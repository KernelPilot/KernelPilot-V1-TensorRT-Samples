#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float get_value_clamp(const float* __restrict__ input, int H, int W, int y, int x) {
    y = max(0, min(H - 1, y));
    x = max(0, min(W - 1, x));
    return __ldg(&input[y * W + x]);
}

__global__ void grid_sample_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grid,
    float* __restrict__ output,
    int H, int W, int H_out, int W_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H_out * W_out) return;

    // Vectorized load of grid coordinates (coalesced memory access)
    const float2 coords = reinterpret_cast<const float2*>(grid)[idx];
    
    // Fast coordinate transformation using FMA
    const float fx = __fmaf_rn(coords.x, 0.5f * W, 0.5f * (W - 1));
    const float fy = __fmaf_rn(coords.y, 0.5f * H, 0.5f * (H - 1));

    // Integer coordinates and fractional parts
    const int x0 = __float2int_rd(fx);
    const int y0 = __float2int_rd(fy);
    const float wx = fx - x0;
    const float wy = fy - y0;

    // Optimized boundary checks and texture-like sampling
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const int x0_clamped = max(x0, 0);
    const int y0_clamped = max(y0, 0);

    // Load all required values with read-only cache
    const float v00 = __ldg(&input[y0_clamped * W + x0_clamped]);
    const float v01 = __ldg(&input[y0_clamped * W + x1]);
    const float v10 = __ldg(&input[y1 * W + x0_clamped]);
    const float v11 = __ldg(&input[y1 * W + x1]);

    // Optimized bilinear interpolation using FMA
    const float v0 = __fmaf_rn(v01 - v00, wx, v00);
    const float v1 = __fmaf_rn(v11 - v10, wx, v10);
    output[idx] = __fmaf_rn(v1 - v0, wy, v0);
}

// test (unchanged)
bool compare(const float* a, const float* b, int size, float tol = 1e-3f) {
    for (int i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

void read_binary(const std::string& fn, float* data, size_t size) {
    std::ifstream in(fn, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fn << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

int main() {
    const int H = 32, W = 32, H_out = 16, W_out = 16;
    const int input_size = H * W;
    const int grid_size = H_out * W_out * 2;
    const int output_size = H_out * W_out;
    const int n_tests = 5;

    float* h_input = new float[input_size];
    float* h_grid = new float[grid_size];
    float* h_ref = new float[output_size];
    float* h_out = new float[output_size];

    float *d_input, *d_grid, *d_out;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_grid, grid_size * sizeof(float));
    cudaMalloc(&d_out, output_size * sizeof(float));

    bool all_pass = true;

    for (int i = 0; i < n_tests; ++i) {
        read_binary("data/gs_input_" + std::to_string(i+1) + ".bin", h_input, input_size);
        read_binary("data/gs_grid_" + std::to_string(i+1) + ".bin", h_grid, grid_size);
        read_binary("data/gs_ref_" + std::to_string(i+1) + ".bin", h_ref, output_size);

        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grid, h_grid, grid_size * sizeof(float), cudaMemcpyHostToDevice);

        // Optimal block size for RTX 3090 Ti (128 threads per block)
        const int threads = 128;
        const int blocks = (H_out * W_out + threads - 1) / threads;
        
        // Launch kernel with optimal configuration
        grid_sample_kernel<<<blocks, threads>>>(d_input, d_grid, d_out, H, W, H_out, W_out);

        cudaMemcpy(h_out, d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare(h_out, h_ref, output_size)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            break;
        }
    }

    if (all_pass) std::cout << "T" << std::endl;

    delete[] h_input; delete[] h_grid; delete[] h_ref; delete[] h_out;
    cudaFree(d_input); cudaFree(d_grid); cudaFree(d_out);
    return 0;
}