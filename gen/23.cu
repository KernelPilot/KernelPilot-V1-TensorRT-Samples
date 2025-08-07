#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define EPSILON 1e-5f
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define ROWS_PER_BLOCK 4

__global__ void normalization_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                   const float* __restrict__ S, const float* __restrict__ B,
                                   int N, int D) {
    // Warp and lane index
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    
    if (row >= N) return;

    // Shared memory for partial sums (2 floats per row)
    __shared__ float smem[ROWS_PER_BLOCK * 2];
    
    // First pass: compute mean and variance
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    // Unrolled loop for better performance
    #pragma unroll(4)
    for (int col = lane_id; col < D; col += WARP_SIZE) {
        float val = X[row * D + col];
        sum += val;
        square_sum += val * val;
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        square_sum += __shfl_down_sync(0xFFFFFFFF, square_sum, offset);
    }
    
    // Store results in shared memory
    if (lane_id == 0) {
        float mean = sum / D;
        float variance = (square_sum / D) - (mean * mean);
        smem[warp_id * 2] = mean;
        smem[warp_id * 2 + 1] = rsqrtf(variance + EPSILON);
    }
    
    __syncthreads();
    
    // Second pass: normalize and scale
    float row_mean = smem[warp_id * 2];
    float row_inv_std = smem[warp_id * 2 + 1];
    
    // Unrolled loop for better performance
    #pragma unroll(4)
    for (int col = lane_id; col < D; col += WARP_SIZE) {
        float val = X[row * D + col];
        float normalized = (val - row_mean) * row_inv_std;
        Y[row * D + col] = normalized * S[col] + B[col];
    }
}

void read_bin(const std::string& fn, float* ptr, size_t size) {
    std::ifstream in(fn, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fn << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(ptr), sizeof(float) * size);
    in.close();
}

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const int D = 128;
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t total = N * D;
        size_t bytes = total * sizeof(float);

        std::string xfile = "data/norm_input_" + std::to_string(idx + 1) + ".bin";
        std::string sfile = "data/norm_scale_" + std::to_string(idx + 1) + ".bin";
        std::string bfile = "data/norm_bias_" + std::to_string(idx + 1) + ".bin";
        std::string rfile = "data/norm_ref_" + std::to_string(idx + 1) + ".bin";

        float *h_x = (float*)malloc(bytes);
        float *h_s = (float*)malloc(D * sizeof(float));
        float *h_b = (float*)malloc(D * sizeof(float));
        float *h_ref = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);

        read_bin(xfile, h_x, total);
        read_bin(sfile, h_s, D);
        read_bin(bfile, h_b, D);
        read_bin(rfile, h_ref, total);

        float *d_x, *d_y, *d_s, *d_b;
        cudaMalloc(&d_x, bytes);
        cudaMalloc(&d_y, bytes);
        cudaMalloc(&d_s, D * sizeof(float));
        cudaMalloc(&d_b, D * sizeof(float));

        cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, h_s, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice);

        int blocks = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        normalization_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_s, d_b, N, D);
        cudaMemcpy(h_out, d_y, bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_out, h_ref, total)) {
            std::cout << "F\n";
            all_pass = false;
            break;
        }

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_s); cudaFree(d_b);
        free(h_x); free(h_s); free(h_b); free(h_out); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}