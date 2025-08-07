#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define N 64
#define ALPHA 1e-4f
#define BETA 0.75f
#define NORM_WINDOW 5

__global__ void lrn_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    extern __shared__ float s_data[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= C) return;
    
    // Precompute constants
    const float alpha_div_n = ALPHA / NORM_WINDOW;
    const int half_window = NORM_WINDOW / 2;
    
    // Load data into shared memory with cooperative loading
    int load_pos = idx - half_window;
    if (load_pos >= 0 && load_pos < C) {
        s_data[tid] = input[load_pos];
    } else {
        s_data[tid] = 0.0f;
    }
    
    // Load right halo with the first warp only
    if (tid < 32 && (tid < half_window)) {
        int halo_pos = idx + blockDim.x - half_window + tid;
        if (halo_pos < C) {
            s_data[blockDim.x + tid] = input[halo_pos];
        } else {
            s_data[blockDim.x + tid] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute sum of squares with loop unrolling
    float sum = 0.0f;
    #pragma unroll
    for (int j = -half_window; j <= half_window; ++j) {
        float val = s_data[tid + half_window + j];
        sum += val * val;
    }
    
    // Fast power approximation using exponentiation by squaring
    float norm_factor = 1.0f + alpha_div_n * sum;
    norm_factor = __expf(BETA * __logf(norm_factor));
    
    // Apply normalization with fused operations
    output[idx] = __fdividef(input[idx], norm_factor);
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> sizes = {64, 256, 1024, 4096, 8192};
    bool all_passed = true;

    for (int idx = 0; idx < sizes.size(); ++idx) {
        int C = sizes[idx];
        std::string input_file = "data/lrn_input_" + std::to_string(idx+1) + ".bin";
        std::string ref_file = "data/lrn_ref_" + std::to_string(idx+1) + ".bin";

        float* h_input = (float*)malloc(C * sizeof(float));
        float* h_ref   = (float*)malloc(C * sizeof(float));
        float* h_out   = (float*)malloc(C * sizeof(float));

        read_binary(input_file, h_input, C);
        read_binary(ref_file, h_ref, C);

        float *d_input, *d_output;
        cudaMalloc(&d_input, C * sizeof(float));
        cudaMalloc(&d_output, C * sizeof(float));
        cudaMemcpy(d_input, h_input, C * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (C + threads - 1) / threads;
        size_t shared_mem_size = (threads + NORM_WINDOW - 1) * sizeof(float);
        lrn_kernel<<<blocks, threads, shared_mem_size>>>(d_input, d_output, C);
        cudaMemcpy(h_out, d_output, C * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_arrays(h_out, h_ref, C)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_out); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_out); free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}