#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

__global__ void unary_exp_kernel(const float* __restrict__ input, float* __restrict__ output, size_t N) {
    const size_t idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Warp-level optimization for Ampere architecture
    if (idx_base + 3 < N) {
        // Process 4 elements per thread with vectorized loads/stores
        float4 in = reinterpret_cast<const float4*>(input)[idx_base/4];
        float4 out;
        out.x = __expf(in.x);
        out.y = __expf(in.y);
        out.z = __expf(in.z);
        out.w = __expf(in.w);
        reinterpret_cast<float4*>(output)[idx_base/4] = out;
    } else {
        // Fallback for remaining elements
        for (int i = 0; i < 4 && (idx_base + i) < N; i++) {
            output[idx_base + i] = __expf(input[idx_base + i]);
        }
    }
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> sizes = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < sizes.size(); ++idx) {
        size_t N = sizes[idx];
        size_t bytes = N * sizeof(float);

        std::string input_file = "data/unary_input_" + std::to_string(idx+1) + ".bin";
        std::string ref_file   = "data/unary_ref_"   + std::to_string(idx+1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_ref   = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_ref, N);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads * 4 - 1) / (threads * 4);
        unary_exp_kernel<<<blocks, threads>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_ref); free(h_output);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_ref); free(h_output);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}