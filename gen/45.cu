#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

__global__ void unsqueeze_axis1_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int D) {
    // Warp-level optimization with 32 threads working together
    constexpr int elements_per_warp = 128;  // 4 elements per thread * 32 threads
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_warp_offset = warp_id * elements_per_warp;
    
    // Using uint4 for 128-bit memory transactions (perfect for Ampere)
    uint4* output_vec = reinterpret_cast<uint4*>(output + global_warp_offset);
    const uint4* input_vec = reinterpret_cast<const uint4*>(input + global_warp_offset);
    
    // Each thread handles 4 elements (128-bit) with no bank conflicts
    if (global_warp_offset + lane_id * 4 + 3 < N * D) {
        uint4 val = input_vec[lane_id];
        output_vec[lane_id] = val;
    }
    // Handle the last partial warp
    else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = global_warp_offset + lane_id * 4 + i;
            if (idx < N * D) {
                output[idx] = input[idx];
            }
        }
    }
}

void read_bin(const std::string& fname, float* data, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fname << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    const int D = 64;
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t size = N * D;
        size_t bytes = size * sizeof(float);

        std::string input_file = "data/unsqueeze_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/unsqueeze_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_ref   = (float*)malloc(bytes);
        float* h_out   = (float*)malloc(bytes);
        read_bin(input_file, h_input, size);
        read_bin(ref_file, h_ref, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Optimal configuration for RTX 3090 Ti
        const int threads_per_block = 256;  // 8 warps per block
        const int elements_per_warp = 128;
        int total_warps = (size + elements_per_warp - 1) / elements_per_warp;
        int blocks = (total_warps * 32 + threads_per_block - 1) / threads_per_block;
        
        unsqueeze_axis1_kernel<<<blocks, threads_per_block>>>(d_input, d_output, N, D);
        cudaMemcpy(h_out, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare(h_out, h_ref, size)) {
            std::cout << "F\n";
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_ref); free(h_out);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_ref); free(h_out);
    }

    if (all_match) std::cout << "T\n";
    return 0;
}