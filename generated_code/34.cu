#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>

__global__ void reverse_sequence_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       const int* __restrict__ seq_lens, 
                                       int B, int S) {
    // Using 1D grid for better memory coalescing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < B * S; i += stride) {
        int batch_idx = i / S;
        int seq_idx = i % S;
        int seq_len = seq_lens[batch_idx];
        
        if (seq_idx < seq_len) {
            // Reverse the sequence up to seq_len
            int reversed_idx = batch_idx * S + (seq_len - 1 - seq_idx);
            output[i] = input[reversed_idx];
        } else {
            // Copy the remaining elements unchanged
            output[i] = input[i];
        }
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

// test
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    std::vector<int> Bs = {64, 128, 256, 512, 1024};
    int S = 128;
    bool all_passed = true;

    for (int idx = 0; idx < Bs.size(); ++idx) {
        int B = Bs[idx];
        int total = B * S;

        std::string input_file = "data/rs_input_" + std::to_string(idx + 1) + ".bin";
        std::string lens_file  = "data/rs_lens_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/rs_ref_" + std::to_string(idx + 1) + ".bin";

        float *h_input = (float*)malloc(total * sizeof(float));
        int* h_lens    = (int*)malloc(B * sizeof(int));
        float* h_ref   = (float*)malloc(total * sizeof(float));
        float* h_output = (float*)malloc(total * sizeof(float));

        read_binary_float(input_file, h_input, total);
        read_binary_int(lens_file, h_lens, B);
        read_binary_float(ref_file, h_ref, total);

        float *d_input, *d_output;
        int* d_lens;
        cudaMalloc(&d_input, total * sizeof(float));
        cudaMalloc(&d_output, total * sizeof(float));
        cudaMalloc(&d_lens, B * sizeof(int));

        cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lens, h_lens, B * sizeof(int), cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        int blockSize = 256;  // Optimal for Ampere architecture
        int numBlocks = (B * S + blockSize - 1) / blockSize;
        reverse_sequence_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_lens, B, S);
        cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_output, h_ref, total)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            break;
        }

        cudaFree(d_input); cudaFree(d_output); cudaFree(d_lens);
        free(h_input); free(h_output); free(h_lens); free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}