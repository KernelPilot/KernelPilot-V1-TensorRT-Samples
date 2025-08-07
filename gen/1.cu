#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

__global__ void relu_kernel(const float* __restrict__ input, 
                           float* __restrict__ output, 
                           const int N) {
    // Using 128 threads per block for better occupancy
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bdim = blockDim.x;
    
    // Process 8 elements per thread for better memory throughput
    const int elements_per_thread = 8;
    const int global_idx = (bid * bdim + tid) * elements_per_thread;
    
    // Using vectorized loads/stores with float4
    if (global_idx + elements_per_thread - 1 < N) {
        // Load 8 elements (2 float4 vectors)
        float4 in1 = *reinterpret_cast<const float4*>(&input[global_idx]);
        float4 in2 = *reinterpret_cast<const float4*>(&input[global_idx + 4]);
        
        // Process with ReLU
        float4 out1, out2;
        out1.x = fmaxf(0.0f, in1.x);
        out1.y = fmaxf(0.0f, in1.y);
        out1.z = fmaxf(0.0f, in1.z);
        out1.w = fmaxf(0.0f, in1.w);
        out2.x = fmaxf(0.0f, in2.x);
        out2.y = fmaxf(0.0f, in2.y);
        out2.z = fmaxf(0.0f, in2.z);
        out2.w = fmaxf(0.0f, in2.w);
        
        // Store results
        *reinterpret_cast<float4*>(&output[global_idx]) = out1;
        *reinterpret_cast<float4*>(&output[global_idx + 4]) = out2;
    }
    else {
        // Handle remaining elements (not multiple of 8)
        for (int i = 0; i < elements_per_thread && global_idx + i < N; ++i) {
            output[global_idx + i] = fmaxf(0.0f, input[global_idx + i]);
        }
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_arrays(const float* a, const float* b, size_t N, float tol = 1e-3f) {
    for (size_t i = 0; i < N; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t bytes = N * sizeof(float);

        std::string in_file  = "data/act_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/act_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input  = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref    = (float*)malloc(bytes);

        read_binary_float(in_file, h_input, N);
        read_binary_float(ref_file, h_ref, N);

        float* d_input;
        float* d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);

        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        const int threads_per_block = 128;  // Better for Ampere architecture
        const int elements_per_thread = 8;
        int blocks = (static_cast<int>(N) + threads_per_block * elements_per_thread - 1) / 
                    (threads_per_block * elements_per_thread);
        blocks = min(blocks, 65535);  // Maximum grid size
        
        relu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, static_cast<int>(N));

        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_arrays(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}