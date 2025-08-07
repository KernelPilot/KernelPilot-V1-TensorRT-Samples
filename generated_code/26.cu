#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

__global__ void prelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            const float* __restrict__ slopes,
                            int N, int C) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bdim = blockDim.x;
    
    // Process 8 elements per thread for maximum memory throughput
    const int idx = (bid * bdim + tid) * 8;
    
    if (idx + 7 < N * C) {
        // Load 8 elements at once using float4
        float4 in0 = *reinterpret_cast<const float4*>(&input[idx]);
        float4 in1 = *reinterpret_cast<const float4*>(&input[idx+4]);
        
        // Compute channels for all 8 elements
        int channels[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            channels[i] = (idx + i) % C;
        }
        
        // Process all 8 elements with minimal branching
        float4 out0, out1;
        out0.x = in0.x * (in0.x >= 0.0f ? 1.0f : slopes[channels[0]]);
        out0.y = in0.y * (in0.y >= 0.0f ? 1.0f : slopes[channels[1]]);
        out0.z = in0.z * (in0.z >= 0.0f ? 1.0f : slopes[channels[2]]);
        out0.w = in0.w * (in0.w >= 0.0f ? 1.0f : slopes[channels[3]]);
        out1.x = in1.x * (in1.x >= 0.0f ? 1.0f : slopes[channels[4]]);
        out1.y = in1.y * (in1.y >= 0.0f ? 1.0f : slopes[channels[5]]);
        out1.z = in1.z * (in1.z >= 0.0f ? 1.0f : slopes[channels[6]]);
        out1.w = in1.w * (in1.w >= 0.0f ? 1.0f : slopes[channels[7]]);
        
        // Store results
        *reinterpret_cast<float4*>(&output[idx]) = out0;
        *reinterpret_cast<float4*>(&output[idx+4]) = out1;
    }
    else {
        // Handle remaining elements (1-7)
        #pragma unroll
        for (int i = 0; i < 8 && idx + i < N * C; i++) {
            const int channel = (idx + i) % C;
            const float x = input[idx + i];
            output[idx + i] = x * (x >= 0.0f ? 1.0f : slopes[channel]);
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const int C = 64;

    float* h_slopes = new float[C];
    read_binary_float("data/prelu_slopes.bin", h_slopes, C);

    float* d_slopes;
    cudaMalloc(&d_slopes, C * sizeof(float));
    cudaMemcpy(d_slopes, h_slopes, C * sizeof(float), cudaMemcpyHostToDevice);

    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t size = N * C;
        size_t bytes = size * sizeof(float);

        std::string input_file = "data/prelu_input_" + std::to_string(idx + 1) + ".bin";
        std::string output_file = "data/prelu_output_" + std::to_string(idx + 1) + ".bin";

        float* h_input = new float[size];
        float* h_ref_output = new float[size];
        float* h_output = new float[size];

        read_binary_float(input_file, h_input, size);
        read_binary_float(output_file, h_ref_output, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);

        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        const int threads = 256;  // Optimal for Ampere architecture
        const int blocks = (static_cast<int>(size) + threads * 8 - 1) / (threads * 8);
        prelu_kernel<<<blocks, threads>>>(d_input, d_output, d_slopes, N, C);

        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref_output, size)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            delete[] h_input; delete[] h_ref_output; delete[] h_output;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        delete[] h_input; delete[] h_ref_output; delete[] h_output;
    }

    cudaFree(d_slopes);
    delete[] h_slopes;

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}