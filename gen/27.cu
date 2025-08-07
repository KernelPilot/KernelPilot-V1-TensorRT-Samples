#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define C 64

__global__ void plugin_square_kernel(const float* __restrict__ input, float* __restrict__ output, size_t total_size) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    // Process 8 elements per thread using vectorized loads/stores
    if (idx + 7 < total_size) {
        // Use float4 for better memory throughput
        float4 in0 = reinterpret_cast<const float4*>(input)[idx/4];
        float4 in1 = reinterpret_cast<const float4*>(input)[idx/4 + 1];
        
        float4 out0, out1;
        out0.x = in0.x * in0.x;
        out0.y = in0.y * in0.y;
        out0.z = in0.z * in0.z;
        out0.w = in0.w * in0.w;
        
        out1.x = in1.x * in1.x;
        out1.y = in1.y * in1.y;
        out1.z = in1.z * in1.z;
        out1.w = in1.w * in1.w;
        
        reinterpret_cast<float4*>(output)[idx/4] = out0;
        reinterpret_cast<float4*>(output)[idx/4 + 1] = out1;
    }
    else {
        // Handle remaining elements
        for (size_t i = 0; i < 8 && (idx + i) < total_size; ++i) {
            output[idx + i] = input[idx + i] * input[idx + i];
        }
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_float_arrays(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t total = N * C;
        size_t bytes = total * sizeof(float);

        std::string input_file = "data/plugin_input_" + std::to_string(idx+1) + ".bin";
        std::string ref_file   = "data/plugin_output_" + std::to_string(idx+1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);

        read_binary_float(input_file, h_input, total);
        read_binary_float(ref_file, h_ref, total);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Using 128 threads per block (4 warps) for better occupancy
        int threads = 128;
        int blocks = (total + threads * 8 - 1) / (threads * 8);
        plugin_square_kernel<<<blocks, threads>>>(d_input, d_output, total);
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_float_arrays(h_output, h_ref, total)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            break;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        free(h_ref);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}