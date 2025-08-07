#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

__global__ void plugin_layer_kernel(const float* __restrict__ input, float* __restrict__ output, int C, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* row = input + idx * C;
    float sum = 0.0f;

    // Process 8 elements at a time for better ILP and memory throughput
    const int unroll_factor = 8;
    int i = 0;
    for (; i <= C - unroll_factor; i += unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            float val = row[i + j];
            sum += val * val;
        }
    }

    // Handle remaining elements
    for (; i < C; ++i) {
        sum += row[i] * row[i];
    }

    // Use precise square root with fast-math optimization
    output[idx] = __fsqrt_rn(sum);
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

bool compare_array(const float* a, const float* b, size_t n, float tol = 1e-2f) {
    for (size_t i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    const int C = 10;
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t bytes = N * C * sizeof(float);
        size_t out_bytes = N * sizeof(float);

        std::string input_file = "data/plugin_input_" + std::to_string(idx+1) + ".bin";
        std::string ref_file   = "data/plugin_ref_"   + std::to_string(idx+1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_ref   = (float*)malloc(out_bytes);
        float* h_output= (float*)malloc(out_bytes);

        read_binary_float(input_file, h_input, N * C);
        read_binary_float(ref_file, h_ref, N);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, out_bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, out_bytes);

        // Optimal block size for Ampere architecture
        const int threads = 512;
        const int blocks = (N + threads - 1) / threads;
        
        // Launch with async execution
        plugin_layer_kernel<<<blocks, threads, 0, 0>>>(d_input, d_output, C, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, N)) {
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