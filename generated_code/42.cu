#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>

__global__ void squeeze_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    // Using 128 threads per block for better occupancy on Ampere
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 8 elements per thread for better memory throughput
    #pragma unroll 8
    for (int i = tid; i < N; i += stride) {
        output[i] = input[i];
    }
}

void read_binary(const std::string& fname, float* data, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << fname << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t bytes = N * sizeof(float);

        std::string in_file  = "data/squeeze_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/squeeze_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input  = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref    = (float*)malloc(bytes);

        read_binary(in_file, h_input, N);
        read_binary(ref_file, h_ref, N);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        int threads = 128;  // Better occupancy on Ampere
        int blocks = min((N + threads * 8 - 1) / (threads * 8), 2048);  // Process 8 elements per thread
        
        squeeze_kernel<<<blocks, threads>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, N)) {
            std::cout << "F\n";
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}