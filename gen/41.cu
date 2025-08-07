#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cfloat>  // Added for FLT_MAX definition

#define C 10  // number of channels
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        const int offset = row * C;
        
        // Find max value with sequential reduction
        float max_val = -FLT_MAX;  // Corrected FLT_MAX usage
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            max_val = fmaxf(max_val, input[offset + c]);
        }
        
        // Compute exponentials and sum with fast math
        float sum = 0.0f;
        float exp_values[C];
        
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            exp_values[c] = __expf(input[offset + c] - max_val);
            sum += exp_values[c];
        }
        
        // Normalize and store with fast reciprocal
        const float inv_sum = __frcp_rn(sum);
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            output[offset + c] = exp_values[c] * inv_sum;
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
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t size = N * C;
        std::string input_file = "data/sm_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/sm_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(size * sizeof(float));
        float* h_ref   = (float*)malloc(size * sizeof(float));
        float* h_out   = (float*)malloc(size * sizeof(float));

        read_binary_float(input_file, h_input, size);
        read_binary_float(ref_file, h_ref, size);

        float *d_input, *d_out;
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_out, size * sizeof(float));
        cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        softmax_kernel<<<blocks, threads>>>(d_input, d_out, N);
        cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_out, h_ref, size)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_out);
            free(h_input); free(h_ref); free(h_out);
            break;
        }

        cudaFree(d_input); cudaFree(d_out);
        free(h_input); free(h_ref); free(h_out);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}