#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void identity_kernel(const float* __restrict__ input, float* __restrict__ output, size_t N) {
    const size_t idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    if (idx < N) {
        const float val0 = input[idx];
        const float val1 = (idx + blockDim.x < N) ? input[idx + blockDim.x] : 0.0f;
        const float val2 = (idx + 2*blockDim.x < N) ? input[idx + 2*blockDim.x] : 0.0f;
        const float val3 = (idx + 3*blockDim.x < N) ? input[idx + 3*blockDim.x] : 0.0f;
        
        output[idx] = val0;
        if (idx + blockDim.x < N) output[idx + blockDim.x] = val1;
        if (idx + 2*blockDim.x < N) output[idx + 2*blockDim.x] = val2;
        if (idx + 3*blockDim.x < N) output[idx + 3*blockDim.x] = val3;
    }
}

// test
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

int main() {
    size_t Ns[] = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int i = 0; i < 5; ++i) {
        size_t N = Ns[i];
        std::string input_file = "data/identity_input_" + std::to_string(i+1) + ".bin";
        std::string ref_file   = "data/identity_ref_" + std::to_string(i+1) + ".bin";

        float* h_input = new float[N];
        float* h_ref   = new float[N];
        float* h_output = new float[N];

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_ref, N);

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads * 4 - 1) / (threads * 4);
        identity_kernel<<<blocks, threads>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            delete[] h_input; delete[] h_ref; delete[] h_output;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        delete[] h_input; delete[] h_ref; delete[] h_output;
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}