#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void cast_float_to_int_kernel(const float* input, int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = static_cast<int>(input[idx]);
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

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

// test
bool compare_arrays(const int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        std::string input_file = "data/cast_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/cast_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(N * sizeof(float));
        int*   h_output = (int*)malloc(N * sizeof(int));
        int*   h_ref    = (int*)malloc(N * sizeof(int));

        read_binary_float(input_file, h_input, N);
        read_binary_int(ref_file, h_ref, N);

        float* d_input;
        int*   d_output;

        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(int));
        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        cast_float_to_int_kernel<<<blocks, threads>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
        
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