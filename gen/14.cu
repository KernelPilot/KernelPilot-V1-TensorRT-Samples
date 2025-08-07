#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>

#define C 32
#define TOL 1e-2f

__global__ void gather_kernel(const float* __restrict__ data, 
                             const int* __restrict__ indices, 
                             float* __restrict__ output, 
                             int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const int col_idx = indices[idx];
        // Clamp the index to valid range [0, C-1] using ternary operator
        const int safe_col_idx = (col_idx < 0) ? 0 : ((col_idx >= C) ? C-1 : col_idx);
        output[idx] = data[idx * C + safe_col_idx];
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
bool compare_arrays(const float* a, const float* b, size_t size, float tol = TOL) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    const size_t Ns[5] = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int test_id = 0; test_id < 5; ++test_id) {
        size_t N = Ns[test_id];
        size_t data_bytes = N * C * sizeof(float);
        size_t indices_bytes = N * sizeof(int);
        size_t output_bytes = N * sizeof(float);

        std::string prefix = "data/gather_";
        std::string data_file = prefix + "data_" + std::to_string(test_id + 1) + ".bin";
        std::string indices_file = prefix + "indices_" + std::to_string(test_id + 1) + ".bin";
        std::string ref_file = prefix + "ref_" + std::to_string(test_id + 1) + ".bin";

        float* h_data = (float*)malloc(data_bytes);
        int* h_indices = (int*)malloc(indices_bytes);
        float* h_output = (float*)malloc(output_bytes);
        float* h_ref = (float*)malloc(output_bytes);

        read_binary_float(data_file, h_data, N * C);
        read_binary_int(indices_file, h_indices, N);
        read_binary_float(ref_file, h_ref, N);

        float *d_data, *d_output;
        int* d_indices;
        cudaMalloc(&d_data, data_bytes);
        cudaMalloc(&d_indices, indices_bytes);
        cudaMalloc(&d_output, output_bytes);

        cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        gather_kernel<<<blocks, threads>>>(d_data, d_indices, d_output, N);

        cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_arrays(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        cudaFree(d_data); cudaFree(d_indices); cudaFree(d_output);
        free(h_data); free(h_indices); free(h_output); free(h_ref);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}