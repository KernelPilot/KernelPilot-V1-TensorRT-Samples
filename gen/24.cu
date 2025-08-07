#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cmath>

__global__ void one_hot_kernel(const int* indices, float* output, float off_value, float on_value, int depth, int N) {
    extern __shared__ int shared_indices[];
    
    // Phase 1: Load indices into shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        shared_indices[tid] = indices[idx];
    }
    __syncthreads();
    
    // Phase 2: Process output with coalesced memory access
    if (idx < N) {
        int class_idx = shared_indices[tid];
        float* output_row = output + idx * depth;
        
        // First set all values to off_value
        for (int d = tid; d < depth; d += blockDim.x) {
            output_row[d] = off_value;
        }
        
        // Then set the on_value at the correct position
        if (class_idx >= 0 && class_idx < depth) {
            output_row[class_idx] = on_value;
        }
    }
}

void read_bin_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

void read_bin_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_outputs(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    const int depth = 10;
    const std::string prefix = "data/onehot_";
    bool all_pass = true;

    for (int t = 0; t < Ns.size(); ++t) {
        int N = Ns[t];
        std::vector<int> h_indices(N);
        std::vector<float> h_values(2);
        std::vector<float> h_ref(N * depth);
        std::vector<float> h_out(N * depth);

        read_bin_int(prefix + std::to_string(t + 1) + "_idx.bin", h_indices.data(), N);
        read_bin_float(prefix + std::to_string(t + 1) + "_val.bin", h_values.data(), 2);
        read_bin_float(prefix + std::to_string(t + 1) + "_ref.bin", h_ref.data(), N * depth);

        int* d_indices;
        float* d_output;
        cudaMalloc(&d_indices, N * sizeof(int));
        cudaMalloc(&d_output, N * depth * sizeof(float));
        cudaMemcpy(d_indices, h_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // Optimized launch configuration
        int threads = 256;  // Optimal for Ampere architecture
        int blocks = (N + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(int);
        
        one_hot_kernel<<<blocks, threads, shared_mem_size>>>(d_indices, d_output, h_values[0], h_values[1], depth, N);

        cudaMemcpy(h_out.data(), d_output, N * depth * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_out.data(), h_ref.data(), N * depth)) {
            std::cout << "F\n";
            all_pass = false;
            break;
        }

        cudaFree(d_indices);
        cudaFree(d_output);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}