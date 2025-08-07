#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define N 1
#define C 4
#define H 8
#define W 8

__global__ void scatter_element_kernel(const float* __restrict__ updates,
                                     const int* __restrict__ indices,
                                     float* __restrict__ output) {
    // Optimized for Ampere architecture (RTX 3090 Ti)
    const int n = blockIdx.z;  // N dimension
    const int c = blockIdx.y;  // C dimension
    const int h = blockIdx.x;  // H dimension
    
    // Each thread handles multiple W elements for better utilization
    const int w_start = threadIdx.x * 4;  // Process 4 elements per thread
    const int w_end = min(w_start + 4, W);

    // Shared memory for coalesced writes
    __shared__ float s_updates[32][4];
    __shared__ int s_indices[32][4];
    
    // Load data in coalesced pattern
    for (int w = w_start; w < w_end; w++) {
        const int idx = ((n * C + c) * H + h) * W + w;
        s_updates[threadIdx.x][w-w_start] = updates[idx];
        s_indices[threadIdx.x][w-w_start] = indices[idx];
    }
    __syncthreads();

    // Scatter with optimized memory access
    for (int w = w_start; w < w_end; w++) {
        const int target_h = s_indices[threadIdx.x][w-w_start];
        const int out_idx = ((n * C + c) * H + target_h) * W + w;
        output[out_idx] = s_updates[threadIdx.x][w-w_start];
    }
}

bool compare(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

void read_binary(const std::string& filename, void* data, size_t bytes) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), bytes);
    in.close();
}

int main() {
    size_t total = N * C * H * W;
    size_t fbytes = total * sizeof(float);
    size_t ibytes = total * sizeof(int);

    bool all_pass = true;

    for (int i = 1; i <= 5; ++i) {
        std::string base = "./data/scatter_";
        std::string data_f = base + "data_" + std::to_string(i) + ".bin";
        std::string idx_f = base + "indices_" + std::to_string(i) + ".bin";
        std::string upd_f = base + "updates_" + std::to_string(i) + ".bin";
        std::string ref_f = base + "ref_" + std::to_string(i) + ".bin";

        float *h_data = new float[total];
        float *h_updates = new float[total];
        int *h_indices = new int[total];
        float *h_ref = new float[total];

        read_binary(data_f, h_data, fbytes);
        read_binary(upd_f, h_updates, fbytes);
        read_binary(idx_f, h_indices, ibytes);
        read_binary(ref_f, h_ref, fbytes);

        float *d_output, *d_updates;
        int* d_indices;

        cudaMalloc(&d_output, fbytes);
        cudaMalloc(&d_updates, fbytes);
        cudaMalloc(&d_indices, ibytes);

        cudaMemcpy(d_output, h_data, fbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_updates, h_updates, fbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices, ibytes, cudaMemcpyHostToDevice);

        // Optimized grid configuration
        dim3 grid(H, C, N);  // H in x-dim for better cache locality
        dim3 block(8);        // 8 threads per block (process 32 elements total)
        scatter_element_kernel<<<grid, block>>>(d_updates, d_indices, d_output);

        float* h_output = new float[total];
        cudaMemcpy(h_output, d_output, fbytes, cudaMemcpyDeviceToHost);

        if (!compare(h_output, h_ref, total)) {
            std::cout << "F\n";
            all_pass = false;
        }

        delete[] h_data; delete[] h_updates; delete[] h_indices;
        delete[] h_ref; delete[] h_output;
        cudaFree(d_output); cudaFree(d_updates); cudaFree(d_indices);

        if (!all_pass) break;
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}