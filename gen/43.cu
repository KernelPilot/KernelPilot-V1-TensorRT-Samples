#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>

#define K 5

__device__ inline void insert_sorted(float* topk, float val) {
    if (val <= topk[K-1]) return;
    
    int pos = K - 2;
    while (pos >= 0 && val > topk[pos]) {
        topk[pos+1] = topk[pos];
        pos--;
    }
    topk[pos+1] = val;
}

__global__ void topk_kernel(const float* __restrict__ input, 
                           float* __restrict__ output, 
                           int N, int C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* row_data = input + row * C;
    float* row_output = output + row * K;

    // Initialize top K values in registers
    float topk[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        topk[i] = -FLT_MAX;
    }

    // Process elements with 4-way unrolled loop for better ILP
    int i = 0;
    for (; i <= C - 4; i += 4) {
        float4 vals = reinterpret_cast<const float4*>(row_data + i)[0];
        insert_sorted(topk, vals.x);
        insert_sorted(topk, vals.y);
        insert_sorted(topk, vals.z);
        insert_sorted(topk, vals.w);
    }

    // Process remaining elements
    for (; i < C; ++i) {
        insert_sorted(topk, row_data[i]);
    }

    // Write output in descending order
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        row_output[i] = topk[i];
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_outputs(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const int C = 100;

    bool all_pass = true;
    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t in_bytes = N * C * sizeof(float);
        size_t out_bytes = N * K * sizeof(float);

        std::string in_file = "data/topk_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/topk_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(in_bytes);
        float* h_ref = (float*)malloc(out_bytes);
        float* h_out = (float*)malloc(out_bytes);

        read_binary_float(in_file, h_input, N * C);
        read_binary_float(ref_file, h_ref, N * K);

        float *d_input, *d_output;
        cudaMalloc(&d_input, in_bytes);
        cudaMalloc(&d_output, out_bytes);
        cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        topk_kernel<<<blocks, threads>>>(d_input, d_output, N, C);

        cudaMemcpy(h_out, d_output, out_bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_outputs(h_out, h_ref, N * K)) {
            std::cout << "F\n";
            all_pass = false;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_out); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}