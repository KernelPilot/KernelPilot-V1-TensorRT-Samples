#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>

#define POOL_SIZE 4
#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void max_pooling_1d(const float* __restrict__ input, float* __restrict__ output, int B, int W) {
    const int pooled_W = W / POOL_SIZE;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < B * pooled_W) {
        const int b = idx / pooled_W;
        const int w = idx % pooled_W;
        const int start = b * W + w * POOL_SIZE;
        
        // Vectorized load for better memory throughput
        float4 window = reinterpret_cast<const float4*>(input + start)[0];
        
        // Efficient max reduction without branches
        float max_val = fmaxf(fmaxf(window.x, window.y), fmaxf(window.z, window.w));
        
        output[idx] = max_val;
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Bs = {8, 16, 32, 64, 128};
    std::vector<int> Ws = {1024, 2048, 4096, 8192, 16384};
    bool all_match = true;

    for (int i = 0; i < Bs.size(); ++i) {
        int B = Bs[i];
        int W = Ws[i];
        int pooled_W = W / POOL_SIZE;
        size_t in_bytes = B * W * sizeof(float);
        size_t out_bytes = B * pooled_W * sizeof(float);

        std::string in_file = "data/pool_in_" + std::to_string(i+1) + ".bin";
        std::string ref_file = "data/pool_ref_" + std::to_string(i+1) + ".bin";

        float* h_in = (float*)malloc(in_bytes);
        float* h_ref = (float*)malloc(out_bytes);
        float* h_out = (float*)malloc(out_bytes);

        read_binary_float(in_file, h_in, B * W);
        read_binary_float(ref_file, h_ref, B * pooled_W);

        float *d_in, *d_out;
        cudaMalloc(&d_in, in_bytes);
        cudaMalloc(&d_out, out_bytes);
        cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

        int threads = BLOCK_SIZE;
        int blocks = (B * pooled_W + threads - 1) / threads;
        max_pooling_1d<<<blocks, threads>>>(d_in, d_out, B, W);
        cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_out, h_ref, B * pooled_W)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_out); free(h_ref);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}