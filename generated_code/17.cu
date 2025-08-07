#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <cmath>

__global__ void conditional_sum_kernel(const float* __restrict__ x, 
                                     const bool* __restrict__ cond, 
                                     float* __restrict__ then_sum, 
                                     float* __restrict__ else_sum, 
                                     int N) {
    // Using 8x unrolling and warp-level optimizations for Ampere architecture
    constexpr int UNROLL = 8;
    __shared__ float then_shared[32];  // Only need 32 elements for warp-level reduction
    __shared__ float else_shared[32];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int global_idx = blockIdx.x * (blockDim.x * UNROLL) + tid;
    
    float thread_then_sum = 0.0f;
    float thread_else_sum = 0.0f;

    // Process UNROLL elements per thread with perfect unrolling
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        const int current_idx = global_idx + i * blockDim.x;
        if (current_idx < N) {
            const bool c = cond[current_idx];
            const float val = x[current_idx];
            thread_then_sum += c ? val : 0.0f;
            thread_else_sum += c ? 0.0f : val;
        }
    }

    // Warp-level reduction using Ampere's efficient warp shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_then_sum += __shfl_down_sync(0xFFFFFFFF, thread_then_sum, offset);
        thread_else_sum += __shfl_down_sync(0xFFFFFFFF, thread_else_sum, offset);
    }

    // First lane in each warp stores partial sum
    if (lane_id == 0) {
        then_shared[warp_id] = thread_then_sum;
        else_shared[warp_id] = thread_else_sum;
    }
    __syncthreads();

    // Final reduction across warps in block
    if (warp_id == 0) {
        float block_then_sum = (lane_id < blockDim.x/32) ? then_shared[lane_id] : 0.0f;
        float block_else_sum = (lane_id < blockDim.x/32) ? else_shared[lane_id] : 0.0f;

        for (int offset = 8; offset > 0; offset >>= 1) {
            block_then_sum += __shfl_down_sync(0xFFFFFFFF, block_then_sum, offset);
            block_else_sum += __shfl_down_sync(0xFFFFFFFF, block_else_sum, offset);
        }

        // Single atomic add per block
        if (lane_id == 0) {
            atomicAdd(then_sum, block_then_sum);
            atomicAdd(else_sum, block_else_sum);
        }
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void read_binary_bool(const std::string& filename, bool* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(bool));
    in.close();
}

bool compare_scalar(float a, float b, float tol = 1e-1f) {
    return fabsf(a - b) < tol;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        std::string x_file = "data/cond_x_" + std::to_string(idx+1) + ".bin";
        std::string cond_file = "data/cond_mask_" + std::to_string(idx+1) + ".bin";
        std::string then_file = "data/cond_then_" + std::to_string(idx+1) + ".bin";
        std::string else_file = "data/cond_else_" + std::to_string(idx+1) + ".bin";

        float* h_x = (float*)malloc(N * sizeof(float));
        bool* h_cond = (bool*)malloc(N * sizeof(bool));
        float h_then_ref, h_else_ref;

        read_binary_float(x_file, h_x, N);
        read_binary_bool(cond_file, h_cond, N);
        read_binary_float(then_file, &h_then_ref, 1);
        read_binary_float(else_file, &h_else_ref, 1);

        float *d_x, *d_then_sum, *d_else_sum;
        bool *d_cond;
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_cond, N * sizeof(bool));
        cudaMalloc(&d_then_sum, sizeof(float));
        cudaMalloc(&d_else_sum, sizeof(float));
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cond, h_cond, N * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemset(d_then_sum, 0, sizeof(float));
        cudaMemset(d_else_sum, 0, sizeof(float));

        const int threads = 256;
        const int blocks = (static_cast<int>(N) + threads * 8 - 1) / (threads * 8);
        conditional_sum_kernel<<<blocks, threads>>>(d_x, d_cond, d_then_sum, d_else_sum, static_cast<int>(N));

        float h_then_sum, h_else_sum;
        cudaMemcpy(&h_then_sum, d_then_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_else_sum, d_else_sum, sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_scalar(h_then_sum, h_then_ref) || !compare_scalar(h_else_sum, h_else_ref)) {
            std::cout << "F\n";
            all_match = false;
            cudaFree(d_x); cudaFree(d_cond); cudaFree(d_then_sum); cudaFree(d_else_sum);
            free(h_x); free(h_cond);
            break;
        }

        cudaFree(d_x); cudaFree(d_cond); cudaFree(d_then_sum); cudaFree(d_else_sum);
        free(h_x); free(h_cond);
    }

    if (all_match) std::cout << "T\n";
    return 0;
}