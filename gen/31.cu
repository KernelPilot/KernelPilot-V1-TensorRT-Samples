#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <cub/cub.cuh>

__global__ void ragged_softmax_kernel(const float* __restrict__ input, 
                                     float* __restrict__ output, 
                                     const int* __restrict__ bounds, 
                                     int Z, int S) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int valid_length = bounds[row];
    
    // Shared memory for warp-level reductions
    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];
    
    // Phase 1: Find max value (warp-level reduction)
    float thread_max = -INFINITY;
    #pragma unroll 4
    for (int col = tid; col < valid_length; col += blockDim.x) {
        thread_max = fmaxf(thread_max, input[row * S + col]);
    }
    
    // Warp-level max reduction
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 16));
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 8));
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 4));
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 2));
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 1));
    
    if (lane_id == 0) {
        smem_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final max reduction across warps
    float row_max = lane_id < blockDim.x/32 ? smem_max[lane_id] : -INFINITY;
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 16));
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 8));
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
    
    // Phase 2: Compute exponentials and sum (warp-level reduction)
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int col = tid; col < valid_length; col += blockDim.x) {
        float val = input[row * S + col];
        float exp_val = expf(val - row_max);
        output[row * S + col] = exp_val;  // Temporarily store exp values
        thread_sum += exp_val;
    }
    
    // Warp-level sum reduction
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, 16);
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, 8);
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, 4);
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, 2);
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, 1);
    
    if (lane_id == 0) {
        smem_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final sum reduction across warps
    float row_sum = lane_id < blockDim.x/32 ? smem_sum[lane_id] : 0.0f;
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 16);
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 8);
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
    
    float inv_row_sum = (row_sum != 0.0f) ? 1.0f / row_sum : 0.0f;
    
    // Phase 3: Normalize and handle invalid positions
    #pragma unroll 4
    for (int col = tid; col < S; col += blockDim.x) {
        if (col < valid_length) {
            output[row * S + col] *= inv_row_sum;
        } else {
            output[row * S + col] = 0.0f;
        }
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<std::pair<int, int>> shapes = {
        {32, 64}, {64, 128}, {128, 256}, {256, 256}, {512, 128}
    };
    bool all_pass = true;

    for (int idx = 0; idx < shapes.size(); ++idx) {
        int Z = shapes[idx].first;
        int S = shapes[idx].second;
        size_t total = Z * S;

        std::string prefix = "data/rs_";
        std::string input_file = prefix + "input_" + std::to_string(idx + 1) + ".bin";
        std::string bounds_file = prefix + "bounds_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = prefix + "ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = new float[total];
        float* h_ref = new float[total];
        float* h_output = new float[total];
        int* h_bounds = new int[Z];

        read_binary_float(input_file, h_input, total);
        read_binary_int(bounds_file, h_bounds, Z);
        read_binary_float(ref_file, h_ref, total);

        float *d_input, *d_output;
        int* d_bounds;
        cudaMalloc(&d_input, total * sizeof(float));
        cudaMalloc(&d_output, total * sizeof(float));
        cudaMalloc(&d_bounds, Z * sizeof(int));

        cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bounds, h_bounds, Z * sizeof(int), cudaMemcpyHostToDevice);

        const int threads = 256;  // Optimal for RTX 3090 Ti
        ragged_softmax_kernel<<<Z, threads>>>(d_input, d_output, d_bounds, Z, S);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, total)) {
            std::cout << "F" << std::endl;
            all_pass = false;
        }

        cudaFree(d_input); cudaFree(d_output); cudaFree(d_bounds);
        delete[] h_input; delete[] h_output; delete[] h_ref; delete[] h_bounds;

        if (!all_pass) break;
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}