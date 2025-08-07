#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_tensor(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

__global__ void padding_kernel(const float* __restrict__ input, float* __restrict__ output,
                              const int N, const int C, const int H, const int W,
                              const int out_H, const int out_W,
                              const int pad_top, const int pad_left) {
    // Warp-level optimization with 4-element vectorization
    constexpr int VEC_SIZE = 4;
    const int total_elements = N * C * out_H * out_W;
    const int vectorized_elements = total_elements / VEC_SIZE;
    
    // Precompute all constants for minimal in-loop calculations
    const int C_out_H_out_W = C * out_H * out_W;
    const int out_H_out_W = out_H * out_W;
    const int C_H_W = C * H * W;
    const int H_W = H * W;
    
    // Process multiple elements per thread with vectorized loads/stores
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         vec_idx < vectorized_elements; 
         vec_idx += gridDim.x * blockDim.x) {
        const int idx = vec_idx * VEC_SIZE;
        
        // Calculate base indices for vectorized processing
        const int n = idx / C_out_H_out_W;
        const int c = (idx % C_out_H_out_W) / out_H_out_W;
        const int h = (idx % out_H_out_W) / out_W;
        const int w = idx % out_W;

        // Vectorized boundary checking and loading
        float4 out_val = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            const int curr_w = w + v;
            if (curr_w < out_W) {  // Prevent out-of-bounds in vectorization
                const int in_h = h - pad_top;
                const int in_w = curr_w - pad_left;
                
                if ((unsigned)in_h < (unsigned)H && (unsigned)in_w < (unsigned)W) {
                    const int input_idx = n * C_H_W + c * H_W + in_h * W + in_w;
                    ((float*)&out_val)[v] = __ldg(&input[input_idx]);
                }
            }
        }
        
        // Vectorized store
        if (idx + VEC_SIZE - 1 < total_elements) {
            *((float4*)(&output[idx])) = out_val;
        } else {
            // Handle remainder elements
            for (int v = 0; v < VEC_SIZE && idx + v < total_elements; ++v) {
                output[idx + v] = ((float*)&out_val)[v];
            }
        }
    }
    
    // Handle remaining elements that didn't fit in vectorization
    const int remaining_start = vectorized_elements * VEC_SIZE;
    for (int idx = remaining_start + blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += gridDim.x * blockDim.x) {
        const int n = idx / C_out_H_out_W;
        const int c = (idx % C_out_H_out_W) / out_H_out_W;
        const int h = (idx % out_H_out_W) / out_W;
        const int w = idx % out_W;

        const int in_h = h - pad_top;
        const int in_w = w - pad_left;
        
        output[idx] = ((unsigned)in_h < (unsigned)H && (unsigned)in_w < (unsigned)W) ?
                     __ldg(&input[n * C_H_W + c * H_W + in_h * W + in_w]) : 0.0f;
    }
}

int main() {
    std::vector<int> sizes = {64, 256, 1024, 4096, 8192};
    int N = 1, C = 3, H = 8, W = 8;
    int pad_top = 1, pad_bottom = -1, pad_left = 2, pad_right = -2;

    bool all_pass = true;

    for (int i = 0; i < sizes.size(); ++i) {
        int batch = sizes[i];
        int in_size = batch * C * H * W;
        int out_H = H + pad_top + pad_bottom;
        int out_W = W + pad_left + pad_right;
        int out_size = batch * C * out_H * out_W;

        std::string in_file = "data/pad_input_" + std::to_string(i + 1) + ".bin";
        std::string ref_file = "data/pad_output_" + std::to_string(i + 1) + ".bin";

        float* h_input = (float*)malloc(in_size * sizeof(float));
        float* h_output = (float*)malloc(out_size * sizeof(float));
        float* h_ref = (float*)malloc(out_size * sizeof(float));

        read_binary_float(in_file, h_input, in_size);
        read_binary_float(ref_file, h_ref, out_size);

        float* d_input, *d_output;
        cudaMalloc(&d_input, in_size * sizeof(float));
        cudaMalloc(&d_output, out_size * sizeof(float));
        cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, out_size * sizeof(float));

        // Optimal launch configuration for RTX 3090 Ti
        const int threads_per_block = 256;
        const int max_blocks = 65535;
        const int blocks = min(max_blocks, (out_size + threads_per_block * 4 - 1) / (threads_per_block * 4));
        
        padding_kernel<<<blocks, threads_per_block>>>(d_input, d_output,
                                                    batch, C, H, W,
                                                    out_H, out_W,
                                                    pad_top, pad_left);

        cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_tensor(h_output, h_ref, out_size)) {
            std::cout << "F" << std::endl;
            all_pass = false;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
        if (!all_pass) break;
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}