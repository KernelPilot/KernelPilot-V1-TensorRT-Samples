#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

template <int ELEMENTS_PER_THREAD>
__launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM
__global__ void scale_kernel(const float* __restrict__ input, float* __restrict__ output,
                           const float* __restrict__ scale, const float* __restrict__ shift,
                           const float power, const int N, const int channels) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int channel_mask = channels - 1;
    
    // Precompute power-related values
    const bool use_power = (power != 1.0f);
    const float p = power;
    
    // Preload scale and shift values into registers
    float s[ELEMENTS_PER_THREAD], sh[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * stride;
        if (idx >= N) continue;
        int channel_idx = idx & channel_mask;
        s[i] = scale ? __ldg(&scale[channel_idx]) : 1.0f;
        sh[i] = shift ? __ldg(&shift[channel_idx]) : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * stride;
        if (idx >= N) return;
        
        // Load input value
        float val = __ldg(&input[idx]);
        
        // Compute the result
        float result = val * s[i] + sh[i];
        if (use_power) {
            result = __powf(result, p);
        }
        
        // Store the result
        output[idx] = result;
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    std::vector<int> Ns = {64, 128, 256, 512, 1024};
    int channels = 128;
    bool all_passed = true;

    // Verify channels is power of two for optimized indexing
    if ((channels & (channels - 1)) != 0) {
        std::cerr << "Error: channels must be power of two for optimal performance" << std::endl;
        return 1;
    }

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int B = Ns[idx];
        int total_size = B * channels;
        
        std::string input_file = "data/scale_input_" + std::to_string(idx + 1) + ".bin";
        std::string scale_file  = "data/scale_scale_" + std::to_string(idx + 1) + ".bin";
        std::string shift_file  = "data/scale_shift_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file    = "data/scale_ref_" + std::to_string(idx + 1) + ".bin";

        float *h_input = (float*)malloc(total_size * sizeof(float));
        float *h_scale = (float*)malloc(channels * sizeof(float));
        float *h_shift = (float*)malloc(channels * sizeof(float));
        float *h_ref   = (float*)malloc(total_size * sizeof(float));
        float* h_output = (float*)malloc(total_size * sizeof(float));

        read_binary_float(input_file, h_input, total_size);
        read_binary_float(scale_file, h_scale, channels);
        read_binary_float(shift_file, h_shift, channels);
        read_binary_float(ref_file, h_ref, total_size);

        float *d_input, *d_output, *d_scale, *d_shift;
        cudaMalloc(&d_input, total_size * sizeof(float));
        cudaMalloc(&d_output, total_size * sizeof(float));
        cudaMalloc(&d_scale, channels * sizeof(float));
        cudaMalloc(&d_shift, channels * sizeof(float));

        cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scale, h_scale, channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_shift, h_shift, channels * sizeof(float), cudaMemcpyHostToDevice);

        float power = 1.0f;
        
        // Optimal configuration for RTX 3090 Ti
        const int elements_per_thread = 4;
        const int threads_per_block = 256;
        
        // Calculate optimal block count
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, 
            scale_kernel<elements_per_thread>, threads_per_block, 0);
        int optimal_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        
        scale_kernel<elements_per_thread><<<optimal_blocks, threads_per_block>>>(
            d_input, d_output, d_scale, d_shift, power, total_size, channels);
        
        cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, total_size)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            break;
        }

        cudaFree(d_input); cudaFree(d_output); cudaFree(d_scale); cudaFree(d_shift);
        free(h_input); free(h_output); free(h_scale); free(h_shift); free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}