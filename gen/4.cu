#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

template <int ELEMENTS_PER_THREAD, int C1, int C2>
__global__ void concat_kernel_optimized(const float* __restrict__ input1, 
                                      const float* __restrict__ input2, 
                                      float* __restrict__ output,
                                      const int N) {
    constexpr int C = C1 + C2;
    const int total_elements = N * C;
    const int tid = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;
    
    // Pre-compute constants for the warp
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int elements_per_warp = 32 * ELEMENTS_PER_THREAD;
    const int warp_offset = blockIdx.x * blockDim.x / 32 * elements_per_warp + warp_id * elements_per_warp;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int element_idx = lane_id + i * 32;
        const int idx = warp_offset + element_idx;
        
        if (idx < total_elements) {
            const int n = idx / C;
            const int c = idx % C;
            
            // Branchless load using conditional move
            const float* src = c < C1 ? input1 : input2;
            const int src_c = c < C1 ? c : (c - C1);
            const int src_stride = c < C1 ? C1 : C2;
            
            // Use vectorized load/store when possible
            output[idx] = src[n * src_stride + src_c];
        }
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

bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    constexpr int C1 = 6;
    constexpr int C2 = 4;
    constexpr int C = C1 + C2;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t in1_size = N * C1;
        size_t in2_size = N * C2;
        size_t out_size = N * C;

        std::string in1_file = "data/concat_input1_" + std::to_string(idx + 1) + ".bin";
        std::string in2_file = "data/concat_input2_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/concat_ref_"    + std::to_string(idx + 1) + ".bin";

        float* h_input1 = (float*)malloc(in1_size * sizeof(float));
        float* h_input2 = (float*)malloc(in2_size * sizeof(float));
        float* h_ref    = (float*)malloc(out_size * sizeof(float));
        float* h_output = (float*)malloc(out_size * sizeof(float));

        read_binary_float(in1_file, h_input1, in1_size);
        read_binary_float(in2_file, h_input2, in2_size);
        read_binary_float(ref_file, h_ref, out_size);

        float *d_input1, *d_input2, *d_output;
        cudaMalloc(&d_input1, in1_size * sizeof(float));
        cudaMalloc(&d_input2, in2_size * sizeof(float));
        cudaMalloc(&d_output, out_size * sizeof(float));

        cudaMemcpy(d_input1, h_input1, in1_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, h_input2, in2_size * sizeof(float), cudaMemcpyHostToDevice);

        // Optimal configuration for RTX 3090 Ti
        constexpr int ELEMENTS_PER_THREAD = 8;
        constexpr int THREADS_PER_BLOCK = 256;
        const int blocks = (out_size + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) / 
                          (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);
        
        concat_kernel_optimized<ELEMENTS_PER_THREAD, C1, C2>
            <<<blocks, THREADS_PER_BLOCK>>>(d_input1, d_input2, d_output, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_arrays(h_output, h_ref, out_size)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_input1); cudaFree(d_input2); cudaFree(d_output);
            free(h_input1); free(h_input2); free(h_ref); free(h_output);
            break;
        }

        cudaFree(d_input1); cudaFree(d_input2); cudaFree(d_output);
        free(h_input1); free(h_input2); free(h_ref); free(h_output);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}