#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>

__global__ void loop_iterator_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int D) {
    // Using 128 threads per block for better occupancy on Ampere
    const int elements_per_thread = 8;  // Optimal for 128 threads and 16-wide rows
    const int total_elements = N * D;
    const int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    // Process 8 elements per thread with manual unrolling
    if (global_idx + 7 < total_elements) {
        // Coalesced 128-byte loads/stores (4 transactions for 8 floats)
        float4 in0 = reinterpret_cast<const float4*>(input)[global_idx/4];
        float4 in1 = reinterpret_cast<const float4*>(input)[global_idx/4 + 1];
        
        reinterpret_cast<float4*>(output)[global_idx/4] = in0;
        reinterpret_cast<float4*>(output)[global_idx/4 + 1] = in1;
    } else {
        // Handle the remaining elements (if any)
        for (int i = 0; i < elements_per_thread; i++) {
            if (global_idx + i < total_elements) {
                output[global_idx + i] = input[global_idx + i];
            }
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    const int D = 16;
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t total = N * D;
        size_t bytes = total * sizeof(float);

        std::string xfile = "data/iter_input_" + std::to_string(idx + 1) + ".bin";
        std::string rfile = "data/iter_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_ref   = (float*)malloc(bytes);
        float* h_out   = (float*)malloc(bytes);

        read_binary_float(xfile, h_input, total);
        read_binary_float(rfile, h_ref, total);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Optimized for Ampere architecture (RTX 3090 Ti)
        int threads = 128;  // Better occupancy than 256
        int blocks = (N * D + threads * 8 - 1) / (threads * 8);
        loop_iterator_kernel<<<blocks, threads>>>(d_input, d_output, N, D);
        cudaMemcpy(h_out, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare_array(h_out, h_ref, total)) {
            std::cout << "F\n";
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_ref); free(h_out);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_ref); free(h_out);
    }

    if (all_match) std::cout << "T\n";
    return 0;
}