#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

__global__ void trip_limit_kernel(int* __restrict__ out, const int trip_limit, const int N) {
    // Warp-level optimization - each warp handles multiple elements
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread to reduce overhead
    #pragma unroll(4)
    for (; idx < N; idx += stride) {
        // Compiler will optimize this empty loop
        for (volatile int i = 0; i < trip_limit; i++) {
            // Prevent loop elimination by compiler
            asm volatile("" : "+r"(i) : : "memory");
        }
        out[idx] = trip_limit;
    }
}

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), sizeof(int) * size);
    in.close();
}

bool compare_array(const int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        std::string trip_file = "data/trip_limit_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file  = "data/trip_ref_" + std::to_string(idx + 1) + ".bin";

        int trip_limit;
        read_binary_int(trip_file, &trip_limit, 1);

        int* h_ref = (int*)malloc(N * sizeof(int));
        int* h_out = (int*)malloc(N * sizeof(int));
        read_binary_int(ref_file, h_ref, N);

        int* d_out;
        cudaMalloc(&d_out, N * sizeof(int));

        // Optimized launch configuration for RTX 3090 Ti
        const int threads = 256;  // Optimal for Ampere architecture
        const int blocks = min(256, (N + threads - 1) / threads);  // Reduced grid size
        
        // Prefetch to L2 cache
        cudaMemPrefetchAsync(d_out, N * sizeof(int), 0);
        
        trip_limit_kernel<<<blocks, threads>>>(d_out, trip_limit, N);
        cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

        if (!compare_array(h_out, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_out);
            free(h_out); free(h_ref);
            break;
        }

        cudaFree(d_out);
        free(h_out); free(h_ref);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}