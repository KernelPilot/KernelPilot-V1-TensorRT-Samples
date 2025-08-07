#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>

__global__ void linspace_kernel(float* out, float alpha, float beta, int N) {
    // Using grid-stride loop for better load balancing and coalesced memory access
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < N; 
         idx += blockDim.x * gridDim.x) {
        out[idx] = alpha + idx * beta;
    }
}

void read_binary_float(const std::string& fname, float* ptr, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << fname << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(ptr), size * sizeof(float));
    in.close();
}

// test
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t bytes = N * sizeof(float);

        std::string dim_file  = "data/fill_dim_"  + std::to_string(idx + 1) + ".bin";
        std::string ref_file  = "data/fill_ref_"  + std::to_string(idx + 1) + ".bin";
        std::string alpha_file = "data/fill_alpha_" + std::to_string(idx + 1) + ".bin";
        std::string beta_file  = "data/fill_beta_"  + std::to_string(idx + 1) + ".bin";

        float alpha, beta;
        read_binary_float(alpha_file, &alpha, 1);
        read_binary_float(beta_file, &beta, 1);

        float* h_ref = (float*)malloc(bytes);
        float* h_out = (float*)malloc(bytes);
        read_binary_float(ref_file, h_ref, N);

        float* d_out;
        cudaMalloc(&d_out, bytes);

        // Optimized launch configuration for RTX 3090 Ti
        int threads = 256;  // Optimal for Ampere architecture
        int blocks = min((N + threads - 1) / threads, 2048);  // Cap blocks for better occupancy
        
        linspace_kernel<<<blocks, threads>>>(d_out, alpha, beta, N);
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_out, h_ref, N)) {
            std::cout << "F\n";
            all_pass = false;
            cudaFree(d_out);
            free(h_out); free(h_ref);
            break;
        }

        cudaFree(d_out);
        free(h_out); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}