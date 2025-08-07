#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void select_kernel(const int* __restrict__ cond, 
                             const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ output,
                             int N) {
    // Process 4 elements per thread for better memory throughput
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 3 < N) {
        // Load 4 elements at once
        int4 c = reinterpret_cast<const int4*>(cond)[idx/4];
        float4 x_vals = reinterpret_cast<const float4*>(x)[idx/4];
        float4 y_vals = reinterpret_cast<const float4*>(y)[idx/4];
        
        // Process vectorized elements
        float4 result;
        result.x = c.x ? x_vals.x : y_vals.x;
        result.y = c.y ? x_vals.y : y_vals.y;
        result.z = c.z ? x_vals.z : y_vals.z;
        result.w = c.w ? x_vals.w : y_vals.w;
        
        reinterpret_cast<float4*>(output)[idx/4] = result;
    } else {
        // Handle remaining elements
        for (int i = idx; i < min(idx + 4, N); i++) {
            output[i] = cond[i] ? x[i] : y[i];
        }
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

// test
bool compare_array(const float* a, const float* b, size_t N, float tol = 1e-3f) {
    for (size_t i = 0; i < N; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];

        std::string cond_file = "data/select_cond_" + std::to_string(idx + 1) + ".bin";
        std::string x_file = "data/select_x_" + std::to_string(idx + 1) + ".bin";
        std::string y_file = "data/select_y_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/select_ref_" + std::to_string(idx + 1) + ".bin";

        std::vector<int> h_cond(N);
        std::vector<float> h_x(N), h_y(N), h_ref(N), h_out(N);

        read_binary_int(cond_file, h_cond.data(), N);
        read_binary_float(x_file, h_x.data(), N);
        read_binary_float(y_file, h_y.data(), N);
        read_binary_float(ref_file, h_ref.data(), N);

        int *d_cond;
        float *d_x, *d_y, *d_out;
        cudaMalloc(&d_cond, N * sizeof(int));
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));

        cudaMemcpy(d_cond, h_cond.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        const int threads = 256;
        const int blocks = min(65535, (static_cast<int>(N) + 4*threads - 1) / (4*threads));
        select_kernel<<<blocks, threads>>>(d_cond, d_x, d_y, d_out, static_cast<int>(N));
        cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_out.data(), h_ref.data(), N)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_cond); cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
            break;
        }

        cudaFree(d_cond); cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}