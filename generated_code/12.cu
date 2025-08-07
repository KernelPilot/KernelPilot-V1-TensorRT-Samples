#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>

template <bool BROADCAST_B, int VECTOR_SIZE>
__global__ void elementwise_add_ultimate(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        size_t N) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;
    if (idx >= N) return;

    // Using vectorized loads with const memory optimization
    const float4 a = reinterpret_cast<const float4*>(A)[idx/4];
    float4 c;

    if (BROADCAST_B) {
        // Broadcast optimization with register caching
        const float b = __ldg(B);  // Cache the broadcast value in register
        c.x = __fadd_rn(a.x, b);
        c.y = __fadd_rn(a.y, b);
        c.z = __fadd_rn(a.z, b);
        c.w = __fadd_rn(a.w, b);
    } else {
        // Coalesced memory access with prefetching
        const float4 b = __ldg(reinterpret_cast<const float4*>(B) + idx/4);
        c.x = __fadd_rn(a.x, b.x);
        c.y = __fadd_rn(a.y, b.y);
        c.z = __fadd_rn(a.z, b.z);
        c.w = __fadd_rn(a.w, b.w);
    }

    // Non-temporal store to bypass cache when writing results
    __stwt(reinterpret_cast<float4*>(C) + idx/4, c);
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];

        float* h_A = new float[N];
        float* h_B = (idx % 2 == 0) ? new float[N] : new float[1];
        float* h_ref = new float[N];
        float* h_C = new float[N];

        read_binary_float("data/elemwise_A_" + std::to_string(idx + 1) + ".bin", h_A, N);
        read_binary_float("data/elemwise_B_" + std::to_string(idx + 1) + ".bin", h_B, (idx % 2 == 0) ? N : 1);
        read_binary_float("data/elemwise_C_" + std::to_string(idx + 1) + ".bin", h_ref, N);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, ((idx % 2 == 0) ? N : 1) * sizeof(float));
        cudaMalloc(&d_C, N * sizeof(float));
        cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, ((idx % 2 == 0) ? N : 1) * sizeof(float), cudaMemcpyHostToDevice);

        // Optimal launch configuration for Ampere architecture
        constexpr int VECTOR_SIZE = 4;
        const int threads = 256;  // Optimal for RTX 3090 Ti
        const int blocks = (N / VECTOR_SIZE + threads - 1) / threads;

        // Prefetch data to L2 cache
        cudaMemPrefetchAsync(d_A, N * sizeof(float), 0);
        cudaMemPrefetchAsync(d_B, ((idx % 2 == 0) ? N : 1) * sizeof(float), 0);

        if (idx % 2 != 0) {
            elementwise_add_ultimate<true, VECTOR_SIZE><<<blocks, threads>>>(d_A, d_B, d_C, N);
        } else {
            elementwise_add_ultimate<false, VECTOR_SIZE><<<blocks, threads>>>(d_A, d_B, d_C, N);
        }

        cudaDeviceSynchronize();
        cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_array(h_C, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_ref;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}