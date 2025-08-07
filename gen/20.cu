#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for matrix multiplication: C = A x B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the row index of C and A
    int row = by * blockDim.y + ty;
    // Calculate the column index of C and B
    int col = bx * blockDim.x + tx;
    
    // Shared memory for tiles of A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    float sum = 0.0f;
    
    // Loop over the tiles of A and B required to compute C element
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Load A tile into shared memory
        if (row < M && (t * 16 + tx) < K) {
            As[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile into shared memory
        if (col < N && (t * 16 + ty) < K) {
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure the tiles are loaded
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize to prevent race conditions
        __syncthreads();
    }
    
    // Write the result to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    // Avoid std::tuple, use parallel arrays
    const int test_cases = 5;
    int Ms[test_cases] = {128, 256, 512, 1024, 2048};
    int Ns[test_cases] = {256, 512, 128, 64, 128};
    int Ks[test_cases] = {64, 128, 256, 128, 64};

    bool all_passed = true;

    for (int idx = 0; idx < test_cases; ++idx) {
        int M = Ms[idx];
        int N = Ns[idx];
        int K = Ks[idx];

        std::string A_file = "data/mm_A_" + std::to_string(idx + 1) + ".bin";
        std::string B_file = "data/mm_B_" + std::to_string(idx + 1) + ".bin";
        std::string C_ref_file = "data/mm_C_" + std::to_string(idx + 1) + ".bin";

        size_t A_size = M * K;
        size_t B_size = K * N;
        size_t C_size = M * N;

        float* h_A = (float*)malloc(A_size * sizeof(float));
        float* h_B = (float*)malloc(B_size * sizeof(float));
        float* h_C = (float*)malloc(C_size * sizeof(float));
        float* h_C_ref = (float*)malloc(C_size * sizeof(float));

        read_binary_float(A_file, h_A, A_size);
        read_binary_float(B_file, h_B, B_size);
        read_binary_float(C_ref_file, h_C_ref, C_size);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, A_size * sizeof(float));
        cudaMalloc(&d_B, B_size * sizeof(float));
        cudaMalloc(&d_C, C_size * sizeof(float));

        cudaMemcpy(d_A, h_A, A_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, B_size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);
        matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare(h_C, h_C_ref, C_size)) {
            std::cout << "F\n";
            all_passed = false;
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            free(h_A); free(h_B); free(h_C); free(h_C_ref);
            break;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
    }

    if (all_passed) std::cout << "T\n";
    return 0;
}