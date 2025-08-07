#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Reduced to fit within shared memory limits

__global__ void einsum_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Output position
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to output matrix
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    const int M = 256;
    const int K = 64;
    const int N = 128;

    std::vector<std::string> A_files = {
        "data/einsum_A_1.bin", "data/einsum_A_2.bin", "data/einsum_A_3.bin", "data/einsum_A_4.bin", "data/einsum_A_5.bin"
    };
    std::vector<std::string> B_files = {
        "data/einsum_B_1.bin", "data/einsum_B_2.bin", "data/einsum_B_3.bin", "data/einsum_B_4.bin", "data/einsum_B_5.bin"
    };
    std::vector<std::string> C_refs = {
        "data/einsum_C_1.bin", "data/einsum_C_2.bin", "data/einsum_C_3.bin", "data/einsum_C_4.bin", "data/einsum_C_5.bin"
    };

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    bool all_pass = true;

    for (int i = 0; i < 5; ++i) {
        float *h_A = new float[size_A];
        float *h_B = new float[size_B];
        float *h_C = new float[size_C];
        float *h_C_ref = new float[size_C];

        read_binary_float(A_files[i], h_A, size_A);
        read_binary_float(B_files[i], h_B, size_B);
        read_binary_float(C_refs[i], h_C_ref, size_C);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A * sizeof(float));
        cudaMalloc(&d_B, size_B * sizeof(float));
        cudaMalloc(&d_C, size_C * sizeof(float));

        cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        einsum_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_arrays(h_C, h_C_ref, size_C)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            break;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}