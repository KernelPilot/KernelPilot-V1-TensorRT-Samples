#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define UNROLL_FACTOR 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int M) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Process multiple elements per thread with unrolling and stride
    #pragma unroll
    for (int i = tid; i < M; i += BLOCK_SIZE * UNROLL_FACTOR) {
        sum += input[row * M + i];
        if (i + BLOCK_SIZE < M) sum += input[row * M + i + BLOCK_SIZE];
        if (i + 2*BLOCK_SIZE < M) sum += input[row * M + i + 2*BLOCK_SIZE];
        if (i + 3*BLOCK_SIZE < M) sum += input[row * M + i + 3*BLOCK_SIZE];
    }

    // Warp-level reduction
    sum = warp_reduce(sum);

    // First thread in warp stores partial sum
    if (tid % 32 == 0) {
        atomicAdd(&output[row], sum);
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

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<std::pair<int, int>> shapes = {
        {1024, 256}, {2048, 512}, {512, 128}, {128, 64}, {256, 1024}
    };

    bool all_match = true;
    for (int idx = 0; idx < shapes.size(); ++idx) {
        int N = shapes[idx].first;
        int M = shapes[idx].second;
        size_t input_size = N * M;
        size_t output_size = N;

        std::string prefix = "data/reduction_";
        std::string input_file = prefix + "input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = prefix + "ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = new float[input_size];
        float* h_ref = new float[output_size];
        float* h_output = new float[output_size];

        read_binary_float(input_file, h_input, input_size);
        read_binary_float(ref_file, h_ref, output_size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));
        cudaMemset(d_output, 0, output_size * sizeof(float));

        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blocks(N);
        dim3 threads(BLOCK_SIZE);
        reduction_kernel<<<blocks, threads>>>(d_input, d_output, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, output_size)) {
            std::cout << "F" << std::endl;
            all_match = false;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        delete[] h_ref;

        if (!all_match) break;
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}