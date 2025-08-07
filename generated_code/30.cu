#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float quantize(float val, float scale) {
    float scaled = val / scale;
    float rounded = rintf(scaled);  // round-to-nearest, ties-to-even
    rounded = fminf(fmaxf(rounded, -128.0f), 127.0f);  // clamp to [-128, 127]
    return rounded * scale;
}

__global__ void quantize_kernel(const float* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = quantize(input[idx], scale);
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
}

// test
bool compare_outputs(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    std::vector<size_t> sizes = {64 * 64, 128 * 128, 256 * 256, 512 * 512, 1024 * 1024};
    bool all_pass = true;

    for (int test_id = 0; test_id < sizes.size(); ++test_id) {
        size_t N = sizes[test_id];
        float scale;

        float* h_input = new float[N];
        float* h_output_ref = new float[N];
        float* h_output = new float[N];

        std::string prefix = "data/quant_" + std::to_string(test_id + 1);
        read_binary_float(prefix + "_input.bin", h_input, N);
        read_binary_float(prefix + "_scale.bin", &scale, 1);
        read_binary_float(prefix + "_ref.bin", h_output_ref, N);

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        quantize_kernel<<<blocks, threads>>>(d_input, d_output, scale, N);
        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_outputs(h_output, h_output_ref, N)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output);
            delete[] h_input; delete[] h_output; delete[] h_output_ref;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        delete[] h_input; delete[] h_output; delete[] h_output_ref;
    }

    if (all_pass)
        std::cout << "T" << std::endl;

    return 0;
}