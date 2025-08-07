#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void constant_fill_kernel(float* output, float value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = value;
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const float constant_value = 42.195f;
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        std::string ref_file = "data/const_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_output = (float*)malloc(N * sizeof(float));
        float* h_ref = (float*)malloc(N * sizeof(float));

        float* d_output;
        cudaMalloc(&d_output, N * sizeof(float));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        constant_fill_kernel<<<blocks, threads>>>(d_output, constant_value, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
        read_binary_float(ref_file, h_ref, N);

        // test
        if (!compare_arrays(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_output);
            free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_output);
        free(h_output); free(h_ref);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}