#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void conditional_output_kernel(const bool* flags, const float* true_vals, const float* false_vals, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = flags[idx] ? true_vals[idx] : false_vals[idx];
    }
}

void read_binary_bool(const std::string& filename, bool* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size);
    in.close();
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

// test
bool compare_array(const float* a, const float* b, int N, float tol = 1e-3f) {
    for (int i = 0; i < N; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> sizes = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_passed = true;

    for (int idx = 0; idx < sizes.size(); ++idx) {
        int N = sizes[idx];

        std::string prefix = "data/cond_" + std::to_string(idx+1);
        std::string flag_file = prefix + "_flag.bin";
        std::string true_file = prefix + "_true.bin";
        std::string false_file = prefix + "_false.bin";
        std::string ref_file = prefix + "_ref.bin";

        // host memory
        bool* h_flag = new bool[N];
        float* h_true = new float[N];
        float* h_false = new float[N];
        float* h_output = new float[N];
        float* h_ref = new float[N];

        read_binary_bool(flag_file, h_flag, N);
        read_binary_float(true_file, h_true, N);
        read_binary_float(false_file, h_false, N);
        read_binary_float(ref_file, h_ref, N);

        // device memory
        bool* d_flag;
        float *d_true, *d_false, *d_output;
        cudaMalloc(&d_flag, N * sizeof(bool));
        cudaMalloc(&d_true, N * sizeof(float));
        cudaMalloc(&d_false, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        cudaMemcpy(d_flag, h_flag, N * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_true, h_true, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_false, h_false, N * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256, blocks = (N + threads - 1) / threads;
        conditional_output_kernel<<<blocks, threads>>>(d_flag, d_true, d_false, d_output, N);

        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_passed = false;
        }

        cudaFree(d_flag); cudaFree(d_true); cudaFree(d_false); cudaFree(d_output);
        delete[] h_flag; delete[] h_true; delete[] h_false; delete[] h_output; delete[] h_ref;

        if (!all_passed) break;
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}