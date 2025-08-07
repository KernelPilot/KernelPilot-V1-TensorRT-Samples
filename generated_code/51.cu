#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>

template <bool predicate>
__global__ void condition_kernel_template(const float* __restrict__ then_branch, 
                                        const float* __restrict__ else_branch, 
                                        float* __restrict__ output, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = predicate ? then_branch[idx] : else_branch[idx];
    }
}

__global__ void condition_kernel(bool predicate, const float* __restrict__ then_branch, 
                               const float* __restrict__ else_branch, 
                               float* __restrict__ output, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = predicate ? then_branch[idx] : else_branch[idx];
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

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

bool compare_arrays(const float* a, const float* b, int N, float tol = 1e-5f) {
    for (int i = 0; i < N; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        std::string pred_file = "data/cond_pred_" + std::to_string(idx + 1) + ".bin";
        std::string then_file = "data/cond_then_" + std::to_string(idx + 1) + ".bin";
        std::string else_file = "data/cond_else_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file  = "data/cond_ref_"  + std::to_string(idx + 1) + ".bin";

        float *h_then = (float*)malloc(N * sizeof(float));
        float *h_else = (float*)malloc(N * sizeof(float));
        float *h_ref  = (float*)malloc(N * sizeof(float));
        float *h_out  = (float*)malloc(N * sizeof(float));
        int*   h_pred = (int*)malloc(sizeof(int));

        read_binary_float(then_file, h_then, N);
        read_binary_float(else_file, h_else, N);
        read_binary_float(ref_file,  h_ref,  N);
        read_binary_int(pred_file,  h_pred, 1);

        float *d_then, *d_else, *d_out;
        bool *d_pred;
        cudaMalloc(&d_then, N * sizeof(float));
        cudaMalloc(&d_else, N * sizeof(float));
        cudaMalloc(&d_out,  N * sizeof(float));
        cudaMalloc(&d_pred, sizeof(bool));
        cudaMemcpy(d_then, h_then, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_else, h_else, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pred, h_pred, sizeof(bool), cudaMemcpyHostToDevice);

        bool cond_scalar = h_pred[0];
        const int threads = 256;
        const int blocks = min(65535, (static_cast<int>(N) + threads - 1) / threads);
        
        // Use template version for known predicate at compile-time
        if (cond_scalar) {
            condition_kernel_template<true><<<blocks, threads>>>(d_then, d_else, d_out, static_cast<int>(N));
        } else {
            condition_kernel_template<false><<<blocks, threads>>>(d_then, d_else, d_out, static_cast<int>(N));
        }
        
        cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_arrays(h_out, h_ref, static_cast<int>(N))) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        cudaFree(d_then); cudaFree(d_else); cudaFree(d_out); cudaFree(d_pred);
        free(h_then); free(h_else); free(h_out); free(h_pred); free(h_ref);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}