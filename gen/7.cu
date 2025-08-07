#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void cumulative_sum_kernel(const float* input, float* output, int N, int C) {
    // Each block processes multiple rows
    const int rows_per_block = 4;
    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    
    if (row >= N) return;
    
    const float* row_input = input + row * C;
    float* row_output = output + row * C;

    // Each thread handles one element (for C=10, we use 10 threads per row)
    if (threadIdx.x < C) {
        float val = row_input[threadIdx.x];
        
        // Warp-level inclusive scan
        for (int offset = 1; offset < C; offset *= 2) {
            float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (threadIdx.x >= offset) val += temp;
        }
        
        row_output[threadIdx.x] = val;
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

bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    const int C = 10;
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t size = N * C;

        std::string input_file = "data/cumsum_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/cumsum_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input  = (float*)malloc(size * sizeof(float));
        float* h_output = (float*)malloc(size * sizeof(float));
        float* h_ref    = (float*)malloc(size * sizeof(float));

        read_binary_float(input_file, h_input, size);
        read_binary_float(ref_file, h_ref, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));

        cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

        // Optimized kernel configuration
        dim3 block(32, 4);  // 32 threads x 4 rows per block
        dim3 grid((N + 3) / 4);  // ceil(N / rows_per_block)
        
        cumulative_sum_kernel<<<grid, block>>>(d_input, d_output, N, C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_arrays(h_output, h_ref, size)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}