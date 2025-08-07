#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>

__global__ void loop_output_concat_kernel(const float* inputs, float* output, int K, int D) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread index is within the bounds of the output tensor
    if (idx < K * D) {
        // Calculate the iteration (k) and dimension (d) indices
        int k = idx / D;  // which iteration (row in output)
        int d = idx % D;   // which dimension (column in output)
        
        // The input for iteration k starts at k * D
        // We just need to copy the input values to the output in the same order
        output[idx] = inputs[k * D + d];
    }
}

void read_binary(const std::string& fname, float* data, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << fname << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    const int D = 128;
    std::vector<int> Ks = {8, 16, 32, 64, 128};
    bool all_pass = true;

    for (int idx = 0; idx < Ks.size(); ++idx) {
        int K = Ks[idx];
        size_t size = K * D;
        size_t bytes = size * sizeof(float);

        std::string in_file  = "data/loop_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/loop_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_input  = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref    = (float*)malloc(bytes);

        read_binary(in_file, h_input, size);
        read_binary(ref_file, h_ref, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        loop_output_concat_kernel<<<blocks, threads>>>(d_input, d_output, K, D);
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare(h_output, h_ref, size)) {
            std::cout << "F\n";
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}