#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>

__global__ void shuffle_kernel(const float* __restrict__ input, float* __restrict__ output, 
                              int N, int C, int H, int W) {
    const int NHW = N * H * W;
    const int CHW = C * H * W;
    const int HW = H * W;
    
    // Each thread handles multiple elements for better utilization
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < NHW * C; 
         idx += blockDim.x * gridDim.x) {
        // Calculate output position in [NHW, C] layout
        int out_nhw = idx / C;
        int out_c = idx % C;
        
        // Calculate input position in original [N, C, H, W] layout
        int in_n = out_nhw / HW;
        int in_hw = out_nhw % HW;
        int in_h = in_hw / W;
        int in_w = in_hw % W;
        
        int input_idx = in_n * CHW + out_c * HW + in_h * W + in_w;
        output[idx] = input[input_idx];
    }
}

void read_binary(const std::string& fname, float* data, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fname << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    const int C = 8, H = 4, W = 4;
    std::vector<int> Ns = {128, 256, 512, 1024, 2048};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t size = N * C * H * W;
        size_t bytes = size * sizeof(float);

        std::string input_file = "data/shuffle_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/shuffle_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        read_binary(input_file, h_input, size);
        read_binary(ref_file, h_ref, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        int threads = 256;
        int blocks = min(65535, (N * C * H * W + threads - 1) / threads);
        shuffle_kernel<<<blocks, threads>>>(d_input, d_output, N, C, H, W);
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare(h_output, h_ref, size)) {
            std::cout << "F\n";
            all_pass = false;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}