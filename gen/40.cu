#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void slice_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const int H, const int W,
    const int start_h, const int start_w,
    const int size_h, const int size_w,
    const int stride_h, const int stride_w) {
    
    // Coalesced memory access pattern
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit for threads outside output bounds
    if (i >= size_h || j >= size_w) return;
    
    // Precompute input offsets
    const int input_row = start_h + i * stride_h;
    const int input_col = start_w + j * stride_w;
    const int input_idx = input_row * W + input_col;
    const int output_idx = i * size_w + j;
    
    // Independent memory operations
    output[output_idx] = input[input_idx];
}

void read_bin(const std::string& fname, float* ptr, size_t size) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fname << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(ptr), size * sizeof(float));
    in.close();
}

bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

int main() {
    std::vector<std::pair<int, int>> shapes = {
        {64, 64}, {128, 128}, {256, 128}, {512, 256}, {1024, 512}
    };

    bool all_pass = true;
    for (int idx = 0; idx < shapes.size(); ++idx) {
        int H = shapes[idx].first;
        int W = shapes[idx].second;

        int start_h = 1, start_w = 2;
        int stride_h = 2, stride_w = 3;
        int size_h = (H - start_h) / stride_h;
        int size_w = (W - start_w) / stride_w;

        size_t input_size = H * W;
        size_t output_size = size_h * size_w;

        std::string in_file  = "data/slice_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/slice_ref_"   + std::to_string(idx + 1) + ".bin";

        float* h_in  = (float*)malloc(input_size * sizeof(float));
        float* h_out = (float*)malloc(output_size * sizeof(float));
        float* h_ref = (float*)malloc(output_size * sizeof(float));
        read_bin(in_file, h_in, input_size);
        read_bin(ref_file, h_ref, output_size);

        float *d_in, *d_out;
        cudaMalloc(&d_in, input_size * sizeof(float));
        cudaMalloc(&d_out, output_size * sizeof(float));
        cudaMemcpy(d_in, h_in, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Optimized block size for RTX 3090 Ti (256 threads per block)
        dim3 threads(16, 16);
        dim3 blocks((size_w + threads.x - 1) / threads.x, 
                   (size_h + threads.y - 1) / threads.y);
        
        // Launch kernel with optimal configuration
        slice_kernel<<<blocks, threads, 0, 0>>>(
            d_in, d_out, H, W, start_h, start_w,
            size_h, size_w, stride_h, stride_w
        );

        cudaMemcpy(h_out, d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (!compare_array(h_out, h_ref, output_size)) {
            std::cout << "F\n";
            all_pass = false;
            cudaFree(d_in); cudaFree(d_out);
            free(h_in); free(h_out); free(h_ref);
            break;
        }

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_out); free(h_ref);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}