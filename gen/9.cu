#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void dequantize_kernel(const int8_t* __restrict__ input,
                                 float* __restrict__ output,
                                 const float scale,
                                 const int zeroPt,
                                 const size_t N) {
    const size_t warp_stride = blockDim.x * gridDim.x * 4;
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 elements per thread with warp-level optimizations
    if (idx + 3 < N) {
        // Coalesced 128-bit memory access
        const int32_t* input32 = reinterpret_cast<const int32_t*>(input + idx);
        int32_t packed = *input32;
        
        // Unpack 4 int8 values
        int8_t vals[4] = {
            static_cast<int8_t>(packed & 0xFF),
            static_cast<int8_t>((packed >> 8) & 0xFF),
            static_cast<int8_t>((packed >> 16) & 0xFF),
            static_cast<int8_t>((packed >> 24) & 0xFF)
        };
        
        // Compute all 4 values in parallel
        float results[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            results[i] = (vals[i] - zeroPt) * scale;
        }
        
        // Coalesced 128-bit memory write
        float4* output4 = reinterpret_cast<float4*>(output + idx);
        *output4 = make_float4(results[0], results[1], results[2], results[3]);
    }
    else {
        // Handle remainder elements
        for (size_t i = idx; i < min(idx + 4, N); ++i) {
            output[i] = (input[i] - zeroPt) * scale;
        }
    }
}

void read_binary_int8(const std::string& filename, int8_t* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open file: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size);
    in.close();
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open file: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_arrays(const float* a, const float* b, size_t size, float tol = 1e-4f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const float scale = 0.05f;
    const int zeroPt = 0;
    bool all_match = true;

    for (int i = 0; i < Ns.size(); ++i) {
        size_t N = Ns[i];
        std::string in_file = "data/deq_input_" + std::to_string(i+1) + ".bin";
        std::string ref_file = "data/deq_ref_" + std::to_string(i+1) + ".bin";

        int8_t* h_input = new int8_t[N];
        float* h_output = new float[N];
        float* h_ref = new float[N];
        int8_t* d_input;
        float* d_output;

        read_binary_int8(in_file, h_input, N);
        read_binary_float(ref_file, h_ref, N);

        cudaMalloc(&d_input, N * sizeof(int8_t));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMemcpy(d_input, h_input, N * sizeof(int8_t), cudaMemcpyHostToDevice);

        // Optimized launch configuration
        int threads = 256;
        int blocks = min(65535, (static_cast<int>(N) + threads * 4 - 1) / (threads * 4));
        dequantize_kernel<<<blocks, threads>>>(d_input, d_output, scale, zeroPt, N);
        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_arrays(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_output);
            delete[] h_input; delete[] h_output; delete[] h_ref;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        delete[] h_input; delete[] h_output; delete[] h_ref;
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}