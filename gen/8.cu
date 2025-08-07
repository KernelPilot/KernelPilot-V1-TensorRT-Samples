#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 3

__global__ void deconv2d_kernel(const float* __restrict__ input, 
                               const float* __restrict__ kernel, 
                               float* __restrict__ output,
                               int H_in, int W_in, int K_h, int K_w) {
    // Shared memory for kernel
    __shared__ float s_kernel[KERNEL_SIZE * KERNEL_SIZE];
    
    // Load kernel into shared memory (coalesced)
    if (threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE) {
        s_kernel[threadIdx.y * KERNEL_SIZE + threadIdx.x] = 
            kernel[threadIdx.y * KERNEL_SIZE + threadIdx.x];
    }
    __syncthreads();

    // Calculate output coordinates
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within output bounds
    if (x_out >= (W_in + K_w - 1) || y_out >= (H_in + K_h - 1)) {
        return;
    }

    // Initialize sum
    float sum = 0.0f;

    // Calculate bounds for input access
    const int y_in_start = max(0, y_out - K_h + 1);
    const int y_in_end = min(H_in - 1, y_out);
    const int x_in_start = max(0, x_out - K_w + 1);
    const int x_in_end = min(W_in - 1, x_out);

    // Unrolled loop for kernel height (K_h = 3)
    #pragma unroll
    for (int y_in = y_in_start; y_in <= y_in_end; ++y_in) {
        const int k_y = y_out - y_in;
        const float* input_row = input + y_in * W_in;
        
        // Unrolled loop for kernel width (K_w = 3)
        #pragma unroll
        for (int x_in = x_in_start; x_in <= x_in_end; ++x_in) {
            const int k_x = x_out - x_in;
            sum += input_row[x_in] * s_kernel[k_y * KERNEL_SIZE + k_x];
        }
    }

    // Write the result to output
    output[y_out * (W_in + K_w - 1) + x_out] = sum;
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
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
    const int H = 32, W = 32;
    const int K_h = 3, K_w = 3;
    const int H_out = H + K_h - 1;
    const int W_out = W + K_w - 1;

    size_t in_size = H * W;
    size_t out_size = H_out * W_out;
    size_t kernel_size = K_h * K_w;

    float h_kernel[kernel_size];
    read_binary_float("data/deconv2d_kernel.bin", h_kernel, kernel_size);

    float* d_kernel;
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    bool all_pass = true;
    for (int i = 0; i < 5; ++i) {
        std::string input_file = "data/deconv2d_input_" + std::to_string(i+1) + ".bin";
        std::string ref_file   = "data/deconv2d_ref_" + std::to_string(i+1) + ".bin";

        float *h_input = new float[in_size];
        float *h_output = new float[out_size];
        float *h_ref = new float[out_size];
        float *d_input, *d_output;

        read_binary_float(input_file, h_input, in_size);
        read_binary_float(ref_file, h_ref, out_size);

        cudaMalloc(&d_input, in_size * sizeof(float));
        cudaMalloc(&d_output, out_size * sizeof(float));
        cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, out_size * sizeof(float));

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((W_out + TILE_SIZE - 1)/TILE_SIZE, (H_out + TILE_SIZE - 1)/TILE_SIZE);
        deconv2d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output, H, W, K_h, K_w);
        cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        if (!compare_arrays(h_output, h_ref, out_size)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output);
            delete[] h_input; delete[] h_output; delete[] h_ref;
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        delete[] h_input; delete[] h_output; delete[] h_ref;
    }

    cudaFree(d_kernel);
    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}