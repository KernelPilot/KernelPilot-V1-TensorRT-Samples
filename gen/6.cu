#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <cmath>

#define TILE_SIZE 16
#define KERNEL_SIZE 3

__global__ void conv2d_kernel(const float* __restrict__ input,
                             const float* __restrict__ weight,
                             const float* __restrict__ bias,
                             float* __restrict__ output,
                             int B, int C_in, int H, int W,
                             int C_out, int K) {
    // Calculate output dimensions
    const int OH = H - K + 1;
    const int OW = W - K + 1;
    
    // Block and thread indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Output position
    const int out_h = by * TILE_SIZE + ty;
    const int out_w = bx * TILE_SIZE + tx;
    const int b = blockIdx.z / C_out;
    const int co = blockIdx.z % C_out;
    
    if (b >= B || co >= C_out || out_h >= OH || out_w >= OW) return;
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int ci = 0; ci < C_in; ++ci) {
        // Loop over kernel rows and columns
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h = out_h + kh;
                int w = out_w + kw;
                
                if (h < H && w < W) {
                    // Input index: [b][ci][h][w]
                    size_t in_idx = b * (C_in * H * W) + ci * (H * W) + h * W + w;
                    // Weight index: [co][ci][kh][kw]
                    size_t w_idx = co * (C_in * K * K) + ci * (K * K) + kh * K + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[co];
    }
    
    // Output index: [b][co][out_h][out_w]
    size_t out_idx = b * (C_out * OH * OW) + co * (OH * OW) + out_h * OW + out_w;
    output[out_idx] = sum;
}

void read_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare(const float* a, const float* b, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    const int B = 2, C_in = 3, H = 8, W = 8, C_out = 4, K = 3;
    const int OH = H - K + 1;
    const int OW = W - K + 1;
    const size_t in_size = B * C_in * H * W;
    const size_t w_size = C_out * C_in * K * K;
    const size_t out_size = B * C_out * OH * OW;
    const size_t bias_size = C_out;

    bool all_match = true;
    for (int idx = 1; idx <= 5; ++idx) {
        std::string prefix = "data/conv_" + std::to_string(idx);
        float *h_in = new float[in_size];
        float *h_w  = new float[w_size];
        float *h_b  = new float[bias_size];
        float *h_out_ref = new float[out_size];
        float *h_out = new float[out_size];

        read_float(prefix + "_input.bin", h_in, in_size);
        read_float(prefix + "_weight.bin", h_w, w_size);
        read_float(prefix + "_bias.bin", h_b, bias_size);
        read_float(prefix + "_output.bin", h_out_ref, out_size);

        float *d_in, *d_w, *d_b, *d_out;
        cudaMalloc(&d_in, in_size * sizeof(float));
        cudaMalloc(&d_w, w_size * sizeof(float));
        cudaMalloc(&d_b, bias_size * sizeof(float));
        cudaMalloc(&d_out, out_size * sizeof(float));

        cudaMemcpy(d_in, h_in, in_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w, w_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bias_size * sizeof(float), cudaMemcpyHostToDevice);

        // Configure grid and block dimensions
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((OW + TILE_SIZE - 1) / TILE_SIZE,
                  (OH + TILE_SIZE - 1) / TILE_SIZE,
                  B * C_out);
        
        conv2d_kernel<<<grid, block>>>(d_in, d_w, d_b, d_out, B, C_in, H, W, C_out, K);
        cudaMemcpy(h_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare(h_out, h_out_ref, out_size)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
        delete[] h_in; delete[] h_w; delete[] h_b; delete[] h_out; delete[] h_out_ref;
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}