#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define C 10

// Optimized nearest neighbor interpolation kernel
__global__ void nearest_neighbor_resize(const float* __restrict__ input, float* __restrict__ output, 
                                       int inputHeight, int inputWidth, int outputHeight, int outputWidth) {
    // Using 2D grid-stride loop for better occupancy
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outputWidth || y >= outputHeight) return;

    // Precompute ratios to avoid redundant calculations
    const float x_ratio = __fdividef(static_cast<float>(inputWidth), outputWidth);
    const float y_ratio = __fdividef(static_cast<float>(inputHeight), outputHeight);
    
    // Calculate source coordinates with fast integer conversion
    const int src_x = min(static_cast<int>(x * x_ratio), inputWidth - 1);
    const int src_y = min(static_cast<int>(y * y_ratio), inputHeight - 1);
    
    // Coalesced memory access pattern
    output[y * outputWidth + x] = input[src_y * inputWidth + src_x];
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
bool compare_array(const float* a, const float* b, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    bool all_match = true;
    
    // Test 5 sets of data
    for (int idx = 0; idx < 5; ++idx) {
        int inputHeight = 10;
        int inputWidth = 10;
        int outputHeight = 20;
        int outputWidth = 20;

        float* h_input = new float[inputHeight * inputWidth];
        float* h_output = new float[outputHeight * outputWidth];
        float* h_ref = new float[outputHeight * outputWidth];

        // Initialize input tensor with random values
        for (int i = 0; i < inputHeight * inputWidth; ++i) {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Perform nearest neighbor resizing using CPU for reference
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                int nearestX = min(inputWidth - 1, max(0, x * inputWidth / outputWidth));
                int nearestY = min(inputHeight - 1, max(0, y * inputHeight / outputHeight));
                h_ref[y * outputWidth + x] = h_input[nearestY * inputWidth + nearestX];
            }
        }

        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, inputHeight * inputWidth * sizeof(float));
        cudaMalloc(&d_output, outputHeight * outputWidth * sizeof(float));
        cudaMemcpy(d_input, h_input, inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);

        // Optimized kernel launch parameters for RTX 3090 Ti
        // Using 256 threads per block (16x16) for better occupancy
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the optimized kernel
        nearest_neighbor_resize<<<numBlocks, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth, outputHeight, outputWidth);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, outputHeight * outputWidth * sizeof(float), cudaMemcpyDeviceToHost);

        // test
        // Compare results
        if (!compare_array(h_output, h_ref, outputHeight * outputWidth)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        delete[] h_ref;
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}