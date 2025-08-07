#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void nonzero_kernel(const float* input, int rows, int cols, int* output, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx >= total) return;

    // Check if current element is non-zero
    if (input[idx] != 0.0f) {
        // Atomic increment to get unique position in output
        int pos = atomicAdd(count, 1);
        
        // Calculate row and column indices
        int row = idx / cols;
        int col = idx % cols;
        
        // Store the coordinates in lexicographical order
        output[pos] = row;
        output[pos + total] = col;  // Using total as offset for column storage
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

// test
bool check_equal(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    std::vector<std::pair<int, int>> shapes = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}
    };
    bool all_pass = true;

    for (int test_id = 0; test_id < shapes.size(); ++test_id) {
        int rows = shapes[test_id].first;
        int cols = shapes[test_id].second;
        int total = rows * cols;
        std::string prefix = "nz_" + std::to_string(test_id + 1);

        // Host input
        std::vector<float> h_input(total);
        read_binary_float(prefix + "_in.bin", h_input.data(), total);

        float* d_input;
        int* d_output;
        int* d_count;
        cudaMalloc(&d_input, total * sizeof(float));
        cudaMalloc(&d_output, total * 2 * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(int));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        nonzero_kernel<<<blocks, threads>>>(d_input, rows, cols, d_output, d_count);

        int h_count;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<int> h_output(2 * h_count);
        cudaMemcpy(h_output.data(), d_output, sizeof(int) * 2 * h_count, cudaMemcpyDeviceToHost);

        std::vector<int> h_ref(2 * h_count);
        read_binary_int(prefix + "_out.bin", h_ref.data(), 2 * h_count);

        // test
        if (!check_equal(h_output, h_ref)) {
            std::cout << "F" << std::endl;
            all_pass = false;
            cudaFree(d_input); cudaFree(d_output); cudaFree(d_count);
            break;
        }

        cudaFree(d_input); cudaFree(d_output); cudaFree(d_count);
    }

    if (all_pass) std::cout << "T" << std::endl;
    return 0;
}