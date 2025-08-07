#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void assert_kernel(const bool* __restrict__ input, int N, bool* error_flag) {
    // Warp-level processing (32 threads)
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int warps_per_grid = gridDim.x * (blockDim.x / 32);
    
    bool thread_error = false;
    
    // Process elements with warp-stride loop for coalesced memory access
    for (int idx = global_warp_id * 32 + lane_id; 
         idx < N && !(*error_flag); 
         idx += warps_per_grid * 32) {
        if (!input[idx]) {
            thread_error = true;
            break;
        }
    }
    
    // Warp-level vote for any error
    bool warp_error = __any_sync(0xFFFFFFFF, thread_error);
    
    // First thread in warp updates error flag atomically
    if (warp_error && lane_id == 0) {
        atomicOr(reinterpret_cast<unsigned int*>(error_flag), 1u);
    }
}

void read_binary_bool(const std::string& filename, bool* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(bool));
    in.close();
}

void read_binary_int(const std::string& filename, int* data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), sizeof(int));
    in.close();
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_match = true;

    for (int i = 0; i < Ns.size(); ++i) {
        size_t N = Ns[i];
        std::string input_file = "data/assert_input_" + std::to_string(i + 1) + ".bin";
        std::string label_file = "data/assert_label_" + std::to_string(i + 1) + ".bin";

        // Host
        bool* h_input = (bool*)malloc(N * sizeof(bool));
        int h_label;

        read_binary_bool(input_file, h_input, N);
        read_binary_int(label_file, &h_label);

        // Device
        bool* d_input;
        bool* d_error_flag;
        bool h_error_flag = false;

        cudaMalloc(&d_input, N * sizeof(bool));
        cudaMalloc(&d_error_flag, sizeof(bool));

        cudaMemcpy(d_input, h_input, N * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_error_flag, &h_error_flag, sizeof(bool), cudaMemcpyHostToDevice);

        // Optimal configuration for RTX 3090 Ti
        int threads = 256;  // 8 warps per block
        int blocks = min(1024, (static_cast<int>(N) + threads - 1) / threads);
        assert_kernel<<<blocks, threads>>>(d_input, static_cast<int>(N), d_error_flag);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_error_flag, d_error_flag, sizeof(bool), cudaMemcpyDeviceToHost);

        bool pass = !h_error_flag;
        if ((int)pass != h_label) {
            all_match = false;
        }

        cudaFree(d_input);
        cudaFree(d_error_flag);
        free(h_input);
    }

    std::cout << (all_match ? "T" : "F") << std::endl;
    return 0;
}