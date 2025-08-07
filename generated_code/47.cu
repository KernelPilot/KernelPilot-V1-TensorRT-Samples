#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>

__global__ void recurrence_kernel(const int* init, const int* step, int* output, int N, int trip_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = init[idx] + trip_count * step[idx];
    }
}

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

// test
bool compare_array(const int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    int trip_count = 100;
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t bytes = N * sizeof(int);

        std::string init_file = "data/rec_init_" + std::to_string(idx + 1) + ".bin";
        std::string step_file = "data/rec_step_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file  = "data/rec_ref_"  + std::to_string(idx + 1) + ".bin";

        int* h_init = (int*)malloc(bytes);
        int* h_step = (int*)malloc(bytes);
        int* h_ref  = (int*)malloc(bytes);
        int* h_out  = (int*)malloc(bytes);

        read_binary_int(init_file, h_init, N);
        read_binary_int(step_file, h_step, N);
        read_binary_int(ref_file, h_ref, N);

        int *d_init, *d_step, *d_out;
        cudaMalloc(&d_init, bytes);
        cudaMalloc(&d_step, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_init, h_init, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_step, h_step, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        recurrence_kernel<<<blocks, threads>>>(d_init, d_step, d_out, N, trip_count);

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_array(h_out, h_ref, N)) {
            std::cout << "F\n";
            all_match = false;
            break;
        }

        cudaFree(d_init); cudaFree(d_step); cudaFree(d_out);
        free(h_init); free(h_step); free(h_ref); free(h_out);
    }

    if (all_match) std::cout << "T\n";
    return 0;
}