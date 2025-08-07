#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

__global__ void shape_kernel(const int64_t* __restrict__ input_shape, int64_t* __restrict__ output, int ndim) {
    // Warp-strided copy with single instruction per thread
    const int idx = threadIdx.x;
    #pragma unroll
    for (int i = idx; i < ndim; i += blockDim.x) {
        output[i] = input_shape[i];
    }
}

// test
bool compare_array(const int64_t* a, const int64_t* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void read_binary_int64(const std::string& filename, int64_t* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int64_t));
    in.close();
}

int main() {
    std::vector<int> shape_lens = {2, 3, 4, 5, 6};
    bool all_match = true;

    for (int i = 0; i < shape_lens.size(); ++i) {
        int ndim = shape_lens[i];
        std::string shape_file = "data/shape_input_" + std::to_string(i + 1) + ".bin";
        std::string ref_file = "data/shape_ref_" + std::to_string(i + 1) + ".bin";

        std::vector<int64_t> h_input_shape(ndim);
        std::vector<int64_t> h_ref(ndim);
        std::vector<int64_t> h_output(ndim);

        read_binary_int64(shape_file, h_input_shape.data(), ndim);
        read_binary_int64(ref_file, h_ref.data(), ndim);

        int64_t *d_input_shape, *d_output;
        cudaMalloc(&d_input_shape, ndim * sizeof(int64_t));
        cudaMalloc(&d_output, ndim * sizeof(int64_t));
        cudaMemcpy(d_input_shape, h_input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Optimal launch configuration for Ampere
        // Using 256 threads (8 warps) for maximum memory throughput
        // Even for small ndim, extra threads will exit immediately
        shape_kernel<<<1, 256, 0, 0>>>(d_input_shape, d_output, ndim);
        cudaMemcpy(h_output.data(), d_output, ndim * sizeof(int64_t), cudaMemcpyDeviceToHost);

        cudaFree(d_input_shape);
        cudaFree(d_output);

        // test
        if (!compare_array(h_output.data(), h_ref.data(), ndim)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}