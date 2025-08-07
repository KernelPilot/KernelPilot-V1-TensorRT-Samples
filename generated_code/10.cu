#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Optimized absolute max reduction with warp shuffles
__global__ void compute_abs_max_kernel(const float* __restrict__ x, float* max_val, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    
    // Load and process 2 elements per thread
    float local_max = 0.0f;
    if (i < N) local_max = fabsf(x[i]);
    if (i + blockDim.x < N) local_max = fmaxf(local_max, fabsf(x[i + blockDim.x]));
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    
    // Store warp results to shared memory
    if (tid % 32 == 0) {
        sdata[tid / 32] = local_max;
    }
    __syncthreads();
    
    // Final reduction for the block
    if (tid < 32) {
        local_max = tid < blockDim.x / 32 ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
        
        // Atomic update to global memory
        if (tid == 0) {
            atomicMax((int*)max_val, __float_as_int(local_max));
        }
    }
}

// Optimized quantization kernel with vectorized loads
__global__ void quantize_kernel(const float* __restrict__ x, int8_t* __restrict__ q, float scale, int N) {
    int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    float inv_scale = 1.0f / scale;
    
    // Process 4 elements per thread
    if (i < N) {
        float val = x[i] * inv_scale;
        q[i] = static_cast<int8_t>(roundf(fminf(fmaxf(val, -128.0f), 127.0f)));
    }
    if (i + blockDim.x < N) {
        float val = x[i + blockDim.x] * inv_scale;
        q[i + blockDim.x] = static_cast<int8_t>(roundf(fminf(fmaxf(val, -128.0f), 127.0f)));
    }
    if (i + 2*blockDim.x < N) {
        float val = x[i + 2*blockDim.x] * inv_scale;
        q[i + 2*blockDim.x] = static_cast<int8_t>(roundf(fminf(fmaxf(val, -128.0f), 127.0f)));
    }
    if (i + 3*blockDim.x < N) {
        float val = x[i + 3*blockDim.x] * inv_scale;
        q[i + 3*blockDim.x] = static_cast<int8_t>(roundf(fminf(fmaxf(val, -128.0f), 127.0f)));
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void read_binary_int8(const std::string& filename, int8_t* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int8_t));
    in.close();
}

// test
bool compare_int8(const int8_t* a, const int8_t* b, size_t size, int tol = 1) {
    for (size_t i = 0; i < size; ++i) {
        if (abs((int)a[i] - (int)b[i]) > tol) {
            return false;
        }
    }
    return true;
}

// test
bool compare_float(float a, float b, float tol = 1e-2f) {
    return fabs(a - b) < tol;
}

int main() {
    std::vector<int> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    bool all_pass = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int N = Ns[idx];
        size_t fbytes = N * sizeof(float);
        size_t ibytes = N * sizeof(int8_t);

        std::string input_file = "data/quant_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_quant  = "data/quant_ref_"   + std::to_string(idx + 1) + ".bin";
        std::string ref_scale  = "data/quant_scale_" + std::to_string(idx + 1) + ".bin";

        float* h_x     = (float*)malloc(fbytes);
        int8_t* h_qref = (int8_t*)malloc(ibytes);
        float h_scale_ref;

        read_binary_float(input_file, h_x, N);
        read_binary_int8(ref_quant, h_qref, N);
        read_binary_float(ref_scale, &h_scale_ref, 1);

        float *d_x, *d_max;
        int8_t* d_q;
        cudaMalloc(&d_x, fbytes);
        cudaMalloc(&d_q, ibytes);
        cudaMalloc(&d_max, sizeof(float));
        cudaMemcpy(d_x, h_x, fbytes, cudaMemcpyHostToDevice);
        cudaMemset(d_max, 0, sizeof(float));

        // Optimized kernel configuration
        const int threads = 256;
        const int blocks_max = (N + threads * 2 - 1) / (threads * 2);
        const int blocks_quant = (N + threads * 4 - 1) / (threads * 4);
        size_t shared_mem_size = (threads / 32) * sizeof(float);
        
        compute_abs_max_kernel<<<blocks_max, threads, shared_mem_size>>>(d_x, d_max, N);
        
        float h_max;
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        float scale = h_max / 127.0f;

        quantize_kernel<<<blocks_quant, threads>>>(d_x, d_q, scale, N);

        int8_t* h_qout = (int8_t*)malloc(ibytes);
        cudaMemcpy(h_qout, d_q, ibytes, cudaMemcpyDeviceToHost);

        // test
        if (!compare_int8(h_qout, h_qref, N, 1) || !compare_float(scale, h_scale_ref)) {
            std::cout << "F\n";
            all_pass = false;
            cudaFree(d_x); cudaFree(d_q); cudaFree(d_max);
            free(h_x); free(h_qref); free(h_qout);
            break;
        }

        cudaFree(d_x); cudaFree(d_q); cudaFree(d_max);
        free(h_x); free(h_qref); free(h_qout);
    }

    if (all_pass) std::cout << "T\n";
    return 0;
}