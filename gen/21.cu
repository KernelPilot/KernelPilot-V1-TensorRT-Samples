#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

__device__ __forceinline__ float compute_iou(const float* box1, const float* box2) {
    float x1 = fmaxf(box1[0], box2[0]);
    float y1 = fmaxf(box1[1], box2[1]);
    float x2 = fminf(box1[2], box2[2]);
    float y2 = fminf(box1[3], box2[3]);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;
    return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
}

__global__ void nms_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    int* __restrict__ selected_indices,
    const int batch_size,
    const int num_boxes,
    const int num_classes,
    const float iou_threshold,
    const float score_threshold,
    const int max_output_boxes_per_class
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;

    const int batch_idx = idx / num_classes;
    const int class_idx = idx % num_classes;
    const int base_offset = batch_idx * num_boxes * num_classes + class_idx;

    // Step 1: gather scores & indices
    float score_buf[512];  // support up to 512 boxes
    int index_buf[512];
    int count = 0;

    for (int i = 0; i < num_boxes; ++i) {
        const float score = scores[base_offset + i * num_classes];
        if (score >= score_threshold) {
            score_buf[count] = score;
            index_buf[count] = i;
            count++;
        }
    }

    // Step 2: sort index_buf by score_buf in descending order (optimized insertion sort)
    for (int i = 1; i < count; ++i) {
        const float current_score = score_buf[i];
        const int current_index = index_buf[i];
        int j = i - 1;
        
        while (j >= 0 && score_buf[j] < current_score) {
            score_buf[j + 1] = score_buf[j];
            index_buf[j + 1] = index_buf[j];
            j--;
        }
        score_buf[j + 1] = current_score;
        index_buf[j + 1] = current_index;
    }

    // Step 3: NMS on sorted boxes
    bool suppressed[512] = {false};
    int selected_boxes[512];
    int num_selected = 0;

    for (int i = 0; i < count && num_selected < max_output_boxes_per_class; ++i) {
        if (suppressed[i]) continue;
        
        selected_boxes[num_selected++] = index_buf[i];
        
        const float* box_i = boxes + (batch_idx * num_boxes + index_buf[i]) * 4;
        
        for (int j = i + 1; j < count; ++j) {
            if (suppressed[j]) continue;
            
            const float* box_j = boxes + (batch_idx * num_boxes + index_buf[j]) * 4;
            
            const float iou = compute_iou(box_i, box_j);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // Step 4 & 5: write selected indices to output and pad with -1
    const int out_base = batch_idx * num_classes * max_output_boxes_per_class + 
                        class_idx * max_output_boxes_per_class;

    // First fill all with -1
    for (int i = 0; i < max_output_boxes_per_class; ++i) {
        selected_indices[out_base + i] = -1;
    }
    
    // Then write the selected indices
    for (int i = 0; i < num_selected; ++i) {
        selected_indices[out_base + i] = selected_boxes[i];
    }
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

bool compare_arrays(const int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    const int batch_size = 2;
    const int num_boxes = 5;
    const int num_classes = 3;
    const int max_output_boxes_per_class = 5;
    const int num_tests = 5;

    float iou_threshold = 0.5f;
    float score_threshold = 0.0f;

    bool all_passed = true;

    for (int t = 1; t <= num_tests; ++t) {
        float* h_boxes = (float*)malloc(batch_size * num_boxes * 4 * sizeof(float));
        float* h_scores = (float*)malloc(batch_size * num_boxes * num_classes * sizeof(float));
        int* h_selected_indices = (int*)malloc(batch_size * num_classes * max_output_boxes_per_class * sizeof(int));
        int* h_ref_indices = (int*)malloc(batch_size * num_classes * max_output_boxes_per_class * sizeof(int));

        std::string box_file = "data/boxes_" + std::to_string(t) + ".bin";
        std::string score_file = "data/scores_" + std::to_string(t) + ".bin";
        std::string ref_file = "data/ref_indices_" + std::to_string(t) + ".bin";

        read_binary(box_file, h_boxes, batch_size * num_boxes * 4);
        read_binary(score_file, h_scores, batch_size * num_boxes * num_classes);

        std::ifstream in(ref_file, std::ios::binary);
        if (!in) {
            std::cerr << "Missing ref file: " << ref_file << std::endl;
            return 1;
        }
        in.read(reinterpret_cast<char*>(h_ref_indices), batch_size * num_classes * max_output_boxes_per_class * sizeof(int));
        in.close();

        float *d_boxes, *d_scores;
        int *d_selected_indices;

        cudaMalloc(&d_boxes, batch_size * num_boxes * 4 * sizeof(float));
        cudaMalloc(&d_scores, batch_size * num_boxes * num_classes * sizeof(float));
        cudaMalloc(&d_selected_indices, batch_size * num_classes * max_output_boxes_per_class * sizeof(int));

        cudaMemcpy(d_boxes, h_boxes, batch_size * num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scores, h_scores, batch_size * num_boxes * num_classes * sizeof(float), cudaMemcpyHostToDevice);

        const int threads = 256;
        const int blocks = (batch_size * num_classes + threads - 1) / threads;
        nms_kernel<<<blocks, threads>>>(
            d_boxes, d_scores, d_selected_indices,
            batch_size, num_boxes, num_classes,
            iou_threshold, score_threshold, max_output_boxes_per_class
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_selected_indices, d_selected_indices,
                   batch_size * num_classes * max_output_boxes_per_class * sizeof(int),
                   cudaMemcpyDeviceToHost);
        
        bool match = compare_arrays(h_selected_indices, h_ref_indices,
                                    batch_size * num_classes * max_output_boxes_per_class);
        if (!match) {
            all_passed = false;
        }

        free(h_boxes);
        free(h_scores);
        free(h_selected_indices);
        free(h_ref_indices);
        cudaFree(d_boxes);
        cudaFree(d_scores);
        cudaFree(d_selected_indices);
    }

    std::cout << (all_passed ? "T" : "F") << std::endl;
    return 0;
}