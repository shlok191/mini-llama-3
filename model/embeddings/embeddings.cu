#include <cuda.h>
#include <cuda_runtime.h>
#include "embedding.h"

namespace CUDAEmbeddings {

__global__ void embedding_forward_kernel(
    const float* __restrict__ weights,
    const int32_t* __restrict__ indices,
    float* __restrict__ output,
    const int32_t batch_size,
    const int32_t seq_length,
    const int32_t embed_dim,
    const int32_t padding_idx,
    const int32_t total_elements) {
    
    // Calculate global thread index
    const int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 32;

    #pragma unroll
    for (int i = 0; i < 32 && (start_idx + i) < total_elements; i++) {
        const int idx = start_idx + i;

        const int embedding_pos = idx % embed_dim;
        const int seq_pos = (idx / embed_dim) % seq_length;
        const int batch_pos = idx / (embed_dim * seq_length);

        const int token_idx = indices[batch_pos * seq_length + seq_pos];

        if (token_idx == padding_idx) {
            output[idx] = 0.0f;  // Use float value 0.0
        } else {
            output[idx] = weights[token_idx * embed_dim + embedding_pos];
        }
    }
}

__global__ void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ indices,
    float* __restrict__ grad_weight,
    const int32_t batch_size,
    const int32_t seq_length,
    const int32_t embed_dim,
    const int32_t num_embeddings,
    const int32_t padding_idx,
    const int32_t total_elements) {

    const int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 32;

    #pragma unroll
    for (int i = 0; i < 32 && (start_idx + i) < total_elements; i++) {
        
        const int idx = start_idx + i;

        const int embed_pos = idx % embed_dim;
        const int seq_pos = (idx / embed_dim) % seq_length;
        const int batch_pos = idx / (embed_dim * seq_length);

        const int token_idx = indices[batch_pos * seq_length + seq_pos];

        if (token_idx != padding_idx) {
            atomicAdd(&grad_weight[token_idx * embed_dim + embed_pos], grad_output[idx]);
        }
    }
}

}

std::vector<torch::Tensor> embedding_forward_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int32_t padding_token_index) {
    
    auto batch_size = indices.size(0);
    auto seq_length = indices.size(1);
    auto embed_dim = weights.size(1);
    
    auto output = torch::zeros(
        {batch_size, seq_length, embed_dim},
        torch::dtype(torch::kFloat32).device(weights.device())  // Use float32 dtype
    );
    
    // Configuring 1D grid and blocks
    const int threads_per_block = 1024;
    const int work_per_thread = 32;

    const int total_elements = batch_size * seq_length * embed_dim;

    // Calculating total blocks in the grid for 1024 threads/block
    const int num_blocks = (total_elements + (threads_per_block * work_per_thread) - 1) / (threads_per_block * work_per_thread);

    // Call the CUDA kernel
    embedding_forward_kernel<<<num_blocks, threads_per_block>>>(
        weights.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        batch_size,
        seq_length,
        embed_dim,
        padding_token_index,
        total_elements
    );

    return {output};
}

std::vector<torch::Tensor> embedding_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int64_t padding_idx) {

    auto batch_size = indices.size(0);
    auto seq_length = indices.size(1);
    auto embed_dim = weights.size(1);
    auto num_embeddings = weights.size(0);

    auto grad_weight = torch::zeros_like(weights);

    const int total_elements = batch_size * seq_length * embed_dim;

    const int threads_per_block = 1024;
    const int work_per_thread = 32;

    // Number of blocks needed to process all elements
    const int num_blocks = (total_elements + (threads_per_block * work_per_thread) - 1) / (threads_per_block * work_per_thread);

    embedding_backward_kernel<<<num_blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        grad_weight.data_ptr<float>(),
        batch_size,
        seq_length,
        embed_dim,
        num_embeddings,
        padding_idx,
        total_elements
    );

    return {grad_weight};
}