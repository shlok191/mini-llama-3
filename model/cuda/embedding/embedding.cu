#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <torch/torch.h>

#define BLOCK_SIZE 32
#define THREAD_SIZE 1

__global__ void embedding_forward_kernel(
    const float* table,
    const int32_t* indices,
    float* output,
    const int32_t seq_length,
    const int32_t embed_dim) {

    // Calculating the global thread index    
    const int thread_idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
   
    // Calculating the token's value this thread must fetch
    const int thread_token_idx = thread_idx * THREAD_SIZE / embed_dim;
   
    // Check sequence position is within bounds
    if (thread_token_idx >= seq_length) {
       return;
    }

    // Finally, copying the elements into the output tensor
    const int output_idx = thread_idx * THREAD_SIZE;
    const int token_offset = output_idx % embed_dim;

    const int table_idx = indices[thread_token_idx] * embed_dim + token_offset;

    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i++){
        output[output_idx + i] = table[table_idx + i];
    }
}

__global__ void embedding_backward_kernel(
    const float* grad_output,
    const int32_t* indices,
    float* grad_table,
    const int32_t seq_length,
    const int32_t embed_dim,
    const int32_t num_embeddings,
    const int32_t total_elements,
    const int32_t padding_idx) {

    // Calculating the global thread index    
    const int thread_idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
   
    // Calculating the token's value this thread must fetch
    const int thread_token_idx = thread_idx * THREAD_SIZE / embed_dim;
   
    // Checking that the sequence position is within bounds
    if (thread_token_idx >= seq_length) {
       return;
    }

    // Skipping the gradient update if this is the padding token!
    if (indices[thread_token_idx] == padding_idx) {
        return;
    }

    // Finally, copying the elements into the output tensor
    const int output_idx = thread_idx * THREAD_SIZE;
    const int token_offset = output_idx % embed_dim;

    const int table_idx = indices[thread_token_idx] * embed_dim + token_offset;

    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i++){
        atomicAdd(&grad_table[table_idx + i], grad_output[output_idx + i]);
    }
}

torch::Tensor embedding_forward_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& table) {
    
    const int batch_size = indices.size(0);
    const int seq_length = indices.size(1);

    const int embed_dim = table.size(1);

    torch::Tensor output = torch::zeros({batch_size, seq_length, embed_dim}, table.device());

    // Creating CUDA streams
    std::vector<cudaStream_t> streams(batch_size);

    // Creating PyTorch streams
    for(int b = 0; b < batch_size; b++){
        cudaStreamCreate(&streams[b]);
    }

    // Configuring a 1D block
    const int total_elements = seq_length * embed_dim;

    // Calculating total blocks in the grid for 1024 threads / block
    const int num_blocks = total_elements / BLOCK_SIZE * THREAD_SIZE;

    for(int b = 0; b < batch_size; b++){
    
        // Call the CUDA kernel
        embedding_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, streams[b]>>>(
            table.data_ptr<float>(),
            indices[b].data_ptr<int32_t>(),
            output[b].data_ptr<float>(),
            seq_length,
            embed_dim);

        cudaStreamSynchronize(streams[b]);
    }

    // Handling the created streams
    for(int b = 0; b < batch_size; b++){

        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    return output;
}

torch::Tensor embedding_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& table,
    const int32_t padding_idx) {

    auto batch_size = indices.size(0);
    auto seq_length = indices.size(1);

    auto embed_dim = table.size(1);
    auto num_embeddings = table.size(0);

    // Creating CUDA streams
    std::vector<cudaStream_t> streams(batch_size);

    // Creating PyTorch streams
    for(int b = 0; b < batch_size; b++){
        cudaStreamCreate(&streams[b]);
    }

    auto grad_weight = torch::zeros({batch_size, num_embeddings, embed_dim}, grad_output.options());

    const int total_elements = seq_length * embed_dim;

    // Number of blocks needed to process all elements
    const int num_blocks = total_elements / (BLOCK_SIZE * THREAD_SIZE);

    for(int i = 0; i < batch_size; i++){

        embedding_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, streams[i]>>>(
            grad_output[i].data_ptr<float>(),
            indices[i].data_ptr<int32_t>(),
            grad_weight[i].data_ptr<float>(),
            seq_length,
            embed_dim,
            num_embeddings,
            total_elements,
            padding_idx
        );
        
        cudaStreamSynchronize(streams[i]);
    }

    for(int b = 0; b < batch_size; b++){

        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    return grad_weight.sum(0);
}