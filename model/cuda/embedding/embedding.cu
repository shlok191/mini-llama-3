#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#define BLOCK_SIZE 512
#define THREAD_SIZE 2

__global__ void embedding_forward_kernel(
   const float* __restrict__ table,
   const int32_t* __restrict__ indices,
   float* __restrict__ output,
   const int32_t seq_length,
   const int32_t embed_dim) {
    
   // Calculate global thread index 
   const int start_index = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
   
   // Each embedding is finished by embed_dim / thread_size threads :)
   const int sequence_position = start_index / (embed_dim / THREAD_SIZE);
   
   // Check sequence position is within bounds
   if (sequence_position >= seq_length) {
       return;
   }
   
   // Each thread will process its index from the embedding vector index * thread_size
   const int sequence_offset = (start_index % (embed_dim / THREAD_SIZE)) * THREAD_SIZE;
      
   // Finally, copying the elements into the output tensor
   const int output_index = sequence_position * embed_dim + sequence_offset;
   const int vocab_idx = indices[sequence_position];
   const int table_index = vocab_idx * embed_dim + sequence_offset;

   #pragma unroll
   for (int i = 0; i < THREAD_SIZE; i++) {
        output[output_index + i] = table[table_index + i];
   }
}

    
__global__ void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ indices,
    float* __restrict__ grad_weight,
    const int32_t seq_length,
    const int32_t embed_dim,
    const int32_t num_embeddings,
    const int32_t total_elements) {

    const int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 32;

    #pragma unroll
    for (int i = 0; i < 32 && (start_idx + i) < total_elements; i++) {
        
        const int idx = start_idx + i;

        const int embed_pos = idx % embed_dim;
        const int seq_pos = (idx / embed_dim) % seq_length;
        const int batch_pos = idx / (embed_dim * seq_length);

        const int token_idx = indices[batch_pos * seq_length + seq_pos];

        atomicAdd(&grad_weight[token_idx * embed_dim + embed_pos], grad_output[idx]);
    }
}

torch::Tensor embedding_forward_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& table) {
    
    const int seq_length = indices.size(0);
    const int  embed_dim = table.size(1);
    torch::Tensor output = torch::zeros({seq_length, embed_dim},table.device());
    
    // Configuring a 1D block
    const int total_elements = seq_length * embed_dim;

    // Calculating total blocks in the grid for 1024 threads/block
    const int num_blocks = (total_elements + (BLOCK_SIZE * THREAD_SIZE) - 1) / (BLOCK_SIZE * THREAD_SIZE);

    // Call the CUDA kernel
    embedding_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        table.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_length,
        embed_dim);

    return output;
}

torch::Tensor embedding_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& table) {

    auto seq_length = indices.size(1);
    auto embed_dim = table.size(1);
    auto num_embeddings = table.size(0);

    auto grad_weight = torch::zeros_like(table);

    const int total_elements = seq_length * embed_dim;

    // Number of blocks needed to process all elements
    const int num_blocks = (total_elements + (BLOCK_SIZE * THREAD_SIZE) - 1) / (BLOCK_SIZE * THREAD_SIZE);

    embedding_backward_kernel<<<num_blocks, THREAD_SIZE>>>(
        grad_output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        grad_weight.data_ptr<float>(),
        seq_length,
        embed_dim,
        num_embeddings,
        total_elements
    );

    return grad_weight;
}