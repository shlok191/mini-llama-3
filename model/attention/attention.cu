#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

// Defines the number of rows processed for the query matrix / cols of the key transposed matrix
#define BLOCK_SIZE 32
#define CHUNK_SIZE 16
#define EMBED_DIM 512 

__global__ void flash_attn_score_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ output,
    const int sequence_length 
){

    // Defining spaces in shared memory to capture partial QK^T matrix :)
    __shared__ float query_tile[CHUNK_SIZE][EMBED_DIM]; // Takes 16 x 512 x 32 / 8 x 1024 = 32 KB
    __shared__ float key_tile[CHUNK_SIZE][EMBED_DIM];   // Takes 16 x 512 x 32 / 8 x 1024 = 32 KB 
    __shared__ float value_tile[CHUNK_SIZE][EMBED_DIM]; // Takes 16 x 512 x 32 / 8 x 1024 = 32 KB
    __shared__ float finished[CHUNK_SIZE][CHUNK_SIZE];  // Takes 16 x 8 x 32 / 8 x 1024   = 0.5 KB memory 

    // For each of the Q, K, and V matrices, each thread loads in 8 elements
    // Each warp in turn, loads in 256 elements per row! 1 row = 2 warps hence :)

    const int NUM_TILES = sequence_length / CHUNK_SIZE;

    for(int i = 0; i < NUM_TILES; i++){

        // First, we populate our shared memory
        const int thread_row = (i * CHUNK_SIZE) + threadIdx.y / 2;
        const int thread_column = (threadIdx.y % 2) * (EMBED_DIM / 2) + threadIdx.x;

        // Each thread pulls in 4 floats at once, twice
        for(int j = 0; j < 2; j++){

            float4* query_tile_ptr = reinterpret_cast<float4*>(&query_tile[thread_row * EMBED_DIM + thread_column + (j * 4)]);
            
        }
    }
}


// Output projection kernel
__global__ void output_projection_kernel(
    const float* __restrict__ attention_out,  // [num_heads][seq_len][head_dim]
    const float* __restrict__ Wo,             // [model_dim][num_heads * head_dim]
    float* __restrict__ output,               // [seq_len][model_dim]
    const int seq_length,
    const int num_heads,
    const int model_dim
) {
    __shared__ float attn_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float wo_tile[HEAD_DIM][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    
    // Iterate over heads and tiles
    for (int h = 0; h < num_heads; h++) {
        const float* head_attn = attention_out + h * seq_length * HEAD_DIM;
        const float* head_wo = Wo + col * (num_heads * HEAD_DIM) + h * HEAD_DIM;
        
        // Load attention output tile
        if (row < seq_length && threadIdx.x < HEAD_DIM) {
            attn_tile[threadIdx.y][threadIdx.x] = 
                head_attn[row * HEAD_DIM + threadIdx.x];
        }
        
        // Load Wo tile
        if (threadIdx.y < HEAD_DIM && col < model_dim) {
            wo_tile[threadIdx.y][threadIdx.x] = head_wo[threadIdx.y];
        }
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < HEAD_DIM; k++) {
            acc += attn_tile[threadIdx.y][k] * wo_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    // Store result
    if (row < seq_length && col < model_dim) {
        output[row * model_dim + col] = acc;
    }
}

// Main function to run multi-head attention
void multi_head_attention(
    const torch::Tensor& queries,     
    const torch::Tensor& keys,         
    const torch::Tensor& values,       
    const torch::Tensor& W_output,     
    torch::Tensor& output,
    const int seq_length, 
    const int num_heads,
    const int model_dim
) {
    // Allocate temporary memory
    float *scores, *attention_out;
    cudaMalloc(&scores, num_heads * seq_length * seq_length * sizeof(float));
    cudaMalloc(&attention_out, num_heads * seq_length * HEAD_DIM * sizeof(float));
    
    // 1. Compute QK^T
    dim3 qk_grid(
        (seq_length + TILE_SIZE - 1) / TILE_SIZE,
        (seq_length + TILE_SIZE - 1) / TILE_SIZE,
        num_heads
    );
    
    dim3 qk_block(TILE_SIZE, TILE_SIZE);
    
    qk_matmul_kernel<<<qk_grid, qk_block>>>(
        queries, keys, scores,
        seq_length, num_heads
    );
    
    // 2. Apply softmax
    dim3 softmax_grid(
        (seq_length + TILE_SIZE - 1) / TILE_SIZE,
        num_heads
    );
    dim3 softmax_block(TILE_SIZE);
    softmax_kernel<<<softmax_grid, softmax_block>>>(
        scores, seq_length, num_heads
    );
    
    // 3. Multiply with values
    dim3 av_grid(
        1,
        (seq_length + TILE_SIZE - 1) / TILE_SIZE,
        num_heads
    );
    dim3 av_block(TILE_SIZE, TILE_SIZE);
    attention_values_kernel<<<av_grid, av_block>>>(
        scores, values, attention_out,
        seq_length, num_heads
    );
    
    // 4. Project output
    dim3 proj_grid(
        (model_dim + TILE_SIZE - 1) / TILE_SIZE,
        (seq_length + TILE_SIZE - 1) / TILE_SIZE
    );
    dim3 proj_block(TILE_SIZE, TILE_SIZE);
    output_projection_kernel<<<proj_grid, proj_block>>>(
        attention_out, Wo, output,
        seq_length, num_heads, model_dim
    );
    
    // Clean up
    cudaFree(scores);
    cudaFree(attention_out);
}