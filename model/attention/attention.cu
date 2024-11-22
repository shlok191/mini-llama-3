#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

// Defines the number of rows processed for the query matrix / cols of the key transposed matrix
#define BLOCK_SIZE 32
#define CHUNK_SIZE 32
#define VECTOR_WIDTH 2 
#define EMBED_DIM 256
#define THREAD_WORK 128

__global__ void flash_attn_score_kernel(
    const float* __restrict__ query,  // Shape: [SEQUENCE LENGTH][EMBEDDING DIM]
    const float* __restrict__ key,    // Shape: [SEQUENCE LENGTH][EMBEDDING DIM]  
    const float* __restrict__ value,  // Shape: [SEQUENCE LENGTH][EMBEDDING DIM]
    float* __restrict__ attn,         // Shape: [SEQUENCE LENGTH][SEQUENCE LENGTH]
    const int sequence_length 
){

    // Defining spaces in shared memory to capture partial QK^T matrix :)
    __shared__ float query_tile[CHUNK_SIZE][EMBED_DIM]; // Takes 32 x 256 x (32 / 8) = 32 KB
    __shared__ float key_tile[CHUNK_SIZE][EMBED_DIM];   // Takes 32 x 256 x (32 / 8) = 32 KB 

    __shared__ float max_tile[CHUNK_SIZE];  // Max for each row
    __shared__ float sum_tile[CHUNK_SIZE];  // Softmax denominator for each row

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        max_tile[i] = -INFINITY;
        sum_tile[i] = 0.0f;
    }

    const float scale = 1.0f / sqrt(EMBED_DIM);

    // For each of the Q, and K matrices, each thread loads in 8 elements
    const int NUM_TILES = sequence_length / CHUNK_SIZE;

    for(int row_offset = 0; row_offset < NUM_TILES; row_offset++){

        // First, we populate our shared memory with query

        // Keeping this in the outer loop since every column of the key matrix will be multiplied with
        // the query matrix row, and keeping it in memory longer is more efficient!
        const int global_row_Q = (row_offset * CHUNK_SIZE) + threadIdx.y;
        int column_Q = threadIdx.x * 4;
        
        #pragma unroll
        for(int j = 0; j < VECTOR_WIDTH; j++){
                
            float4* query_ptr = reinterpret_cast<float4*>(&query[global_row_Q * EMBED_DIM + column_Q]);
            float4* shared_ptr = reinterpret_cast<float4*>(&query_tile[threadIdx.y][column_Q]);

            *shared_ptr = *query_ptr;
            
            // Strided fetches :)
            column_Q += BLOCK_SIZE * 4;
        }
        
        __syncthreads();

        for(int col_offset = 0; col_offset < NUM_TILES; col_offset++){
            
            // Now, we pull in the value matrix!
            const int global_row_K = (col_offset * CHUNK_SIZE) + threadIdx.y;
            int column_K = threadIdx.x * 4;
            
            #pragma unroll
            for(int j = 0; j < VECTOR_WIDTH; j++){
                    
                float4* key_ptr = reinterpret_cast<float4*>(&key[global_row_K * EMBED_DIM + column_K]);
                float4* shared_ptr = reinterpret_cast<float4*>(&key_tile[threadIdx.y][column_K]);

                *shared_ptr = *key_ptr;
                
                // Strided fetches :)
                column_K += BLOCK_SIZE * 4; 
            }
            
            // Ensuring the entire key matrix is loaded into shared memory before we proceed!
            __syncthreads();
            
            float score = 0.0f;

            #pragma unroll
            for(int i = 0; i < EMBED_DIM; i++){
                score += query_tile[threadIdx.y][i] * key_tile[threadIdx.x][i];
            }


            // Calculating scaled attention score
            score *= scale;

            // Using atomicMax to update the maximum score for this row
            atomicMax(&max_tile[threadIdx.y], score);
            __syncthreads();

            // Calculating the exp(score - max) for numerical stability
            float exp_score = expf(score - max_tile[threadIdx.y]);

            // Updating the running sum for softmax denominator
            atomicAdd(&sum_tile[threadIdx.y], exp_score);
            __syncthreads();

            // Write the final normalized attention score
            int row_idx = row_offset * CHUNK_SIZE + threadIdx.y;
            int col_idx = col_offset * CHUNK_SIZE + threadIdx.x;

            attn[row_idx * sequence_length + col_idx] = exp_score / sum_tile[threadIdx.y];

            __syncthreads();
        }
    }
}

// Output projection kernel
__global__ void value_output_projection_kernel(
    const float* __restrict__ value,    // [seq_len][head_dim]
    const float* __restrict__ output,   // [model_dim][num_heads * head_dim]
    float* __restrict__ attn,          //  [seq_len][model_dim]
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
torch::Tensor multi_head_attention(
    const torch::Tensor& query,     
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