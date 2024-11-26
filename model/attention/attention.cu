#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <iostream>

// Defines the number of rows processed for the query matrix / cols of the key transposed matrix
#define BLOCK_SIZE 32
#define CHUNK_SIZE 32
#define VECTOR_WIDTH 2 
#define EMBED_DIM 256
#define THREAD_WORK 128

// Note to self and possible viwers: max_rows keeps track of maximum per row as per the flash attention paper :)
// Same goes for the sum_rows which saves the sum of exp(Scoreij) per row for the softmax calculation!

__global__ void calculate_attention_scores(
    const float* __restrict__ query,  // Shape: [SEQUENCE LENGTH][EMBEDDING DIM]
    const float* __restrict__ key,    // Shape: [SEQUENCE LENGTH][EMBEDDING DIM]  
    float* __restrict__ max_rows,     // Shape: [SEQUENCE LENGTH] 
    float* __restrict__ sum_rows,     // Shape: [SEQUENCE LENGTH]
    float* __restrict__ output,       // Shape: [SEQUENCE LENGTH][SEQUENCE LENGTH]
    const int sequence_length 
){

    // Defining spaces in shared memory to capture partial QK^T matrix :)
    __shared__ float query_tile[CHUNK_SIZE][EMBED_DIM]; // Takes 32 x 256 x (32 / 8) = 32 KB
    __shared__ float key_tile[CHUNK_SIZE][EMBED_DIM];   // Takes 32 x 256 x (32 / 8) = 32 KB 
    __shared__ float scores[CHUNK_SIZE][CHUNK_SIZE];    // Takes 32 x 32 x (32 / 8)  = 4  KB

    const float scale = 1.0f / sqrt(EMBED_DIM);

    // For each of the Q, and K matrices, each thread loads in 8 elements
    const int num_blocks = sequence_length / CHUNK_SIZE;

    // Add debug prints at key points
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Kernel started with sequence_length: %d\n", sequence_length);
    }

    for(int query_block = 0; query_block < num_blocks; query_block++){

        const int global_row_query = (query_block * CHUNK_SIZE) + threadIdx.y;
        const int shared_row_query = threadIdx.y;

        const int column_query = threadIdx.x * 8;

        #pragma unroll
        for(int mem_fetch = 0; mem_fetch < VECTOR_WIDTH; mem_fetch++){

            const float4* query_ptr = reinterpret_cast<const float4*>(&query[(global_row_query * EMBED_DIM) + column_query + (mem_fetch * 4)]);
            float4* shared_query_ptr = reinterpret_cast<float4*>(&query_tile[shared_row_query][column_query + (mem_fetch * 4)]);

            *shared_query_ptr = *query_ptr;
        }

        // Ensuring that the query matrix is pulled in to SMEM!
        __syncthreads();
            
        for(int key_block = 0; key_block < 1; key_block++){

            if (threadIdx.y == 0 && threadIdx.x == 0){

                printf("Inside!");
                // We move down global memory by chunk size x num. of iterations + this thread's appropriate row :)
                const int global_row = (key_block * CHUNK_SIZE) + threadIdx.y;
                const int shared_row = threadIdx.y;

                // The column index will be the same for the shared and global memories
                const int column = threadIdx.x * 8;

                // Each warp fetches one row of size embedding dimension for the keys
                #pragma unroll
                for(int mem_fetch = 0; mem_fetch < VECTOR_WIDTH; mem_fetch++){
                        
                    // Fetching the key matrix chunkc
                    const float4* key_ptr = reinterpret_cast<const float4*>(&key[(global_row * EMBED_DIM) + column + (mem_fetch * 4)]);
                    float4* shared_key_ptr = reinterpret_cast<float4*>(&key_tile[shared_row][column + (mem_fetch * 4)]);

                    *shared_key_ptr = *key_ptr;
                }
            }
            // Ensuring the entire matrix has been pulled in
            __syncthreads();
            
            // float score = 0.0f;

            // #pragma unroll
            // for(int i = 0; i < EMBED_DIM; i++){
            //     score += query_tile[threadIdx.y][i] * key_tile[threadIdx.x][i];
            // }
        
            // // Calculating scaled attention score
            // score *= scale;

            // // Storing the score into the shared memory
            // scores[threadIdx.y][threadIdx.x] = score;

            // __syncthreads();

            // // Updating the running sum for softmax denominator
            // if (threadIdx.x == 0) {

            //     float prev_max = max_rows[global_row_query];
            //     float local_max = -INFINITY;

            //     for (int i = 0; i < CHUNK_SIZE; i++) {
            //         local_max = max(local_max, scores[threadIdx.y][i]);
            //     }

            //     max_rows[global_row_query] = max(prev_max, local_max);

            //     // Adjust sum_rows if max_rows changed
            //     float sum_val = sum_rows[global_row_query];

            //     if (max_rows[global_row_query] > prev_max) {
            //         sum_val = sum_val * expf(prev_max - max_rows[global_row_query]);
            //     }

            //     // Recompute exponentials with updated max_rows and update scores
            //     float exp_scores_sum = 0.0f;

            //     for (int i = 0; i < CHUNK_SIZE; i++) {
            //         float exp_score = expf(scores[threadIdx.y][i] - max_rows[global_row_query]);
            //         scores[threadIdx.y][i] = exp_score;
            //         exp_scores_sum += exp_score;
            //     }

            //     sum_rows[global_row_query] = sum_val + exp_scores_sum;
            // }

            // __syncthreads();

            // // Normalize scores
            // scores[threadIdx.y][threadIdx.x] = scores[threadIdx.y][threadIdx.x] / sum_rows[global_row_query];

            // // **Write the normalized scores to global output array**
            // // Calculate global indices for query and key positions
            // const int global_query_idx = (query_block * CHUNK_SIZE) + threadIdx.y;
            // const int global_key_idx = (key_block * CHUNK_SIZE) + threadIdx.x;

            // // Write to the output array
            // output[global_query_idx * sequence_length + global_key_idx] = scores[threadIdx.y][threadIdx.x];

            // __syncthreads();
        }
    }
}

torch::Tensor calculate_attention_scores_cuda(torch::Tensor query,torch::Tensor key) {
    
    // Get the dimensions
    const int sequence_length = query.size(0);

    std::cout << query.sizes() << std::endl;

    // Create output tensors
    auto output = torch::zeros({sequence_length, sequence_length}, query.options());
    auto max_rows = torch::full({sequence_length}, -INFINITY, query.options());
    auto sum_rows = torch::zeros({sequence_length}, query.options());
    
    // Calculate grid and block dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(1);
    
    // Launch kernel
    calculate_attention_scores<<<blocks, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        max_rows.data_ptr<float>(),
        sum_rows.data_ptr<float>(),
        output.data_ptr<float>(),
        sequence_length
    );
    
    // Add these lines to see printf output
    cudaDeviceSynchronize();  // Wait for kernel to finish
    fflush(stdout);          // Flush stdout buffer

    return output;
}