#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <iostream>

// NOTE: Assuming each row of blocks in a grid processes one attention head!

/***************************************************************************
*
* NAPKIN CALCULATIONS
* -------------------
*
* Total shared memory required:
* 
* Query Tile:  32 x 256 x 4  = 32 KB
* Output Tile: 32 x 256 x 4  = 32 KB
*
* Key Tile:    16 x 256 x 4  = 16 KB
* Value Tile:  16 x 256 x 4  = 16 KB
*
* Intermediate Tile: 32 x 16 x 4 = 2 KB
*
* Thus, total shared memoray usage per block = 98 KB :)
* Total threads per block = 1024
*
* Overall, this is quite good utilization per SM memory and thread wise!
*
***************************************************************************/

// Defines the number of rows processed for the query matrix / cols of the key transposed matrix
#define BLOCK_DIM 32
#define QUERY_ROWS 32
#define KV_ROWS 16
#define EMBED_DIM 256

#define MAX_SEQUENCE_LENGTH 128

__global__ void calculate_attention_scores(
    const float* __restrict__ query,  // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* __restrict__ key,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* __restrict__ value,  // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ max_rows,     // Shape: [SEQUENCE LENGTH] 
    float* __restrict__ sum_rows,     // Shape: [SEQUENCE LENGTH]
    float* __restrict__ output        // Shape: [SEQUENCE LENGTH][EMBED DIM]
){
    
    // I dynamically allocate memory since it exceeds 48 KB :)
    extern __shared__ float shared_memory[];

    // Defining shared memory blocks
    float* query_tile_base = shared_memory;
    float* output_tile_base = &shared_memory[QUERY_ROWS * EMBED_DIM];

    float* key_tile_base = &shared_memory[2 * QUERY_ROWS * EMBED_DIM];
    float* value_tile_base = &shared_memory[2 * QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM];

    // Keeps track of the softmax values!
    float* softmax_sum = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM)];
    float* softmax_max = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM) + BLOCK_DIM];

    // Stores the intermediate output
    float* intermediate_out_base = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM)];

    // Defining some helpful macros!
    #define query_tile(row, col) query_tile_base[(row) * EMBED_DIM + (col)]
    #define output_tile(row, col) output_tile_base[(row) * EMBED_DIM + (col)]
    #define key_tile(row, col) key_tile_base[(row) * EMBED_DIM + (col)]
    #define value_tile(row, col) value_tile_base[(row) * EMBED_DIM + (col)]
    #define intermediate_out(row, col) intermediate_out_base[(row) * KV_ROWS + (col)]

    // Defining the scaling factor to reduce softmax distribution sharpness
    const float scale = 1.0f / sqrt(EMBED_DIM);

    // Fetching in the query matrix!
    const int row_offset = blockIdx.x * BLOCK_DIM;
    const int col_offset = threadIdx.x * 4;

    // Keeps track of the prior maximum value
    float prev_max = softmax_max[threadIdx.y];

    // Helper variables for QK^T calculation
    float score = 0.0f;
    int thread_offset = (threadIdx.x % 2) * 128;

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 128){

        const float4* global_query_ptr = reinterpret_cast<const float4*>(&query[(row_offset + threadIdx.y) * EMBED_DIM + col_offset + j]);
        float4* shared_query_ptr = reinterpret_cast<float4*>(&query_tile(threadIdx.y, col_offset + j));

        *shared_query_ptr = *global_query_ptr;
    }

    __syncthreads();

    // Loading in the kv-cache to process the attention matrix row-wise in a tiled fashion!

    #pragma unroll
    for(int i = 0; i < MAX_SEQUENCE_LENGTH; i += KV_ROWS){
        
        /***************************************************************
        *
        * We fetch [16 x 256] block of the key and value matrices each.
        *
        * First 16 rows of the block thus fetch the key matrix.
        * Latter 16 rows of the block then fetch the value matrix!
        *
        ***************************************************************/

        // Fetching in the key matrix
        if(threadIdx.y < 16){

            #pragma unroll
            for(int j = 0; j < EMBED_DIM; j += 128){

                const float4* global_key_ptr = reinterpret_cast<const float4*>(&key[(i + threadIdx.y) * EMBED_DIM + col_offset + j]);
                float4* shared_key_ptr = reinterpret_cast<float4*>(&key_tile(threadIdx.y, col_offset + j));

                *shared_key_ptr = *global_key_ptr;
            }
        }

        // Fetching in the value matrix
        else{
            
            #pragma unroll
            for(int j = 0; j < EMBED_DIM; j += 128){

                const float4* global_value_ptr = reinterpret_cast<const float4*>(&key[(i + threadIdx.y) * EMBED_DIM + col_offset + j]);
                float4* shared_value_ptr = reinterpret_cast<float4*>(&key_tile(threadIdx.y, col_offset + j));

                *shared_value_ptr = *global_value_ptr;
            }
        }

        __syncthreads();

        /******************************************************************************************
        *
        * Now, we calculate the QK^T matrix will keeping track of the sum and the maximum values!
        * Since we have 2x threads than output values to calculate, 2 threads calculate one value
        *
        * Lastly, we utilize efficient warp level functions for calculating the softmax values!
        *
        ******************************************************************************************/

        if(threadIdx.x < 16){

            #pragma unroll
            for(int j = 0; j < EMBED_DIM; j++){
                score += query_tile(threadIdx.y, thread_offset + j) * key_tile(threadIdx.x / 2, thread_offset + j);
            }

            score = score * scale; 

            // Finding the maximum value in this warp
            float warp_max = score;
            
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 8));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 4));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 2));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 1));

            // Only thread 0 needs to write to shared memory
            if(threadIdx.x == 0) {
                softmax_max[threadIdx.y] = max(softmax_max[threadIdx.y], warp_max);
            }

            __syncthreads();

            // Taking away the maximum value for numerical stability
            score = expf(score - softmax_max[threadIdx.y]);

            // Storing the score
            intermediate_out(threadIdx.y, threadIdx.x) = score;

            // Calculating the warp sum
            float warp_sum = score;

            // Reduce within our 16 threads to get the row sum
            warp_sum += __shfl_down_sync(0xffff, warp_sum, 8);
            warp_sum += __shfl_down_sync(0xffff, warp_sum, 4);
            warp_sum += __shfl_down_sync(0xffff, warp_sum, 2);
            warp_sum += __shfl_down_sync(0xffff, warp_sum, 1);

            // Finally, updating the softmax!
            if(threadIdx.x == 0){   
                softmax_sum[threadIdx.y] = expf(prev_max - softmax_max[threadIdx.y]) * softmax_sum[threadIdx.y] + warp_sum;
            }
        }
    }

    __syncthreads();

    float denominator = expf(prev_max - softmax_max[threadIdx.y]);

    // Rescaling the current output matrix

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 32){
        output_tile(threadIdx.y, threadIdx.x + j) /= denominator;
    }

    __syncthreads();

    // Finally, multiplying with the value matrix!

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 32){

        float output_value = 0.0f;

        #pragma unroll
        for(int k = 0; k < KV_ROWS; k++){
            output_value += intermediate_out(threadIdx.y, k) * value_tile(k, j);
        }

        output_tile(threadIdx.y, j) += output_value;
        
        __syncthreads();
    }

    // Lastly, we normalize and write the output tile to the global output matrix
    denominator = softmax_sum[threadIdx.y];

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 32){
        output_tile(threadIdx.y, j) /= denominator;
    }

    __syncthreads();

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 128){
        
        float4* output_ptr = reinterpret_cast<float4*>(&output[(row_offset + threadIdx.y) * QUERY_ROWS + col_offset + j]);
        float4* shared_output_ptr = reinterpret_cast<float4*>(&output_tile(threadIdx.y, col_offset + j));

        *output_ptr = *shared_output_ptr;
    }
}

torch::Tensor calculate_attention_scores_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    
    // Create output tensors
    auto output = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, query.options());

    auto sum_rows = torch::zeros({MAX_SEQUENCE_LENGTH}, query.options());
    auto max_rows = torch::full({MAX_SEQUENCE_LENGTH}, -INFINITY, query.options());   

    // Calculate grid and block dimensions
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / BLOCK_DIM);

    size_t shared_mem_size = 2 * sizeof(float) * (BLOCK_DIM * EMBED_DIM + 16 * EMBED_DIM + BLOCK_DIM) + 
        sizeof(float) * QUERY_ROWS * KV_ROWS;
    
    // Need to increase shared memory size limit for the kernel
    cudaFuncSetAttribute(
        calculate_attention_scores,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    // Launch kernel
    calculate_attention_scores<<<blocks, threads, shared_mem_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        max_rows.data_ptr<float>(),
        sum_rows.data_ptr<float>(),
        output.data_ptr<float>()
    );

    return output;
}