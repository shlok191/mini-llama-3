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
#define MAX_SEQUENCE_LENGTH 256

__global__ void calculate_attention_scores(
    const float* query,  // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* key,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* value,  // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* max_rows,     // Shape: [SEQUENCE LENGTH] 
    float* sum_rows,     // Shape: [SEQUENCE LENGTH]
    float* output,       // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* logsumexp     // Shape: [SEQUENCE LENGTH]
){
    
    /****************************************************************
    *
    * I utlize the first section to only define helper variables :)
    * 
    ****************************************************************/

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
    float* softmax_logexp = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM)];

    // Stores the intermediate output
    float* intermediate_out_base = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM) + BLOCK_DIM];

    // Defining some helpful macros
    #define query_tile(row, col) query_tile_base[(row) * EMBED_DIM + (col)]
    #define output_tile(row, col) output_tile_base[(row) * EMBED_DIM + (col)]
    #define key_tile(row, col) key_tile_base[(row) * EMBED_DIM + (col)]
    #define value_tile(row, col) value_tile_base[(row) * EMBED_DIM + (col)]
    #define intermediate_out(row, col) intermediate_out_base[(row) * KV_ROWS + (col)]

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < EMBED_DIM; i += 32){
        output_tile(threadIdx.y, threadIdx.x + i) = 0.0f;
    }

    // Defining the scaling factor to reduce softmax distribution sharpness
    const float scale = 1.0f / sqrt(EMBED_DIM);

    // Fetching in the query matrix!
    const int row_offset = blockIdx.x * BLOCK_DIM;
    const int col_offset = threadIdx.x * 4;

    // Keeps track of the prior maximum value
    float prev_max = -INFINITY;
    float score = 0.0f;

    /***********************************
    *
    * Beginning the actual processing!
    * 
    ***********************************/


    // Fetching the query matrix!

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

                const float4* global_value_ptr = reinterpret_cast<const float4*>(&value[(i + threadIdx.y - 16) * EMBED_DIM + col_offset + j]);
                float4* shared_value_ptr = reinterpret_cast<float4*>(&value_tile(threadIdx.y - 16, col_offset + j));

                *shared_value_ptr = *global_value_ptr;
            }
        }

        __syncthreads();

        /******************************************************************************************
        *
        * Now, we calculate the QK^T matrix will keeping track of the sum and the maximum values!
        * Since we have 2x threads than output values to calculate, one thread calculates one value
        *
        * Lastly, we utilize efficient warp level functions for calculating the softmax values!
        *
        ******************************************************************************************/
       
        if(threadIdx.x < 16){

            #pragma unroll
            for(int j = 0; j < EMBED_DIM; j++){
                score += query_tile(threadIdx.y, j) * key_tile(threadIdx.x, j);
            }

            score = score * scale; 

            float warp_max = score;

            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 8));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 4));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 2));
            warp_max = max(warp_max, __shfl_down_sync(0xffff, warp_max, 1));

            // Only thread 0 needs to write to shared memory
            if(threadIdx.x == 0) {
                softmax_max[threadIdx.y] = max(softmax_max[threadIdx.y], warp_max);
            }
        }   

        __syncthreads();
        
        if(threadIdx.x < 16){
            
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

        __syncthreads();

        float denominator = expf(prev_max - softmax_max[threadIdx.y]);

        // Rescaling the current output matrix

        #pragma unroll
        for(int j = 0; j < EMBED_DIM; j += 32){
            output_tile(threadIdx.y, threadIdx.x + j) *= denominator;
        }

        __syncthreads();
    
        // Finally, multiplying with the value matrix!
        #pragma unroll
        for(int j = 0; j < EMBED_DIM; j += 32){

            float output_value = 0.0f;

            #pragma unroll
            for(int k = 0; k < KV_ROWS; k++){
                output_value += intermediate_out(threadIdx.y, k) * value_tile(k, j + threadIdx.x);
            }

            output_tile(threadIdx.y, threadIdx.x + j) += output_value;
        }

        // Updating our previous maximum value
        prev_max = softmax_max[threadIdx.y];
        score = 0.0f;

        if(threadIdx.x == 0){
            softmax_logexp[threadIdx.y] = prev_max + logf(softmax_sum[threadIdx.y]);
        }

        __syncthreads();
    }

    // Storing the logsum exponential value
    if(threadIdx.x == 0){
        logsumexp[row_offset + threadIdx.y] = softmax_logexp[threadIdx.y];
    }

    // Lastly, we normalize and write the output tile to the global output matrix!
    float denominator = softmax_sum[threadIdx.y];

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 32){
        output_tile(threadIdx.y, threadIdx.x + j) /= denominator;
    }

    #pragma unroll
    for(int j = 0; j < EMBED_DIM; j += 128){
        
        float4* output_ptr = reinterpret_cast<float4*>(&output[(row_offset + threadIdx.y) * EMBED_DIM + col_offset + j]);
        float4* shared_output_ptr = reinterpret_cast<float4*>(&output_tile(threadIdx.y, col_offset + j));

        *output_ptr = *shared_output_ptr;
    }
}

/***************************************************************************
*
* NAPKIN CALCULATIONS
* -------------------
*
* Total shared memory required:
* 
* Query Tile:   16 x 256 x 4   = 16 KB
* d_Query Tile:  16 x 256 x 4  = 16 KB
*
* Output Tile:  16 x 256 x 4   = 16 KB
* d_Output Tile: 16 x 256 x 4  = 16 KB
*
* Key Tile:    8 x 256 x 4  = 8 KB
* d_Key Tile:  8 x 256 x 4  = 8 KB
*
* Value Tile:   8 x 256 x 4  = 8 KB
* d_Value Tile: 8 x 256 x 4  = 8 KB
* 
* Intermediate Tile: 32 x 16 x 4 = 2 KB
*
* Thus, total shared memoray usage per block = 98 KB :)
* Total threads per block = 1024
*
* Overall, this is quite good utilization per SM memory and thread wise!
*
***************************************************************************/

#define KV_ROWS_BACK 8
#define QUERY_ROWS_BACK 16

__global__ void calculate_attention_scores_backwards(
    const float* query,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* key,      // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* value,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* output,   // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* d_query,        // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* d_key,          // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* d_value,        // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* d_output,       // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* logexp,         // Shape: [SEQUENCE LENGTH] 
    float* D               // Shape: [SEQUENCE LENGTH]
){

    // First, we calculate the D matrix
    // I dynamically allocate memory since it exceeds 48 KB :)
    extern __shared__ float shared_memory[];

    // Defining shared memory blocks for the weights
    float* query_tile_base    = shared_memory;
    float* key_tile_base      = &shared_memory[QUERY_ROWS_BACK * EMBED_DIM];
    float* value_tile_base    = &shared_memory[(QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* output_tile_base   = &shared_memory[(QUERY_ROWS_BACK + 2 * KV_ROWS_BACK) * EMBED_DIM];

    // Defining shared memory blocks for the derivatives
    float* d_query_tile_base  = &shared_memory[2 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* d_key_tile_base    = &shared_memory[(3 * QUERY_ROWS_BACK + 2 * KV_ROWS_BACK) * EMBED_DIM];
    float* d_value_tile_base  = &shared_memory[3 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* d_output_tile_base = &shared_memory[(3 * QUERY_ROWS_BACK + 4 * KV_ROWS_BACK) * EMBED_DIM];
    
    // Storing normalization and D values    
    float* scores_tile_base   = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* d_P_tile_base      = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + KV_ROWS_BACK * QUERY_ROWS_BACK];
    float* d_S_tile_base      = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + 2 * KV_ROWS_BACK * QUERY_ROWS_BACK]; 

    // Defining some helper macros :)
    #define query_tile(row, col)    query_tile_base[(row) * EMBED_DIM + (col)]
    #define key_tile(row, col)      key_tile_base[(row) * EMBED_DIM + (col)]
    #define value_tile(row, col)    value_tile_base[(row) * EMBED_DIM + (col)]
    #define output_tile(row, col)   output_tile_base[(row) * EMBED_DIM + (col)]

    // Tile access macros for gradients
    #define d_query_tile(row, col)  d_query_tile_base[(row) * EMBED_DIM + (col)]
    #define d_key_tile(row, col)    d_key_tile_base[(row) * EMBED_DIM + (col)]
    #define d_value_tile(row, col)  d_value_tile_base[(row) * EMBED_DIM + (col)]
    #define d_output_tile(row, col) d_output_tile_base[(row) * EMBED_DIM + (col)]

    // Intermediate calculation tiles
    #define scores_tile(row, col)   scores_tile_base[(row) * KV_ROWS_BACK + (col)]
    #define d_P_tile(row, col)      d_P_tile_base[(row) * KV_ROWS_BACK + (col)]
    #define d_S_tile(row, col)      d_S_tile_base[(row) * KV_ROWS_BACK + (col)]
    
    // Fetching the logexp value

    #pragma unroll
    for(int i = 0; i < MAX_SEQUENCE_LENGTH; i += KV_ROWS_BACK){
        
        // Fetching in the query, key and the derivative blocks
        if(threadIdx.y < KV_ROWS_BACK){

            #pragma unroll
            for(int j = threadIdx.x * 4; j < EMBED_DIM; j += 128){
                
                const float4* key_ptr = reinterpret_cast<const float4*>(&key[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_key_ptr = reinterpret_cast<float4*>(&key_tile(threadIdx.y, j));

                const float4* value_ptr = reinterpret_cast<const float4*>(&value[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_value_ptr = reinterpret_cast<float4*>(&value_tile(threadIdx.y, j));
                
                float4* d_key_ptr = reinterpret_cast<float4*>(&d_key[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_d_key_ptr = reinterpret_cast<float4*>(&d_key_tile(threadIdx.y, j));
                
                float4* d_value_ptr = reinterpret_cast<float4*>(&d_value[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_d_value_ptr = reinterpret_cast<float4*>(&d_value_tile(threadIdx.y, j));

                *shared_key_ptr = *key_ptr;
                *shared_value_ptr = *value_ptr;
                *shared_d_key_ptr = *d_key_ptr;
                *shared_d_value_ptr = *d_value_ptr;
                
            }
        }

        for(int j = 0; j < MAX_SEQUENCE_LENGTH; j += QUERY_ROWS_BACK){

            // Loading in the query, d_query, output and d_output tiles
            if(threadIdx.y < QUERY_ROWS_BACK){

                #pragma unroll
                for(int k = threadIdx.x * 4; k < EMBED_DIM; k += 128) {
                    
                    const float4* query_ptr = reinterpret_cast<const float4*>(&query[(j + threadIdx.y) * EMBED_DIM + k]);
                    float4* shared_query_ptr = reinterpret_cast<float4*>(&query_tile(threadIdx.y, k));
                    
                    const float4* output_ptr = reinterpret_cast<const float4*>(&output[(j + threadIdx.y) * EMBED_DIM + k]);
                    float4* shared_output_ptr = reinterpret_cast<float4*>(&output_tile(threadIdx.y, k));
                    
                    float4* d_output_ptr = reinterpret_cast<float4*>(&d_output[(j + threadIdx.y) * EMBED_DIM + k]);
                    float4* shared_d_output_ptr = reinterpret_cast<float4*>(&d_output_tile(threadIdx.y, k));
                    
                    float4* d_query_ptr = reinterpret_cast<float4*>(&d_query[(j + threadIdx.y) * EMBED_DIM + k]);
                    float4* shared_d_query_ptr = reinterpret_cast<float4*>(&d_query_tile(threadIdx.y, k));
                    
                    *shared_query_ptr = *query_ptr;
                    *shared_output_ptr = *output_ptr;
                    *shared_d_output_ptr = *d_output_ptr;
                    *shared_d_query_ptr = *d_query_ptr;
                }
            }

            float sum = 0.0f;
            
            int key_offset = (threadIdx.y % 2) * 4 + threadIdx.x / 8;
            int row_offset = (threadIdx.x % 8) * 32;

            __syncthreads();

            // Calculating the score of dimension [16 x 8] and 2 warps calculate one row of values
            #pragma unroll
            for(int k = 0; k < 32; k++){
                sum += query_tile(threadIdx.y / 2, row_offset + k) * key_tile(key_offset, row_offset + k);
            }

            __syncthreads();
            
            sum += __shfl_down_sync(0xff, sum, 4);
            sum += __shfl_down_sync(0xff, sum, 2);
            sum += __shfl_down_sync(0xff, sum, 1);

            __syncthreads();

            float logexpval = logexp[j + threadIdx.y];

            // Storing the values and fetching the log exp value
            if(threadIdx.x % 8 == 0){
                scores_tile(threadIdx.y / 2, key_offset) = expf(sum - logexpval);
            }

            __syncthreads();

            // Defining the row and column offsets for each thread
            row_offset = threadIdx.y / 4;
            int col_offset = (threadIdx.y % 4) * 64 + threadIdx.x * 2;

            sum = 0.0f;

            #pragma unroll
            for(int k = col_offset; k < col_offset + 2; k++){
                
                sum = 0.0f;

                #pragma unroll
                for(int l = 0; l < 16; l++){
                    sum += scores_tile(l, row_offset) * d_output_tile(l, k);
                }

                d_value_tile(row_offset, k) += sum;
            }

            sum = 0;

            __syncthreads();

            #pragma unroll
            for(int k = 0; k < 32; k++){
                sum += d_output_tile(threadIdx.y / 2, row_offset + k) * value_tile(key_offset, row_offset + k);
            }

            __syncthreads();

            sum += __shfl_down_sync(0xff, sum, 4);
            sum += __shfl_down_sync(0xff, sum, 2);
            sum += __shfl_down_sync(0xff, sum, 1);

            __syncthreads();

            if(threadIdx.x % 8 == 0){
                d_P_tile(threadIdx.y / 2, (threadIdx.y % 2) * 4 + threadIdx.x / 8) = sum - D[row_offset];;
            }

            __syncthreads();

            sum = 0.0f;

            // Now we calculate the dS value
            if(threadIdx.y < 16){

                if(threadIdx.x < 8){
                    d_S_tile(threadIdx.y, threadIdx.x) = scores_tile(threadIdx.y, threadIdx.x) * d_P_tile(threadIdx.y, threadIdx.x);
                }
            }

            __syncthreads();

            col_offset = (threadIdx.y % 2) * 128 + threadIdx.x * 4;

            // Now we update the dQ matrix!
            #pragma unroll
            for(int k = 0; k < 4; k++){
                
                sum = 0.0f;

                #pragma unroll
                for(int l = 0; l < KV_ROWS_BACK; l++){
                    sum += d_S_tile(threadIdx.y / 2, l) * key_tile(l, col_offset + k);
                }

                d_query_tile(threadIdx.y / 2, col_offset + k) += sum;
            }

            __syncthreads();

            #pragma unroll
            for(int k = threadIdx.x * 4; k < EMBED_DIM; k += 128){

                float4* d_query_ptr = reinterpret_cast<float4*>(&d_query[(j + threadIdx.y) * EMBED_DIM + k]);
                float4* shared_d_query_ptr = reinterpret_cast<float4*>(&d_query_tile(threadIdx.y, k));

                *d_query_ptr = *shared_d_query_ptr;
            
            }

            // Calculating dK
            #pragma unroll
            for(int k = 0; k < 2; k++){
                
                float sum = 0.0f;

                #pragma unroll
                for(int l = 0; l < KV_ROWS_BACK; l++){
                    sum += d_S_tile(threadIdx.y / 2, l) * query_tile(l, col_offset + k);
                }

                d_key_tile(threadIdx.y / 2, col_offset + k) += sum;
            }

            __syncthreads();
        }

        if(threadIdx.y < KV_ROWS_BACK){

            #pragma unroll
            for(int j = threadIdx.x * 4; j < EMBED_DIM; j += 128){
                
                float4* d_key_ptr = reinterpret_cast<float4*>(&d_key[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_d_key_ptr = reinterpret_cast<float4*>(&d_key_tile(threadIdx.y, j));

                *d_key_ptr = *shared_d_key_ptr;
            }

            #pragma unroll
            for(int j = threadIdx.x * 4; j < EMBED_DIM; j += 128){

                float4* d_value_ptr = reinterpret_cast<float4*>(&d_value[(i + threadIdx.y) * EMBED_DIM + j]);
                float4* shared_d_value_ptr = reinterpret_cast<float4*>(&d_value_tile(threadIdx.y, j));

                *d_value_ptr = *shared_d_value_ptr;   
            }
        }
    }
}

std::vector<torch::Tensor> calculate_attention_scores_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    
    // Create output tensors
    auto output = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, query.options());

    auto sum_rows = torch::zeros({MAX_SEQUENCE_LENGTH}, query.options());
    auto max_rows = torch::full({MAX_SEQUENCE_LENGTH}, -INFINITY, query.options());   
    auto logsumexp = torch::zeros({MAX_SEQUENCE_LENGTH}, query.options());

    // Calculate grid and block dimensions
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(1);

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
        output.data_ptr<float>(),
        logsumexp.data_ptr<float>()
    );

    return std::vector<torch::Tensor>{output, logsumexp};
}

std::vector<torch::Tensor> calculate_attention_scores_backward_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output, torch::Tensor d_output, torch::Tensor logexp){
        
    // Create output tensors
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_value = torch::zeros_like(value);

    // Create D tensor
    torch::Tensor D = (d_output.cuda() * output).sum(1);

    // Calculate grid and block dimensions
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / BLOCK_DIM);

    // Defining the shared memory requirements
    size_t shared_mem_size = sizeof(float) * (4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + 3 * KV_ROWS_BACK * QUERY_ROWS_BACK);

    // Need to increase shared memory size limit for the kernel
    cudaFuncSetAttribute(
        calculate_attention_scores_backwards,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    // Launch kernel
    calculate_attention_scores_backwards<<<blocks, threads, shared_mem_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        d_query.data_ptr<float>(),
        d_key.data_ptr<float>(),
        d_value.data_ptr<float>(),
        d_output.data_ptr<float>(),
        logexp.data_ptr<float>(),
        D.data_ptr<float>()
    );

    return std::vector<torch::Tensor>{d_query, d_key, d_value};
}