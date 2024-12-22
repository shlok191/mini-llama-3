#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

// Each CUDA stream handles one attention head!
#define MAX_SEQUENCE_LENGTH 256
#define EMBED_DIM 256
#define NUM_HEADS 4

std::vector<torch::Tensor> split_matrices(torch::Tensor input, const int batch_size){
    
    // This stores each 256 embedding dimension tensor
    std::vector<torch::Tensor> split_matrices;
    
    for(int b = 0; b < batch_size; b++){

        // Fetching the batch tensor
        auto batch_tensor = input.select(0, b);

        for(int i = 0; i < NUM_HEADS; i++){

            auto split = batch_tensor.slice(1, i * EMBED_DIM, (i + 1) * EMBED_DIM).clone();
            split_matrices.push_back(split);
        }
    }

    return split_matrices;
}

std::vector<torch::Tensor> split_vectors(torch::Tensor input) {

    std::vector<torch::Tensor> split_vectors;
    
    auto batch_size = input.size(0);

    for(int b = 0; b < batch_size; b++) {

        // Fetching the batch tensor
        auto batch_tensor = input.select(0, b);

        for(int i = 0; i < NUM_HEADS; i++) {

            auto split = batch_tensor.select(0, i).clone();
            split_vectors.push_back(split);
        }
    }

    return split_vectors;
}

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

__global__ void calculate_attention_scores(
    const float* query,  // Shape: [BATCH SIZE][SEQUENCE LENGTH][EMBED DIM]
    const float* key,    // Shape: [BATCH SIZE][SEQUENCE LENGTH][EMBED DIM]
    const float* value,  // Shape: [BATCH SIZE][SEQUENCE LENGTH][EMBED DIM]
    float* max_rows,     // Shape: [BATCH SIZE][SEQUENCE LENGTH] 
    float* sum_rows,     // Shape: [BATCH SIZE][SEQUENCE LENGTH]
    float* output        // Shape: [BATCH SIZE][SEQUENCE LENGTH][EMBED DIM]
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
    
    if (threadIdx.x == 0) {
        softmax_max[threadIdx.y] = -INFINITY;
        softmax_sum[threadIdx.y] = 0.0f;
    }

    __syncthreads();

    // Stores the intermediate output
    float* intermediate_out_base = &shared_memory[2 * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM)];

    // Defining some helpful macros
    #define query_tile(row, col) query_tile_base[(row) * EMBED_DIM + (col)]
    #define output_tile(row, col) output_tile_base[(row) * EMBED_DIM + (col)]
    #define key_tile(row, col) key_tile_base[(row) * EMBED_DIM + (col)]
    #define value_tile(row, col) value_tile_base[(row) * EMBED_DIM + (col)]
    #define intermediate_out(row, col) intermediate_out_base[(row) * KV_ROWS + (col)]
    
    // Defining the scaling factor to reduce softmax distribution sharpness
    const float scale = 1.0f / sqrt(EMBED_DIM);
    
    #pragma unroll
    for(int i = 0; i < EMBED_DIM; i += 32){
        output_tile(threadIdx.y, threadIdx.x + i) = 0.0f;
    }

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

        __syncwarp();
        
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

        float denominator = expf(prev_max - softmax_max[threadIdx.y]) + 1e-6;

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

        __syncthreads();
    }

    // Storing the logsum exponential value
    if(threadIdx.x == 0){
        max_rows[row_offset + threadIdx.y] = softmax_max[threadIdx.y];
        sum_rows[row_offset + threadIdx.y] = softmax_sum[threadIdx.y];
    }

    // Lastly, we normalize and write the output tile to the global output matrix!
    float denominator = softmax_sum[threadIdx.y] + 1e-6;

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
    const float* __restrict__ query,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* __restrict__ key,      // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* __restrict__ value,    // Shape: [SEQUENCE LENGTH][EMBED DIM]
    const float* __restrict__ output,   // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ d_query,        // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ d_key,          // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ d_value,        // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ d_output,       // Shape: [SEQUENCE LENGTH][EMBED DIM]
    float* __restrict__ max_rows,       // Shape: [SEQUENCE LENGTH] 
    float* __restrict__ sum_rows,       // Shape: [SEQUENCE LENGTH]
    float* __restrict__ D               // Shape: [SEQUENCE LENGTH]
){

    // First, we calculate the D matrix
    // I dynamically allocate memory since it exceeds 48 KB :)
    extern __shared__ float shared_memory[];

    // Defining shared memory blocks for all weights with QUERY_ROWS_BACK rows :)
    float* __restrict__ query_tile_base    = shared_memory;
    float* __restrict__ output_tile_base   = &shared_memory[QUERY_ROWS_BACK * EMBED_DIM];
    float* __restrict__ d_query_tile_base  = &shared_memory[2 * QUERY_ROWS_BACK * EMBED_DIM];
    float* __restrict__ d_output_tile_base = &shared_memory[3 * QUERY_ROWS_BACK * EMBED_DIM];
    
    // Defining shared memory blocks for all weights with KV_ROWS_BACK now!
    float* __restrict__ key_tile_base      = &shared_memory[4 * QUERY_ROWS_BACK * EMBED_DIM];
    float* __restrict__ value_tile_base    = &shared_memory[(4 * QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* __restrict__ d_key_tile_base    = &shared_memory[(4 * QUERY_ROWS_BACK + 2 * KV_ROWS_BACK) * EMBED_DIM];
    float* __restrict__ d_value_tile_base  = &shared_memory[(4 * QUERY_ROWS_BACK + 3 * KV_ROWS_BACK) * EMBED_DIM];
    
    // Storing normalization and D values    
    float* __restrict__ scores_tile_base   = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM];
    float* __restrict__ d_P_tile_base      = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + KV_ROWS_BACK * QUERY_ROWS_BACK];
    float* __restrict__ d_S_tile_base      = &shared_memory[4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + 2 * KV_ROWS_BACK * QUERY_ROWS_BACK]; 

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

    const int x_offset = (threadIdx.y % 2) * 128 + threadIdx.x * 4;
    const int y_offset = (threadIdx.y % 2) * 4 + threadIdx.x / 8;
    
    const float scale = 1.0 / sqrt(EMBED_DIM);

    // Denotes the offset per block!
    const int i = blockIdx.x * KV_ROWS_BACK; 

    /********************************************************
    *
    * Two rows of the block fetches one row of the matrices!
    * This is done in float4 vectorized fetches for KV :)
    *  
    ********************************************************/

    if(threadIdx.y < 16){ 
    
        const float4* key_ptr = reinterpret_cast<const float4*>(&key[(i + threadIdx.y / 2) * EMBED_DIM + x_offset]);
        float4* shared_key_ptr = reinterpret_cast<float4*>(&key_tile(threadIdx.y / 2, x_offset));

        const float4* value_ptr = reinterpret_cast<const float4*>(&value[(i + threadIdx.y / 2) * EMBED_DIM + x_offset]);
        float4* shared_value_ptr = reinterpret_cast<float4*>(&value_tile(threadIdx.y / 2, x_offset));
        
        float4* d_key_ptr = reinterpret_cast<float4*>(&d_key[(i + threadIdx.y / 2) * EMBED_DIM + x_offset]);
        float4* shared_d_key_ptr = reinterpret_cast<float4*>(&d_key_tile(threadIdx.y / 2, x_offset));
        
        float4* d_value_ptr = reinterpret_cast<float4*>(&d_value[(i + threadIdx.y / 2) * EMBED_DIM + x_offset]);
        float4* shared_d_value_ptr = reinterpret_cast<float4*>(&d_value_tile(threadIdx.y / 2, x_offset));

        *shared_key_ptr = *key_ptr;
        *shared_value_ptr = *value_ptr;
        *shared_d_key_ptr = *d_key_ptr;
        *shared_d_value_ptr = *d_value_ptr;
    }
    
    // Now, we process the query blocks!

    for(int j = 0; j < MAX_SEQUENCE_LENGTH; j += QUERY_ROWS_BACK){
        
        /********************************************************
        *
        * Two rows of the block fetches one row of the matrix
        * We utilize all threads for the Q, O, dQ, dO fetches :)
        *  
        ********************************************************/

        const int local_offset = (j + threadIdx.y / 2) * EMBED_DIM;
        
        const float4* query_ptr = reinterpret_cast<const float4*>(&query[local_offset + x_offset]);
        float4* shared_query_ptr = reinterpret_cast<float4*>(&query_tile(threadIdx.y / 2, x_offset));
        
        const float4* output_ptr = reinterpret_cast<const float4*>(&output[local_offset + x_offset]);
        float4* shared_output_ptr = reinterpret_cast<float4*>(&output_tile(threadIdx.y / 2, x_offset));
        
        float4* d_output_ptr = reinterpret_cast<float4*>(&d_output[local_offset + x_offset]);
        float4* shared_d_output_ptr = reinterpret_cast<float4*>(&d_output_tile(threadIdx.y / 2, x_offset));
        
        float4* d_query_ptr = reinterpret_cast<float4*>(&d_query[local_offset + x_offset]);
        float4* shared_d_query_ptr = reinterpret_cast<float4*>(&d_query_tile(threadIdx.y / 2, x_offset));
        
        *shared_query_ptr = *query_ptr;
        *shared_output_ptr = *output_ptr;
        *shared_d_output_ptr = *d_output_ptr;
        *shared_d_query_ptr = *d_query_ptr;

        // Ensuring all move operations are complete before we proceed :)
        __syncthreads();

        /******************************************************************
        *
        * For Calculating QK^T of shape 16 x 8, each thread can do 1/8th
        * of the work! This allows each thread to contribute and we use 
        * super-fast warp level actions to sum up calculated values :)
        *
        ******************************************************************/
        

        int row_offset = (threadIdx.x % 8) * 32;
        float sum = 0.0f;

        // Calculating the score of dimension [16 x 8] and 2 warps calculate one row of values
        for(int k = 0; k < 32; k++){
            sum += query_tile(threadIdx.y / 2, row_offset + k) * key_tile(y_offset, row_offset + k);
        }
        
        __syncwarp();

        sum += __shfl_down_sync(0xff, sum, 4);
        sum += __shfl_down_sync(0xff, sum, 2);
        sum += __shfl_down_sync(0xff, sum, 1);

        __syncwarp();

        // Since 8 threads contribute to 1 value, one every 8 thread should store too!
        if(threadIdx.x % 8 == 0){
            sum *= scale;
            scores_tile(threadIdx.y / 2, y_offset) = expf(sum - max_rows[j + threadIdx.y / 2]) / sum_rows[j + threadIdx.y / 2];
        }

        __syncthreads();
        
        /**********************************************************************
        *
        * Calculating derivative of Value
        * --------------------------------
        *
        * To calculate dV, we follow the following formula:  
        * dV = dV + S^T x dO; where S is calculated above :)
        * 
        * Since we require 8 x 256 values, each thread calculates 2 values!
        *
        **********************************************************************/

        // Defining the row and column offsets for each thread
        row_offset = threadIdx.y / 4;
        const int offset = (threadIdx.y % 4) * 64 + threadIdx.x * 2;

        for(int k = 0; k < 2; k++){
            
            sum = 0.0f;

            for(int l = 0; l < 16; l++){
                sum += scores_tile(l, row_offset) * d_output_tile(l, k + offset);
            }

            d_value_tile(row_offset, k + offset) += sum;
        }


        /************************************************************************
        *
        * Now we calculate the dS used to update dQ, and dK. Here are the steps:
        * 
        * 1. dS = dO x V^T
        * 2. dS = S x dS - (rowsum(dO @ O))
        *
        ************************************************************************/

        // First we calculate dO x V^T!

        row_offset = (threadIdx.x % 8) * 32;
        sum = 0;

        for(int k = 0; k < 32; k++){
            sum += d_output_tile(threadIdx.y / 2, row_offset + k) * value_tile(y_offset, row_offset + k);
        }

        __syncwarp();

        sum += __shfl_down_sync(0xff, sum, 4);
        sum += __shfl_down_sync(0xff, sum, 2);
        sum += __shfl_down_sync(0xff, sum, 1);

        __syncwarp();

        // Storing the value into shared memory
        if(threadIdx.x % 8 == 0){
            d_P_tile(threadIdx.y / 2, y_offset) = sum - D[j + threadIdx.y / 2];
        }

        __syncthreads();

        // Now we calculate the dS value! :)
        if(threadIdx.y < KV_ROWS_BACK){

            if(threadIdx.x < QUERY_ROWS_BACK){
                d_S_tile(threadIdx.x, threadIdx.y) = scores_tile(threadIdx.x, threadIdx.y) * d_P_tile(threadIdx.x, threadIdx.y);
            }
        }

        __syncthreads();

        /******************************
        *
        * Now we calculate the dQ!
        * 1. dQ = dS x K
        *
        ******************************/

        // Now we update the dQ matrix!
        for(int k = 0; k < 4; k++){
            
            sum = 0.0f;

            for(int l = 0; l < KV_ROWS_BACK; l++){
                sum += d_S_tile(threadIdx.y / 2, l) * key_tile(l, x_offset + k);
            }

            d_query_tile(threadIdx.y / 2, x_offset + k) += sum * scale;
        }

        __syncthreads();

        // Storing the dQuery values
        // We use atomic additions to avoid inter-block race conditions!

        float* d_query_base = &d_query[local_offset + x_offset];
        float* shared_d_query_base = &d_query_tile(threadIdx.y / 2, x_offset);
        
        atomicAdd(&d_query_base[0], shared_d_query_base[0]);
        atomicAdd(&d_query_base[1], shared_d_query_base[1]);
        atomicAdd(&d_query_base[2], shared_d_query_base[2]);
        atomicAdd(&d_query_base[3], shared_d_query_base[3]);

        // Ensuring all move operations are complete before we proceed :)
        __syncthreads();

        /*********************************
        *
        * Finally, we calculate dK!
        * 
        * 1. dK = dK + dS^T x Q 
        *
        *********************************/

        const int key_offset = (threadIdx.y % 4) * 64 + threadIdx.x * 2;

        // Calculating dK
        for(int k = 0; k < 2; k++){
            
            sum = 0.0f;

            for(int l = 0; l < QUERY_ROWS_BACK; l++){
                sum += d_S_tile(l, threadIdx.y / 4) * query_tile(l, key_offset + k);
            }

            d_key_tile(threadIdx.y / 4, key_offset + k) += sum * scale;
        }
    }

    __syncthreads();

    // Storing the dV, dK matrices back into global memory!
    if(threadIdx.y < 16){ 
        
        const int o = (i + threadIdx.y / 2) * EMBED_DIM + x_offset;

        float4* d_key_ptr = reinterpret_cast<float4*>(&d_key[o]);
        float4* shared_d_key_ptr = reinterpret_cast<float4*>(&d_key_tile(threadIdx.y / 2, x_offset));
    
        float4* d_value_ptr = reinterpret_cast<float4*>(&d_value[o]);
        float4* shared_d_value_ptr = reinterpret_cast<float4*>(&d_value_tile(threadIdx.y / 2, x_offset));

        *d_value_ptr = *shared_d_value_ptr;
        *d_key_ptr = *shared_d_key_ptr;
    }
}

std::vector<torch::Tensor> calculate_attention_scores_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    
    const int batch_size = query.size(0);

    // Create output tensors
    auto output = torch::zeros({batch_size, MAX_SEQUENCE_LENGTH, EMBED_DIM}, query.options());
    auto sum_rows = torch::zeros({batch_size, MAX_SEQUENCE_LENGTH}, query.options());
    auto max_rows = torch::full({batch_size, MAX_SEQUENCE_LENGTH}, -INFINITY, query.options());   
    auto logsumexp = torch::zeros({batch_size, MAX_SEQUENCE_LENGTH}, query.options());

    // Defining CUDA streams for parallelization across batches
    std::vector<cudaStream_t> streams(batch_size);

    // Creating PyTorch streams
    for(int b = 0; b < batch_size; b++){
        cudaStreamCreateWithPriority(&streams[b], cudaStreamNonBlocking, -1);
    }

    // Calculate grid and block dimensions
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / BLOCK_DIM);

    size_t shared_mem_size = 2 * sizeof(float) * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM) + 
        sizeof(float) * QUERY_ROWS * KV_ROWS;
    
    // Need to increase shared memory size limit for the kernel
    cudaFuncSetAttribute(
        calculate_attention_scores,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    for(int b = 0; b < batch_size; b++){
        
        // Launching the kernel
        calculate_attention_scores<<<blocks, threads, shared_mem_size, streams[b]>>>(
            query[b].data_ptr<float>(),
            key[b].data_ptr<float>(),
            value[b].data_ptr<float>(),
            max_rows[b].data_ptr<float>(),
            sum_rows[b].data_ptr<float>(),
            output[b].data_ptr<float>()
        );
    }

    // Making sure all streams are synchronized before proceeding
    for(int b = 0; b < batch_size; b++) {
        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    return std::vector<torch::Tensor>{output, max_rows, sum_rows};
}

std::vector<torch::Tensor> calculate_multihead_attention_scores_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {

    auto batch_size = query.size(0);
    const int total_streams = batch_size * NUM_HEADS;

    // Splitting the needed tensors into head tensors
    auto query_splits = split_matrices(query, batch_size);
    auto key_splits = split_matrices(key, batch_size);
    auto value_splits = split_matrices(value, batch_size);
    
    // Preparing output vectors
    std::vector<torch::Tensor> all_outputs(total_streams);
    std::vector<torch::Tensor> all_max_rows(total_streams);
    std::vector<torch::Tensor> all_sum_rows(total_streams);
    
    // Create CUDA streams for parallel execution
    std::vector<cudaStream_t> streams(total_streams);
    
    for(int i = 0; i < total_streams; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, -1);
    }

    // Specifying the shape for each attention head
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / BLOCK_DIM);

    size_t shared_mem_size = 2 * sizeof(float) * (QUERY_ROWS * EMBED_DIM + KV_ROWS * EMBED_DIM + BLOCK_DIM) +
        sizeof(float) * QUERY_ROWS * KV_ROWS;
    
    cudaFuncSetAttribute(
        calculate_attention_scores,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    // Process each batch and head
    for(int b = 0; b < batch_size; b++) {

        for(int i = 0; i < NUM_HEADS; i++) {
        
            const int split_idx = b * NUM_HEADS + i;

            auto output = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, query.options());
            auto sum_rows = torch::zeros({MAX_SEQUENCE_LENGTH}, query.options());
            auto max_rows = torch::full({MAX_SEQUENCE_LENGTH}, -INFINITY, query.options());
            
            calculate_attention_scores<<<blocks, threads, shared_mem_size, streams[split_idx]>>>(
                static_cast<float*>(query_splits[split_idx].data_ptr()),
                static_cast<float*>(key_splits[split_idx].data_ptr()),
                static_cast<float*>(value_splits[split_idx].data_ptr()),
                max_rows.data_ptr<float>(),
                sum_rows.data_ptr<float>(),
                output.data_ptr<float>()
            );

            // Ensure computation is complete before copying
            cudaStreamSynchronize(streams[split_idx]);

            all_outputs[split_idx] = output.clone();
            all_max_rows[split_idx] = max_rows.clone();
            all_sum_rows[split_idx] = sum_rows.clone();
        }
    }

    // Synchronize all streams before tensor reorganization
    for(int i = 0; i < total_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Organize outputs by batch first, then concatenate heads
    std::vector<torch::Tensor> batch_outputs(batch_size);
    
    for(int b = 0; b < batch_size; b++) {
    
        // Creating separate head outputs to merge in!
        std::vector<torch::Tensor> head_outputs(NUM_HEADS);
    
        for(int h = 0; h < NUM_HEADS; h++) {
            head_outputs[h] = all_outputs[b * NUM_HEADS + h];
        }
    
        batch_outputs[b] = torch::cat(head_outputs, 1);
    }

    // Finally, everything is merged back in
    auto final_output = torch::stack(batch_outputs, 0);

    // Organize max_rows and sum_rows similarly
    std::vector<torch::Tensor> batch_max_rows(batch_size);
    std::vector<torch::Tensor> batch_sum_rows(batch_size);
    
    // Processing all max_rows for all batches
    for(int b = 0; b < batch_size; b++) {
    
        // Similarly, we do this head by head
        std::vector<torch::Tensor> head_max_rows;
        std::vector<torch::Tensor> head_sum_rows;

        head_max_rows.reserve(NUM_HEADS);
        head_sum_rows.reserve(NUM_HEADS);

        for(int h = 0; h < NUM_HEADS; h++) {
            head_max_rows.push_back(all_max_rows[b * NUM_HEADS + h].to(query.device()));
            head_sum_rows.push_back(all_sum_rows[b * NUM_HEADS + h].to(query.device()));
        }

        // Stacking the max rows and sum rows vectors
        batch_max_rows[b] = torch::stack(head_max_rows, 0).to(query.device());
        batch_sum_rows[b] = torch::stack(head_sum_rows, 0).to(query.device());
    }

    // Performing the final stack, and we're done!
    torch::Tensor final_max_rows = torch::stack(batch_max_rows, 0);
    torch::Tensor final_sum_rows = torch::stack(batch_sum_rows, 0);

    // Clean up streams
    for(int i = 0; i < total_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return std::vector<torch::Tensor>{final_output, final_max_rows, final_sum_rows};
}

std::vector<torch::Tensor> calculate_attention_scores_backward_cuda(
    torch::Tensor query, 
    torch::Tensor key, 
    torch::Tensor value, 
    torch::Tensor output, 
    torch::Tensor d_output, 
    torch::Tensor max_rows, 
    torch::Tensor sum_rows){
        
    auto batch_size = query.size(0);

    // Create output tensors
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_value = torch::zeros_like(value);
    
    // Create D tensor for each batch
    torch::Tensor D = torch::sum(d_output * output, -1, true);

    // Calculate grid and block dimensions
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / KV_ROWS_BACK);

    // Define shared memory size
    size_t shared_mem_size = sizeof(float) * (4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + 
        3 * KV_ROWS_BACK * QUERY_ROWS_BACK);

    // Set maximum shared memory size for the kernel
    cudaFuncSetAttribute(
        calculate_attention_scores_backwards,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    // Create CUDA streams for parallel execution across batches
    std::vector<cudaStream_t> streams(batch_size);
    
    // Initialize streams with high priority for backward pass
    for(int b = 0; b < batch_size; b++) {
        cudaStreamCreateWithPriority(&streams[b], cudaStreamNonBlocking, -1);
    }

    // Launch kernels in parallel using streams
    for(int b = 0; b < batch_size; b++) {

        calculate_attention_scores_backwards<<<blocks, threads, shared_mem_size, streams[b]>>>(
            query[b].data_ptr<float>(),
            key[b].data_ptr<float>(),
            value[b].data_ptr<float>(),
            output[b].data_ptr<float>(),
            d_query[b].data_ptr<float>(),
            d_key[b].data_ptr<float>(),
            d_value[b].data_ptr<float>(),
            d_output[b].data_ptr<float>(),
            max_rows[b].data_ptr<float>(),
            sum_rows[b].data_ptr<float>(),
            D[b].data_ptr<float>()
        );

        cudaStreamSynchronize(streams[b]);
    }

    // Synchronize and cleanup streams
    for(int b = 0; b < batch_size; b++) {
        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    return std::vector<torch::Tensor>{d_query, d_key, d_value};
}

std::vector<torch::Tensor> calculate_multihead_attention_backward_cuda(
    torch::Tensor query, 
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor d_output,
    torch::Tensor max_rows,
    torch::Tensor sum_rows) {

    const int batch_size = query.size(0);
    const int total_streams = batch_size * NUM_HEADS;

    // Splitting the needed tensors into head tensors
    auto query_splits = split_matrices(query, batch_size);
    auto key_splits = split_matrices(key, batch_size);
    auto value_splits = split_matrices(value, batch_size);
    auto output_splits = split_matrices(output, batch_size);
    auto d_output_splits = split_matrices(d_output, batch_size);

    auto max_rows_splits = split_vectors(max_rows);
    auto sum_rows_splits = split_vectors(sum_rows);

    // Preparing output vectors
    std::vector<torch::Tensor> d_queries(total_streams);
    std::vector<torch::Tensor> d_keys(total_streams);
    std::vector<torch::Tensor> d_values(total_streams);

    // Create CUDA streams for parallel execution
    std::vector<cudaStream_t> streams(total_streams);

    for(int i = 0; i < total_streams; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, -1);
    }

    // Specifying the shape for each attention head
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(MAX_SEQUENCE_LENGTH / KV_ROWS_BACK);
    
    // Defining the shared memory requirements
    size_t shared_mem_size = sizeof(float) * (4 * (QUERY_ROWS_BACK + KV_ROWS_BACK) * EMBED_DIM + 3 * KV_ROWS_BACK * QUERY_ROWS_BACK);

    // Need to increase shared memory size limit for the kernel
    cudaFuncSetAttribute(
        calculate_attention_scores_backwards,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    // Processing each head using its own stream
    for(int b = 0; b < batch_size; b++){

        for(int i = 0; i < NUM_HEADS; i++) {

            const int split_idx = b * NUM_HEADS + i;

            // Pre-allocate output tensors
            d_queries[split_idx] = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, query.options());
            d_keys[split_idx] = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, key.options());
            d_values[split_idx] = torch::zeros({MAX_SEQUENCE_LENGTH, EMBED_DIM}, value.options());

            // Calculating the D values
            torch::Tensor D = (d_output_splits[split_idx] * output_splits[split_idx]).sum(1);

            // Launch kernel
            calculate_attention_scores_backwards<<<blocks, threads, shared_mem_size, streams[split_idx]>>>(
                static_cast<float*>(query_splits[split_idx].data_ptr()),
                static_cast<float*>(key_splits[split_idx].data_ptr()),
                static_cast<float*>(value_splits[split_idx].data_ptr()),
                static_cast<float*>(output_splits[split_idx].data_ptr()),
                d_queries[split_idx].data_ptr<float>(),
                d_keys[split_idx].data_ptr<float>(),
                d_values[split_idx].data_ptr<float>(),
                static_cast<float*>(d_output_splits[split_idx].data_ptr()),
                static_cast<float*>(max_rows_splits[split_idx].data_ptr()),
                static_cast<float*>(sum_rows_splits[split_idx].data_ptr()),
                D.data_ptr<float>()
            );

            cudaStreamSynchronize(streams[split_idx]);
        }
    }

    // Synchronizing all streams
    for(int i = 0; i < total_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Organize derivatives by batch first, then concatenate them by heads
    std::vector<torch::Tensor> batch_derivatives_queries(batch_size);
    std::vector<torch::Tensor> batch_derivatives_values(batch_size);
    std::vector<torch::Tensor> batch_derivatives_keys(batch_size);
    
    for(int b = 0; b < batch_size; b++) {
    
        // Creating separate head outputs to merge in!
        std::vector<torch::Tensor> d_queries_batch;
        std::vector<torch::Tensor> d_values_batch;
        std::vector<torch::Tensor> d_keys_batch;

        for(int h = 0; h < NUM_HEADS; h++) {

            const int batch_idx = b * NUM_HEADS + h;

            d_queries_batch.push_back(d_queries[batch_idx]);
            d_values_batch.push_back(d_values[batch_idx]);
            d_keys_batch.push_back(d_keys[batch_idx]);
            
        }
    
        batch_derivatives_queries[b] = torch::cat(d_queries_batch, 1);
        batch_derivatives_values[b] = torch::cat(d_values_batch, 1);
        batch_derivatives_keys[b] = torch::cat(d_keys_batch, 1);
    }

    auto d_queries_final = torch::stack(batch_derivatives_queries, 0);
    auto d_values_final = torch::stack(batch_derivatives_values, 0);
    auto d_keys_final = torch::stack(batch_derivatives_keys, 0);

    return std::vector<torch::Tensor>{d_queries_final, d_keys_final, d_values_final};
}