#include <torch/extension.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>

// Constants for block dimensions
#define BLOCK_SIZE 16
#define TILE_SIZE 64
#define THREAD_SIZE  4 // Loading in 64 bytes per vectorized loading operation
#define VECTOR_WIDTH 4 // The chunk in which we are loading in values (here, 4 values per mem. operation)

// Each kernel utilizes 32 KB of shared memory when we use float32 values!
// Therefore, we can run a total of 3 blocks per SM :)
// This also maximizes active warps per SM!

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Forward pass kernel implementation
__global__ void linear_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const int in_features,
    const int out_features,
    const int batch_size,
    const int sequence_length) {
    
    // Defining shared memory for the block
    __shared__ float X_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float weights_shared[TILE_SIZE][TILE_SIZE];

    // Calculating the block position in the broader output matrix
    const int batch_idx = blockIdx.z;
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;

    const int batch_offset = batch_idx * sequence_length * in_features;
    const int output_batch_offset = batch_idx * sequence_length * out_features;

    // Moving the X pointer to the appropriate place in memory
    X += batch_offset;
    output += output_batch_offset;

    // Storing the 2D results generated by our thread in thread-local memory
    float values[THREAD_SIZE][THREAD_SIZE] = {0.0f};

    const int num_tiles = in_features / TILE_SIZE;

    #pragma unroll
    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {

        // Keep in mind, each thread pulls in COLUMN-WISE!
        // This is important, since then each warp will pull consecutive rows :)
        #pragma unroll
        for (int row_offset = 0; row_offset < TILE_SIZE; row_offset += BLOCK_SIZE) {
            
            int shared_row = row_offset + threadIdx.y;
            int global_row_X = block_row + shared_row; // global row for the input
            int global_row_W = tile_index * TILE_SIZE + shared_row; // global row for the weights

            int shared_col = threadIdx.x * VECTOR_WIDTH;
            int global_col_X = tile_index * TILE_SIZE + shared_col;
            int global_col_W = block_col + shared_col;

            // Reinterpret shared memory pointers as float4*
            float4* X_shared_ptr = reinterpret_cast<float4*>(&X_shared[shared_row][shared_col]);
            float4* weights_shared_ptr = reinterpret_cast<float4*>(&weights_shared[shared_row][shared_col]);

            // Reinterpret global memory pointers as float4*
            const float4* X_global_ptr = reinterpret_cast<const float4*>(&X[global_row_X * in_features + global_col_X]);
            const float4* weights_global_ptr = reinterpret_cast<const float4*>(&weights[global_row_W * out_features + global_col_W]);
            
            // Load data into shared memory
            *X_shared_ptr = *X_global_ptr;
            *weights_shared_ptr = *weights_global_ptr;

            __syncthreads();
        }

        // Compute the dot-product for the memory currently loaded in
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE; i++) {

                #pragma unroll
                for (int j = 0; j < THREAD_SIZE; j++) {
                    values[i][j] += X_shared[threadIdx.y + i * BLOCK_SIZE][k] * weights_shared[k][threadIdx.x + j * BLOCK_SIZE];
                }
            }
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final results to the output matrix
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE; i++) {
        
        int output_row = block_row + threadIdx.y + i * BLOCK_SIZE;
        
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE; j++) {

            int output_col = block_col + threadIdx.x + j * BLOCK_SIZE;
            output[output_row * out_features + output_col] = values[i][j];
        }
    }
}

__global__ void linear_backward_input_kernel(
    const float* __restrict__ grad_output,  // [sequence_length × out_features]
    const float* __restrict__ weights_T,    // [out_features × embedding_dim]
    float* __restrict__ grad_input,         // [sequence_length × embedding_dim]
    const int embedding_dim,                
    const int out_features,
    const int batch_size,
    const int sequence_length
) {

    // Shared memory for tiling
    __shared__ float grad_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float weights_shared[TILE_SIZE][TILE_SIZE];

    // Block position in output matrix
    const int batch_idx = blockIdx.z;
    const int block_row = blockIdx.y * TILE_SIZE;  // sequence dimension
    const int block_col = blockIdx.x * TILE_SIZE;  // embedding dimension

    const int grad_output_offset = batch_idx * sequence_length * out_features;
    const int grad_input_offset = batch_idx * sequence_length * embedding_dim;

    grad_output += grad_output_offset;
    grad_input += grad_input_offset;

    // Thread local accumulator
    float values[THREAD_SIZE][THREAD_SIZE] = {0};

    // Each block calculates one block of values, requiring us to traverse dL/dY columns or weights^T (dY/dX)'s rows!
    const int num_tiles = out_features / TILE_SIZE;

    #pragma unroll
    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {
        
        #pragma unroll
        for (int row_offset = 0; row_offset < TILE_SIZE; row_offset += BLOCK_SIZE) {
            
            int shared_row = row_offset + threadIdx.y;
            int global_row_grad_output = block_row + shared_row; // global row for the input
            int global_row_weight_T = tile_index * TILE_SIZE + shared_row; // global row for the weights

            int shared_col = threadIdx.x * VECTOR_WIDTH;
            int global_col_grad_output = tile_index * TILE_SIZE + shared_col;
            int global_col_weight_T = block_col + shared_col;

            // Reinterpret shared memory pointers as float4*
            float4* grad_shared_ptr = reinterpret_cast<float4*>(&grad_shared[shared_row][shared_col]);
            float4* weights_shared_ptr = reinterpret_cast<float4*>(&weights_shared[shared_row][shared_col]);

            // Reinterpret global memory pointers as float4*
            const float4* grad_global_ptr = reinterpret_cast<const float4*>(&grad_output[global_row_grad_output * out_features + global_col_grad_output]);
            const float4* weights_global_ptr = reinterpret_cast<const float4*>(&weights_T[global_row_weight_T * embedding_dim + global_col_weight_T]);
            
            // Load data into shared memory
            *grad_shared_ptr = *grad_global_ptr;
            *weights_shared_ptr = *weights_global_ptr;

            __syncthreads();
        }

        // Compute matrix multiplication for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
        
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE; i++) {
                
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE; j++) {
                    values[i][j] += grad_shared[threadIdx.y + i * BLOCK_SIZE][k] * weights_shared[k][threadIdx.x + j * BLOCK_SIZE];
                }
            }
        }

        __syncthreads();
    }

    // Write results to grad_input
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE; i++) {
        
        const int output_row = block_row + threadIdx.y + i * BLOCK_SIZE;
        
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE; j++) {

            const int output_col = block_col + threadIdx.x + j * BLOCK_SIZE;
            grad_input[output_row * embedding_dim + output_col] = values[i][j];
        }
    }
}

// Kernel for computing gradient with respect to weights (dL/dW)
// This will actually be used to update our weights :)
__global__ void linear_backward_weight_kernel(
    const float* __restrict__ input_T,     // [batch_idx x embedding_dim x seq_length]
    const float* __restrict__ grad_output, // [batch_idx x seq_length x out_features]
    float* __restrict__ grad_weights,      // [batch_idx x embedding_dim x out_features]
    const int out_features,
    const int sequence_length,
    const int embedding_dim,
    const int batch_size) {
    
    __shared__ float grad_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float input_shared[TILE_SIZE][TILE_SIZE];

    const int batch_idx = blockIdx.z;
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;

    const int input_offset = batch_idx * embedding_dim * sequence_length;
    const int grad_output_offset = batch_idx * sequence_length * out_features;
    const int grad_weights_offset = batch_idx * embedding_dim * out_features;

    input_T += input_offset;
    grad_output += grad_output_offset;
    grad_weights += grad_weights_offset;

    float values[THREAD_SIZE][THREAD_SIZE] = {0};

    const int num_tiles = sequence_length / TILE_SIZE;

    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {
        
        // Loading grad_output and input into shared memory
        for (int row_offset = 0; row_offset < TILE_SIZE; row_offset += BLOCK_SIZE) {
            
            int shared_row = row_offset + threadIdx.y;
            int global_row_input = block_row + shared_row;
            int global_row_grad = tile_index * TILE_SIZE + shared_row;

            int shared_col = threadIdx.x * VECTOR_WIDTH;
            int global_col_input = tile_index * TILE_SIZE + shared_col;;
            int global_col_grad = block_col + shared_col;

            float4* input_shared_ptr = reinterpret_cast<float4*>(&input_shared[shared_row][shared_col]);
            float4* grad_shared_ptr = reinterpret_cast<float4*>(&grad_shared[shared_row][shared_col]);

            const float4* input_global_ptr = reinterpret_cast<const float4*>(&input_T[global_row_input * sequence_length + global_col_input]);
            const float4* grad_global_ptr = reinterpret_cast<const float4*>(&grad_output[global_row_grad * out_features + global_col_grad]);
            
            *input_shared_ptr = *input_global_ptr;
            *grad_shared_ptr = *grad_global_ptr;
        }

        __syncthreads();

        // Computing partial results
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {

            #pragma unroll
            for (int i = 0; i < THREAD_SIZE; i++) {
            
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE; j++) {
                    values[i][j] += input_shared[threadIdx.y + i * BLOCK_SIZE][k] * grad_shared[k][threadIdx.x + j * BLOCK_SIZE];
                }
            }
        }

        __syncthreads();
    }

    // Writing the results back into global memory
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE; i++) {
        
        int output_row = block_row + threadIdx.y + i * BLOCK_SIZE;
    
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE; j++) {
            
            int output_col = block_col + threadIdx.x + j * BLOCK_SIZE;
            grad_weights[output_row * out_features + output_col] = values[i][j];
            
        }
    }
}

// C++ wrapper for the forward pass
torch::Tensor linear_forward_cuda(
    const torch::Tensor& X,
    const torch::Tensor& weights) {
    
    // Ensure the inputs are on CUDA and are contiguous
    CHECK_CUDA(X);
    CHECK_CUDA(weights);

    // Get dimensions
    const int batch_size = X.size(0);
    const int sequence_length = X.size(1);
    const int in_features = X.size(2);
    const int out_features = weights.size(1);  // weights is [in_features, out_features]

    // Create output tensor with same dtype as input
    auto output = torch::empty(
        {batch_size, sequence_length, out_features},  
        X.options()
    );

    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(out_features / TILE_SIZE, sequence_length / TILE_SIZE, batch_size);

    // Launch kernel with half precision pointers
    linear_forward_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(X.data_ptr<float>()),
        reinterpret_cast<const float*>(weights.data_ptr<float>()),
        reinterpret_cast<float*>(output.data_ptr<float>()),
        in_features,
        out_features,
        batch_size,
        sequence_length
    );

    return output;
}

torch::Tensor linear_backward_weights_cuda(
    const torch::Tensor& grad_output,  // [batch_idx x sequence_length × out_features]
    const torch::Tensor& input_T       // [batch_idx x embedding_dim x sequence_length]
) {

    // Input validation
    CHECK_CUDA(grad_output);
    CHECK_CUDA(input_T);

    // Get dimensions
    const int batch_size = input_T.size(0);
    const int embedding_dim = input_T.size(1);
    const int sequence_length = input_T.size(2);
    const int out_features = grad_output.size(2);

    // Create output tensors with correct shapes
    auto per_batch_grad_weights = torch::empty(
        {batch_size, embedding_dim, out_features},  // Same shape as input
        input_T.options()
    );

    // Calculate grid dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // For grad_weights (dL/dW = X^T × dL/dY)
    dim3 blocks_weight(
        out_features / TILE_SIZE,   // Columns of output
        embedding_dim / TILE_SIZE,  // Rows of output
        batch_size
    );

    linear_backward_weight_kernel<<<blocks_weight, threads>>>(
        reinterpret_cast<const float*>(input_T.data_ptr<float>()),
        reinterpret_cast<const float*>(grad_output.data_ptr<float>()),
        reinterpret_cast<float*>(per_batch_grad_weights.data_ptr<float>()),
        out_features,
        sequence_length,
        embedding_dim,
        batch_size
    );

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, 
        "Error in linear backward: ", cudaGetErrorString(cudaGetLastError()));

    // Add the gradients across the batches!
    return per_batch_grad_weights.sum(0);
}

torch::Tensor linear_backward_inputs_cuda(
    const torch::Tensor& grad_output,  // [sequence_length × out_features]
    const torch::Tensor& weights_T     // [out_features × embedding_dim]
) {

    // Input validation
    CHECK_CUDA(grad_output);
    CHECK_CUDA(weights_T);

    // Get dimensions
    const int batch_size = grad_output.size(0);
    const int sequence_length = grad_output.size(1);
    const int embedding_dim = weights_T.size(1);
    const int out_features = grad_output.size(2);

    // Create output tensors with correct shapes
    auto grad_input = torch::empty(
        {batch_size, sequence_length, embedding_dim},  // Same shape as input
        weights_T.options()
    );

    // Calculate grid dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // For grad_input (dL/dX = dL/dY × W^T)
    dim3 blocks_input(
        embedding_dim / TILE_SIZE,   // Columns of output
        sequence_length / TILE_SIZE, // Rows of output
        batch_size
    );

    linear_backward_input_kernel<<<blocks_input, threads>>>(
        reinterpret_cast<const float*>(grad_output.data_ptr<float>()),
        reinterpret_cast<const float*>(weights_T.data_ptr<float>()),
        reinterpret_cast<float*>(grad_input.data_ptr<float>()),
        embedding_dim,
        out_features,
        batch_size,
        sequence_length
    );

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, 
        "Error in linear backward: ", cudaGetErrorString(cudaGetLastError()));

    return grad_input;
}