#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Defining forward pass 
std::vector<torch::Tensor> calculate_attention_scores_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value);
std::vector<torch::Tensor> calculate_multihead_attention_scores_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value);

// Defining backward pass
std::vector<torch::Tensor> calculate_attention_scores_backward_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output, torch::Tensor d_output, torch::Tensor max_rows, torch::Tensor sum_rows);
std::vector<torch::Tensor> calculate_multihead_attention_backward_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output, torch::Tensor d_output, torch::Tensor max_rows, torch::Tensor sum_rows);