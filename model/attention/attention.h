#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

std::vector<torch::Tensor> calculate_attention_scores_cuda(torch::Tensor query,torch::Tensor key, torch::Tensor value);
std::vector<torch::Tensor> calculate_attention_scores_backward_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output, torch::Tensor d_output, torch::Tensor logexp);
