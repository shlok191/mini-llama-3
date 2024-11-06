#pragma once
#include <torch/extension.h>

// Method definition of the forward pass for the embedding layer
std::vector<torch::Tensor> embedding_forward_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int32_t padding_token_index);

// Method definition of the backward pass for the embedding layer
std::vector<torch::Tensor> embedding_backward_cuda(
    const torch::Tensor& gradient_output,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int32_t padding_token_index);
