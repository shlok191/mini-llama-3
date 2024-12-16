#pragma once
#include <torch/extension.h>

// Method definition of the forward pass for the embedding layer
torch::Tensor embedding_forward_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& table);

// Method definition of the backward pass for the embedding layer
torch::Tensor embedding_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& table);
