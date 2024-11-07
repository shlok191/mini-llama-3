#pragma once
#include <torch/extension.h>

// Method definition of the forward pass for a Linear layer
std::vector<torch::Tensor> linear_forward_cuda(
    const torch::Tensor& X,
    const torch::Tensor& weights,
    const torch::Tensor& bias);

// Method definition of the backward pass for a Linear layer
std::vector<torch::Tensor> linear_backward_cuda(
    const torch::Tensor& gradient_output,
    const torch::Tensor& X,
    const torch::Tensor& weights,
    const torch::Tensor& bias);
