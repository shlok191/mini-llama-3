#pragma once
#include <torch/extension.h>

// Method definition of the forward pass for a Linear layer
torch::Tensor linear_forward_cuda(
    const torch::Tensor& X,
    const torch::Tensor& weights);

torch::Tensor linear_backward_inputs_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& weights_T);

torch::Tensor linear_backward_weights_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_T);