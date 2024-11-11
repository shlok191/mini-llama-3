#pragma once
#include <torch/extension.h>

// Method definition of the forward pass for a Linear layer
torch::Tensor linear_forward_cuda(
    const torch::Tensor& X,
    const torch::Tensor& weights);