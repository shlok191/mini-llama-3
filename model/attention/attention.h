#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

torch::Tensor calculate_attention_scores_cuda(torch::Tensor query,torch::Tensor key);