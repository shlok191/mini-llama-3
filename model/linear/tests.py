import torch
import time
from cuda_kernels import linear_forward
import nvtx

@nvtx.annotate(color="blue")
def wrapper(X, weights):
    return linear_forward(X, weights)

# Set device to CUDA
device = torch.device('cuda')

# Define dimensions (ensure they are multiples of 256)
in_features = 512
out_features = 1024

# Generate random input data and weights
X = torch.randn(64, in_features, device=device, dtype=torch.float32)
weights = torch.randn(in_features, out_features, device=device, dtype=torch.float32)

# Standard PyTorch matrix multiplication
def standard_linear(X, weights):
    return torch.matmul(X, weights)

# Warm-up GPU
for _ in range(10):
    _ = standard_linear(X, weights)
    _ = linear_forward(X, weights)

# Test correctness
with torch.no_grad():
    
    # Output from custom CUDA function
    output_custom = wrapper(X, weights)
    
    # Output from standard matrix multiplication
    output_standard = standard_linear(X, weights)

    # Compare outputs
    max_difference = (output_custom - output_standard).abs().max().item()
    
    print(f'Maximum difference between outputs: {max_difference}')
    if max_difference < 1e-3:
        print('Outputs are close enough.')
    else:
        print('Outputs differ significantly!')

# Measure execution time
num_runs = 25

# Time custom CUDA function
start_time = time.time()
for _ in range(num_runs):
    output_custom = linear_forward(X, weights)
torch.cuda.synchronize()
custom_time = (time.time() - start_time) / num_runs

# Time standard matrix multiplication
start_time = time.time()
for _ in range(num_runs):
    output_standard = standard_linear(X, weights)
torch.cuda.synchronize()
standard_time = (time.time() - start_time) / num_runs

print(f'Custom CUDA linear layer time per run: {custom_time * 1000:.3f} ms')
print(f'Standard matrix multiplication time per run: {standard_time * 1000:.3f} ms')

speedup = standard_time / custom_time
print(f'Speedup: {speedup:.2f}x')
