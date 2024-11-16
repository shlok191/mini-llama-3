import torch
import time
from cuda_kernels import embedding_forward
import nvtx

@nvtx.annotate(color="blue")
def wrapper(indices, table):
    return embedding_forward(indices, table)

# Set device to CUDA
device = torch.device('cuda')

# Define dimensions (using typical transformer dimensions)
vocab_size = 32000
embed_dim = 4096
seq_length = 2048
batch_size = 32

# Generate random input data
indices = torch.randint(0, vocab_size, (seq_length,), device=device, dtype=torch.int32)
embedding_table = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float32)

# Standard PyTorch embedding
def standard_embedding(indices, table):
    return torch.nn.functional.embedding(indices, table)

# Warm-up GPU
print("Warming up GPU...")
for _ in range(10):
    _ = standard_embedding(indices, embedding_table)
    _ = embedding_forward(indices, embedding_table)

# Test correctness
print("\nTesting correctness...")
with torch.no_grad():
    # Output from custom CUDA function
    output_custom = wrapper(indices, embedding_table)
    
    # Output from standard embedding
    output_standard = standard_embedding(indices, embedding_table)
    
    # Compare outputs
    max_difference = (output_custom - output_standard).abs().max().item()
    print(f'Maximum difference between outputs: {max_difference}')
    
    if max_difference < 1e-3:
        print('Outputs are close enough.')
    else:
        print('Outputs differ significantly!')
        
    # Additional statistics
    mean_diff = (output_custom - output_standard).abs().mean().item()
    print(f'Mean absolute difference: {mean_diff}')

# Measure execution time
print("\nMeasuring performance...")
num_runs = 100

# Time custom CUDA function
torch.cuda.synchronize()
start_time = time.time()
for _ in range(num_runs):
    output_custom = embedding_forward(indices, embedding_table)
torch.cuda.synchronize()
custom_time = (time.time() - start_time) / num_runs

# Time standard embedding
torch.cuda.synchronize()
start_time = time.time()
for _ in range(num_runs):
    output_standard = standard_embedding(indices, embedding_table)
torch.cuda.synchronize()
standard_time = (time.time() - start_time) / num_runs

# Print performance results
print(f'\nPerformance Results (averaged over {num_runs} runs):')
print(f'Custom CUDA embedding time: {custom_time * 1000:.3f} ms')
print(f'Standard embedding time: {standard_time * 1000:.3f} ms')
print(f'Speedup: {standard_time / custom_time:.2f}x')

# Memory usage statistics
def get_size_str(tensor):
    return f'{tensor.element_size() * tensor.nelement() / (1024 * 1024):.2f} MB'

print('\nMemory Usage:')
print(f'Embedding table: {get_size_str(embedding_table)}')
print(f'Input indices: {get_size_str(indices)}')
print(f'Output tensor: {get_size_str(output_custom)}')

# Additional performance metrics
print('\nAdditional Metrics:')
elements_per_second = (batch_size * seq_length * embed_dim) / custom_time
print(f'Elements processed per second: {elements_per_second/1e9:.2f} billion')
bandwidth = (batch_size * seq_length * embed_dim * 4) / (custom_time * 1e9)  # GB/s
print(f'Effective bandwidth: {bandwidth:.2f} GB/s')
