import torch
import mini_llama
import time
import math

def test_attention_implementation():
    # Set the dimensions
    sequence_length = 512
    embedding_dim = 256

    # Create random input tensors on the GPU
    query = torch.randn(sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')
    key = torch.randn(sequence_length, embedding_dim,
                      dtype=torch.float32,
                      device='cuda')
    value = torch.randn(sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')

    # Time your CUDA implementation
    start_time = time.perf_counter()
    cuda_output, _, _ = mini_llama.cuda.attention_forward(query, key, value)
    
    torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    cuda_time = time.perf_counter() - start_time

    # Time the PyTorch implementation
    start_time = time.perf_counter()
    scores = torch.matmul(query, key.transpose(0, 1))
    scores = scores / (embedding_dim ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(attention_weights, value)
    
    torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    pytorch_time = time.perf_counter() - start_time

    # Compare the results
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance
    
    max_diff = torch.max(torch.abs(cuda_output - pytorch_output))

    print(f"Maximum difference between implementations: {max_diff}")
    is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("✅ Test passed! CUDA implementation matches PyTorch")
    else:
        print("❌ Test failed! Implementations produce different results")

        # Print detailed statistics for debugging
        print("\nDetailed comparison:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_output - pytorch_output))}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_output - pytorch_output))}")

        # Sample a few positions for comparison
        sample_idx = torch.randint(0, sequence_length, (5,))
        print("\nSample values comparison:")
        for idx in sample_idx:
            idx = idx.item()  # Convert tensor to int
            print(f"\nPosition {idx}:")
            print(f"CUDA output: {cuda_output[idx][:5]}")  # First 5 values
            print(f"PyTorch output: {pytorch_output[idx][:5]}")


    print("\nTiming Results:")
    print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time/cuda_time:.2f}x")
    
def test_attention_backward(num_runs=100):
    """
    Test and benchmark the backward pass of our CUDA attention implementation
    against PyTorch's native implementation.
    
    Args:
        num_runs (int): Number of runs for timing comparison
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create input tensors on GPU
    query = torch.randn(512, 256, device='cuda', requires_grad=True)
    key = torch.randn(512, 256, device='cuda', requires_grad=True)
    value = torch.randn(512, 256, device='cuda', requires_grad=True)
    
    # Timing CUDA implementation
    cuda_times = []
    print(f"\nBenchmarking CUDA implementation ({num_runs} runs)...")
    
    for i in range(num_runs):
    
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        output, max_rows, sum_rows = mini_llama.cuda.attention_forward(query, key, value)
        
        grad_output = torch.randn_like(output)
        
        grad_query, grad_key, grad_value = mini_llama.cuda.attention_backward(
            query, key, value, output, grad_output, max_rows, sum_rows
        )
        
        end_time.record()
        
        torch.cuda.synchronize()
        cuda_times.append(start_time.elapsed_time(end_time))
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} CUDA runs")
    
    # Timing PyTorch implementation
    torch_times = []
    print(f"\nBenchmarking PyTorch implementation ({num_runs} runs)...")
    
    for i in range(num_runs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        scores = torch.matmul(query, key.transpose(0, 1)) / 16.0
        attention_weights = torch.softmax(scores, dim=-1)
        torch_output = torch.matmul(attention_weights, value)
        torch_output.backward(grad_output)
        end_time.record()
        
        torch.cuda.synchronize()
        torch_times.append(start_time.elapsed_time(end_time))
        
        # Reset gradients
        if(i is not num_runs - 1):
            query.grad = None
            key.grad = None
            value.grad = None
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} PyTorch runs")
    
    # Calculate statistics
    cuda_mean = sum(cuda_times) / len(cuda_times)
    cuda_std = (sum((x - cuda_mean) ** 2 for x in cuda_times) / len(cuda_times)) ** 0.5
    
    torch_mean = sum(torch_times) / len(torch_times)
    torch_std = (sum((x - torch_mean) ** 2 for x in torch_times) / len(torch_times)) ** 0.5
    
    # Compare gradients
    print("\nGradient Comparison:")
    print(f"Maximum difference in query gradients: {torch.max(torch.abs(grad_query - query.grad)).item():.6f}")
    print(f"Maximum difference in key gradients: {torch.max(torch.abs(grad_key - key.grad)).item():.6f}")
    print(f"Maximum difference in value gradients: {torch.max(torch.abs(grad_value - value.grad)).item():.6f}")
    
    # Print timing results
    print("\nTiming Results:")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")
    
    # Assert gradient correctness
    assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-3), "Query gradients don't match!"
    assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), "Value gradients don't match!"
    print("\nAll gradient checks passed!")


if __name__ == "__main__":
    
    test_attention_implementation()
    print("\n")
    test_attention_backward()