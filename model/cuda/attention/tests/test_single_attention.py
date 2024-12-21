import torch
from mini_llama.cuda import attention_forward, attention_backward
import time
import math

def test_attention_implementation(num_runs = 1):
    # Set the dimensions
    batch_size = 1
    sequence_length = 512
    embedding_dim = 256
    
    for _ in range(num_runs):
        
        # Create random input tensors on the GPU with batch dimension
        # Shape: [batch_size, sequence_length, embedding_dim]
        
        query = torch.randn(batch_size, sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')
        
        key = torch.randn(batch_size, sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')
        
        value = torch.randn(batch_size, sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')

        # Time your CUDA implementation
        start_time = time.perf_counter()
        cuda_output, _, _ = attention_forward(query, key, value)
        
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start_time

        # Time the PyTorch implementation with batching
        start_time = time.perf_counter()
        
        # Reshape inputs to separate heads
        batch_size, seq_len, _ = query.shape
        query_reshaped = query.view(batch_size, seq_len, embedding_dim)
        key_reshaped = key.view(batch_size, seq_len, embedding_dim)
        value_reshaped = value.view(batch_size, seq_len, embedding_dim)

        # Calculate attention scores for each batch and head
        scores = torch.matmul(query_reshaped, key_reshaped.transpose(-2, -1))
        
        scores = scores / math.sqrt(embedding_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        
        pytorch_output = torch.matmul(attention_weights, value_reshaped)
        
        # Reshape back to original format
        # [batch, heads, seq, dim] -> [batch, seq, heads*dim]
        pytorch_output = pytorch_output.view(
            batch_size, seq_len, embedding_dim)
        
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start_time

        print(f"CUDA Output Shape: {cuda_output.shape}")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        
        # Compare the results
        rtol = 1e-3
        atol = 1e-5
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
            batch_idx = torch.randint(0, batch_size, (2,))
            seq_idx = torch.randint(0, sequence_length, (3,))
            
            print("\nSample values comparison:")
            for b in batch_idx:
            
                for s in seq_idx:
                    b, s = b.item(), s.item()
                    print(f"\nBatch {b}, Position {s}:")
                    print(f"CUDA output: {cuda_output[b, s, :5]}")  # First 5 values
                    print(f"PyTorch output: {pytorch_output[b, s, :5]}")

        print("\nTiming Results:")
        print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")
        print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
        print(f"Speedup: {pytorch_time/cuda_time:.2f}x")

def test_attention_backward(num_runs=100):
    
    torch.manual_seed(42)
    
    # Define dimensions
    batch_size = 16
    sequence_length = 256
    embedding_dim = 256
    
    # Create input tensors with batch dimension
    query = torch.randn(batch_size, sequence_length, embedding_dim, 
        device='cuda', requires_grad=True)
    
    key = torch.randn(batch_size, sequence_length, embedding_dim, 
        device='cuda', requires_grad=True)
    
    value = torch.randn(batch_size, sequence_length, embedding_dim, 
        device='cuda', requires_grad=True)
    
    # Timing CUDA implementation
    cuda_times = []
    torch_times = []
    
    print(f"\nBenchmarking CUDA implementation ({num_runs} runs)...")
    
    for i in range(num_runs):
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        output, max_rows, sum_rows = attention_forward(query, key, value)
        grad_output = torch.randn_like(output)
        
        grad_query, grad_key, grad_value = attention_backward(
            query, key, value, output, grad_output, max_rows, sum_rows
        )
        
        end_time.record()
        torch.cuda.synchronize()
        
        cuda_times.append(start_time.elapsed_time(end_time))
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} CUDA runs")
    
        # Timing PyTorch implementation
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Reshape for multi-head attention
        query_reshaped = query.view(batch_size, sequence_length, embedding_dim)
        key_reshaped = key.view(batch_size, sequence_length, embedding_dim)
        value_reshaped = value.view(batch_size, sequence_length, embedding_dim)
        
        scores = torch.matmul(query_reshaped, key_reshaped.transpose(-2, -1)) / math.sqrt(embedding_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        
        torch_output = torch.matmul(attention_weights, value_reshaped)
        torch_output = torch_output.view(batch_size, sequence_length, embedding_dim)
        torch_output.backward(grad_output)
        
        end_time.record()
        
        torch.cuda.synchronize()
        torch_times.append(start_time.elapsed_time(end_time))
        
        assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-3), "Query gradients don't match!"
        assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-3), "Key gradients don't match!"
        assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), "Value gradients don't match!"
        
        if(i != num_runs - 1):
            query.grad = None
            key.grad = None
            value.grad = None
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} PyTorch runs")

        
    # Calculate statistics and print results
    cuda_mean = sum(cuda_times) / len(cuda_times)
    cuda_std = (sum((x - cuda_mean) ** 2 for x in cuda_times) / len(cuda_times)) ** 0.5
    
    torch_mean = sum(torch_times) / len(torch_times)
    torch_std = (sum((x - torch_mean) ** 2 for x in torch_times) / len(torch_times)) ** 0.5
    
    print("\nGradient Comparison:")
    print(f"Maximum difference in query gradients: {torch.max(torch.abs(grad_query - query.grad)).item():.6f}")
    print(f"Maximum difference in key gradients: {torch.max(torch.abs(grad_key - key.grad)).item():.6f}")
    print(f"Maximum difference in value gradients: {torch.max(torch.abs(grad_value - value.grad)).item():.6f}")
    
    print("\nTiming Results:")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")
    
    # Assert gradient correctness
    assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-3), "Query gradients don't match!"
    assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-3), "Key gradients don't match!"
    assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), "Value gradients don't match!"
    print("\nAll gradient checks passed!")
    
if __name__ == "__main__":
    
    test_attention_backward(100)