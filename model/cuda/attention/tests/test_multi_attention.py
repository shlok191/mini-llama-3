import torch
import mini_llama
import math
import time

def test_MHA_attention_backward(num_runs=1):
    """
    Test and benchmark the backward pass of our MHA CUDA attention implementation
    against PyTorch's native implementation!
    
    Args:
        num_runs (int): Number of runs for timing comparison
    """
    
    # Setting the seed to help with debugging
    torch.manual_seed(42)
    
    # Set the dimensions
    sequence_length = 256
    embedding_dim_total = 1024
    num_heads = 4
    
    # Compare the results per head
    rtol = 1e-3
    atol = 1e-5
    
    head_dim = embedding_dim_total // num_heads
    
    # Create initial random input tensors on the GPU with requires_grad
    query = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True)
    
    key = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True)
    
    value = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True)
    
    # Timing CUDA implementation
    cuda_times = []
    print(f"\nBenchmarking CUDA implementation ({num_runs} runs)...")
    
    for i in range(num_runs):
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        cuda_output, max_rows, sum_rows = mini_llama.cuda.multi_attention_forward(query, key, value)
        grad_output = torch.randn_like(cuda_output)

        # CUDA multi-head backward pass
        grad_query, grad_key, grad_value = mini_llama.cuda.multi_attention_backward(
            query, key, value, cuda_output, grad_output, max_rows, sum_rows
        )
        
        end_time.record()
        
        torch.cuda.synchronize()
        cuda_times.append(start_time.elapsed_time(end_time))
    
    # Timing PyTorch implementation
    torch_times = []
    print(f"\nBenchmarking PyTorch implementation ({num_runs} runs)...")
    
    for i in range(num_runs):
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        torch_head_outputs = []
        
        for head in range(num_heads):
            
            # Fetching individual head outputs
            q_head = query[:, head * head_dim : (head + 1) * head_dim] 
            k_head = key[:, head * head_dim : (head + 1) * head_dim]
            v_head = value[:, head * head_dim : (head + 1) * head_dim]
            
            # Calculating the scores
            scores = torch.matmul(q_head, k_head.transpose(0, 1)) / (head_dim ** 0.5)
            
            # Calculating the softmax
            attention_weights = torch.softmax(scores, dim=-1)
            
            # Finally, multiplying with the value head
            head_output = torch.matmul(attention_weights, v_head)
            
            torch_head_outputs.append(head_output)
        
        # Concatenate head outputs
        torch_output = torch.cat(torch_head_outputs, dim=1)
        
        # Backward pass
        torch_output.backward(grad_output)
        end_time.record()
        
        torch.cuda.synchronize()
        torch_times.append(start_time.elapsed_time(end_time))
        
        # Reset gradients
        if(i is not num_runs - 1):
            query.grad = None
            key.grad = None
            value.grad = None
    
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
    assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-3), "Key gradients don't match!"
    assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), "Value gradients don't match!"
    print("\nAll gradient checks passed!")
    
def test_MHA_attention_implementation():
    
    # Setting the seed to help with debugging
    torch.manual_seed(42)
    
    # Set the dimensions
    sequence_length = 256
    embedding_dim_total = 1024
    num_heads = 4
    
    # Compare the results per head
    rtol = 1e-3
    atol = 1e-5
    
    head_dim = embedding_dim_total // num_heads
    
    # Create initial random input tensors on the GPU
    query = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')
    
    key = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')
    
    value = torch.randn(sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')

    # Timing the CUDA operation!
    start_time = time.perf_counter()
    cuda_output, _, _ = mini_llama.cuda.multi_attention_forward(query, key, value)
    
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start_time
        
    # PyTorch implementation with proper multi-head handling
    start_time = time.perf_counter()

    # Process each head independently
    head_outputs = []
    
    for head in range(num_heads):
        
        # Get current head's queries, keys, values (create fresh copies for safety)
        q_head = query[:, (head) * 256:(head + 1) * 256]
        k_head = key[:, (head) * 256:(head + 1) * 256]
        v_head = value[:, (head) * 256:(head + 1) * 256]

        scores = torch.matmul(q_head, k_head.transpose(0, 1))
        scores = scores / (256 ** 0.5)
        
        attention_weights = torch.softmax(scores, dim=-1)
        pytorch_output = torch.matmul(attention_weights, v_head)
            
        head_outputs.append(pytorch_output)
    
    # Concatenate head outputs
    pytorch_output = torch.cat(head_outputs, dim=1)
    
    pytorch_time = time.perf_counter() - start_time
    
    print(f"\nComparing each head independently:")
    
    all_heads_match = True
    
    for head in range(num_heads):
        
        # Extract each head's portion
        start_idx = head * head_dim
        end_idx = (head + 1) * head_dim
        
        # Create copies of the segments we're comparing
        cuda_head = cuda_output[:, start_idx:end_idx]
        pytorch_head = pytorch_output[:, start_idx:end_idx]
                
        print(f"Head {head} shapes - CUDA: {cuda_head.shape}, PyTorch: {pytorch_head.shape}")
        
        # Compare statistics for this head
        max_diff = torch.max(torch.abs(cuda_head - pytorch_head))
        mean_diff = torch.mean(torch.abs(cuda_head - pytorch_head))
        std_diff = torch.std(torch.abs(cuda_head - pytorch_head))
        is_close = torch.allclose(cuda_head, pytorch_head, rtol=rtol, atol=atol)
        
        if is_close:
            print(f"✅ Head {head} matches!")
        
        else:
            all_heads_match = False
            print(f"❌ Head {head} differs!")
            print(f"  Maximum difference: {max_diff}")
            print(f"  Mean absolute error: {mean_diff}")
            print(f"  Standard deviation of error: {std_diff}")
            
    # Overall comparison
    print("\nOverall comparison:")
    print(f"CUDA output shape: {cuda_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    final_max_diff = torch.max(torch.abs(cuda_output - pytorch_output))
    print(f"Maximum difference between implementations: {final_max_diff}")
    
    final_is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)
    
    if final_is_close and all_heads_match:
        print("✅ Test passed! CUDA implementation matches PyTorch")
    
    else:
        print("❌ Test failed! Implementations produce different results")


    print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
    print(f"\nSpeedup: {pytorch_time / cuda_time:.2f}x")
    
if __name__ == "__main__":
    
    test_MHA_attention_implementation() 
    test_MHA_attention_backward()
    