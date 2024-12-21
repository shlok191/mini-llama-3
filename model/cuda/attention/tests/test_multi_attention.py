import torch
from mini_llama.cuda import multi_attention_forward, multi_attention_backward
import time

def test_MHA_attention_backward(batch_size=4, num_runs=1):
    """
    Test and benchmark the backward pass of our MHA CUDA attention implementation
    against PyTorch's native implementation with batch support!
    
    Args:
        batch_size (int): Number of sequences in the batch
        num_runs (int): Number of runs for timing comparison
    """
    
    torch.manual_seed(42)
    
    # Set the dimensions
    sequence_length = 256
    embedding_dim_total = 1024
    num_heads = 4
    head_dim = embedding_dim_total // num_heads
    
    rtol = 1e-3
    atol = 1e-5
    
    # Create batched input tensors
    query = torch.randn(batch_size, sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True)
    
    key = torch.randn(batch_size, sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True)
    
    value = torch.randn(batch_size, sequence_length, embedding_dim_total,
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
        
        cuda_output, max_rows, sum_rows = multi_attention_forward(query, key, value)
        grad_output = torch.randn_like(cuda_output)

        grad_query, grad_key, grad_value = multi_attention_backward(
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
        
        torch_outputs = []
        
        # Process each batch
        for b in range(batch_size):
            torch_head_outputs = []
            
            for head in range(num_heads):
                # Fetching individual head outputs for current batch
                q_head = query[b, :, head * head_dim : (head + 1) * head_dim] 
                k_head = key[b, :, head * head_dim : (head + 1) * head_dim]
                v_head = value[b, :, head * head_dim : (head + 1) * head_dim]
                
                scores = torch.matmul(q_head, k_head.transpose(0, 1)) / (head_dim ** 0.5)
                attention_weights = torch.softmax(scores, dim=-1)
                head_output = torch.matmul(attention_weights, v_head)
                
                torch_head_outputs.append(head_output)
            
            # Concatenate head outputs for current batch
            batch_output = torch.cat(torch_head_outputs, dim=1)
            torch_outputs.append(batch_output)
        
        # Stack batch outputs
        torch_output = torch.stack(torch_outputs)
        
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
    
    print("\nTiming Results:")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")
    
    # Assert gradient correctness
    assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-3), "Query gradients don't match!"
    assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-3), "Key gradients don't match!"
    assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), "Value gradients don't match!"
    print("\nAll gradient checks passed!")

def test_MHA_attention_implementation(batch_size=4):
    """
    Test the forward pass of our MHA CUDA attention implementation
    against PyTorch's native implementation with batch support!
    
    Args:
        batch_size (int): Number of sequences in the batch
    """
    
    torch.manual_seed(42)
    
    sequence_length = 256
    embedding_dim_total = 1024
    num_heads = 4
    head_dim = embedding_dim_total // num_heads
    
    rtol = 1e-3
    atol = 1e-5
    
    # Create batched input tensors
    query = torch.randn(batch_size, sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')
    
    key = torch.randn(batch_size, sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')
    
    value = torch.randn(batch_size, sequence_length, embedding_dim_total,
        dtype=torch.float32,
        device='cuda')

    # Time CUDA implementation
    start_time = time.perf_counter()
    cuda_output, _, _ = multi_attention_forward(query, key, value)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start_time
        
    # PyTorch implementation
    start_time = time.perf_counter()
    pytorch_outputs = []
    
    # Process each batch
    for b in range(batch_size):
        head_outputs = []
        
        for head in range(num_heads):
            # Get current head's queries, keys, values for current batch
            q_head = query[b, :, head * head_dim:(head + 1) * head_dim]
            k_head = key[b, :, head * head_dim:(head + 1) * head_dim]
            v_head = value[b, :, head * head_dim:(head + 1) * head_dim]

            scores = torch.matmul(q_head, k_head.transpose(0, 1))
            scores = scores / (head_dim ** 0.5)
            
            attention_weights = torch.softmax(scores, dim=-1)
            head_output = torch.matmul(attention_weights, v_head)
                
            head_outputs.append(head_output)
        
        # Concatenate head outputs for current batch
        batch_output = torch.cat(head_outputs, dim=1)
        pytorch_outputs.append(batch_output)
    
    # Stack batch outputs
    pytorch_output = torch.stack(pytorch_outputs)
    pytorch_time = time.perf_counter() - start_time
    
    print(f"\nComparing results across batches and heads:")
    
    all_matches = True
    
    # Compare each batch and head
    for b in range(batch_size):
        print(f"\nBatch {b}:")
        for head in range(num_heads):
            start_idx = head * head_dim
            end_idx = (head + 1) * head_dim
            
            cuda_head = cuda_output[b, :, start_idx:end_idx]
            pytorch_head = pytorch_output[b, :, start_idx:end_idx]
                    
            max_diff = torch.max(torch.abs(cuda_head - pytorch_head))
            is_close = torch.allclose(cuda_head, pytorch_head, rtol=rtol, atol=atol)
            
            if is_close:
                print(f"✅ Head {head} matches!")
            else:
                all_matches = False
                print(f"❌ Head {head} differs! Maximum difference: {max_diff}")
            
    # Overall comparison
    print("\nOverall comparison:")
    print(f"CUDA output shape: {cuda_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    final_max_diff = torch.max(torch.abs(cuda_output - pytorch_output))
    print(f"Maximum difference between implementations: {final_max_diff}")
    
    final_is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)
    
    if final_is_close and all_matches:
        print("✅ Test passed! CUDA implementation matches PyTorch")
    else:
        print("❌ Test failed! Implementations produce different results")

    print(f"\nCUDA implementation time: {cuda_time * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
    
if __name__ == "__main__":
    BATCH_SIZE = 4
    test_MHA_attention_implementation(BATCH_SIZE)
    #test_MHA_attention_backward(BATCH_SIZE)