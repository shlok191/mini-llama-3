import torch
from mini_llama.cuda import multi_attention_forward, multi_attention_backward
import time
import statistics

def test_MHA_attention_implementation(num_runs=1):
    """
    Test the forward pass of our MHA CUDA attention implementation
    against PyTorch's native implementation with batch support.
    
    Args:
        num_runs (int): Number of times to run the test
    """
    
    # Presentation matters :)
    print("=" * 80)
    print(f"Testing CUDA attention implementation for multi headed attention processing...")
    print("=" * 80)
    
    # Setting the dimensions
    batch_size = 64
    sequence_length = 320
    embedding_dim_total = 1024
    num_heads=4
    head_dim = embedding_dim_total // num_heads
    
    rtol = 1e-3
    atol = 1e-5
    
    print("\nConfiguration:")
    print(f"Sequence Length: {sequence_length}")
    print(f"Total Embedding Dimension: {embedding_dim_total}")
    print(f"Number of attention heads: {num_heads}")
    
    cuda_times = []
    pytorch_times = []
    
    for run in range(num_runs):
        
        print(f"\n{'-' * 80}")
        print(f"Beginning run {run}...")
        print(f"{'-' * 80}")
        
        # Randomly deciding padding sequence lengths (starts at 75-100% of max length)
        curr_seq_lens = torch.randint(
            int(0.75 * sequence_length),
            sequence_length + 1,
            (batch_size,),
            device='cuda',
            dtype=torch.int32
        )
        
        curr_seq_lens = curr_seq_lens.tolist()
        
        # Creating batched input tensors
        query = torch.randn(batch_size, sequence_length, embedding_dim_total,
            dtype=torch.float32,
            device='cuda')
        
        key = torch.randn(batch_size, sequence_length, embedding_dim_total,
            dtype=torch.float32,
            device='cuda')
        
        value = torch.randn(batch_size, sequence_length, embedding_dim_total,
            dtype=torch.float32,
            device='cuda')

        # Zeroing out embeddings for padded positions
        for b in range(batch_size):
            query[b, curr_seq_lens[b]:] = 0
            key[b, curr_seq_lens[b]:] = 0
            value[b, curr_seq_lens[b]:] = 0
            
        # Time CUDA implementation
        start_time = time.perf_counter()
        cuda_output, _, _ = multi_attention_forward(query, key, value, curr_seq_lens)
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start_time
        
        cuda_times.append(cuda_time)
        
        # Timing the PyTorch implementation
        start_time = time.perf_counter()
        torch_batch_outputs = []
    
        # Processing each of our batches
        for b in range(batch_size):
            
            head_outputs = []
            seq_len = curr_seq_lens[b] 
            
            for head in range(num_heads):
                
                # Extracting the tensors for this attention head
                q_head = query[b, :, head * head_dim:(head + 1) * head_dim]
                k_head = key[b, :, head * head_dim:(head + 1) * head_dim]
                v_head = value[b, :, head * head_dim:(head + 1) * head_dim]
                
                # Calculating the raw attention scores
                scores = torch.matmul(q_head, k_head.transpose(0, 1)) / (head_dim ** 0.5)
                
                # Creating an attention mask for this sequence
                attention_mask = torch.zeros((sequence_length, sequence_length), device='cuda', dtype=torch.bool)
                
                # Adding causal masking + padding masking
                attention_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
                attention_mask[:, seq_len:] = True
                
                attention_mask = attention_mask.to("cuda")
                
                scores = scores.masked_fill(attention_mask, -1e10)
                
                # Calculating the softmax
                attention_weights = torch.softmax(scores, dim=-1)
                
                # Apply attention weights to values
                head_output = torch.matmul(attention_weights, v_head)
                head_outputs.append(head_output)
            
            # Concatenate all head outputs for this batch
            batch_output = torch.cat(head_outputs, dim=-1)
            torch_batch_outputs.append(batch_output)
        
        pytorch_output = torch.stack(torch_batch_outputs, dim=0)
        pytorch_time = time.perf_counter() - start_time

        pytorch_times.append(pytorch_time)
        
        print(f"\nComparing each batch and head independently (batch_size={batch_size}):")
        
        all_matches = True
        
        for b in range(batch_size):

            print(f"\nBatch {b}:")
            
            for head in range(num_heads):
                start_idx = head * head_dim
                end_idx = (head + 1) * head_dim
                
                cuda_head = cuda_output[b, :, start_idx:end_idx]
                pytorch_head = pytorch_output[b, :, start_idx:end_idx]
                
                is_close = torch.allclose(cuda_head, pytorch_head, rtol=rtol, atol=atol)
                
                if is_close:
                    print(f"✓ Head {head} matches!")
                    
                else:
                    all_matches = False
                    max_diff = torch.max(torch.abs(cuda_head - pytorch_head))
                    mean_diff = torch.mean(torch.abs(cuda_head - pytorch_head))
                    std_diff = torch.std(torch.abs(cuda_head - pytorch_head))
                    
                    print(f"✗ Head {head} differs!")
                    print(f"  Maximum difference: {max_diff}")
                    print(f"  Mean absolute error: {mean_diff}")
                    print(f"  Standard deviation of error: {std_diff}\n")
                
        # Printing out statistics
        print("\nOverall comparison:")
        print(f"CUDA output shape: {cuda_output.shape}")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        
        final_max_diff = torch.max(torch.abs(cuda_output - pytorch_output))
        print(f"Maximum difference between implementations: {final_max_diff}")
        
        final_is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)
        
        if final_is_close and all_matches:
            print("✓ Test passed! CUDA implementation matches PyTorch")
        else:
            print("✗ Test failed! Implementations produce different results")

        # Printing out time differences
        print("\nTiming Results:")
        print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")
        print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
        print(f"Speedup: {pytorch_time/cuda_time:.2f}x")

        print(f"{'-' * 80}")
        
    # Giving out final time speedup values!
    cuda_time_mean = statistics.mean(cuda_times)    
    pytorch_time_mean = statistics.mean(pytorch_times)
    
    print(f"\n{'=' * 80}")
    print("Final Timing Results:")
    print(f"{'-' * 80}")
    
    print(f"CUDA implementation time: {cuda_time_mean * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time_mean * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time/cuda_time:.2f}x")
    
def test_MHA_attention_backward(num_runs=100):
    
    # Presentation matters :)
    print("=" * 80)
    print(f"Testing CUDA backward attention implementation for multi headed attention processing...")
    print("=" * 80)
    
     # Setting the dimensions
    batch_size = 64
    sequence_length = 320
    embedding_dim_total = 1024
    num_heads=4
    head_dim = embedding_dim_total // num_heads
    
    rtol = 5e-2
    atol = 5e-2
    
    print("\nConfiguration:")
    print(f"Sequence Length: {sequence_length}")
    print(f"Total Embedding Dimension: {embedding_dim_total}")
    print(f"Number of attention heads: {num_heads}")
    
    cuda_times = []
    pytorch_times = []
    
    cuda_times = []
    pytorch_times = []
    
    print(f"\nBenchmarking CUDA implementation ({num_runs} runs)...")
           
    for run in range(num_runs):
        
        print(f"\n{'-' * 80}")
        print(f"Beginning run {run}...")
        print(f"{'-' * 80}")
        
        # Randomly deciding padding sequence lengths (starts at 75-100% of max length)
        curr_seq_lens = torch.randint(
            int(0.75 * sequence_length),
            sequence_length + 1,
            (batch_size,),
            device='cuda',
            dtype=torch.int32
        )
        
        curr_seq_lens = curr_seq_lens.tolist()
        
        # Create input tensors with batch dimension
        query = torch.randn(batch_size, sequence_length, embedding_dim_total, 
            device='cuda', requires_grad=False)
        
        key = torch.randn(batch_size, sequence_length, embedding_dim_total, 
            device='cuda', requires_grad=False)
        
        value = torch.randn(batch_size, sequence_length, embedding_dim_total, 
            device='cuda', requires_grad=False)
        

        for b in range(batch_size):
            
            # Create mask of shape [sequence_length]
            mask = torch.arange(sequence_length, device='cuda') < curr_seq_lens[b]
            
            # Expand mask to match embedding dimension: [sequence_length, embedding_dim]
            mask = mask.unsqueeze(-1).expand(-1, embedding_dim_total)
            
            # Apply mask through multiplication instead of in-place assignment
            query[b] = query[b] * mask
            key[b] = key[b] * mask
            value[b] = value[b] * mask
        
        query = query.clone().detach().requires_grad_(True)
        key = key.clone().detach().requires_grad_(True)
        value = value.clone().detach().requires_grad_(True)
        
        # Timing the CUDA performance
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Calculating the forward pass
        cuda_output, max_rows, sum_rows = multi_attention_forward(query.clone(), key.clone(), value.clone(), curr_seq_lens)
        grad_output = torch.randn_like(cuda_output)
        
        # Calculating the backward pass
        grad_query, grad_key, grad_value = multi_attention_backward(
            query, key, value, cuda_output, grad_output, max_rows, sum_rows, curr_seq_lens
        )
        
        end_time.record()
        torch.cuda.synchronize()
        
        # Storing the times for statistics!
        cuda_time = start_time.elapsed_time(end_time)
        cuda_times.append(cuda_time)
        
        # Timing PyTorch implementation
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        torch_batch_outputs = []
        start_time.record()
        
        # Processing each of our batches
        for b in range(batch_size):
            
            head_outputs = []
            seq_len = curr_seq_lens[b] 
            
            for head in range(num_heads):
                
                # Extracting the tensors for this attention head
                q_head = query[b, :, head * head_dim:(head + 1) * head_dim]
                k_head = key[b, :, head * head_dim:(head + 1) * head_dim]
                v_head = value[b, :, head * head_dim:(head + 1) * head_dim]
                
                # Calculating the raw attention scores
                scores = torch.matmul(q_head, k_head.transpose(0, 1)) / (head_dim ** 0.5)
                
                # Creating an attention mask for this sequence
                attention_mask = torch.zeros((sequence_length, sequence_length), device='cuda', dtype=torch.bool)
                
                # Adding causal masking + padding masking
                attention_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
                attention_mask[:, seq_len:] = True
                
                attention_mask = attention_mask.to("cuda")
                
                scores = scores.masked_fill(attention_mask, -1e10)
                
                # Calculating the softmax
                attention_weights = torch.softmax(scores, dim=-1)
                
                # Apply attention weights to values
                head_output = torch.matmul(attention_weights, v_head)
                head_outputs.append(head_output)
            
            # Concatenate all head outputs for this batch
            batch_output = torch.cat(head_outputs, dim=-1)
            torch_batch_outputs.append(batch_output)
        
        pytorch_output = torch.stack(torch_batch_outputs, dim=0)
        pytorch_output.backward(grad_output)
        
        end_time.record()
        torch.cuda.synchronize()
        
        pytorch_time = start_time.elapsed_time(end_time)
        pytorch_times.append(pytorch_time)
        
        print(f"\nComparing each batch and head independently (batch_size={batch_size}):")
        
        for b in range(batch_size):

            print(f"Batch {b}: \n")
            
            # Printing out important metrics
            print("\nGradient Comparison:")
            
            print(f"Maximum difference in query gradients: {torch.max(torch.abs(grad_query - query.grad)).item():.6f}")
            print(f"Maximum difference in key gradients: {torch.max(torch.abs(grad_key - key.grad)).item():.6f}")
            print(f"Maximum difference in value gradients: {torch.max(torch.abs(grad_value - value.grad)).item():.6f}")
        
            assert torch.allclose(grad_query, query.grad, rtol=rtol, atol=atol), "Query gradients don't match!"
            assert torch.allclose(grad_key, key.grad, rtol=rtol, atol=atol), "Key gradients don't match!"
            assert torch.allclose(grad_value, value.grad, rtol=rtol, atol=atol), "Value gradients don't match!"

            print("✓ Test passed! CUDA implementation matches PyTorch")
        # Printing out statistics
        print("\nOverall comparison:")
        print(f"CUDA output shape: {cuda_output.shape}")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        
        # Printing out time differences
        print("\nTiming Results:")
        print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")
        print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")
        print(f"Speedup: {pytorch_time/cuda_time:.2f}x")

        print(f"{'-' * 80}")
        
    # Giving out final time speedup values!
    cuda_time_mean = statistics.mean(cuda_times)    
    pytorch_time_mean = statistics.mean(pytorch_times)
    
    print(f"\n{'=' * 80}")
    print("Final Timing Results:")
    print(f"{'-' * 80}")
    
    print(f"CUDA implementation time: {cuda_time_mean * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time_mean * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time/cuda_time:.2f}x")
    
if __name__ == "__main__":
    
    test_MHA_attention_backward(10)