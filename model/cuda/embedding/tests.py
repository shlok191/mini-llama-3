import torch
from mini_llama.cuda import embedding_forward, embedding_backward
import time

def test_embedding_forward():
    """
    Tests the CUDA embedding forward function with batch support by comparing 
    its output to PyTorch's native implementation. This test verifies both 
    correctness and performance of batched embedding lookups.
    """
    print("\n=== Testing Batched Embedding Forward CUDA Function ===")
    
    # Initialize test dimensions that represent real usage scenarios
    batch_size = 32      # Number of sequences in each batch
    vocab_size = 1024    # Size of the vocabulary
    embed_dim = 256      # Dimension of each embedding vector
    seq_length = 512     # Length of each sequence in the batch
    
    # Create embedding table that will be shared across the batch
    table = torch.randn(vocab_size, embed_dim, 
                       dtype=torch.float32,
                       device='cuda')
    
    # Create batched indices tensor, including edge cases in each batch
    indices = torch.zeros(batch_size, seq_length, dtype=torch.int32, device='cuda')
    for b in range(batch_size):
        # First token is always 0 (edge case)
        indices[b, 0] = 0
        # Middle tokens are random
        indices[b, 1:-1] = torch.randint(1, vocab_size-1, (seq_length-2,), dtype=torch.int32)
        # Last token is vocab_size-1 (edge case)
        indices[b, -1] = vocab_size-1
    
    print(f"Input shapes:")
    print(f"Indices: {indices.shape} (batch_size × sequence_length)")
    print(f"Embedding table: {table.shape} (vocab_size × embedding_dim)")
    
    # Compute embeddings using our CUDA implementation
    print("\nComputing CUDA embeddings...")
    cuda_output = embedding_forward(indices, table)
    print(f"CUDA output shape: {cuda_output.shape}")
    
    # Compute expected embeddings using PyTorch
    print("Computing PyTorch embeddings for comparison...")
    torch_embed = torch.nn.Embedding.from_pretrained(table)
    torch_output = torch_embed(indices)
    print(f"PyTorch output shape: {torch_output.shape}")
    
    # Compare results between implementations
    max_diff = torch.max(torch.abs(cuda_output - torch_output))
    print(f"\nMaximum difference in outputs: {max_diff:.6f}")
    
    # Check if results match within tolerances
    is_close = torch.allclose(cuda_output, torch_output, rtol=1e-3, atol=1e-3)
    
    if is_close:
        print("✅ Batched embedding lookup matches PyTorch implementation")
    else:
        print("❌ Batched embedding mismatch detected")
        print("\nDetailed error analysis:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_output - torch_output)):.6f}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_output - torch_output)):.6f}")
        
        # Show specific examples from different batches if there's a mismatch
        different_positions = torch.where(torch.abs(cuda_output - torch_output).sum(dim=2) > 1e-3)
        if len(different_positions[0]) > 0:
            batch_idx = different_positions[0][0].item()
            seq_idx = different_positions[1][0].item()
            token_id = indices[batch_idx, seq_idx].item()
            print(f"\nExample mismatch in batch {batch_idx}, position {seq_idx}, token_id {token_id}:")
            print(f"CUDA output: {cuda_output[batch_idx, seq_idx, :5]}")
            print(f"PyTorch output: {torch_output[batch_idx, seq_idx, :5]}")
            print(f"Original embedding: {table[token_id, :5]}")
    
    # Verify embedding properties across batches
    print("\nVerifying embedding properties...")
    for b in range(min(3, batch_size)):  # Check first few batches
        for i in range(min(3, seq_length)):  # Check first few positions
            token_id = indices[b, i].item()
            assert torch.allclose(cuda_output[b, i], table[token_id], rtol=1e-3, atol=1e-3), \
                f"Mismatch in batch {b}, position {i} for token {token_id}"
    print("✅ Direct lookup verification passed for sampled batch positions")
    
    # Performance testing
    print("\nPerformance testing...")
    num_runs = 100
    cuda_times = []
    torch_times = []
    
    # Time CUDA implementation with batching
    print(f"Running batched CUDA implementation ({num_runs} times)...")
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = embedding_forward(indices, table)
        end.record()
        
        torch.cuda.synchronize()
        cuda_times.append(start.elapsed_time(end))
    
    # Time PyTorch implementation with batching
    print(f"Running batched PyTorch implementation ({num_runs} times)...")
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = torch_embed(indices)
        end.record()
        
        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))
    
    # Calculate and display performance statistics
    cuda_mean = sum(cuda_times) / len(cuda_times)
    cuda_std = (sum((x - cuda_mean) ** 2 for x in cuda_times) / len(cuda_times)) ** 0.5
    
    torch_mean = sum(torch_times) / len(torch_times)
    torch_std = (sum((x - torch_mean) ** 2 for x in torch_times) / len(torch_times)) ** 0.5
    
    print(f"\nBatched Timing Results (averaged over {num_runs} runs):")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")
    
    # Test batch-specific edge cases
    print("\nTesting batch-specific edge cases...")
    
    # Test with batch where all sequences are identical
    same_indices = torch.ones(batch_size, seq_length, dtype=torch.int32, device='cuda')
    try:
        same_output = embedding_forward(same_indices, table)
        print("✅ Passed: Handling batch of identical sequences")
    except Exception as e:
        print(f"❌ Failed: Error with batch of identical sequences: {str(e)}")
    
    # Test with minimal batch size and sequence length
    tiny_indices = torch.zeros((1, 1), dtype=torch.int32, device='cuda')
    try:
        tiny_output = embedding_forward(tiny_indices, table)
        print("✅ Passed: Handling minimal batch and sequence size")
    except Exception as e:
        print(f"❌ Failed: Error with minimal batch and sequence: {str(e)}")
    
    print("\n=== Batched Embedding Forward Test Complete ===")
    
def test_embedding_backward():
    """
    Tests the raw CUDA embedding backward function by comparing its output 
    to PyTorch's native implementation. This test isolates just the backward
    pass computation to verify gradient calculation correctness.
    """
    print("\n=== Testing Embedding Backward CUDA Function ===")
    
    # Initialize test dimensions that represent real usage scenarios
    batch_size = 32      # Number of sequences in each batch
    vocab_size = 1024    # Size of the vocabulary
    embed_dim = 256      # Dimension of each embedding vector
    seq_length = 512     # Length of each sequence in the batch
    
    # Create embedding table that will be shared across the batch
    table = torch.randn(vocab_size, embed_dim, 
                       dtype=torch.float32,
                       device='cuda')
    
    # Create batched indices tensor, including edge cases in each batch
    indices = torch.zeros(batch_size, seq_length, dtype=torch.int32, device='cuda')
    
    for b in range(batch_size):
        # First token is always 0 (edge case)
        indices[b, 0] = 0
        # Middle tokens are random
        indices[b, 1:-1] = torch.randint(1, vocab_size-1, (seq_length-2,), dtype=torch.int32)
        # Last token is vocab_size-1 (edge case)
        indices[b, -1] = vocab_size-1
        
    # Create test tensors
    # grad_output represents gradients coming from the next layer
    grad_output = torch.randn(batch_size, seq_length, embed_dim, 
                        dtype=torch.float32,
                        device='cuda')
    
    # Setting the padding index 0 to 0
    table[0, :] = torch.zeros(embed_dim, device='cuda', dtype=torch.float32)
    
    # Get gradients using our CUDA implementation
    print("Computing CUDA gradients...")
    cuda_grad_table = embedding_backward(grad_output, indices, table, 0)
    
    torch.cuda.synchronize()
    
    # Compute expected gradients using PyTorch
    print("Computing PyTorch gradients for comparison...")
    
    # Create a PyTorch embedding layer with same weights
    torch_embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0).cuda()
    torch_embed.weight.data.copy_(table)
    
    # Do forward and backward passes
    output = torch_embed(indices)
    output.backward(grad_output)
    
    torch_grad_table = torch_embed.weight.grad
    
    # Compare results
    max_diff = torch.max(torch.abs(cuda_grad_table - torch_grad_table))
    print(f"\nMaximum difference in gradients: {max_diff:.6f}")
    
    # Check if results are close enough
    is_close = torch.allclose(cuda_grad_table, torch_grad_table, rtol=1e-3, atol=1e-3)
    
    if is_close:
        print("✅ Gradient computation matches PyTorch implementation")
    else:
        print("❌ Gradient mismatch detected")
        print("\nDetailed error analysis:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_grad_table - torch_grad_table)):.6f}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_grad_table - torch_grad_table)):.6f}")
        
        # Show example of mismatched gradients
        different_indices = torch.where(torch.abs(cuda_grad_table - torch_grad_table) > 1e-3)[0]
        if len(different_indices) > 0:
            idx = different_indices[0].item()
            print(f"\nExample mismatch at embedding index {idx}:")
            print(f"CUDA gradients: {cuda_grad_table[idx][:5]}")
            print(f"PyTorch gradients: {torch_grad_table[idx][:5]}")
    
    # Performance test
    print("\nPerformance testing...")
    num_runs = 100
    cuda_times = []
    torch_times = []
    
    # Time CUDA implementation
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = embedding_backward(grad_output, indices, table, 0)
        end.record()
        
        torch.cuda.synchronize()
        cuda_times.append(start.elapsed_time(end))
    
    # Time PyTorch implementation
    torch_embed.weight.grad = None  # Reset gradients
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = torch_embed(indices)
        output.backward(grad_output)
        end.record()
        
        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))
        torch_embed.weight.grad = None  # Reset for next iteration
    
    # Calculate statistics
    cuda_mean = sum(cuda_times) / len(cuda_times)
    cuda_std = (sum((x - cuda_mean) ** 2 for x in cuda_times) / len(cuda_times)) ** 0.5
    
    torch_mean = sum(torch_times) / len(torch_times)
    torch_std = (sum((x - torch_mean) ** 2 for x in torch_times) / len(torch_times)) ** 0.5
    
    print(f"\nTiming Results (averaged over {num_runs} runs):")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")

if __name__ == "__main__":
    #test_embedding_forward()
    test_embedding_backward()