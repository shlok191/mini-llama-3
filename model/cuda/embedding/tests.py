import torch
from mini_llama.cuda import embedding_forward, embedding_backward
import time

def test_embedding_forward():
    """
    Tests the raw CUDA embedding forward function by comparing its output
    to PyTorch's native implementation. This test verifies both correctness
    and performance of the embedding lookup operation.
    """
    print("\n=== Testing Embedding Forward CUDA Function ===")
    
    # Initialize test dimensions that are representative of real usage
    vocab_size = 1024    # Size of the vocabulary
    embed_dim = 256      # Dimension of each embedding vector
    seq_length = 512     # Length of the input sequence
    
    # Create test tensors with controlled data for easier debugging
    # The embedding table contains the vectors we'll look up
    table = torch.randn(vocab_size, embed_dim, 
                       dtype=torch.float32,
                       device='cuda')
    
    # Create indices that will exercise different parts of the embedding table
    # We include edge cases like first and last indices
    indices = torch.cat([
        torch.zeros(1, dtype=torch.int32),                     # First index
        torch.randint(1, vocab_size-1, (seq_length-2,),        # Random middle indices
                     dtype=torch.int32),
        torch.tensor([vocab_size-1], dtype=torch.int32)        # Last index
    ]).cuda()
    
    # Compute embeddings using our CUDA implementation
    print("Computing CUDA embeddings...")
    cuda_output = embedding_forward(indices, table)
    
    # Compute expected embeddings using PyTorch for comparison
    print("Computing PyTorch embeddings for comparison...")
    torch_embed = torch.nn.Embedding.from_pretrained(table)
    torch_output = torch_embed(indices)
    
    # Compare the results between implementations
    max_diff = torch.max(torch.abs(cuda_output - torch_output))
    print(f"\nMaximum difference in outputs: {max_diff:.6f}")
    
    # Check if results are close within acceptable tolerances
    is_close = torch.allclose(cuda_output, torch_output, rtol=1e-3, atol=1e-3)
    
    if is_close:
        print("✅ Embedding lookup matches PyTorch implementation")
    else:
        print("❌ Embedding mismatch detected")
        print("\nDetailed error analysis:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_output - torch_output)):.6f}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_output - torch_output)):.6f}")
        
        # If there's a mismatch, show specific examples to help debugging
        different_indices = torch.where(torch.abs(cuda_output - torch_output).sum(dim=1) > 1e-3)[0]
        if len(different_indices) > 0:
            idx = different_indices[0].item()
            token_id = indices[idx].item()
            print(f"\nExample mismatch for token_id {token_id} at sequence position {idx}:")
            print(f"CUDA output: {cuda_output[idx][:5]}")
            print(f"PyTorch output: {torch_output[idx][:5]}")
            print(f"Original embedding: {table[token_id][:5]}")
    
    # Verify specific properties that should hold
    print("\nVerifying embedding properties...")
    
    # Check that lookups match the original table
    for i in range(min(5, seq_length)):  # Check first few indices
        token_id = indices[i].item()
        assert torch.allclose(cuda_output[i], table[token_id], rtol=1e-3, atol=1e-3), \
            f"Mismatch in direct lookup for token {token_id}"
    print("✅ Direct lookup verification passed")
    
    # Performance testing with multiple runs for stable measurements
    print("\nPerformance testing...")
    num_runs = 100
    cuda_times = []
    torch_times = []
    
    # Time CUDA implementation
    print(f"Running CUDA implementation ({num_runs} times)...")
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = embedding_forward(indices, table)
        end.record()
        
        torch.cuda.synchronize()
        cuda_times.append(start.elapsed_time(end))
    
    # Time PyTorch implementation
    print(f"Running PyTorch implementation ({num_runs} times)...")
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
    
    print(f"\nTiming Results (averaged over {num_runs} runs):")
    print(f"CUDA Implementation: {cuda_mean:.3f} ms ± {cuda_std:.3f} ms")
    print(f"PyTorch Implementation: {torch_mean:.3f} ms ± {torch_std:.3f} ms")
    print(f"Speedup: {torch_mean/cuda_mean:.2f}x")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with sequence of all same indices
    same_indices = torch.ones(seq_length, dtype=torch.int32, device='cuda')
    
    try:
        same_output = embedding_forward(same_indices, table)
        print("✅ Passed: Handling sequence of identical indices")
    
    except Exception as e:
        print(f"❌ Failed: Error with identical indices: {str(e)}")
    
    # Test with very short sequence
    short_indices = torch.tensor([0], dtype=torch.int32, device='cuda')
    
    try:
        short_output = embedding_forward(short_indices, table)
        print("✅ Passed: Handling single-element sequence")
        
    except Exception as e:
        print(f"❌ Failed: Error with single-element sequence: {str(e)}")
    
    print("\n=== Embedding Forward Test Complete ===")
    
def test_embedding_backward():
    """
    Tests the raw CUDA embedding backward function by comparing its output 
    to PyTorch's native implementation. This test isolates just the backward
    pass computation to verify gradient calculation correctness.
    """
    print("\n=== Testing Embedding Backward CUDA Function ===")
    
    # Initialize test dimensions
    vocab_size = 1024   # Size of vocabulary
    embed_dim = 1024    # Embedding dimension
    seq_length = 512    # Length of input sequence
    
    # Create test tensors
    # grad_output represents gradients coming from the next layer
    grad_output = torch.randn(seq_length, embed_dim, 
                        dtype=torch.float32,
                        device='cuda')
    
    # indices represents the token IDs used in the forward pass
    indices = torch.randint(0, vocab_size, (seq_length,), 
                        dtype=torch.int32,
                        device='cuda')
    
    # table represents the embedding weights
    table = torch.randn(vocab_size, embed_dim, 
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
    # test_embedding_forward()
    test_embedding_backward()