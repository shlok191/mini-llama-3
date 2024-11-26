import torch
from cuda_kernels import calculate_attention_scores  # your compiled CUDA module
import numpy as np

def test_attention_scores():
    # Test parameters
    sequence_length = 64
    embed_dim = 256
    torch.manual_seed(42)
    
    # Create random input tensors
    query = torch.randn(sequence_length, embed_dim, 
                       dtype=torch.float32, device='cuda')
    
    key = torch.randn(sequence_length, embed_dim, 
                     dtype=torch.float32, device='cuda')
    
    key_copy = key.clone()
    
    # PyTorch reference implementation
    with torch.no_grad():
        # Calculate QK^T
        qk = torch.matmul(query, key_copy.transpose(0, 1))
        # Scale
        qk = qk / np.sqrt(embed_dim)
        # Softmax
        attention_ref = torch.softmax(qk, dim=-1)

    print(key.shape)
    # Your CUDA implementation
    attention_cuda = calculate_attention_scores(
        query.contiguous(), key.contiguous())
    
    # Compare results
    max_diff = torch.max(torch.abs(attention_ref - attention_cuda)).item()
    is_close = torch.allclose(attention_ref, attention_cuda, 
                            rtol=1e-5, atol=1e-5)
    
    print(f"\nTest Results:")
    print(f"Max difference: {max_diff}")
    print(f"Tests passing: {is_close}")
    
    if not is_close:
        # Find worst disagreement
        diff = torch.abs(attention_ref - attention_cuda)
        max_idx = torch.argmax(diff)
        row = max_idx // sequence_length
        col = max_idx % sequence_length
        print("\nLargest disagreement:")
        print(f"Position: [{row}, {col}]")
        print(f"Reference: {attention_ref[row, col].item()}")
        print(f"CUDA: {attention_cuda[row, col].item()}")
        
        # Show small regions around largest difference
        r1, r2 = max(0, row-2), min(sequence_length, row+3)
        c1, c2 = max(0, col-2), min(sequence_length, col+3)
        print("\nReference region:")
        print(attention_ref[r1:r2, c1:c2])
        print("\nCUDA region:")
        print(attention_cuda[r1:r2, c1:c2])

if __name__ == "__main__":
    # Test different sequence lengths
    for seq_len in [64, 128]:
        print(f"\nTesting sequence length: {seq_len}")
        test_attention_scores()
        
    # Test numerical stability
    print("\nTesting with very small values...")
    query = torch.ones(32, 256, device='cuda') * 1e-5
    key = torch.ones(32, 256, device='cuda') * 1e-5
    
    print("\nTesting with very large values...")
    query = torch.ones(32, 256, device='cuda') * 1e5
    key = torch.ones(32, 256, device='cuda') * 1e5