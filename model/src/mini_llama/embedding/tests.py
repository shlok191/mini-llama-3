import torch
import torch.nn as nn
from mini_llama.embedding import Embedding

def test_embedding():
    """
    Comprehensive test suite for the Embedding layer that combines multiple test cases
    into a single function. Tests initialization, forward pass, padding, gradients,
    and edge cases.
    """
    # Configuration setup
    config = {
        'vocab_size': 1024,
        'embed_dims': 1024,
        'padding_token_index': 0,
        'init_method': "xavier"
    }
    
    # Extract common values
    vocab_size = config['vocab_size']
    embed_dims = config['embed_dims']
    padding_idx = config['padding_token_index']
    
    print("\n=== Testing Embedding Layer ===")
    
    # Test 1: Initialization with different methods
    print("\nTesting initialization...")
    for init_method in ["xavier", "normal", "kaiming-he"]:
        embed = Embedding(
            vocab_size=vocab_size,
            embed_dims=embed_dims,
            padding_token_index=padding_idx,
            init_method=init_method
        ).cuda()
        
        # Check non-padding embeddings are not all zeros
        non_padding_embeddings = embed.embedding_table[1:]
        assert not torch.allclose(non_padding_embeddings, torch.zeros_like(non_padding_embeddings).cuda()), \
            f"Error: {init_method} initialization resulted in all zeros"
            
        # Check padding token is zero
        padding_embedding = embed.embedding_table[padding_idx]
        assert torch.allclose(padding_embedding, torch.zeros(embed_dims).cuda()), \
            f"Error: Padding token not zero with {init_method} initialization"
        print(f"✓ {init_method} initialization passed")
    
    # Test 2: Forward pass shape
    print("\nTesting forward pass shapes...")
    embed = Embedding(**config).cuda()
    seq_length = 256
    indices = torch.randint(0, vocab_size, (seq_length,), dtype=torch.int32).cuda()
    
    with torch.no_grad():
        output = embed(indices)
        assert output.shape == (seq_length, embed_dims), \
            f"Error: Expected shape {(seq_length, embed_dims)}, got {output.shape}"
    print("✓ Forward pass shapes passed")
    
    # Test 3: Embedding values correctness
    print("\nTesting embedding lookup values...")
    result = embed(indices)
    expected = embed.embedding_table[indices]
    assert torch.allclose(result, expected), "Error: Embedding lookup produced incorrect values"
    print("✓ Embedding lookup values passed")
    
    # Test 4: Padding token handling
    print("\nTesting padding token behavior...")
    padding_indices = torch.ones(seq_length, dtype=torch.int32).cuda() * padding_idx
    
    with torch.no_grad():
        padding_output = embed(padding_indices)
    
    assert torch.allclose(padding_output, torch.zeros_like(padding_output)), \
        "Error: Padding tokens produced non-zero embeddings"
    
    print("✓ Padding token handling passed")
    
    # Test 5: Gradient flow
    print("\nTesting gradient flow...")
    indices = torch.randint(0, vocab_size, (seq_length,), dtype=torch.int32).cuda()
    initial_params = embed.embedding_table.clone()

    embed.embedding_table.retain_grad()
    
    # Forward and backward pass
    output = embed(indices)
    loss = output.sum()
    loss.backward()
    
    assert embed.embedding_table.grad is not None, "Error: No gradients computed"
    assert not torch.allclose(embed.embedding_table.grad, torch.zeros_like(embed.embedding_table.grad)), \
        "Error: Gradients are all zeros"
    
    # Test parameter update
    optimizer = torch.optim.Adam([embed.embedding_table], lr=0.01)
    optimizer.step()
    assert not torch.allclose(embed.embedding_table, initial_params), \
        "Error: Parameters didn't update after optimization"
    print("✓ Gradient flow passed")
    
    print("\n=== All tests passed successfully! ===")

if __name__ == "__main__":
    test_embedding()