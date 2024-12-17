import torch
import pytest
import math
from torch.testing import assert_close

# Import your implementation
from rope import RoPEmbedding  # Adjust import path as needed

def test_rope_initialization():
    """
    Test that RoPE embedding initializes correctly with various dimensions
    and properly enforces constraints.
    """
    # Should initialize successfully with even dimensions
    rope = RoPEmbedding(dim=4)
    assert rope.dim == 4
    assert rope.inv_frequencies.shape == (2,)  # dim//2 frequencies
    
    # Should raise assertion error for odd dimensions
    with pytest.raises(AssertionError):
        RoPEmbedding(dim=3)
    
    # Test frequency calculation
    rope = RoPEmbedding(dim=4, base=10000)
    expected_freqs = 1.0 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
    assert_close(rope.inv_frequencies, expected_freqs)

def test_rope_forward_shape():
    """
    Test that the forward pass maintains input tensor shapes.
    """
    dim = 6
    seq_len = 8
    rope = RoPEmbedding(dim=dim)
    
    # Test with various input shapes
    x = torch.randn(seq_len, dim)
    output = rope(x)
    assert output.shape == (seq_len, dim)
    
    # Should raise assertion error if dim doesn't match
    wrong_dim = torch.randn(seq_len, dim + 2)
    with pytest.raises(AssertionError):
        rope(wrong_dim)

def test_rope_rotation_properties():
    """
    Test fundamental properties of the rotary embeddings:
    1. Rotation should be position-dependent
    2. Same position should get same rotation
    3. Different positions should get different rotations
    """
    dim = 4
    rope = RoPEmbedding(dim=dim)
    
    # Create simple input where each position is the same
    x = torch.ones(3, dim)  # 3 positions, all ones
    output = rope(x)
    
    # Different positions should get different rotations
    assert not torch.allclose(output[0], output[1])
    assert not torch.allclose(output[1], output[2])
    
    # Same input at same position should get same rotation
    x_repeated = torch.ones(2, dim)
    output_1 = rope(x_repeated)
    output_2 = rope(x_repeated)
    assert_close(output_1, output_2)

def test_rope_complex_rotation():
    """
    Test that RoPE correctly implements complex number rotation properties.
    We test this by checking if consecutive pairs of dimensions behave like
    complex numbers under rotation.
    """
    dim = 4
    rope = RoPEmbedding(dim=dim, base=10000)
    
    # Create input where each pair can be interpreted as a complex number
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # Two complex numbers: 1+0i each
    output = rope(x)
    
    # For each complex pair, magnitude should be preserved
    for i in range(0, dim, 2):
        original_magnitude = torch.sqrt(x[0,i]**2 + x[0,i+1]**2)
        rotated_magnitude = torch.sqrt(output[0,i]**2 + output[0,i+1]**2)
        assert_close(original_magnitude, rotated_magnitude)

def test_rope_numerical_stability():
    """
    Test RoPE's numerical stability with very long sequences
    and extreme values.
    """
    dim = 4
    rope = RoPEmbedding(dim=dim)
    
    # Test with long sequence
    long_seq = torch.randn(10000, dim)
    output_long = rope(long_seq)
    assert torch.isfinite(output_long).all()  # No NaN or inf values
    
    # Test with very large values
    large_values = torch.ones(10, dim) * 1e6
    output_large = rope(large_values)
    assert torch.isfinite(output_large).all()
    
    # Test with very small values
    small_values = torch.ones(10, dim) * 1e-6
    output_small = rope(small_values)
    assert torch.isfinite(output_small).all()

def test_rope_device_compatibility():
    """
    Test that RoPE works correctly across different devices
    when available.
    """
    dim = 4
    rope = RoPEmbedding(dim=dim)
    x = torch.randn(5, dim)
    
    # Test on CPU
    output_cpu = rope(x)
    assert output_cpu.device.type == 'cpu'
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        rope_cuda = rope.cuda()
        x_cuda = x.cuda()
        output_cuda = rope_cuda(x_cuda)
        assert output_cuda.device.type == 'cuda'
        # Results should match across devices
        assert_close(output_cuda.cpu(), output_cpu)

def test_rope_gradients():
    """
    Test that gradients flow correctly through the RoPE layer.
    """
    dim = 4
    rope = RoPEmbedding(dim=dim)
    x = torch.randn(5, dim, requires_grad=True)
    
    # Forward pass
    output = rope(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Gradient checks
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()  # No NaN or inf in gradients
    assert x.grad.shape == x.shape

if __name__ == "__main__":
    pytest.main([__file__])