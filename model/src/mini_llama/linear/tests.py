import torch
import pytest
from torch import nn
from torch.testing import assert_close

# Import your implementation - adjust path as needed
from mini_llama.linear import Linear, FunctionalLinear

def test_initialization():
    """
    Test if weights are initialized correctly using different methods.
    We check basic statistical properties that should hold true for each initialization.
    """
    in_features, out_features = 512, 256
    
    # Test Kaiming He initialization (default)
    layer = Linear(in_features, out_features, init_method="kaiming-he")
    weights = layer.weights.data
    
    # The mean should be close to 0 and standard deviation should be reasonable
    assert abs(weights.mean().item()) < 0.1
    assert 0.01 < weights.std().item() < 1.0
    
    # Test invalid initialization - should raise an error
    with pytest.raises(AssertionError):
        Linear(in_features, out_features, init_method="invalid")

def test_forward_pass():
    """
    Test the forward pass of our layer by comparing it with PyTorch's nn.Linear.
    This ensures our implementation produces correct outputs.
    """
    # Set up dimensions
    seq_len, in_features, out_features = 128, 512, 256
    
    # Create our custom layer
    custom_layer = Linear(in_features, out_features, device="cuda")
    
    # Create equivalent PyTorch layer without bias
    torch_layer = nn.Linear(in_features, out_features, bias=False).cuda()
    # PyTorch uses transposed weights, so we need to align them
    torch_layer.weight.data = custom_layer.weights.T
    
    # Create input tensor
    x = torch.randn(seq_len, in_features, device="cuda")
    
    # Compare outputs
    custom_output = custom_layer(x)
    torch_output = torch_layer(x)
    
    # Check if outputs match within a small tolerance
    assert_close(custom_output, torch_output, rtol=1e-4, atol=1e-4)

def test_backward_pass():
    """
    Test the backward pass by checking if gradients are computed correctly
    and have the expected shapes.
    """
    seq_len, in_features, out_features = 128, 512, 256
    
    # Create layer and input
    layer = Linear(in_features, out_features, device="cuda")
    x = torch.randn(seq_len, in_features, device="cuda", requires_grad=True)
    
    # Forward pass
    output = layer(x)
    
    # Create a gradient to backpropagate
    # (in real use, this would come from the loss function)
    grad_output = torch.randn_like(output)
    
    # Backward pass
    output.backward(grad_output)
    
    # Check that gradients exist and have correct shapes
    assert x.grad is not None
    assert x.grad.shape == (seq_len, in_features)
    assert layer.weights.grad is not None
    assert layer.weights.shape == (in_features, out_features)
    
    # Check that gradients are finite (no NaN or inf values)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weights.grad).all()


if __name__ == "__main__":
    pytest.main([__file__])