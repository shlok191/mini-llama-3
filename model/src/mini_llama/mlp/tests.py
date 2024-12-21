import pytest
import torch
import torch.nn as nn
from mini_llama.mlp import MLP

@pytest.fixture
def mlp_config():
    """Provides standard configuration for testing"""
    return {
        'hidden_size': 1024,
        'intermediate_size': 2048,
        'activation_fn': nn.SiLU()
    }

def test_mlp_output_shape(mlp_config):
    """Tests if the MLP maintains the expected output shape"""
    # Arrange
    mlp = MLP(**mlp_config)
    seq_length = 512
    
    x = torch.randn(seq_length, mlp_config['hidden_size']).to("cuda:0")

    # Act
    with torch.no_grad():
        output = mlp(x)
    
    # Assert
    expected_shape = (seq_length, mlp_config['hidden_size'])
    
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {output.shape}"

def test_mlp_activation_function(mlp_config):
    """Tests if the activation function is being applied correctly"""
    
    # Use ReLU for easier testing of activation behavior    
    mlp_config['activation_fn'] = nn.ReLU()
    mlp = MLP(**mlp_config)
    
    # Create input with negative values
    x = torch.randn(512, mlp_config['hidden_size']).to("cuda:0")
    
    # Making all values negative
    x = -torch.abs(x)
    
    # Act
    with torch.no_grad():
        gate_output = mlp.activation_func(mlp.gate_proj(x))
    
    assert torch.all(gate_output >= 0), \
        "Activation function (ReLU) is not working as expected"

def test_mlp_gradient_flow(mlp_config):
    """Tests if gradients flow properly through all components"""
    
    # Arrange
    mlp = MLP(**mlp_config).to("cuda:0")
    x = torch.randn(512, mlp_config['hidden_size'], requires_grad=True).to("cuda:0")
    x.retain_grad()
    
    # Act
    output = mlp(x)
    loss = output.sum()
    loss.backward()
    
    # Assert
    assert x.grad is not None, "Input gradient is None"
    assert mlp.gate_proj.weights.grad is not None, "Gate projection gradient is None"
    assert mlp.up_proj.weights.grad is not None, "Up projection gradient is None"
    assert mlp.down_proj.weights.grad is not None, "Down projection gradient is None"
    assert not torch.isnan(x.grad).any(), "NaN gradients detected in input"

def test_mlp_zero_input(mlp_config):
    """Tests MLP behavior with zero input"""
    
    # Arrange
    mlp = MLP(**mlp_config)
    
    x = torch.zeros(512, mlp_config['hidden_size']).to("cuda:0")
    
    # Act
    with torch.no_grad():
        output = mlp(x)
    
    # Assert
    # Zero input should produce zero output due to multiplicative gating
    assert torch.allclose(output, torch.zeros_like(output)), \
        "Zero input did not produce zero output"
        

if __name__ == "__main__":
    pytest.main([__file__])