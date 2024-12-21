import pytest
import torch
import torch.nn as nn
from mini_llama.decoder import DecoderLayer

@pytest.fixture
def decoder_config():
    """Provides a standard configuration for testing"""

    return {
        'hidden_size': 1024,
        'num_heads': 4,
        'intermediate_size': 2048,
        'rope_dim': 1024,
        'dropout': 0.1,
        'activation_fn': nn.SiLU()
    }

def test_decoder_output_shape(decoder_config):
    """Tests if the decoder maintains the expected output shape"""
    # Arrange

    decoder = DecoderLayer(**decoder_config)
    seq_length = 512

    x = torch.randn(seq_length, decoder_config['hidden_size'], requires_grad=True).to("cuda:0")
    
    with torch.no_grad():
        output = decoder(x)
    
    # Assert
    assert output.shape == (seq_length, decoder_config['hidden_size']), \
        f"Expected shape {(seq_length, decoder_config['hidden_size'])}, got {output.shape}"

def test_decoder_gradient_flow(decoder_config):
    """Tests if gradients can flow through the decoder properly"""
    
    # Arrange
    decoder = DecoderLayer(**decoder_config)
    seq_length = 512
    
    x = torch.randn(seq_length, decoder_config['hidden_size'], requires_grad=True).to("cuda:0")
    x.retain_grad()
    
    # Act
    output = decoder(x)
    loss = output.sum()
    loss.backward()
    
    # Assert
    assert x.grad is not None, "Gradient did not flow back to input"
    assert all(p.grad is not None for p in decoder.parameters()), \
        "Some decoder parameters did not receive gradients"
    assert not torch.isnan(x.grad).any(), "NaN gradients detected"

if __name__ == "__main__":
    pytest.main([__file__])