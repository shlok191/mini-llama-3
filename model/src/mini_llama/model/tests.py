import pytest
import torch
import torch.nn as nn
from mini_llama.model import MiniLlamaForCausalLM

def model_config():
    """Provides a standard test configuration for the model"""
    
    return {
        'vocab_size': 128,        
        'embedding_dim': 1024,      
        'num_decoder_layers': 4,  
        'num_attn_heads': 4,
        'mlp_layer_intermediate_dim': 2048,
        'dropout': 0.0,     
        'padding_idx': 0
    }

def test_model_forward_shape():
    """Tests if the model maintains expected shapes during forward pass"""
    
    # Arrange
    config = model_config()
    model = MiniLlamaForCausalLM(**config)
    
    seq_length = 256
    input_ids = torch.randint(0, config['vocab_size'], (seq_length,))
    
    # Act
    with torch.no_grad():
        logits = model(input_ids, labels=None)
    
    # Assert
    expected_shape = (seq_length, config['vocab_size'])
    
    assert logits.shape == expected_shape, \
        f"Expected logits shape {expected_shape}, got {logits.shape}"

def test_model_loss_backward():
    """Tests if gradients flow properly through the model during backward pass"""
    # Arrange
    
    config = model_config()
    model = MiniLlamaForCausalLM(**config)
    
    seq_length = 256
    
    input_ids = torch.randint(0, config['vocab_size'], (seq_length,))
    labels = torch.randint(0, config['vocab_size'], (seq_length,))

    # Storing initial parameters for comparison
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Act
    loss = model(input_ids, labels)
    loss.backward()
    
    # Assert
    # Check if gradients exist and aren't zero
    for name, param in model.named_parameters():
        
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
            f"Zero gradient for {name}"
        
    # Check if parameters changed after backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    
    for name, param in model.named_parameters():
        assert not torch.allclose(param, initial_params[name]), \
            f"Parameter {name} didn't update after optimization step"

def test_loss_computation():
    """Tests if the loss computation works correctly and gives reasonable values"""
    
    # Arrange
    config = model_config()
    model = MiniLlamaForCausalLM(**config)
    
    seq_length = 256
    input_ids = torch.randint(0, config['vocab_size'], (seq_length,))
    
    # Create labels that match input_ids perfectly (should give low loss)
    labels = input_ids.clone()
    
    # Act
    perfect_match_loss = model(input_ids, labels)
    
    # Create random labels (should give high loss)
    random_labels = torch.randint(0, config['vocab_size'], (seq_length,))
    random_match_loss = model(input_ids, random_labels)
    
    # Assert
    assert perfect_match_loss < random_match_loss, \
        "Loss for perfect prediction should be lower than random prediction"
    
if __name__ == "__main__":
    pytest.main([__file__])