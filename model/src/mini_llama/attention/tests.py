import pytest
import torch
from mini_llama.attention import MultiHeadedAttention

@pytest.fixture
def mha_config():
    """
    Provides standard configuration parameters for MultiHeadedAttention.
    These values can be reused across multiple tests.
    """
    return {
        'seq_length': 256,
        'hidden_size': 1024,
        'num_heads': 4,
        'rope_dim': 1024,
        'dropout_rate': 0.1
    }

@pytest.fixture
def mha_module(mha_config):
    """
    Creates a MultiHeadedAttention module with standard configuration.
    The module is moved to CUDA and returned for testing.
    """
    module = MultiHeadedAttention(
        dropout=mha_config['dropout_rate'],
        hidden_size=mha_config['hidden_size'],
        num_heads=mha_config['num_heads'],
        rope_dim=mha_config['rope_dim']
    ).cuda()
    return module

@pytest.fixture
def input_tensor(mha_config):
    """
    Creates a standard input tensor for testing.
    The tensor is moved to CUDA and has gradients enabled.
    """
    return torch.randn(
        mha_config['seq_length'], 
        mha_config['hidden_size'],
        requires_grad=True,
        device="cuda:0"
    )

def test_initialization(mha_module, mha_config):
    """
    Tests if the MultiHeadedAttention module is initialized correctly with
    the expected parameters and computed values.
    """
    assert mha_module.hidden_size == mha_config['hidden_size'], \
        "Hidden size not properly initialized"
    
    assert mha_module.num_heads == mha_config['num_heads'], \
        "Number of heads not properly initialized"
    
    expected_head_dim = mha_config['hidden_size'] // mha_config['num_heads']
    assert mha_module.head_dim == expected_head_dim, \
        "Head dimension calculation incorrect"

def test_output_shape(mha_module, input_tensor, mha_config):
    """
    Verifies that the module produces output tensors of the correct shape,
    maintaining the input dimensions through the forward pass.
    """
    output = mha_module(input_tensor)
    expected_shape = (mha_config['seq_length'], mha_config['hidden_size'])
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} doesn't match expected {expected_shape}"

def test_dropout_behavior(mha_module, input_tensor):
    """
    Tests if dropout is properly applied during training and disabled during evaluation.
    We expect different outputs in training mode due to dropout.
    """
    # Test in eval mode (dropout disabled)
    mha_module.eval()
    output_no_dropout = mha_module(input_tensor)
    
    # Test in training mode (dropout enabled)
    mha_module.train()
    output_with_dropout = mha_module(input_tensor)
    
    # Outputs should differ due to dropout in training mode
    assert not torch.allclose(output_no_dropout, output_with_dropout, atol=1e-3), \
        "Dropout not affecting output in training mode"

def test_max_sequence_length(mha_module, mha_config):
    """
    Verifies that the module can handle inputs up to the maximum specified sequence length.
    """
    max_input = torch.randn(
        mha_config['seq_length'], 
        mha_config['hidden_size'],
        device="cuda:0"
    )
    max_output = mha_module(max_input)
    
    expected_shape = (mha_config['seq_length'], mha_config['hidden_size'])
    assert max_output.shape == expected_shape, \
        "Failed to handle maximum sequence length"

def test_gradient_flow(mha_module, input_tensor):
    """
    Checks if gradients properly flow through the module during backpropagation.
    This ensures the module is properly connected in the computational graph.
    """
    output = mha_module(input_tensor)
    loss = output.sum()
    loss.backward()
    
    # Check gradients on input
    assert input_tensor.grad is not None, \
        "Gradients not flowing to input"
    
    # Check gradients on projection layers
    assert mha_module.q_proj.weights.grad is not None, \
        "Gradients not flowing to query projection"
    assert mha_module.k_proj.weights.grad is not None, \
        "Gradients not flowing to key projection"
    assert mha_module.v_proj.weights.grad is not None, \
        "Gradients not flowing to value projection"
        
if __name__ == "__main__":
    pytest.main([__file__])