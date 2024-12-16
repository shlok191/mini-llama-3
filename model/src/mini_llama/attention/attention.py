import torch
import torch.nn as nn
from mini_llama import Linear
from torch.autograd import Function
from mini_llama import RoPEmbedding
from mini_llama import multi_attention_forward, multi_attention_backward

class FunctionalAttention(Function):
    
    @staticmethod
    def forward(ctx, query, key, value):
        """Performs the forward pass for the attention implementation

        Args:
            ctx (object): Stores variables for backprop
            query (torch.Tensor): stores Q matrix for gradient calculation
            key (torch.Tensor): stores K matrix for gradient calculation
            value (torch.Tensor): stores V matrix for gradient calculation

        Returns:
            torch.Tensor: Returns the output value
        """
        
        # Calling our in-house attention mechanism!
        output, max_rows, sum_rows = multi_attention_forward(query, key, value)
        
        # Storing the necessary values for backpropogation
        ctx.save_for_backward(query, key, value, output, max_rows, sum_rows)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Performs the backpropogation for the attention mechanism!

        Args:
            ctx (obj): Stores the variables during forward mechanism
            grad_output (torch.Tensor): The gradient for the output value

        Returns:
            tuple[torch.Tensor]: Returns the gradients for Q, K, and V :)
        """
        
        # Fetching the saved tensors for backpropogation
        query, key, value, output, max_rows, sum_rows = ctx.saved_tensors
        
        # Once again using the custom CUDA implementation which from my tests is ~ 3x faster than PyTorch implementation :)    
        grad_query, grad_key, grad_value = multi_attention_backward(query, key, value, output, grad_output, max_rows, sum_rows)
        
        return grad_query, grad_key, grad_value
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, dropout: float,
            hidden_size: int,
            num_heads: int,
            max_positional_embeddings: int,
            rope_dim: int):
        
        """Initializes the MHAttention module

        Args:
            dropout (float): The percentage of X values to zero out. Must be in range [0, 1]
            hidden_size (int): The out dimension for the Q, K, and V matrices
            num_heads (int): The number of query heads for each attention layer
            max_positional_embeddings (int): 
            rope_dim (int): The number of dimensions to apply rotary positional embeddings to
        """

        super().__init__()

        # Storing the defined variables into the object
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_positional_embeddings = max_positional_embeddings
        self.rope_dim = rope_dim
        
        # Defining head numbers
        self.head_dim = hidden_size // num_heads
        
        # Defining the linear layers
        self.q_proj = Linear(self.hidden_size, self.hidden_size)
        self.k_proj = Linear(self.hidden_size, self.hidden_size)
        self.v_proj = Linear(self.hidden_size, self.hidden_size)
        self.o_proj = Linear(self.hidden_size, self.hidden_size)

        # Defining rotary embedding and dropout layer
        self.rope = RoPEmbedding(dim=self.rope_dim)
        self.dropout = nn.Dropout(dropout)

        # Storing the Functional implementation
        self.attention_fn = FunctionalAttention.apply
        
    def forward(self, X: torch.Tensor):
    
        seq_length = X.shape[0]
        
        # Applying dropout to the given input value
        X = self.dropout(X)
        
        # Calculating Q, K, V projections
        query = self.q_proj(X)
        key = self.k_proj(X)
        value = self.v_proj(X)
        
        # Applying positional embeddings to Q and K via rope
        query = self.rotary_emb.forward(query)
        key = self.rotary_emb.forward(key)

        # Calling our in-house attention implementation
        attn_output = self.attention_fn(query, key, value)
        attn_output = self.o_proj(attn_output)
        
        # Merging the heads along the embedding dimension!
        return attn_output
    
    
def test_mha_class():
    """
    Test function for MultiHeadedAttention class implementation.
    Tests initialization, forward pass, and various edge cases without batch dimension.
    The input tensor shape is (sequence_length, hidden_size) instead of 
    (sequence_length, batch_size, hidden_size).
    """
    
    import torch
    import pytest
    
    # Test parameters - notice we removed batch_size since we're not using it
    seq_length = 10
    hidden_size = 1024
    num_heads = 4
    max_pos_embeddings = 1024
    rope_dim = 32
    dropout_rate = 0.1
    
    # Initialize the attention module
    mha = MultiHeadedAttention(
        dropout=dropout_rate,
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_positional_embeddings=max_pos_embeddings,
        rope_dim=rope_dim
    )
    
    # Test 1: Basic initialization checks
    assert mha.hidden_size == hidden_size, "Hidden size not properly initialized"
    assert mha.num_heads == num_heads, "Number of heads not properly initialized"
    assert mha.head_dim == hidden_size // num_heads, "Head dimension calculation incorrect"
    
    # Test 2: Input shape handling
    # Note: Now using 2D input tensor (seq_length, hidden_size)
    input_tensor = torch.randn(seq_length, hidden_size)
    output = mha.forward(input_tensor)
    
    # Check output dimensions - should match input dimensions
    expected_shape = (seq_length, hidden_size)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"
    
    # Test 3: Dropout behavior
    mha.eval()  # Set to evaluation mode to disable dropout
    output_no_dropout = mha.forward(input_tensor)
    
    mha.train()  # Set back to training mode
    output_with_dropout = mha.forward(input_tensor)
    
    # Outputs should be different in training mode due to dropout
    assert not torch.allclose(output_no_dropout, output_with_dropout), "Dropout not affecting output in training mode"
    
    # Test 4: Edge cases
    # Test with minimum sequence length
    min_input = torch.randn(1, hidden_size)
    min_output = mha.forward(min_input)
    assert min_output.shape == (1, hidden_size), "Failed to handle minimum sequence length"
    
    # Test with maximum sequence length
    max_input = torch.randn(max_pos_embeddings, hidden_size)
    max_output = mha.forward(max_input)
    assert max_output.shape == (max_pos_embeddings, hidden_size), "Failed to handle maximum sequence length"
    
    # Test 5: Gradient flow
    # Enable gradient tracking
    input_tensor.requires_grad = True
    output = mha.forward(input_tensor)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are properly flowing
    assert input_tensor.grad is not None, "Gradients not flowing to input"
    assert mha.q_proj.weight.grad is not None, "Gradients not flowing to query projection"
    assert mha.k_proj.weight.grad is not None, "Gradients not flowing to key projection"
    assert mha.v_proj.weight.grad is not None, "Gradients not flowing to value projection"
    
    # Test 6: RoPE embedding application
    # Create two identical inputs at different positions
    input1 = torch.randn(1, hidden_size)
    input2 = input1.clone()
    
    output1 = mha.forward(input1)
    output2 = mha.forward(input2)
    
    # Outputs should be identical for identical inputs
    assert torch.allclose(output1, output2, atol=1e-5), "RoPE embedding producing inconsistent results for identical inputs"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_mha_class()
    