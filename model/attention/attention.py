import torch
import torch.nn as nn
from ..linear.linear import Linear
from torch.autograd import Function
from ..rope.rope import RoPEmbedding
from mini_llama import attention_forward, attention_backward

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
        output, max_rows, sum_rows = attention_forward(query, key, value)
        
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
        grad_query, grad_key, grad_value = attention_backward(query, key, value, output, grad_output, max_rows, sum_rows)
        
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

        # Reshaping Q, K, V to num_heads subtensors splitting along embedding dimensions!
        query = query.view(-1, self.num_heads, self.hidden_size)
        key = key.view(-1, self.num_heads, self.hidden_size)
        value = value.view(-1, self.num_heads, self.hidden_size)
        
        # Applying positional embeddings to Q and K via rope :)
        query = self.rotary_emb.forward(query)
        key = self.rotary_emb.forward(key)

        # Calling our in-house attention implementation :)
        attn_output = self.attention_fn(query, key, value)
        attn_output = self.o_proj(attn_output)
        
        # Merging the heads along the embedding dimension!
        return attn_output
    