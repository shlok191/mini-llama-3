import torch
import torch.nn as nn
from mini_llama.linear import Linear
from torch.autograd import Function
from mini_llama.rope import RoPEmbedding
from mini_llama.cuda import multi_attention_forward, multi_attention_backward

from typing import List

class FunctionalAttention(Function):
    
    @staticmethod
    def forward(ctx, query, key, value, curr_seq_lens):
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
        output, max_rows, sum_rows = multi_attention_forward(query, key, value, curr_seq_lens)
        
        # Storing the necessary values for backpropogation
        ctx.save_for_backward(query, key, value, output, max_rows, sum_rows)
        ctx.curr_seq_lens = curr_seq_lens
        
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
        curr_seq_lens = ctx.curr_seq_lens
        
        # Once again using the custom CUDA implementation which from my tests is ~ 3x faster than PyTorch implementation :)    
        grad_query, grad_key, grad_value = multi_attention_backward(query, key, value, output, grad_output, max_rows, sum_rows, curr_seq_lens)
        
        return grad_query, grad_key, grad_value, None
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, dropout: float,
            hidden_size: int,
            num_heads: int,
            rope_dim: int):
        
        """Initializes the MHAttention module

        Args:
            dropout (float): The percentage of X values to zero out. Must be in range [0, 1]
            hidden_size (int): The out dimension for the Q, K, and V matrices
            num_heads (int): The number of query heads for each attention layer
            rope_dim (int): The number of dimensions to apply rotary positional embeddings to
        """

        super().__init__()

        # Storing the defined variables into the object
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rope_dim = rope_dim
        
        # Defining head numbers
        self.head_dim = hidden_size // num_heads
        
        # Defining the linear layers
        self.q_proj = Linear(self.hidden_size, self.hidden_size).to("cuda")
        self.k_proj = Linear(self.hidden_size, self.hidden_size).to("cuda")
        self.v_proj = Linear(self.hidden_size, self.hidden_size).to("cuda")
        self.o_proj = Linear(self.hidden_size, self.hidden_size).to("cuda")
    
        # Defining rotary embedding and dropout layer
        self.rope = RoPEmbedding(dim=self.rope_dim)
        self.dropout = nn.Dropout(dropout)

        # Storing the Functional implementation
        self.attention_fn = FunctionalAttention.apply
        
    def forward(self, X: torch.Tensor, curr_seq_lens: List[int]):
    
        assert not torch.isnan(X).any()
        
        # Applying dropout to the given input value
        X = self.dropout(X)
        
        # Calculating Q, K, V projections
        query = self.q_proj(X)
        key = self.k_proj(X)
        value = self.v_proj(X)
        
        # Updating shapes to assist with RoPE application
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        embed_dim = self.head_dim * self.num_heads
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
        # Applying positional embeddings to Q and K via rope
        query = self.rope.forward(query)
        key = self.rope.forward(key)

        # Converting back to original shape for the attention implementation
        query = query.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        key = key.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        value = value.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
        # Calling our in-house attention implementation!
        attn_output = self.attention_fn(query, key, value, curr_seq_lens)
        final_output = self.o_proj(attn_output)
        
        # Merging the heads along the embedding dimension!
        return final_output