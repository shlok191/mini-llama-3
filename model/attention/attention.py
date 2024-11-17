import torch
import torch.nn as nn
from torch.autograd import Function
from ..linear.linear import Linear
from ..rope.rope import RoPE

class FunctionalSDPAttention(Function):
   
    def __init__(self):
        pass

class SDPAttention(nn.Module):
   
    def __init__(self, dropout: float,
                 hidden_size: int,
                 num_heads: int,
                 key_value_heads: int,
                 max_positional_embeddings: int,
                 rope_dim: int,
                 layer_id: int,
                 bias: bool):
        
        super().__init__()

        # Storing the defined variables into the object
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_value_heads = key_value_heads
        self.max_positional_embeddings = max_positional_embeddings
        self.layer_id = layer_id
        self.bias = bias
        self.rope_dim = rope_dim
        
        # Defining the derivative values
        self.head_dim = hidden_size // num_heads
        self.key_value_groups = self.num_heads // self.key_value_heads
        
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.bias)
        self.k_proj = Linear(self.hidden_size, self.key_value_heads * self.head_dim, bias=self.bias)
        self.v_proj = Linear(self.hidden_size, self.key_value_heads * self.head_dim, bias=self.bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.bias)

        self.rotary_emb = RoPE(dim=self.rope_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    
        # Project hidden states to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(-1, self.num_heads, hidden_states.size(1), self.head_dim)
        key = key.view(-1, self.key_value_heads, hidden_states.size(1), self.head_dim)
        value = value.view(-1, self.key_value_heads, hidden_states.size(1), self.head_dim)
        
        # Apply RoPE to queries and keys
        query = self.rotary_emb.forward(query)
        key = self.rotary_emb.forward(key)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # Broadcast to match (batch, heads, seq, seq)
    
        # Call attention mechanism (functional or built-in)
        attn_output, attn_weights = scaled_dot_product_attention(
            query, key, value, mask=attention_mask, dropout=self.dropout
        )
        
        # Merge heads and project back
        attn_output = attn_output.view(-1, hidden_states.size(1), self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
    