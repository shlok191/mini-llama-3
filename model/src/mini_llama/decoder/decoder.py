import torch
import torch.nn as nn
from mini_llama.attention import MultiHeadedAttention
from mini_llama.rmsnorm import RMSNorm
from mini_llama.mlp import MLP
from typing import List

class DecoderLayer(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rope_dim: int,
        dropout: float = 0.1,
        activation_fn: nn.Module = nn.SiLU()):
        
        """Initializes a Llama decoder layer
        
        Args:
            hidden_size (int): The dimension of the model's hidden states
            num_heads (int): Number of attention heads
            intermediate_size (int): Size of the MLP's intermediate layer
            rope_dim (int): Dimension for rotary position embeddings
            dropout (float, optional): Dropout probability. Defaults to 0.1
            activation_fn (nn.Module, optional): Activation function for MLP. Defaults to SiLU
        """
        
        super().__init__()
        
        # Defining our custom CUDA Multi Headed Attention block
        self.attention = MultiHeadedAttention(
            dropout=dropout,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_dim=rope_dim
        )
        
        self.attention_norm = RMSNorm(dim=hidden_size)
        
        # Defining the post attention MLP layer
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_fn=activation_fn
        )
        
        # Defining layer normalization
        self.input_layer_norm = RMSNorm(dim=hidden_size)
        self.post_attn_norm = RMSNorm(dim=hidden_size)
        
    def forward(self, X: torch.Tensor, curr_seq_lens: List[int]) -> torch.Tensor:
        """Forward pass of the decoder layer
        
        Args:
            X (torch.Tensor): Input tensor of shape (seq_len, hidden_size)
            curr_seq_lens (List[int]): The length of the non-padded tokens for the batch
            
        Returns:
            torch.Tensor: Output tensor of shape (seq_len, hidden_size)
        """
        
        # Self-attention block with residual connection
        norm_X = self.input_layer_norm(X).to("cuda:0")
        
        X = X + self.attention(norm_X, curr_seq_lens)
        
        # Having the output past through the feed forward layer
        norm_X = self.post_attn_norm(X).to("cuda:0")
        X = X + self.mlp(norm_X).to("cuda:0")
        
        return X