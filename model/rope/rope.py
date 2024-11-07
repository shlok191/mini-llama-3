import torch
import torch.nn as nn

class RoPEmbedding(nn.Module):
    
    def __init__(self, dim: int):
        """
        Creating a custom, simplified Rotary-Positional (RoPE) embedding

        Args:
            dim (int): Dimensionality of the embeddings. Should be even.
        """
        
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE."
        self.dim = dim
        
        # Precomputing sinusoidal frequencies for the embeddings (assuming max seq length of 10,000)
        inv_frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Registering the inverse frequencies
        self.register_buffer('inv_frequencies', inv_frequencies)

    def forward(self, x: torch.Tensor):
        """
        Apply the RoPE embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        
        # Getting the appropriate dimensions
        _, seq_len, dim = x.shape
        
        assert dim == self.dim, "Input dimension must match embedding dimension."
        
        # Computing the positional embeddings
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        # Calculating the positional embeddings for each token (i) and frequency (j)
        sinusoidal_inp = torch.einsum("i,j->ij", positions, self.inv_frequencies)
        
        # Sin will be multiplied with odd places, and cosine with the even ones!
        sin, cos = sinusoidal_inp.sin(), sinusoidal_inp.cos()
        
        # Interleaving dimensions to apply RoPE
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated_X = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        return rotated_X