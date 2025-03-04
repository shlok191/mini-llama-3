import torch
import torch.nn as nn
import math

class RoPEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        """
        Initialize Rotary Position Embeddings (RoPE) for single-batch inputs.
        
        This implementation assumes inputs will always have shape 
        (batch_size, num_heads, seq_len, dim),
        
        Args:
            dim (int): Embedding dimension (must be even)
            base (int): Base for the frequency calculations (default: 10000 as in original paper)
        """
        super().__init__()
        assert dim % 2 == 0, f"Embedding dimension must be even for RoPE, got {dim}"
        self.dim = dim
        
        # Create frequency bands using geometric sequence
        # Each pair of dimensions shares the same frequency
        exponents = torch.arange(0, dim, 2).float() / dim
        inv_frequencies = 1.0 / (base ** exponents)
        
        # Register the frequencies as a buffer (persistent but not a parameter)
        self.register_buffer('inv_frequencies', inv_frequencies)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim)
        
        Returns:
            torch.Tensor: Transformed tensor with same shape as input (batch_size, num_heads, seq_len, dim)
        """
        
        batch_size, num_heads, seq_len, dim = x.shape
        assert dim == self.dim, f"Input dimension {dim} must match embedding dimension {self.dim}"
        
        # Creating position indices for the sequence
        positions = torch.arange(seq_len, device=x.device).float()
        
        # Computing rotation angles for each position and frequency
        # Shape: (seq_len, dim//2)
        angles = positions.unsqueeze(1) * self.inv_frequencies.to(x.device)
        
        # Compute sin and cos for the rotation
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(1)  # (batch_size, num_heads, seq_len, dim//2)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(1)  # (batch_size, num_heads, seq_len, dim//2)
        
        # Split input into even and odd dimensions
        x_even = x[..., ::2]  # (batch_dim, num_heads, seq_len, dim//2)
        x_odd = x[..., 1::2]  # (batch_dim, num_heads, seq_len, dim//2)
        
        # Apply rotation using the rotation matrix:
        # [cos θ, -sin θ]
        # [sin θ,  cos θ]
        rotated = torch.empty_like(x)
        
        rotated[..., ::2] = x_even * cos - x_odd * sin    # Real part
        rotated[..., 1::2] = x_even * sin + x_odd * cos   # Imaginary part
        
        return rotated