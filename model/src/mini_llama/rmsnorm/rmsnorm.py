import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        """
        Args:
            dim (int): The dimension of the input feature.
            eps (float): A small value to avoid division by zero.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim)).to("cuda:0")  # Learnable scaling parameter

    def forward(self, x):
        """
        Forward pass of RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute the RMS of the input
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.scale * (x / rms)
