import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim (int): The dimension of the input feature.
            eps (float): A small value to avoid division by zero.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, requires_grad=True), requires_grad=True).to("cuda:0")  # Learnable scaling parameter

    def forward(self, x):
        """
        Forward pass of RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        
        x_clipped = torch.clamp(x, -1e6, 1e6)  # Prevent extreme values
        
        # Compute the RMS of the input
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.scale * (x / rms)
