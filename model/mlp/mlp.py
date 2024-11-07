import torch
import torch.nn as nn
from model import Linear

class MLPLayer(nn.Module):
    
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool, activation_fn: nn.Module):
        """Defines a custom MLP layer from the Llama paper

        Args:
            hidden_size (int: The hidden size of the logits
            intermediate_size (int): The output dimensions of the final gate
            bias (bool): Whether or not to have bias values for the gates
            activation_fn (nn.Module): The activation function to use
        """
        super().__init__()

        # Storing the given variables
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.activation_fn = activation_fn
        
        # Utilizing the custom linear layers :)
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = Linear(self.hidden_size, self.intermediate_size, bias=bias)
        
        # The activation function (could be ReLU, LeakyReLU, orrrr SwiGLU?)
        self.activation_func = activation_fn
        
    def forward(self, X: torch.Tensor):
        """Passes the logits through the MLP layer

        Args:
            X (torch.Tensor): The input values

        Returns:
            torch.Tensor: The output values
        """
        
        # Processing X through the gate layer and the up layer
        gate_projection = self.activation_func(self.gate_proj(X))
        up_projection = self.up_proj(X)
        
        # Processing the output values through the down projection
        down_projection = self.down_proj(gate_projection * up_projection)
        
        return down_projection