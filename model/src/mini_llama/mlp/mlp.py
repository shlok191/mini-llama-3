import torch
import torch.nn as nn
from mini_llama.linear import Linear

class MLP(nn.Module):
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation_fn: nn.Module):
        """Defines a custom MLP layer from the Llama paper

        Args:
            hidden_size (int: The hidden size of the logits
            intermediate_size (int): The output dimensions of the final gate
            activation_fn (nn.Module): The activation function to use
        """
        super().__init__()

        # Storing the given variables
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Utilizing the custom linear layers :)
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size).to("cuda:0")
        self.up_proj = Linear(self.hidden_size, self.intermediate_size).to("cuda:0")
        self.down_proj = Linear(self.intermediate_size, self.hidden_size).to("cuda:0")
        
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
        assert not torch.isnan(down_projection).any()
        
        return down_projection