import torch
import torch.nn as nn
from torch.autograd import Function
from cuda_kernels import linear_forward, linear_backward

class FunctionalLinear(Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weights: torch.Tensor):
        
        """Generates the output for a given set of weights and inputs, also stores the inputs and weights for backpropogation

        Args:
            ctx: The context for storing variables
            x (torch.Tensor): The input torch tensor
            weights (torch.Tensor): The weights
            
        Returns:
            torch.tensor: The output of the matrix multiplication
        """
        
        # Saving the tensors needed for backward pass
        ctx.save_for_backward(x, weights)
        
        # Calculating the forward value :)
        return linear_forward(x, weights)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        
        """Calculates the gradient w.r.t weights and inputs

        args:
            ctx: The context for accessing stored variables
            
            grad_output (torch.Tensor): The gradient of the output of this layer -- calculated by the next 
            layer or the loss func. if this is the final layer :)
            
        Returns:
            torch.Tensor, torch.Tensor: Returns the gradient w.r.t to the inputs and the weights respectively
        """
        
        # Retrieving the weights and the inputs of the layer
        x, weights = ctx.saved_tensors
        
        # Getting the gradient for the inputs and the weights
        # One is passed backwards, and one updates the weights!
        grad_input, grad_weights = linear_backward(grad_output, x, weights)
        
        # Return gradients for each input in same order as forward
        return grad_input, grad_weights

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 init_method: str = "kaiming-he",
                 device: str = "cuda"):
        
        """Custom Linear layer implementation with CUDA backend.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            init_method (str): Weight initialization method ("xavier", "normal", "kaiming-he")
            device (str): Device to store the layer ("cuda" or "cpu")
        """
        
        super().__init__()
        
        # Storing the dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Initializing weights according to the strategy specified
        self.weights = nn.Parameter(
            torch.empty(in_features, out_features, 
                       dtype=torch.float32,
                       device=device)
        )
        
        self._initialize_layer(init_method)

    def _initialize_layer(self, method: str):
        """Initializes the layer weights by custom strategies from popular research papers
        
        Args:
            method (str): Initialization method to use for this particular linear layer
        """
        
        valid_methods = ["xavier", "normal", "kaiming-he"]

        # Ensuring a valid initialization method is specified
        assert method in valid_methods, f"Invalid initialization method. Must be one of {valid_methods}"
            
        # We support 3 forms of initialization!
        if method == "xavier":
            nn.init.xavier_normal_(self.weights)
 
        elif method == "normal":
            nn.init.normal_(self.weights)
            
        else:
            nn.init.kaiming_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer
        
        Args:
            x (torch.Tensor): Input tensor of shape (sequence_length x input_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length x output_features)
        """
        
        return FunctionalLinear.apply(x, self.weights)

    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return f"in_features={self.in_features}, out_features={self.out_features}"