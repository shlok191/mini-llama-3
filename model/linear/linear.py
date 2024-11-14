import torch
import torch.nn as nn
from torch.autograd import Function
from cuda_kernels import linear_forward, linear_backward

class FunctionalLinear(Function):
    
    @staticmethod
    def forward(ctx, weights: torch.Tensor, bias: torch.Tensor, X: torch.Tensor):
        
        ctx.save_for_backward(X, bias, weights)
        
        output = linear_forward(weights, bias, X)
        return output

    @staticmethod
    def backward(ctx, gradient_output):
        
        X, bias, weights = ctx.saved_tensors
        grad_weight = linear_backward(gradient_output, weights, bias, X)[0]
        
        return None, grad_weight, None

class Linear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool, init_method: str, device: str):
        """ Defining a custom Linear / Dense layer implementation with a tailored CUDA backend!

        Args:
            in_features (int): The number of incoming features
            out_features (int): The number of values to put out per input value in the batch
            bias (bool): Whether or not to have a bias term
            init_method (str): The method with which to initialize the layer values
            device (str): The device on which to store the layer
        """

        super().__init__()
        
        # Storing the class variables
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        
        # Defining the weights matrix and a custom bias parameter if requested
        self.weights = torch.nn.Parameter(torch.empty([in_features, out_features], dtype=torch.float32), requires_grad=True)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=torch.float32))
        
        else:
            self.register_parameter('bias', None)

        # Initializing the linear layer
        self._initialize_layer(method=init_method)
        
        # Moving the parameters to the requested device
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)

    def _initialize_layer(self, method=None):
        
        # Checking for a valid method name!
        assert method in ["xavier", "normal", "kaiming-he"], "Please specifiy a valid initialization method!"
        
        if method == "xavier":
            nn.init.xavier_normal_(self.weights)
            
        elif method == "normal":
            nn.init.normal_(self.weights)
            
        else:
            nn.init.kaiming_normal_(self.weights)
    
    def forward(self, X):
        
        return FunctionalLinear.apply(self.weights, self.bias, X)
