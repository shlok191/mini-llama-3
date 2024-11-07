import torch
import torch.nn as nn
from torch.autograd import Function
import math

class FunctionalEmbedding(Function):
    
    @staticmethod
    def forward(ctx, indices, weights, padding_token_value):
        
        ctx.save_for_backward(indices, weights)
        ctx.padding_token_value = padding_token_value
        
        output = torch.ops.custom.embedding_forward(indices, weights, padding_token_value)[0]
        return output
    
    @staticmethod
    def backward(ctx, gradient_output):
        
        indices, weights = ctx.saved_tensors
        grad_weight = torch.ops.custom.embedding_backward(gradient_output, indices, weights, ctx.padding_token_value)[0]
        
        return None, grad_weight, None

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings: int = 16384, embed_dims: int = 4196, padding_token_value: int = -100, init_method="xavier"):
        """Custom Implementation of the Embedding Layer with self-implemented CUDA backend :)

        Args:
            num_embeddings (int): The number of tokens for which to have custom embeddings
            embed_dims (int, optional): The number of scalar values for each embedding vector. Defaults to 4196.
            padding_token_value (int, optional): The representation for padding tokens. Defaults to -100.
            init_method (str, optional): The way with which to initialize the layer. Defaults to "xavier".
        """
        
        super().__init__()

        # Storing the passed variables
        self.num_embeddings = num_embeddings
        self.embed_dims = embed_dims
        self.padding_token_value = padding_token_value
        
        # Creating the embedding lookup table
        self.embedding_table = torch.nn.Parameter(torch.empty(num_embeddings, embed_dims, dtype=torch.float16), requires_grad=True)
        
        # Initializing the table
        self._initialize_table(init_method)
        
    def _initialize_table(self, method=None):
        """Initializes the embedding table with the specified initialization method

        Args:
            method (str, optional): The way with which to initialize the layer
        """
        
        assert method in ["xavier", "normal", "kaiming-he"], "Please specifiy a valid initialization method!"
        
        if method == "xavier":
            nn.init.xavier_normal_(self.embedding_table)
            
        elif method == "normal":
            nn.init.normal_(self.embedding_table)
            
        else:
            nn.init.kaiming_normal_(self.embedding_table)
            
    def forward(self, indices: torch.Tensor):
        """Performs the forward pass for the given embedding indices

        Args:
            indices (torch.Tensor): A batched Torch Tensor describing the indices to fetch

        Returns:
            torch.Tensor: Returns the requested embedding vectors
        """
        
        # Input validation
        assert indices.dim() != 2, f"Expected 2D input, got {indices.dim()}D input"
        
        return FunctionalEmbedding.apply(
            indices, 
            self.embedding_table,
            self.padding_token_value
        )
        