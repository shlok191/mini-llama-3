import torch
import torch.nn as nn

from mini_llama.embedding import Embedding
from mini_llama.rmsnorm import RMSNorm
from mini_llama.rope import RoPEmbedding
from mini_llama.decoder import DecoderLayer
from mini_llama.linear import Linear

class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int=8192,
        embedding_dim: int = 1024,
        num_decoder_layers: int = 4,
        num_attn_heads: int = 4,
        mlp_layer_intermediate_dim: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        """Initializes our custom model :)
        
        Args:
            vocab_size (int): The total amount of custom tokens we can have
            embedding_dim (int): The size of the 1D embedding vectors
            num_decoder_layers (int): The number of decoder layers
            num_attn_heads (int): The number of attention heads / decoder layer
            mlp_layer_intermediate_dim (int): The intermediate dim for the MLP layer
            dropout (float, optional): Dropout rate. Defaults to 0.1
            padding_idx (int, optional): Token index for padding. Defaults to 1
        """
        
        super().__init__()
        
        # Defining our custom embedding layer        
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embed_dims=embedding_dim,
            padding_token_index=padding_idx,
            init_method="xavier"
        )
        
        # Defining a stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=embedding_dim,
                num_heads=num_attn_heads,
                intermediate_size=mlp_layer_intermediate_dim,
                rope_dim=embedding_dim,
                dropout=dropout,
                activation_fn=nn.SiLU()
            ) for _ in range(num_decoder_layers)
        ])
        
        # Defining the Root-Mean-Squared Normalization at the end
        self.norm = RMSNorm(dim=embedding_dim)
        self.rope = RoPEmbedding(dim=embedding_dim)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (seq_len)
            
        Returns:
            torch.Tensor: Output embeddings of shape (seq_len, hidden_size)
        """
        
        # Hardcoded value to simplifiy implementation!
        assert input_ids.shape[0] == 256
        
        # Get embeddings using custom CUDA implementation
        hidden_states = self.embedding(input_ids)
        
        # Pass through each decoder layer
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)
            
        # Final normalization
        hidden_states = self.norm(hidden_states)
        hidden_states = self.rope(hidden_states)
        
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        embedding_dim: int = 1024,
        num_decoder_layers: int = 4,
        num_attn_heads: int = 4,
        mlp_layer_intermediate_dim: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 1
    ):
        """Initialize the Llama model for causal language modeling
        
        Args:
            vocab_size (int): The total amount of custom tokens we can have
            embedding_dim (int): The size of the 1D embedding vectors
            num_decoder_layers (int): The number of decoder layers
            num_attn_heads (int): The number of attention heads / decoder layer
            mlp_layer_intermediate_dim (int): The intermediate dim for the MLP layer
            dropout (float, optional): Dropout rate. Defaults to 0.1
            padding_idx (int, optional): Token index for padding. Defaults to 1
        """
        super().__init__()
        
        # Main model
        self.model = LlamaModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_decoder_layers=num_decoder_layers,
            num_attn_heads=num_attn_heads,
            mlp_layer_intermediate_dim=mlp_layer_intermediate_dim,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Language modeling head
        self.lm_head = Linear(embedding_dim, vocab_size)
        
        # Weight tying
        self.lm_head.weights = self.model.embed_tokens.embedding_table
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (seq_len,)
            
        Returns:
            torch.Tensor: Logits of shape (seq_len, vocab_size)
        """
        # Get hidden states from base model
        hidden_states = self.model(input_ids)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits