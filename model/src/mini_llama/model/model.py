import torch
import torch.nn as nn

from mini_llama.embedding import Embedding
from mini_llama.rmsnorm import RMSNorm
from mini_llama.rope import RoPEmbedding
from mini_llama.decoder import DecoderLayer
from mini_llama.linear import Linear

class MiniLlamaModel(nn.Module):
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
            vocab_size=vocab_size,
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
        
        # # Final normalization
        hidden_states = self.norm(hidden_states)
        hidden_states = self.rope(hidden_states)
        
        return hidden_states

class MiniLlamaForCausalLM(nn.Module):
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
        self.model = MiniLlamaModel(
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
        self.lm_head.weights = nn.Parameter(self.model.embedding.embedding_table.T)  
              
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generates logits for each input ID
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (seq_len,)
            labels (torch.Tensor): Labels of the shape (seq_len, )
            
        Returns:
            torch.Tensor: Logits of shape (seq_len, vocab_size)
        """
        
        # First, we calculate the logits!
        logits = self.model(input_ids)
        logits = self.lm_head(logits)
        
        # If no labels provided, we simply return the logits
        if labels is None:
            return logits
            
        # Otherwise we compute the loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].to(dtype=torch.int64).contiguous()
        
        # Computing the cross entropy loss!
        loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """Generates text given a prompt by sampling from the model's outputs.
        
        Args:
            input_ids (torch.Tensor): Starting sequence of tokens (the prompt)
            max_length (int): Maximum sequence length to generate
            temperature (float): Controls randomness in sampling. Lower values make it more deterministic
            top_k (int): Number of highest probability tokens to consider for sampling
            
        Returns:
            torch.Tensor: Generated sequence of tokens including the prompt
        """
        
        # Making sure we don't exceed maximum sequence length!
        assert max_length <= 256, "Model only supports sequences up to length 256"
        
        # Initializing our current sequence with the question to be answered
        current_sequence = input_ids.clone()
        
        while current_sequence.shape[0] < max_length:
            
            with torch.no_grad():
                logits = self.forward(current_sequence, labels=None)
                
            # We only really need the last logit!
            next_token_logits = logits[-1, :]
            
            # Applying temperature sampling
            next_token_logits = next_token_logits / temperature
            
            # We will use one of the top-K samples that we will pick :)
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Creating a distribution from the filtered logits
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Now, we sample from the probabilities
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_index]
            
            # Adding the current token to the final sequence!
            current_sequence = torch.cat([current_sequence, next_token])
        
        return current_sequence