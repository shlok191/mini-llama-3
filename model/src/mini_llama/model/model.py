# Importing torch libraries
import torch
import torch.nn as nn

# Importing custom layers
from mini_llama.embedding import Embedding
from mini_llama.rmsnorm import RMSNorm
from mini_llama.rope import RoPEmbedding
from mini_llama.decoder import DecoderLayer
from mini_llama.linear import Linear
from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer

# Importing PyTorch Lightning libraries
import lightning as L
import wandb
from typing import List

class MiniLlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int=8192,
        embedding_dim: int = 1024,
        context_length: int = 1024,
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
            context_length (int): The maximum amount of positional relationships supported
            num_decoder_layers (int): The number of decoder layers
            num_attn_heads (int): The number of attention heads / decoder layer
            mlp_layer_intermediate_dim (int): The intermediate dim for the MLP layer
            dropout (float, optional): Dropout rate. Defaults to 0.1
            padding_idx (int, optional): Token index for padding. Defaults to 0
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
                rope_dim=context_length,
                dropout=dropout,
                activation_fn=nn.SiLU()
            ) for _ in range(num_decoder_layers)
        ])
        
        # Defining the Root-Mean-Squared Normalization at the end
        self.norm = RMSNorm(dim=embedding_dim)
        self.rope = RoPEmbedding(dim=context_length)
        
    def forward(self, input_ids: torch.Tensor, curr_seq_lens: List[int]) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (seq_len)
            curr_seq_lens (List[int]): The length of the non-padded tokens for the batch
            
        Returns:
            torch.Tensor: Output embeddings of shape (seq_len, hidden_size)
        """

        print(input_ids.shape)
        
        # Hardcoded value to simplifiy implementation!
        assert input_ids.shape[-1] == 320
        assert not torch.isnan(input_ids).any()
        
        # Get embeddings using custom CUDA implementation
        hidden_states = self.embedding(input_ids)
        
        # Pass through each decoder layer
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, curr_seq_lens)
        
        # # Final normalization
        hidden_states = self.norm(hidden_states)
        hidden_states = self.rope(hidden_states)
        
        return hidden_states

class MiniLlamaForCausalLM(L.LightningModule):  
    def __init__(
        self,
        vocab_size: int = 8192,
        embedding_dim: int = 1024,
        num_decoder_layers: int = 4,
        num_attn_heads: int = 4,
        mlp_layer_intermediate_dim: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 0,
        tokenizer_path: str = "/root/mini-llama-3/model/src/tokenizers/tokenizer_configs/pirate_tokenizer_8K.json"
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
            tokenizer_path (str): The path to the tokenizer
        """
        super().__init__()
        
        # We will monitor all hyperparams except our model :)
        self.save_hyperparameters(ignore=["model"])
        
        self.learning_rate = 5e-4
        self.weight_decay = 5e-2
        self.max_steps = 1e6
        self.tokenizer = MiniLlamaTokenizer.load(tokenizer_path)
        self.validation_step_outputs = []
        self.tokenizer_path = tokenizer_path
        
        # Letting the first 1000 steps involve warmup
        self.warmup_steps = 1e3
        
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
        
        # Weight tying to the embedding table!
        self.lm_head.weights = nn.Parameter(self.model.embedding.embedding_table.T)  
              
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, curr_seq_lens: List[int]) -> torch.Tensor:
        """Generates logits for each input ID
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (seq_len,)
            labels (torch.Tensor): Labels of the shape (seq_len, )
            curr_seq_lens (List[int]): The length of the non-padded sequences for the batch
            
        Returns:
            torch.Tensor: Logits of shape (seq_len, vocab_size)
        """
        
        # First, we calculate the logits!
        logits = self.model(input_ids, curr_seq_lens)
        logits = self.lm_head(logits)
        
        # If no labels provided, we simply return the logits
        if labels is None:
            return logits
            
        # Otherwise we compute the loss
        shift_logits = logits[..., :-1, :].contiguous()

        # Computing the cross entropy loss!
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        
        # Collapsing every dimension except the last one!
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 320,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = 2
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
        
        print("Beginning generation...")
        
        # Making sure we don't exceed maximum sequence length!
        assert max_length <= 320, "Model only supports sequences up to length 320"
        
        # Initializing our current sequence with the question to be answered
        current_sequence = input_ids.clone()
        
        # Adding padding to match max_length
        if current_sequence.shape[0] < max_length:
            
            padding = torch.zeros(max_length - current_sequence.shape[0], dtype=current_sequence.dtype, device='cuda')
            current_sequence = torch.cat([current_sequence, padding])
        
        while True:
            
            # Find the position of the first padding token
            pad_positions = (current_sequence == 0).nonzero(as_tuple=True)[0]
        
            # If no padding tokens left, stop generation
            if len(pad_positions) == 0:
                break
                
            first_pad_pos = pad_positions[0].item()
            
            with torch.no_grad():
                logits = self.forward(current_sequence, labels=None, curr_seq_lens=[first_pad_pos])
                
            # We only really need the last logit!
            next_token_logits = logits[first_pad_pos - 1]
            next_token_logits = torch.clamp(next_token_logits, min=-1e-3, max=1e3)
            
            # Applying temperature sampling
            next_token_logits = next_token_logits / temperature
            
            # We will use one of the top-K samples that we will pick :)
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            top_k_logits = torch.clamp(top_k_logits, min=-1e-6, max=1e6)
            
            if torch.isnan(top_k_logits).any():
                raise ValueError("Logits contain NaN values.")

            elif torch.isinf(top_k_logits).any():
                raise ValueError("Logits contain INF values.")
            
            # Creating a distribution from the filtered logits
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Now, we sample from the probabilities
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_index]
            
            # Check if we generated an EOS token
            if next_token.item() == eos_token_id:
                
                # Place the EOS token and zero out the rest
                current_sequence[first_pad_pos] = eos_token_id
                
                # Add padding if need be
                if first_pad_pos + 1 < len(current_sequence):
                    current_sequence[first_pad_pos + 1:] = 0
                
                break
            
            # Adding in our brand new token!
            current_sequence[first_pad_pos] = next_token
        
        return current_sequence
    
    def training_step(self, batch, batch_idx):
        
        inputs = batch['input_ids']
        labels = batch['labels']
        curr_seq_lens = batch['curr_seq_lens']
        
        # Processing the inputs to get our loss
        loss = self.forward(inputs, labels, curr_seq_lens)
        
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Fetching the active learning rate
        scheduler = self.lr_schedulers()
        
        # Updating the LR according to CosineAnnealing :)
        if(self.global_step > self.warmup_steps):
            scheduler.step()
        
        if scheduler is not None:
            
            lr = scheduler.get_last_lr()[0]
            self.log("learning_rate", lr, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        inputs = batch['input_ids']
        labels = batch['labels']
        curr_seq_lens = batch['curr_seq_lens']
        
        loss = self.forward(inputs, labels, curr_seq_lens)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("validation_perplexity", perplexity, on_step=True, on_epoch=True)
        
        # Store for epoch end validation
        self.validation_step_outputs.append({"val_loss": loss, "val_perplexity": perplexity})
        
        # Generating sample text every N validation steps
        if batch_idx == 0:
            
            # Taking the first sequence from batch as prompt
            prompt = inputs[:50]
            
            # Generate continuation
            generated = self.tokenizer.decode(self.generate(
                prompt,
                max_length=320,
                temperature=1.0,
                top_k=50
            ).cpu().tolist())
            
            # Logging the generated text
            self.logger.experiment.log({
               "generated_samples": wandb.Table(
                   columns=["prompt", "generated"],
                   data=[[self.tokenizer.decode(prompt.cpu().tolist()), generated]]
               )
            })
        
        return loss
        
        
    def on_validation_epoch_end(self):
        
        # Calculating the mean loss and perplexity
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_ppl = torch.stack([x["val_perplexity"] for x in self.validation_step_outputs]).mean()
        
        # Log epoch-level metrics
        self.log("validation_epoch_loss", avg_loss)
        self.log("validation_epoch_perplexity", avg_ppl)
        
        # Clear saved outputs
        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
        
        # Creating and AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)  # Using betas recommended for LLMs
        )
        
        # Creating a scheduler with Cosine Annealing to update the LR as we go through
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }