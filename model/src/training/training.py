import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from mini_llama.model import MiniLlamaForCausalLM
from tiny_stories import TinyStoriesDataset

from mini_llama.cuda import linear_forward, linear_backward_weights, linear_backward_inputs
from mini_llama.cuda import embedding_forward, embedding_backward
from mini_llama.cuda import multi_attention_forward, multi_attention_backward

def train_mini_llama():

    # Defining our configuration for training
    config = {
        "batch_size": 64,
        "num_workers": 8,
        "vocab_size": 8192,
        "embedding_dim": 1024,
        "num_decoder_layers": 4,
        "num_attn_heads": 4,
        "mlp_layer_intermediate_dim": 2048,
        "dropout": 0.1,
        "padding_idx": 0,
        "tokenizer_path": "/root/mini-llama-3/model/src/tokenizers/tokenizer_configs/pirate_tokenizer_8K.json",
        "checkpoint_dir": "/root/mini-llama-3/checkpoints/tiny-stories-model-checkpoints",
        "max_training_steps": 1e5,
        "sequences_stride": 64
    }

    # Initializing Weights & Biases for experiment tracking
    wandb_logger = WandbLogger(
        project="Mini-Llama-3",
        config=config
    )

    # Creating our datasets
    train_dataset = TinyStoriesDataset(
        parquet_file_path="/root/mini-llama-3/datasets/tiny_stories_train_tokenized.parquet",
        output_file_path="/root/mini-llama-3/datasets/tiny_stories_train_sequences.parquet",
        stride=config["sequences_stride"],
        column_name="pirate_tokens"
    )
    
    val_dataset = TinyStoriesDataset(
        parquet_file_path="/root/mini-llama-3/datasets/tiny_stories_val_tokenized.parquet",
        output_file_path="/root/mini-llama-3/datasets/tiny_stories_val_sequences.parquet",
        stride=config["sequences_stride"],
        column_name="pirate_tokens"
    )

    # Creating dataloaders for efficient batch processing
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True
    )

    # Initializing our model
    model = MiniLlamaForCausalLM(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_decoder_layers=config["num_decoder_layers"],
        num_attn_heads=config["num_attn_heads"],
        mlp_layer_intermediate_dim=config["mlp_layer_intermediate_dim"],
        dropout=config["dropout"],
        padding_idx=config["padding_idx"],
        tokenizer_path=config["tokenizer_path"]
    )
    
    model = model.to("cuda")
    
    # Set up callbacks for checkpointing and learning rate monitoring
    callbacks = [
        ModelCheckpoint(
            dirpath=config["checkpoint_dir"],
            filename="mini-llama-{epoch:02d}-{validation_epoch_loss:.2f}",
            save_top_k=3,
            monitor="validation_epoch_loss",
            mode="min"
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    # Create our trainer
    trainer = L.Trainer(
        max_steps=config["max_training_steps"],
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        val_check_interval=1500
    )

    # Start training!
    trainer.fit(
        model, 
        train_loader,
        val_loader)

if __name__ == "__main__":
    train_mini_llama()
