from mini_llama.model import MiniLlamaForCausalLM
from lightning.loggers import WandbLogger

# Example of how to set up training with the completed model
def setup_training(
    model: MiniLlamaForCausalLM,
    train_dataloader,
    val_dataloader,
    project_name: str = "Mini-Llama-3"
):
    """Sets up Lightning training with WandB logging and checkpointing"""
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=project_name,
        log_model="all"
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="minillama-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=model.max_steps,
        devices="cuda:0",
        gradient_clip_val=1.0,
        val_check_interval=1000,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    
    return trainer

if __name__ == "__main__":
    
    trainer = setup_training()
    trainer.train()
    