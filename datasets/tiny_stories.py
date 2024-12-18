from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np

class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        split="train",
        max_length=256,
        stride=4,
        cache_dir=None
    ):
        """Initializes the TinyStories dataset processor.
        
        Args:
            split (str): Dataset split to use ("train" or "validation")
            max_length (int): Maximum sequence length for processing
            stride (int): Stride length for sliding window
            cache_dir (str): Directory to cache the downloaded dataset
        """
        
        super().__init__()
        
        # Loading in the Tiny Stories dataset
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir
        )
        
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        
        # Processing all available Tiny Stories :)
        self._process_stories()
    
    def _process_stories(self):
        """Processes all stories into training examples using sliding windows."""
        
        for story_dict in self.dataset:
            
            # Separating the story
            story = story_dict['text']
        
            # Create sliding windows across the raw text
            for i in range(0, len(story) - self.max_length + 1, self.stride):
                
                # Creating our window
                window = story[i:i + self.max_length]
                
                # Making sure we only keep full-length windows
                if len(window) == self.max_length:
                    
                    # Input is all characters except last and the target is all except first!
                    input_text = window[:-1]  
                    target_text = window[1:]   
                    
                    self.examples.append({
                        "input_ids": input_text,
                        "labels": target_text
                    })

    
    def __len__(self):
        """Returns the number of processed examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Gets a single training example.
        
        Returns:
            tuple: (input_ids, labels) as torch tensors
        """
        example = self.examples[idx]
        
        # Convert to torch tensors
        input_ids = torch.tensor(example["input_ids"], dtype=torch.int32)
        labels = torch.tensor(example["labels"], dtype=torch.int32)
        
        return input_ids, labels

# Example usage
def create_dataloaders(batch_size=1, num_workers=4):
    """Creates train and validation dataloaders.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TinyStoriesDataset(split="train")
    val_dataset = TinyStoriesDataset(split="validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Important for GPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    
    a, b = create_dataloaders()
    print(a)