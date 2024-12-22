from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
import json
from typing import List

def load_pirate_stories(path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train.jsonl") -> List[str]:
    """Loads the TinyStories - pirated version :)
    
    Args:
        path (str): Path to the JSONL file
        
    Returns:
        List[str]: Lists of the stories as strings
    """
    
    stories = []
    
    print("Loading stories from JSONL...")
    count = 0
    
    with open(path, 'r') as file:
        
        # Read in stories line by line    
        for line in file:
            
            story = json.loads(line)
            
            stories.append(story['original'])
            stories.append(story['pirate'])
        
    return stories

def train_tokenizer(stories: list, vocab_size: int = 8192, iterations: int = 8192 - 99) -> MiniLlamaTokenizer:
    """Trains our custom BPE tokenizer on the dataset provided
    
    Args:
        stories (list): List of stories to train on
        vocab_size (int): Total tokens we will possibly store
        iterations (int): Number of times we wish to merge most adjacent occuring tokens
        
    Returns:
        MiniLlamaTokenizer: Our trained tokenizer :)
    """
    
    print(f"\nTraining tokenizer with vocab size {vocab_size} and {iterations} iterations...")
    
    # Defining a tokenizer object
    tokenizer = MiniLlamaTokenizer(vocab_size=vocab_size, iterations=iterations)
    
    # Training!
    tokenizer.train(stories)
    
    return tokenizer

def test_tokenizer(
        tokenizer: MiniLlamaTokenizer,
        validation_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train.jsonl",
        num_samples: int = 3) -> None:
    """Tests the tokenizer on some sample stories
    
    Args:
        tokenizer (MiniLlamaTokenizer): Trained tokenizer to test
        validation_path (str): The path to the validation dataset
        num_samples (int): Number of samples to test
    """
    
    print("\nTesting tokenizer on samples:")
    
    # Soem test texts
    test_texts = load_pirate_stories(validation_path)
    
    # Printing out our tokenized entries
    for i, text in enumerate(test_texts[:num_samples]):
        
        text = text.replace('\n', '')
        
        print(f"\nSample {i+1}:")
        print(f"\nOriginal: {text}")
        
        encoded = tokenizer.encode(text)
        print(f"\nEncoded: {encoded[:10]}... (length: {len(encoded)})")
        
        decoded = tokenizer.decode(encoded)
        print(f"\nDecoded: {decoded}")

        # Allowing for proper demarcations
        print(f"{'=' * 100}")
        
if __name__ == "__main__":
    
    # # Load stories
    # stories = load_pirate_stories()
    # print(f"Loaded {len(stories)} stories")
    
    # # Train tokenizer
    # tokenizer = train_tokenizer(stories)
    
    # # Test it
    # test_tokenizer(tokenizer)
    
    # # Save tokenizer
    save_path = "pirate_tokenizer_8K.json"
    
    # print(f"\nSaving tokenizer to {save_path}")
    # tokenizer.save(save_path)
    
    # Verifying loading works
    print("\nVerifying loaded tokenizer:")
    
    loaded_tokenizer = MiniLlamaTokenizer.load(save_path)
    test_tokenizer(loaded_tokenizer)
