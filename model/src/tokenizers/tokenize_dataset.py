from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
from typing import List
import json
from tqdm.auto import tqdm

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
        for line in tqdm(file, desc='🏴‍☠️ Plunderin\' Stories!', colour='green'):
            
            story = json.loads(line)
            
            # Maintaining some of the original text :)
            if count % 2 == 0:
                stories.append(story['original'])
                
            stories.append(story['pirate'])
            
    return stories

def tokenize_dataset(
        tokenizer: MiniLlamaTokenizer,
        dataset_path: str,
        output_path: str) -> None:
    """Tokenizes the dataset and saves it to a new file
    
    Args:
        tokenizer (MiniLlamaTokenizer): Tokenizer to use
        dataset_path (str): Path to the dataset to tokenize
        output_path (str): Path to save the tokenized dataset
    """
    
    # Load dataset
    dataset = load_pirate_stories(dataset_path)
    
    # Tokenize
    count = 0
    
    tokenized_dataset = []
    
    for story in tqdm(dataset, desc='🏴‍☠️ Tokenizin\' Stories!', colour='green'):
        
        story = story.replace('"', '\"').lower()
        tokenized_story = tokenizer.encode(story)
            
        tokenized_dataset.append(tokenizer.encode(story))
        count += 1
        
    # Save
    with open(output_path, "w") as f:
        
        for tokenized_story in tokenized_dataset:
            f.write(f"{tokenized_story}\n")
            
    print(f"Tokenized dataset saved to {output_path}")
    
if __name__ == "__main__":
    # Load stories
    stories = load_pirate_stories()
    
    # Loading in the tokenizer
    tokenizer = MiniLlamaTokenizer(None, 8100, 8160)
    tokenizer = tokenizer.load("./pirate_tokenizer_8k.json")
    
    # Tokenizing the validation set
    tokenize_dataset(
        tokenizer, 
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation.jsonl",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation_tokenized.jsonl"
    )
    
    tokenize_dataset(
        tokenizer, 
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation.jsonl",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation_tokenized.jsonl"
    )