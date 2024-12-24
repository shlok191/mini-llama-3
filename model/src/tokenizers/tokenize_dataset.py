from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
from typing import List
import json
from tqdm.auto import tqdm
import pandas as pd

def create_sequences_from_tokens(
        parquet_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/tiny_stories_train_tokenized.parquet",
        output_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/tiny_stories_train_sequences.parquet",
        column_name: str = "original_tokens",
        sequence_length: int = 256,
        stride: int = 32,
        padding_idx: int = 0,
        eos_idx: int = 2):
    """Generates uniform length sequences from a tokenized dataset

    Args:
        parquet_path (str, optional): The path to the input parquet path
        output_path (str, optional): The path to the output parquet path
        column_name (str, optional): The column ID to fetch
        sequence_length (int, optional): The maximum sequence length of the sequence
        stride (int, optional): The stride to take when creating sequences
        padding_idx (int, optional): The padding idx of the tokenizer
        eos_idx (int, optional): The EOS idx of the tokenizer
    """

    # Reading in the parquet file
    dataset = pd.read_parquet(parquet_path, columns=[column_name])[column_name].values.tolist()
    
    sequences = []
    
    for story in tqdm(dataset, colour="green", desc="Creating sequences from tokenized stories..."):
        
        # Converting the story to a list in python
        story = story.tolist()
        
        curr_start = 0
        curr_end = sequence_length
        
        eos_position = story.index(eos_idx)
        
        # Only generate sequences until the EOS token is not crossed!
        while(curr_end <= eos_position):
            
            sequence = story[curr_start:curr_end]
            sequence = sequence + [padding_idx] * (320 - sequence_length)
            
            # Pushing forward our indices
            curr_end += stride
            curr_start += stride

            # Finally, pushing the sequence to the list
            sequences.append(sequence)
    
    # Now, we store the sequences into a parquet file!
    df = pd.DataFrame(sequences)
        
    print("\nWe are finished now! Here is some analysis:\n")

    print(df.describe())
    print(df.head(5))
    
    df.to_parquet(output_path)
      
def load_pirate_stories(path: str = "/Users/sabarwal/work/projects/mini-llama-3/datasets/pirate_stories_train.jsonl") -> List[str]:
    """Loads the TinyStories - pirated version :)
    
    Args:
        path (str): Path to the JSONL file
        
    Returns:
        List[str]: Lists of the stories as strings
    """
    
    stories = {"original": [], "pirate": []}
    
    print("Loading stories from JSONL...")
    count = 0
    
    with open(path, 'r') as file:
        
        # Read in stories line by line    
        for line in tqdm(file, desc='🏴‍☠️ Plunderin\' Stories!', colour='green'):
            
            story = json.loads(line)
        
            # Only keeping 33% of the original stories!
            if count % 3 == 0:
                stories["original"].append(story['original'])
                
            stories["pirate"].append(story['pirate'])
            
            count += 1
            
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
    
    # Loading in the dataset
    dataset = load_pirate_stories(dataset_path)
    
    # Stores the tokenized dataset
    tokenized_dataset = {"original": [], "pirate": []}
    
    # Taking all of the available stories!
    for story in tqdm(dataset["pirate"], desc='🏴‍☠️ Tokenizin\' Stories!', colour='green'):
        
        story = story.replace('"', '\"')            
        tokenized_dataset["pirate"].append(tokenizer.encode(story, 512))
    
    # Taking all of the available original stories!
    for story in tqdm(dataset["original"], desc='Tokenizing the original TinyStories! :)', colour='green'):
        
        story = story.replace('"', '\"')           
        tokenized_dataset["original"].append(tokenizer.encode(story, 512))
    
    # Saving finally!
    original_df = pd.DataFrame({
        'original_tokens': tokenized_dataset["original"],
    })
    
    pirate_df = pd.DataFrame({
        'pirate_tokens': tokenized_dataset["pirate"],
    })
    
    original_df.to_parquet(output_path.replace(".json", "_original.parquet"))
    pirate_df.to_parquet(output_path.replace(".json", "_pirate.parquet"))
    
    # Save to Parquet
    print(f"Tokenized dataset saved to {output_path}!")
    
if __name__ == "__main__":
    
    # Loading in the tokenizer
    tokenizer = MiniLlamaTokenizer.load("/root/mini-llama-3/model/src/tokenizers/pirate_tokenizer_8K.json")
    
    # Tokenizing the training set first
    print("Tokenizing the training set...\n")
    
    tokenize_dataset(
        tokenizer, 
        "/root/mini-llama-3/datasets/pirate_stories_train.jsonl",
        "/root/mini-llama-3/datasets/pirate_stories_train_tokenized.json"
    )
    
    # Tokenizing the validation set next!
    print("\nTokenizing the validation set...\n")
    
    tokenize_dataset(
        tokenizer, 
        "/root/mini-llama-3/datasets/pirate_stories_validation.jsonl",
        "/root/mini-llama-3/datasets/pirate_stories_validation_tokenized.json"
    )