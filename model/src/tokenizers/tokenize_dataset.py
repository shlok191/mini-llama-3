from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
from typing import List
import json
from tqdm.auto import tqdm
import pandas as pd

def convert_json_to_parquet(json_path: str, parquet_path_1: str, parquet_path_2: str) -> None:
    """Converts JSON to more memory-fridnedly Parquet format

    Args:
        json_path (str): The path to the JSON files
        parquet_path (str): The path to the Parquet files
    """
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    non_512_pirate = 0
    non_512_original = 0
    
    for entry in data['original']:
        
        if len(entry) != 512:
            non_512_original += 1
    
    for entry in data['pirate']:
        
        if len(entry) != 512:
            non_512_pirate += 1
        
    print(f"Original stories not of length 512: {non_512_original}")
    print(f"Pirate stories not of length 512: {non_512_pirate}")
    
    # Convert to pandas DataFrame
    original_df = pd.DataFrame({
        'original_tokens': data['original'],
    })
    
    pirate_df = pd.DataFrame({
        'pirate_tokens': data['pirate'],
    })
    
    # Save as Parquet
    original_df.to_parquet(parquet_path_1)
    pirate_df.to_parquet(parquet_path_2)
    
    print(f"Converted {json_path} to Parquet format at {parquet_path_1}")
    
def load_pirate_stories(path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train.jsonl") -> List[str]:
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
            
            # Only keeping 25% of the original stories!
            if count % 4 == 0:
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
    # tokenizer = MiniLlamaTokenizer.load("/Users/sabarwal/work/projects/mini-llama-3/model/src/tokenizers/pirate_tokenizer_8K.json")
    
    # # Tokenizing the training set first
    # print("Tokenizing the training set...\n")
    
    # tokenize_dataset(
    #     tokenizer, 
    #     "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train.jsonl",
    #     "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train_tokenized.json"
    # )
    
    # # Tokenizing the validation set next!
    # print("\nTokenizing the validation set...\n")
    
    # tokenize_dataset(
    #     tokenizer, 
    #     "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation.jsonl",
    #     "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation_tokenized.json"
    # )
    
    # Converting the tokenized JSON files to Parquet format
    convert_json_to_parquet(
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train_tokenized.json",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/original_stories_train_tokenized.parquet",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train_tokenized.parquet"
    )
    
    convert_json_to_parquet(
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation_tokenized.json",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/original_stories_validation_tokenized.parquet",
        "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_validation_tokenized.parquet"
    )