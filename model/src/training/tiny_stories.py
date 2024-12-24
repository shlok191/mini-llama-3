import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm.auto import tqdm
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def create_sequences_from_tokens(
        parquet_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/tiny_stories_train_tokenized.parquet",
        output_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/tiny_stories_train_sequences.parquet",
        column_name: str = "original_tokens",
        sequence_length: int = 256,
        max_sequence_length: int = 320,
        stride: int = 32,
        padding_idx: int = 0,
        eos_idx: int = 2):
    """Generates uniform length sequences from a tokenized dataset

    Args:
        parquet_path (str): The path to the input parquet path
        output_path (str): The path to the output parquet path
        column_name (str): The column ID to fetch
        sequence_length (int): The maximum sequence length of the sequence
        max_sequence_length (int): The amount of padding to be added is determined with this
        stride (int): The stride to take when creating sequences
        padding_idx (int): The padding idx of the tokenizer
        eos_idx (int): The EOS idx of the tokenizer
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
            
            sequence = story[curr_start : curr_end]
            sequence = sequence + [padding_idx] * (max_sequence_length - sequence_length)
            
            # Pushing forward our indices
            curr_end += stride
            curr_start += stride

            # Finally, pushing the sequence to the list
            sequences.append(sequence)
    
    # Now, we store the sequences into a parquet file!
    table = pa.table({'text': sequences})
    pq.write_table(table, output_path)

class TinyStoriesDataset(Dataset):
    
    def __init__(self, 
        parquet_file_path: str = "/root/mini-llama-3/datasets/pirate_stories_train_tokenized_original.parquet",
        output_file_path: str = "/root/mini-llama-3/datasets/original_stories_train_sequences.parquet",
        column_name: str = "original_tokens",
        seq_len: int = 256,
        stride: int = 32,
        max_length: int = 320,
        padding_idx: int = 0,
        eos_idx: int = 2):
        """Creates sequences from a given list of original tokenized stories

        Args:
            parquet_file_path (str): The parquet file location containing the tokenized stories
            output_file_path (str): The location to store all the generated sequences to 
            column_name (str): The name of the column to read in
            seq_len (int): The number of non-padding tokens accepted. Defaults to 256.
            stride (int): The stride to take in the sliding window approach. Defaults to 32.
            max_length (int): The maximum length of the sequence created (includes padding). Defaults to 320.
            padding_idx (int): The index of the padding token. Defaults to 0.
            eos_idx (int): The index of the EOS token. Defaults to 2.
        """
        
        # Checking if the output file exists and loading it in if present
        if not os.path.exists(output_file_path):
            
            # Creating the stories
            create_sequences_from_tokens(
                parquet_path=parquet_file_path, 
                output_path=output_file_path,
                column_name=column_name,
                sequence_length=seq_len,
                stride=stride,
                max_sequence_length=max_length,
                padding_idx=padding_idx,
                eos_idx=eos_idx
            )
        
        # Reading in the parquet file
        df = pd.read_parquet(output_file_path)
        sequences_array = df.values.tolist()
        
        sequences = []
        for sequence in sequences_array:
            sequences.append(sequence[0])
             
        sequences = np.array(sequences)
            
        # Converting to tensor of type int32
        self.sequences = torch.tensor(sequences, dtype=torch.int32)
        
    def __getitem__(self, idx):
        
        input_ids = self.sequences[idx]
        labels = input_ids[1:]
        
        padding_positions = (input_ids == 0).nonzero(as_tuple=True)[0]
        curr_seq_lens = padding_positions[0].item() if len(padding_positions) > 0 else len(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "curr_seq_lens": curr_seq_lens
        }
    
    def __len__(self):
        return self.sequences.shape[0]
    

if __name__ == "__main__":
    
    test_stories = TinyStoriesDataset(
        parquet_file_path="/root/mini-llama-3/datasets/tiny_stories_val_tokenized.parquet",
        output_file_path="/root/mini-llama-3/datasets/tiny_stories_val_sequences.parquet",
    )
    
    print(len(test_stories))
    print(len(test_stories[0]['input_ids']))