from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional
import json
from tqdm.auto import tqdm

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

class ParallelBPETrainer:
    
    def __init__(self, vocab_size, iterations, word_freqs, merges, string_to_tokens, tokens_to_string):
        
        self.vocab_size = vocab_size
        self.iterations = iterations
        self.word_freqs = word_freqs
        self.merges = merges
        self.string_to_tokens = string_to_tokens
        self.tokens_to_string = tokens_to_string
        
        self.num_processes = cpu_count() - 1 or 1

    @staticmethod
    def get_pairs_for_chunk(word_freq_items):
        """Calculate pair frequencies for a chunk of words"""
        
        pairs = defaultdict(int)
        
        for word, freq in word_freq_items:
            symbols = word.split()
        
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        
        return dict(pairs)

    def merge_pair_in_word(self, pair, word, freq):
        """Merge a given pair in a single word"""
        
        parts = word.split()
        i = 0
        
        while i < len(parts) - 1:
        
            if (parts[i], parts[i + 1]) == pair:
                parts[i:i + 2] = [''.join(pair)]
        
            else:
                i += 1
        
        return ' '.join(parts), freq

    def parallel_get_pair_frequencies(self, word_freqs):
        """Calculate pair frequencies in parallel"""
        
        # Split word_freqs into chunks for each process
        items = list(word_freqs.items())
        chunk_size = max(1, len(items) // self.num_processes)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Process chunks in parallel
        with Pool(self.num_processes) as pool:
            chunk_results = pool.map(self.get_pairs_for_chunk, chunks)

        # Combine results from all chunks
        combined_pairs = defaultdict(int)
        
        for chunk_pairs in chunk_results:
            for pair, freq in chunk_pairs.items():
                combined_pairs[pair] += freq

        return combined_pairs

    def parallel_merge_tokens(self, best_pair, word_freqs):
        """Merge tokens in parallel"""
        items = list(word_freqs.items())
        
        chunk_size = max(1, len(items) // self.num_processes)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Process chunks in parallel
        merge_func = partial(self.merge_pair_in_word, best_pair)
        
        with Pool(self.num_processes) as pool:
            merged_chunks = pool.starmap(merge_func, items)

        return dict(merged_chunks)

    def train(self):
        """Train the BPE tokenizer using parallel processing"""
        
        # Convert words to space-separated character sequences
        word_freqs = {' '.join(word): freq for word, freq in self.word_freqs.items()}

        for i in tqdm(range(self.iterations), desc='🏴‍☠️ Training the BPE Tokenizer...', colour='green'):
        
            # Get pair frequencies in parallel
            pairs = self.parallel_get_pair_frequencies(word_freqs)
            
            if not pairs:
                break

            # Finding the most frequent pair
            most_freq = max(pairs.items(), key=lambda x: x[1])
            best_pair = most_freq[0]

            # Merging the pair in parallel
            word_freqs = self.parallel_merge_tokens(best_pair, word_freqs)

            # Update vocabulary
            self.merges[best_pair] = ''.join(best_pair)
            self.string_to_tokens[''.join(best_pair)] = len(self.tokens_to_string)
            self.tokens_to_string[len(self.tokens_to_string)] = ''.join(best_pair)

            # Check vocabulary size
            if len(self.string_to_tokens) >= self.vocab_size:
                print(f"Reached maximum vocabulary size: {self.vocab_size}")
                break

            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges. Vocabulary size: {len(self.string_to_tokens)}")

        return self.merges, self.string_to_tokens, self.tokens_to_string
    
class MiniLlamaTokenizer:
    """This is a simplified implementation of the Mini-LLama Tokenizer. I am using Byte-Pair encoding as
    suggested by the sentencepiece paper from Google with iterations=2048.
    
    If time permits, I will use concurrency to parallelize the process of counting adjacent tokens as well!
    """
    
    def __init__(self, text_corpus: Optional[List[str]], iterations: int, vocab_size: int = 8192):
        """Initializes and trains the tokenizer on the given texts for [iterations] iterations

        Args:
            text_corpus (Optional[List[str]]): A list of texts or None (None in case of loading tokenizer)
            iterations (int): The number of times to merge the most frequently adjacent tokens
            vocab_size (int): The total amount of tokens storable in the memory
        """

        # Storing all of the important variables
        self.iterations = iterations
        self.vocab_size = vocab_size
        
        # Converts tokens to strings and vice versa!
        self.string_to_tokens: Dict[str, int] = dict()
        self.tokens_to_string: Dict[int, str] = dict()
        
        self.merges: Dict[Tuple[str, str], str] = {}
        self.word_freqs: Dict[str, int] = defaultdict(int)
        
        # Defining some helpful special tokens!
        self.special_tokens = ["<padding>", "<begin_of_sentence>", "<end_of_sentence>",
            "<unknown>", "</w>"]

        # Initializing the vocabulary with all of basic english characters + </w> special token
        for token in self.special_tokens:
            
            self.string_to_tokens[token] = len(self.string_to_tokens)
            self.tokens_to_string[len(self.tokens_to_string)] = token
        
        # Adding some basic tokens!
        for c in 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/\'\"':
            
            if c not in self.string_to_tokens:
                
                self.string_to_tokens[c] = len(self.string_to_tokens)
                self.tokens_to_string[len(self.tokens_to_string)] = c
            
        if text_corpus is not None:
            
            # Processing the corpus to build initial word frequencies
            for text in tqdm(text_corpus, colour='green', desc='🏴‍☠️ Building The BPE Tokenizer...'):
                
                words = text.lower().strip().split()
                
                for word in words:
                
                    # Counting the word frequencies but not adding to the vocabulary yet
                    self.word_freqs[word + "</w>"] += 1
                    
                    
    def merge_tokens(self, pair: Tuple[str, str], word_frequency: Dict[str, int]) -> Dict[str, int]:
        """Merges all of the token pairs with the highest adjacency frequency counts

        Args:
            pair (Tuple[str, str]): All pairs of adjacent tokens
            word_frequency (Dict[str, int]): The frequencies of words that occur

        Returns:
            Dict[str, int]: Returns a dict of new word frequencies
        """
        
        new_word_frequency = {}
        
        # Creating a pattern that ignores the special sequences and skips all other tokens around
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        # Creating a joined pair
        replacement = ''.join(pair)
        
        for word, freq in word_frequency.items():
            
            # Creating new words
            new_word = pattern.sub(replacement, word)
            new_word_frequency[new_word] = freq
            
        return new_word_frequency
    
    def train(self) -> None:
        """Defines the sets of tokens for the tokenizer using BPE encoding given the corpus and iterations count"""
        
        # Convert words to space-separated character sequences
        word_freqs = {' '.join(word): freq for word, freq in self.word_freqs.items()}
        
        # Defining a trainer to parallelize this task
        trainer = ParallelBPETrainer(
            vocab_size=self.vocab_size,
            iterations=self.iterations,
            word_freqs=word_freqs,
            merges=self.merges,
            string_to_tokens=self.string_to_tokens,
            tokens_to_string=self.tokens_to_string
        )
        
        # Beginning parallel training
        merges, string_to_tokens, tokens_to_string = trainer.train()
        
        # Storing everything into object instance
        self.merges = merges
        self.string_to_tokens = string_to_tokens
        self.tokens_to_string = tokens_to_string
        
    def encode(self, text: str, max_length: int = 1024, truncation: bool = False) -> List[int]:
        """Optimized version of the encoder"""
        
        # Pre-compile regex patterns for common operations
        
        if not hasattr(self, '_split_pattern'):
            self._split_pattern = re.compile(r'\s+')
        
        # Pre-fetch commonly used token IDs
        
        if not hasattr(self, '_common_tokens'):
        
            self._common_tokens = {
                'bos': self.string_to_tokens['<begin_of_sentence>'],
                'eos': self.string_to_tokens['<end_of_sentence>'],
                'unk': self.string_to_tokens['<unknown>'],
            }
        
        # Process text in batches
        words = self._split_pattern.split(text.lower().strip())
        
        encoded = []
        
        # Create a cache for processed subwords
        if not hasattr(self, '_word_cache'):
            self._word_cache = {}
        
        # Process each word
        for word in words:
        
            # Check cache first
            cache_key = word + "</w>"
        
            if cache_key in self._word_cache:
                encoded.extend(self._word_cache[cache_key])
                continue
                
            # Process new word
            current = ' '.join(list(cache_key))
        
            while True:
        
                # Try to apply merges efficiently
                changed = False
        
                for pair, merge in self.merges.items():
                    pair_str = ' '.join(pair)
                    
                    if pair_str in current:
                        current = current.replace(pair_str, merge)
                        changed = True
                        break
                
                if not changed:
                    break
            
            # Convert to token IDs efficiently
            tokens = []
            
            for string in current.split():
                tokens.append(
                    self.string_to_tokens.get(string, self._common_tokens['unk'])
                )
            
            # Cache the result
            self._word_cache[cache_key] = tokens
            encoded.extend(tokens)
        
        # Handle truncation if needed
        if truncation and len(encoded) > max_length - 2:
            encoded = encoded[:max_length - 2]
        
        # Add BOS and EOS tokens
        return [self._common_tokens['bos']] + encoded + [self._common_tokens['eos']]

    def clear_cache(self):
        """Clear the word cache if needed"""
        if hasattr(self, '_word_cache'):
            self._word_cache.clear()
            
    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a string efficiently
        
        Args:
            token_ids (List[int]): The list of token IDs to convert
            
        Returns:
            str: The decoded string
        """
        # Pre-fetch special tokens if not already cached
        if not hasattr(self, '_special_tokens_set'):
            self._special_tokens_set = {"<padding>", "<begin_of_sentence>", "<end_of_sentence>"}
        
        # Pre-allocate list with estimated size
        text = []
        text_append = text.append  # Local reference for faster append
        
        # Process tokens more efficiently
        for token in token_ids:
            # Use get() with default value instead of multiple checks
            token_string = self.tokens_to_string.get(token, "<unknown>")
            
            # Skip special tokens efficiently using set lookup
            if token_string in self._special_tokens_set:
                continue
                
            text_append(token_string)
        
        # Join and process the final string efficiently
        if not text:
            return ""
            
        # Fast string joining and </w> replacement
        result = "".join(text)
        if "</w>" in result:
            result = result.replace("</w>", " ")
        
        return result.strip()
        
    def save(self, path: str) -> None:
        """Save the tokenizer properties in a JSON file

        Args:
            path (str): The path where to store the JSON
        """
        
        # Defining a dictionary to store the associated values
        save_dict = {
            'string_to_tokens': self.string_to_tokens,  # Store complete dictionary
            'tokens_to_string': {str(k): v for k, v in self.tokens_to_string.items()},  # Convert int keys to str for JSON
            'merges': {' '.join(key): value for key, value in self.merges.items()},
            'special_tokens': self.special_tokens
        }
    
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> 'MiniLlamaTokenizer':
        """Loads in a JSON configuration for the BPE tokenizer
        ( Defining as a classmethod since it is used before an obj is defined! )

        Args:
            path (str): The path of the saved configuration file

        Returns:
            MiniLlamaTokenizer: The tokenizer object
        """
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = MiniLlamaTokenizer(None, iterations=0)
        
        # Saving the respective dictionaries
        tokenizer.string_to_tokens = data['string_to_tokens']
        tokenizer.tokens_to_string = {int(k): v for k, v in data['tokens_to_string'].items()}
    
        tokenizer.merges = {tuple(key.split()): value for key, value in data['merges'].items()}
        tokenizer.special_tokens = data['special_tokens']
        
        return tokenizer

    def example_usage():
        
        # Example texts
        
        texts = [
            "Savvy? Not all treasure is silver and gold, mate.",
            "Why is the rum always gone?",
            "This is the day you will always remember as the day you almost caught Captain Jack Sparrow!",
        ]
        
        # Initialize and train tokenizer
        tokenizer = MiniLlamaTokenizer(texts, iterations=1000)
        tokenizer.train()
        
        # Test encoding/decoding
        test_text = "Why is the rum gone?"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original: {test_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        
        return tokenizer


if __name__ == "__main__":
    
    # tokenizer = MiniLlamaTokenizer.example_usage()
    
    # print("-" * 50)
    # print("Beginning saving and loading: \n")
    
    # tokenizer.save("tokenizer.json")
    
    tokenizer = MiniLlamaTokenizer.load(path="tokenizer.json")
    print(tokenizer.string_to_tokens.__len__())
    
    encoded_tokens = tokenizer.encode("Why is the rum gone?")
    print(f"Encoded tokens: {encoded_tokens}")
    
    decoded_string = tokenizer.decode(encoded_tokens)
    print(f"Decoded string: {decoded_string}")