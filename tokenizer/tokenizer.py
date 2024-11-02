from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional
import json

class MiniLlamaTokenizer:
    
    """This is a simplified implementation of the Mini-LLama Tokenizer. I am using Byte-Pair encoding as
    suggested by the sentencepiece paper from Google with iterations=5.
    
    If time permits, I will use concurrency to parallelize the process of counting adjacent tokens as well!
    """
    
    def __init__(self, text_corpus: List[str], iterations: int):
        """ Initializes and trains the tokenizer on the given texts for [iterations] iterations

        Args:
            text_corpus (List[str]): A list of texts
            iterations (int): The number of times to merge the most frequently adjacent tokens
        """

        # Storing all of the important variables
        self.iterations = iterations
        
        # Converts tokens to strings and vice versa!
        self.string_to_tokens: Dict[str, int] = dict()
        self.tokens_to_string: Dict[int, str] = dict()
        
        self.merges: Dict[Tuple[str, str], str] = {}
        self.word_freqs: Dict[str, int] = defaultdict(int)
        
        # Defining some helpful special tokens!
        self.special_tokens = set(["<padding>", "<begin_of_sentence>", "<end_of_sentence>",
            "<unknown>", "<assistant>", "<human>", "</w>"])

        # Initializing the vocabulary with all of basic english characters + </w> special token
        for key in self.special_tokens:
            
            self.string_to_tokens[key] = len(self.string_to_tokens)
            self.tokens_to_string[len(self.tokens_to_string)] = key
            
        for text in text_corpus:
            
            # Splitting the text into individual letter
            words = text.lower().strip().split()
            
            for word in words:
                
                if word + "</w>" in self.string_to_tokens:
                    continue
                
                # Adding a token to identify the end of word
                word = word + "</w>" 
                self.word_freqs[word] += 1
                
                self.string_to_tokens[word] = len(self.string_to_tokens)
                self.tokens_to_string[len(self.tokens_to_string)] = word
                
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
        
        for i in range(self.iterations):

            # Get pair frequencies
            pairs = defaultdict(int)
        
            for word, freq in word_freqs.items():
                
                # Taking everything by each space-separated unit -- initially alphabet, combines into larger tokens :) 
                symbols = word.split()
                
                # Adding up the frequency for all adjacent words
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += freq
                    
            if not pairs:
                break
                
            # Finding the most frequent pair on the basis of the frequencies
            most_freq = max(pairs.items(), key=lambda x: x[1])
            best_pair = most_freq[0]
            
            # Merging the pair in our vocabulary
            word_freqs = self.merge_tokens(best_pair, word_freqs)
            self.merges[best_pair] = ''.join(best_pair)
            
            # Adding in the new token!
            self.string_to_tokens[''.join(best_pair)] = len(self.tokens_to_string)
            self.tokens_to_string[len(self.tokens_to_string)] = ''.join(best_pair)
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges. Vocabulary size: {len(self.string_to_tokens)}")
    
    def encode(self, text: str) -> List[int]:
        """Converts the given text into a series of token IDs

        Args:
            text (str): The text to convert to a list of tokens

        Returns:
            List[int]: The list of tokens!
        """
        
        # Splitting everything into words and adding the BOS and EOS tokens
        words = text.lower().strip().split()
        encoded = []
        
        for word in words:
            
            word = word + "</w>"
            word = ' '.join(list(word))
            
            while True:
                
                # Trying to apply merges
                changed = False
                
                for pair, merge in self.merges.items():
                
                    if ' '.join(pair) in word:
                        
                        word = word.replace(' '.join(pair), merge)
                        changed = True
                        break
                    
                if not changed:
                    break
            
            # Convert to token IDs
            for string in word.split():
                
                if string in self.string_to_tokens:
                    encoded.append(self.string_to_tokens[string])
                
                else:
                    encoded.append(self.string_to_tokens["<unknown>"])
        
        return [self.string_to_tokens['<begin_of_sentence>']] + encoded + [self.string_to_tokens['<end_of_sentence>']]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a string

        Args:
            token_ids (List[int]): The list of token IDs to convert

        Returns:
            str: The decoded string
        """
        
        text = []
        
        for token in token_ids:
                
            if token in self.tokens_to_string:
                text.append(self.tokens_to_string[token])
            
            else:
                text.append("<unknown>")              
        
        text = ''.join(text).replace('</w>', ' ').strip()
        return text
    
    def save(self, path: str) -> None:
        """Save tokenizer vocabulary and merges."""
        save_dict = {
            'vocab': list(self.string_to_tokens),
            'merges': {' '.join(k): v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str, text_corpus: Optional[List[str]] = None) -> 'MiniLlamaTokenizer':
        """Load saved tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = cls(text_corpus or [""], iterations=0)
        tokenizer.vocab = set(data['vocab'])
        tokenizer.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
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
        tokenizer = MiniLlamaTokenizer(texts, iterations=100)
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
    
    tokenizer = MiniLlamaTokenizer.example_usage()
    
    # tokenizer.save("tokenizer.json")
    # tokenizer = MiniLlamaTokenizer.load("tokenizer.json")
    print(tokenizer.decode(tokenizer.encode("Why is the rum gone?")))