import json
from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
from typing import List, Dict
from tqdm import tqdm
import random

def load_stories(path: str) -> Dict[str, List[str]]:
    """Load stories from JSONL file
    
    Args:
        path (str): Path to the JSONL file
        
    Returns:
        Dict[str, List[str]]: Dictionary containing original and pirate stories
    """
    stories = {"original": [], "pirate": []}
    
    print("Loading stories from JSONL...")
    with open(path, 'r') as file:
        for line in tqdm(file, desc='🏴‍☠️ Loading stories'):
            story = json.loads(line)
            stories["original"].append(story['original'])
            stories["pirate"].append(story['pirate'])
    
    return stories

def run_tokenizer_tests(
    tokenizer_path: str = "/Users/sabarwal/work/projects/mini-llama-3/model/src/tokenizers/pirate_tokenizer_8K.json",
    stories_path: str = "/Users/sabarwal/work/projects/mini-llama-3/dataset/pirate_stories_train.jsonl"
) -> None:
    """Run comprehensive tests on our tokenizer using raw story data
    
    Args:
        tokenizer_path (str): Path to the trained tokenizer
        stories_path (str): Path to JSONL file with stories
    """
    print("\n🏴‍☠️ Starting tokenizer tests...\n")
    
    # Load the trained tokenizer
    print("Loading tokenizer...")
    tokenizer = MiniLlamaTokenizer.load(tokenizer_path)
    
    # Load story data
    print("Loading stories...")
    stories = load_stories(stories_path)
    
    def test_basic_tokenization():
        """Test basic tokenization functionality"""
        print("\n1. Testing basic tokenization...")
        
        # Test both original and pirate stories
        for story_type in ['original', 'pirate']:
            print(f"\nTesting {story_type} stories:")
            
            # Take random samples
            test_stories = random.sample(stories[story_type], 5)
            
            for i, story in enumerate(test_stories):
                
                story = story.replace('\n', '')
                
                # Encode and decode
                tokens = tokenizer.encode(story, max_length=512)
                decoded_text = tokenizer.decode(tokens)
                
                # Compare original and decoded text
                if story.strip() == decoded_text.strip():
                    print(f"✅ Story {i+1} preserved")
                else:
                    print(f"❌ Story {i+1} mismatch!")
                    print(f"Original: {story[:1000]}...")
                    print(f"Decoded:  {decoded_text[:1000]}...")
                    
                # Check token length
                if len(tokens) == 512:
                    print(f"✅ Token length correct (512)")
                else:
                    print(f"❌ Token length incorrect: {len(tokens)}")
    
    def test_length_constraints():
        """Test various length constraints"""
        print("\n2. Testing length constraints...")
        
        test_lengths = [32, 64, 128, 256, 512]
        test_story = random.choice(stories['pirate'])
        
        for length in test_lengths:
            tokens = tokenizer.encode(test_story, max_length=length)
            
            if len(tokens) != length:
                print(f"❌ Length mismatch for max_length={length}:")
                print(f"Expected: {length}, Got: {len(tokens)}")
            else:
                print(f"✅ Length constraint satisfied for max_length={length}")
                
                # Additional check: decode and ensure story is still readable
                decoded = tokenizer.decode(tokens)
                if len(decoded.strip()) > 0:
                    print(f"  ✅ Decoded text is non-empty")
                else:
                    print(f"  ❌ Decoded text is empty")
    
    def test_special_tokens():
        """Test special tokens handling"""
        print("\n3. Testing special tokens...")
        
        # Test with both original and pirate stories
        for story_type in ['original', 'pirate']:
            sample_text = random.choice(stories[story_type])
            tokens = tokenizer.encode(sample_text, max_length=512)
            
            # Check BOS token
            bos_token = tokens[0]
            bos_expected = tokenizer.encode("<begin_of_sentence>", max_length=512)[0]
            if bos_token == bos_expected:
                print(f"✅ Begin of sentence token correct for {story_type}")
            else:
                print(f"❌ Begin of sentence token incorrect for {story_type}")
            
            # Check EOS token
            eos_token = tokens[-1]
            eos_expected = tokenizer.encode("<end_of_sentence>", max_length=512)[0]
            if eos_token == eos_expected:
                print(f"✅ End of sentence token correct for {story_type}")
            else:
                print(f"❌ End of sentence token incorrect for {story_type}")
            
            # Check padding tokens if any
            padding_token = tokenizer.encode("<padding>", max_length=512)[0]
            if padding_token in tokens:
                pad_positions = [i for i, t in enumerate(tokens) if t == padding_token]
                if all(pos > tokens.index(eos_expected) for pos in pad_positions):
                    print(f"✅ Padding tokens correctly placed for {story_type}")
                else:
                    print(f"❌ Padding tokens incorrectly placed for {story_type}")
    
    def test_unknown_tokens():
        """Test handling of unknown tokens"""
        print("\n4. Testing unknown token handling...")
        
        test_cases = [
            "こんにちは世界",  # Japanese
            "🏴‍☠️👋👾",       # Emojis
            "¯\_(ツ)_/¯",    # ASCII art
            "√∆∂∫",         # Math symbols
            # Add some mixed cases
            "Arrr こんにちは matey!",
            "Shiver me 🏴‍☠️ timbers",
            "¯\_(ツ)_/¯ walked the plank"
        ]
        
        unknown_token_id = tokenizer.encode("<unknown>", max_length=512)[0]
        
        for test_text in test_cases:
            tokens = tokenizer.encode(test_text, max_length=512)
            decoded = tokenizer.decode(tokens)
            
            print(f"\nTesting: {test_text}")
            if unknown_token_id in tokens:
                print(f"✅ Unknown token present")
                print(f"Decoded: {decoded}")
            else:
                print(f"❌ No unknown token used")
                print(f"Decoded: {decoded}")
    
    def test_pirate_specific():
        """Test pirate-specific tokenization"""
        print("\n5. Testing pirate-specific tokenization...")
        
        # Extract some pirate-specific phrases from our dataset
        pirate_stories = stories['pirate'][:100]  # Take first 100 stories
        common_phrases = [
            "Arrr",
            "matey",
            "Yo ho ho",
            "walk the plank",
            "shiver me timbers",
            "pieces of eight",
            "sailed the seven seas",
            "treasure"
        ]
        
        for phrase in common_phrases:
            # Find a story containing this phrase if possible
            matching_stories = [s for s in pirate_stories if phrase.lower() in s.lower()]
            
            if matching_stories:
                test_story = random.choice(matching_stories)
                tokens = tokenizer.encode(test_story, max_length=512)
                decoded = tokenizer.decode(tokens)
                
                if phrase.lower() in decoded.lower():
                    print(f"✅ Successfully preserved: {phrase}")
                else:
                    print(f"❌ Failed to preserve: {phrase}")
                    print(f"Original story: {test_story[:100]}...")
                    print(f"Decoded story: {decoded[:100]}...")
            else:
                print(f"⚠️ No test story found containing: {phrase}")
    
    def test_robustness():
        """Test robustness with edge cases"""
        print("\n6. Testing robustness with edge cases...")
        
        edge_cases = {
            "empty": "",
            "spaces": "      ",
            "very_long": "a" * 1000,
            "repeated_puncts": "!?!?!?!?!",
            "mixed_spaces": "\n\t  \r\n  \t",
            "mixed_content": "Arr! 🏴‍☠️ The こんにちは crew found √∆ treasure!",
            "repeated_words": "matey matey matey matey",
            "special_chars_only": "!@#$%^&*()",
        }
        
        for case_name, text in edge_cases.items():
            try:
                tokens = tokenizer.encode(text, max_length=512)
                decoded = tokenizer.decode(tokens)
                print(f"\nTesting {case_name}:")
                print(f"✅ Successfully encoded and decoded")
                print(f"Original: {text[:50]}")
                print(f"Decoded:  {decoded[:50]}")
                print(f"Token length: {len(tokens)}")
            except Exception as e:
                print(f"❌ Failed on {case_name}: {str(e)}")
    
    def test_consistency():
        """Test tokenization consistency"""
        print("\n7. Testing tokenization consistency...")
        
        test_story = random.choice(stories['pirate'])
        
        # Test multiple encodings of the same text
        print("\nTesting multiple encodings of same text:")
        encodings = [tokenizer.encode(test_story, max_length=512) for _ in range(5)]
        
        if all(encoding == encodings[0] for encoding in encodings):
            print("✅ All encodings consistent")
        else:
            print("❌ Inconsistent encodings detected")
        
        # Test with slight modifications
        print("\nTesting with whitespace variations:")
        variations = [
            test_story,
            test_story + "  ",
            "  " + test_story,
            test_story.replace("  ", " ")
        ]
        
        decoded_variations = [
            tokenizer.decode(tokenizer.encode(v, max_length=512))
            for v in variations
        ]
        
        if all(d.strip() == decoded_variations[0].strip() for d in decoded_variations):
            print("✅ Consistent handling of whitespace variations")
        else:
            print("❌ Inconsistent handling of whitespace variations")
    
    # Run all tests
    test_basic_tokenization()
    test_length_constraints()
    test_special_tokens()
    test_unknown_tokens()
    test_pirate_specific()
    test_robustness()
    test_consistency()
    
    print("\n🏴‍☠️ All tests completed! Arr!")

if __name__ == "__main__":
    run_tokenizer_tests()