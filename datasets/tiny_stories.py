from datasets import load_dataset
import arrr
from tqdm.auto import tqdm
import json
from typing import Dict, List
import random

def pirate_speak(text: str) -> str:
    """Enhances pirate translation specifically tuned for children's stories

    Args:
        text (str): The string to pirate-ify!

    Returns:
        str: The rowdier sentence
    """
    
    # We attempt to target a lot of Tiny Stories lingo to be converted to pirate speech :)
    
    pirate_vocab = {
        # Family & People
        'mom': 'dear mother',
        'mother': 'dear mother',
        'dad': 'old salt',
        'father': 'old salt',
        'sister': 'shipmate',
        'brother': 'fellow sailor',
        'baby': 'tiny sailor',
        'child': 'young buccaneer',
        'children': 'young buccaneers',
        'kid': 'cabin boy',
        'kids': 'cabin boys',
        'friend': 'matey',
        'friends': 'mateys',
        'teacher': 'wise old man',
        
        # Common TinyStories activities
        'share': 'divide the spoils',
        'help': 'lend a hand',
        'learn': 'gather wisdom',
        'sleep': 'rest in the hammock',
        'sleeping': "restin\' in the hammock",
        'eat': 'feast',
        'eating': "feastin\'",
        
        # Common animals in stories
        'dog': 'sea dog',
        'cat': "ship\'s cat",
        'bird': 'parrot',
        'fish': 'sea creature',
        'rabbit': 'land lubber',
        'mouse': 'bilge rat',
        
        # Emotions & States
        'happy': 'jolly',
        'sad': 'down in the dumps',
        'angry': 'fierce as a storm',
        'scared': 'shiverin\'',
        'tired': 'beat',
        'excited': 'full of grog',
        
        # Places
        'room': 'cabin',
        
        # Common objects
        'food': 'grub',
        
        # Basic actions
        'walk': 'swagger',
        'walking': "swaggerin\'",
        'run': 'scurry',
        'running': "scurryin\'",
        'look': 'spy',
        'looking': "spyin\'",
        'say': 'declare',
        'said': 'declared',
        'think': 'reckon',
        'thought': 'reckoned',
        'want': 'fancy',
        'wanted': 'fancied',
        
        # Common words in TinyStories
        'little': 'wee',
        'big': 'mighty',
        'small': 'tiny',
        'bad': 'rotten',
        'best': 'finest',
        'favorite': 'treasured',
        'different': 'strange',
        'beautiful': 'fair as calm seas',
        
        # Basic pirate vocabulary
        'yes': 'aye',
        'no': 'nay',
        'my': 'me',
        'you': 'ye',
        'your': 'yer',
        'is': 'be',
    }
    
    # Converting some common suffixes into pirate lingo
    suffix_patterns = {
        'ing': "in\'",
        'ings': "in\'s",
    }
    
    text = arrr.translate(text, word_replacements=pirate_vocab, suffix_replacements=None)
        
    # Adding occasional pirate endings (about 65% chance)
    pirate_endings = [
        " Batten down the hatches!",
        " Splice the mainbrace!",
        " Thar she blows!",
        " Arrr!",
        " Weigh anchor and hoist the mizzen!",
        " Savvy?",
        " Dead men tell no tales!",
        " Cleave him to the brisket!",
        " Blimey!",
        " Blow me down!",
        " Avast ye!",
        " Yo ho ho.",
        " Shiver me timbers!",
        " Blistering barnacles!",
        " Ye floundering nincompoop.",
        " Thundering typhoons!",
        " Sling yer hook!",
        " Yo ho ho!",
        " And that\'s the truth of it!",
        " On me word as a pirate!",
        " Or I\'ll walk the plank!",
        " May the wind be at our backs!"
    ]
    
    # There is a 65%  chance of a pirate ending! :)
    if random.random() < 0.65:
        text = text + random.choice(pirate_endings)
    
    return text


def convert_to_pirate_stories() -> List[Dict[str, str]]:
    """
    Loads the Tiny Stories dataset, converts each story to pirate speak,
    and returns the processed dataset!
    
    Returns:
        List[Dict[str, str]]: List of dictionaries containing the original & pirated versions
    """

    # Loading in the Tiny Stories dataset
    print("Loading Tiny Stories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Initializes a list to store processed pirate stories!
    pirate_stories_train = []
    pirate_stories_validation = []
    
    # Processing the training split
    print("Converting stories to pirate speak...")
    
    for story in tqdm(dataset['train'], desc='🏴‍☠️ Plunderin\' stories!', colour='green'):
        
        try:
            # The arrr library in action :)
            pirate_version = pirate_speak(story['text'])
            
            # Storing both versions of the story
            story_dict = {
                'original': story['text'],
                'pirate': pirate_version
            }
            
            pirate_stories_train.append(story_dict)
        
        # Catch any exceptions if they happen!
        except Exception as e:
            print(f"Error processing story: {str(e)}")
            continue
    
    for story in tqdm(dataset['validation'], desc='🏴‍☠️ Plunderin\' stories!', colour='green'):
        
        try:
            # The arrr library in action :)
            pirate_version = pirate_speak(story['text'])
            
            # Storing both versions of the story
            story_dict = {
                'original': story['text'],
                'pirate': pirate_version
            }
            
            pirate_stories_validation.append(story_dict)
        
        # Catch any exceptions if they happen!
        except Exception as e:
            print(f"Error processing story: {str(e)}")
            continue
    
    
    # Printing out some samples!
    print("\n=== Sample Conversions from Training Set ===")

    samples = random.sample(pirate_stories_train, 5)
    
    for i, sample in enumerate(samples, 1):
    
        print(f"\nSample {i}:")
    
        print(f"Original:\n{sample['original']}")
        print(f"\nPirate Version:\n{sample['pirate']}")
        
        print("\n" + "="*80)
        
        
    return pirate_stories_train, pirate_stories_validation 


def save_stories(stories: List[Dict[str, str]], filename: str = "pirate_stories.jsonl") -> None:
    """
    Saves the processed stories to a JSONL file
    
    Args:
        stories (List[Dict[str, str]]): The processed stories
        filename (str): Name of output file
    """
    
    print(f"Saving {len(stories)} stories to {filename}...")
    
    with open(filename, 'w') as f:
        
        for story in stories:
            f.write(json.dumps(story) + '\n')


if __name__ == "__main__":
    
    # Converting the stories!
    pirate_stories_train, pirate_stories_validation = convert_to_pirate_stories()
    
    # Saving them to disk :)
    save_stories(pirate_stories_train, "pirate_stories_train.jsonl")
    save_stories(pirate_stories_validation, "pirate_stories_validation.jsonl")
