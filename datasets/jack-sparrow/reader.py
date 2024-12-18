from screenplay_pdf_to_json import convert # Super helpful free library available online!
from typing import List, Dict

def read_script(script_loc: str) -> List[Dict]:
    """Processes a PDF version of a script and returns all of the scenes in dictionary format

    Args:
        script_loc (str): The absolute location of the script to read and process
    
    Returns:
        List[Dict]: Includes the scene information, context and dialogues
    """
    
    return_list = []
    
    # Reading in the script using the smash cut package
    with open(script_loc, 'rb') as script:
        script_json = convert(script, 0)
    
    for pages in script_json[1:]:
        
        # Going through each scene on each page
        for scenes in pages['content']:
            return_list.append(scenes)
            
    return return_list

def get_character_scenes(script_loc: str, character: str) -> List[Dict]:
    """Given a character name, fetches all the scenes said by the character in the script

    Args:
        script_loc (str): The absolute location of the script to read and process
        character (str): The character whose scenes to isolate and return
        
    Returns:
        List[dict]: A list of scenes involving the character
    """
    
    # Reading in the script
    scenes = read_script(script_loc)
    filtered_scenes = []
    
    for scene in scenes:
        
        # Going through all of the lines and other associated information of the scene
        for scene_content in scene['scene']:
            
            # Seeing if we can find the character in the lines!
            if scene_content['type'] == 'CHARACTER' and scene_content['content']['character'] == character:
                
                filtered_scenes.append(scene)
                break
    
    return filtered_scenes



if __name__ == "__main__":
    
    # Fetching all jack sparrow scenes!
    jack_sparrow_scenes = get_character_scenes("/Users/sabarwal/work/projects/mini-llama-3/jack-sparrow/scripts/dead-mans-chest.pdf", character='JACK')
    
    for idx, scene in enumerate(jack_sparrow_scenes):
        
        print(f"Scene number {idx}: \n\n {scene} \n\n {'-' * 128}")