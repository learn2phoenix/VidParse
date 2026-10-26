import os
import json
import re
import argparse
from collections import defaultdict

# --- Configuration ---
FPS = 15.0  # Frames per second for GTEA, as requested

def parse_gtea_file(file_path):
    """
    Parses a single GTEA .txt file and extracts a list of action tuples.
    
    Returns:
        list: A list of (start_frame, end_frame, action_name) tuples.
    """
    
    # Regex to capture <verb><nouns>(start_frame-end_frame)
    # Group 1: verb
    # Group 2: nouns
    # Group 3: start_frame
    # Group 4: end_frame
    action_regex = re.compile(r'<([^>]+)><([^>]+)> \(([0-9]+)-([0-9]+)\).*')
    
    action_tuples = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                match = action_regex.match(line)
                
                if match:
                    verb = match.group(1)
                    nouns = match.group(2)
                    start_frame = int(match.group(3))
                    end_frame = int(match.group(4))
                    
                    action_name = f"{verb} {nouns}"
                    
                    # Handle GTEA's 1-frame overlaps (e.g., end 123, start 122)
                    # We'll fix this properly in the gap-filling logic
                    action_tuples.append((start_frame, end_frame, action_name))
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # Sort by start frame to ensure correct order
    return sorted(action_tuples, key=lambda x: x[0])

def main(args):
    gtea_dir = args.gtea_dir
    output_json = args.output_json

    # --- PASS 1: Collect all actions to build the action maps ---
    
    actions_per_recipe = defaultdict(set)
    # Store raw parsed data to process in Pass 2
    segments_to_process = defaultdict(list)
    
    # Regex to extract recipe name, e.g., "Hotdog" from "S1_Hotdog_C1.txt"
    recipe_regex = re.compile(r'^[A-Za-z0-9]+_([^_]+)_')

    print(f"PASS 1: Scanning directory and parsing actions: {gtea_dir}")
    
    for filename in sorted(os.listdir(gtea_dir)):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(gtea_dir, filename)
        video_id = os.path.splitext(filename)[0]

        recipe_match = recipe_regex.search(filename)
        if not recipe_match:
            print(f"Warning: Could not infer recipe name from {filename}. Skipping.")
            continue
        
        recipe_name = recipe_match.group(1) 
        
        # Parse the file to get raw action tuples
        raw_action_tuples = parse_gtea_file(file_path)
        
        if raw_action_tuples:
            segments_to_process[recipe_name].append((video_id, raw_action_tuples))
            # Add all actions to the set for this recipe
            for _, _, action_name in raw_action_tuples:
                actions_per_recipe[recipe_name].add(action_name)
        else:
            print(f"Skipped {filename} (no actions found or error)")

    if not segments_to_process:
        print("Error: No .txt files were successfully processed. Aborting.")
        return

    # --- Build the master action2idx maps ---
    
    final_json_data = {}
    all_recipe_idx_maps = {}
    all_found_recipes = sorted(list(segments_to_process.keys()))
    
    print(f"\nFound {len(all_found_recipes)} recipes: {all_found_recipes}")

    for recipe_name in all_found_recipes:
        action_2_idx = {"BG": 0}
        for i, action_name in enumerate(sorted(list(actions_per_recipe[recipe_name]))):
            action_2_idx[action_name] = i + 1
        
        all_recipe_idx_maps[recipe_name] = action_2_idx
        
        # Prepare the final JSON structure
        final_json_data[recipe_name] = {
            "action2idx": action_2_idx,
            "segments": []
        }

    # --- PASS 2: Build contiguous timelines with BG gaps ---

    print(f"\nPASS 2: Generating full timelines for all videos...")

    for recipe_name, segments_list in segments_to_process.items():
        action_2_idx = all_recipe_idx_maps[recipe_name] # Get the map for this recipe
        
        for video_id, raw_actions in segments_list:
            
            # These will be the final parallel lists
            action_list_ids = []
            time_stamp_list = []
            error_desc_list = []
            
            current_frame = 0 # Start from frame 0
            
            for (start_frame, end_frame, action_name) in raw_actions:
                
                # 1. Check for a gap between current_frame and this action's start
                if start_frame > current_frame:
                    # Gap detected! Add a "BG" segment
                    bg_start_frame = current_frame
                    bg_end_frame = start_frame - 1
                    
                    action_list_ids.append(0)
                    error_desc_list.append("BG")
                    time_stamp_list.append([
                        round(bg_start_frame / FPS, 1),
                        round(bg_end_frame / FPS, 1)
                    ])
                
                # 2. Handle overlaps (e.g., GTEA's 1-frame overlap)
                if start_frame < current_frame:
                    # This action overlaps with the previous one.
                    # Adjust its start_frame to avoid temporal issues.
                    start_frame = current_frame
                    
                # 3. Process the action (if it's still valid after adjustment)
                if end_frame >= start_frame:
                    action_id = action_2_idx.get(action_name)
                    
                    action_list_ids.append(action_id)
                    error_desc_list.append(action_name)
                    time_stamp_list.append([
                        round(start_frame / FPS, 1),
                        round(end_frame / FPS, 1)
                    ])
                    
                    # 4. Update current_frame to be the *next* frame
                    current_frame = end_frame + 1
                else:
                    # This action was completely swallowed by an overlap
                    print(f"Warning: Action '{action_name}' in {video_id} was skipped due to overlap.")

            # --- End of loop for this video ---
            
            # Create the final segment_data dictionary
            segment_data = {
                "video_id": video_id,
                "labels": {
                    "action": action_list_ids,
                    "time_stamp": time_stamp_list,
                    "error_description": error_desc_list
                    # 'action_type' is intentionally omitted as requested
                }
            }
            final_json_data[recipe_name]["segments"].append(segment_data)
        
        print(f"Finished processing timeline for recipe: '{recipe_name}'")

    # --- Save the final JSON file ---
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, indent=4)
        print(f"\nSuccessfully created EgoPer-compatible JSON at: {output_json}")
        
        # Print a sample for verification
        if all_found_recipes:
            sample_recipe = all_found_recipes[0]
            print("\n--- Sample Output (First Recipe) ---")
            print(f"Recipe: {sample_recipe}")
            print("Action Map (sample):", dict(list(final_json_data[sample_recipe]['action2idx'].items())[:5]), "...")
            print("Segment (sample):")
            print(json.dumps(final_json_data[sample_recipe]['segments'][0], indent=2))
            print("------------------------------------")
        
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GTEA .txt annotations to an EgoPer-style JSON file, "
                    "separated by recipe and including full action/timestamp timelines."
    )
    
    parser.add_argument(
        '--gtea_dir', 
        type=str, 
        required=True, 
        help="Path to the directory containing GTEA .txt annotation files."
    )
    parser.add_argument(
        '--output_json', 
        type=str, 
        required=True, 
        help="Path to save the output EgoPer-style annotation.json file."
    )
    
    args = parser.parse_args()
    main(args)