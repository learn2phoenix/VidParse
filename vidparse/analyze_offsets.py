import pandas as pd
import numpy as np
import os
import json
import argparse
import sys
from collections import defaultdict
from typing import Dict, Any, Set, List
from scipy.stats import weibull_min

# --- Helper class from analyze_durations.py ---
class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Helper functions from analyze_durations.py ---
def load_training_list(training_video_list_path: str, recipe: str) -> Set[str]:
    """
    Loads the list of training video filenames from a .txt file.
    """
    video_filenames = set()
    try:
        with open(training_video_list_path, 'r') as f:
            video_filenames = {line.strip() for line in f if line.strip() and recipe in line}
        print(f"Loaded {len(video_filenames)} training video names.")
    except FileNotFoundError:
        print(f"Error: Training video list not found at {training_video_list_path}")
    return video_filenames

def load_recipe_segments(annotation_path: str, recipe_name: str) -> List[Dict[str, Any]]:
    """
    Loads the main annotation file and returns the list of segments
    for the specified recipe.
    """
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return []
    
    try:
        with open(annotation_path, 'r') as f:
            all_metadata = json.load(f)
        
        if recipe_name not in all_metadata:
            print(f"Error: Recipe '{recipe_name}' not found in annotation file.")
            return []
            
        recipe_data = all_metadata[recipe_name].get('segments', [])
        print(f"Found {len(recipe_data)} total video segments for recipe '{recipe_name}'.")
        return recipe_data
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing annotation file: {e}")
        return []

# --- Main logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze start-to-start temporal offsets between action pairs."
    )
    parser.add_argument(
        '--annotation_file', 
        type=str, 
        required=True,
        help="Path to the main annotation JSON file (e.g., unified_metadata.json)."
    )
    parser.add_argument(
        '--recipe', 
        type=str, 
        required=True,
        help="The specific recipe name to process."
    )
    parser.add_argument(
        '--training_video_list', 
        type=str, 
        required=True,
        help="Path to a .txt file containing training video filenames."
    )
    parser.add_argument(
        '--output_stats_json', 
        type=str, 
        required=True,
        help="Path to save the temporal offset statistics as a JSON file."
    )
    parser.add_argument(
        '--ignore_action',
        type=str,
        default=None,
        help="An action label (e.g., 'BG') to completely remove from analysis, "
             "readjusting subsequent timestamps as if it were edited out."
    )
    
    args = parser.parse_args()

    if args.ignore_action:
        print(f"--- Processing Recipe: {args.recipe} (IGNORING action: '{args.ignore_action}') ---")
    else:
        print(f"--- Processing Recipe: {args.recipe} ---")
    
    training_set = load_training_list(args.training_video_list, args.recipe)
    if not training_set: sys.exit(1)
    
    recipe_segments = load_recipe_segments(args.annotation_file, args.recipe)
    if not recipe_segments: sys.exit(1)

    print("Collecting temporal offsets from training videos...")
    
    # Structure: all_offsets[action_i_name][action_j_name] = [list of offsets]
    all_offsets = defaultdict(lambda: defaultdict(list))
    videos_processed_count = 0
    
    for video_meta in recipe_segments:
        video_id = video_meta.get('video_id')
        if not video_id or video_id not in training_set:
            continue
        videos_processed_count += 1
        
        try:
            timestamps = video_meta['labels']['time_stamp']
            action_names = video_meta['labels']['error_description']
            
            # --- START: MODIFIED TIMELINE-ADJUSTING LOGIC ---
            # This logic builds a new, simulated list of actions
            # by removing the 'ignored' action and shifting all
            # subsequent timestamps left by the duration of the
            # removed segments.
            
            video_actions = []
            total_duration_removed_s = 0.0

            for action_name, time_pair in zip(action_names, timestamps):
                start_time = time_pair[0]
                end_time = time_pair[1]
                duration = end_time - start_time
                
                if duration <= 0: # Skip invalid, zero-length segments
                    continue

                # Check if this is the action to ignore
                if args.ignore_action and action_name == args.ignore_action:
                    # 1. Add its duration to the total amount to be removed
                    total_duration_removed_s += duration
                    # 2. Do NOT add this action to the list (it's "edited out")
                    continue
                
                # If we're here, it's an action we want to keep.
                # We must adjust its start time based on all the
                # "ignored" segments we've seen *so far*.
                adjusted_start_time = start_time - total_duration_removed_s

                video_actions.append({
                    'name': action_name,
                    'start': adjusted_start_time # Use the new, adjusted time
                })
            
            # --- END: MODIFIED TIMELINE-ADJUSTING LOGIC ---


            # 2. "FIRST-OCCURRENCE" LOGIC (This part is unchanged)
            # This logic now runs on the 'video_actions' list which
            # has had ignored actions removed and timestamps adjusted.
            for i in range(len(video_actions)):
                action_i = video_actions[i]
                action_i_name = action_i['name']
                
                seen_subsequent_actions = set()

                for j in range(i + 1, len(video_actions)):
                    action_j = video_actions[j]
                    action_j_name = action_j['name']
                    
                    if action_j_name not in seen_subsequent_actions:
                        offset_s = action_j['start'] - action_i['start']
                        
                        if offset_s > 0:
                            all_offsets[action_i_name][action_j_name].append(offset_s)
                        
                        seen_subsequent_actions.add(action_j_name)
                    
        except KeyError:
            print(f"Warning: Skipping video '{video_id}' due to malformed metadata.")
    
    print(f"Processed {videos_processed_count} training videos.")

    # 3. Calculate statistics for all collected offsets
    print("\nCalculating statistics...")
    final_stats = defaultdict(dict)
    
    for action_i_name, partners in all_offsets.items():
        for action_j_name, offsets_list in partners.items():
            if not offsets_list: continue
            offsets_np = np.array(offsets_list)

            dist_params = None
            # Need at least 3-4 samples to get a decent fit
            if len(offsets_np) > 3:
                try:
                    # Fit a Weibull distribution
                    # floc=0 forces the distribution to start at 0 (durations can't be negative)
                    shape, loc, scale = weibull_min.fit(offsets_np, floc=0)
                    dist_params = {
                        'dist': 'weibull_min',
                        'shape': shape,
                        'loc': loc,
                        'scale': scale
                    }
                    print(f"  - ActionPair: '{action_i_name},{action_j_name}' (count={len(offsets_np)}) ... successfully fit Weibull dist.")
                except Exception as e:
                    print(f"  - ActionPair: '{action_i_name},{action_j_name}' (count={len(offsets_np)}) ... Weibull fit failed: {e}")
            else:
                    print(f"  - ActionPair: '{action_i_name},{action_j_name}' (count={len(offsets_np)}) ... skipping distribution fit (not enough data).")

            stats = {
                'mean_s': offsets_np.mean(), 'std_s': offsets_np.std(),
                'min_s': offsets_np.min(), 'max_s': offsets_np.max(),
                '25_percentile_s': np.percentile(offsets_np, 25),
                '50_percentile_s (median)': np.percentile(offsets_np, 50),
                '75_percentile_s': np.percentile(offsets_np, 75),
                'count': int(len(offsets_np)),
                'dist_fit': dist_params
            }
            final_stats[action_i_name][action_j_name] = stats

    print(f"Calculated offset statistics for {len(final_stats)} root actions.")

    # 4. Save the final stats to the output JSON
    if final_stats:
        try:
            output_dir = os.path.dirname(args.output_stats_json)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            with open(args.output_stats_json, 'w') as f:
                json.dump(final_stats, f, indent=2, cls=NpEncoder)
            print(f"\n✅ Successfully saved temporal offset statistics to: {args.output_stats_json}")
        except Exception as e:
            print(f"\n❌ Error saving stats to JSON: {e}")
    else:
        print("No offset statistics were generated.")

    print("\n--- Done ---")

if __name__ == "__main__":
    main()