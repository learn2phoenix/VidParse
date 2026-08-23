import json
import os
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Any

# This will act as our in-memory cache, just like the global variable in app.py
_annotation_cache: Dict[str, Dict[str, Any]] = {}

def load_annotations_and_get_metadata(
    annotation_path: str, 
    video_filename: str,
    recipe_name: str
) -> Dict[str, Any] | None:
    """
    Loads a main annotation file (if not already cached) and retrieves the
    metadata for a specific video.

    This function mimics the efficient loading strategy of app.py by converting
    the annotation list to a dictionary keyed by filename for fast lookups.

    Args:
        annotation_path (str): The file path to the main annotation JSON file
                               (e.g., 'unified_metadata.json').
        video_filename (str): The filename of the video to look up (e.g., 'P01_12.mp4').

    Returns:
        Dict[str, Any] | None: A dictionary containing the video's metadata,
                                or None if the video is not found.
    """
    global _annotation_cache
    
    # --- Step 1: Load and Cache the annotation file (only runs once) ---
    if not _annotation_cache:
        print(f"Cache is empty. Loading and processing annotation file: {annotation_path}...")
        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file not found at {annotation_path}")
            return None
        
        try:
            with open(annotation_path, 'r') as f:
                metadata_list = json.load(f)

            # Convert list to a dictionary keyed by filename for fast lookup
            _annotation_cache = {entry['video_id']: entry for entry in metadata_list[recipe_name]['segments']}
            print(f"Successfully loaded and cached metadata for {len(_annotation_cache)} videos.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing annotation file: {e}")
            _annotation_cache = {} # Prevent retrying a bad file
            return None

    # --- Step 2: Fetch the metadata for the requested video ---
    return _annotation_cache.get(video_filename)

# (This is the function from the previous step, unchanged)
def segment_features_by_action(
    video_features: np.ndarray, 
    video_metadata: Dict[str, Any], 
    video_fps: float = 10.0
) -> Dict[str, List[np.ndarray]]:
    """
    Splits a video's frame-wise features into segments based on sub-action boundaries,
    parsing the new metadata structure.

    Args:
        video_features (np.ndarray): An (M, D) numpy array containing the features
                                     for all M frames in the video.
        video_metadata (dict): The metadata dictionary for the video, which must
                               contain a 'labels' key with 'time_stamp' and
                               'error_description' lists.
        video_fps (float): The frame rate at which the features were extracted.

    Returns:
        Dict[str, List[np.ndarray]]: A dictionary where keys are the sub-action
                                     names and values are a list of feature
                                     matrices for each occurrence.
    """
    segmented_features = defaultdict(list)
    
    # --- MODIFIED PART: Access the new data structure ---
    try:
        timestamps = video_metadata['labels']['time_stamp']
        action_names = video_metadata['labels']['error_description']
    except KeyError:
        print("Warning: Metadata is missing 'labels', 'time_stamp', or 'error_description'. Returning empty dict.")
        return dict(segmented_features)
        
    if len(timestamps) != len(action_names):
        print("Warning: Mismatch between length of timestamps and action names. Aborting.")
        return dict(segmented_features)
    # --- END MODIFICATION ---
        
    total_frames = video_features.shape[0]
    
    # Iterate through the parallel lists using an index
    for i in range(len(timestamps)):
        action_name = action_names[i]
        time_pair = timestamps[i]
        start_time, end_time = time_pair[0], time_pair[1]
        
        # The rest of the logic is the same as before
        start_frame = int(np.floor(start_time * video_fps))
        end_frame = int(np.ceil(end_time * video_fps))
        
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        if start_frame < end_frame:
            action_feature_segment = video_features[start_frame:end_frame, :]
            segmented_features[action_name].append(action_feature_segment)
            
    return dict(segmented_features)

# ==============================================================================
# Example: Putting It All Together
# ==============================================================================
if __name__ == '__main__':
        
    # --- 1. Set up command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Segment video features by sub-action based on an annotation file."
    )
    parser.add_argument(
        '--annotation_file', 
        type=str, 
        required=True, 
        help="Path to the main annotation JSON file (e.g., unified_metadata.json)."
    )
    parser.add_argument(
        '--feature_file', 
        type=str, 
        required=True, 
        help="Path to the video's feature file (e.g., /path/to/video_01.npy)."
    )
    parser.add_argument(
        '--recipe', 
        type=str, 
        required=True, 
        help="Which recipe the video belongs to."
    )
    parser.add_argument(
        '--fps', 
        type=float, 
        default=10.0, 
        help="The FPS at which features were extracted (default: 10.0)."
    )
    args = parser.parse_args()

    # --- 2. Infer video filename from the feature file path ---
    # os.path.basename -> 'video_01.npy'
    base_with_ext = os.path.basename(args.feature_file)
    # os.path.splitext -> ('video_01', '.npy')
    base_name = os.path.splitext(base_with_ext)[0]
    # Reconstruct the filename to match the keys in the annotation file
    video_filename_key = f"{base_name}"

    print(f"Inferred video key: '{video_filename_key}'")

    # --- 3. Run the processing functions ---
    try:
        # a) Load the numpy feature file
        video_features = np.load(args.feature_file)
        print(f"Loaded feature file with shape: {video_features.shape}")

        # b) Fetch the metadata for our target video
        metadata = load_annotations_and_get_metadata(args.annotation_file, video_filename_key, args.recipe)
        if metadata:
            print(f"Successfully fetched metadata for '{video_filename_key}'")
            
            # c) Use the fetched metadata to segment the features
            action_features = segment_features_by_action(
                video_features,
                metadata,
                video_fps=args.fps
            )
            
            # d) Print the results
            print("\n--- Segmentation Results ---")
            if not action_features:
                print("No action segments were found or extracted.")
            else:
                for action_name, features_list in action_features.items():
                    print(f"Action: '{action_name}'")
                    for i, features in enumerate(features_list):
                        print(f"  - Instance {i+1} feature shape: {features.shape}")
        else:
            print(f"Error: Could not find metadata for '{video_filename_key}' in the annotation file.")

    except FileNotFoundError:
        print(f"Error: The feature file was not found at {args.feature_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")