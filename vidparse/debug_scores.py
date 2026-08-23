import numpy as np
import pandas as pd
import argparse
import os
import json
from multiprocessing import Pool, cpu_count

# --- Import functions directly from your existing matching.py script ---
# This ensures we use the exact same logic for segmentation and cost calculation.
from matching import segment_test_video, calculate_min_dtw_cost
from segments import load_annotations_and_get_metadata
from feat_comparison import align_videos_with_provided_code


def calculate_aggregate_score(
    segment_features: np.ndarray,
    action_prototypes: list,
    drop_penalty: float,
    n: int,
    metric: str = 'cosine'
) -> float:
    """
    Calculates DTW scores for a segment against all prototypes of an action,
    and returns the average of the top n lowest scores. If n=1, this is
    equivalent to finding the minimum score.
    """
    if not action_prototypes:
        return float('inf')

    all_scores = []
    for proto_features in action_prototypes:
        if proto_features.ndim == 1:
            proto_features = proto_features.reshape(-1, 1)
        result = align_videos_with_provided_code(proto_features, segment_features, drop_penalty, metric=metric)
        score = result.get('normalized_score', float('inf'))
        all_scores.append(score)

    if not all_scores or not np.isfinite(all_scores).any():
        return float('inf')

    all_scores.sort()
    
    # Take the top n scores, or all if fewer than n are available
    top_n_scores = all_scores[:n]
    
    # Return the average
    return np.mean(top_n_scores)


# This is the worker function for parallel processing.
def _worker_calculate_all_scores(args):
    """
    Worker function to compute the DTW score for one micro-segment against one action.
    """
    micro_segment, action_name, prototypes_for_action, drop_penalty, avg_top_n = args
    score = calculate_aggregate_score(micro_segment, prototypes_for_action, drop_penalty, avg_top_n)
    return action_name, score

def generate_debug_scores_csv(
    feature_file: str,
    prototypes_file: str,
    output_csv_path: str,
    segment_duration: float,
    fps: float,
    drop_penalty: float,
    avg_top_n: int = 1,
    use_gt_segments: bool = False,
    annotation_file: str = None,
    recipe: str = None
):
    """
    Generates a CSV file with DTW scores for each micro-segment against all actions.
    """
    print("--- Loading Files ---")
    try:
        test_features = np.load(feature_file)
        prototypes = np.load(prototypes_file, allow_pickle=True).item()
        print(f"Loaded feature file with shape: {test_features.shape}")
        print(f"Loaded {len(prototypes)} actions from prototypes file.")

        first_action = next(iter(prototypes))
        if prototypes[first_action]:
            first_proto_shape = prototypes[first_action][0].shape
            print(f"  -> Sample prototype shape for action '{first_action}' is {first_proto_shape}.")
            if first_proto_shape[0] < fps * 5: # Heuristic: if shorter than 5s, likely micro
                 print("  -> These appear to be micro-prototypes.")
            else:
                 print("  -> These appear to be full-length prototypes.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    segments_to_process = []
    if use_gt_segments:
        print("\n--- Using Ground-Truth Segmentation ---")
        video_id = os.path.splitext(os.path.basename(feature_file))[0]
        metadata = load_annotations_and_get_metadata(annotation_file, video_id, recipe)
        if not metadata:
            print(f"Error: Could not load metadata for video '{video_id}'. Aborting.")
            return
            
        try:
            timestamps = metadata['labels']['time_stamp']
            action_names = metadata['labels']['error_description']
            total_frames = test_features.shape[0]

            for i, (time_pair, gt_label) in enumerate(zip(timestamps, action_names)):
                start_time, end_time = time_pair[0], time_pair[1]
                start_frame = int(np.floor(start_time * fps))
                end_frame = int(np.ceil(end_time * fps))
                
                start_frame = max(0, start_frame)
                end_frame = min(total_frames, end_frame)

                if start_frame < end_frame:
                    feature_segment = test_features[start_frame:end_frame, :]
                    segments_to_process.append({
                        "index": i,
                        "features": feature_segment,
                        "gt_label": gt_label,
                        "start_time": start_time,
                        "end_time": end_time
                    })
            print(f"Loaded {len(segments_to_process)} ground-truth segments.")
        except KeyError:
            print("Error: Annotation file is missing required 'labels', 'time_stamp', or 'error_description' keys.")
            return
    else:
        print("\n--- Using Fixed-Duration Micro-Segmentation ---")
        micro_segments = segment_test_video(test_features, segment_duration, fps)
        for i, segment_features in enumerate(micro_segments):
            segments_to_process.append({
                "index": i,
                "features": segment_features,
                "gt_label": "N/A", # No GT label for fixed segments
                "start_time": i * segment_duration,
                "end_time": (i + 1) * segment_duration
            })
    
    all_action_names = sorted(prototypes.keys())
    results_list = []

    print(f"\n--- Calculating Scores for {len(segments_to_process)} Segments ---")

    # 2. Iterate through each micro-segment and calculate scores for all actions
    for i, segment_info in enumerate(segments_to_process):
        print(f"  Processing segment {i+1}/{len(segments_to_process)} (GT: {segment_info['gt_label']})...")
        
        tasks = []
        for action_name in all_action_names:
            task_args = (segment_info['features'], action_name, prototypes.get(action_name, []), drop_penalty, avg_top_n)
            tasks.append(task_args)
            
        # Parallelize the score calculation for the current segment
        scores_for_segment = {}
        if tasks:
            num_workers = min(cpu_count(), len(tasks))
            with Pool(processes=num_workers) as pool:
                action_score_pairs = pool.map(_worker_calculate_all_scores, tasks)
                scores_for_segment = dict(action_score_pairs)

        # 3. Store the results for this segment
        row = {
            "segment_index": segment_info['index'],
            # Add the ground-truth label to the output row.
            "gt_label": segment_info['gt_label'],
            "start_time_s": f"{segment_info['start_time']:.2f}",
            "end_time_s": f"{segment_info['end_time']:.2f}",
        }
        for action_name in all_action_names:
            row[action_name] = scores_for_segment.get(action_name, float('inf'))
            
        results_list.append(row)
    # 4. Convert to a pandas DataFrame and save as CSV
    if results_list:
        print("\n--- Generating CSV Output ---")
        df = pd.DataFrame(results_list)
        
        # Reorder columns to include the new gt_label column.
        cols = ["segment_index", "gt_label", "start_time_s", "end_time_s"] + all_action_names
        df = df[cols]
        
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Successfully saved debug scores to: {output_csv_path}")
        
        print("\n--- Score Matrix (Snippet) ---")
        print(df.head())
    else:
        print("No results were generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a CSV of DTW scores for each video segment against all action prototypes for debugging."
    )
    parser.add_argument('--feature_file', type=str, required=True, help="Path to the test video's .npy feature file.")
    parser.add_argument('--prototypes_file', type=str, required=True, help="Path to the .npy file containing action prototypes.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the output debug CSV file.")
    
    # Parameters should match matching.py for consistency
    parser.add_argument('--drop_penalty', type=float, default=0.5, help="Drop penalty for the underlying DTW.")
    parser.add_argument('--fps', type=float, default=10.0, help="FPS at which features were extracted.")

    parser.add_argument('--use_gt_segments', action='store_true', help="If set, use ground-truth segments from annotations instead of fixed-duration ones.")
    parser.add_argument('--segment_duration', type=float, default=2.0, help="Duration of each micro-segment in seconds (only used if --use_gt_segments is NOT set).")
    parser.add_argument('--annotation_file', type=str, help="Path to the annotation JSON file (REQUIRED if --use_gt_segments is set).")
    parser.add_argument('--recipe', type=str, help="Name of the recipe (REQUIRED if --use_gt_segments is set).")
    
    parser.add_argument('--avg_top_n', type=int, default=1, help="Average the top N lowest scores instead of taking the single minimum. Default is 1 (equivalent to minimum).")

    args = parser.parse_args()
    if args.use_gt_segments and (not args.annotation_file or not args.recipe):
        parser.error("--annotation_file and --recipe are required when --use_gt_segments is set.")

    generate_debug_scores_csv(
        feature_file=args.feature_file,
        prototypes_file=args.prototypes_file,
        output_csv_path=args.output_csv,
        fps=args.fps,
        drop_penalty=args.drop_penalty,
        use_gt_segments=args.use_gt_segments,
        segment_duration=args.segment_duration,
        annotation_file=args.annotation_file,
        recipe=args.recipe,
        avg_top_n=args.avg_top_n
    )
