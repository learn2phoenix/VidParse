import numpy as np
import pandas as pd
import argparse
import os
import cv2
import json
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# --- Import functions from your existing scripts ---
# (Ensure these scripts are in your PYTHONPATH)
from segments import load_annotations_and_get_metadata
from matching import segment_test_video
from prototypes import load_detections_and_generate_mask
# We use the aggregate score function from debug_scores.py
# This function handles the "average top-n" logic per action.
from debug_scores import calculate_aggregate_score 


def _worker_calculate_all_scores(task_args):
    """
    Helper function to unpack arguments for pool.map()
    """
    segment_features, action_name, proto_list, drop_penalty, avg_top_n, current_video_id, metric = task_args
    
    lovo_proto_list = [p['features'] for p in proto_list if p['video_id'] != current_video_id] 
    # This imported function does the heavy lifting
    score = calculate_aggregate_score(
        segment_features,
        lovo_proto_list,
        drop_penalty,
        avg_top_n,
        metric=metric
    )
    return action_name, score


def get_gt_for_segment(start_time, end_time, video_metadata, fps):
    """
    Finds all ground-truth actions that overlap with a given micro-segment.
    Also determines if the segment spans a boundary.
    """
    try:
        timestamps = video_metadata['labels']['time_stamp']
        action_names = video_metadata['labels']['error_description']
    except KeyError:
        print("Warning: Metadata missing 'labels' or 'time_stamp'. Cannot determine GT.")
        return [], False

    overlapping_actions = []
    # Add a small buffer to handle floating point precision
    buffer = 1.0 / (fps * 2) 
    
    for gt_action, (gt_start, gt_end) in zip(action_names, timestamps):
        # Check for overlap: (gt_start < seg_end) and (gt_end > seg_start)
        if (gt_start < (end_time - buffer)) and (gt_end > (start_time + buffer)):
            overlapping_actions.append(gt_action)
            
    # Remove duplicates
    unique_actions = sorted(list(set(overlapping_actions)))
    
    # A boundary exists if there is more than one unique, non-"BG" action
    non_bg_actions = [action for action in unique_actions if action.upper() != 'BG']
    has_boundary = len(non_bg_actions) > 1
    
    return unique_actions, has_boundary


def run_micro_segment_analysis(args, video_features, video_metadata, prototypes, all_action_names, num_workers, current_video_id, hand_mask):
    """
    Runs the Top-K analysis on fixed-duration micro-segments.
    """
    print("\n" + "="*50)
    print("--- Running Analysis: Micro-Segments ---")
    print(f"Output file: {args.output_csv}")
    print("="*50)

    # --- 1. Skip if output already exists ---
    if os.path.exists(args.output_csv):
        print(f"Skipping: Micro-segment file already exists: {args.output_csv}")
        return

    # --- 2. Segment the test video ---
    micro_segments = segment_test_video(
        video_features, args.segment_duration, args.fps
    )
    print(f"Split video into {len(micro_segments)} micro-segments.")

    results_list = []

    # --- 3. Main loop: Process each micro-segment (SERIAL) ---
    for i, segment_features in enumerate(micro_segments):
        start_time = i * args.segment_duration
        end_time = min((i + 1) * args.segment_duration, video_features.shape[0] / args.fps)

        gt_actions, has_boundary = get_gt_for_segment(start_time, end_time, video_metadata, args.fps)
        gt_actions_str = ";".join(gt_actions) if gt_actions else "BG"

        # 1. Align mask to current 10fps segment frames
        m_start = int(i * args.segment_duration * args.fps)
        m_end = int(m_start + len(segment_features))
        segment_mask = hand_mask[m_start:m_end]
        target_len = len(segment_features)
        if len(segment_mask) < target_len:      # short mask -> pad
            segment_mask = np.concatenate(
                [segment_mask, np.zeros(target_len - len(segment_mask), dtype=int)])
        segment_mask = segment_mask[:target_len]
        
        # 2. Filter segment features (Cosine logic)
        masked_feats = segment_features[segment_mask == 1]
        tasks = []
        # 3. Skip scoring if no hands are in the segment
        if masked_feats.size == 0:
            action_scores = {name: float('inf') for name in all_action_names}
        else:
            for action_name in all_action_names:
                # Pass the metric to the worker
                task_args = (masked_feats, action_name, prototypes.get(action_name, []), 
                             args.drop_penalty, args.avg_top_n, current_video_id, args.distance_metric)
                tasks.append(task_args)

        action_scores = {}
        if tasks:
            with Pool(processes=num_workers) as pool:
                action_score_pairs = pool.map(_worker_calculate_all_scores, tasks)
                action_scores = dict(action_score_pairs)

        # 3c. Get Top-K results
        sorted_scores = sorted(action_scores.items(), key=lambda item: item[1])

        # 3d. Store results
        row = {
            "segment_index": i,
            "start_time_s": f"{start_time:.2f}",
            "end_time_s": f"{end_time:.2f}",
            "gt_actions": gt_actions_str,
            "has_boundary": has_boundary
        }
        row.update(action_scores)
        results_list.append(row)

    # --- 4. Save all results to the CSV file ---
    if results_list:
        df = pd.DataFrame(results_list)
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False, float_format='%.4f')
        print(f"Successfully saved micro-segment scores to: {args.output_csv}")
    else:
        print("No micro-segments were processed. No output file created.")


def run_gt_segment_analysis(args, video_features, video_metadata, prototypes, all_action_names, num_workers, current_video_id, hand_mask):
    """
    Runs the Top-K analysis on ground-truth action segments.
    """
    # --- 1. Define and check output path ---
    base_dir = os.path.dirname(args.output_csv)
    base_name = os.path.splitext(os.path.basename(args.output_csv))[0]
    output_csv_gt = os.path.join(base_dir, f"{base_name}_gt_segments.csv")
    
    print("\n" + "="*50)
    print("--- Running Analysis: GT-Segments ---")
    print(f"Output file: {output_csv_gt}")
    print("="*50)

    if os.path.exists(output_csv_gt):
        print(f"Skipping: GT-segment file already exists: {output_csv_gt}")
        return

    # --- 2. Extract GT Segments from Metadata ---
    gt_segments_to_process = []
    try:
        timestamps = video_metadata['labels']['time_stamp']
        action_names = video_metadata['labels']['error_description']
        total_frames = video_features.shape[0]

        for i, (time_pair, gt_label) in enumerate(zip(timestamps, action_names)):
            start_time, end_time = time_pair[0], time_pair[1]
            start_frame = int(np.floor(start_time * args.fps))
            end_frame = int(np.ceil(end_time * args.fps))
            
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)

            if start_frame < end_frame:
                feature_segment = video_features[start_frame:end_frame, :]
                gt_segments_to_process.append({
                    "index": i,
                    "features": feature_segment,
                    "gt_label": gt_label,
                    "start_time": start_time,
                    "end_time": end_time
                })
        print(f"Extracted {len(gt_segments_to_process)} GT segments.")
    except KeyError:
        print("Error: Annotation file missing keys. Cannot extract GT segments.")
        return

    # --- 3. Main loop: Process each GT segment ---
    results_list = []
    for segment_info in gt_segments_to_process:
        segment_features = segment_info['features']
        
        # Apply mask alignment to the GT segment indices
        m_start = int(np.floor(segment_info['start_time'] * args.fps))
        m_end = int(np.ceil(segment_info['end_time'] * args.fps))
        segment_mask = hand_mask[m_start:m_end]
        target_len = len(segment_features)
        if len(segment_mask) < target_len:      # short mask -> pad
            segment_mask = np.concatenate(
                [segment_mask, np.zeros(target_len - len(segment_mask), dtype=int)])
        segment_mask = segment_mask[:target_len]
        
        # Filter features based on HOD reliability
        masked_feats = segment_features[segment_mask == 1]
        tasks = []
        action_scores = {}
        if masked_feats.size == 0:
            action_scores = {name: float('inf') for name in all_action_names}
        else:
            for action_name in all_action_names:
                proto_list = prototypes.get(action_name, [])
                # Ensure distance_metric is passed here as well
                task_args = (masked_feats, action_name, proto_list, args.drop_penalty, args.avg_top_n, current_video_id, args.distance_metric)
                tasks.append(task_args)

            with Pool(processes=num_workers) as pool:
                action_score_pairs = pool.map(_worker_calculate_all_scores, tasks)
                action_scores = dict(action_score_pairs)

        row = {
            "segment_index": segment_info['index'],
            "start_time_s": f"{segment_info['start_time']:.2f}",
            "end_time_s": f"{segment_info['end_time']:.2f}",
            "gt_action": segment_info['gt_label'],
        }
        row.update(action_scores)
        results_list.append(row)

    # --- 4. Save all results to the CSV file ---
    if results_list:
        df = pd.DataFrame(results_list)
        output_dir = os.path.dirname(output_csv_gt)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv_gt, index=False, float_format='%.4f')
        print(f"Successfully saved GT-segment scores to: {output_csv_gt}")
    else:
        print("No GT-segments were processed. No output file created.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate top-k DTW scores for all micro-segments in a video."
    )
    # --- Input/Output Files ---
    parser.add_argument('--video_file', type=str, required=True,
                        help="Path to the test video's .npy feature file.")
    parser.add_argument('--prototypes_file', type=str, required=True,
                        help="Path to the .npy file containing action prototypes.")
    parser.add_argument('--annotation_file', type=str, required=True,
                        help="Path to the main annotation JSON file.")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Path to save the output CSV file for this video.")
    
    # --- Job Parameters ---
    parser.add_argument('--recipe', type=str, required=True,
                        help="The recipe name (e.g., 'tea') for annotation lookup.")
    parser.add_argument('--k', type=int, default=5,
                        help="Number of top-k actions to save.")
    
    # --- DTW/Segment Parameters (should match prototypes.py/matching.py) ---
    parser.add_argument('--avg_top_n', type=int, default=3,
                        help="Average the top N lowest scores per action.")
    parser.add_argument('--segment_duration', type=float, default=3.0,
                        help="Duration of each micro-segment in seconds. "
                             "3.0 is the published setting.")
    parser.add_argument('--drop_penalty', type=float, default=0.5,
                        help="Drop penalty for the underlying DTW.")
    parser.add_argument('--fps', type=float, default=10.0,
                        help="FPS at which features were extracted.")
    
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['dtw', 'cosine'])
    parser.add_argument('--detections_dir', type=str, required=True)
    parser.add_argument('--raw_video_dir', type=str, required=True) # To get original FPS for mask
    parser.add_argument('--conf_thresh', type=float, default=0.75,
                        help="Confidence threshold for hand detection.")
    
    args = parser.parse_args()

    # --- 1. Skip if output already exists (for SLURM reruns) ---
    if os.path.exists(args.output_csv):
        print(f"Skipping: Output file already exists: {args.output_csv}")
        return

    print(f"Processing: {args.video_file}")

    # --- 2. Load all necessary files ---
    try:
        video_features = np.load(args.video_file)
        prototypes = np.load(args.prototypes_file, allow_pickle=True).item()
        
        video_id = os.path.splitext(os.path.basename(args.video_file))[0]
        video_metadata = load_annotations_and_get_metadata(
            args.annotation_file, video_id, args.recipe
        )
        if not video_metadata:
            print(f"Error: Could not load metadata for video '{video_id}'. Aborting.")
            return
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # ... existing loading code ...
    video_id = os.path.splitext(os.path.basename(args.video_file))[0]
    
    # 1. Map FPS for Mask Alignment
    det_fps = 30.0 
    vid_path = os.path.join(args.raw_video_dir, f"{video_id}.mp4")
    if os.path.exists(vid_path):
        cap = cv2.VideoCapture(vid_path)
        det_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    # 2. Load and Aligned 10fps Hand Mask
    det_path = os.path.join(args.detections_dir, video_id, 'detections.json')
    hand_mask = load_detections_and_generate_mask(
        det_path, len(video_features), args.fps, det_fps, args.conf_thresh
    )

    all_action_names = sorted(prototypes.keys())
    # results_list = []
    num_workers = cpu_count()
    print(f"Using {num_workers} workers for parallel action comparison.")

    # --- Run Mode 1: Micro-Segment Analysis ---
    try:
        run_micro_segment_analysis(args, video_features, video_metadata, prototypes, all_action_names, num_workers, video_id, hand_mask)
    except Exception as e:
        print(f"Error during micro-segment analysis: {e}")

    # --- Run Mode 2: GT-Segment Analysis ---
    try:
        run_gt_segment_analysis(args, video_features, video_metadata, prototypes, all_action_names, num_workers, video_id, hand_mask)
    except Exception as e:
        print(f"Error during GT-segment analysis: {e}")

    print(f"\n--- Dual Analysis Complete for: {args.video_file} ---")


if __name__ == '__main__':
    main()