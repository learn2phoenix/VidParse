import os
import pickle
import glob
import json
import argparse
import pandas as pd
import numpy as np
import Levenshtein
from collections import defaultdict
from typing import List, Dict, Any

_processed_recipe_cache: Dict[str, Dict[str, Any]] = {}

# --- Try to import openpyxl for Excel writing ---
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: 'openpyxl' library not found. To save .xlsx reports, please install it: pip install openpyxl")

from segmentation_metrics import get_labels_start_end_time, edit_score, f_score, acc_score


def get_video_metadata(
    full_annotation_data: Dict[str, Any], 
    video_filename: str,
    recipe_name: str
) -> Dict[str, Any] | None:
    """
    Gets video metadata from the already-loaded full annotation data,
    using a cache for processed recipe segments.
    """
    global _processed_recipe_cache
    
    # Check if we have already processed and cached this recipe
    if recipe_name not in _processed_recipe_cache:
        # This is the first time we're seeing this recipe.
        recipe_data = full_annotation_data.get(recipe_name)
        
        if not recipe_data or 'segments' not in recipe_data:
            print(f"Warning: Recipe '{recipe_name}' or 'segments' not found in annotation file.")
            _processed_recipe_cache[recipe_name] = {} # Cache the failure
        else:
            # Process the segment list into a fast-lookup dict
            video_dict = {entry['video_id']: entry for entry in recipe_data['segments']}
            _processed_recipe_cache[recipe_name] = video_dict
            print(f"Cached {len(video_dict)} video metadata entries for recipe '{recipe_name}'.")
    
    # Now, return the requested video from the cache
    return _processed_recipe_cache[recipe_name].get(video_filename)


def generate_report_sheet(writer, sheet_name, all_stats, per_video_stats, stats_key, all_recipe_frame_metrics):
    """
    Generates a report (console and Excel) for a specific stats_key ('no_bg' or 'with_bg').
    Reports ONLY Segment-Level Accuracy in summary tables.
    """
    print(f"\n\n{'='*70}")
    print(f"--- GENERATING REPORT: {sheet_name} (Logic: {stats_key}) ---")
    print(f"{'='*70}")
    
    current_row = 0
    all_recipes = sorted(all_stats.keys())
    
    grand_total_stats = defaultdict(lambda: {'correct': 0, 'total_gt': 0, 'total_pred': 0})
    overall_recipe_tally = []

    grand_total_frame_metrics = {
        'Accuracy': [], 'Edit': [], 'F1@10': [], 'F1@25': [], 'F1@50': [], 'P@50': [], 'R@50': []
    }

    for recipe in all_recipes:
        print(f"--- Recipe: {recipe} ---")
        
        recipe_action_stats = all_stats[recipe]
        report_data = []
        
        # --- Get stats for the specified key ('no_bg' or 'with_bg') ---
        # breakpoint()
        valid_actions = [action for action, stats in recipe_action_stats.items() if stats[stats_key]['total_gt'] > 0 or stats[stats_key]['total_pred'] > 0]
        # valid_actions = [action for action in valid_actions if action is not None]
        for action in sorted(valid_actions):
            stats = recipe_action_stats[action][stats_key]
            c = stats['correct']
            t_gt = stats['total_gt']
            t_pred = stats['total_pred']
            
            grand_total_stats[action]['correct'] += c
            grand_total_stats[action]['total_gt'] += t_gt
            grand_total_stats[action]['total_pred'] += t_pred

            precision = (c / t_pred) * 100 if t_pred > 0 else 0
            recall = (c / t_gt) * 100 if t_gt > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report_data.append({
                "Action": action,
                "F1-Score (Seg)": f1, 
                "Precision (Seg)": precision,
                "Recall": recall,
                "Correct": c,
                "Total_GT": t_gt,
                "Total_Pred": t_pred
            })
        
        df = pd.DataFrame(report_data)
        if df.empty:
            print("No stats to report for this recipe.")
            continue
            
        print(df.to_string(index=False, float_format="%.2f"))

        recipe_video_stats = per_video_stats.get(recipe, {})
        recipe_total_seg_correct = sum(v[stats_key]['Correct'] for v in recipe_video_stats.values())
        recipe_total_seg_segments = sum(v['Total'] for v in recipe_video_stats.values())
        overall_recipe_seg_acc = (recipe_total_seg_correct / recipe_total_seg_segments) * 100 if recipe_total_seg_segments > 0 else 0
        
        recipe_summary_str_seg = f"** Overall Recipe Segment-Level Accuracy: {overall_recipe_seg_acc:.2f}% ({recipe_total_seg_correct} / {recipe_total_seg_segments}) **"
        
        print(f"\n{recipe_summary_str_seg}\n")

        recipe_frame_metrics = all_recipe_frame_metrics.get(recipe, {})
        avg_frame_acc = np.mean(list(recipe_frame_metrics.get('Accuracy', {}).values()) or [0])
        avg_frame_edit = np.mean(list(recipe_frame_metrics.get('Edit', {}).values()) or [0])
        avg_frame_f1_10 = np.mean(list(recipe_frame_metrics.get('F1@10', {}).values()) or [0])
        avg_frame_f1_25 = np.mean(list(recipe_frame_metrics.get('F1@25', {}).values()) or [0])
        avg_frame_f1_50 = np.mean(list(recipe_frame_metrics.get('F1@50', {}).values()) or [0])
        
        # Add to grand totals
        grand_total_frame_metrics['Accuracy'].extend(recipe_frame_metrics.get('Accuracy', {}).values())
        grand_total_frame_metrics['Edit'].extend(recipe_frame_metrics.get('Edit', {}).values())
        grand_total_frame_metrics['F1@10'].extend(recipe_frame_metrics.get('F1@10', {}).values())
        grand_total_frame_metrics['F1@25'].extend(recipe_frame_metrics.get('F1@25', {}).values())
        grand_total_frame_metrics['F1@50'].extend(recipe_frame_metrics.get('F1@50', {}).values())
        grand_total_frame_metrics['P@50'].extend(recipe_frame_metrics.get('P@50', {}).values())
        grand_total_frame_metrics['R@50'].extend(recipe_frame_metrics.get('R@50', {}).values())

        recipe_frame_summary_str = (
            f"** Overall Recipe Frame-Level Metrics (MoV):\n"
            f"** Accuracy: {avg_frame_acc:.2f}%\n"
            f"** Edit Score: {avg_frame_edit:.2f}\n"
            f"** F1@.10:    {avg_frame_f1_10:.2f}\n"
            f"** F1@.25:    {avg_frame_f1_25:.2f}\n"
            f"** F1@.50:    {avg_frame_f1_50:.2f}"
        )
        print(f"{recipe_frame_summary_str}\n")
        
        overall_recipe_tally.append({
            "Recipe": recipe,
            "Segment Acc": overall_recipe_seg_acc,
            "Frame Acc": avg_frame_acc,
            "Edit Score": avg_frame_edit,
            "F1@50": avg_frame_f1_50,
            "Correct (Seg)": recipe_total_seg_correct,
            "Total (Seg)": recipe_total_seg_segments
        })

        # --- Write to Excel (Per-Recipe) ---
        if writer:
            pd.DataFrame([f"--- Recipe: {recipe} ---"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
            current_row += 1
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += len(df) + 1 # +1 for header
            pd.DataFrame([recipe_summary_str_seg]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
            current_row += 1 
            for line in recipe_frame_summary_str.split('\n'):
                 pd.DataFrame([line]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
                 current_row += 1
            current_row += 1 


    # --- 6. Report Grand Totals ---
    print("=========================================================")
    print("--- Overall Action Performance (All Recipes) ---")
    
    grand_report_data = []
    valid_grand_actions = [action for action, stats in grand_total_stats.items() if stats['total_gt'] > 0 or stats['total_pred'] > 0]

    for action in sorted(valid_grand_actions):
        stats = grand_total_stats[action]
        c = stats['correct']
        t_gt = stats['total_gt']
        t_pred = stats['total_pred']

        precision = (c / t_pred) * 100 if t_pred > 0 else 0
        recall = (c / t_gt) * 100 if t_gt > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        grand_report_data.append({
            "Action": action,
            "F1-Score (Seg)": f1,
            "Precision (Seg)": precision,
            "Recall (Seg)": recall,
            "Correct": c,
            "Total_GT": t_gt,
            "Total_Pred": t_pred
        })

    df_grand = pd.DataFrame(grand_report_data)
    if not df_grand.empty:
        print(df_grand.to_string(index=False, float_format="%.2f"))
    
    print("\n--- Overall Recipe Performance ---")
    df_recipes = pd.DataFrame(overall_recipe_tally)
    # breakpoint()
    # Re-order columns for clarity
    df_recipes = df_recipes[[
        "Recipe", 
        "Segment Acc", "F1@50", "Frame Acc", "Edit Score",
        "Correct (Seg)", "Total (Seg)"
    ]]
    print(df_recipes.to_string(index=False, float_format="%.2f"))

    print("\n--- CSV for Excel (F1@50, Frame Acc, Edit Score) ---")
    print_str = []
    for row in overall_recipe_tally:
        # Prints: 34.50,67.20,12.50
        print_str.append(f"{row['F1@50']:.2f},{row['Frame Acc']:.2f},{row['Edit Score']:.2f}")
    print(",".join(print_str))

    # --- Grand Total Segment Accuracy (matches recipe totals) ---
    grand_total_seg_correct = sum(s['Correct (Seg)'] for s in overall_recipe_tally)
    grand_total_seg_segments = sum(s['Total (Seg)'] for s in overall_recipe_tally)
    grand_total_seg_acc = (grand_total_seg_correct / grand_total_seg_segments) * 100 if grand_total_seg_segments > 0 else 0
    grand_total_summary_str = f"** Grand Total Segment-Level Accuracy: {grand_total_seg_acc:.2f}% ({grand_total_seg_correct} / {grand_total_seg_segments}) **"
    
    print("\n=========================================================")
    print(grand_total_summary_str)
    
    grand_avg_frame_acc = np.mean(grand_total_frame_metrics.get('Accuracy', [0]))
    grand_avg_frame_edit = np.mean(grand_total_frame_metrics.get('Edit', [0]))
    grand_avg_frame_f1_10 = np.mean(grand_total_frame_metrics.get('F1@10', [0]))
    grand_avg_frame_f1_25 = np.mean(grand_total_frame_metrics.get('F1@25', [0]))
    grand_avg_frame_f1_50 = np.mean(grand_total_frame_metrics.get('F1@50', [0]))
    grand_avg_frame_p_50 = np.mean(grand_total_frame_metrics.get('P@50', [0]))
    grand_avg_frame_r_50 = np.mean(grand_total_frame_metrics.get('R@50', [0]))
    
    grand_frame_summary_str = (
        f"** Grand Total Frame-Level Metrics (MoV):\n"
        f"** Accuracy: {grand_avg_frame_acc:.2f}%\n"
        f"** Edit Score: {grand_avg_frame_edit:.2f}\n"
        f"** F1@.10:    {grand_avg_frame_f1_10:.2f}\n"
        f"** F1@.25:    {grand_avg_frame_f1_25:.2f}\n"
        f"** F1@.50:    {grand_avg_frame_f1_50:.2f}\n"
        f"** P@.50:    {grand_avg_frame_p_50:.2f}\n"
        f"** R@.50:    {grand_avg_frame_r_50:.2f}"
    )
    print(grand_frame_summary_str)

    # --- Write Grand Totals to Excel ---
    if writer:
        pd.DataFrame(["========================================================="]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        pd.DataFrame(["--- Overall Action Performance (All Recipes) ---"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        if not df_grand.empty:
            df_grand.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(df_grand) + 2

        pd.DataFrame(["--- Overall Recipe Performance ---"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        df_recipes.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(df_recipes) + 2

        pd.DataFrame(["========================================================="]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        pd.DataFrame([grand_total_summary_str]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        for line in grand_frame_summary_str.split('\n'):
             pd.DataFrame([line]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
             current_row += 1

def calculate_statistics(
    results_dir: str, 
    video_list_file: str, 
    annotation_file: str, 
    fps: float,
    eval_levels: List[int],
    output_excel_file: str = None,
    ignore_action: str = "BG"
):
    """
    Evaluates Full-Video Matching results.
    
    CHANGES FROM COMPRESSED LOGIC:
    1. We NO LONGER shift timestamps or compress the GT timeline.
       Matching now runs on the full video (including BG segments), so predictions 
       are in real-time coordinates. We must respect the original GT timestamps.
    
    2. We use 'ignore_action' solely for FILTERING during metric calculation.
       If GT at frame t is 'ignore_action' (e.g. BG), we exclude that frame 
       from the Accuracy/F1 calculation.
    """
    
    # --- 1. Read Expected Video List ---
    expected_video_ids = set()
    try:
        with open(video_list_file, 'r') as f:
            for line in f:
                if '/' in line:
                    video_id = line.strip().split('/')[-1].split('.')[0]
                else:
                    video_id = line.strip()
                if video_id:
                    expected_video_ids.add(video_id)
        if not expected_video_ids:
            print("Warning: Video list file is empty or could not be read.")
        else:
            print(f"Loaded {len(expected_video_ids)} expected video IDs from {video_list_file}")
    except Exception as e:
        print(f"Error reading video list file {video_list_file}: {e}")
        return

    # --- 2. Find Processed Result Files ---
    search_path = os.path.join(results_dir, '**', '*_results.pkl')
    pkl_files = glob.glob(search_path, recursive=True)

    if not pkl_files:
        print(f"Error: No '*_results.pkl' files found in {results_dir} or its subdirectories.")
        if expected_video_ids:
            print("\n--- Missing Video Report ---")
            print(f"Found 0 results. All {len(expected_video_ids)} expected videos are missing.")
            for video_id in sorted(list(expected_video_ids)):
                print(f"  - {video_id}")
        return

    print(f"Found {len(pkl_files)} result files. Processing...\n")
    print(f"Targeting IGNORE CLASS: '{ignore_action}' (Frames with this GT label will be skipped in metrics)")

    # --- 3. Initialize Excel Writer (if applicable) ---
    writer = None
    if output_excel_file:
        if OPENPYXL_AVAILABLE:
            try:
                writer = pd.ExcelWriter(output_excel_file, engine='openpyxl')
                print(f"\nWill save Excel report to: {output_excel_file}")
            except Exception as e:
                print(f"\nWarning: Could not initialize Excel writer: {e}")
                writer = None
        else:
            print("\nWarning: Cannot save Excel report. 'openpyxl' is not installed.")

    master_all_stats = {}
    master_per_video_stats = {}
    master_all_recipe_frame_metrics = {}
    
    eval_levels_sorted = sorted(list(set(eval_levels))) 
    print(f"Will evaluate at the following levels: {eval_levels_sorted}%")

    for level in eval_levels_sorted:
        master_all_stats[level] = defaultdict(lambda: defaultdict(lambda: {
            'no_bg': {'correct': 0, 'total_gt': 0, 'total_pred': 0},
            'with_bg': {'correct': 0, 'total_gt': 0, 'total_pred': 0}
        }))
        master_per_video_stats[level] = defaultdict(lambda: defaultdict(dict))
        master_all_recipe_frame_metrics[level] = defaultdict(lambda: defaultdict(dict))


    found_video_ids = set()

    # --- 4. Accumulate Statistics ---
    try:
        with open(annotation_file, 'r') as f:
            full_annotation_data = json.load(f)
        print(f"Loaded full annotation file: {annotation_file}")
    except Exception as e:
        print(f"FATAL: Could not load annotation file {annotation_file} to get action maps. Error: {e}")
        return
    all_inference_times = []
    all_fps_speeds = []

    for pkl_path in pkl_files:
        try:
            recipe = os.path.basename(os.path.dirname(pkl_path))
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            if "inference_time_seconds" in data:
                all_inference_times.append(data["inference_time_seconds"])
                all_fps_speeds.append(data["fps_speed"])
            
            action2id_map = full_annotation_data.get(recipe, {}).get('action2idx')
            if not action2id_map:
                print(f"Warning: Could not find 'action2idx' map for recipe '{recipe}'. Skipping video {pkl_path}.")
                continue
            id2action_map = {v: k for k, v in action2id_map.items()}

            # 1. Determine the video ID
            video_id_from_data = data.get('video_id')
            video_id_from_name = os.path.basename(pkl_path).replace('_results.pkl', '')
            
            video_id = video_id_from_data
            if not video_id:
                video_id = video_id_from_name 
                print(f"Warning: 'video_id' key not in {pkl_path}. Inferred '{video_id}' from filename.")

            # 2. Check if this video is in our expected list
            if video_id not in expected_video_ids:
                continue
            
            # 3. If we are here, the video was expected AND found.
            #    Add it to found_video_ids for the final missing report.
            found_video_ids.add(video_id)

            # --- Continue with processing for this valid video ---
            original_best_path = data['best_hypothesis']['path']
            gt_history = data['gt_history_per_segment']
            original_num_segments = len(original_best_path)

            # Mismatch check is loose now because GT history from matching might include BG, might not, depending on how it was collected.
            # But let's trust the logic.
            
            for level in eval_levels_sorted:
                all_stats = master_all_stats[level]
                per_video_stats = master_per_video_stats[level]
                all_recipe_frame_metrics = master_all_recipe_frame_metrics[level]

                num_segments = int(np.ceil(original_num_segments * (level / 100.0)))
                
                if num_segments == 0: continue
                    
                best_path = original_best_path[:num_segments]
                
                current_video_correct_segments_no_bg = 0
                current_video_correct_segments_with_bg = 0
                total_video_segments = num_segments 

                for i in range(num_segments): 
                    pred_action = best_path[i]
                    raw_gt_actions = set(gt_history[i])

                    # Logic 1: No-BG (Filter out ignore_action)
                    gt_actions_no_bg = raw_gt_actions - {'BG'} # Robustness

                    if gt_actions_no_bg:
                        if pred_action != "BG" and pred_action in gt_actions_no_bg:
                            all_stats[recipe][pred_action]['no_bg']['total_pred'] += 1
                            all_stats[recipe][pred_action]['no_bg']['correct'] += 1
                        elif pred_action != "BG":
                            all_stats[recipe][pred_action]['no_bg']['total_pred'] += 1
                        for gt_action in gt_actions_no_bg:
                            all_stats[recipe][gt_action]['no_bg']['total_gt'] += 1
                    
                    is_correct_segment_no_bg = False
                    if pred_action != "BG" and pred_action in gt_actions_no_bg:
                        is_correct_segment_no_bg = True
                    if is_correct_segment_no_bg:
                        current_video_correct_segments_no_bg += 1

                    # Logic 2: With-BG (Include ignore_action)
                    gt_actions_with_bg = raw_gt_actions.copy()
                    is_correct_segment_with_bg = False
                    
                    if pred_action != "BG" and pred_action in gt_actions_with_bg:
                        is_correct_segment_with_bg = True
                        all_stats[recipe][pred_action]['with_bg']['correct'] += 1
                        all_stats[recipe][pred_action]['with_bg']['total_pred'] += 1
                    elif pred_action == "BG" and (ignore_action in gt_actions_with_bg or 'BG' in gt_actions_with_bg):
                        is_correct_segment_with_bg = True
                        all_stats[recipe]['BG']['with_bg']['correct'] += 1
                        all_stats[recipe]['BG']['with_bg']['total_pred'] += 1
                    elif pred_action == "BG":
                        all_stats[recipe]['BG']['with_bg']['total_pred'] += 1
                    else:
                        all_stats[recipe][pred_action]['with_bg']['total_pred'] += 1

                    if ignore_action in gt_actions_with_bg or 'BG' in gt_actions_with_bg:
                        all_stats[recipe]["BG"]['with_bg']['total_gt'] += 1
                    else:
                        for gt_action in gt_actions_with_bg:
                            all_stats[recipe][gt_action]['with_bg']['total_gt'] += 1
                    
                    if is_correct_segment_with_bg:
                        current_video_correct_segments_with_bg += 1
                
                video_accuracy_no_bg = (current_video_correct_segments_no_bg / total_video_segments) * 100 if total_video_segments > 0 else 0
                video_accuracy_with_bg = (current_video_correct_segments_with_bg / total_video_segments) * 100 if total_video_segments > 0 else 0
                
                per_video_stats[recipe][video_id] = {
                    'no_bg': {'Correct': current_video_correct_segments_no_bg, 'Accuracy': video_accuracy_no_bg},
                    'with_bg': {'Correct': current_video_correct_segments_with_bg, 'Accuracy': video_accuracy_with_bg},
                    'Total': total_video_segments
                }
                
                try:
                    # --- FRAME LEVEL METRICS (FULL VIDEO, NO COMPRESSION) ---
                    video_metadata = get_video_metadata(full_annotation_data, video_id, recipe)
                    if not video_metadata: continue

                    try:
                        gt_timestamps = video_metadata['labels']['time_stamp']
                        gt_action_names = video_metadata['labels']['error_description']
                    except KeyError: continue

                    if not gt_timestamps: continue

                    # 1. Build FULL timeline (No compression/shifting)
                    original_gt_segments = list(zip(gt_action_names, gt_timestamps))
                    
                    # 2. Get total duration
                    if not original_gt_segments: continue
                    max_end_time = max(t[1] for _, t in original_gt_segments)
                    total_frames = int(max_end_time * fps)
                    if total_frames == 0: continue

                    # 3. Build GT frames
                    # Initialize with default "BG" (or ignore_action)
                    gt_frames = [ignore_action if ignore_action else "BG"] * total_frames
                    
                    for action_name, (start_time, end_time) in original_gt_segments:
                        # If the annotation explicitly says BG, we leave it as BG.
                        if action_name == ignore_action or action_name == "BG": 
                            continue

                        start_frame = int(start_time * fps)
                        end_frame = min(int(end_time * fps), total_frames)
                        if start_frame < end_frame:
                            gt_frames[start_frame:end_frame] = [action_name] * (end_frame - start_frame)
                    
                    # 4. Build Pred frames
                    pred_frames = []
                    # CASE A: Variable Duration (New Online Output)
                    if 'path_durations' in data['best_hypothesis']:
                        path_durations = data['best_hypothesis']['path_durations']
                        # Safety check: lengths must match
                        if len(path_durations) != len(best_path):
                            print(f"Warning: Path length ({len(best_path)}) != Durations length ({len(path_durations)}) for {video_id}. Truncating to min.")
                            min_len = min(len(best_path), len(path_durations))
                            best_path = best_path[:min_len]
                            path_durations = path_durations[:min_len]

                        for pred_action, dur_s in zip(best_path, path_durations):
                            lbl = pred_action if pred_action != "BG" else (ignore_action if ignore_action else "BG")
                            # Convert seconds to frames
                            seg_frames = int(round(dur_s * fps))
                            pred_frames.extend([lbl] * seg_frames)

                    # CASE B: Fixed Duration (Old/Offline Output)
                    elif 'segment_duration' in data:
                        seg_dur = data['segment_duration']
                        frames_per_segment = int(seg_dur * fps)
                        for pred_action in best_path: 
                            lbl = pred_action if pred_action != "BG" else (ignore_action if ignore_action else "BG")
                            pred_frames.extend([lbl] * frames_per_segment)
                    
                    # CASE C: Failure
                    else:
                        print(f"Error: No timing info ('segment_duration' or 'path_durations') found for {video_id}. Skipping frame metrics.")
                        continue
                    
                    # 5. Align
                    n_frames = min(len(gt_frames), len(pred_frames))
                    if n_frames == 0: continue
                        
                    gt_frames_aligned = gt_frames[:n_frames]
                    pred_frames_aligned = pred_frames[:n_frames]
                    
                    # 6. Filter (The Metric Logic)
                    gt_frames_filtered = []
                    pred_frames_filtered = []
                    
                    target_ignore = ignore_action if ignore_action else "BG"

                    for k in range(n_frames):
                        # WE ONLY EVALUATE FRAMES WHERE GT IS NOT BG
                        if gt_frames_aligned[k] != target_ignore and gt_frames_aligned[k] != "BG":
                            gt_frames_filtered.append(gt_frames_aligned[k])
                            pred_frames_filtered.append(pred_frames_aligned[k])

                    if gt_frames_filtered:
                        acc = acc_score(pred_frames_filtered, gt_frames_filtered)
                        edit = edit_score(pred_frames_aligned, gt_frames_aligned) # Edit score usually on full sequence? 
                        # Edit score on filtered sequence is safer for "Action only" metrics
                        # But standard is often full sequence. 
                        # Given user prompt "process only frames which are non-BG", let's apply metrics to filtered.
                        # However, Edit distance on disjoint chunks is weird. 
                        # Let's keep Edit Score on aligned (full) but Accuracy/F1 on filtered.
                        # Actually, if we filter BG, edit distance of the remaining "action clumps" is what we want.
                        
                        f1_scores = f_score(pred_frames_filtered, gt_frames_filtered, overlap_thresholds=[0.10, 0.25, 0.50])
                        
                        all_recipe_frame_metrics[recipe]['Accuracy'][video_id] = acc
                        all_recipe_frame_metrics[recipe]['Edit'][video_id] = edit
                        all_recipe_frame_metrics[recipe]['F1@10'][video_id] = f1_scores[0.10]['f1']
                        all_recipe_frame_metrics[recipe]['F1@25'][video_id] = f1_scores[0.25]['f1']
                        all_recipe_frame_metrics[recipe]['F1@50'][video_id] = f1_scores[0.50]['f1']
                        all_recipe_frame_metrics[recipe]['P@50'][video_id] = f1_scores[0.50]['precision']
                        all_recipe_frame_metrics[recipe]['R@50'][video_id] = f1_scores[0.50]['recall']
                    else:
                        pass 

                except Exception as e:
                    print(f"Error during frame-level metric calculation for {video_id}: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"Error processing file {pkl_path}: {e}")
    

    # --- 5. Report Statistics (Unchanged) ---
    for level in eval_levels_sorted:
        print(f"\n\n{'#'*80}")
        print(f"###   EVALUATION FOR {level}% OF BEAM PATH   ###")
        print(f"{'#'*80}\n")
        
        all_stats = master_all_stats[level]
        per_video_stats = master_per_video_stats[level]
        all_recipe_frame_metrics = master_all_recipe_frame_metrics[level]
        
        sheet_name_no_bg = f'Report_NoBG_{level}pct'
        sheet_name_with_bg = f'Report_WithBG_{level}pct'

        if writer:
            generate_report_sheet(writer, sheet_name_no_bg, all_stats, per_video_stats, 'no_bg', all_recipe_frame_metrics)
            generate_report_sheet(writer, sheet_name_with_bg, all_stats, per_video_stats, 'with_bg', all_recipe_frame_metrics)
        else:
            if level == eval_levels_sorted[0]: 
                print("\n--- No Excel writer. Printing reports to console ---")
            generate_report_sheet(None, sheet_name_no_bg, all_stats, per_video_stats, 'no_bg', all_recipe_frame_metrics)
            generate_report_sheet(None, sheet_name_with_bg, all_stats, per_video_stats, 'with_bg', all_recipe_frame_metrics)

    if writer:
        print("\n--- Writing Per-Video Accuracy Sheets to Excel ---")
        video_report_data_all = []
        for level in eval_levels_sorted:
            per_video_stats = master_per_video_stats[level]
            all_recipe_frame_metrics = master_all_recipe_frame_metrics[level]
            for recipe, video_data in per_video_stats.items():
                for vid, data in video_data.items():
                    video_report_data_all.append({
                        'Level_pct': level,
                        'Recipe': recipe,
                        'Video': vid, 
                        'Total Segments': data['Total'],
                        'Seg Acc (No-BG)': data['no_bg']['Accuracy'], 
                        'Seg Acc (With-BG)': data['with_bg']['Accuracy'], 
                        'Frame Acc': all_recipe_frame_metrics[recipe].get('Accuracy', {}).get(vid, np.nan),
                        'Edit Score': all_recipe_frame_metrics[recipe].get('Edit', {}).get(vid, np.nan),
                        'F1@50': all_recipe_frame_metrics[recipe].get('F1@50', {}).get(vid, np.nan),
                        'P@50': all_recipe_frame_metrics[recipe].get('P@50', {}).get(vid, np.nan),
                        'R@50': all_recipe_frame_metrics[recipe].get('R@50', {}).get(vid, np.nan),
                    })
        
        if video_report_data_all:
             df_video = pd.DataFrame(video_report_data_all)
             df_video = df_video.sort_values(by=['Level_pct', 'Recipe', 'Frame Acc'], ascending=[True, True, False])
             cols = ['Level_pct', 'Recipe', 'Video', 'Total Segments', 'Seg Acc (No-BG)', 'Seg Acc (With-BG)', 'Frame Acc', 'Edit Score', 'F1@50']
             df_video = df_video[cols]
             df_video.to_excel(writer, sheet_name="All_Videos_Summary", index=False)
             print("Done writing per-video summary sheet.")

    if writer:
        try:
            writer.close()
            print(f"\nSuccessfully saved Excel report to {output_excel_file}")
        except Exception as e:
            print(f"\nError: Could not save Excel file: {e}")

    if all_inference_times:
        avg_time = sum(all_inference_times) / len(all_inference_times)
        avg_fps = sum(all_fps_speeds) / len(all_fps_speeds)
        print("\n--- Inference Speed Report ---")
        print(f"Average Inference Time per Video: {avg_time:.4f} seconds")
        print(f"Average Processing Speed: {avg_fps:.2f} FPS")

    print("\n=========================================================")
    print("--- Missing Video Report ---")
    missing_videos = expected_video_ids - found_video_ids
    if not missing_videos:
        print(f"All {len(expected_video_ids)} expected videos were found and processed.")
    else:
        print(f"Found {len(found_video_ids)} / {len(expected_video_ids)} expected videos.")
        print(f"The following {len(missing_videos)} videos were NOT found:")
        for video_id in sorted(list(missing_videos)):
            print(f"  - {video_id}")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate beam search results from .pkl files.")
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--video_list_file', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--output_excel_file', type=str, default=None)
    parser.add_argument('--eval_levels', type=int, nargs='+', default=[100])
    parser.add_argument('--ignore_action', type=str, default="BG", help="Label for Background class to filter out of metrics (default: BG).")
    
    args = parser.parse_args()
    
    calculate_statistics(
        args.results_dir, 
        args.video_list_file, 
        args.annotation_file,
        args.fps,
        args.eval_levels, 
        args.output_excel_file,
        args.ignore_action
    )