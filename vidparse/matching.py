import numpy as np
import pandas as pd
import argparse
import os
import pickle
import json
import re
from collections import defaultdict

from segments import load_annotations_and_get_metadata
from matching_utils import (
    segment_test_video, format_path_for_display, visualize_beam_state,
    filter_features_by_action, calculate_min_dtw_cost,
    get_gt_actions_for_segment, get_gt_actions_from_compressed_timeline,
    get_cost_from_fitted_dist 
)
from decoding_components import (
    DecoderConfig, GraphCandidateGenerator, FreeCandidateGenerator, Scorer, BeamSelector
)

def compute_global_stats(all_stats):
    """Aggregates all action stats to find global Mean and Std."""
    means = []
    stds = []
    
    for action_key, sub_dict in all_stats.items():
        # Handle structure: {'Action': {'mean': x}} vs {'Action': {'Action': ...}}
        target = sub_dict
        if action_key in sub_dict:
             target = sub_dict[action_key]
             
        if 'mean' in target and 'std' in target:
            means.append(target['mean'])
            stds.append(target['std'])
            
    if not means:
        print("Warning: No valid stats found. Using defaults.")
        return {'mu': 0.5, 'sigma': 0.2}
        
    return {
        'mu': float(np.mean(means)),
        'sigma': float(np.mean(stds))
    }

# --- NEW: Mask Logic ---
def smooth_binary_signal(signal, gap_fill_size, min_segment_size):
    signal = np.array(signal, dtype=int)
    n = len(signal)
    
    # 1. Fill Gaps
    i = 0
    while i < n:
        if signal[i] == 0:
            j = i
            while j < n and signal[j] == 0: j += 1
            if i > 0 and j < n and (j - i) <= gap_fill_size:
                signal[i:j] = 1
            i = j
        else: i += 1
            
    # 2. Remove Short Segments
    i = 0
    while i < n:
        if signal[i] == 1:
            j = i
            while j < n and signal[j] == 1: j += 1
            if (j - i) < min_segment_size:
                signal[i:j] = 0
            i = j
        else: i += 1
    return signal

def generate_hand_mask(det_file, total_frames, fps, conf_thresh, fill_gap_sec, min_dur_sec):
    if not det_file or not os.path.exists(det_file):
        return np.ones(total_frames, dtype=int) 

    try:
        with open(det_file, 'r') as f: data = json.load(f)
    except: return np.ones(total_frames, dtype=int)

    frame_indices = set()
    num_pattern = re.compile(r'(\d+)')
    
    # Convert seconds to frames
    gap_fill = int(fill_gap_sec * fps)
    min_dur = int(min_dur_sec * fps)

    for key, dets in data.items():
        matches = num_pattern.findall(key)
        if not matches: continue
        idx = int(matches[-1]) - 1 
        # Use dynamic confidence threshold
        if any(d['class'] == 'hand' and d['score'] >= conf_thresh for d in dets):
            frame_indices.add(idx)

    pred_mask = np.zeros(total_frames, dtype=int)
    for idx in frame_indices:
        if idx < total_frames: pred_mask[idx] = 1
            
    return smooth_binary_signal(pred_mask, gap_fill, min_dur)

def beam_search_decoder(
    test_features: np.ndarray,
    prototypes: dict,
    config: DecoderConfig,
    candidate_generator,
    scorer,
    output_dir: str,
    hand_mask: np.ndarray = None,
    global_stats: dict = None
):
    """Refactored Beam Search Decoder with Checkpointing, Smart Pruning, and GT Viz."""
    os.makedirs(output_dir, exist_ok=True)
    test_micro_segments = segment_test_video(test_features, config.segment_duration, config.fps)
    
    # --- Checkpoint / Init Logic ---
    checkpoint_file_path = os.path.join(output_dir, "beam_search_checkpoint.pkl")
    hyp_id_counter = 0
    start_segment_index = 0
    frames_per_segment = int(config.segment_duration * config.fps)
    
    initial_hypothesis = {
        'path': [], 'cost': 0.0, 'ranking_score': 0.0, 'last_action': None,
        'observed_actions': set(), 'id': hyp_id_counter, 'parent_id': -1, 'step': -1,
        'continuous_action_count': 0, 'last_action_blank': False,
        'continuous_blank_count': 0, 'action_start_times': {}, 'observed_action_counts': {},
        'bg_suppressed': False, 'suppression_ref_cost': None
    }
    
    if os.path.exists(checkpoint_file_path):
        print(f"--- Found checkpoint. Resuming from {checkpoint_file_path} ---")
        try:
            with open(checkpoint_file_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            start_segment_index = checkpoint_data['last_processed_segment_index'] + 1
            beam = checkpoint_data['beam']
            beam_history = checkpoint_data['beam_history']
            hyp_id_counter = checkpoint_data['hyp_id_counter']
            gt_history = checkpoint_data.get('gt_history', []) 
            print(f"Resuming from segment {start_segment_index}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            hyp_id_counter += 1
            beam = [initial_hypothesis]
            beam_history = [beam]
            gt_history = []
    else:
        hyp_id_counter += 1
        beam = [initial_hypothesis]
        beam_history = [beam]
        gt_history = []

    if start_segment_index >= len(test_micro_segments):
         return beam[0], beam_history, gt_history # Already done

    # Init Components
    beam_selector = BeamSelector(config)

    # --- GT Handling for Visualization ---
    video_id = os.path.splitext(os.path.basename(config.test_video_feature_path))[0]
    video_metadata_for_gt = {}
    compressed_gt_segments = None 
    
    try:
        video_metadata_for_gt = load_annotations_and_get_metadata(
            config.annotation_file_path, video_id, config.recipe
        )
        if config.ignore_action and video_metadata_for_gt:
            print(f"Building compressed GT timeline, ignoring '{config.ignore_action}'...")
            compressed_gt_segments = []
            total_duration_removed_s = 0.0
            
            try:
                original_gt_timestamps = video_metadata_for_gt['labels']['time_stamp']
                original_gt_action_names = video_metadata_for_gt['labels']['error_description']

                for action_name, (start_time, end_time) in zip(original_gt_action_names, original_gt_timestamps):
                    duration = end_time - start_time
                    if duration <= 0: continue

                    if action_name == config.ignore_action:
                        total_duration_removed_s += duration
                        continue
                    
                    adjusted_start = start_time - total_duration_removed_s
                    adjusted_end = end_time - total_duration_removed_s
                    
                    compressed_gt_segments.append({
                        'name': action_name,
                        'start': adjusted_start,
                        'end': adjusted_end
                    })
            except KeyError:
                print("Warning: Metadata missing standard labels. Skipping compressed GT build.")
                
    except Exception as e:
        print(f"Warning: Could not load GT for visualization: {e}")

    # --- Main Loop ---
    for i in range(start_segment_index, len(test_micro_segments)):
        micro_segment = test_micro_segments[i]
        print(f"Processing segment {i+1}/{len(test_micro_segments)}...")

        # --- 1. Compute Hand Probability (P_hand) for this segment ---
        m_start = i * frames_per_segment
        m_end = min((i + 1) * frames_per_segment, len(hand_mask))
        segment_mask = hand_mask[m_start:m_end]
        
        raw_segment_p_hand = 1.0
        if len(segment_mask) > 0:
            raw_segment_p_hand = np.mean(segment_mask)
        hands_visible = (raw_segment_p_hand > 0.5)

        # --- 2. Calculate Dynamic BG Cost (The "Floating Baseline") ---
        # If P_hand=0, Cost=0. If P_hand=1, Cost=Mu + Alpha*Std
        if np.isnan(global_stats['sigma']):
            global_stats['sigma'] = 0
        bg_mean_cost = global_stats['mu'] + (config.bg_alpha * global_stats['sigma'])
        bg_mean_cost = max(0.0, bg_mean_cost)
        
        # 3. Get ALL Unique Candidates across the entire beam
        all_unique_candidates = set()
        for hyp in beam:
            cands = candidate_generator.get_candidates(hyp, hyp['continuous_action_count'])
            all_unique_candidates.update(cands)

        # 4. Compute/Cache DTW Scores for this segment
        segment_match_cache = {}
        for action in all_unique_candidates:
            match_result = calculate_min_dtw_cost(micro_segment, prototypes.get(action, []), config.drop_penalty)
            segment_match_cache[action] = match_result

        # 5. Expand Beam
        potential_real_hyps = []
        potential_blank_hyps = []
        any_good_match_globally = False
        
        for hypothesis in beam:
            is_prev_suppressed = hypothesis.get('bg_suppressed', False)
            candidates = candidate_generator.get_candidates(hypothesis, hypothesis['continuous_action_count'], hands_visible)
            best_real_nll = float('inf')
            
            # A. Try Real Actions
            for action in sorted(list(candidates)):
                new_hyp = pickle.loads(pickle.dumps(hypothesis))
                new_hyp['parent_id'] = hypothesis['id']
                new_hyp['id'] = hyp_id_counter
                hyp_id_counter += 1
                new_hyp['step'] = i

                # --- 1. Determine Counts Early (Needed for Trigger Check) ---
                # We need to know what the count WOULD be if we picked this action
                if action == hypothesis['last_action']:
                     current_action_count = hypothesis['continuous_action_count'] + 1
                else:
                     current_action_count = 1

                # --- SUPPRESSION & COST LOGIC ---
                
                # Check 1: Are we CONTINUING a suppression?
                # (Parent was suppressed + Hands still missing)
                # Note: Generator already restricted 'candidates' to only this action if true.
                continuing_suppression = (is_prev_suppressed and not hands_visible)

                # Check 2: Are we TRIGGERING a new suppression?
                # (Parent NOT suppressed + Hands missing + Trigger Action + Count met)
                newly_triggered = (
                    not is_prev_suppressed and
                    config.bg_suppress_trigger and 
                    action == config.bg_suppress_trigger and 
                    current_action_count >= config.bg_suppress_trigger_count
                )

                is_suppressed_now = continuing_suppression or newly_triggered

                dtw_cost = 0.0
                ref_cost = None

                if is_suppressed_now:
                    # --- OPTIMIZATION: SKIP MATCHING ---
                    # Just copy the previous step cost.
                    # This ensures the cost doesn't fluctuate based on "BG" similarity.
                    ref_cost = hypothesis.get('suppression_ref_cost')
                    if ref_cost is None:
                        ref_cost = hypothesis.get('step_cost', 0.0)
                    dtw_cost = hypothesis.get('step_cost', 0.0)
                    
                    # No prototype match info updates needed
                    new_hyp['bg_suppressed'] = True
                else:
                    match_res = segment_match_cache[action]
                    raw_score = match_res['min_cost']

                    if match_res['best_proto_id']:
                        new_hyp['prototype_match'] = {
                            'proto_id': match_res['best_proto_id'],
                            'clip_path': match_res['clip_path'],
                            'alignment': match_res['best_alignment'],
                            'raw_score': raw_score
                        }
                    
                    # Costs
                    if config.cost_function == 'raw':
                        dtw_cost = raw_score
                    else:
                        dtw_cost = scorer.get_dtw_cost(raw_score, action)

                    act_stats = scorer.all_stats.get(action, {}).get(action, {})
                    act_sigma = act_stats.get('std', global_stats['sigma'])
                    if act_sigma is None or act_sigma == 0: act_sigma = global_stats['sigma']
                    
                    # --- Apply Mask Penalty ---
                    # Cost = DTW + Lambda * (1 - P_hand) * Sigma
                    if act_sigma is None or np.isnan(act_sigma):
                        missing_hand_penalty = 0.0
                    else:
                        missing_hand_penalty = config.mask_lambda * (1.0 - raw_segment_p_hand) * act_sigma
                    dtw_cost += missing_hand_penalty
                    new_hyp['bg_suppressed'] = False
                    
                    # 4. Global "Good Match" Check (for pruning blanks later)
                    # ... (keep existing good match logic) ...
                    stats = scorer.all_stats.get(action, {}).get(action, {})
                    if stats.get('mean') is not None:
                        sigma = config.good_match_sigma
                        if action == hypothesis['last_action']: sigma *= config.continuation_sigma_multiplier
                        if raw_score < (stats['mean'] + sigma * stats['std']):
                            any_good_match_globally = True
                
                # If ignore_history is True, we start ranking from 0.0 (local only).
                base_score = 0.0 if config.ignore_history else hypothesis['ranking_score']
                duration_cost = scorer.get_duration_cost(hypothesis, action)
                transition_cost = scorer.get_transition_cost(hypothesis, action)
                # assert duration_cost == 0
                # assert transition_cost == 0
                total_step_cost = dtw_cost + duration_cost + transition_cost
                
                # Update State
                new_hyp['path'].append(action)
                new_hyp['ranking_score'] = total_step_cost + base_score
                new_hyp['cost'] += dtw_cost 
                new_hyp['last_action'] = action
                new_hyp['last_action_blank'] = False
                new_hyp['observed_actions'].add(action)
                new_hyp['continuous_action_count'] = current_action_count
                if current_action_count == 1:
                     current_total_count = new_hyp['observed_action_counts'].get(action, 0)
                     new_hyp['observed_action_counts'][action] = current_total_count + 1
                     if action not in new_hyp['action_start_times']:
                        new_hyp['action_start_times'][action] = i * config.segment_duration

                # Commit Suppression State
                new_hyp['bg_suppressed'] = is_suppressed_now
                new_hyp['suppression_ref_cost'] = ref_cost
                new_hyp['step_cost'] = dtw_cost
                potential_real_hyps.append(new_hyp)

            should_skip_bg = False
            if is_prev_suppressed and not hands_visible:
                should_skip_bg = True
            elif candidates and config.bg_suppress_trigger: 
                 # Edge case: We just entered the "New Trigger" state in the loop above.
                 # If we did, we shouldn't offer BG as an alternative.
                 # However, checking this purely from 'candidates' is hard.
                 # Easier approach: If ANY of the children from this parent are suppressed, skip BG.
                 if any(child['bg_suppressed'] for child in potential_real_hyps if child['parent_id'] == hypothesis['id']):
                     should_skip_bg = True

            # B. Handle BLANK
            if not config.ignore_action and not should_skip_bg: 
                # 1. Determine Visibility (Same as Real Actions)
                hands_visible_in_segment = (raw_segment_p_hand > 0.5)
                is_prev_suppressed = hypothesis.get('bg_suppressed', False)
                # Strictly BLOCK BG if we are currently suppressed and hands are still missing.
                # This forces the decoder to stick with the Real Action (which uses ref_cost).
                if is_prev_suppressed and not hands_visible_in_segment:
                    continue

                new_blank_count = hypothesis.get('continuous_blank_count', 0) + 1

                blank_hyp = pickle.loads(pickle.dumps(hypothesis))
                blank_hyp['parent_id'] = hypothesis['id']
                blank_hyp['id'] = hyp_id_counter
                blank_hyp['step'] = i
                hyp_id_counter += 1
                blank_hyp['path'].append("BG")
                
                # 2. Determine Suppression for Blank
                # Blanks CANNOT trigger suppression, they can only inherit or release (via HOD)
                if hands_visible_in_segment:
                    is_suppressed_blank = False
                else:
                    is_suppressed_blank = is_prev_suppressed
                    
                # 3. Calculate Cost
                effective_p_hand_blank = 1.0 if is_suppressed_blank else raw_segment_p_hand
                blank_bg_cost = bg_mean_cost * (1 + effective_p_hand_blank)

                # Apply costs + penalty
                blank_hyp['ranking_score'] += blank_bg_cost
                blank_hyp['cost'] += blank_bg_cost
                
                blank_hyp['last_action_blank'] = True
                blank_hyp['continuous_blank_count'] = new_blank_count
                
                # Update State
                blank_hyp['bg_suppressed'] = is_suppressed_blank
                
                # Pass forward the ref_cost (though likely unused until we switch back to an action)
                # If released, clear it.
                if not is_suppressed_blank:
                    blank_hyp['suppression_ref_cost'] = None

                if 'step_cost' not in blank_hyp: 
                    blank_hyp['step_cost'] = blank_bg_cost
                else:
                    blank_hyp['step_cost'] += blank_bg_cost
                
                potential_blank_hyps.append(blank_hyp)

        next_beam_candidates = []
        if any_good_match_globally:
            # SCENARIO A: Good match found somewhere. Prune continuous blanks.
            next_beam_candidates.extend(potential_real_hyps)
            for hyp in potential_blank_hyps:
                if hyp['continuous_blank_count'] <= 1:
                    next_beam_candidates.append(hyp)
        else:
            # SCENARIO B: No good match anywhere. Keep everything.
            next_beam_candidates = potential_real_hyps + potential_blank_hyps

        # 4. Smart Pruning
        beam = beam_selector.select_next_beam(next_beam_candidates)
        beam_history.append(beam)

        # 5. Visualization Logic
        start_time = i * config.segment_duration
        end_time = (i + 1) * config.segment_duration
        current_gt_actions = set()

        if compressed_gt_segments is not None:
            current_gt_actions = get_gt_actions_from_compressed_timeline(
                start_time, end_time, compressed_gt_segments
            )
        elif video_metadata_for_gt:
            current_gt_actions = get_gt_actions_for_segment(
                start_time, end_time, video_metadata_for_gt
            )
        
        gt_history.append(current_gt_actions)
        
        if (i % 10 == 0) or (i == len(test_micro_segments) - 1):
             best_id_so_far = beam[0]['id'] if beam else -1
             visualize_beam_state(
                beam_history=beam_history,
                current_step=i,
                output_dir=output_dir,
                start_time=start_time,
                end_time=end_time,
                gt_actions=current_gt_actions,
                gt_history=gt_history,
                best_hyp_id=best_id_so_far
            )

        # 6. Save Checkpoint
        checkpoint_data = {
            'last_processed_segment_index': i,
            'beam': beam,
            'beam_history': beam_history,
            'hyp_id_counter': hyp_id_counter,
            'gt_history': gt_history
        }
        try:
            with open(checkpoint_file_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception: pass

    if os.path.exists(checkpoint_file_path):
        try: os.remove(checkpoint_file_path)
        except: pass

    if len(beam_history) >= 2:
        print("\n--- Selecting best hypothesis from the 2nd-to-last segment ---")
        final_beam_to_use = beam_history[-2]
    else:
        print("\n--- Warning: Video is < 2 segments. Using final beam as fallback. ---")
        final_beam_to_use = beam

    if not final_beam_to_use:
        final_best = beam[0]
    else:
        final_best = sorted(final_beam_to_use, key=lambda h: h['ranking_score'])[0]

    return final_best, beam_history, gt_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--feature_file', type=str, required=True)
    parser.add_argument('--prototypes_file', type=str, required=True)
    parser.add_argument('--task_graph_file', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--stats_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--action_duration_stats_file', type=str, required=True)
    parser.add_argument('--temporal_offset_stats_file', type=str, required=True)
    parser.add_argument('--transition_probs_file', type=str, required=True)
    parser.add_argument('--dump_file', type=str, required=True)
    
    # Optional Flags
    parser.add_argument('--use_graph', action='store_true')
    parser.add_argument('--no_graph', action='store_false', dest='use_graph')
    parser.set_defaults(use_graph=True)
    
    parser.add_argument('--transition_weight', type=float, default=1.0)
    parser.add_argument('--duration_weight', type=float, default=1.0)
    
    # Other params
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--segment_duration', type=float, default=0.5)
    parser.add_argument('--drop_penalty', type=float, default=0.5)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--ignore_action', type=str, default=None)
    
    # Restored blank penalty separation
    parser.add_argument('--blank_penalty', type=float, default=0.6, help="True Cost of a blank.")
    parser.add_argument('--ranking_blank_cost', type=float, default=0.1, help="Heuristic cost for pruning blanks.")

    parser.add_argument('--continuation_bonus', type=float, default=0.2)
    parser.add_argument('--continuation_penalty', type=float, default=1.0)
    parser.add_argument('--short_duration_penalty', type=float, default=1.0)
    parser.add_argument('--temporal_offset_penalty', type=float, default=0.5)
    parser.add_argument('--transition_penalty', type=float, default=0.5)
    parser.add_argument('--good_match_sigma', type=float, default=1.0)
    parser.add_argument('--continuation_sigma_multiplier', type=float, default=1.25)
    parser.add_argument('--cost_function', type=str, default='nll', choices=['nll', 'cdf', 'raw'])
    parser.add_argument('--pruning_metric', type=str, default='ranking_score')
    parser.add_argument('--ignore_history', action='store_true')
    parser.add_argument('--force_distinct_actions', action='store_true')

    parser.add_argument('--detections_file', type=str, default=None)
    parser.add_argument('--mask_lambda', type=float, default=1.0)
    parser.add_argument('--bg_alpha', type=float, default=0.0)
    parser.add_argument('--conf_thresh', type=float, default=0.75, help="Confidence threshold for hand detection.")
    parser.add_argument('--fill_gap', type=float, default=0.2, help="Max gap size (in seconds) to fill.")
    parser.add_argument('--min_duration', type=float, default=0.2, help="Min duration (in seconds) for valid FG.")

    parser.add_argument('--bg_suppress_trigger', type=str, default=None, help="Action that triggers ignoring HOD BG signal")
    parser.add_argument('--bg_suppress_trigger_count', type=int, default=2)


    args = parser.parse_args()

    # 1. Load Data
    test_features = np.load(args.feature_file)
    prototypes = np.load(args.prototypes_file, allow_pickle=True).item()
    with open(args.task_graph_file) as f: task_graph = json.load(f).get(args.recipe)
    with open(args.annotation_file) as f: annot_data = json.load(f)
    with open(args.stats_file) as f: all_stats = json.load(f)
    with open(args.action_duration_stats_file) as f: dur_stats = json.load(f)
    with open(args.transition_probs_file) as f: trans_probs = json.load(f).get(args.recipe, {})
    
    # 1. Compute Global Stats
    global_stats = compute_global_stats(all_stats)
    
    # 2. Load Mask
    total_frames = int(test_features.shape[0]) 
    hand_mask = generate_hand_mask(args.detections_file, total_frames, args.fps, args.conf_thresh, args.fill_gap, args.min_duration)

    # 3. Config
    config = DecoderConfig(
        use_task_graph=args.use_graph,
        transition_penalty_weight=args.transition_penalty,
        continuation_penalty_weight=args.continuation_penalty,
        continuation_bonus_weight=args.continuation_bonus,
        temporal_offset_penalty_weight=args.temporal_offset_penalty,
        beam_width=args.beam_width,
        segment_duration=args.segment_duration,
        fps=args.fps,
        drop_penalty=args.drop_penalty,
        test_video_feature_path=args.feature_file,
        annotation_file_path=args.annotation_file,
        recipe=args.recipe,
        ignore_action=args.ignore_action,
        
        # Scoring Params
        ranking_blank_cost=args.ranking_blank_cost,
        good_match_sigma=args.good_match_sigma, # Pass config param
        continuation_sigma_multiplier=args.continuation_sigma_multiplier, # Pass config param
        
        cost_function=args.cost_function,
        pruning_metric=args.pruning_metric,
        
        # Pass weights to components
        short_duration_penalty_weight=args.short_duration_penalty,
        ignore_history=args.ignore_history,
        force_distinct_actions=args.force_distinct_actions,
        mask_lambda=args.mask_lambda,
        bg_alpha=args.bg_alpha,
        true_blank_cost=0.0,

        bg_suppress_trigger=args.bg_suppress_trigger,
        bg_suppress_trigger_count=args.bg_suppress_trigger_count
    )

    # 3. Setup Components
    action_to_id = annot_data[args.recipe]['action2idx']
    id_to_action = {v: k for k, v in action_to_id.items()}
    
    # Deriving video_id regardless of filtering, so it is available for dumps
    video_id = os.path.splitext(os.path.basename(args.feature_file))[0]

    if args.ignore_action:
        video_metadata = load_annotations_and_get_metadata(args.annotation_file, video_id, args.recipe)
        test_features = filter_features_by_action(test_features, video_metadata, args.ignore_action, args.fps)

    if config.use_task_graph:
        generator = GraphCandidateGenerator(
            task_graph, id_to_action, action_to_id, dur_stats, config.segment_duration
        )
    else:
        all_actions = set(prototypes.keys())
        generator = FreeCandidateGenerator(all_actions)

    scorer = Scorer(config, duration_stats=dur_stats, transition_probs=trans_probs, all_stats=all_stats)

    # 4. Run
    best_hyp, history, gt_history = beam_search_decoder(
        test_features, prototypes, config, generator, scorer, args.output_dir,
        hand_mask=hand_mask, 
        global_stats=global_stats
    )

    # 5. Save Results
    print(f"Final Path Cost: {best_hyp['ranking_score']:.4f}")
    formatted = format_path_for_display(best_hyp['path'], args.segment_duration)
    print(pd.DataFrame(formatted).to_string(index=False) if formatted else "No actions.")

    try:
        with open(args.dump_file, 'wb') as f:
            pickle.dump({
                "video_id": video_id,
                "segment_duration": args.segment_duration,
                "best_hypothesis": best_hyp,
                "beam_history": history,
                "gt_history_per_segment": gt_history,
                "formatted_path": formatted
            }, f)
        print(f"Saved results to {args.dump_file}")
    except Exception as e:
        print(f"Error saving dump: {e}")