import os
import math
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from scipy.stats import norm, weibull_min, gamma
from scipy.spatial.distance import cdist

from segments import load_annotations_and_get_metadata, segment_features_by_action
from feat_comparison import align_videos_with_provided_code
from scipy.spatial.distance import cosine


def calculate_avg_cosine_cost(micro_segment: np.ndarray, action_prototypes: list):
    """
    Calculates cost by comparing the Mean Feature Vector of the segment
    vs. the Mean Feature Vector of the prototype.
    Ignores temporal structure. Fast and robust.
    """
    best_result = {
        'min_cost': float('inf'),
        'best_proto_id': None,
        'best_alignment': None, # No alignment in global averaging
        'clip_path': None
    }
    
    if not action_prototypes or micro_segment.shape[0] == 0:
        return best_result

    # 1. Compute Mean of the Micro-Segment (Collapse Time)
    # Shape: (D,)
    segment_mean = np.mean(micro_segment, axis=0)
    
    for proto in action_prototypes:
        proto_features = proto['features']
        if proto_features.size == 0: continue
        
        # 2. Compute Mean of the Prototype
        # Shape: (D,)
        proto_mean = np.mean(proto_features, axis=0)
        
        # 3. Compute Cosine Distance (0.0 to 2.0)
        # 0.0 = Identical average
        # 1.0 = Orthogonal
        try:
            dist = cosine(segment_mean, proto_mean)
        except:
            dist = float('inf')
            
        if dist < best_result['min_cost']:
            best_result['min_cost'] = dist
            best_result['best_proto_id'] = proto.get('id', 'unknown')
            best_result['clip_path'] = proto.get('clip_path', None)
            
    return best_result


def filter_features_by_action(
    features: np.ndarray, 
    video_metadata: dict, 
    action_to_ignore: str, 
    fps: float
) -> np.ndarray:
    """
    Filters a feature array by removing all frames corresponding
    to a specified 'action_to_ignore' (e.g., 'BG').
    
    Returns a new, shorter feature array.
    """
    if not action_to_ignore:
        print("No action_to_ignore specified. Returning original features.")
        return features

    if not video_metadata:
        print(f"Warning: Cannot filter features by '{action_to_ignore}' without video metadata. Returning original features.")
        return features

    try:
        timestamps = video_metadata['labels']['time_stamp']
        action_names = video_metadata['labels']['error_description']
    except KeyError:
        print("Warning: Metadata missing 'labels'. Returning original features.")
        return features

    # 1. Create a list of "keep" segments (start_frame, end_frame)
    keep_segments = []
    
    for action_name, (start_time, end_time) in zip(action_names, timestamps):
        if action_name == action_to_ignore:
            continue
            
        # This is an action we want to keep. Convert times to frames.
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Ensure frame indices are within bounds
        end_frame = min(end_frame, features.shape[0])
        start_frame = min(start_frame, end_frame)
        
        if start_frame < end_frame:
            keep_segments.append((start_frame, end_frame))
            
    if not keep_segments:
        print(f"Warning: Filtering by '{action_to_ignore}' resulted in zero features. Returning empty array.")
        return np.empty((0, features.shape[1]), dtype=features.dtype)

    # 2. Concatenate the features from these segments
    filtered_feature_list = [
        features[start:end, :] for start, end in keep_segments
    ]
    
    new_features = np.concatenate(filtered_feature_list, axis=0)
    
    print(f"Filtered features: Original size {features.shape[0]}, New size {new_features.shape[0]} (Removed {features.shape[0] - new_features.shape[0]} frames of '{action_to_ignore}')")
    
    return new_features


def get_gt_actions_for_segment(start_time: float, end_time: float, video_metadata: dict) -> set:
    """
    Finds all ground-truth actions that overlap with a given time segment.
    """
    gt_actions = set()
    try:
        timestamps = video_metadata['labels']['time_stamp']
        action_names = video_metadata['labels']['error_description']
    except KeyError:
        # No labels found, return empty set
        return gt_actions

    # Add a small buffer to handle floating point precision
    buffer = 1e-4 
    
    for gt_action, (gt_start, gt_end) in zip(action_names, timestamps):
        # Check for overlap: (gt_start < seg_end) and (gt_end > seg_start)
        if (gt_start < (end_time - buffer)) and (gt_end > (start_time + buffer)):
            gt_actions.add(gt_action)
            
    return gt_actions


def segment_test_video(features: np.ndarray, segment_duration_seconds: float, fps: float) -> list:
    """
    Splits a video's features into uniform, non-overlapping chunks (micro-segments).
    """
    if features.ndim == 1:
        print(f"Warning: features are 1D. Reshaping to (N, 1).")
        features = features.reshape(-1, 1)

    frames_per_segment = int(segment_duration_seconds * fps)
    num_segments = math.ceil(features.shape[0] / frames_per_segment)
    
    micro_segments = []
    for i in range(num_segments):
        start_frame = i * frames_per_segment
        end_frame = min((i + 1) * frames_per_segment, features.shape[0])
        if start_frame < end_frame:
            micro_segments.append(features[start_frame:end_frame, :])
            
    print(f"Segmented test video into {len(micro_segments)} micro-segments of ~{segment_duration_seconds}s each.")
    return micro_segments


def format_path_for_display(path: list, segment_duration: float):
    """
    Collapses a raw path (e.g., ['A', 'A', 'B']) into a structured, timed format.
    """
    if not path:
        return []

    collapsed_path = []
    current_action = path[0]
    start_time = 0.0
    
    for i in range(1, len(path)):
        if path[i] != current_action:
            end_time = i * segment_duration
            if current_action != "BLANK":
                collapsed_path.append({
                    "action": current_action,
                    "start_time": f"{start_time:.2f}s",
                    "end_time": f"{end_time:.2f}s"
                })
            current_action = path[i]
            start_time = end_time
    
    # Add the final action segment
    end_time = len(path) * segment_duration
    if current_action != "BLANK":
        collapsed_path.append({
            "action": current_action,
            "start_time": f"{start_time:.2f}s",
            "end_time": f"{end_time:.2f}s"
        })
        
    return collapsed_path


def visualize_beam_state(
        beam_history: list,
        current_step: int,
        output_dir: str,
        start_time: float,
        end_time: float,
        gt_actions: set,
        gt_history: list,
        best_hyp_id: int = -1
    ):
    """
    Generates a PNG image visualizing the current state of the beam search.
    Only renders the last 10 steps to keep the graph readable.
    """
    LOOKBACK_WINDOW = 10
    min_render_step = max(0, current_step - LOOKBACK_WINDOW)
    
    g = graphviz.Digraph(comment=f'Beam Search - Step {current_step}')
    gt_str = ", ".join(sorted(list(gt_actions))) if gt_actions else "N/A"
    
    # Update label to indicate the window being shown
    g.attr(
        rankdir='TB', 
        label=f"Beam State (Steps {min_render_step}-{current_step}) | Time: {start_time:.2f}s - {end_time:.2f}s\nGT: {gt_str}", 
        labelloc="t"
    )
    g.attr(size="8.5,100")

    # 1. Create a lookup of all hypotheses ever created
    # (We still need the full lookup for backtracking active paths properly)
    hyp_lookup = {}
    for step_beams in beam_history:
        for hyp in step_beams:
            hyp_lookup[hyp['id']] = hyp
            
    # 2. Find all node and edge IDs that are part of the final (current) beam
    # (Logic remains exactly the same here to correctly identify red/blue paths)
    active_elements = set() 
    active_hypotheses = beam_history[-1] 
    
    queue = [hyp['id'] for hyp in active_hypotheses]
    while queue:
        hyp_id = queue.pop()
        if hyp_id in active_elements: continue     
        active_elements.add(hyp_id)
        if hyp_id not in hyp_lookup: continue
        hyp = hyp_lookup[hyp_id]
        parent_id = hyp['parent_id']
        if parent_id != -1:
            active_elements.add((parent_id, hyp_id))
            if parent_id not in active_elements: queue.append(parent_id)

    best_path_elements = set()
    if best_hyp_id != -1 and best_hyp_id in hyp_lookup:
        queue = [best_hyp_id]
        while queue:
            hyp_id = queue.pop()
            if hyp_id in best_path_elements: continue
            best_path_elements.add(hyp_id) 
            if hyp_id not in hyp_lookup: continue
            hyp = hyp_lookup[hyp_id]
            parent_id = hyp['parent_id']
            if parent_id != -1:
                best_path_elements.add((parent_id, hyp_id))
                if parent_id not in best_path_elements: queue.append(parent_id)

    # Only iterate through the history slice that is within the window
    # We use slicing on beam_history to skip early steps
    history_slice = beam_history[min_render_step:]
    
    for step_beams in history_slice:
        for hyp in step_beams:
            node_id = hyp['id']
            parent_id = hyp['parent_id']
            
            # (Color logic remains the same)
            action = hyp['path'][-1] if hyp['path'] else "START"
            
            is_correct = False
            if action == "START": is_correct = True
            else:
                node_step_index = hyp['step']
                node_gt_actions = set()
                if node_step_index < len(gt_history):
                    node_gt_actions = gt_history[node_step_index]
                is_correct = (action in node_gt_actions) or (action == "BLANK" and not node_gt_actions)

            if node_id in best_path_elements:
                final_color = 'red'; pen_width = '3.0'
            elif node_id in active_elements:
                final_color = 'blue'; pen_width = '1.0'
            else:
                final_color = 'gray'; pen_width = '1.0'

            if final_color == 'red': fill_color = '#e0ffe0' if is_correct else '#ffe0e0'
            elif final_color == 'blue': fill_color = '#e0ffe0' if is_correct else '#e0e0ff'
            else: fill_color = '#f0f0f0'

            label = (
                f"Action: {action}\n"
                f"NLL Cost: {hyp.get('step_cost', 0.0):.2f}\n"
                f"Full Cost: {hyp['cost']:.2f}\n"
                f"Rank: {hyp['ranking_score']:.2f}\n"
            )
            
            g.node(str(node_id), label=label, color=final_color, fontcolor='black', style='filled', fillcolor=fill_color)
            
            # --- CHANGE 3: Conditional Edge Drawing ---
            # Only draw the edge if the parent is ALSO within the render window.
            # If hyp['step'] is exactly min_render_step, we treat it as a root for this graph.
            if parent_id != -1 and hyp['step'] > min_render_step:
                if (parent_id, node_id) in best_path_elements: edge_color = 'red'
                elif (parent_id, node_id) in active_elements: edge_color = 'blue'
                else: edge_color = 'gray'
                g.edge(str(parent_id), str(node_id), color=edge_color)

    # 4. Render the graph
    output_filename = os.path.join(output_dir, f"beam_step_{current_step:03d}")
    try:
        g.render(output_filename, format='pdf', cleanup=True)
        print(f"    -> Saved beam visualization: {output_filename}.png")
    except Exception as e:
        pass


def compare_segment_pair(args_tuple):
    """
    Wrapper function to call Drop-DTW on a pair of feature segments.
    Designed to be used with multiprocessing.Pool.
    """
    features1, features2, drop_penalty = args_tuple
    try:
        result = align_videos_with_provided_code(features1, features2, drop_penalty)
        return result['normalized_score']
    except Exception:
        # print("Found an exception")
        return np.inf

def create_action_similarity_matrix(
    feature_path1: str,
    feature_path2: str,
    annotation_path: str,
    recipe1: str,
    recipe2: str,
    output_dir: str, # <-- New parameter
    fps: float = 10.0,
    drop_penalty: float = 0.5
) -> pd.DataFrame:
    """
    Compares all sub-action segments between two videos and generates a 
    similarity matrix of their alignment scores.
    """
    # (Steps 1-4 are the same as before: load data, segment, run comparisons)
    features1 = np.load(feature_path1)
    features2 = np.load(feature_path2)
    
    video_id1 = os.path.splitext(os.path.basename(feature_path1))[0]
    video_id2 = os.path.splitext(os.path.basename(feature_path2))[0]
    
    metadata1 = load_annotations_and_get_metadata(annotation_path, video_id1, recipe1)
    metadata2 = load_annotations_and_get_metadata(annotation_path, video_id2, recipe2)

    if not metadata1 or not metadata2:
        print("Error: Could not retrieve metadata for one or both videos.")
        return pd.DataFrame()

    segments1 = segment_features_by_action(features1, metadata1, fps)
    segments2 = segment_features_by_action(features2, metadata2, fps)

    comparison_tasks = []
    task_info = []
    flat_segments1 = [(name, i, seg) for name, instances in segments1.items() for i, seg in enumerate(instances)]
    flat_segments2 = [(name, i, seg) for name, instances in segments2.items() for i, seg in enumerate(instances)]

    for (name1, _, seg1) in flat_segments1:
        for (name2, _, seg2) in flat_segments2:
            comparison_tasks.append((seg1, seg2, drop_penalty))
            task_info.append({'action1': name1, 'action2': name2})
    
    print(f"Prepared {len(comparison_tasks)} pairwise DTW comparisons...")

    num_workers = min(cpu_count(), len(comparison_tasks))
    with Pool(processes=num_workers) as pool:
        print(f"Starting parallel processing with {num_workers} workers...")
        scores = pool.map(compare_segment_pair, comparison_tasks)
    
    # compare_segment_pair(comparison_tasks[1])
    print("All comparisons finished.")
    aggregated_scores = defaultdict(lambda: defaultdict(list))
    for i, score in enumerate(scores):
        info = task_info[i]
        aggregated_scores[info['action1']][info['action2']].append(score)

    unique_actions1 = sorted(segments1.keys())
    unique_actions2 = sorted(segments2.keys())
    
    similarity_matrix = pd.DataFrame(index=unique_actions1, columns=unique_actions2, dtype=float)

    for action1, action2 in product(unique_actions1, unique_actions2):
        if action2 in aggregated_scores[action1]:
            min_score = min(aggregated_scores[action1][action2])
            similarity_matrix.loc[action1, action2] = min_score
            
    similarity_matrix.fillna(np.inf, inplace=True)

    # --- New plotting logic ---
    if not similarity_matrix.empty:
        # **Debug Step:** Check the range of scores before plotting.
        min_val = similarity_matrix.min().min()
        max_val = similarity_matrix.max().max()
        print(f"\nDEBUG: Score range of the matrix is [{min_val:.4f}, {max_val:.4f}]")
        
        # **Assertion:** Enforce that scores are non-negative distances.
        # If this line fails, it confirms the problem is in the data generation, not the plotting.
        assert min_val >= 0, "Error: Negative distance scores found, check your feature generation or distance metric."

        print("Generating similarity matrix plot...")
        plt.figure(figsize=(max(12, 0.5 * len(unique_actions2)), max(10, 0.5 * len(unique_actions1))))
        
        # should_annotate = similarity_matrix.size <= 225


        # **Improved Heatmap:** Use a "coolwarm" map where blue is "cool" (low score, good match)
        # and red is "warm" (high score, bad match). We explicitly set the range from 0 to 1.
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm_r", # Reversed coolwarm: Low scores are red ("hot"), high scores are blue ("cold")
            linewidths=.5,
            vmin=0,   # Set a fixed minimum for the color scale
            vmax=1.0  # Set a fixed maximum for a more stable color representation
        )
        
        plt.title(f"Action Similarity Matrix\n(Video: {video_id1} vs. {video_id2})", fontsize=16)
        plt.xlabel(f"Actions in Video: {video_id2}", fontsize=12)
        plt.ylabel(f"Actions in Video: {video_id1}", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        output_path = os.path.join(output_dir, f"similarity_{video_id1}_vs_{video_id2}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved plot to: {output_path}")
            
    return similarity_matrix

def get_cost_from_fitted_dist(value: float, 
                              action_stats: dict, 
                              cost_function: str = 'nll', 
                              default_high_cost: float = 20.0) -> float:
    """
    Calculates a cost from a value given a pre-calculated distribution.
    
    This function checks for distributions in this priority order:
    1. 'gmm_fit':   Uses a Gaussian Mixture Model (most flexible).
    2. 'dist_fit': Uses a specific fitted distribution (e.g., 'gamma', 'weibull_min').
    3. 'mean'/'std': Falls back to a simple Normal distribution.
    
    Cost Functions:
    'nll':   Negative Log-Likelihood (-log(pdf(x))).
             Asks "How *probable* is this exact score?"
    'cdf':   Negative Log-Survival-Function (-log(sf(x))).
             Asks "How *surprising* is this score (or higher)?"
    """
    
    if value <= 0:
        # Ensure value is slightly positive for logpdf/logsf calculation
        value = 1e-6 

    try:
        log_value = -np.inf # Default to -infinity
        
        # --- 1. NEW: Try GMM fit first ---
        gmm_params = action_stats.get('gmm_fit')
        if gmm_params:
            weights = gmm_params.get('weights')
            means = gmm_params.get('means')
            covariances = gmm_params.get('covariances')
            best_k = gmm_params.get('best_k')

            # Check for valid GMM data
            if (weights is not None and means is not None and covariances is not None and
                best_k is not None and len(weights) == best_k and 
                len(means) == best_k and len(covariances) == best_k):
                
                total_prob = 0.0
                try:
                    for i in range(best_k):
                        w = weights[i]
                        # Extract scalar mean and std from GMM output
                        m = means[i][0] 
                        s = np.sqrt(covariances[i][0][0])
                        
                        if s < 1e-6: # Avoid division by zero / singular component
                            continue

                        if cost_function == 'nll':
                            # PDF: sum(w_i * pdf_i)
                            total_prob += w * norm.pdf(value, loc=m, scale=s)
                        elif cost_function == 'cdf':
                            # Survival Function: sum(w_i * sf_i)
                            total_prob += w * norm.sf(value, loc=m, scale=s)
                    
                    if total_prob < 1e-300: # np.log(0) is -inf
                        log_value = -np.inf 
                    else:
                        log_value = np.log(total_prob)
                    
                    # Check if GMM calculation was successful
                    if not (np.isinf(log_value) or np.isnan(log_value)):
                        cost = -log_value
                        return min(cost, default_high_cost)
                    
                except Exception:
                    # GMM calculation failed, pass through to fallback
                    pass 
            
            # If GMM failed or was invalid, we fall through...

        # --- 2. Try 'dist_fit' (Gamma, Weibull) ---
        dist_params = action_stats.get('dist_fit')
        dist = None
        params = {}

        if dist_params:
            dist_name = dist_params.get('dist')
            if dist_name == 'weibull_min':
                dist = weibull_min
                params = {'shape': dist_params['shape'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
            elif dist_name == 'gamma':
                dist = gamma
                params = {'a': dist_params['a'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
            else:
                # Unknown dist_fit, fall through to Normal fallback
                pass
        
        if dist is None:
            # --- 3. Fallback to Normal distribution ---
            mean = action_stats.get('mean')
            std = action_stats.get('std')
            if mean is None or std is None or np.isnan(std) or std < 1e-6:
                return default_high_cost # Cannot compute
            
            dist = norm
            params = {'loc': mean, 'scale': std}

        # --- 4. Calculate log-probability (for dist_fit or Normal) ---
        if cost_function == 'nll':
            log_value = dist.logpdf(value, **params)
        elif cost_function == 'cdf':
            log_value = dist.logsf(value, **params)
        else:
             return default_high_cost

        # --- 5. Check for invalid results and convert to cost ---
        if np.isinf(log_value) or np.isnan(log_value):
            if log_value == -np.inf:
                # This means probability is 0. 
                # -log(0) = inf. This is a very high cost.
                return default_high_cost * 2 
            return default_high_cost
            
        cost = -log_value
        
        # Cap the cost
        return min(cost, default_high_cost)

    except (KeyError, ValueError, TypeError) as e:
        # Error unpacking params or during calculation
        print(f"Warning: NLL calculation error. {e}")
        return default_high_cost
    

def get_nll_cost(
    raw_score: float, 
    action_stats: dict, 
    cost_function: str = 'nll'
) -> float:
    """
    Calculates the cost of a raw score, given the pre-calculated 
    distribution (mean, std) for that action.
    
    Can use two methods:
    1. 'nll': Negative Log-Likelihood (from logpdf).
    2. 'cdf': Negative Log-Survival-Function (from log(1-cdf)).
             This is good for "lower is better" scores.
    """
    # A very high cost for invalid inputs
    DEFAULT_HIGH_COST = 50.0 

    # if not action_stats or 'mean' not in action_stats or 'std' not in action_stats:
    #     # No stats for this action, return a high cost (or fallback to raw score)
    #     return DEFAULT_HIGH_COST

    mean = action_stats.get('mean')
    std = action_stats.get('std')

    # If std is 0 or NaN (e.g., only 1 sample), we can't use it.
    # Also check if mean is missing.
    if mean is None or std is None or np.isnan(std) or std < 1e-6:
        # Cannot compute a valid cost
        return DEFAULT_HIGH_COST

    cost = DEFAULT_HIGH_COST # Default
    
    if cost_function == 'nll':
        # NLL is the negative of log-probability. Lower is better.
        # This can be negative if prob density > 1.
        log_prob = norm.logpdf(raw_score, loc=mean, scale=std)
        cost = -log_prob

    elif cost_function == 'cdf':
        # Use the log of the survival function (1 - CDF).
        # This measures the log-probability of getting a score *this high or higher*.
        # A low (good) raw_score gives sf=1, logsf=0, cost=0. (Good)
        # A high (bad) raw_score gives sf=0, logsf=-inf, cost=+inf. (Good)
        log_sf = norm.logsf(raw_score, loc=mean, scale=std)
        
        # -log_sf gives us a positive cost. Lower is better.
        cost = -log_sf

    # Cap the cost to prevent a single insane outlier
    # from creating an infinite cost.
    return min(cost, DEFAULT_HIGH_COST)


def get_nll_from_dist(value: float, dist_params: dict, default_high_cost: float = 20.0) -> float:
    """
    Calculates the Negative Log-Likelihood (NLL) of a value given
    a pre-calculated distribution's parameters.
    
    A high cost (e.g., 20.0) is equivalent to a tiny probability 
    (e.g., e^-20), so it's a strong penalty.
    """
    if not dist_params or dist_params.get('dist') != 'weibull_min':
        # No valid distribution parameters, return a high cost
        return default_high_cost

    try:
        shape = dist_params['shape']
        loc = dist_params['loc']
        scale = dist_params['scale']
        
        # Ensure value is slightly positive for logpdf calculation
        if value <= 0:
            value = 1e-6 
            
        log_prob = weibull_min.logpdf(value, shape, loc=loc, scale=scale)
        
        # Check for invalid log_prob (e.g., -inf, nan)
        if np.isinf(log_prob) or np.isnan(log_prob):
            return default_high_cost
            
        # NLL is the negative of the log-probability
        cost = -log_prob
        
        # Cap the cost to prevent a single bad value from dominating
        return min(cost, default_high_cost)

    except (KeyError, ValueError, TypeError):
        # Error unpacking params or during calculation
        return default_high_cost
    
def calculate_min_dtw_cost(
    micro_segment: np.ndarray,
    action_prototypes: list,
    drop_penalty: float
) -> float:
    """
    Calculates the alignment cost of a micro-segment against all prototypes for a
    given action. Returns both the min_cost and the detailed metadata of the best matching prototype.
    """
    best_result = {
        'min_cost': float('inf'),
        'best_proto_id': None,
        'best_alignment': None, # [(input_frame, proto_frame), ...]
        'clip_path': None
    }
    
    if not action_prototypes:
        return best_result
        
    for proto in action_prototypes:
        proto_features = proto['features']
        if proto_features.ndim == 1:
            proto_features = proto_features.reshape(-1, 1)

        alignment = align_videos_with_provided_code(proto_features, micro_segment, drop_penalty)
        score = alignment.get('normalized_score', float('inf'))
        matches = alignment.get('matches', [])
        if not matches:
            score = float('inf')
            
        if score < best_result['min_cost']:
            best_result['min_cost'] = score
            best_result['best_proto_id'] = proto.get('id', 'unknown')
            best_result['best_alignment'] = alignment.get('matches', [])
            best_result['clip_path'] = proto.get('clip_path', None)
            
    return best_result


def get_gt_actions_from_compressed_timeline(
    segment_start_time: float, 
    segment_end_time: float, 
    compressed_gt_segments: list
) -> set:
    """
    Finds all GT actions that overlap with a given time segment,
    using a pre-computed compressed GT timeline.
    """
    gt_actions = set()
    # Add a small buffer to handle floating point precision
    buffer = 1e-4 
    
    for gt_segment in compressed_gt_segments:
        gt_start = gt_segment['start']
        gt_end = gt_segment['end']
        gt_action = gt_segment['name']
        
        # Check for overlap: (gt_start < seg_end) and (gt_end > seg_start)
        if (gt_start < (segment_end_time - buffer)) and (gt_end > (segment_start_time + buffer)):
            gt_actions.add(gt_action)
            
    return gt_actions