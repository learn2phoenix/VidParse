import numpy as np
import pandas as pd
import argparse
import os
import pickle
import json
import re
import time
import cv2
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks, convolve2d
from scipy.special import logsumexp

from segments import load_annotations_and_get_metadata
from matching_utils import (
    format_path_for_display, visualize_beam_state,
    filter_features_by_action, calculate_min_dtw_cost, calculate_avg_cosine_cost,
    get_gt_actions_for_segment, get_gt_actions_from_compressed_timeline
)
from decoding_components import (
    DecoderConfig, GraphCandidateGenerator, FreeCandidateGenerator, Scorer, BeamSelector
)

class OnlineAdaptiveZScoreDetector:
    """Baseline 3: Adaptive Rolling Z-Score.
       Auto-calibrates to the scale of DINOv2 distances using local statistics."""
    def __init__(self, window_size=20, z_threshold=.5, min_dist=20):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_dist = min_dist
        self.last_emitted_boundary = -999
        self.prev_feature = None
        self.dist_history = []

    def process_stream(self, new_features, start_idx):
        new_bounds = []
        for i, feat in enumerate(new_features):
            global_idx = start_idx + i
            
            if self.prev_feature is not None:
                norm1 = np.linalg.norm(feat) + 1e-6
                norm2 = np.linalg.norm(self.prev_feature) + 1e-6
                sim = np.dot(feat, self.prev_feature) / (norm1 * norm2)
                dist = 1.0 - sim
                
                # Wait until we have a full window to compute reliable local stats
                if len(self.dist_history) == self.window_size:
                    local_mean = np.mean(self.dist_history)
                    local_std = np.std(self.dist_history) + 1e-6
                    
                    z_score = (dist - local_mean) / local_std
                    
                    # Trigger if the spike is significant and cooldown has passed
                    if z_score > self.z_threshold and (global_idx - self.last_emitted_boundary) >= self.min_dist:
                        new_bounds.append(global_idx)
                        self.last_emitted_boundary = global_idx
                        
                self.dist_history.append(dist)
                
                # Keep the window rolling
                if len(self.dist_history) > self.window_size:
                    self.dist_history.pop(0) 
                    
            self.prev_feature = feat
            
        return new_bounds

class OnlineBOCDEdgeDetector:
    """Baseline 3: Bayesian Online Change Point Detection (Adams & MacKay 2007).
       Operates on the 1D frame-to-frame distance signal using a Gaussian 
       observation model with unknown mean and fixed variance."""
    def __init__(self, hazard_rate=40, prior_var=0.0005, min_dist=20, change_threshold=0.05):
        self.hazard_rate = hazard_rate
        self.prior_var = prior_var
        self.min_dist = min_dist
        self.change_threshold = change_threshold
        
        self.last_emitted_boundary = -999
        self.prev_feature = None
        
        # Log-probabilities of run lengths P(r_t | x_{1:t})
        self.log_R = np.array([0.0]) 
        
        # Sufficient statistics for the Gaussian prior/posterior (Unknown Mean)
        self.prior_mu = 0.0
        self.prior_kappa = 1.0  # Precision scale
        
        self.mu_params = np.array([self.prior_mu])
        self.kappa_params = np.array([self.prior_kappa])
        
        # Capping the run-length tracking prevents memory explosions
        self.max_tracked_run_length = 1000 

    def process_stream(self, new_features, start_idx):
        new_bounds = []
        
        for i, feat in enumerate(new_features):
            global_idx = start_idx + i
            
            if self.prev_feature is not None:
                # 1. Compute 1D Observation (Cosine Distance)
                norm1 = np.linalg.norm(feat) + 1e-6
                norm2 = np.linalg.norm(self.prev_feature) + 1e-6
                sim = np.dot(feat, self.prev_feature) / (norm1 * norm2)
                x_t = 1.0 - sim
                
                # 2. Evaluate Predictive Probability: log P(x_t | r_{t-1})
                # Using Gaussian predictive distribution: N(mu, prior_var + prior_var/kappa)
                predictive_vars = self.prior_var + (self.prior_var / self.kappa_params)
                log_pred_probs = -0.5 * np.log(2 * np.pi * predictive_vars) - \
                                 ((x_t - self.mu_params)**2 / (2 * predictive_vars))
                
                # 3. Calculate Growth Probabilities
                log_H = np.log(1.0 / self.hazard_rate)
                log_1_minus_H = np.log(1.0 - (1.0 / self.hazard_rate))
                
                log_growth_probs = self.log_R + log_pred_probs + log_1_minus_H
                
                # 4. Calculate Changepoint Probability
                log_cp_prob = logsumexp(self.log_R + log_pred_probs + log_H)
                
                # 5. Update Run Length Distribution
                new_log_R = np.append(log_cp_prob, log_growth_probs)
                
                # Normalize log probabilities
                self.log_R = new_log_R - logsumexp(new_log_R)
                
                # 6. Check for Boundary Trigger
                # If the probability of a changepoint (run length = 0) spikes
                cp_probability = np.exp(self.log_R[0])
                if cp_probability > self.change_threshold and (global_idx - self.last_emitted_boundary) >= self.min_dist:
                    new_bounds.append(global_idx)
                    self.last_emitted_boundary = global_idx
                    
                # 7. Update Sufficient Statistics
                new_kappa = self.kappa_params + 1.0
                new_mu = (self.kappa_params * self.mu_params + x_t) / new_kappa
                
                self.kappa_params = np.append(self.prior_kappa, new_kappa)
                self.mu_params = np.append(self.prior_mu, new_mu)
                
                # 8. Truncate arrays to save memory
                if len(self.log_R) > self.max_tracked_run_length:
                    self.log_R = self.log_R[:self.max_tracked_run_length]
                    self.log_R -= logsumexp(self.log_R) # Renormalize
                    self.kappa_params = self.kappa_params[:self.max_tracked_run_length]
                    self.mu_params = self.mu_params[:self.max_tracked_run_length]
                    
            self.prev_feature = feat
            
        return new_bounds

class OnlineNaiveEdgeDetector:
    """Baseline 1: Emits a boundary if frame-to-frame cosine distance exceeds a threshold."""
    def __init__(self, threshold=0.025, min_dist=20):
        self.threshold = threshold
        self.min_dist = min_dist
        self.last_emitted_boundary = -999
        self.prev_feature = None

    def process_stream(self, new_features, start_idx):
        new_bounds = []
        for i, feat in enumerate(new_features):
            global_idx = start_idx + i
            if self.prev_feature is not None:
                norm1 = np.linalg.norm(feat) + 1e-6
                norm2 = np.linalg.norm(self.prev_feature) + 1e-6
                sim = np.dot(feat, self.prev_feature) / (norm1 * norm2)
                dist = 1.0 - sim
                
                if dist > self.threshold and (global_idx - self.last_emitted_boundary) >= self.min_dist:
                    new_bounds.append(global_idx)
                    self.last_emitted_boundary = global_idx
            self.prev_feature = feat
        return new_bounds


class OnlineABDEdgeDetector:
    """Baseline 2: Online adaptation of Fast and Unsupervised Action Boundary Detection.
       Uses causal smoothing, adjacent distance, and NMS (find_peaks)."""
    def __init__(self, smooth_window=5, min_dist=20, peak_height=0.0005):
        self.smooth_window = smooth_window
        self.min_dist = min_dist
        self.peak_height = peak_height
        self.last_emitted_boundary = -999
        self.buffer = np.empty((0, 0)) 
        self.frame_indices = []
        self.dist_history = []

    def process_stream(self, new_features, start_idx):
        if len(self.buffer) == 0:
            self.buffer = new_features
        else:
            self.buffer = np.vstack([self.buffer, new_features])
        
        self.frame_indices.extend(list(range(start_idx, start_idx + len(new_features))))
        new_bounds = []
        
        last_scored_idx = self.dist_history[-1][0] if self.dist_history else -1
        
        for i in range(len(self.buffer)):
            global_frame = self.frame_indices[i]
            if global_frame <= last_scored_idx:
                continue
                
            # Need enough frames to compute two adjacent smoothed windows
            if i >= self.smooth_window:
                curr_window = self.buffer[i - self.smooth_window + 1 : i + 1]
                prev_window = self.buffer[i - self.smooth_window : i]
                
                curr_smooth = np.mean(curr_window, axis=0)
                prev_smooth = np.mean(prev_window, axis=0)
                
                norm1 = np.linalg.norm(curr_smooth) + 1e-6
                norm2 = np.linalg.norm(prev_smooth) + 1e-6
                sim = np.dot(curr_smooth, prev_smooth) / (norm1 * norm2)
                dist = 1.0 - sim
                
                self.dist_history.append((global_frame, dist))
                
        # Peak detection (NMS)
        if len(self.dist_history) > 5:
            scores_arr = np.array([x[1] for x in self.dist_history])
            frames_arr = np.array([x[0] for x in self.dist_history])
            
            peaks, _ = find_peaks(scores_arr, height=self.peak_height, distance=self.min_dist)
            
            for p_idx in peaks:
                found_frame = frames_arr[p_idx]
                if found_frame > self.last_emitted_boundary and (found_frame - self.last_emitted_boundary) >= self.min_dist:
                    new_bounds.append(found_frame)
                    self.last_emitted_boundary = found_frame
                    
        # Smart cleanup to prevent memory leaks
        if len(self.buffer) > 1000:
            keep = max(self.smooth_window * 4, 200)
            self.buffer = self.buffer[-keep:]
            self.frame_indices = self.frame_indices[-keep:]
            self.dist_history = self.dist_history[-keep:]
            
        return new_bounds

class OnlineTSMEdgeDetector:
    def __init__(self, kernel_width=10, embed_dim=1024, max_duration_frames=None,
                 peak_height=9.0, min_distance=20):
        """
        Args:
            kernel_width: Half-size of the checkerboard kernel (lookahead/history).
                          If kernel_width=10, the window is 20x20.
        """
        self.kernel_width = kernel_width
        self.window_size = kernel_width * 2
        self.max_duration_frames = max_duration_frames
        self.peak_height = peak_height
        self.min_distance = min_distance
        
        # Create the Checkerboard Kernel (Gaussian Tapered)
        self.kernel = self._create_checkerboard_kernel(kernel_width)
        self.buffer = []
        self.frame_indices = []
        self.novelty_scores = [] 
        self.detected_boundaries = []
        self.last_emitted_boundary = -999
        
    def _create_checkerboard_kernel(self, width):
        """
        Creates a Gaussian-tapered checkerboard kernel.
        Structure:
          [ 1, -1 ]
          [-1,  1 ]
        """
        # 1. Base Checkerboard (2x2 blocks of size 'width')
        cb = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((width, width)))
        
        # 2. Gaussian Taper (Radially decaying weights to focus on the center/diagonal)
        # Create a coordinate grid centered at 0
        axis = np.linspace(-1, 1, width * 2)
        x, y = np.meshgrid(axis, axis)
        gaussian = np.exp(-(x**2 + y**2) / 0.5) 
        return cb * gaussian

    def compute_local_novelty(self, features):
        """
        Computes the novelty score for the *center* of the given feature window.
        1. Compute TSM (Self-Similarity) for the window.
        2. Correlate with Checkerboard Kernel.
        """
        # Normalize features
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        feats_norm = features / norm
        
        # Compute Local TSM (S)
        # Shape: (2*width, 2*width)
        S = np.dot(feats_norm, feats_norm.T)
        
        # Correlate S with Kernel K
        # Since both are symmetric, simple element-wise mult + sum is fine
        score = np.sum(S * self.kernel)
        return score

    def process_stream(self, new_features, start_idx):
        """
        Ingest new features and detect boundaries.
        Fix: Correctly maps global processing progress to local buffer indices 
             to allow infinite stream length.
        """
        # 1. Update Buffer
        if len(self.buffer) == 0:
            self.buffer = new_features
            self.frame_indices = list(range(start_idx, start_idx + len(new_features)))
        else:
            self.buffer = np.vstack([self.buffer, new_features])
            self.frame_indices.extend(list(range(start_idx, start_idx + len(new_features))))
            
        new_bounds = []
        
        # 2. Determine Start Index relative to LOCAL buffer
        # We need to find the first frame in the buffer that we haven't scored yet.
        # The score corresponds to the center of the window.
        
        last_scored_global_idx = -1
        if self.novelty_scores:
            last_scored_global_idx = self.novelty_scores[-1][0]
            
        target_global_center = last_scored_global_idx + 1
        if not self.frame_indices: return []
            
        buffer_start_global = self.frame_indices[0]
        
        # The index in 'self.buffer' that corresponds to 'target_global_center'
        target_local_center_idx = target_global_center - buffer_start_global
        
        # To compute the score for 'target_local_center_idx', we need a window 
        # that starts at (center - kernel_width).
        start_search_i = target_local_center_idx - self.kernel_width
        
        # Clamp: We can't start before the beginning of the buffer
        start_search_i = max(0, start_search_i)
        
        # We stop when the window extends beyond the available buffer
        # Window range: [i : i + window_size]
        end_search_i = len(self.buffer) - self.window_size
        
        # 3. Compute Scores
        # Note: We use end_search_i + 1 because range is exclusive at the end
        for i in range(start_search_i, end_search_i + 1):
            window_feats = self.buffer[i : i + self.window_size]
            center_buf_idx = i + self.kernel_width
            current_global_frame = self.frame_indices[center_buf_idx]
            if current_global_frame <= last_scored_global_idx: continue
            score = self.compute_local_novelty(window_feats)
            self.novelty_scores.append((current_global_frame, score))

        # 4. Peak Detection
        if len(self.novelty_scores) > 5:
            self._detect_peaks(new_bounds)

        if self.max_duration_frames is not None and self.frame_indices:
            latest_frame = self.frame_indices[-1]
            effective_last = self.last_emitted_boundary if self.last_emitted_boundary != -999 else 0
            
            # Check if the distance since the last boundary exceeds our limit
            if (latest_frame - effective_last) >= self.max_duration_frames:
                forced_boundary = effective_last + self.max_duration_frames
                
                # Deduplication check
                if forced_boundary > self.last_emitted_boundary:
                    self.detected_boundaries.append(forced_boundary)
                    new_bounds.append(forced_boundary)
                    self.last_emitted_boundary = forced_boundary

        # 5. Cleanup Buffer (Smart Truncation)
        # We enforce a max buffer size to prevent memory explosion on long videos.
        # But we MUST keep enough overlap (history) for the next kernel calculation.
        MAX_BUFFER = 2000 
        if len(self.buffer) > MAX_BUFFER:
            keep = max(self.window_size * 4, 400) 
            self.buffer = self.buffer[-keep:]
            self.frame_indices = self.frame_indices[-keep:]
            
        return new_bounds

    def _detect_peaks(self, new_bounds_list):
        """
        Uses scipy.signal.find_peaks on the recent score history.
        """
        if len(self.novelty_scores) < 5:
            return
        search_data = self.novelty_scores
        scores_arr = np.array([x[1] for x in search_data])
        frames_arr = np.array([x[0] for x in search_data])
        peaks_local_indices, _ = find_peaks(
            scores_arr, height=self.peak_height, distance=self.min_distance)
        
        for p_idx in peaks_local_indices:
            found_frame = frames_arr[p_idx]
            
            # 1. Deduplication: Only process if we haven't emitted this frame yet
            if found_frame <= self.last_emitted_boundary:
                continue
            
            # 2. Global Distance Constraint:
            # find_peaks handles distance inside 'scores_arr', but we must
            # also respect distance relative to the *previously emitted* boundary
            # which might be outside the current lookback window.
            if (found_frame - self.last_emitted_boundary) < 20:
                continue
            
            # If valid:
            self.detected_boundaries.append(found_frame)
            new_bounds_list.append(found_frame)
            self.last_emitted_boundary = found_frame

# ==============================================================================
# 2. Variable Duration Scorer
# ==============================================================================

class VariableScorer(Scorer):
    """
    Subclass of Scorer that overrides duration costs to handle 
    variable-length segments instead of fixed counts.
    """
    def get_duration_cost(self, hypothesis, action, current_segment_duration_s):
        """
        Calculates cost based on ACTUAL accumulated time, not step counts.
        """
        if self.config.ignore_history:
            return 0.0
            
        prev_action = hypothesis['last_action']
        total_duration_cost = 0.0
        is_breaking_action = (action != prev_action and prev_action is not None and not hypothesis['last_action_blank'])
        
        if is_breaking_action and self.config.short_duration_penalty_weight > 0:
            prev_stats = self.duration_stats.get(prev_action, {})
            dist_params = prev_stats.get('dist_fit')
            if dist_params:
                duration_ran_s = hypothesis.get('continuous_duration_s', 0.0)
                from matching_utils import get_nll_from_dist
                breaking_nll = get_nll_from_dist(duration_ran_s, dist_params)
                total_duration_cost += (breaking_nll * self.config.short_duration_penalty_weight)

        # --- 2. Continuation Cost (Running Long) ---
        if action == prev_action and action != "BLANK":
            if self.config.continuation_penalty_weight > 0:
                current_duration_s = hypothesis.get('continuous_duration_s', 0.0) + current_segment_duration_s
                stats = self.duration_stats.get(action, {}).get('dist_fit')
                if stats:
                    from scipy.stats import weibull_min
                    try:
                        # Log Survival Function: log(P(T > t))
                        log_sf = weibull_min.logsf(current_duration_s, stats['shape'], loc=stats['loc'], scale=stats['scale'])
                        if np.isinf(log_sf) or np.isnan(log_sf): total_duration_cost += 50.0 
                        else: total_duration_cost += (-log_sf * self.config.continuation_penalty_weight)
                    except: total_duration_cost += 5.0

        return total_duration_cost

# ==============================================================================
# 3. Main Decoder & Utils
# ==============================================================================

def compute_global_stats(all_stats):
    """Aggregates all action stats to find global Mean and Std."""
    means = []
    stds = []
    for action_key, sub_dict in all_stats.items():
        target = sub_dict.get(action_key, sub_dict)
        if 'mean' in target: means.append(target['mean'])
        if 'std' in target: stds.append(target['std'])
    if not means: return {'mu': 0.5, 'sigma': 0.2}
    means = np.array(means); means = means[np.isfinite(means)]
    stds = np.array(stds); stds = stds[np.isfinite(stds)]
    return {'mu': float(np.mean(means)), 'sigma': float(np.mean(stds))}

def generate_hand_mask(det_data, total_feature_frames, feature_fps, det_fps, conf_thresh, fill_gap_sec, min_dur_sec):
    """
    Generates a binary mask mapped to the FEATURE timeline.
    """
    if not det_data:
        return np.ones(total_feature_frames, dtype=int)

    frame_indices = set()
    num_pattern = re.compile(r'(\d+)')
    gap_fill = int(fill_gap_sec * feature_fps)
    min_dur = int(min_dur_sec * feature_fps)
    fps_ratio = feature_fps / det_fps

    for key, dets in det_data.items():
        matches = num_pattern.findall(key)
        if not matches: continue
        raw_idx = int(matches[-1]) - 1 
        feature_idx = int(round(raw_idx * fps_ratio))
        if any(d['class'] == 'hand' and d['score'] >= conf_thresh for d in dets):
            frame_indices.add(feature_idx)

    pred_mask = np.zeros(total_feature_frames, dtype=int)
    for idx in frame_indices:
        if idx < total_feature_frames: pred_mask[idx] = 1
            
    # --- Simple Smoothing (Opening/Closing) ---
    # 1. Fill Gaps
    i = 0
    n = len(pred_mask)
    while i < n:
        if pred_mask[i] == 0:
            j = i
            while j < n and pred_mask[j] == 0: j += 1
            # If gap is small enough, fill it
            if i > 0 and j < n and (j - i) <= gap_fill:
                pred_mask[i:j] = 1
            i = j
        else: i += 1
    
    # 2. Remove Short Bursts
    i = 0
    while i < n:
        if pred_mask[i] == 1:
            j = i
            while j < n and pred_mask[j] == 1: j += 1
            # If duration is too short, kill it
            if (j - i) < min_dur:
                pred_mask[i:j] = 0
            i = j
        else: i += 1
        
    return pred_mask


def beam_search_decoder(
    test_features: np.ndarray,
    prototypes: dict,
    config: DecoderConfig,
    candidate_generator,
    scorer,
    output_dir: str,
    hand_mask: np.ndarray = None,
    global_stats: dict = None,
    distance_metric: str = 'cosine',
    max_duration_frames: int = None,
    boundary_method: str = 'tsme',
    tsm_kernel_seconds: float = 2.0,
    tsm_peak_height: float = 9.0,
    tsm_min_distance: int = 20
):
    os.makedirs(output_dir, exist_ok=True)

    if boundary_method == 'tsme':
        print("Initializing Online TSM Edge Detector...")
        detector = OnlineTSMEdgeDetector(
            kernel_width=max(2, int(tsm_kernel_seconds * config.fps)),
            max_duration_frames=max_duration_frames,
            peak_height=tsm_peak_height,
            min_distance=tsm_min_distance)
    elif boundary_method == 'abd':
        print("Initializing Online ABD Edge Detector...")
        detector = OnlineABDEdgeDetector(smooth_window=int(config.fps / 2.0))
    elif boundary_method == 'naive':
        print("Initializing Naive Edge Detector...")
        detector = OnlineNaiveEdgeDetector()
    elif boundary_method == 'zscore':
        print("Initializing Adaptive Z-Score Edge Detector...")
        detector = OnlineAdaptiveZScoreDetector()
    elif boundary_method == 'bocd':
        print("Initializing Bayesian Online Change Point Detector...")
        detector = OnlineBOCDEdgeDetector()
    else:
        raise ValueError(f"Unknown boundary_method: {boundary_method}")
    
    feature_buffer = np.empty((0, test_features.shape[1])) 
    buffer_start_global_idx = 0 
    
    hyp_id_counter = 0
    initial_hypothesis = {
        'path': [], 'path_durations': [], 'cost': 0.0, 'ranking_score': 0.0, 'last_action': None,
        'observed_actions': set(), 'id': hyp_id_counter, 'parent_id': -1, 'step': -1,
        'continuous_duration_s': 0.0, 'last_action_blank': False,
        'continuous_blank_count': 0, 'bg_suppressed': False, 'suppression_ref_cost': None,
        'accumulated_hand_penalty': 0.0,
        'last_raw_score': 0.0, 
        'clean_action_count': 0
    }
    hyp_id_counter += 1
    beam = [initial_hypothesis]
    beam_history = [beam]
    gt_history = []
    beam_selector = BeamSelector(config)

    video_id = os.path.splitext(os.path.basename(config.test_video_feature_path))[0]
    video_metadata_for_gt = {}
    try:
        video_metadata_for_gt = load_annotations_and_get_metadata(config.annotation_file_path, video_id, config.recipe)
    except: pass

    recent_best_scores = deque(maxlen=20) 
    fallback_score = global_stats['mu'] if global_stats else 0.2

    # --- Core Beam Step ---
    def _process_single_step(current_beam, micro_segment, seg_start_time, seg_end_time, current_step, current_hyp_id):
        segment_duration_s = len(micro_segment) / config.fps
        
        m_start = int(seg_start_time * config.fps)
        m_end = int(seg_end_time * config.fps)
        m_end = min(m_end, len(hand_mask))
        m_start = min(m_start, len(hand_mask))
        segment_mask = hand_mask[m_start:m_end]
        
        if len(segment_mask) < len(micro_segment):
            pad = len(micro_segment) - len(segment_mask)
            segment_mask = np.concatenate([segment_mask, np.zeros(pad, dtype=int)])
        elif len(segment_mask) > len(micro_segment):
            segment_mask = segment_mask[:len(micro_segment)]

        raw_segment_p_hand = 1.0
        if len(segment_mask) > 0: raw_segment_p_hand = np.mean(segment_mask)
        LOW_VISIBILITY_THRESH = 0.10

        is_suppression_active = False
        if config.bg_suppress_trigger:
            is_suppression_active = any(h.get('bg_suppressed', False) for h in current_beam)

        # Apply Inertia ONLY if:
        # 1. Hands are not visible
        # 2. We are not at the very start
        # 3. We are NOT currently in a suppressed background state
        if raw_segment_p_hand < LOW_VISIBILITY_THRESH and current_step > 0 and not is_suppression_active:
            next_beam = []
            next_hyp_id = current_hyp_id 
            for hyp in current_beam:
                new_hyp = pickle.loads(pickle.dumps(hyp))
                new_hyp['continuous_duration_s'] += segment_duration_s
                inertia_cost = 0.1 * segment_duration_s
                new_hyp['ranking_score'] += inertia_cost
                new_hyp['cost'] += inertia_cost
                new_hyp['step'] = current_step
                new_hyp['parent_id'] = hyp['id']
                new_hyp['id'] = next_hyp_id
                next_hyp_id += 1
                action_to_append = hyp['last_action']
                if action_to_append is None: action_to_append = "BG"
                new_hyp['path'].append(action_to_append)
                new_hyp['last_action'] = action_to_append
                new_hyp['path_durations'].append(segment_duration_s)
                next_beam.append(new_hyp)
            
            beam_history.append(next_beam)
            current_gt_actions = set()
            if video_metadata_for_gt: 
                current_gt_actions = get_gt_actions_for_segment(seg_start_time, seg_end_time, video_metadata_for_gt)
            gt_history.append(current_gt_actions)
            # if (current_step % 5 == 0):
            #     visualize_beam_state(
            #         beam_history=beam_history, 
            #         current_step=current_step, 
            #         output_dir=output_dir, 
            #         start_time=seg_start_time, 
            #         end_time=seg_end_time, 
            #         gt_actions=current_gt_actions, 
            #         gt_history=gt_history
            #     )
                
            return next_beam, next_hyp_id
        
        hands_visible = (raw_segment_p_hand > 0.5)
        filtered_segment = micro_segment
        if distance_metric == 'cosine':
            filtered_segment = micro_segment[segment_mask == 1]

        all_unique_candidates = set()
        for hyp in current_beam:
            cands = candidate_generator.get_candidates(hyp) 
            all_unique_candidates.update(cands)

        segment_match_cache = {}
        for action in all_unique_candidates:
            if distance_metric == 'cosine':
                if len(filtered_segment) == 0:
                     match_result = {'min_cost': float('inf'), 'best_proto_id': None, 'clip_path': None}
                else:
                     match_result = calculate_avg_cosine_cost(filtered_segment, prototypes.get(action, []))
            else:
                match_result = calculate_min_dtw_cost(micro_segment, prototypes.get(action, []), config.drop_penalty)
            segment_match_cache[action] = match_result

        candidate_scores = [res['min_cost'] for res in segment_match_cache.values()]
        valid_scores = [s for s in candidate_scores if s != float('inf')]
        
        reference_score = None
        if valid_scores:
            best_local = min(valid_scores)
            recent_best_scores.append(best_local)
            reference_score = best_local
        else:
            if recent_best_scores: reference_score = np.median(recent_best_scores)
            else: reference_score = fallback_score

        base_bg_cost = reference_score + config.ranking_blank_cost
        base_bg_cost = max(0.0, base_bg_cost)
        scaled_bg_cost = base_bg_cost * segment_duration_s

        potential_real_hyps = []
        potential_blank_hyps = []
        any_good_match_globally = False
        next_hyp_id = current_hyp_id

        # breakpoint()
        for hypothesis in current_beam:
            valid_real_action_count = 0
            is_prev_suppressed = hypothesis.get('bg_suppressed', False)
            candidates = candidate_generator.get_candidates(hypothesis, hands_visible)
            
            valid_scores_for_hyp = []
            for cand in candidates:
                if cand in segment_match_cache:
                    valid_scores_for_hyp.append(segment_match_cache[cand]['min_cost'])
            
            valid_scores_for_hyp.sort()
            
            # Default to infinity if no candidates
            top2_score_threshold = float('inf')
            
            if valid_scores_for_hyp:
                # If we have >= 2 items, pick the 2nd best score as threshold
                # If we only have 1 item, pick that 1 score
                idx = min(len(valid_scores_for_hyp) - 1, 1) 
                top2_score_threshold = valid_scores_for_hyp[idx]

            for action in sorted(list(candidates)):
                new_hyp = pickle.loads(pickle.dumps(hypothesis))
                
                # --- GAP SMOOTHING (A->BG->A) ---
                is_gap_fill_event = False
                gap_refund = 0.0
                gap_inject_cost = 0.0
                
                match_res = segment_match_cache[action]
                raw_score = match_res['min_cost']
                
                stats = scorer.all_stats.get(action, {}).get(action, {})
                
                # Condition 1: Current Action matches Parent's Parent (A ... BG ... A)
                # We check hypothesis['path'] history. 
                # path[-1] is BG. path[-2] should be A.
                # Only triggering if the BG gap is short (<= 2 steps)
                if hypothesis['last_action_blank'] and \
                   len(hypothesis['path']) >= 2 and \
                   hypothesis['path'][-2] == action and \
                   hypothesis['continuous_blank_count'] <= 2:

                    # Condition 2: Current "A" is not a "very bad match"
                    # Threshold = Mean + 3*Sigma
                    if stats and 'mean' in stats and 'std' in stats:
                        bad_match_thresh = stats['mean'] + (2.0 * stats['std'])
                        if raw_score < bad_match_thresh:
                            is_gap_fill_event = True
                            
                if is_gap_fill_event:
                    # 1. RETROACTIVE FIX:
                    # Modify previous BG step to look like A.
                    new_hyp['path'][-1] = action # Replace BG with A
                    new_hyp['last_action'] = action
                    new_hyp['last_action_blank'] = False
                    
                    # 2. COST REPLACEMENT:
                    # Refund the BG cost paid in previous step
                    gap_refund = hypothesis.get('step_cost', 0.0)
                    
                    # Calculate "Average Cost" of A (dataset mean)
                    avg_raw_score = stats['mean']
                    # Use scorer to convert Average Raw -> Average Cost
                    # Pass a dummy hypothesis or None because we just want the static cost of 'mean'
                    avg_cost_rate = scorer.get_dtw_cost(avg_raw_score, action, None, 1.0) 
                    gap_inject_cost = avg_cost_rate * hypothesis.get('step_duration', segment_duration_s)
                    
                    new_hyp['ranking_score'] += (gap_inject_cost - gap_refund)
                    new_hyp['cost'] += (gap_inject_cost - gap_refund)
                    
                    # 3. STATE STITCHING:
                    # We need to recover the continuous count of A *before* the BG.
                    # Walk backwards from path[-2]
                    recovered_count = 0
                    p_idx = -2
                    while abs(p_idx) <= len(new_hyp['path']) and new_hyp['path'][p_idx] == action:
                        recovered_count += 1
                        p_idx -= 1
                    
                    new_hyp['continuous_action_count'] = recovered_count + 1
                    new_hyp['continuous_duration_s'] = (recovered_count + 1) * segment_duration_s
                    new_hyp['accumulated_hand_penalty'] = 0.0

                # --- PREREQ SKIPPING / IMPUTATION LOGIC ---
                if config.allow_prereq_skipping and config.use_task_graph:
                    # Check if this action is missing prerequisites
                    missing_prereqs = candidate_generator.get_missing_prerequisites(action, hypothesis)
                    
                    if missing_prereqs:
                        # 1. Apply Penalty (Soft Constraint)
                        # We apply penalty per missing prereq, or just a flat one? Flat is safer.
                        penalty = config.prereq_miss_penalty
                        new_hyp['ranking_score'] += penalty
                        new_hyp['cost'] += penalty
                        
                        # 2. Recalibration (Imputation)
                        # Mark missing prereqs as verified so we don't block future steps
                        if 'verified_actions' not in new_hyp: new_hyp['verified_actions'] = set()
                        for mp in missing_prereqs:
                            new_hyp['verified_actions'].add(mp)
                            # Optional: Add to observed_actions too? Yes, for 'revisit' logic.
                            new_hyp['observed_actions'].add(mp)


                if action != new_hyp['last_action']:
                    new_hyp['clean_action_count'] = 0
                    new_hyp['accumulated_hand_penalty'] = 0.0
                else:
                    # Inherit the counter from parent if action is the same
                    new_hyp['clean_action_count'] = new_hyp.get('clean_action_count', 0)
                    new_hyp['accumulated_hand_penalty'] = new_hyp.get('accumulated_hand_penalty', 0.0)
                
                if 'verified_actions' not in new_hyp: new_hyp['verified_actions'] = set()
                if 'continuous_action_count' not in new_hyp: new_hyp['continuous_action_count'] = 0
                new_hyp['parent_id'] = hypothesis['id']
                new_hyp['id'] = next_hyp_id
                next_hyp_id += 1
                new_hyp['step'] = current_step
                
                if action == new_hyp['last_action']: 
                    new_hyp['continuous_duration_s'] = new_hyp.get('continuous_duration_s', 0.0) + segment_duration_s
                    new_hyp['continuous_action_count'] = new_hyp['continuous_action_count'] + 1
                else: 
                    new_hyp['continuous_duration_s'] = segment_duration_s 
                    new_hyp['continuous_action_count'] = 1

                is_clean_step = (raw_segment_p_hand > 0.25) and (not is_prev_suppressed)

                if is_clean_step:
                    new_hyp['clean_action_count'] += 1

                VERIFICATION_THRESHOLD = 2
                if new_hyp['clean_action_count'] >= VERIFICATION_THRESHOLD:
                    if action not in new_hyp['verified_actions']: new_hyp['verified_actions'].add(action)

                continuing_suppression = (is_prev_suppressed and not hands_visible)
                newly_triggered = False
                if (not is_prev_suppressed and config.bg_suppress_trigger and action == config.bg_suppress_trigger):
                    newly_triggered = True 

                is_suppressed_now = continuing_suppression or newly_triggered
                dtw_cost = 0.0
                ref_cost = None
                current_step_hand_penalty = 0.0

                match_res = segment_match_cache[action]
                raw_score = match_res['min_cost']

                missing_hand_penalty = config.mask_lambda * (1.0 - raw_segment_p_hand)
                missing_hand_penalty *= (segment_duration_s / 1.0)

                if is_suppressed_now:
                    prev_step_cost = hypothesis.get('step_cost', 0.0)
                    prev_step_dur = hypothesis.get('step_duration', 1.0)
                    if prev_step_dur <= 0: prev_step_dur = 1.0
                    cost_rate = prev_step_cost / prev_step_dur
                    dtw_cost = cost_rate * segment_duration_s
                    ref_cost = hypothesis.get('suppression_ref_cost')
                    new_hyp['bg_suppressed'] = True
                    current_step_hand_penalty = 0.0
                else:
                    if match_res['best_proto_id']:
                        new_hyp['prototype_match'] = {'proto_id': match_res['best_proto_id'], 'clip_path': match_res['clip_path'], 'raw_score': raw_score}
                    
                    if config.cost_function == 'raw': 
                        dtw_cost = raw_score * segment_duration_s
                    else:
                        dtw_cost = scorer.get_dtw_cost(raw_score, action, new_hyp, raw_segment_p_hand) * segment_duration_s

                    current_step_hand_penalty = missing_hand_penalty
                    dtw_cost += missing_hand_penalty
                    new_hyp['bg_suppressed'] = False

                    stats = scorer.all_stats.get(action, {}).get(action, {})
                    if stats.get('mean') is not None:
                            if raw_score < (stats['mean'] + config.good_match_sigma * stats['std']):
                                any_good_match_globally = True
                    
                is_valid_score = (raw_score != float('inf'))
                is_top2_candidate = (raw_score <= top2_score_threshold + 1e-6) and not (action == config.bridge_start_action)
                is_bridge_fill = (action == config.bridge_fill_action)

                if (is_top2_candidate or is_bridge_fill) and is_valid_score and new_hyp['accumulated_hand_penalty'] > 0:
                    forgiveness_amount = new_hyp['accumulated_hand_penalty']
                    new_hyp['ranking_score'] -= forgiveness_amount
                    new_hyp['cost'] -= forgiveness_amount
                    new_hyp['accumulated_hand_penalty'] = 0.0

                new_hyp['accumulated_hand_penalty'] += current_step_hand_penalty

                duration_cost = scorer.get_duration_cost(new_hyp, action, segment_duration_s)
                transition_cost = scorer.get_transition_cost(hypothesis, action)
                total_step_cost = dtw_cost + duration_cost + transition_cost

                if np.isinf(total_step_cost): continue
                
                new_hyp['path'].append(action)
                new_hyp['path_durations'].append(segment_duration_s)
                new_hyp['ranking_score'] += total_step_cost
                new_hyp['cost'] += dtw_cost
                new_hyp['last_action'] = action
                new_hyp['last_action_blank'] = False
                new_hyp['observed_actions'].add(action)
                new_hyp['step_cost'] = dtw_cost
                new_hyp['step_duration'] = segment_duration_s
                new_hyp['suppression_ref_cost'] = ref_cost
                new_hyp['last_raw_score'] = raw_score 
                
                potential_real_hyps.append(new_hyp)
                valid_real_action_count += 1

            should_skip_bg = False
            if is_prev_suppressed and not hands_visible: should_skip_bg = True
            if raw_segment_p_hand > 0.75: should_skip_bg = True
            if valid_real_action_count == 0: should_skip_bg = False
            
            if not config.ignore_action and not should_skip_bg:
                    hands_visible_in_segment = (raw_segment_p_hand > 0.5)
                    if is_prev_suppressed and not hands_visible_in_segment: pass
                    else:
                        blank_hyp = pickle.loads(pickle.dumps(hypothesis))
                        blank_hyp['parent_id'] = hypothesis['id']
                        blank_hyp['id'] = next_hyp_id
                        next_hyp_id += 1
                        blank_hyp['step'] = current_step
                        blank_hyp['path'].append("BG")
                        blank_hyp['path_durations'].append(segment_duration_s)
                        
                        if hands_visible_in_segment: is_suppressed_blank = False
                        else: is_suppressed_blank = is_prev_suppressed
                        
                        effective_p_hand = 1.0 if is_suppressed_blank else raw_segment_p_hand
                        this_step_bg_cost = scaled_bg_cost * (1 + effective_p_hand)
                        
                        blank_hyp['ranking_score'] += this_step_bg_cost
                        blank_hyp['cost'] += this_step_bg_cost
                        blank_hyp['last_action_blank'] = True
                        blank_hyp['continuous_blank_count'] += 1
                        blank_hyp['continuous_action_count'] = 0
                        blank_hyp['clean_action_count'] = 0
                        blank_hyp['bg_suppressed'] = is_suppressed_blank
                        blank_hyp['step_cost'] = this_step_bg_cost
                        blank_hyp['step_duration'] = segment_duration_s
                        blank_hyp['last_raw_score'] = 100.0 
                        
                        potential_blank_hyps.append(blank_hyp)

        next_beam_candidates = potential_real_hyps
        if any_good_match_globally:
                for hyp in potential_blank_hyps:
                    if hyp['continuous_blank_count'] <= 2: next_beam_candidates.append(hyp)
        else: next_beam_candidates.extend(potential_blank_hyps)
        new_beam = beam_selector.select_next_beam(next_beam_candidates)
        beam_history.append(new_beam)
        current_gt_actions = set()
        if video_metadata_for_gt: current_gt_actions = get_gt_actions_for_segment(seg_start_time, seg_end_time, video_metadata_for_gt)
        gt_history.append(current_gt_actions)
        # if (current_step % 5 == 0):
        #     visualize_beam_state(beam_history=beam_history, current_step=current_step, output_dir=output_dir, start_time=seg_start_time, end_time=seg_end_time, gt_actions=current_gt_actions, gt_history=gt_history)
        return new_beam, next_hyp_id

    chunk_size = int(config.fps * 1.0)
    step_counter = 0 
    print(f"Starting Stream Processing (Chunk Size: {chunk_size} frames)...")
    for stream_ptr in range(0, len(test_features), chunk_size):
        chunk = test_features[stream_ptr : stream_ptr + chunk_size]
        detected_boundaries = detector.process_stream(chunk, start_idx=stream_ptr)
        if feature_buffer.shape[0] == 0: feature_buffer = chunk
        else: feature_buffer = np.vstack([feature_buffer, chunk])
        if not detected_boundaries: continue
        for global_boundary_idx in detected_boundaries:
            cut_idx = global_boundary_idx - buffer_start_global_idx
            if cut_idx <= 0 or cut_idx > len(feature_buffer): continue
            micro_segment = feature_buffer[:cut_idx]
            seg_start_time = buffer_start_global_idx / config.fps
            seg_end_time = global_boundary_idx / config.fps
            print(f"  -> Boundary Trigger! Processing Step {step_counter}: {seg_start_time:.2f}s - {seg_end_time:.2f}s")
            beam, hyp_id_counter = _process_single_step(beam, micro_segment, seg_start_time, seg_end_time, step_counter, hyp_id_counter)
            feature_buffer = feature_buffer[cut_idx:]
            buffer_start_global_idx += cut_idx
            step_counter += 1

    if len(feature_buffer) > 0:
        remaining_frames = len(feature_buffer)
        seg_start_time = buffer_start_global_idx / config.fps
        seg_end_time = (buffer_start_global_idx + remaining_frames) / config.fps
        print(f"  -> End of Stream Flush! Processing Final Step {step_counter}: {seg_start_time:.2f}s - {seg_end_time:.2f}s")
        beam, hyp_id_counter = _process_single_step(beam, feature_buffer, seg_start_time, seg_end_time, step_counter, hyp_id_counter)
        step_counter += 1

    # if gt_history:
    #     visualize_beam_state(
    #         beam_history=beam_history,
    #         current_step=step_counter - 1,
    #         output_dir=output_dir,
    #         start_time=seg_start_time,
    #         end_time=seg_end_time,
    #         gt_actions=gt_history[-1],
    #         gt_history=gt_history
    #     )

    if not beam: return None, [], []
    final_best = sorted(beam, key=lambda h: h['ranking_score'])[0]
    return final_best, beam_history, gt_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Paths
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
    parser.add_argument('--video_dir', type=str, required=True)
    
    # Flags & Params
    parser.add_argument('--use_graph', action='store_true')
    parser.add_argument('--no_graph', action='store_false', dest='use_graph')
    parser.set_defaults(use_graph=True)
    parser.add_argument('--transition_weight', type=float, default=1.0)
    parser.add_argument('--duration_weight', type=float, default=1.0)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--segment_duration', type=float, default=3.0,
                        help='Segment length in seconds. 3.0 is the published setting.')
    parser.add_argument('--drop_penalty', type=float, default=0.5)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--ignore_action', type=str, default=None)
    parser.add_argument('--blank_penalty', type=float, default=0.6)
    parser.add_argument('--ranking_blank_cost', type=float, default=0.1)
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
    parser.add_argument('--conf_thresh', type=float, default=0.75)
    parser.add_argument('--fill_gap', type=float, default=0.2)
    parser.add_argument('--min_duration', type=float, default=0.2)
    parser.add_argument('--bg_suppress_trigger', type=str, default=None)
    parser.add_argument('--bg_suppress_trigger_count', type=int, default=2)
    parser.add_argument('--max_graph_hops', type=int, default=1)
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['dtw', 'cosine'])
    
    # Bridge Params
    parser.add_argument('--bridge_start', type=str, default=None)
    parser.add_argument('--bridge_fill', type=str, default=None)
    parser.add_argument('--bridge_end', type=str, default=None)
    parser.add_argument('--bridge_missing_thresh', type=float, default=0.5)

    # Prereq Skipping Params
    parser.add_argument('--allow_skipping', action='store_true')
    parser.add_argument('--prereq_miss_penalty', type=float, default=1.0)

    parser.add_argument('--max_segment_duration', type=float, default=None, 
                        help="Hard maximum segment duration in seconds. Triggers an automatic boundary if exceeded.")
    parser.add_argument('--tsm_kernel_seconds', type=float, default=2.0,
                        help='Checkerboard kernel half-width in seconds. Paper: 2.0 on EgoPER.')
    parser.add_argument('--tsm_peak_height', type=float, default=9.0,
                        help='Novelty peak height threshold. Paper: 9 on EgoPER.')
    parser.add_argument('--tsm_min_distance', type=int, default=20,
                        help='Minimum gap d between boundaries, in frames. Paper: 20 on EgoPER (the d of Table 4c).')
    parser.add_argument('--boundary_method', type=str, default='tsme', 
                        choices=['tsme', 'abd', 'naive', 'bocd', 'zscore'], 
                        help="Select the boundary detection algorithm for ablation.")

    args = parser.parse_args()

    test_features = np.load(args.feature_file)
    prototypes = np.load(args.prototypes_file, allow_pickle=True).item()
    with open(args.task_graph_file) as f: task_graph = json.load(f).get(args.recipe)
    with open(args.annotation_file) as f: annot_data = json.load(f)
    with open(args.stats_file) as f: all_stats = json.load(f)
    with open(args.action_duration_stats_file) as f: dur_stats = json.load(f)
    with open(args.transition_probs_file) as f: trans_probs = json.load(f).get(args.recipe, {})
    
    global_stats = compute_global_stats(all_stats)
    total_frames = int(test_features.shape[0])
    
    video_id = os.path.splitext(os.path.basename(args.feature_file))[0]
    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
    
    det_fps = 30.0 
    video_total_frames = 0
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            det_fps = cap.get(cv2.CAP_PROP_FPS)
            video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Loaded Video Metadata: {video_id} | FPS: {det_fps:.2f} | Frames: {video_total_frames}")
            cap.release()
        else:
            print(f"Warning: Could not open video file {video_path}. Using default 30.0 FPS.")
    else:
        print(f"Warning: Video file not found at {video_path}. Using default 30.0 FPS.")

    det_data = {}
    if args.detections_file and os.path.exists(args.detections_file):
        try:
            with open(args.detections_file, 'r') as f: 
                det_data = json.load(f)
            det_keys_count = len(det_data)
            if video_total_frames > 0:
                diff = abs(det_keys_count - video_total_frames)
                if diff > max(20, video_total_frames * 0.05):
                    print(f"\n[WARNING] Mismatch in detection keys vs video frames!")
        except Exception as e:
            print(f"Error loading detections: {e}")

    total_frames = int(test_features.shape[0])
    
    hand_mask = generate_hand_mask(
        det_data, 
        total_frames, 
        args.fps, 
        det_fps, 
        args.conf_thresh, 
        args.fill_gap, 
        args.min_duration
    )


    config = DecoderConfig(
        use_task_graph=args.use_graph,
        max_graph_hops=args.max_graph_hops,
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
        ranking_blank_cost=args.ranking_blank_cost,
        good_match_sigma=args.good_match_sigma,
        continuation_sigma_multiplier=args.continuation_sigma_multiplier,
        cost_function=args.cost_function,
        pruning_metric=args.pruning_metric,
        short_duration_penalty_weight=args.short_duration_penalty,
        ignore_history=args.ignore_history,
        force_distinct_actions=args.force_distinct_actions,
        mask_lambda=args.mask_lambda,
        bg_alpha=args.bg_alpha,
        true_blank_cost=0.0,
        bg_suppress_trigger=args.bg_suppress_trigger,
        bg_suppress_trigger_count=args.bg_suppress_trigger_count,
        # Bridge Params
        bridge_start_action=args.bridge_start,
        bridge_fill_action=args.bridge_fill,
        bridge_end_action=args.bridge_end,
        bridge_missing_hands_threshold=args.bridge_missing_thresh,
        # Prereq Skipping Params
        allow_prereq_skipping=args.allow_skipping,
        prereq_miss_penalty=args.prereq_miss_penalty
    )

    action_to_id = annot_data[args.recipe]['action2idx']
    id_to_action = {v: k for k, v in action_to_id.items()}
    video_id = os.path.splitext(os.path.basename(args.feature_file))[0]

    if args.ignore_action:
        video_metadata = load_annotations_and_get_metadata(args.annotation_file, video_id, args.recipe)
        test_features = filter_features_by_action(test_features, video_metadata, args.ignore_action, args.fps)

    if config.use_task_graph:
        generator = GraphCandidateGenerator(task_graph, id_to_action, action_to_id, dur_stats, args.segment_duration, args.max_graph_hops)
    else:
        all_actions = set(prototypes.keys())
        generator = FreeCandidateGenerator(all_actions)

    scorer = VariableScorer(config, duration_stats=dur_stats, transition_probs=trans_probs, all_stats=all_stats)
    max_frames = int(args.max_segment_duration * args.fps) if args.max_segment_duration else None

    start_time = time.perf_counter()

    best_hyp, history, gt_history = beam_search_decoder(
        test_features, prototypes, config, generator, scorer, args.output_dir,
        hand_mask=hand_mask, global_stats=global_stats,
        distance_metric=args.distance_metric,
        max_duration_frames=max_frames,
        boundary_method=args.boundary_method,
        tsm_kernel_seconds=args.tsm_kernel_seconds,
        tsm_peak_height=args.tsm_peak_height,
        tsm_min_distance=args.tsm_min_distance
    )

    end_time = time.perf_counter()
    inference_time_seconds = end_time - start_time
    frames_processed = len(test_features)
    fps_speed = frames_processed / inference_time_seconds if inference_time_seconds > 0 else 0

    print(f"Final Path Cost: {best_hyp['ranking_score']:.4f}")
    
    try:
        with open(args.dump_file, 'wb') as f:
            pickle.dump({
                "video_id": video_id,
                "best_hypothesis": best_hyp,
                "beam_history": history,
                "gt_history_per_segment": gt_history,
                "inference_time_seconds": inference_time_seconds,
                "frames_processed": frames_processed,
                "fps_speed": fps_speed
            }, f)
        print(f"Saved results to {args.dump_file}")
    except Exception as e:
        print(f"Error saving dump: {e}")