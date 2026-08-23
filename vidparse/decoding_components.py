import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from scipy.stats import norm, weibull_min, gamma
from collections import deque, defaultdict

# Import necessary utilities
from matching_utils import get_cost_from_fitted_dist, get_nll_from_dist

@dataclass
class DecoderConfig:
    # --- Structural Switches ---
    use_task_graph: bool = True
    max_graph_hops : int = 1
    
    # --- Weights / Multipliers ---
    transition_penalty_weight: float = 1.0
    continuation_bonus_weight: float = 1.0 
    continuation_penalty_weight: float = 1.0 
    short_duration_penalty_weight: float = 1.0
    temporal_offset_penalty_weight: float = 0.5
    diversity_penalty_weight: float = 0.5
    
    # --- Core Parameters ---
    beam_width: int = 5
    segment_duration: float = 2.0
    fps: float = 10.0
    drop_penalty: float = 0.5
    
    # --- Scoring Params ---
    cost_function: str = 'nll' 
    ranking_blank_cost: float = 0.1
    true_blank_cost: float = 0.6
    good_match_sigma: float = 1.0
    continuation_sigma_multiplier: float = 1.25
    pruning_metric: str = 'ranking_score'
    distance_metric: str = 'cosine' # 'dtw' or 'cosine'

    # --- Mask / Sigma Params ---
    mask_lambda: float = 1.0
    bg_alpha: float = 0.0
    bg_suppress_trigger: str = None
    bg_suppress_trigger_count: int = 2
    
    # --- Blank Handling ---
    max_continuous_blanks: int = 2
    continuous_blank_penalty: float = 10.0

    # --- GT Logic ---
    test_video_feature_path: str = ""
    annotation_file_path: str = ""
    recipe: str = ""
    ignore_action: Optional[str] = None

    ignore_history: bool = False
    force_distinct_actions: bool = False

    # --- Bridge / Tunnel Logic ---
    bridge_start_action: Optional[str] = None # e.g., 'A'
    bridge_end_action: Optional[str] = None   # e.g., 'C'
    bridge_fill_action: Optional[str] = None # e.g., 'B' (The "Hidden Hands" action)
    bridge_confidence_sigma: float = 1.0      
    bridge_missing_hands_threshold: float = 0.5

    # --- Graph Relaxation (Smart Skipping) ---
    allow_prereq_skipping: bool = True
    prereq_miss_penalty: float = 2.0  # Cost penalty for skipping a prereq (Soft Constraint)


class CandidateGenerator:
    """Base class for determining valid next actions."""
    def get_candidates(self, hypothesis: Dict, hands_visible: bool = True) -> Set[str]:
        raise NotImplementedError
    
    def get_missing_prerequisites(self, action: str, hypothesis: Dict) -> Set[str]:
        """Returns a set of action names that are required but missing for the given action."""
        return set()


class FreeCandidateGenerator(CandidateGenerator):
    """Allows any action at any time (Graph disabled)."""
    def __init__(self, all_action_names: Set[str]):
        self.all_action_names = all_action_names

    def get_candidates(self, hypothesis: Dict, hands_visible: bool = True) -> Set[str]:
        candidates = set(self.all_action_names) - {'BG', 'BLANK'} 
        if hypothesis.get('bg_suppressed', False) and not hands_visible:
            return {hypothesis['last_action']}
        if hypothesis['last_action']:
            candidates.add(hypothesis['last_action'])
        return candidates


class GraphCandidateGenerator(CandidateGenerator):
    """Restricts actions based on the Task Graph."""
    def __init__(self, task_graph: Dict, id_to_action: Dict, action_to_id: Dict, duration_stats: Dict, segment_duration: float, max_hops: int = 1):
        self.task_graph = task_graph
        self.id_to_action = id_to_action
        self.action_to_id = action_to_id
        self.duration_stats = duration_stats
        self.segment_duration = segment_duration
        self.dependency_graph = task_graph.get('dependency_graph', {})
        self.max_hops = max_hops
        
        # Pre-compute start nodes
        self.start_node_ids = set()
        for step_id_str, details in task_graph.get('dependency_graph', {}).items():
            if [] in details.get('prerequisites', []):
                self.start_node_ids.add(int(step_id_str))
        self.start_node_names = {self.id_to_action[nid] for nid in self.start_node_ids if nid in self.id_to_action}

        self.adj = defaultdict(list)
        for u, v in self.task_graph.get('first_visit_edges', []):
            self.adj[u].append((v, 'first'))
        for u, v in self.task_graph.get('revisit_edges', []):
            self.adj[u].append((v, 'revisit'))

    def has_sufficient_duration(self, action_id: int, hypothesis: Dict) -> bool:
        """Checks if a previously observed action was seen for a sufficient duration."""
        action_name = self.id_to_action.get(action_id)
        if not action_name: return False

        observed_count = hypothesis.get('observed_action_counts', {}).get(action_name, 0)
        if observed_count == 0: return False
        
        stats = self.duration_stats.get(action_name, {})
        duration_floor_s = stats.get('25_percentile_s')

        if duration_floor_s is not None:
            observed_duration_s = observed_count * self.segment_duration
            return observed_duration_s >= duration_floor_s
        else:
            return observed_count > 1
        
    def _are_prerequisites_met(self, act_id: int, observed_ids: Set[int]) -> bool:
        """Helper to check prerequisites given a set of observed IDs."""
        prereq_option_lists = self.dependency_graph.get(str(act_id), {}).get('prerequisites', [])
        if not prereq_option_lists or prereq_option_lists == [[]]: return True

        for prereq_list in prereq_option_lists:
            if not prereq_list: return True
            current_option_met = True
            for pid in prereq_list:
                if pid in observed_ids: continue
                if self.dependency_graph.get(str(pid), {}).get('is_omittable', False): continue
                
                # Relaxation logic (check if prereq was "short" enough to be skipped)
                p_name = self.id_to_action.get(pid)
                if p_name:
                    mean_s = self.duration_stats.get(p_name, {}).get('mean_s')
                    if mean_s is not None and mean_s < (2 * self.segment_duration): continue
                
                current_option_met = False
                break 
            if current_option_met: return True
        return False
        
    def get_missing_prerequisites(self, action: str, hypothesis: Dict) -> Set[str]:
        """
        Identifies which specific prerequisites are missing for a target action.
        Used for the 'Recalibration' step.
        """
        if action not in self.action_to_id: return set()
        act_id = self.action_to_id[action]
        
        # Get already verified IDs
        verified_ids = set()
        for name in hypothesis.get('verified_actions', set()):
            if name in self.action_to_id:
                verified_ids.add(self.action_to_id[name])
                
        prereq_option_lists = self.dependency_graph.get(str(act_id), {}).get('prerequisites', [])
        if not prereq_option_lists or prereq_option_lists == [[]]: return set()

        # We assume the user wants the "best" path, but if multiple options exist,
        # we return the missing items from the *first* option list that isn't satisfied.
        # This is a simplification. Ideally, we pick the path with the fewest missing items.
        
        best_missing_set = None
        min_missing_count = float('inf')

        for prereq_list in prereq_option_lists:
            current_missing = set()
            for pid in prereq_list:
                if pid not in verified_ids:
                     # If omittable, don't count it as missing for penalty purposes, 
                     # but we might still want to return it to mark it "done".
                     # For now, stick to strict missing.
                     if not self.dependency_graph.get(str(pid), {}).get('is_omittable', False):
                        current_missing.add(pid)
            
            if len(current_missing) == 0:
                return set() # Met!
            
            if len(current_missing) < min_missing_count:
                min_missing_count = len(current_missing)
                best_missing_set = current_missing
        
        # Convert IDs back to names
        missing_names = set()
        if best_missing_set:
            for pid in best_missing_set:
                if pid in self.id_to_action:
                    missing_names.add(self.id_to_action[pid])
        
        return missing_names

    def get_candidates(self, hypothesis: Dict, hands_visible: bool = True) -> Set[str]:
        last_action = hypothesis['last_action']
        if hypothesis.get('bg_suppressed', False) and not hands_visible:
            return {last_action} if last_action else set()

        real_observed_actions = {a for a in hypothesis.get('observed_actions', set()) if a != 'BG'}
        
        if not last_action or (last_action == "BG" and not real_observed_actions):
            candidates = set(self.start_node_names)
            if last_action == "BG":
                candidates.add("BG")
            return candidates

        candidates = set()
        candidates.add(last_action)
        
        last_action_id = self.action_to_id.get(last_action)
        if last_action_id is None:
            return candidates

        touched_ids = set()
        for name in hypothesis.get('observed_actions', set()):
            if name in self.action_to_id:
                touched_ids.add(self.action_to_id[name])

        verified_ids = set()
        for name in hypothesis.get('verified_actions', set()):
            if name in self.action_to_id:
                verified_ids.add(self.action_to_id[name])

        # BFS Traversal
        queue = deque([(last_action_id, 0)])
        visited_in_search = {last_action_id}

        while queue:
            curr_id, depth = queue.popleft()
            if depth >= self.max_hops: continue

            for next_id, edge_type in self.adj.get(curr_id, []):
                if next_id in visited_in_search: continue

                is_valid_transition = False
                is_skippable_transition = False

                if edge_type == 'first':
                    if next_id not in verified_ids:
                        if self._are_prerequisites_met(next_id, touched_ids):
                            is_valid_transition = True
                        else:
                            # --- SMART SKIP LOGIC ---
                            # If direct prereqs are NOT met, we still include it as a candidate
                            # but it will be penalized downstream.
                            # We limit this to nodes that are *directly* connected to our BFS front
                            # to prevent exploding the search space.
                            is_skippable_transition = True 
                
                elif edge_type == 'revisit':
                    if next_id in touched_ids:
                         is_valid_transition = True
                
                if is_valid_transition or is_skippable_transition:
                    if next_id in self.id_to_action:
                        candidates.add(self.id_to_action[next_id])
                    
                    visited_in_search.add(next_id)
                    queue.append((next_id, depth + 1))
                
        return candidates


class Scorer:
    """Calculates costs for actions."""
    def __init__(self, config: DecoderConfig, duration_stats: Dict, transition_probs: Dict, all_stats: Dict):
        self.config = config
        self.duration_stats = duration_stats
        self.transition_probs = transition_probs
        self.all_stats = all_stats

    def _get_threshold(self, action: str, sigma_multiplier: float = 1.0) -> float:
        """Helper to get a 'Weak Match' threshold for an action."""
        stats = self.all_stats.get(action, {}).get(action)
        if not stats or 'mean' not in stats: return 100.0
        return stats['mean']

    def get_dtw_cost(self, raw_score: float, action: str, hypothesis: Dict = None, hands_visible_prob: float = 1.0) -> float:
        """
        Calculates Standard Cost + Trigger-based Bridge Overrides.
        Bypasses duration and feature-matching constraints for Action B transitions.
        """
        base_cost = 0.0
        stats = self.all_stats.get(action, {}).get(action)
        if stats is not None:
            base_cost = get_cost_from_fitted_dist(raw_score, stats, self.config.cost_function)
        else:
            base_cost = raw_score

        # Check if Bridge Configuration is active
        if not (self.config.bridge_start_action and self.config.bridge_fill_action and self.config.bridge_end_action):
            return base_cost

        if not hypothesis:
            return base_cost

        A = self.config.bridge_start_action
        B = self.config.bridge_fill_action
        C = self.config.bridge_end_action
        prev_action = hypothesis['last_action']
        
        # Threshold for "Empty Hands"
        # Even if boundaries are imperfect, low hand prob is a stronger indicator of B than features are of A.
        hands_missing = (1.0 - hands_visible_prob) > self.config.bridge_missing_hands_threshold

        # --- SCENARIO 1: A -> B (The Hard Transition) ---
        if prev_action == A and action == B:
            if hands_missing:
                # Force the transition to B regardless of visual feature match
                return 0.05 
            return base_cost

        # --- SCENARIO 2: In B (Maintenance & Exit) ---
        if prev_action == B:
            # Maintenance (B -> B): If hands are gone, B is physically certain.
            if action == B:
                if hands_missing:
                    return 0.01 # Keep cost extremely low to allow B to be very long
                else:
                    # If hands reappear, B is now unlikely. Force base_cost (likely high) 
                    # to push the beam toward C.
                    return base_cost

            # Exit to C (B -> C): Allow exit on feature match, even if hands are still partially obscured
            if action == C:
                stats_C = self.all_stats.get(C, {}).get(C, {})
                if stats_C:
                    # Use a looser threshold for C: if it's within 1 sigma of C's mean, 
                    # we allow the exit from B even if visibility is low.
                    confidence_thresh = stats_C['mean'] + stats_C['std']
                    if raw_score < confidence_thresh:
                        return base_cost 
                return base_cost + 20.0 # Block exit if C is not detected

            # Block B -> A (Physical impossibility in procedural tasks)
            if action == A:
                return base_cost + 100.0

        return base_cost

    def get_duration_cost(self, hypothesis: Dict, action: str) -> float:
        if self.config.ignore_history:
            return 0.0
        prev_action = hypothesis['last_action']
        total_duration_cost = 0.0

        is_breaking_action = (
            action != prev_action and 
            prev_action is not None and 
            not hypothesis['last_action_blank']
        )
        
        if is_breaking_action and self.config.short_duration_penalty_weight > 0:
            prev_stats = self.duration_stats.get(prev_action, {})
            dist_params = prev_stats.get('dist_fit')
            if dist_params:
                duration_ran_s = hypothesis['continuous_action_count'] * self.config.segment_duration
                breaking_nll = get_nll_from_dist(duration_ran_s, dist_params)
                total_duration_cost += (breaking_nll * self.config.short_duration_penalty_weight)

        if action == prev_action and action != "BLANK":
            if self.config.continuation_penalty_weight > 0:
                if hypothesis['last_action_blank']:
                     current_duration = (hypothesis['continuous_action_count'] + hypothesis['continuous_blank_count'] + 1) * self.config.segment_duration
                else:
                     current_duration = (hypothesis['continuous_action_count'] + 1) * self.config.segment_duration
                
                stats = self.duration_stats.get(action, {}).get('dist_fit')
                if stats:
                    try:
                        log_sf = weibull_min.logsf(current_duration, stats['shape'], loc=stats['loc'], scale=stats['scale'])
                        if np.isinf(log_sf) or np.isnan(log_sf):
                             total_duration_cost += 50.0 
                        else:
                             total_duration_cost += (-log_sf * self.config.continuation_penalty_weight)
                    except: 
                        total_duration_cost += 5.0

        return total_duration_cost

    def get_transition_cost(self, hypothesis: Dict, action: str) -> float:
        if self.config.transition_penalty_weight == 0: return 0.0
        if self.config.ignore_history:
            return 0.0
            
        prev_action = hypothesis['last_action']
        if prev_action and action != prev_action and action != "BLANK" and not hypothesis['last_action_blank']:
            prob = self.transition_probs.get(prev_action, {}).get(action, 0.0)
            if prob < 1e-9:
                return 20.0 * self.config.transition_penalty_weight
            return -np.log(prob) * self.config.transition_penalty_weight
        return 0.0


class BeamSelector:
    """
    State-based Beam Selection with Diversity Priority.
    """
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.score_key = config.pruning_metric

    def select_next_beam(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return []

        # 1. Deduplication (Always respect State Uniqueness including Count)
        best_hyp_by_state = {}
        
        for hyp in candidates:
            state_key = (hyp['last_action'], hyp['continuous_action_count'])
            if state_key not in best_hyp_by_state or \
               hyp[self.score_key] < best_hyp_by_state[state_key][self.score_key]:
                best_hyp_by_state[state_key] = hyp
        
        survivors = list(best_hyp_by_state.values())
        
        # 2. Selection Strategy
        if not self.config.force_distinct_actions:
            survivors.sort(key=lambda h: h[self.score_key])
            return survivors[:self.config.beam_width]
            
        else:
            # Diversity Priority
            grouped_by_action = defaultdict(list)
            for hyp in survivors:
                grouped_by_action[hyp['last_action']].append(hyp)
            
            for action in grouped_by_action:
                grouped_by_action[action].sort(key=lambda h: h[self.score_key])

            next_beam = []
            
            # Round 1: Top 1 from EACH action group
            first_picks = []
            for action, hyps in grouped_by_action.items():
                first_picks.append(hyps[0])
            first_picks.sort(key=lambda h: h[self.score_key])
            
            for hyp in first_picks:
                if len(next_beam) < self.config.beam_width:
                    next_beam.append(hyp)
            
            # Round 2: Fill remaining
            if len(next_beam) < self.config.beam_width:
                remaining_candidates = []
                for action, hyps in grouped_by_action.items():
                    for h in hyps[1:]:
                        remaining_candidates.append(h)
                
                remaining_candidates.sort(key=lambda h: h[self.score_key])
                slots_needed = self.config.beam_width - len(next_beam)
                next_beam.extend(remaining_candidates[:slots_needed])
                
            next_beam.sort(key=lambda h: h[self.score_key])
            return next_beam