import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

# We assume the file 'exact_dp.py' is in the same directory or accessible.
# from baselines.DropDTW.dp.exact_dp import double_drop_dtw as exact_drop_dtw

def align_videos_with_provided_code(features1, features2, drop_penalty=0.5, metric='cosine'):
    """
    Aligns two video feature sequences using the provided exact_dp.py implementation.

    This function first computes a pairwise cosine distance matrix and a drop
    cost vector, then passes them to the provided Drop-DTW algorithm.

    Args:
        features1 (np.ndarray): An (M, D) numpy array for the first video (prototypes/steps).
        features2 (np.ndarray): An (N, D) numpy array for the second video (video clips).
        drop_penalty (float): A constant cost for dropping any frame from the second video.

    Returns:
        dict: A dictionary containing the alignment results:
            'normalized_score': Total cost divided by the number of matches. Lower is better.
            'raw_score': The raw alignment cost from the algorithm.
            'matches': A list of tuples [(idx1, idx2), ...] for matched frames.
            'drops1': List of dropped frame indices from video 1 (always empty for this implementation).
            'drops2': List of dropped frame indices from video 2.
    """
    if metric == 'cosine':
        if features1.size == 0 or features2.size == 0:
            return {'normalized_score': float('inf'), 'matches': []}
        
        # Calculate mean vectors (Assumes features are already masked)
        mu1 = np.mean(features1, axis=0)
        mu2 = np.mean(features2, axis=0)
        
        # Cosine distance (1 - similarity)
        dist = cosine(mu1, mu2)
        
        return {
            'normalized_score': dist,
            'raw_score': dist,
            'matches': [(0, 0)], # Dummy match for normalization logic
            'drops1': [],
            'drops2': []
        }
    
    K, D1 = features1.shape
    N, D2 = features2.shape
    assert D1 == D2, "Feature dimensions must match."

    # 1. Compute the pairwise match costs (zx_costs).
    # We use cosine distance, where lower values mean more similar features.
    # The shape will be (K, N) or (M, N).
    match_costs = cdist(features1, features2, 'cosine')

    # print(f"Max match cost: {match_costs.max():.4f}, Min match cost: {match_costs.min():.4f}")
    # 2. Compute the drop costs for the second video.
    # We'll use a constant penalty for dropping any frame.
    # The shape will be (N,).
    # drop_costs = np.full(N, fill_value=drop_penalty)
    z_drop_costs = np.full(K, fill_value=drop_penalty) # For features1
    x_drop_costs = np.full(N, fill_value=drop_penalty) # For features2

    # 3. Run the provided Drop-DTW algorithm.
    # This implementation only allows drops from the second sequence ('x').
    # raw_score, path, drops2_indices = exact_drop_dtw(
    #     zx_costs=match_costs,
    #     drop_costs=drop_costs,
    #     exclusive=True
    # )
    # raw_score, path, drops2_indices, drops1_indices = exact_drop_dtw(
    #     pairwise_zx_costs=match_costs,
    #     x_drop_costs=x_drop_costs,
    #     z_drop_costs=z_drop_costs,
    # )
    # # 4. Process the path into a more readable format.
    # # The returned path is 1-based due to DP table padding. Convert to 0-based.
    # # A match is where zi is not 0 (meaning a step is assigned).
    # matches = []
    # # The path is returned from end to start, so we reverse it.
    # for zi, xi in reversed(path):
    #     if zi > 0 and xi > 0:
    #         # Check if the previous state was a match or a drop to avoid duplicates
    #         if len(matches) == 0 or (matches[-1][0] != zi - 1 and matches[-1][1] != xi - 1):
    #              matches.append((zi - 1, xi - 1))

    # A more robust way to get matches directly from the traceback logic
    # labels = exact_drop_dtw(match_costs, drop_costs, return_labels=True)
    # matches = []
    # for frame_idx, step_id in enumerate(labels):
    #     if step_id > 0:
    #         matches.append((int(step_id) - 1, frame_idx))

    # matches = []
    # # This simplified logic assumes the path contains matched pairs; a full
    # # traceback implementation would be required for perfect accuracy.
    # # A simple approach is to find frames that are NOT dropped.
    # matched_indices1 = sorted(list(set(range(K)) - set(drops1_indices)))
    # matched_indices2 = sorted(list(set(range(N)) - set(drops2_indices)))
    
    # # This gives us which frames are matched, but not the pairing.
    # # For a placeholder, let's assume a simple 1-to-1 mapping if lengths are same
    # if len(matched_indices1) == len(matched_indices2):
    #     matches = list(zip(matched_indices1, matched_indices2))

    # 3. Run the modified Drop-DTW algorithm.
    raise NotImplementedError(
        "The Drop-DTW alignment branch is not part of the released pipeline. "
        "Every published result uses metric='cosine', which returns above. "
        "To use DTW, install Drop-DTW (https://github.com/SamsungLabs/Drop-DTW) "
        "and import double_drop_dtw as exact_drop_dtw here.")
    raw_score, labels, drops1_indices  = exact_drop_dtw(
        pairwise_zx_costs=match_costs,
        x_drop_costs=x_drop_costs,
        z_drop_costs=z_drop_costs,
    )
    
    # 4. Process the labels array to get a list of explicit matches.
    matches = []
    # A non-zero label indicates a match.
    for frame_idx, step_id in enumerate(labels):
        if step_id > 0:
            # The returned step_id is 1-based, so convert to 0-based index.
            matches.append((int(step_id) - 1, frame_idx))

    # 5. Calculate a normalized score.
    if len(matches) > 0:
        normalized_score = raw_score / len(matches)
    else:
        normalized_score = raw_score

    # For debugging, you can also find dropped frames easily.
    drops2 = [i for i, label in enumerate(labels) if label == 0]

    # breakpoint()

    # print(f"Raw_score: {raw_score}, Matches: {len(matches)}, Normalized Score: {normalized_score:.4f}")
    return {
        'normalized_score': normalized_score,
        'raw_score': raw_score,
        'matches': matches,
        'drops1': drops1_indices, # This implementation doesn't drop from the first sequence
        'drops2': drops2
    }

# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    # You will need scipy for this example:
    # pip install scipy

    # --- Create dummy video feature data ---
    D = 64
    core_action = np.random.rand(50, D)
    
    # Video 1 is the "prototype sequence"
    video1_features = core_action
    
    # Video 2 has the core action with noise at the beginning
    noise_prefix = np.random.rand(20, D) * 0.3
    video2_features = np.vstack([noise_prefix, core_action + np.random.rand(50, D) * 0.05])

    print(f"Shape of video 1 features (K, D): {video1_features.shape}")
    print(f"Shape of video 2 features (N, D): {video2_features.shape}\n")

    # --- Run the alignment ---
    # Use a drop_penalty that is lower than the likely cost of a bad match
    # (cosine distance is between 0 and 2).
    alignment = align_videos_with_provided_code(
        video1_features,
        video2_features,
        drop_penalty=0.4
    )

    # --- Print the results ---
    print("--- Alignment Results using your provided code ---")
    print(f"Normalized Score (lower is better): {alignment['normalized_score']:.4f}")
    print(f"Number of Matched Frames: {len(alignment['matches'])}")
    print(f"\nFrames Dropped from Video 1 (should be empty):")
    print(f"  -> {alignment['drops1']}")
    print(f"\nFrames Dropped from Video 2 (should be the noisy prefix):")
    print(f"  -> {alignment['drops2']}")
    print(f"\nFirst 5 Matched Frame Pairs (idx_video1, idx_video2):")
    for match in alignment['matches'][:5]:
        print(f"  -> {match}")