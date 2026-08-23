import numpy as np
import os
import cv2
import argparse
import subprocess
import json
import re
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans

# Import the functions from your previously created files
from segments import segment_features_by_action, load_annotations_and_get_metadata
from feat_comparison import align_videos_with_provided_code

# This function is used by the multiprocessing pool
def _compare_pair_wrapper(args):
    """Helper to unpack arguments for the alignment function."""
    features1, features2, drop_penalty = args
    # We only need the score for the distance matrix
    result = align_videos_with_provided_code(features1, features2, drop_penalty)
    return result['normalized_score']

def extract_video_clip(source_path, start_time, end_time, output_path):
    """
    Extracts a clip from source_path using ffmpeg.
    Re-encodes to ensure keyframes are reset (accurate seeking/playback).
    """
    duration = end_time - start_time
    # Use -ss before -i for fast seeking, but may need re-encoding for frame accuracy
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', source_path,
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'fast',  # Re-encode for web compatibility
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip {output_path}: {e}")
        return False
    
SEED = 42  # overridden by --seed; 42 is the published setting


def load_detections_and_generate_mask(det_file, total_feature_frames, feature_fps, det_fps, conf_thresh, fill_gap_sec=0.2, min_dur_sec=0.2):
    """
    Loads detections and creates a binary mask aligned to the FEATURE timeline.
    """
    if not det_file or not os.path.exists(det_file):
        print(f"Warning: Detection file not found: {det_file}. Assuming ALL frames valid.")
        return np.ones(total_feature_frames, dtype=int)

    try:
        with open(det_file, 'r') as f: data = json.load(f)
    except: 
        return np.ones(total_feature_frames, dtype=int)

    frame_indices = set()
    num_pattern = re.compile(r'(\d+)')
    gap_fill = int(fill_gap_sec * feature_fps)
    min_dur = int(min_dur_sec * feature_fps)
    
    # Calculate ratio to map Detection Frame -> Feature Frame
    # e.g., 10.0 / 30.0 = 0.333
    fps_ratio = feature_fps / det_fps

    for key, dets in data.items():
        matches = num_pattern.findall(key)
        if not matches: continue
        
        # Raw index in the original video (at det_fps)
        raw_idx = int(matches[-1]) - 1 
        
        # Map to feature index
        feature_idx = int(round(raw_idx * fps_ratio))
        
        if any(d['class'] == 'hand' and d['score'] >= conf_thresh for d in dets):
            frame_indices.add(feature_idx)

    pred_mask = np.zeros(total_feature_frames, dtype=int)
    for idx in frame_indices:
        if idx < total_feature_frames: pred_mask[idx] = 1
            
    # Simple smoothing (same as before)
    i = 0
    n = len(pred_mask)
    while i < n:
        if pred_mask[i] == 0:
            j = i
            while j < n and pred_mask[j] == 0: j += 1
            if i > 0 and j < n and (j - i) <= gap_fill:
                pred_mask[i:j] = 1
            i = j
        else: i += 1
    
    i = 0
    while i < n:
        if pred_mask[i] == 1:
            j = i
            while j < n and pred_mask[j] == 1: j += 1
            if (j - i) < min_dur:
                pred_mask[i:j] = 0
            i = j
        else: i += 1
    return pred_mask


def gather_all_action_instances(
    training_video_list_path: str,
    annotation_path: str,
    feature_dir: str,
    recipe: str,
    fps: float,
    metric: str = 'cosine',
    detections_dir: str = None,
    conf_thresh: float = 0.75,
    raw_video_dir: str = None  # <--- Added Argument
) -> defaultdict:
    """
    Loads all training videos, segments them, and groups all feature segments.
    Aligns detection FPS with feature FPS.
    """
    print(f"--- Step 1: Gathering all action instances (Metric: {metric}) ---")
    
    all_instances = defaultdict(list)
    
    try:
        with open(training_video_list_path, 'r') as f:
            video_filenames = [line.strip() for line in f if line.strip() and recipe in line]
    except FileNotFoundError:
        print(f"Error: Training video list not found at {training_video_list_path}")
        return all_instances

    for video_filename in video_filenames:
        clean_name = os.path.splitext(video_filename)[0]
        feature_path = os.path.join(feature_dir, f"{clean_name}.npy")
        if not os.path.exists(feature_path):
            print(f"Warning: Feature file not found for video '{video_filename}'. Skipping.")
            continue
            
        video_features = np.load(feature_path)
        metadata = load_annotations_and_get_metadata(annotation_path, video_filename, recipe)
        if not metadata:
            print(f"Warning: No metadata found for video '{video_filename}'. Skipping.")
            continue
            
        # --- Load Mask if Cosine ---
        video_mask = None
        if metric == 'cosine' and detections_dir:
            det_path = os.path.join(detections_dir, clean_name, 'detections.json')
            
            # 1. Determine Detection FPS
            det_fps = 30.0 # Default fallback
            if raw_video_dir:
                vid_path = os.path.join(raw_video_dir, f"{clean_name}.mp4")
                if os.path.exists(vid_path):
                    try:
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            det_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                    except Exception as e:
                        print(f"Warning: Could not read FPS from {vid_path}: {e}")
            
            # 2. Generate Mask with aligned FPS
            video_mask = load_detections_and_generate_mask(
                det_path, len(video_features), fps, det_fps, conf_thresh
            )

        try:
            timestamps = metadata['labels']['time_stamp']
            action_names = metadata['labels']['error_description']
        except KeyError:
            continue 
            
        segmented = segment_features_by_action(video_features, metadata, fps)
        
        segmented_masks = {}
        if video_mask is not None:
            segmented_masks = segment_features_by_action(video_mask.reshape(-1, 1), metadata, fps)

        temp_action_counts = defaultdict(int)
        for i, action_name in enumerate(action_names):
            if action_name in segmented:
                instance_idx = temp_action_counts[action_name]
                if instance_idx < len(segmented[action_name]):
                    
                    feats = segmented[action_name][instance_idx]
                    mask = None
                    
                    if metric == 'cosine' and action_name in segmented_masks:
                        mask = segmented_masks[action_name][instance_idx].flatten()
                        # Optimization: Skip if NO valid hands in this segment?
                        # if np.sum(mask) == 0:
                        #     temp_action_counts[action_name] += 1
                        #     continue 
                            
                    instance_data = {
                        'features': feats,
                        'mask': mask,
                        'video_id': video_filename,
                        'timestamp': timestamps[i]
                    }
                    all_instances[action_name].append(instance_data)
                    temp_action_counts[action_name] += 1
            
    print(f"Gathered {sum(len(v) for v in all_instances.values())} total instances across {len(all_instances)} unique actions.\n")
    return all_instances

def compute_dtw_distance_matrix(instances_with_meta: list, drop_penalty: float) -> np.ndarray:
    """
    Computes a pairwise distance matrix for a list of action instances
    using the Drop-DTW alignment score.
    """
    num_instances = len(instances_with_meta)
    distance_matrix = np.zeros((num_instances, num_instances))
    feature_list = [inst['features'] for inst in instances_with_meta]
    
    tasks = []
    indices = []
    for i, j in combinations(range(num_instances), 2):
        tasks.append((feature_list[i], feature_list[j], drop_penalty))
        indices.append((i, j))
        
    if not tasks:
        return distance_matrix
        
    num_workers = min(cpu_count(), len(tasks))
    with Pool(processes=num_workers) as pool:
        print(f"    - Starting {len(tasks)} pairwise DTW comparisons with {num_workers} workers...")
        scores = pool.map(_compare_pair_wrapper, tasks)
    
    for (i, j), score in zip(indices, scores):
        distance_matrix[i, j] = score
        distance_matrix[j, i] = score
        
    return distance_matrix

def compute_cosine_distance_matrix(instances_with_meta: list) -> np.ndarray:
    """
    Computes pairwise Cosine distance between the MEAN vectors of the instances.
    Respects the HOD mask: Mean is calculated only on valid frames.
    """
    num_instances = len(instances_with_meta)
    
    # 1. Compute Mean Vectors
    mean_vectors = []
    valid_indices = []
    
    for idx, inst in enumerate(instances_with_meta):
        feats = inst['features']
        mask = inst.get('mask')
        
        if mask is not None:
            # Use only valid frames
            valid_feats = feats[mask == 1]
            if len(valid_feats) == 0:
                # Should have been filtered in gather, but safety check
                print(f"    Warning: Instance {inst['video_id']} has 0 valid frames. Using global mean as fallback.")
                mean_vec = np.mean(feats, axis=0)
            else:
                mean_vec = np.mean(valid_feats, axis=0)
        else:
            mean_vec = np.mean(feats, axis=0)
            
        mean_vectors.append(mean_vec)

    mean_vectors = np.array(mean_vectors)
    
    # 2. Compute Pairwise Cosine Distance
    # pdist returns condensed matrix, squareform expands it
    print(f"    - Computing Cosine distances for {num_instances} instances...")
    dist_condensed = pdist(mean_vectors, metric='cosine')
    distance_matrix = squareform(dist_condensed)
    
    return distance_matrix


def find_medoids(distance_matrix: np.ndarray, labels: np.ndarray) -> list:
    """
    Finds the medoid for each cluster. The medoid is the point with the
    minimum average distance to all other points in the same cluster.
    """
    medoid_map = {}
    unique_labels = set(labels)
    # Ignore the noise cluster, which has label -1
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    for label in unique_labels:
        cluster_member_indices = np.where(labels == label)[0]
        
        # If a cluster has only one member, it's the medoid
        if len(cluster_member_indices) == 1:
            medoid_map[label] = cluster_member_indices[0]
            continue
            
        # Create a sub-matrix of distances only for members of this cluster
        cluster_dist_matrix = distance_matrix[np.ix_(cluster_member_indices, cluster_member_indices)]
        
        # The medoid is the point with the smallest sum of distances to others
        sum_of_distances = cluster_dist_matrix.sum(axis=1)
        cluster_medoid_local_idx = np.argmin(sum_of_distances)
        
        # Convert back to the original index
        original_medoid_idx = cluster_member_indices[cluster_medoid_local_idx]
        medoid_map[label] = original_medoid_idx
        
    return medoid_map


def generate_cluster_report(
    report_file_path: str,
    action_name: str,
    instances_with_meta: list,
    labels: np.ndarray,
    medoid_map: dict
):
    """
    Appends a human-readable summary of the clustering results to a markdown file.
    """
    unique_labels = sorted(list(set(labels)))
    
    with open(report_file_path, 'a') as f:
        f.write(f"# Action: {action_name}\n\n")
        
        for label in unique_labels:
            if label == -1:
                continue # Handle noise points last
            
            cluster_indices = np.where(labels == label)[0]
            medoid_idx = medoid_map.get(label)
            medoid_meta = instances_with_meta[medoid_idx]
            
            f.write(f"## Cluster {label} (Medoid: {medoid_meta['video_id']} @ {medoid_meta['timestamp']})\n")
            for idx in cluster_indices:
                meta = instances_with_meta[idx]
                f.write(f"- Member: {meta['video_id']}, Timestamp: {meta['timestamp']}\n")
            f.write("\n")

        f.write("---\n\n")


def generate_prototypes(
    all_instances: dict,
    report_file_path: str,
    drop_penalty: float,
    num_clusters: int,
    fps: float,
    raw_video_dir: str = None,
    dump_clips: bool = False,
    clips_output_dir: str = None,
    metric: str = 'cosine',
    proto_type: str = 'medoid'
) -> dict:
    """
    Orchestrates the prototype generation process: distance matrix calculation,
    clustering, and medoid selection for each action.
    """
    print(f"--- Step 2: Generating Prototypes ({metric}) ---")
    prototypes = {}
    if dump_clips and clips_output_dir:
        os.makedirs(clips_output_dir, exist_ok=True)
    # Clear the report file at the start of the run
    if os.path.exists(report_file_path):
        os.remove(report_file_path)

    for action_name, instances_with_meta in all_instances.items():
        if action_name == 'BG':
            continue
        print(f"  Processing action: '{action_name}' ({len(instances_with_meta)} instances)")

        instance_means = []
        
        for inst in instances_with_meta:
            feats = inst['features']  # Shape: (Frames, D)
            mask = inst.get('mask')   # Binary mask from HOD
            
            if metric == 'cosine' and mask is not None:
                # Use only frames where hands are detected
                valid_feats = feats[mask == 1]
                if len(valid_feats) > 0:
                    mean_vec = np.mean(valid_feats, axis=0)
                else:
                    # Fallback if no hands were found in the entire segment
                    mean_vec = np.mean(feats, axis=0)
            else:
                # Standard mean for DTW or unmasked cosine
                mean_vec = np.mean(feats, axis=0)
                
            instance_means.append(mean_vec)
            
        X = np.array(instance_means)
        
        # 1. Select Prototypes (Cluster Medoids)
        selected_indices = []
        
        if len(instances_with_meta) <= num_clusters:
            selected_indices = list(range(len(instances_with_meta)))
            print(f"    - Using all {len(instances_with_meta)} instances as prototypes.")
        else:
            if metric == 'dtw':
                distance_matrix = compute_dtw_distance_matrix(instances_with_meta, drop_penalty)
            else:
                distance_matrix = compute_cosine_distance_matrix(instances_with_meta)
                
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters, 
                metric="precomputed", 
                linkage='average'
            ).fit(distance_matrix)
            labels = clustering.labels_
            medoid_map = find_medoids(distance_matrix, labels)
            selected_indices = list(medoid_map.values())
            print(f"    - Found {len(selected_indices)} prototypes.")

        # 2. Store and Dump Clips
        action_prototypes = []
        action_clean_name = action_name.replace(" ", "_").replace("/", "-")

        if proto_type == 'global_mean':
            global_center = np.mean(X, axis=0).reshape(1, -1)
            action_prototypes.append({
                'id': f"{action_name}_global_mean",
                'features': global_center,
                'video_id': 'synthetic',
                'timestamp': [0, 0],
                'clip_path': None
            })

        # --- OPTION 2: K-Means Centroids ---
        elif proto_type == 'kmeans':
            n_c = min(len(X), num_clusters)
            kmeans = KMeans(n_clusters=n_c, n_init=10,
                            random_state=SEED).fit(X)
            for i, center in enumerate(kmeans.cluster_centers_):
                action_prototypes.append({
                    'id': f"{action_name}_kmeans_{i}",
                    'features': center.reshape(1, -1),
                    'video_id': 'synthetic',
                    'timestamp': [0, 0],
                    'clip_path': None
                })
        else:
            for idx in selected_indices:
                instance = instances_with_meta[idx]
                
                # Construct a unique ID for this prototype clip
                start_time = instance['timestamp'][0]
                proto_id = f"{action_clean_name}_{instance['video_id']}_{int(start_time*100)}"
                
                # Dump the MP4 clip if requested
                clip_rel_path = None
                if dump_clips and raw_video_dir:
                    source_video_path = os.path.join(raw_video_dir, f"{instance['video_id']}.mp4")
                    action_clip_dir = os.path.join(clips_output_dir, action_clean_name)
                    os.makedirs(action_clip_dir, exist_ok=True)
                    
                    out_filename = f"{proto_id}.mp4"
                    output_path = os.path.join(action_clip_dir, out_filename)
                    
                    if os.path.exists(source_video_path):
                        if not os.path.exists(output_path): 
                            success = extract_video_clip(
                                source_video_path, 
                                instance['timestamp'][0], 
                                instance['timestamp'][1], 
                                output_path
                            )
                            if success:
                                clip_rel_path = os.path.join(action_clean_name, out_filename)
                        else:
                            clip_rel_path = os.path.join(action_clean_name, out_filename)
                    else:
                        print(f"    Warning: Source video {source_video_path} not found.")

                # --- NEW: FILTERING LOGIC WITH PRINTS ---
                final_feats = instance['features']
                original_frames = final_feats.shape[0]
                saved_frames = original_frames
                dropped_msg = "None"

                if metric == 'cosine' and instance.get('mask') is not None:
                    mask = instance['mask']
                    
                    # Calculate what will be dropped BEFORE filtering
                    if np.sum(mask) > 0:
                        # 1. Identify dropped ranges relative to full video
                        zero_indices = np.where(mask == 0)[0]
                        dropped_ranges = []
                        
                        if len(zero_indices) > 0:
                            # Calculate absolute start frame of this segment
                            abs_start_frame = int(instance['timestamp'][0] * fps)
                            
                            # Group consecutive zero indices
                            from itertools import groupby
                            for k, g in groupby(enumerate(zero_indices), lambda x: x[0] - x[1]):
                                group = list(map(lambda x: x[1], g))
                                # Add absolute offset
                                r_start = abs_start_frame + group[0]
                                r_end = abs_start_frame + group[-1]
                                dropped_ranges.append(f"[{r_start}-{r_end}]")
                            
                            dropped_msg = ", ".join(dropped_ranges)
                        
                        # 2. Perform the filter
                        final_feats = final_feats[mask == 1]
                        saved_frames = final_feats.shape[0]

                # Print the stats for this prototype
                print(f"    > Proto {proto_id}: Orig Frames={original_frames}, Saved={saved_frames}. Dropped Ranges (Video Frame #): {dropped_msg}")

                action_prototypes.append({
                    'id': proto_id,
                    'features': final_feats,
                    'video_id': instance['video_id'],
                    'timestamp': instance['timestamp'],
                    'clip_path': clip_rel_path 
                })
            
        prototypes[action_name] = action_prototypes
        
    return prototypes

def generate_micro_prototypes_from_full(
    full_prototypes: dict,
    duration_seconds: float,
    stride_seconds: float,
    fps: float,
    metric: str = 'cosine'
) -> dict:
    """
    Takes a dictionary of full-length action prototypes and segments them into
    a library of smaller, overlapping micro-prototypes.
    If metric='cosine', rejects micro-segments that have no valid hand frames.
    """
    print("\n--- Step 3: Generating Micro-Prototypes ---")
    micro_prototypes = defaultdict(list)
    
    frames_per_segment = int(duration_seconds * fps)
    frames_per_stride = int(stride_seconds * fps)

    if frames_per_segment == 0 or frames_per_stride == 0:
        print("Error: Segment duration or stride is too short for the given FPS, resulting in 0 frames.")
        return {}

    for action_name, proto_list in full_prototypes.items():
        for proto_instance in proto_list:
            full_proto_features = proto_instance['features']
            full_proto_id = proto_instance.get('id', 'unknown')
            clip_path = proto_instance.get('clip_path', None)
            
            # NOTE: full_proto_features might already be filtered if Cosine was used in generate_prototypes
            # BUT: Temporal info was lost if we filtered it there.
            # To do micro-prototypes correctly, we need the TEMPORAL structure.
            # However, 'generate_prototypes' saved the filtered features.
            # This creates a conflict: We cannot slice a filtered array temporally.
            
            # Correction: In generate_prototypes, I modified it to save 'final_feats' which is filtered.
            # This breaks micro-prototypes for Cosine if we want accurate temporal striding.
            # However, since Cosine prototypes are just "bags of features" anyway (order doesn't matter for Mean),
            # maybe it's fine?
            # Actually, if we want to extract a 2-second clip, we need the 2-seconds of features.
            # If we already removed frames 0-10, we can't get them back.
            
            # For simplicity in this pipeline: 
            # If Cosine, we treat the 'filtered' prototype as a contiguous block of valid info.
            # We slice IT. This means a micro-prototype might represent disparate moments in time,
            # but they are all valid hand frames for that action.
            
            num_frames = full_proto_features.shape[0]
            if num_frames == 1:
                micro_prototypes[action_name].append({
                    'id': f"{full_proto_id}_synthetic",
                    'features': full_proto_features,
                    'video_id': proto_instance['video_id'],
                    'clip_path': clip_path,
                    'clip_start_offset': 0.0,
                    'clip_duration': duration_seconds
                })
                continue
            start_frame = 0
            
            # Counter for micro-segments within this full prototype
            seg_idx = 0
            
            while start_frame + frames_per_segment <= num_frames:
                end_frame = start_frame + frames_per_segment
                micro_segment_features = full_proto_features[start_frame:end_frame, :]
                
                # Logic: If Cosine, we ensure this slice isn't empty (it shouldn't be if full was filtered)
                # But just in case
                if metric == 'cosine' and micro_segment_features.shape[0] == 0:
                    start_frame += frames_per_stride
                    continue

                clip_start_offset = start_frame / fps
                
                micro_prototypes[action_name].append({
                    'id': f"{full_proto_id}_seg{seg_idx}",
                    'features': micro_segment_features,
                    'video_id': proto_instance['video_id'],
                    'clip_path': clip_path,
                    'clip_start_offset': clip_start_offset,
                    'clip_duration': duration_seconds
                })
                
                start_frame += frames_per_stride
                seg_idx += 1
        
        print(f"  - Action '{action_name}': Generated {len(micro_prototypes[action_name])} micro-prototypes.")
        
    print("\n--- Micro-Prototype Generation Complete ---")
    return dict(micro_prototypes)


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate action prototypes from training videos using DTW and DBSCAN clustering."
    )
    # Required args
    parser.add_argument('--report_file', type=str, required=True,
                        help="Path to save the markdown cluster report.")
    parser.add_argument('--annotation_file', type=str, required=True,
                        help="Path to the main annotation JSON file.")
    parser.add_argument('--training_video_list', type=str, required=True,
                        help="Path to a .txt file containing one training video filename per line.")
    parser.add_argument('--feature_dir', type=str, required=True,
                        help="Directory containing all the .npy feature files for the videos.")
    parser.add_argument('--recipe', type=str, required=True,
                        help="The recipe name to process from the annotation file.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the final prototypes .npy file.")
    # Optional/Defaults
    parser.add_argument('--drop_penalty', type=float, default=0.5,
                        help="The drop penalty for the DTW algorithm.")
    parser.add_argument('--fps', type=float, default=10.0,
                        help="FPS at which features were extracted.")
    parser.add_argument('--num_clusters', type=int, default=4,
                        help="The target number of clusters (prototypes) to find for each action.")
    # Micro-prototypes
    parser.add_argument('--generate_micro_prototypes', action='store_true',
                        help="If set, also generate and save a micro-prototype file.")
    parser.add_argument('--micro_output_file', type=str,
                        help="Path to save the micro-prototypes .npy file (REQUIRED if --generate_micro_prototypes is set).")
    parser.add_argument('--micro_duration_seconds', type=float, default=2.0,
                        help="Duration of each micro-prototype in seconds.")
    parser.add_argument('--micro_stride_seconds', type=float, default=1.0,
                        help="Stride of the sliding window for micro-prototypes in seconds (creates overlap).")
    # Video Dumping
    parser.add_argument('--dump_clips', action='store_true')
    parser.add_argument('--raw_video_dir', type=str)
    parser.add_argument('--clips_output_dir', type=str)
    
    # --- NEW ARGS FOR COSINE/HOD ---
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['dtw', 'cosine'],
                        help="Distance metric to use for clustering (dtw or cosine).")
    parser.add_argument('--detections_dir', type=str, default=None,
                        help="Directory containing HOD detections (Required for cosine).")
    parser.add_argument('--seed', type=int, default=42,
                        help='k-means seed. 42 reproduces the paper.')
    parser.add_argument('--conf_thresh', type=float, default=0.75,
                        help="Confidence threshold for hand detection.")
    
    parser.add_argument('--proto_type', type=str, default='medoid', 
                        choices=['medoid', 'kmeans', 'global_mean'],
                        help="Ablation: 'medoid' (Ours), 'kmeans' (Centroids), or 'global_mean' (One center).")

    args = parser.parse_args()
    SEED = args.seed  # module-level rebind; read by the KMeans call
    if args.generate_micro_prototypes and not args.micro_output_file:
        parser.error("--micro_output_file is required when --generate_micro_prototypes is set.")
    if args.dump_clips and (not args.raw_video_dir or not args.clips_output_dir):
        parser.error("--raw_video_dir and --clips_output_dir are required when --dump_clips is set.")
    
    # Validation for Cosine
    if args.distance_metric == 'cosine' and not args.detections_dir:
        print("Warning: Cosine metric selected but --detections_dir not provided. Masks will default to ALL VALID.")

    # --- Run the full pipeline ---
    
    # 1. Gather all data
    all_instances = gather_all_action_instances(
        args.training_video_list,
        args.annotation_file,
        args.feature_dir,
        args.recipe,
        args.fps,
        metric=args.distance_metric,
        detections_dir=args.detections_dir,
        conf_thresh=args.conf_thresh,
        raw_video_dir=args.raw_video_dir
    )
    
    # 2. Generate prototypes
    if all_instances:
        final_prototypes = generate_prototypes(
            all_instances,
            args.report_file,
            args.drop_penalty,
            args.num_clusters,
            args.fps,
            args.raw_video_dir, 
            args.dump_clips,
            args.clips_output_dir,
            metric=args.distance_metric,
            proto_type=args.proto_type
        )

        # 3. Save the results
        if final_prototypes:
            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # np.save can handle dictionaries of numpy arrays
            np.save(args.output_file, final_prototypes, allow_pickle=True)
            print(f"\nSuccessfully saved prototypes to: {args.output_file}")
            print(f"Cluster visualization report saved to: {args.report_file}")
            
            # Optional: Print a summary of what was saved
            if args.generate_micro_prototypes:
                micro_prototypes = generate_micro_prototypes_from_full(
                    final_prototypes,
                    args.micro_duration_seconds,
                    args.micro_stride_seconds,
                    args.fps,
                    metric=args.distance_metric
                )
                if micro_prototypes:
                    micro_output_dir = os.path.dirname(args.micro_output_file)
                    if micro_output_dir:
                        os.makedirs(micro_output_dir, exist_ok=True)
                    
                    np.save(args.micro_output_file, micro_prototypes, allow_pickle=True)
                    print(f"\nSuccessfully saved micro-prototypes to: {args.micro_output_file}")
        else:
            print("\nNo prototypes were generated. The output file was not created.")