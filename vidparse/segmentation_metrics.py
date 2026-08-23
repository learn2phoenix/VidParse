import numpy as np
import Levenshtein as lev
from itertools import groupby

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["BG", "BLANK"]):
    # modified edit_score to remove consecutive duplicates after filtering out background
    recognized_no_bg = [a for a in recognized if not a in bg_class]
    ground_truth_no_bg = [a for a in ground_truth if not a in bg_class]
    P = [k for k, g in groupby(recognized_no_bg)]
    Y = [k for k, g in groupby(ground_truth_no_bg)]
    #P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

# def edit_score(recognized, ground_truth):
#     """
#     Calculates the Levenshtein edit distance between two segment sequences.
    
#     BG/BLANK segments are ignored in this calculation.
    
#     Args:
#         recognized (list): Frame-wise predictions.
#         ground_truth (list): Frame-wise ground truth.
        
#     Returns:
#         int: The normalized edit distance (0-100).
#     """
#     # Convert frame sequences to segment sequences
#     rec_segments = get_labels_start_end_time(recognized)
#     gt_segments = get_labels_start_end_time(ground_truth)
    
#     # Filter out BG/BLANK actions
#     rec_segments_filtered = [s[0] for s in rec_segments if s[0] not in ["BG", "BLANK"]]
#     gt_segments_filtered = [s[0] for s in gt_segments if s[0] not in ["BG", "BLANK"]]
    
#     if len(gt_segments_filtered) == 0:
#         if len(rec_segments_filtered) == 0:
#             return 100 # Perfect match (both empty)
#         else:
#             return 0 # All predictions are insertions
            
#     # Use python-Levenshtein to get edit distance
#     distance = lev.distance(''.join(rec_segments_filtered), ''.join(gt_segments_filtered))
    
#     # Normalize by the length of the GT sequence
#     max_len = max(len(rec_segments_filtered), len(gt_segments_filtered))
#     if max_len == 0:
#         return 100.0 # Should be covered above, but as a safeguard
        
#     # Normalized score (higher is better)
#     # This is 1 - (distance / max_len)
#     score = (1.0 - (float(distance) / max_len)) * 100
#     return score


def f_score(recognized, ground_truth, overlap_thresholds, bg_class=["BG", "BLANK"]):
    results = {}
    for overlap in overlap_thresholds:
        p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        # Calculate P, R, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[overlap] = {
            'f1': f1 * 100, # Store as percentage
            'precision': precision,
            'recall': recall
            }

    return results
        

def acc_score(recognized, ground_truth):
    """
    Calculates simple frame-wise accuracy, EXCLUDING all frames
    where the ground_truth is 'BG'.
    
    Args:
        recognized (list): Frame-wise predictions.
        ground_truth (list): Frame-wise ground truth.
        
    Returns:
        float: The accuracy score (0-100).
    """
    if len(recognized) != len(ground_truth):
        raise ValueError("Recognized and ground_truth sequences must have the same length.")
    
    if len(recognized) == 0:
        return 100.0 # Both empty
        
    filtered_rec = []
    filtered_gt = []
    
    # Filter out all frames where the ground truth is BG
    for rec, gt in zip(recognized, ground_truth):
        if gt == "BG":
            continue
            
        # If we are here, the GT is a real action.
        filtered_rec.append(rec)
        filtered_gt.append(gt)

    if not filtered_gt:
        # This means the GT video had 0 non-BG frames.
        # Check if the prediction *also* had 0 non-BG frames.
        for rec in recognized: # Check the original prediction
             if rec != "BLANK" and rec != "BG":
                 return 0.0 # We predicted an action when we shouldn't have. 100% wrong.
        return 100.0 # Both GT and Pred were all background. Perfect.
    
    # Now, calculate accuracy ONLY on the non-BG frames
    correct = 0
    for rec, gt in zip(filtered_rec, filtered_gt):
        # A "BLANK" prediction on a real action is an error.
        if rec == gt:
            correct += 1
            
    return (correct / len(filtered_gt)) * 100

