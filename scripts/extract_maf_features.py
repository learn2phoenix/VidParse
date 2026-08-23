#!/usr/bin/env python
"""Extract Manipulation-Anchored Features (MAFs) — step 0 of VidParse.

For every sampled frame:

  1. take the hand and active-object boxes from a frozen hand-object detector,
  2. form the minimum spanning box over them (the *interaction envelope*),
  3. run a frozen DINOv2 ViT-L/14 and keep the patch tokens whose 14x14 cell
     intersects that box,
  4. average those tokens.

The result is one (T, 1024) float32 array per video: a training-free masked
attention that follows the hands rather than the room. Frames with no detection
above `--conf_thresh` fall back to the mean over all patch tokens, so the
timeline never has holes.

This is the only stage that wants a GPU. Everything downstream is CPU-only.

    python extract_maf_features.py \
        --video_dir  DATA/Videos \
        --det_dir    DATA/hod \
        --out_dir    DATA/feats_dinov2_hod_enclosed_10fps \
        --fps 10 --conf_thresh 0.75

Detections are read as one JSON per video, `<det_dir>/<video_id>/detections.json`:

    {"frame_000048.jpg": [{"class": "hand", "bbox": [x1, y1, x2, y2],
                           "score": 0.986, "side": "right", "state": 0}, ...],
     "frame_000049.jpg": [], ...}

keyed by frame of the *source* video; `--det_fps` says what rate those keys are
at, so they can be resampled onto the feature timeline. Any detector with this
output shape works; the paper uses Shan et al., CVPR 2020
(https://github.com/ddshan/hand_object_detector).
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import cv2

PATCH = 14          # DINOv2 ViT-L/14
SHORT_SIDE = 224    # resize so the short side is a whole number of patches


def spanning_box(dets, conf_thresh):
    """Minimum box covering every hand and active object above threshold."""
    keep = [d['bbox'] for d in dets
            if d.get('score', 0) >= conf_thresh and d.get('class') in ('hand', 'targetobject', 'object')]
    if not keep:
        return None
    a = np.asarray(keep, dtype=np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())


def token_mask(box, src_wh, grid_hw):
    """Boolean mask over the (gh, gw) patch grid for patches meeting `box`."""
    gh, gw = grid_hw
    if box is None:
        return np.ones(gh * gw, dtype=bool)
    W, H = src_wh
    x1, y1, x2, y2 = box
    # box -> grid coordinates
    c1, c2 = int(np.floor(x1 / W * gw)), int(np.ceil(x2 / W * gw))
    r1, r2 = int(np.floor(y1 / H * gh)), int(np.ceil(y2 / H * gh))
    m = np.zeros((gh, gw), dtype=bool)
    m[max(0, r1):min(gh, max(r1 + 1, r2)), max(0, c1):min(gw, max(c1 + 1, c2))] = True
    return m.reshape(-1) if m.any() else np.ones(gh * gw, dtype=bool)


def load_detections(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    num = re.compile(r'(\d+)')
    out = {}
    for key, dets in raw.items():
        m = num.findall(key)
        if m:
            out[int(m[-1])] = dets
    return out


def extract(video_path, det_path, model, device, fps, conf_thresh, det_fps):
    dets = load_detections(det_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {video_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det_rate = det_fps or src_fps
    stride = max(1, int(round(src_fps / fps)))

    feats, idx = [], -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue

        H, W = frame.shape[:2]
        box = spanning_box(dets.get(int(round(idx / src_fps * det_rate)), []), conf_thresh)

        scale = SHORT_SIDE / min(H, W)
        gh = int(round(H * scale / PATCH)); gw = int(round(W * scale / PATCH))
        img = cv2.resize(frame, (gw * PATCH, gh * PATCH))
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div_(255)
        x = x.permute(2, 0, 1)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        with torch.no_grad():
            out = model.forward_features(x[None].to(device))
            tokens = out['x_norm_patchtokens'][0]           # (gh*gw, 1024)

        m = token_mask(box, (W, H), (gh, gw))
        sel = tokens[torch.from_numpy(m).to(device)]
        feats.append(sel.mean(0).float().cpu().numpy())

    cap.release()
    return np.stack(feats).astype(np.float32) if feats else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--det_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fps', type=float, default=10.0,
                    help='feature rate; the paper uses 10 for EgoPER')
    ap.add_argument('--det_fps', type=float, default=None,
                    help='rate the detection keys are indexed at (default: video fps)')
    ap.add_argument('--conf_thresh', type=float, default=0.75)
    ap.add_argument('--model', default='dinov2_vitl14')
    ap.add_argument('--skip_existing', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('facebookresearch/dinov2', a.model).to(device).eval()

    videos = sorted(f for f in os.listdir(a.video_dir) if f.endswith(('.mp4', '.MP4', '.avi')))
    for i, name in enumerate(videos, 1):
        vid = os.path.splitext(name)[0]
        out = os.path.join(a.out_dir, f'{vid}.npy')
        if a.skip_existing and os.path.exists(out):
            continue
        feats = extract(os.path.join(a.video_dir, name),
                        os.path.join(a.det_dir, vid, 'detections.json'),
                        model, device, a.fps, a.conf_thresh, a.det_fps)
        if feats is None:
            print(f'[{i}/{len(videos)}] {vid}: no frames decoded, skipped')
            continue
        np.save(out, feats)
        print(f'[{i}/{len(videos)}] {vid}: {feats.shape}')


if __name__ == '__main__':
    main()
