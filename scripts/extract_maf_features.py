#!/usr/bin/env python
"""Extract Manipulation-Anchored Features (MAFs) -- step 0 of VidParse.

This is the extractor that produced the paper's EgoPER and GTEA features (the
enclosing mode of the original `dino_roi_extractor.py`), with OpenCV in place of
decord for decoding. Its output matches the paper's feature files at mean cosine
>= 0.99998 on full EgoPER test videos (one per recipe) and on GTEA videos.

For every sampled frame:

  1. take every saved hand and object box from the hand-object detector,
  2. form the minimum box enclosing all of them,
  3. run DINOv2 ViT-L/14 (Hugging Face `facebook/dinov2-large`) on the frame,
     preprocessed by its image processor: shortest edge 256 (bicubic), center
     crop 224, i.e. a 16x16 patch grid,
  4. average the final-layer patch tokens of the grid cells the box covers.

Two properties to be aware of, both part of what produced the paper's numbers:

* The box is mapped onto the 16x16 grid in full-frame normalized coordinates
  (floor(x1/W*16) .. ceil(x2/W*16), inclusive, clipped to the grid). It is not
  adjusted for the center crop, so on 16:9 video the selected cells are not the
  ones that physically overlap the box except near the frame center.
* Frames with no detection take the previous frame's feature; frames before the
  first detection take the first valid feature. A video with no detection at all
  is all zeros.

Frames are sampled at true `--fps` (10 for EgoPER, 15 for GTEA): feature t comes
from source frame int(t * native_fps / fps).

    python extract_maf_features.py \
        --video_dir  DATA/Videos \
        --det_dir    DATA/hod \
        --out_dir    DATA/feats_dinov2_hod_enclosed_10fps \
        --fps 10

Detections are one JSON per video, `<det_dir>/<video_id>/detections.json`, keyed
by 0-based frame of the *source* video:

    {"frame_000048.jpg": [{"class": "hand", "bbox": [x1, y1, x2, y2],
                           "score": 0.986, "side": "right", "state": 0}, ...],
     "frame_000049.jpg": [], ...}

The paper ran the detector of Shan et al., CVPR 2020
(https://github.com/ddshan/hand_object_detector) on every source frame and kept
its saved boxes (score >= 0.5) as they are; there is no further threshold here.

GTEA's detections were run on extracted frames keyed 1-based, `frame_0001.jpg`
for the first frame, and the paper's GTEA features read those keys 0-based, so
each frame took the previous frame's box (and frame 0 none). To reproduce the
paper's GTEA features:

    python extract_maf_features.py --video_dir GTEA/Videos --det_dir GTEA/hod \
        --out_dir GTEA/feats_dinov2_hod_enclosed_15fps --fps 15 \
        --det_key 'frame_{:04d}.jpg' --det_offset 0

(`--det_offset 1` aligns the boxes with their own frames instead.)
"""
import argparse
import json
import os

import cv2
import numpy as np

GRID = 16   # 224 / 14


def sample_indices(n_frames, native_fps, fps):
    """Source frame index for each feature step, at true `fps`."""
    if fps is None or fps >= native_fps:
        return list(range(n_frames))
    duration = n_frames / native_fps
    idx = [int(i * native_fps / fps) for i in range(int(duration * fps))]
    return [i for i in idx if i < n_frames]


def grid_slice(dets, W, H):
    """Enclosing box of all detections -> inclusive (py1, py2, px1, px2) on the grid,
    in full-frame normalized coordinates (not adjusted for the center crop)."""
    x1 = min(d['bbox'][0] for d in dets); y1 = min(d['bbox'][1] for d in dets)
    x2 = max(d['bbox'][2] for d in dets); y2 = max(d['bbox'][3] for d in dets)
    px1, py1 = int(np.floor(x1 / W * GRID)), int(np.floor(y1 / H * GRID))
    px2, py2 = int(np.ceil(x2 / W * GRID)), int(np.ceil(y2 / H * GRID))
    px1, py1, px2, py2 = [max(0, min(GRID - 1, v)) for v in (px1, py1, px2, py2)]
    return py1, max(py2, py1), px1, max(px2, px1)


def fill(feats, dim):
    """Forward-fill missing steps, then back-fill the leading ones with the first
    valid feature. All-missing -> zeros."""
    out, last = [], None
    for f in feats:
        last = f if f is not None else last
        out.append(last)
    first = next((f for f in out if f is not None), np.zeros(dim, np.float32))
    return np.stack([first if f is None else f for f in out]).astype(np.float32)


def extract(video_path, detections, processor, model, device, fps, batch_size,
            det_key='frame_{:06d}.jpg', det_offset=0):
    import torch
    from PIL import Image
    torch.set_grad_enabled(False)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {video_path}')
    native = cap.get(cv2.CAP_PROP_FPS)
    idx = sample_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), native, fps)
    steps = {}
    for t, fi in enumerate(idx):
        steps.setdefault(fi, []).append(t)

    feats = [None] * len(idx)
    batch = []   # (steps, rgb frame, dets)

    def flush():
        if not batch:
            return
        inputs = processor(images=[Image.fromarray(b[1]) for b in batch], return_tensors='pt').to(device)
        tok = model(**inputs).last_hidden_state[:, 1:, :]                    # drop CLS
        tok = tok.transpose(1, 2).reshape(len(batch), -1, GRID, GRID)
        for k, (ts, rgb, dets) in enumerate(batch):
            py1, py2, px1, px2 = grid_slice(dets, rgb.shape[1], rgb.shape[0])
            f = tok[k, :, py1:py2 + 1, px1:px2 + 1].mean(dim=(1, 2)).float().cpu().numpy()
            for t in ts:
                feats[t] = f
        batch.clear()

    i, last_wanted = -1, max(idx) if idx else -1
    while i < last_wanted:                       # stream; frames are never all in memory
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        dets = detections.get(det_key.format(i + det_offset), []) if i in steps else []
        if not dets:
            continue
        batch.append((steps[i], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dets))
        if len(batch) == batch_size:
            flush()
    flush()
    cap.release()
    return fill(feats, model.config.hidden_size) if idx else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--det_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fps', type=float, default=10.0,
                    help='feature rate; the paper uses 10 for EgoPER and 15 for GTEA')
    ap.add_argument('--model', default='facebook/dinov2-large')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--det_key', default='frame_{:06d}.jpg',
                    help="detection key per source frame; GTEA's files use 'frame_{:04d}.jpg'")
    ap.add_argument('--det_offset', type=int, default=0,
                    help='added to the 0-based frame index to form the key (GTEA: 0 reproduces the paper, '
                         '1 aligns each box with its own frame)')
    ap.add_argument('--videos', nargs='*', default=None, help='video ids (default: every video in --video_dir)')
    ap.add_argument('--skip_existing', action='store_true')
    a = ap.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel
    os.makedirs(a.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()

    videos = a.videos or sorted(os.path.splitext(f)[0] for f in os.listdir(a.video_dir)
                                if f.endswith(('.mp4', '.MP4', '.avi')))
    for n, vid in enumerate(videos, 1):
        out = os.path.join(a.out_dir, f'{vid}.npy')
        if a.skip_existing and os.path.exists(out):
            continue
        path = next((os.path.join(a.video_dir, vid + e) for e in ('.mp4', '.MP4', '.avi')
                     if os.path.exists(os.path.join(a.video_dir, vid + e))), None)
        det_path = os.path.join(a.det_dir, vid, 'detections.json')
        dets = json.load(open(det_path)) if os.path.exists(det_path) else {}
        feats = extract(path, dets, processor, model, device, a.fps, a.batch_size,
                        a.det_key, a.det_offset) if path else None
        if feats is None:
            print(f'[{n}/{len(videos)}] {vid}: no video or no frames, skipped')
            continue
        np.save(out + '.tmp.npy', feats)          # rename into place, so a killed job
        os.replace(out + '.tmp.npy', out)         # never leaves a truncated file for --skip_existing
        print(f'[{n}/{len(videos)}] {vid}: {feats.shape}', flush=True)


if __name__ == '__main__':
    main()
