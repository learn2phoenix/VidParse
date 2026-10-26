# VidParse: Online Parsing of Egocentric Procedures Like a Pro

<p align="center">
  <a href="https://learn2phoenix.github.io"><b>Anubhav Gupta</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://architkam.github.io/"><b>Archit Kambhamettu</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://vatsalag99.github.io/"><b>Vatsal Agarwal</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.cs.umd.edu/~pulkit/"><b>Pulkit Kumar</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="http://www.cs.umd.edu/~abhinav/"><b>Abhinav Shrivastava</b></a>
</p>

<p align="center">
  University of Maryland, College Park
</p>

<p align="center">
  <b>ECCV 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.27562"><img src="https://img.shields.io/badge/arXiv-VidParse-red" alt="Paper PDF"></a>
  <a href="https://learn2phoenix.github.io/VidParse"><img src="https://img.shields.io/badge/Project_Page-VidParse-green" alt="Project Page"></a>
  <a href="https://github.com/robert80203/EgoPER_official"><img src="https://img.shields.io/badge/Dataset-EgoPER-blue" alt="EgoPER"></a>
</p>

<p align="center">
  <a href="https://learn2phoenix.github.io/VidParse">
    <img src="assets/parse.gif" width="100%"
      alt="A tea video playing beside its induced task graph, each step highlighting as the parser commits to it">
  </a>
</p>

<p align="center"><sub>Tea, parsed online. Filled boxes are steps already committed, the
highlighted box is the current one. First 45 seconds, playback sped up 3&times;.
<a href="https://learn2phoenix.github.io/VidParse">Full version on the project page.</a></sub></p>

---

## Overview

VidParse turns a streaming first-person video into an ordered sequence of action
steps, **online and without a single gradient update**. There is no training
loop in this repository, and no checkpoint to download, because the method never
learns any parameters of its own.

Three ideas, in the order the pipeline applies them:

1. **Manipulation-Anchored Features (MAFs).** A frozen DINOv2 ViT-L/14 gives
   patch tokens; a frozen hand–object detector gives the interaction box. We
   average only the patch tokens inside that box. The result is a training-free
   masked-attention mechanism that looks at what the hands are doing instead of
   at the kitchen behind them.
2. **Training-free boundary detection.** A Gaussian-tapered checkerboard kernel
   slides down the diagonal of an online temporal similarity matrix. Peaks in
   the resulting novelty signal are action boundaries. Nothing is fitted; the
   detector sees each frame once.
3. **Graph-constrained beam search.** Segments are scored against k-means action
   prototypes, then decoded by a beam search that may only follow edges of a
   task graph induced from the training sequences. Transitions the procedure
   does not allow are assigned infinite cost and pruned.

> **Scope.** This release covers **EgoPER** and **GTEA**. Every EgoPER number in
> the paper reproduces from this code; for GTEA, see the note under
> [Results on GTEA](#results-on-gtea).

## Results on EgoPER

Frame accuracy excluding background, edit distance, and segmental F1 at IoU
{0.1, 0.25, 0.5}, over the 53 test videos of the ProTAS split.

| Method | Inference | Train-free | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
|---|---|---|---|---|---|---|---|
| MS-TCN | offline | ✗ | 87.52 | 92.34 | 92.60 | 91.81 | 86.12 |
| MS-TCN | online | ✗ | 25.40 | 44.81 | 44.09 | 31.78 | 15.72 |
| ProTAS | online | ✗ | 76.61 | 65.50 | 64.26 | 62.59 | 51.31 |
| **VidParse** | **online** | **✓** | **80.69** | **88.69** | **91.14** | **88.53** | **77.61** |

Per recipe, from the same run:

| | coffee | oatmeal | pinwheels | quesadilla | tea |
|---|---|---|---|---|---|
| Acc | 75.82 | 80.40 | 74.58 | 83.58 | 86.42 |
| Edit | 76.45 | 91.57 | 81.78 | 91.90 | 96.74 |
| F1@0.5 | 73.37 | 69.70 | 57.94 | 89.71 | 91.97 |

What the representation buys, with the decoder held fixed:

| Representation | MAF | Level | Acc | Edit | F1@0.5 |
|---|---|---|---|---|---|
| DINOv2 ViT-L, full frame | ✗ | frame | 69.09 | 82.68 | 61.42 |
| DINOv2 ViT-B | ✓ | frame | 78.25 | 88.68 | 73.46 |
| EgoVLPv2 | ✗ | clip | 53.40 | 62.75 | 43.66 |
| EgoVLPv2 | ✓ | clip | 60.11 | 70.61 | 52.10 |
| EgoVLPv2 | ✗ | frame | 78.10 | 89.35 | 73.02 |
| EgoVLPv2 | ✓ | frame | 80.32 | 91.46 | 77.48 |
| **DINOv2 ViT-L (ours)** | **✓** | **frame** | **80.69** | **88.69** | **77.61** |

Beam width against accuracy:

| B | Acc | Edit | F1@0.5 |
|---|---|---|---|
| 1 | 58.51 | 77.50 | 52.08 |
| 3 | 76.30 | 86.03 | 72.44 |
| 5 | 78.47 | 86.89 | 74.47 |
| **10** | **80.69** | **88.69** | **77.61** |

## Results on GTEA

The four standard leave-one-subject-out splits (7 test videos each), averaged
unweighted — the same aggregation as the baseline rows.

| Method | Inference | Train-free | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
|---|---|---|---|---|---|---|---|
| MS-TCN | offline | ✗ | 79.35 | 84.46 | 86.54 | 83.79 | 71.86 |
| MS-TCN | online | ✗ | 47.28 | 60.26 | 66.75 | 60.12 | 40.29 |
| ProTAS | online | ✗ | 73.19 | 71.81 | 72.89 | 68.94 | 54.87 |
| **VidParse** | **online** | **✓** | **82.66** | **89.33** | **87.56** | **85.08** | **74.03** |

Per split:

| split (held-out subject) | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
|---|---|---|---|---|---|
| 1 (S1) | 89.51 | 88.70 | 92.59 | 90.61 | 81.25 |
| 2 (S2) | 70.90 | 83.05 | 75.44 | 71.74 | 61.44 |
| 3 (S3) | 85.92 | 93.74 | 92.87 | 91.58 | 80.25 |
| 4 (S4) | 84.29 | 91.82 | 89.35 | 86.38 | 73.16 |

This is the method as the paper describes it, with the EgoPER beam width of 10.
The GTEA column of the paper's Table 1 is split 1 alone (89.1 / 87.4 / 91.9 /
89.9 / 80.5), while the baseline rows beside it are 4-split means; this update
reports the 4-split mean, the like-for-like comparison. Split 1 differs slightly
from the paper's figure because the paper's GTEA run used an earlier
prototype-matching code path.

## Install

```bash
git clone https://github.com/learn2phoenix/VidParse.git
cd VidParse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. The pipeline is CPU-only — a GPU is needed only to extract
features (step 0 below), not to parse. `pygraphviz` needs Graphviz headers
(`apt install graphviz libgraphviz-dev` or `brew install graphviz`).

## Data

[EgoPER](https://github.com/robert80203/EgoPER_official) (Lee et al., CVPR 2024),
which we do not redistribute: its 216 normal videos, EgoPER's own
`annotation.json`, and the ProTAS split, at 10 fps. The published numbers use
that split with one coffee training video left out, `coffee_u1_a1_normal_018`
(162 training, 53 test videos); keeping it moves the 53-video mean by about
−0.2 Acc and +0.7 Edit. Lay it out as:

```
<DATA_ROOT>/
  annotation.json                                   # EgoPER's annotation file
  Videos/                                           # raw mp4s; needed for feature extraction only
  hod/<video_id>/detections.json                    # hand-object detections
  feats_dinov2_hod_enclosed_10fps/<video_id>.npy    # MAFs, one (T, 1024) array per video
  splits/combined_train.txt, combined_test.txt
```

Point `data.root` in the config at `<DATA_ROOT>` and the rest resolves relative
to it.

[GTEA](https://cbs.ic.gatech.edu/fpv/) (Fathi et al., CVPR 2011): 28 videos,
the cleaned 71-class action labels, and the standard four split bundles (as
distributed with MS-TCN), at the native 15 fps. Convert the labels once with
`scripts/convert_gtea_to_json.py --gtea_dir <GTEA_ROOT>/labels --output_json
<GTEA_ROOT>/gtea.json`, and lay it out as:

```
<GTEA_ROOT>/
  gtea.json                                         # from scripts/convert_gtea_to_json.py
  labels/<video_id>.txt                             # the 71-class GTEA labels
  Videos/<video_id>.mp4
  hod/<video_id>/detections.json
  feats_dinov2_hod_enclosed_15fps/<video_id>.npy    # MAFs, step 0 (GTEA flags below)
  splits/train.split{1..4}.bundle, test.split{1..4}.bundle
```

### Step 0 — features

MAFs are the only input the parser needs. `scripts/extract_maf_features.py` is
the extractor that produced the paper's features. Run the frozen
[hand–object detector](https://github.com/ddshan/hand_object_detector) of Shan
et al. on every source frame first; our launch script uses its ResNet-101 100K
checkpoint (`faster_rcnn_1_8_132028`) at the default thresholds, so the saved
boxes have score ≥ 0.5. For each frame sampled at true 10 fps (15
on GTEA), the extractor takes the box enclosing every saved hand and object
detection and averages the final-layer patch tokens of DINOv2 ViT-L/14
(`facebook/dinov2-large`) that the box covers. Three details matter for
reproducing the numbers exactly:

- The model sees the Hugging Face image processor's input: shortest edge 256,
  center crop 224, a 16×16 grid. The box is mapped onto that grid in full-frame
  coordinates, not adjusted for the crop, so on 16:9 video the pooled cells are
  not exactly the ones under the box.
- A frame without a detection takes the previous frame's feature; frames before
  the first detection take the first valid one.
- GTEA's detection files are keyed `frame_0001.jpg` from 1 and the paper's GTEA
  features read them from 0, so each frame took the previous frame's box. Pass
  `--fps 15 --det_key 'frame_{:04d}.jpg' --det_offset 0` to reproduce them.

```bash
python scripts/extract_maf_features.py --video_dir <DATA_ROOT>/Videos \
    --det_dir <DATA_ROOT>/hod --out_dir <DATA_ROOT>/feats_dinov2_hod_enclosed_10fps --fps 10
```

## Run

One config file, one command:

```bash
python run.py --config configs/egoper.json --stage all
python run.py --config configs/egoper.json --stage parse --workers 16
```

Stages run in order and each skips work already on disk, so an interrupted run
resumes:

| stage | what it does | output |
|---|---|---|
| `graph` | induce the task graph and transition probabilities from the **training** split | `egoper_task_graphs.json`, `transition_probs.json` |
| `prototypes` | k-means action prototypes and micro-prototypes per recipe | `prototypes/` |
| `scores` | score training segments against prototypes; collect duration/offset statistics | `segment_scores/`, `stats/` |
| `parse` | **the method**: online boundary detection + graph-constrained beam search on each test video | `parses/` |
| `eval` | segmentation metrics, per-recipe and overall | `report.xlsx` |

`--dry-run` prints the exact commands without touching anything. `--workers`
defaults to cores − 1.

GTEA runs one split at a time (`--split 1` … `4`, each into its own output
directory), or all four with their mean:

```bash
python run.py --config configs/gtea.json --split 2 --stage all
python scripts/gtea_4fold.py --config configs/gtea.json --workers 16
```

Every setting the paper varied lives in `configs/egoper.json`, and each section
of that file carries a `_note` saying what its knobs do. Nothing is hidden in the
code: the runner reads the config and passes every value explicitly.

## Reproduction

Every EgoPER row above was re-measured from this code on Python 3.13.7 (the
paper ran 3.11.9) and matches the published value to the precision the paper
prints — the main row across all 53 test videos and all five recipes, every
feature-ablation row, and the whole beam-width sweep. Prototype arrays match the
archived ones at `max |Δ| = 0`.

The GTEA table above was produced by `scripts/gtea_4fold.py` with
`configs/gtea.json`, from this code. GTEA uses the same method as EgoPER, with
its own settings: 15 fps, 3 centroids per action, a 15-frame kernel, peak height
5, minimum gap 10, 1.5 s segments. `configs/gtea.json` carries each with a
`_note`.

The rows we did **not** re-run are the MS-TCN and ProTAS baselines; those come
from the ProTAS codebase, not this one.


**The task graph is induced from the training split.** "Training-free" means no
gradient updates and no learned parameters — not that the method sees no
training data. Prototypes and the task graph are both built from training
videos; test videos inform neither.

**Two config knobs are dataset-specific priors, not general machinery.** The
config names actions explicitly:

```json
"bg_suppress_trigger": "Microwave for X seconds",
"bridge_start": "Put bowl in microwave",
"bridge_end":   "Remove bowl from microwave"
```

These bridge a step whose visual evidence is a closed microwave door — the
failure mode the paper's Limitations section names. Set them to `null` to turn
the rule off. They apply to one recipe (oatmeal) and are inert elsewhere.


## Repository layout

```
run.py                       config-driven local runner
configs/egoper.json          the published EgoPER settings
configs/gtea.json            the GTEA settings (four splits, --split 1..4)
vidparse/
  parse.py                   online boundary detection + graph-constrained beam search
  prototypes.py              k-means action prototypes
  generate_segment_scores.py segment-to-prototype scoring
  task_graphs.py             task graph induction
  build_transition_probs.py  transition statistics
  decoding_components.py     beam search cost terms
  matching_utils.py          shared matching helpers
  segmentation_metrics.py    Acc / Edit / F1@k
  evaluate.py                per-recipe and overall report
scripts/extract_maf_features.py
scripts/convert_gtea_to_json.py  GTEA labels -> the annotation format the pipeline reads
scripts/gtea_4fold.py        all four GTEA splits and their mean
tests/test_pipeline.py       kernel shape, novelty peaks, config completeness
```

## Citation

```bibtex
@inproceedings{gupta2026vidparse,
  title         = {VidParse: Online Parsing of Egocentric Procedures Like a Pro},
  author        = {Gupta, Anubhav and Kambhamettu, Archit and Agarwal, Vatsal
                   and Kumar, Pulkit and Shrivastava, Abhinav},
  booktitle     = {European Conference on Computer Vision (ECCV)},
  year          = {2026},
  eprint        = {2608.27562},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## Acknowledgments

Supported in part by NSF CAREER Award #2238769 to Abhinav Shrivastava. We thank
the authors of DINOv2, the hand–object detector, ProTAS and EgoPER, whose
released models and data this work builds on.

## License

MIT, for the code in this repository. The datasets and the frozen models it
calls carry their own licenses.
