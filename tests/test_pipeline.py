"""Tests for the parts of VidParse that fail silently.

Not a unit-test suite for its own sake. Each test here corresponds to a way this
pipeline can produce plausible, wrong numbers without raising:

  * the boundary detector finding nothing, or everything
  * the checkerboard kernel losing its structure
  * the beam search quietly ignoring the task graph
  * a config knob not reaching the process that needs it

Run with:  python -m pytest tests -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'vidparse')
sys.path.insert(0, PKG)

import parse as vp  # noqa: E402


# ---------------------------------------------------------------------------
# boundary detection
# ---------------------------------------------------------------------------

def synthetic_stream(dims=1024, block=60, n_blocks=4, noise=0.02, seed=0):
    """A feature stream with n_blocks piecewise-constant regions."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_blocks):
        centre = rng.normal(size=dims)
        centre /= np.linalg.norm(centre)
        out.append(centre[None] + noise * rng.normal(size=(block, dims)))
    return np.concatenate(out).astype(np.float32)


def run_detector(feats, chunk=10, **kw):
    """Feed the stream through the real online entry point, in chunks."""
    det = vp.OnlineTSMEdgeDetector(**kw)
    emitted = []
    for start in range(0, len(feats), chunk):
        got = det.process_stream(feats[start:start + chunk], start)
        if got:
            emitted.extend(got if isinstance(got, (list, tuple)) else [got])
    return det, emitted


def test_checkerboard_kernel_is_balanced():
    """The kernel must sum to ~0, or novelty picks up absolute similarity."""
    det = vp.OnlineTSMEdgeDetector(kernel_width=8)
    k = det.kernel
    assert k.shape == (16, 16), k.shape
    assert abs(float(k.sum())) < 1e-6, f'kernel sums to {k.sum()}, not zero'
    # opposite quadrants share a sign; adjacent ones do not
    q = k.shape[0] // 2
    assert np.sign(k[:q, :q].sum()) == np.sign(k[q:, q:].sum())
    assert np.sign(k[:q, :q].sum()) != np.sign(k[:q, q:].sum())


def test_kernel_width_scales_the_window():
    for w in (5, 10, 20):
        det = vp.OnlineTSMEdgeDetector(kernel_width=w)
        assert det.window_size == 2 * w
        assert det.kernel.shape == (2 * w, 2 * w)


def test_detector_exposes_the_paper_knobs():
    """peak_height / min_distance must be settable -- they differ per dataset."""
    det = vp.OnlineTSMEdgeDetector(kernel_width=10, peak_height=5.0, min_distance=10)
    assert det.peak_height == 5.0
    assert det.min_distance == 10


def test_novelty_peaks_at_real_transitions():
    """A four-block stream should produce novelty maxima near the seams."""
    block = 60
    feats = synthetic_stream(block=block, n_blocks=4)
    det, _ = run_detector(feats, kernel_width=20, peak_height=0.0, min_distance=5)
    scores = det.novelty_scores
    assert scores, 'detector produced no novelty signal at all'
    frames = np.array([s[0] for s in scores])
    values = np.array([s[1] for s in scores])
    seams = [block, 2 * block, 3 * block]
    top = frames[np.argsort(values)[-3:]]
    for t in top:
        assert min(abs(t - s) for s in seams) <= 25, \
            f'novelty peak at frame {t} is not near any true seam {seams}'


def test_constant_stream_has_no_strong_novelty():
    """No transitions in, no boundaries out."""
    feats = synthetic_stream(block=240, n_blocks=1, noise=0.01)
    det, _ = run_detector(feats, kernel_width=20, peak_height=0.0, min_distance=5)
    values = np.array([s[1] for s in det.novelty_scores])
    det2, _ = run_detector(synthetic_stream(block=60, n_blocks=4),
                           kernel_width=20, peak_height=0.0, min_distance=5)
    values2 = np.array([s[1] for s in det2.novelty_scores])
    assert values.size and values2.size, 'no novelty produced'
    assert values.max() < values2.max(), \
        'a constant stream scored as novel as a segmented one'


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

CONFIGS = [os.path.join(ROOT, 'configs', f)
           for f in sorted(os.listdir(os.path.join(ROOT, 'configs')))
           if f.endswith('.json')]


@pytest.mark.parametrize('path', CONFIGS)
def test_config_has_every_knob_the_runner_passes(path):
    """A missing key means argparse's default silently replaces the paper's."""
    cfg = json.load(open(path))
    for section in ('boundary', 'prototypes', 'scoring', 'decoding', 'data'):
        assert section in cfg, f'{path} has no {section!r} section'
    for key in ('kernel_seconds', 'peak_height', 'min_distance'):
        assert key in cfg['boundary'], f'{path}: boundary.{key} missing'
    for key in ('beam_width', 'segment_duration', 'blank_penalty',
                'continuation_bonus', 'continuation_penalty',
                'short_duration_penalty', 'temporal_offset_penalty',
                'transition_penalty'):
        assert key in cfg['decoding'], (
            f'{path}: decoding.{key} missing -- parse.py defaults this to a '
            f'non-zero value, which is not what the paper ran')
    assert 'segment_duration' in cfg['scoring'], f'{path}: scoring.segment_duration missing'


@pytest.mark.parametrize('path', CONFIGS)
def test_runner_emits_every_decoding_key(path):
    """The dry run must put each configured knob on the command line."""
    sys.path.insert(0, ROOT)
    import run as runner
    cfg = json.load(open(path))
    for name in runner.SCALAR_DECODING_FLAGS:
        if name in cfg['decoding'] and cfg['decoding'][name] is not None:
            assert isinstance(cfg['decoding'][name], (int, float, str)), \
                f'{path}: decoding.{name} is not a scalar'


def test_egoper_config_matches_the_paper():
    """The published EgoPER settings, pinned so a stray edit is caught."""
    ego = json.load(open(os.path.join(ROOT, 'configs', 'egoper.json')))
    assert ego['fps'] == 10.0
    assert ego['boundary']['kernel_seconds'] == 2.0
    assert ego['boundary']['peak_height'] == 9.0
    assert ego['boundary']['min_distance'] == 20
    assert ego['prototypes']['num_clusters'] == 9
    assert ego['prototypes']['proto_type'] == 'kmeans'
    assert ego['prototypes']['micro_duration_seconds'] == 4.0
    assert ego['prototypes']['micro_stride_seconds'] == 2.0
    assert ego['decoding']['beam_width'] == 10
    assert ego['scoring']['segment_duration'] == 3.0
    assert ego['distance_metric'] == 'cosine'
    for k in ('continuation_bonus', 'continuation_penalty', 'short_duration_penalty',
              'temporal_offset_penalty', 'transition_penalty'):
        assert ego['decoding'][k] == 0.0, f'decoding.{k} must be 0.0 to match the paper'


def test_prototypes_seed_defaults_to_the_published_value():
    for path in CONFIGS:
        cfg = json.load(open(path))
        assert cfg['prototypes'].get('seed', 42) == 42, \
            f'{path} ships a seed other than the published 42'


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module', [
    'parse.py', 'prototypes.py', 'generate_segment_scores.py',
    'task_graphs.py', 'build_transition_probs.py', 'evaluate.py',
    'analyze_scores.py', 'analyze_durations.py', 'analyze_offsets.py',
])
def test_entry_point_runs(module):
    """Every stage script must at least import and parse arguments."""
    env = dict(os.environ)
    env['PYTHONPATH'] = PKG + os.pathsep + env.get('PYTHONPATH', '')
    r = subprocess.run([sys.executable, module, '--help'], cwd=PKG,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, f'{module} --help failed:\n{r.stderr[-800:]}'


def test_parse_exposes_the_lifted_constants():
    env = dict(os.environ)
    env['PYTHONPATH'] = PKG + os.pathsep + env.get('PYTHONPATH', '')
    r = subprocess.run([sys.executable, 'parse.py', '--help'], cwd=PKG,
                       capture_output=True, text=True, env=env)
    for flag in ('--tsm_kernel_seconds', '--tsm_peak_height', '--tsm_min_distance',
                 '--boundary_method'):
        assert flag in r.stdout, f'{flag} is not on the parse.py command line'
