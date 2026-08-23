#!/usr/bin/env python
"""VidParse — config-driven local runner.

Runs the full parsing pipeline on one machine. No cluster, no scheduler: each
stage is a pool of local worker processes, and every stage skips work whose
output is already on disk, so an interrupted run resumes where it stopped.

    python run.py --config configs/egoper.json --stage all
    python run.py --config configs/egoper.json --stage parse --workers 16
    python run.py --config configs/egoper.json --stage all --dry-run

Stages, in order:

    graph        induce the task graph and transition probabilities from the
                 training split
    prototypes   k-means action prototypes (and micro-prototypes) per recipe
    scores       score every training segment against the prototypes, and
                 collect the duration / offset statistics the decoder needs
    parse        the actual method: online boundary detection + graph-constrained
                 beam search over each test video
    eval         segmentation metrics and the per-recipe report

Everything the paper varied lives in the config file; configs/egoper.json holds
the settings behind the EgoPER columns of Tables 1-3.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, 'vidparse')
STAGES = ['graph', 'prototypes', 'scores', 'parse', 'eval']


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    for key in ('dataset', 'data', 'output_dir', 'recipes'):
        if key not in cfg:
            sys.exit(f"config is missing required key: {key!r}")
    root = cfg['data']['root']
    # every data path is relative to data.root unless it is already absolute
    for k, v in cfg['data'].items():
        if k != 'root' and isinstance(v, str) and not os.path.isabs(v):
            cfg['data'][k] = os.path.join(root, v)
    cfg['output_dir'] = os.path.abspath(os.path.expanduser(cfg['output_dir']))
    return cfg


def paths(cfg):
    o = cfg['output_dir']
    p = {
        'root': o,
        'task_graph': os.path.join(o, f"{cfg['dataset']}_task_graphs.json"),
        'transitions': os.path.join(o, 'transition_probs.json'),
        'prototypes': os.path.join(o, 'prototypes'),
        'scores': os.path.join(o, 'segment_scores'),
        'stats': os.path.join(o, 'stats'),
        'parses': os.path.join(o, 'parses'),
        'viz': os.path.join(o, 'viz'),
        'report': os.path.join(o, 'report.xlsx'),
        'test_list': os.path.join(o, 'test_videos.txt'),
    }
    for k in ('root', 'prototypes', 'scores', 'stats', 'parses', 'viz'):
        os.makedirs(p[k], exist_ok=True)
    for recipe in cfg['recipes']:
        for k in ('scores', 'parses', 'viz'):
            os.makedirs(os.path.join(p[k], recipe), exist_ok=True)
    return p


def video_list(list_path, recipe):
    """Video ids for one recipe, from a split bundle or a plain list."""
    out = []
    with open(list_path) as f:
        for line in f:
            vid = line.strip().replace('.txt', '')
            if not vid:
                continue
            if recipe.lower() in vid.lower():
                out.append(vid)
    return out


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _run_one(cmd, log_path):
    """Run one command, capture its output, return (ok, cmd, log_path)."""
    with open(log_path, 'w') as log:
        r = subprocess.run(cmd, cwd=PKG, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'PYTHONPATH': PKG + os.pathsep
                                + os.environ.get('PYTHONPATH', '')})
    return r.returncode == 0, cmd, log_path


def run_pool(tasks, workers, label, dry_run=False):
    """tasks: list of (cmd, expected_output, log_path). Skips finished work."""
    todo = [t for t in tasks if not (t[1] and os.path.exists(t[1]))]
    skipped = len(tasks) - len(todo)
    print(f"[{label}] {len(tasks)} task(s), {skipped} already done, {len(todo)} to run")
    if dry_run:
        for cmd, _, _ in todo[:3]:
            print('   ', ' '.join(cmd))
        if len(todo) > 3:
            print(f'    ... and {len(todo) - 3} more')
        return True
    if not todo:
        return True

    t0 = time.time()
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, cmd, log): cmd for cmd, _, log in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ok, cmd, log = fut.result()
            if not ok:
                failures.append((cmd, log))
            print(f"\r[{label}] {i}/{len(todo)}  failures={len(failures)}", end='', flush=True)
    print(f"\r[{label}] {len(todo)}/{len(todo)} done in {time.time() - t0:.0f}s, "
          f"{len(failures)} failure(s)")

    # a task that exits 0 but writes nothing is still a failure
    missing = [t for t in todo if t[1] and not os.path.exists(t[1])]
    if missing:
        print(f"[{label}] WARNING: {len(missing)} task(s) produced no output; "
              f"first: {missing[0][1]}")
    for cmd, log in failures[:3]:
        print(f"[{label}] failed: {' '.join(cmd[:4])} ...\n          log: {log}")
    return not failures and not missing


def py(*args):
    return [sys.executable] + list(args)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_graph(cfg, p, args):
    d = cfg['data']
    logs = os.path.join(p['root'], 'logs'); os.makedirs(logs, exist_ok=True)
    tasks = [
        (py('task_graphs.py', '--dataset', cfg['dataset'],
            '--data_source', d['annotations'], '--output_dir', p['root'],
            '--train_list', d['train_list']),
         p['task_graph'], os.path.join(logs, 'graph.log')),
    ]
    if not run_pool(tasks, 1, 'graph', args.dry_run) and not args.dry_run:
        return False
    tasks = [
        (py('build_transition_probs.py', '--dataset', cfg['dataset'],
            '--data_source', d['annotations'], '--output_json', p['transitions'],
            '--train_list', d['train_list']),
         p['transitions'], os.path.join(logs, 'transitions.log')),
    ]
    return run_pool(tasks, 1, 'transitions', args.dry_run)


def stage_prototypes(cfg, p, args):
    d, j = cfg['data'], cfg['prototypes']
    logs = os.path.join(p['root'], 'logs'); os.makedirs(logs, exist_ok=True)
    tasks = []
    for recipe in cfg['recipes']:
        out = os.path.join(p['prototypes'], f'{recipe}_prototypes.npy')
        micro = os.path.join(p['prototypes'], f'{recipe}_micro_prototypes.npy')
        cmd = py('prototypes.py',
                 '--annotation_file', d['annotations'],
                 '--training_video_list', d['train_list'],
                 '--feature_dir', os.path.join(d['features'], cfg['features']['name']),
                 '--recipe', recipe,
                 '--output_file', out,
                 '--report_file', os.path.join(p['prototypes'], f'{recipe}_report.md'),
                 '--fps', str(cfg['fps']),
                 '--num_clusters', str(j['num_clusters']),
                 '--drop_penalty', str(cfg.get('drop_penalty', 0.1)),
                 '--conf_thresh', str(j.get('conf_thresh', 0.75)),
                 '--seed', str(j.get('seed', 42)),
                 '--distance_metric', cfg.get('distance_metric', 'cosine'),
                 '--detections_dir', d['detections'],
                 # required, not cosmetic: prototypes.py reads the source video's
                 # frame rate from here to map detection frames onto the feature
                 # timeline. Without it det_fps silently falls back to 30 and every
                 # hand-object mask shifts.
                 '--raw_video_dir', d.get('videos', ''))
        if j.get('proto_type'):
            cmd += ['--proto_type', j['proto_type']]
        expected = out
        if cfg['prototypes'].get('micro', True):
            cmd += ['--generate_micro_prototypes',
                    '--micro_output_file', micro,
                    '--micro_duration_seconds', str(j['micro_duration_seconds']),
                    '--micro_stride_seconds', str(j['micro_stride_seconds'])]
            expected = micro
        tasks.append((cmd, expected, os.path.join(logs, f'proto_{recipe}.log')))
    return run_pool(tasks, min(args.workers, len(tasks)), 'prototypes', args.dry_run)


def _proto_file(cfg, p, recipe):
    name = '_micro_prototypes.npy' if cfg['prototypes'].get('micro', True) else '_prototypes.npy'
    return os.path.join(p['prototypes'], recipe + name)


def stage_scores(cfg, p, args):
    d, j = cfg['data'], cfg['scoring']
    logs = os.path.join(p['root'], 'logs'); os.makedirs(logs, exist_ok=True)
    tasks = []
    for recipe in cfg['recipes']:
        for vid in video_list(d['train_list'], recipe):
            out = os.path.join(p['scores'], recipe, f'{vid}.csv')
            cmd = py('generate_segment_scores.py',
                     '--video_file', os.path.join(d['features'], cfg['features']['name'], f'{vid}.npy'),
                     '--prototypes_file', _proto_file(cfg, p, recipe),
                     '--annotation_file', d['annotations'],
                     '--recipe', recipe,
                     '--output_csv', out,
                     '--k', str(j['k']),
                     '--avg_top_n', str(j['avg_top_n']),
                     '--segment_duration', str(j['segment_duration']),
                     '--drop_penalty', str(cfg.get('drop_penalty', 0.1)),
                     '--fps', str(cfg['fps']),
                     '--distance_metric', cfg.get('distance_metric', 'cosine'),
                     '--detections_dir', d['detections'],
                     '--raw_video_dir', d.get('videos', ''),
                     '--conf_thresh', str(j.get('conf_thresh', 0.75)))
            tasks.append((cmd, out, os.path.join(logs, f'score_{vid}.log')))
    ok = run_pool(tasks, args.workers, 'scores', args.dry_run)

    # per-recipe statistics the decoder reads at parse time
    stats = []
    for recipe in cfg['recipes']:
        for script, flag, name in (
                ('analyze_scores.py', '--output_stats_json', 'score_stats'),
                ('analyze_durations.py', '--output_stats_json', 'duration_stats'),
                ('analyze_offsets.py', '--output_stats_json', 'offset_stats')):
            out = os.path.join(p['stats'], f'{recipe}_{name}.json')
            if name == 'score_stats':
                cmd = py(script, '--scores_dir', os.path.join(p['scores'], recipe), flag, out)
            else:
                cmd = py(script, '--annotation_file', cfg['data']['annotations'],
                         '--training_video_list', cfg['data']['train_list'],
                         '--recipe', recipe, flag, out)
                if name == 'offset_stats' and cfg.get('analyzers', {}).get('ignore_action'):
                    cmd += ['--ignore_action', cfg['analyzers']['ignore_action']]
            stats.append((cmd, out, os.path.join(p['root'], 'logs', f'{name}_{recipe}.log')))
    if stats:
        ok = run_pool(stats, args.workers, 'stats', args.dry_run) and ok
    return ok


def stage_parse(cfg, p, args):
    d, j, b = cfg['data'], cfg['decoding'], cfg['boundary']
    logs = os.path.join(p['root'], 'logs'); os.makedirs(logs, exist_ok=True)
    tasks, all_test = [], []
    for recipe in cfg['recipes']:
        vids = video_list(d['test_list'], recipe)
        all_test.extend(vids)
        for vid in vids:
            out = os.path.join(p['parses'], recipe, f'{vid}_results.pkl')
            cmd = py('parse.py',
                     '--feature_file', os.path.join(d['features'], cfg['features']['name'], f'{vid}.npy'),
                     '--prototypes_file', _proto_file(cfg, p, recipe),
                     '--task_graph_file', p['task_graph'],
                     '--transition_probs_file', p['transitions'],
                     '--annotation_file', d['annotations'],
                     '--recipe', recipe,
                     '--stats_file', os.path.join(p['stats'], f'{recipe}_score_stats.json'),
                     '--action_duration_stats_file', os.path.join(p['stats'], f'{recipe}_duration_stats.json'),
                     '--temporal_offset_stats_file', os.path.join(p['stats'], f'{recipe}_offset_stats.json'),
                     '--output_dir', os.path.join(p['viz'], recipe, vid),
                     '--dump_file', out,
                     '--video_dir', d.get('videos', ''),
                     '--detections_file', os.path.join(d['detections'], vid, 'detections.json'),
                     '--fps', str(cfg['fps']),
                     '--distance_metric', cfg.get('distance_metric', 'cosine'),
                     # --- boundary detector (source constants in the paper) ---
                     '--boundary_method', b.get('method', 'tsme'),
                     '--tsm_kernel_seconds', str(b['kernel_seconds']),
                     '--tsm_peak_height', str(b['peak_height']),
                     '--tsm_min_distance', str(b['min_distance']),
                     # --- beam search ---
                     '--drop_penalty', str(cfg.get('drop_penalty', 0.1)))
            cmd += ['--use_graph'] if j.get('use_graph', True) else ['--no_graph']
            # Pass every numeric knob the config names, explicitly. parse.py's
            # argparse defaults are NOT the published settings -- five of the
            # penalty terms default non-zero and the paper sets them to 0.0, so
            # anything omitted here silently changes the result.
            for name in SCALAR_DECODING_FLAGS:
                if name in j and j[name] is not None:
                    cmd += [f'--{name}', str(j[name])]
            for name in BOOL_DECODING_FLAGS:
                if j.get(name):
                    cmd += [f'--{name}']
            for name in STRING_DECODING_FLAGS:
                if j.get(name):
                    cmd += [f'--{name}', str(j[name])]
            if cfg.get('feature_chunk_size'):
                cmd += ['--feature_chunk_size', str(cfg['feature_chunk_size'])]
            tasks.append((cmd, out, os.path.join(logs, f'parse_{vid}.log')))
    if not args.dry_run:
        with open(p['test_list'], 'w') as f:
            f.write('\n'.join(sorted(set(all_test))))
    return run_pool(tasks, args.workers, 'parse', args.dry_run)


def stage_eval(cfg, p, args):
    logs = os.path.join(p['root'], 'logs'); os.makedirs(logs, exist_ok=True)
    cmd = py('evaluate.py',
             '--results_dir', p['parses'],
             '--video_list_file', cfg['data']['test_list'],
             '--annotation_file', cfg['data']['annotations'],
             '--fps', str(cfg['fps']),
             '--output_excel_file', p['report'])
    if cfg.get('eval', {}).get('ignore_action'):
        cmd += ['--ignore_action', cfg['eval']['ignore_action']]
    return run_pool([(cmd, p['report'], os.path.join(logs, 'eval.log'))],
                    1, 'eval', args.dry_run)


# Every scalar knob parse.py accepts. The config is the single source of truth;
# relying on argparse defaults for any of these silently departs from the paper.
SCALAR_DECODING_FLAGS = [
    'beam_width', 'segment_duration', 'blank_penalty', 'continuation_bonus',
    'continuation_penalty', 'short_duration_penalty', 'temporal_offset_penalty',
    'transition_penalty', 'good_match_sigma', 'continuation_sigma_multiplier',
    'cost_function', 'pruning_metric', 'mask_lambda', 'bg_alpha', 'conf_thresh',
    'fill_gap', 'min_duration', 'max_graph_hops', 'max_segment_duration',
    'transition_weight', 'duration_weight', 'prereq_miss_penalty',
    'bg_suppress_trigger_count', 'bridge_missing_thresh',
]
BOOL_DECODING_FLAGS = ['force_distinct_actions', 'allow_skipping', 'ignore_history']
STRING_DECODING_FLAGS = ['bg_suppress_trigger', 'bridge_start', 'bridge_fill', 'bridge_end']


RUNNERS = {'graph': stage_graph, 'prototypes': stage_prototypes,
           'scores': stage_scores, 'parse': stage_parse, 'eval': stage_eval}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage', default='all', choices=STAGES + ['all'])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help='parallel worker processes (default: cores - 1)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would run, touch nothing')
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)
    print(f"dataset={cfg['dataset']}  features={cfg['features']['name']}  "
          f"fps={cfg['fps']}  workers={args.workers}")
    print(f"output -> {p['root']}\n")

    todo = STAGES if args.stage == 'all' else [args.stage]
    for name in todo:
        if not RUNNERS[name](cfg, p, args):
            print(f"\nstage {name!r} did not complete cleanly; stopping.")
            print(f"logs are under {os.path.join(p['root'], 'logs')}")
            return 1
    print(f"\ndone. report: {p['report']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
