#!/usr/bin/env python
"""GTEA, all four leave-one-subject-out splits, and their mean.

Runs run.py once per split (each split has its own output tree and skips work
already on disk), then reads each split's grand-total metrics from its eval log
and prints the per-split rows and the unweighted 4-split mean -- the same
aggregation as the baseline rows (MS-TCN, ProTAS) the paper compares against.

    python scripts/gtea_4fold.py --config configs/gtea.json --workers 16
    python scripts/gtea_4fold.py --config configs/gtea.json --report-only
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run import load_config  # noqa: E402

METRICS = [('Accuracy', 'Acc'), ('Edit Score', 'Edit'),
           ('F1@.10', 'F1@.1'), ('F1@.25', 'F1@.25'), ('F1@.50', 'F1@.5')]


def grand_total(eval_log):
    """The '** Grand Total Frame-Level Metrics' block printed by evaluate.py."""
    text = open(eval_log).read()
    block = text.split('Grand Total Frame-Level Metrics')[-1]
    out = {}
    for key, name in METRICS:
        m = re.search(r'\*\* ' + re.escape(key) + r':\s*([\d.]+)', block)
        if not m:
            sys.exit(f'{eval_log}: no {key!r} in the grand-total block')
        out[name] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'gtea.json'))
    ap.add_argument('--splits', nargs='+', default=['1', '2', '3', '4'])
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--report-only', action='store_true',
                    help='skip running; only read the existing eval logs')
    args = ap.parse_args()

    rows = {}
    for s in args.splits:
        if not args.report_only:
            cmd = [sys.executable, os.path.join(ROOT, 'run.py'), '--config', args.config,
                   '--split', s, '--stage', 'all']
            if args.workers:
                cmd += ['--workers', str(args.workers)]
            print(f'=== split {s} ===', flush=True)
            if subprocess.run(cmd).returncode:
                sys.exit(f'split {s} failed')
        out = load_config(args.config, s)['output_dir']
        rows[s] = grand_total(os.path.join(out, 'logs', 'eval.log'))

    names = [n for _, n in METRICS]
    print('\n| split | ' + ' | '.join(names) + ' |')
    print('|---|' + '---|' * len(names))
    for s, r in rows.items():
        print(f'| {s} | ' + ' | '.join(f'{r[n]:.2f}' for n in names) + ' |')
    mean = {n: sum(r[n] for r in rows.values()) / len(rows) for n in names}
    print(f'| **mean of {len(rows)}** | ' + ' | '.join(f'**{mean[n]:.2f}**' for n in names) + ' |')


if __name__ == '__main__':
    main()
