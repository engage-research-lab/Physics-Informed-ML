"""Evaluates a final checkpoint on the 232-scenario test set, or on the 3-scenario sample.

    python scripts/evaluate.py --model A             full test set (needs the full prepared dataset)
    python scripts/evaluate.py --model A --sample    the 3 sample scenarios in data/ (no download needed)

Writes results/metrics/model_a_test_results.json (or model_b_...). With --sample it writes
model_a_sample_results.json instead, so the published test-set results are not overwritten.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, 'data', 'sample_3_scenarios.npz')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', choices=['A', 'B'], default='A')
parser.add_argument('--sample', action='store_true', help='use the 3 sample scenarios shipped in data/')
parser.add_argument('--batch-size', type=int, default=8, help='lower this if you run out of memory')
args = parser.parse_args()

env = dict(os.environ)
if args.sample:
    env['GEOTHERMAL_DATA_PATH'] = SAMPLE
data = env.get('GEOTHERMAL_DATA_PATH') or os.path.join(ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')
if not os.path.exists(data):
    sys.exit(f'Prepared dataset not found at {data}\nSee data/README.md, or run with --sample.')
name = 'model_' + args.model.lower()
out_name = f"{name}_{'sample' if args.sample else 'test'}_results.json"
sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, 'src', 'model_a', 'evaluate_test_set.py'),
                          '--ckpt-path', os.path.join(ROOT, 'checkpoints', name, 'best.pth'),
                          '--out-name', out_name, '--batch-size', str(args.batch_size)], env=env))
