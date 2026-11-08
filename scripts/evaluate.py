"""Evaluates a final checkpoint on the 232-scenario test set, or on the 4 sample scenarios.

    python scripts/evaluate.py --model A             full test set (needs the full prepared dataset)
    python scripts/evaluate.py --model A --sample    the 4 sample scenarios in data/ (no download needed)

Writes results/metrics/model_a_test_results.json (or model_b_...). With --sample it writes
model_a_sample_results.json instead, so the published test-set results are not overwritten, and
prints each sample scenario's errors next to the values from the published test-set run.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, 'data', 'sample_scenarios.npz')
METRICS = os.path.join(ROOT, 'results', 'metrics')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', choices=['A', 'B'], default='A')
parser.add_argument('--sample', action='store_true', help='use the 4 sample scenarios shipped in data/')
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
code = subprocess.call([sys.executable, os.path.join(ROOT, 'src', 'model_a', 'evaluate_test_set.py'),
                        '--ckpt-path', os.path.join(ROOT, 'checkpoints', name, 'best.pth'),
                        '--out-name', out_name, '--batch-size', str(args.batch_size)], env=env)
if code or not args.sample:
    sys.exit(code)

# Side-by-side comparison with the published test-set run
new = json.load(open(os.path.join(METRICS, out_name)))['per_scenario']
ref = json.load(open(os.path.join(METRICS, f'{name}_test_results.json')))['per_scenario']
cols = [('re_t', 'Temperature'), ('re_p', 'Pressure'), ('re_wells', 'Wells')]
print(f'\n=== Model {args.model}: mean relative error (%), this run vs. published test-set run ===')
print(f"{'Scenario':<12}" + ''.join(f'{label:>24}' for _, label in cols))
for i, sid in enumerate(new['sample_id']):
    j = ref['sample_id'].index(sid)
    print(f'{sid:<12}' + ''.join(f'{new[k][i]:>11.3f} vs {ref[k][j]:>9.3f}' for k, _ in cols))
print('Small differences (from about the 4th significant digit) come from CPU vs. GPU arithmetic.')
