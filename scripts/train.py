"""Trains a model from scratch. A GPU is strongly recommended.

    python scripts/train.py --model A --seed 1

Checkpoints go to checkpoints/model_a_seed1/ (seed 0 uses checkpoints/model_a/, which holds the
published best.pth and would be overwritten, so use a seed other than 0 to keep it).
Needs the prepared dataset.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('GEOTHERMAL_DATA_PATH') or os.path.join(ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')
SCRIPTS = {'A': os.path.join('model_a', 'comsol_fno_train.py'), 'B': os.path.join('model_b', 'comsol_fno_train_pinn.py')}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', choices=['A', 'B'], default='A')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

if not os.path.exists(DATA):
    sys.exit(f'Prepared dataset not found at {DATA}\nSee data/README.md.')
env = dict(os.environ, TRAIN_SEED=str(args.seed))
sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, 'src', SCRIPTS[args.model])], env=env))
