"""Builds the prepared dataset from the raw COMSOL export.

    python scripts/prepare_data.py --raw-npz <raw .npz> --summary-csv <summary .csv>

Writes data/fno_lhs_v1_prepared_phase1.npz unless --out is given.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
if '--out' not in args and '-h' not in args and '--help' not in args:
    args += ['--out', os.path.join(ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')]
sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, 'src', 'data_preparation', 'prepare_dataset.py')] + args))
