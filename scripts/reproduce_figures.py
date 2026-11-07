"""Regenerates the paper's figures into results/figures/.

    python scripts/reproduce_figures.py                     all figures
    python scripts/reproduce_figures.py --figures 3 4 10    only these (each script makes its whole group)
    python scripts/reproduce_figures.py --list              show which figure needs what

Figs. 3, 4 and 10 run from the saved metrics. Everything else needs the prepared dataset. Figs. 11-13 first need
the ensemble run, which takes about 10 minutes on a 12-thread CPU and is skipped when its output already exists.
Fig. 1 is a schematic with no script. The graphical abstract is built separately (see paper/graphical_abstract/).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
DATA = os.environ.get('GEOTHERMAL_DATA_PATH') or os.path.join(ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')
ENSEMBLE = os.path.join(ROOT, 'results', 'metrics', 'ensemble_figure_data.pkl')
CAPTURE = 'fig11_13_capture_ensemble.py'
# script -> (figures it makes, needs the dataset, needs the ensemble run)
SCRIPTS = {
    'fig02_07_training_and_depth_change.py': (['2', '7'], True, False),
    'fig03_04_10_error_distributions.py': (['3', '4', '10'], False, False),
    'fig05a_real_01051.py': (['5a'], True, False),
    'fig05b_error_vs_initial_pressure.py': (['5b'], True, False),
    'fig06_3d_fields.py': (['6'], True, False),
    'fig08_09_A2_A3_depth_galleries.py': (['8', '9', 'A2', 'A3'], True, False),
    'fig11_12_ensemble.py': (['11', '12'], True, True),
    'fig13_controls.py': (['13'], True, True),
    'figA1_well_trajectories.py': (['A1'], True, False),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--figures', nargs='+', help='figure numbers, e.g. 3 5a 11 A1 (default: all)')
parser.add_argument('--list', action='store_true', help='show the figure-to-script table and exit')
args = parser.parse_args()

if args.list:
    for script, (figs, data, ens) in SCRIPTS.items():
        need = 'dataset + ensemble run' if ens else 'dataset' if data else 'nothing extra'
        print(f"{', '.join(figs):<14}{script:<44}{need}")
    sys.exit()

known = {f for figs, _, _ in SCRIPTS.values() for f in figs}
wanted = set(args.figures) if args.figures else known
if wanted - known:
    sys.exit(f"Unknown figure(s) {sorted(wanted - known)}. Choose from: {', '.join(sorted(known))}")
chosen = [s for s, (figs, _, _) in SCRIPTS.items() if wanted & set(figs)]

if any(SCRIPTS[s][1] for s in chosen) and not os.path.exists(DATA):
    sys.exit(f'Prepared dataset not found at {DATA}\nSee data/README.md. Figs. 3, 4 and 10 run without it.')
if any(SCRIPTS[s][2] for s in chosen) and not os.path.exists(ENSEMBLE):
    chosen.insert(0, CAPTURE)
for script in chosen:
    print(f'--- {script}', flush=True)
    code = subprocess.call([sys.executable, os.path.join(FIG_DIR, script)])
    if code:
        sys.exit(f'{script} failed with exit code {code}')
print('Done. Figures are in results/figures/.')
