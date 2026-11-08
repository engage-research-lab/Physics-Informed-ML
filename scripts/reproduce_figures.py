"""Regenerates the paper's figures into results/figures/.

    python scripts/reproduce_figures.py                     all figures
    python scripts/reproduce_figures.py --figures 3 4 10    only these (each script makes its whole group)
    python scripts/reproduce_figures.py --sample            the figures that the 4 sample scenarios in data/ cover
    python scripts/reproduce_figures.py --list              show which figure needs what

Figs. 3, 4 and 10 run from the saved metrics. With --sample, Figs. 2, 5a, 6, 7, 8, 9, A2 and A3 also run, from the
sample scenarios, and come out identical to the published ones. Everything else needs the full prepared dataset.
Figs. 11-13 first need the ensemble run, which takes about 10 minutes on a 12-thread CPU and is skipped when its
output already exists. Fig. 1 is a schematic with no script. The graphical abstract is built separately (see
paper/graphical_abstract/).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
SAMPLE = os.path.join(ROOT, 'data', 'sample_scenarios.npz')
ENSEMBLE = os.path.join(ROOT, 'results', 'metrics', 'ensemble_figure_data.pkl')
CAPTURE = 'fig11_13_capture_ensemble.py'
# script -> (figures it makes, needs a dataset, needs the ensemble run, runs on the sample)
SCRIPTS = {
    'fig02_07_training_and_depth_change.py': (['2', '7'], True, False, True),
    'fig03_04_10_error_distributions.py': (['3', '4', '10'], False, False, True),
    'fig05a_real_01051.py': (['5a'], True, False, True),
    'fig05b_error_vs_initial_pressure.py': (['5b'], True, False, False),
    'fig06_3d_fields.py': (['6'], True, False, True),
    'fig08_09_A2_A3_depth_galleries.py': (['8', '9', 'A2', 'A3'], True, False, True),
    'fig11_12_ensemble.py': (['11', '12'], True, True, False),
    'fig13_controls.py': (['13'], True, True, False),
    'figA1_well_trajectories.py': (['A1'], True, False, False),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--figures', nargs='+', help='figure numbers, e.g. 3 5a 11 A1 (default: all)')
parser.add_argument('--sample', action='store_true', help='use the 4 sample scenarios shipped in data/')
parser.add_argument('--list', action='store_true', help='show the figure-to-script table and exit')
args = parser.parse_args()

if args.list:
    for script, (figs, data, ens, smp) in SCRIPTS.items():
        if ens:
            need = 'full dataset + ensemble run'
        elif data:
            need = 'sample or full dataset' if smp else 'full dataset'
        else:
            need = 'nothing extra'
        print(f"{', '.join(figs):<14}{script:<44}{need}")
    sys.exit()

if args.sample:
    os.environ['GEOTHERMAL_DATA_PATH'] = SAMPLE
DATA = os.environ.get('GEOTHERMAL_DATA_PATH') or os.path.join(ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')
pool = {s: v for s, v in SCRIPTS.items() if v[3] or not args.sample}
known = {f for v in pool.values() for f in v[0]}
wanted = set(args.figures) if args.figures else known
if wanted - known:
    where = ' from the sample' if args.sample else ''
    sys.exit(f"Figure(s) {sorted(wanted - known)} not available{where}. Choose from: {', '.join(sorted(known))}")
chosen = [s for s, v in pool.items() if wanted & set(v[0])]

if any(SCRIPTS[s][1] for s in chosen) and not os.path.exists(DATA):
    sys.exit(f'Prepared dataset not found at {DATA}\nSee data/README.md, or use --sample. '
             'Figs. 3, 4 and 10 run without any data.')
if any(SCRIPTS[s][2] for s in chosen) and not os.path.exists(ENSEMBLE):
    chosen.insert(0, CAPTURE)
for script in chosen:
    print(f'--- {script}', flush=True)
    code = subprocess.call([sys.executable, os.path.join(FIG_DIR, script)])
    if code:
        sys.exit(f'{script} failed with exit code {code}')
print('Done. Figures are in results/figures/.')
