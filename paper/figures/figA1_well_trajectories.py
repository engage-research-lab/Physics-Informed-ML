"""Journal version of Fig. A.1: mean simulated and predicted well-output trajectories over the 232 test
scenarios, drawn at its printed width with no code-style titles. The averages come from
compute_avg_trajectories in src/model_a/plot_wells_avg_trajectory.py, used unchanged."""
import os, sys
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, MODEL); os.chdir(MODEL)
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_wells_avg_trajectory as pw

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'font.family': 'DejaVu Sans'})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = np.load(pw.PREPARED_DATA_PATH, allow_pickle=True)['well_time_years']
pred, true = pw.compute_avg_trajectories(pw.BEST_CKPT_PATH, 8, device)
labels = ['Production temperature (K)', 'Injection-well pressure (MPa)', 'Production-well pressure (MPa)']
fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.6), sharex=True, layout='constrained')
for c, (ax, lab) in enumerate(zip(axes, labels)):
    ax.plot(t, true[:, c], color='#4C72B0', lw=1.4, label='Simulated (mean)')
    ax.plot(t, pred[:, c], color='#C44E52', lw=1.1, ls='--', label='Model A (mean)')
    ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(loc='best')
    ax.annotate(f'({"abc"[c]})', (-0.13, 1.0), xycoords='axes fraction', fontweight='bold', fontsize=9, va='top')
    d = pred[:, c] - true[:, c]
    print(lab, 'mean bias %.3f, sd %.3f, max |dev| %.3f' % (d.mean(), d.std(), np.abs(d).max()))
axes[-1].set_xlabel('Year')
fig.savefig(os.path.join(OUT, 'figA1.png'), dpi=400); plt.close(fig)
print('saved figA1')
