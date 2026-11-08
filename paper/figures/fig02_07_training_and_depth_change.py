"""Journal versions of Fig. 2 (training and validation loss) and Fig. 7 (mean absolute change from
Year 0 by depth and year for three test scenarios), drawn at their printed width with no code-style
titles. Data and computations are the same as the original training_loss_figure.py and
depth_time_heatmap_other_scenarios.py scripts."""
import json, os, sys
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, MODEL); os.chdir(MODEL)
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from comsol_fno_train import ComsolFnoDataset, denormalize, PREPARED_DATA_PATH

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'font.family': 'DejaVu Sans'})
DPI = 400

# ---------------------------------------------------------------- Fig. 2
rows = [json.loads(l) for l in open(os.path.join(OUT, '..', 'logs', 'fig2_representative_run.jsonl')) if l.strip()]
ep_rows = [r for r in rows if 'epoch' in r]
ep = [r['epoch'] for r in ep_rows]
best = max(r['epoch'] for r in ep_rows if r.get('is_best'))
fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.6), sharex=True, layout='constrained')
for ax, key, lab in [(axes[0], 't', 'Temperature loss'), (axes[1], 'p', 'Pressure loss')]:
    ax.plot(ep, [r[f'train_loss_{key}'] for r in ep_rows], color='#4C72B0', lw=1, label='Training')
    ax.plot(ep, [r[f'val_loss_{key}'] for r in ep_rows], color='#C44E52', lw=1, label='Validation')
    ax.axvline(best, color='green', ls='--', lw=1)
    ax.set_yscale('log'); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(loc='center right')
axes[0].text(best, axes[0].get_ylim()[1], f' best epoch {best}', color='green', va='top', fontsize=7.5)
axes[1].set_xlabel('Epoch')
fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=DPI); plt.close(fig)
print('fig2 best epoch', best)


# ---------------------------------------------------------------- Fig. 7
def valid_layers(mask, n_z):
    return np.array([mask[:, :, zi, 0].any() for zi in range(n_z)])


def change_heatmap(field, mask, valid, n_z):
    grid = np.full((n_z, field.shape[-1]), np.nan)
    for zi in range(n_z):
        if not valid[zi]:
            continue
        for yi in range(field.shape[-1]):
            m = mask[:, :, zi, yi] & mask[:, :, zi, 0]
            if m.any():
                grid[zi, yi] = np.mean(np.abs(field[:, :, zi, yi] - field[:, :, zi, 0])[m])
    return grid


raw = np.load(PREPARED_DATA_PATH, allow_pickle=True)
ids = [str(raw['sample_id'][i]) for i in raw['test_idx']]
ds = ComsolFnoDataset(PREPARED_DATA_PATH, split='test')
z_m = raw['z_m']
years = np.arange(0, 41, 5)
fig, axes = plt.subplots(3, 2, figsize=(6.8, 7.6), layout='constrained')
for r, sid in enumerate(['real_00008', 'real_01456', 'real_01528']):
    it = ds[ids.index(sid)]
    T = denormalize(torch.from_numpy(it['T_K']), ds.stats['field__T_K__mean'], ds.stats['field__T_K__std']).numpy()
    P = denormalize(torch.from_numpy(it['p_MPa']), ds.stats['field__p_MPa__mean'], ds.stats['field__p_MPa__std']).numpy()
    mT, mP = it['mask_T'] > 0.5, it['mask_P'] > 0.5
    well_z = z_m[np.argwhere(it['well_map'] > 0.5)[0][2]]
    n_z = len(z_m)
    for c, (F, M, name, cmap, unit) in enumerate([(T, mT, 'Temperature', 'inferno', '°C'), (P, mP, 'Pressure', 'magma', 'MPa')]):
        v = valid_layers(M, n_z); h = change_heatmap(F, M, v, n_z); rows_ = np.where(v)[0]
        ax = axes[r, c]
        im = ax.pcolormesh(years, z_m[rows_], h[rows_], shading='nearest', cmap=cmap)
        ax.axhline(well_z, color='cyan', ls='--', lw=1, label='Well depth')
        cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label(f'Mean |change from Year 0| ({unit})')
        ax.set_xlabel('Year')
        if c == 0:
            ax.set_ylabel(f'({"abc"[r]}) {sid}' + chr(10) + 'Depth, z (m)')
        if r == 0:
            ax.set_title(name, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        print(sid, name, 'valid layers', int(v.sum()))
fig.savefig(os.path.join(OUT, 'fig7.png'), dpi=DPI); plt.close(fig)
print('saved fig2, fig7')
