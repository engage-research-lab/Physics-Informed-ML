"""Fig. 5a: simulated and predicted (Models A and B) production-well pressure for real_01051.

Usage: python fig5a_real_01051.py   ->  results/figures/fig5a.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common as C

STYLE = {'A': dict(color='#4C72B0', ls='--'), 'B': dict(color='#DD8452', ls=':')}


def main():
    device = C.get_device()
    raw = C.load_prepared()
    ds = C.ComsolFnoDataset(C.PREPARED_DATA_PATH, split='test')
    item = ds[list(raw['sample_id'][raw['test_idx']]).index('real_01051')]
    mean, std = ds.stats['well__P_prod_MPa__mean'], ds.stats['well__P_prod_MPa__std']
    t = raw['well_time_years']
    true_curve = C.denormalize(torch.from_numpy(item['wells'][:, 2]), mean, std).numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(t, true_curve, color='black', lw=2, label='Simulated (COMSOL)')
    for which in ('A', 'B'):
        model = C.load_model(which, device)
        with torch.no_grad():
            out = model(torch.from_numpy(item['x_static'])[None].to(device),
                        torch.from_numpy(item['params'])[None].to(device),
                        torch.from_numpy(item['well_map'])[None].to(device))
        pred = C.denormalize(out['wells'][0, :, 2], mean, std).cpu().numpy()
        print(f'Model {which}: year 0 {pred[0]:.2f} MPa, year 40 {pred[-1]:.2f} MPa')
        ax.plot(t, pred, lw=2, label=f'Model {which} prediction', **STYLE[which])
    print(f'COMSOL:  year 0 {true_curve[0]:.3f} MPa, year 40 {true_curve[-1]:.3f} MPa')
    ax.set_xlabel('Year')
    ax.set_ylabel('Production-well pressure (MPa, gauge)')
    ax.set_xlim(0, 40)
    ax.set_ylim(bottom=0)
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.98, '(a)', fontsize=14, fontweight='bold', va='top')
    path = C.out_path('fig5a.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('saved', path)


if __name__ == '__main__':
    main()
