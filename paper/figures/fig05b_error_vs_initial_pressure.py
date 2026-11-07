"""Fig. 5b: Model A production-pressure relative error against the simulated initial
production pressure (232 test scenarios), above the training-set distribution of that pressure.

Usage: python fig5b_error_vs_initial_pressure.py   ->  results/figures/fig5b.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common as C


def main():
    raw = C.load_prepared()
    ids = list(raw['sample_id'])
    p0 = raw['y_wells'][:, 0, 2]
    ev = json.loads((C.METRICS_DIR / 'model_a_test_results.json').read_text())['per_scenario']
    sid, re = ev['sample_id'], np.array(ev['re_P_prod_MPa'])
    x = np.array([p0[ids.index(s)] for s in sid])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={'height_ratios': [1, 2.6]})
    ax0.hist(p0[raw['train_idx']], bins=np.arange(0, 26, 1.0), color='#8C8C8C', edgecolor='white')
    ax0.set_ylabel('Training\nscenarios')
    ax0.set_yscale('log')
    ax0.set_title('Initial production pressure, training set (n = 1,074)', loc='left', fontsize=11)
    ax1.scatter(x, re, s=22, color='#4C72B0', alpha=0.75, edgecolor='none', label='Test scenario')
    i = sid.index('real_01051')
    ax1.scatter([x[i]], [re[i]], s=60, color='#C44E52', zorder=3)
    ax1.annotate('real_01051', (x[i], re[i]), xytext=(x[i] + 0.6, re[i]), va='center', fontsize=10, color='#C44E52')
    ax1.axhline(np.median(re), color='black', ls=':', lw=1.2, label=f'Test median = {np.median(re):.1f}%')
    ax1.set_yscale('log')
    ax1.set_xlabel('Simulated production pressure at year 0 (MPa, gauge)')
    ax1.set_ylabel('Production-pressure RE (%)')
    ax1.set_title('Model A production-pressure error, 232 test scenarios', loc='left', fontsize=11)
    ax1.legend(loc='upper right')
    ax0.grid(alpha=0.3)
    ax1.grid(alpha=0.3)
    fig.text(0.01, 0.995, '(b)', fontsize=14, fontweight='bold', va='top')
    path = C.out_path('fig5b.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'corr(log RE, P0) = {np.corrcoef(np.log10(re), x)[0, 1]:.2f}; '
          f'training scenarios below 5 MPa: {int((p0[raw["train_idx"]] < 5).sum())}')
    print('saved', path)


if __name__ == '__main__':
    main()
