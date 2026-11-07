"""Image panels for the four-panel graphical abstract (build_graphical_abstract.js):

  geology.png        Panel 1: schematic geothermal doublet (surface plant, overburden,
                     injector and producer) around the simulated year-40 temperature on the
                     vertical plane through both wells, taken from COMSOL for the
                     highest-injection-rate scenario of scenario cluster A (real_01463).
  decline_scatter.png  Panel 4: surrogate vs COMSOL 40-year production-temperature decline
                     for all 1,535 scenarios.

Needs results/metrics/uq_results_v2.json (cluster assignments) and the Model A well
predictions cached by reviewer_metrics.py.
Usage: python prepare_panels.py   ->  results/figures/graphical_abstract/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402

OUT = C.OUT_DIR / 'graphical_abstract'
INJ, PROD, NAVY = '#2E86AB', '#C0392B', '#1F3B57'
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 17})


def cluster_a_high_flow(raw):
    """Global index and sample id of the cluster-A scenario with the highest injection rate."""
    labels = np.array(json.loads((C.METRICS_DIR / 'uq_results_v2.json').read_text())['cluster_assignments'])
    q = raw['x_params'][:, list(raw['parameter_names']).index('inj_rate')]
    members = np.where(labels == 0)[0]
    g = int(members[np.argmax(q[members])])
    return g, str(raw['sample_id'][g])


def section(raw, g):
    """Year-40 COMSOL temperature (deg C) on the x-z plane through the wells, with the
    rows (z indices) of the pressure (reservoir) domain."""
    wm = raw['x_static_fields'][g, 5]
    inj, prod = np.argwhere(wm > 0.5)[0], np.argwhere(wm < -0.5)[0]
    j = inj[1]
    T = raw['T_K'][g][:, j, :, -1] - 273.15
    mT, mP = raw['mask_T'][g][:, j, :, -1], raw['mask_P'][g][:, j, :, -1]
    zT, zP = np.where(mT.any(0))[0], np.where(mP.any(0))[0]
    return np.where(mT, T, np.nan)[:, zT], zT, zP, inj, prod


def wavy(x, y0, amp, ph):
    return y0 + amp * np.sin(x * 2.2 + ph) + 0.4 * amp * np.sin(x * 5.1 + 2 * ph)


def geology(raw, g):
    T, zT, zP, inj, prod = section(raw, g)
    nx, nz = T.shape
    fig = plt.figure(figsize=(3.6, 6.0))
    ax = fig.add_axes([0, 0.135, 1, 0.865])
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 15.6)
    ax.axis('off')
    x = np.linspace(0, 10, 300)

    # sky, plant and ground surface
    ax.add_patch(Rectangle((0, 13.2), 10, 2.4, color='#EAF3FA', zorder=0))
    ax.add_patch(Rectangle((3.2, 13.2), 3.6, 0.9, color='#8B97A3', zorder=2))
    ax.add_patch(Polygon([[3.2, 14.1], [5.0, 14.75], [6.8, 14.1]], color='#6B7785', zorder=2))
    for cx in (3.9, 5.0, 6.1):
        ax.add_patch(Rectangle((cx - 0.22, 13.45), 0.44, 0.4, color='#DDE6EE', zorder=3))
    ax.text(5.0, 15.05, 'Heat plant', ha='center', fontsize=17.5, color=NAVY, weight='bold')

    # overburden strata (schematic), top to bottom
    tops = [13.2, 12.2, 11.0, 9.9, 8.9]
    cols = ['#C9B48F', '#B39B74', '#D2C3A3', '#A88F6C', '#BFAE8E']
    bounds = [wavy(x, t, 0.08 if k else 0.0, k) for k, t in enumerate(tops)] + [np.full_like(x, 8.0)]
    for k in range(len(tops)):
        ax.fill_between(x, bounds[k + 1], bounds[k], facecolor=cols[k], edgecolor='#00000030',
                        hatch=('..', '--', '..', '--', '..')[k], zorder=1, lw=0)
    ax.text(9.8, 8.45, 'Overburden', fontsize=16, color='#3A2E1C', style='italic', ha='right',
            bbox=dict(fc='#EFE6D2', ec='none', alpha=0.85, pad=1.5), zorder=8)
    ax.plot(x, np.full_like(x, 8.0), color='#5A4A33', lw=1.2, zorder=4)

    # simulated section: cap rock + reservoir, year-40 temperature from COMSOL, with lithology textures
    y0, y1 = 1.05, 8.0
    ext = [0, 10, y0, y1]
    im = ax.imshow(T.T, origin='lower', extent=ext, cmap='RdYlBu_r', aspect='auto',
                   interpolation='bilinear', zorder=2, vmin=np.nanmin(T), vmax=np.nanmax(T))
    dz = (y1 - y0) / nz
    r0 = y0 + (zP[0] - zT[0]) * dz
    r1 = y0 + (zP[-1] - zT[0] + 1) * dz
    for lo, hi, hatch in ((y0, r0, '--'), (r0, r1, '..'), (r1, y1, '--')):
        ax.add_patch(Rectangle((0, lo), 10, hi - lo, fill=False, hatch=hatch, edgecolor='#00000038', lw=0, zorder=3))
    for yy in (r0, r1):
        ax.plot([0, 10], [yy, yy], color='#2B2B2B', lw=1.3, zorder=4)
    lab = dict(fontsize=15.5, color='white', weight='bold', zorder=5, va='center',
               bbox=dict(fc='#00000055', ec='none', pad=1.5))
    ax.text(0.2, (r1 + y1) / 2, 'Cap rock', **lab)
    ax.text(0.2, (y0 + r0) / 2, 'Cap rock', **lab)
    ax.text(9.8, r1 - 0.45, 'Reservoir', ha='right', **lab)

    # wells from the plant down to the screened interval
    xw = {'i': (inj[0] + 0.5) / nx * 10, 'p': (prod[0] + 0.5) / nx * 10}
    zw = y0 + (inj[2] - zT[0] + 0.5) * dz
    for key, col, name, flow, sgn in (('i', INJ, 'Injector', 'cold water', -1), ('p', PROD, 'Producer', 'hot water', 1)):
        xs = xw[key]
        ax.plot([xs, xs], [13.25, zw], color='#2B2B2B', lw=5, zorder=6)
        ax.plot([xs, xs], [13.25, zw], color=col, lw=3, zorder=7)
        xt = xs + sgn * 0.3
        ha = 'left' if sgn > 0 else 'right'
        ax.text(xt, 12.55, name, ha=ha, va='center', fontsize=16.5, color=col, weight='bold', zorder=8,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
        xa = xs + sgn * 0.55
        ya = (11.85, 10.65) if sgn < 0 else (10.65, 11.85)
        ax.add_patch(FancyArrowPatch((xa, ya[0]), (xa, ya[1]), arrowstyle='-|>', mutation_scale=20, color=col, lw=3, zorder=8))
        ax.text(xs + sgn * 0.95, 11.25, flow.replace(' ', '\n'), ha=ha, va='center', fontsize=15, color=col, weight='bold',
                zorder=8, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
    ax.add_patch(FancyArrowPatch((xw['i'] + 0.3, zw), (xw['p'] - 0.3, zw), arrowstyle='-|>',
                                 mutation_scale=18, color='white', lw=2.4, zorder=8))
    ax.text(5.0, 0.6, 'COMSOL temperature, year 40 (not to scale)', ha='center', fontsize=13.5,
            color='#3A3A3A', style='italic')

    # colour bar
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=14, length=2)
    cb.set_label('Temperature (°C)', fontsize=15.5, labelpad=1)
    fig.savefig(OUT / 'geology.png', dpi=300)
    plt.close(fig)


def decline_scatter(raw):
    true = raw['y_wells'][:, :, 0]
    pred = C.cached_wells_model_a()[:, :, 0]
    dt, dp = true[:, 0] - true[:, -1], pred[:, 0] - pred[:, -1]
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    ax.scatter(dt, dp, s=7, color='#D9822B', alpha=0.55, lw=0)
    lim = [min(dt.min(), dp.min()) - 0.5, max(dt.max(), dp.max()) + 0.5]
    ax.plot(lim, lim, color=NAVY, lw=1.3, ls='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('COMSOL decline (K)', fontsize=16)
    ax.set_ylabel('Surrogate decline (K)', fontsize=16)
    ax.tick_params(labelsize=14)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    r = np.corrcoef(dt, dp)[0, 1]
    ax.text(0.04, 0.95, f'1,535 scenarios\nr = {r:.2f}', transform=ax.transAxes, va='top', fontsize=15, color=NAVY)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / 'decline_scatter.png', dpi=300)
    plt.close(fig)
    print(f'decline: median COMSOL {np.median(dt):.2f} K, surrogate {np.median(dp):.2f} K, r = {r:.3f}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = C.load_prepared()
    g, sid = cluster_a_high_flow(raw)
    geology(raw, g)
    print('section from', sid)
    decline_scatter(raw)
    print('saved', OUT)


if __name__ == '__main__':
    main()
