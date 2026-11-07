"""Journal layout of Fig. 13, drawn at the printed width (6.8 in) from the arrays that
uq_analysis_mc_v2.py computes (captured by fig11_13_capture_ensemble.py)."""
import os, sys, pickle
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
sys.path.insert(0, MODEL)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
PANELS = os.path.join(HERE, 'panels')
os.makedirs(PANELS, exist_ok=True)
S = pickle.load(open(os.path.join(HERE, '..', 'metrics', 'ensemble_figure_data.pkl'), 'rb'))
COL = {0: '#4169e1', 1: '#e7298a', 2: '#7b2cbf'}
MRK = {0: 'o', 1: 's', 2: '^'}
NAME = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'font.family': 'DejaVu Sans', 'mathtext.default': 'regular'})
DPI = 400

# ---------------------------------------------------------------- (a)
T, dT, cl = S['T_end'], S['dT_pct'], S['clusters']
order = np.argsort(dT); xs = np.arange(len(order))
fig, ax1 = plt.subplots(figsize=(6.8, 2.7), layout='constrained')
ax2 = ax1.twinx()
for c in range(3):
    m = cl[order] == c
    ax1.scatter(xs[m], T[order][m], s=3, alpha=0.55, c=COL[c], marker=MRK[c], lw=0)
    ax2.scatter(xs[m], dT[order][m], s=3, alpha=0.35, c=COL[c], marker=MRK[c], lw=0)
ax1.axhline(np.median(T), color='#1d3557', lw=1, ls='--')
ax2.axhline(np.median(dT), color='#e63946', lw=1, ls='-.')
ax1.set_xlabel('Scenarios sorted by 40-year temperature decline (lowest to highest)')
ax1.set_ylabel('Year-40 production temperature (K)', color='#1d3557')
ax2.set_ylabel('40-year temperature decline (%)', color='#e63946')
ax1.tick_params(axis='y', labelcolor='#1d3557'); ax2.tick_params(axis='y', labelcolor='#e63946')
ax1.grid(True, alpha=0.2, ls=':')
h = [Line2D([0], [0], marker=MRK[c], color=COL[c], ls='None', ms=4, label=NAME[c]) for c in range(3)]
h += [Line2D([0], [0], color='#1d3557', lw=1, ls='--', label=f'Median temperature ({np.median(T):.1f} K)'),
      Line2D([0], [0], color='#e63946', lw=1, ls='-.', label=f'Median decline ({np.median(dT):.1f}%)')]
ax1.legend(handles=h, loc='upper left', ncol=2, framealpha=0.9)
fig.savefig(os.path.join(PANELS, 'fig13a.png'), dpi=DPI); plt.close(fig)

# ---------------------------------------------------------------- (b)
pT, pP, mT, info = S['pred_T'], S['pred_P'], S['mask_T'], S['info']
z = pT.shape[3] // 2
Tsl = [np.where(mT[i, :, :, z], pT[i, :, :, z], np.nan) for i in range(len(info))]
Psl = [pP[i, :, :, z] for i in range(len(info))]
tmin, tmax = np.nanmin(Tsl), np.nanmax(Tsl)
pmin, pmax = np.nanmin(Psl), np.nanmax(Psl)
ext = [-1.0, 1.0, -0.6, 0.6]  # km, domain 2,000 m x 1,200 m centred on the origin
fig = plt.figure(figsize=(6.8, 3.75), layout='constrained')
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.09])
axes = np.array([[fig.add_subplot(gs[r, k]) for k in range(4)] for r in range(3)])
for i, inf in enumerate(info):
    r, hi = inf['cluster'], inf['suffix'] != 'Low Flow'
    aT, aP = axes[r, int(hi)], axes[r, 2 + int(hi)]
    imT = aT.imshow(Tsl[i].T, origin='lower', extent=ext, cmap='inferno', vmin=tmin, vmax=tmax, interpolation='nearest')
    imP = aP.imshow(Psl[i].T, origin='lower', extent=ext, cmap='viridis', vmin=pmin, vmax=pmax, interpolation='nearest')
    lab = f"ṁ = {inf['inj_rate_kgs']:.1f} kg/s, D = {inf['well_spacing_m']:.0f} m"
    for a in (aT, aP):
        a.set_title(lab, fontsize=6.8, pad=2)
        a.set_xticks([]); a.set_yticks([])
for r in range(3):
    axes[r, 0].set_ylabel(NAME[r], fontweight='bold')
for k, t in enumerate(['Temperature, low flow', 'Temperature, high flow', 'Pressure, low flow', 'Pressure, high flow']):
    axes[0, k].annotate(t, (0.5, 1.32), xycoords='axes fraction', ha='center', fontweight='bold')
fig.colorbar(imT, cax=fig.add_subplot(gs[3, 0:2]), orientation='horizontal', label='Temperature (K)')
fig.colorbar(imP, cax=fig.add_subplot(gs[3, 2:4]), orientation='horizontal', label='Pressure (MPa)')
fig.savefig(os.path.join(PANELS, 'fig13b.png'), dpi=DPI); plt.close(fig)

# ---------------------------------------------------------------- compose
from panel_font import panel_font
F = panel_font()
ims = [Image.open(os.path.join(PANELS, p)).convert('RGB') for p in ('fig13a.png', 'fig13b.png')]
W = max(i.width for i in ims); pad = 70
c = Image.new('RGB', (W, sum(i.height + pad for i in ims)), 'white'); d = ImageDraw.Draw(c); y = 0
for n, im in enumerate(ims):
    d.text((0, y + 5), f'({"ab"[n]})', font=F, fill='black'); c.paste(im, (0, y + pad)); y += im.height + pad
c.save(os.path.join(HERE, 'fig13.png'), dpi=(DPI, DPI)); print('fig13', c.size)
for i in info:
    print(i)
