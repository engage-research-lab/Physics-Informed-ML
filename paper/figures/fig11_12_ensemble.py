"""Journal layouts of Figs. 11 and 12, drawn at the printed width (6.8 in) from the arrays that
uq_analysis_mc_v2.py computes (captured by fig11_13_capture_ensemble.py). Units: MWth, GJ, K, MPa."""
import os, pickle
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
PANELS = os.path.join(HERE, 'panels')
os.makedirs(PANELS, exist_ok=True)
S = pickle.load(open(os.path.join(HERE, '..', 'metrics', 'ensemble_figure_data.pkl'), 'rb'))
COL = ['#4169e1', '#e7298a', '#7b2cbf']
NAME = ['Cluster A', 'Cluster B', 'Cluster C']
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'font.family': 'DejaVu Sans', 'mathtext.default': 'regular'})
DPI = 400
from panel_font import panel_font
F = panel_font()


def stack(parts, out):
    ims = [Image.open(os.path.join(PANELS, p)).convert('RGB') for p in parts]
    W = max(i.width for i in ims); pad = 70
    c = Image.new('RGB', (W, sum(i.height + pad for i in ims)), 'white'); d = ImageDraw.Draw(c); y = 0
    for n, im in enumerate(ims):
        d.text((0, y + 5), f'({"ab"[n]})', font=F, fill='black'); c.paste(im, (0, y + pad)); y += im.height + pad
    c.save(os.path.join(HERE, out), dpi=(DPI, DPI)); print(out, c.size)


# ---------------------------------------------------------------- Fig. 11
kpis, cl = S['plot_kpi_distributions_boxplots'][0][0], S['plot_kpi_distributions_boxplots'][0][1]
mw, en, dt = kpis['MW_th'], kpis['Cum_Energy'], kpis['Delta_T_pct']
print('median capacity %.2f MW, max decline %.2f%%' % (np.median(mw), dt.max()))
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), layout='constrained')
for ax, v, lab, scale in zip(axes, [mw, en, dt],
                             ['Thermal power capacity (MW$_{th}$)', 'Cumulative thermal energy (10$^{7}$ GJ)', '40-year temperature decline (%)'],
                             [1, 1e-7, 1]):
    data = [v[cl == c] * scale for c in range(3)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3.5),
                    medianprops=dict(color='white', lw=1.4), flierprops=dict(marker='o', markersize=2, alpha=0.5))
    for patch, c in zip(bp['boxes'], COL):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks([1, 2, 3], [f'{NAME[c][-1]}\n(n = {(cl == c).sum():,})' for c in range(3)])
    ax.set_xlabel('Scenario cluster'); ax.set_ylabel(lab)
    ax.grid(True, axis='y', alpha=0.25, ls=':')
fig.savefig(os.path.join(PANELS, 'fig11a.png'), dpi=DPI); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 2.6), layout='constrained')
xg = np.linspace(mw.min() - 2, mw.max() + 2, 500)
for lab, m, c in [('Decline below 2%', dt < 2, '#4361ee'), ('Decline 2% to 4%', (dt >= 2) & (dt < 4), '#f77f00'),
                  ('Decline 4% to 6%', (dt >= 4) & (dt < 6), '#2d6a4f')]:
    y = gaussian_kde(mw[m], bw_method='scott')(xg)
    ax.plot(xg, y, color=c, lw=1.5, label=f'{lab} (n = {m.sum():,})'); ax.fill_between(xg, y, color=c, alpha=0.15)
ax.axvline(np.median(mw), color='#1d3557', ls='--', lw=1, label=f'Ensemble median ({np.median(mw):.1f} MW$_{{th}}$)')
ax.set_xlabel('Thermal power capacity at Year 40 (MW$_{th}$)'); ax.set_ylabel('Probability density')
ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(bottom=0); ax.legend(loc='upper right'); ax.grid(True, alpha=0.25, ls=':')
fig.savefig(os.path.join(PANELS, 'fig11b.png'), dpi=DPI); plt.close(fig)
stack(['fig11a.png', 'fig11b.png'], 'fig11.png')

# ---------------------------------------------------------------- Fig. 12
t, pooled, n_total = S['plot_mc_percentile_fan'][0][:3]
t = np.array(t)
_, per_cluster, cl2, _ = S['plot_mc_percentile_fan_by_reservoir'][0][:4]
CH = [('T_prod_K', 'Production temperature (K)', '#d62828'), ('P_inj_MPa', 'Injection-well pressure (MPa)', '#1d6fa5'),
      ('dP_MPa', 'Pressure difference (MPa)', '#f77f00'), ('P_prod_MPa', 'Production-well pressure (MPa)', '#2a9d8f')]


def band(ax, bd, color):
    g = lambda k: np.array(bd[k])
    ax.fill_between(t, g('p10'), g('p90'), color=color, alpha=0.15, lw=0)
    ax.fill_between(t, g('p25'), g('p75'), color=color, alpha=0.4, lw=0)
    ax.plot(t, g('p50'), color=color, lw=1.3)
    ax.set_xlim(0, 40); ax.grid(True, alpha=0.25, ls=':')


fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9), layout='constrained')
for ax, (k, lab, c) in zip(axes, CH):
    band(ax, pooled[k], c); ax.set_title(lab.replace(' (', chr(10) + '('), fontsize=7.5); ax.set_xlabel('Year')
fig.savefig(os.path.join(PANELS, 'fig12a.png'), dpi=DPI); plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(6.8, 4.6), layout='constrained', sharex=True)
for col in range(3):
    for row, (k, lab, _) in enumerate(CH[:3]):
        ax = axes[row, col]
        band(ax, per_cluster[col][k], COL[col])
        if row == 0:
            ax.set_title(f'{NAME[col]} (n = {(cl2 == col).sum():,})', color=COL[col], fontweight='bold')
        if col == 0:
            ax.set_ylabel(lab.replace(' (', '\n('))
        if row == 2:
            ax.set_xlabel('Year')
fig.savefig(os.path.join(PANELS, 'fig12b.png'), dpi=DPI); plt.close(fig)
stack(['fig12a.png', 'fig12b.png'], 'fig12.png')
for c in range(3):
    b = per_cluster[c]['T_prod_K']['p50']; print(NAME[c], 'median T %.1f -> %.1f K' % (b[0], b[-1]))
