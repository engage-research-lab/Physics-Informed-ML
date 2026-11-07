"""Journal versions of Figs. 3, 4 and 10, drawn at the printed width (6.8 in) with one common
font size, from the per-scenario test-set errors in the repo's results/metrics JSON files
(the same inputs as results_figures.py / results_figures_v2.py)."""
import json, os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image, ImageDraw, ImageFont

EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'metrics')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
PANELS = os.path.join(OUT, 'panels')
os.makedirs(PANELS, exist_ok=True)
A = json.load(open(os.path.join(EV, 'model_a_test_results.json')))['per_scenario']
B = json.load(open(os.path.join(EV, 'model_b_test_results.json')))['per_scenario']
CA, CB, TGT = '#4C72B0', '#DD8452', '#C44E52'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
                     'font.family': 'DejaVu Sans', 'mathtext.default': 'regular'})
DPI = 400
from panel_font import panel_font
F = panel_font()
OUTPUTS = [('Temperature', 're_t', False), ('Pressure', 're_p', False), ('Well outputs', 're_wells', True)]
WELLS = [('Production temperature', 're_T_prod_K', False), ('Injection pressure', 're_P_inj_MPa', False),
         ('Production pressure', 're_P_prod_MPa', True)]


def kde(v, log_x):
    v = np.asarray(v, float)
    if log_x:
        x = np.log10(v[v > 0]); xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
        return 10 ** xs, gaussian_kde(x)(xs)
    xs = np.linspace(0, v.max() * 1.15, 400)   # relative errors are non-negative
    return xs, gaussian_kde(v)(xs)


def finish(ax, log_x, ylabel):
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel('Relative error (%)' + (', log scale' if log_x else ''))
    ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
    ax.legend(loc='upper right', framealpha=0.9)


def stack(parts, out):
    ims = [Image.open(os.path.join(PANELS, p)).convert('RGB') for p in parts]
    W = max(i.width for i in ims); pad = 70
    c = Image.new('RGB', (W, sum(i.height + pad for i in ims)), 'white'); d = ImageDraw.Draw(c); y = 0
    for n, im in enumerate(ims):
        d.text((0, y + 5), f'({"ab"[n]})', font=F, fill='black'); c.paste(im, (0, y + pad)); y += im.height + pad
    c.save(os.path.join(OUT, out), dpi=(DPI, DPI)); print(out, c.size)


# ---------------------------------------------------------------- Fig. 3
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), layout='constrained')
for ax, (lab, key, log_x) in zip(axes, OUTPUTS):
    v = np.array(A[key]); m = v.mean()
    edges = np.logspace(np.log10(v[v > 0].min()), np.log10(v.max()), 40) if log_x else 40
    ax.hist(v, bins=edges, color=CA, alpha=0.75, edgecolor='white', lw=0.6)
    ax.axvline(m, color='black', ls='--', lw=1, label=f'Mean = {m:.2f}%')
    if not log_x:
        ax.set_xlim(0, v.max() * 1.15)
        ax.text(0.97, 0.62, f'2% target is\n{2.0 / m:.1f}× the mean\n(off-axis)', transform=ax.transAxes,
                ha='right', va='top', color=TGT, fontsize=7)
    ax.set_title(lab); finish(ax, log_x, 'Number of test scenarios')
fig.savefig(os.path.join(PANELS, 'fig3a.png'), dpi=DPI); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), layout='constrained')
for ax, (lab, key, log_x) in zip(axes, WELLS):
    v = np.array(A[key]); xs, ys = kde(v, log_x)
    ax.plot(xs, ys, color=CA, lw=1.3); ax.fill_between(xs, ys, color=CA, alpha=0.35)
    ax.axvline(v.mean(), color='black', ls='--', lw=1, label=f'Mean = {v.mean():.2f}%')
    ax.axvline(np.median(v), color='black', ls=':', lw=1, label=f'Median = {np.median(v):.2f}%')
    ax.set_title(lab); finish(ax, log_x, 'Density')
fig.savefig(os.path.join(PANELS, 'fig3b.png'), dpi=DPI); plt.close(fig)
stack(['fig3a.png', 'fig3b.png'], 'fig3.png')

# ---------------------------------------------------------------- Fig. 4
cats = ['none', 'anomaly_targeted', 'lhs_explore']
foc = np.array(A['design_focus'])
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), layout='constrained')
for ax, (lab, key, log_x) in zip(axes, OUTPUTS):
    v = np.array(A[key])
    ax.boxplot([v[foc == c] for c in cats], widths=0.55, patch_artist=True,
               boxprops=dict(facecolor='#c6d4ea', edgecolor='black', lw=0.8),
               medianprops=dict(color='#ff7f0e', lw=1.4), flierprops=dict(marker='o', ms=2.5, alpha=0.6))
    ax.set_xticks([1, 2, 3], [c.replace('_', '_\n') + f'\n(n = {(foc == c).sum()})' for c in cats], fontsize=6.8)
    ax.set_title(lab); ax.set_ylabel('Relative error (%)' + (', log scale' if log_x else ''))
    if log_x:
        ax.set_yscale('log')
    ax.grid(alpha=0.3, axis='y')
fig.savefig(os.path.join(OUT, 'fig4.png'), dpi=DPI); plt.close(fig)
print('fig4 counts', {c: int((foc == c).sum()) for c in cats})

# ---------------------------------------------------------------- Fig. 10
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), layout='constrained')
for ax, (lab, key, log_x) in zip(axes, OUTPUTS):
    for ps, col, name in [(A, CA, 'Model A'), (B, CB, 'Model B')]:
        v = np.array(ps[key]); xs, ys = kde(v, log_x)
        ax.plot(xs, ys, color=col, lw=1.3, label=f'{name} (mean = {v.mean():.2f}%)')
        ax.fill_between(xs, ys, color=col, alpha=0.25); ax.axvline(v.mean(), color=col, ls='--', lw=1)
    ax.set_title(lab); finish(ax, log_x, 'Density')
fig.savefig(os.path.join(OUT, 'fig10.png'), dpi=DPI); plt.close(fig)
for lab, key, _ in OUTPUTS + WELLS:
    print(lab, 'A mean %.2f median %.2f | B mean %.2f median %.2f' % (np.mean(A[key]), np.median(A[key]), np.mean(B[key]), np.median(B[key])))
