"""Figs. 8, 9, A.2 and A.3 (journal versions) of the per-depth galleries (Figs. 8, 9, A.2, A.3), drawn at the printed
size (6.8 in wide) so fonts are 8-9 pt in print, with no code-style titles. Data selection
(layers, colour scales, masks) reuses the functions in the repo's plot_field_maps.py unchanged."""
import os, sys
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'model_a')
sys.path.insert(0, MODEL); os.chdir(MODEL)
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import plot_field_maps as pfm
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
PANELS = os.path.join(OUT, 'panels')
os.makedirs(PANELS, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'font.family': 'DejaVu Sans'})
LAB = {'T_K': ('Temperature (°C)', 'Predicted − true (°C)'), 'p_MPa': ('Pressure (MPa)', 'Predicted − true (MPa)')}


def layers(field0, inj, z_m):
    valid_z = np.where(~np.isnan(field0[inj[0], inj[1], :]))[0]
    sh, dp = valid_z[np.argmax(z_m[valid_z])], valid_z[np.argmin(z_m[valid_z])]
    return np.unique(np.linspace(dp, sh, pfm.N_DEPTH_LAYERS).round().astype(int))


def gallery(fields, levels, cmap, extend, cbar_label, inj, prod, grid, out, inj_color, edge, contour=False):
    x_m, y_m, z_m = grid['x_m'], grid['y_m'], grid['z_m']
    li = layers(fields[0], inj, z_m)
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    fig, axes = plt.subplots(len(fields), len(li), figsize=(6.8, 0.45 + 0.6 * len(fields)),
                             squeeze=False, layout='constrained')
    for row, (year, f) in enumerate(zip(pfm.YEARS_TO_PLOT, fields)):
        for col, zi in enumerate(li):
            ax = axes[row][col]
            if cmap is pfm.XRAY_GLOW_CMAP:
                ax.set_facecolor('black')
            cs = ax.contourf(X, Y, f[:, :, zi], levels=levels, cmap=cmap, extend=extend)
            if contour:
                ax.contour(X, Y, f[:, :, zi], levels=levels, colors='#39ff14', linewidths=0.5)
            ax.plot(x_m[inj[0]], y_m[inj[1]], 'o', color=inj_color, ms=3.5, mec=edge, mew=0.7, zorder=10)
            ax.plot(x_m[prod[0]], y_m[prod[1]], 'o', color='red', ms=3.5, mec=edge, mew=0.7, zorder=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f'{z_m[zi]:.0f} m'.replace('-', '−'))
            if col == 0:
                ax.set_ylabel(f'Year {year}')
    cb = fig.colorbar(cs, ax=axes, shrink=0.85, pad=0.02, aspect=25)
    cb.set_label(cbar_label)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('-', '−')))
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print('saved', out)


raw = np.load(pfm.PREPARED_DATA_PATH, allow_pickle=True)
ids = [str(raw['sample_id'][i]) for i in raw['test_idx']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for sid in (sys.argv[1:] or ['real_00008', 'real_01456', 'real_01528']):
    ds, item, meta, grid = pfm.load_scenario(ids.index(sid))
    assert meta['sample_id'] == sid
    for field in ['T_K', 'p_MPa']:
        t, p, inj, prod = pfm.predict_and_extract(ds, item, pfm.BEST_CKPT_PATH, device, field=field)
        diff, dlev = pfm._compute_diff_scale(t, p)
        gallery(diff, dlev, pfm.XRAY_GLOW_CMAP, 'both', LAB[field][1], inj, prod, grid,
                f'{PANELS}/diff_{field}_{sid}.png', 'white', 'black')
        vmin = np.nanmin([np.nanmin(f) for f in t]); vmax = np.nanmax([np.nanmax(f) for f in t])
        gallery(t, np.linspace(vmin, vmax, 16), 'inferno' if field == 'T_K' else 'magma', 'neither', LAB[field][0],
                inj, prod, grid, f'{PANELS}/true_{field}_{sid}.png', 'black', 'white', contour=(field == 'p_MPa'))


# ---------------------------------------------------------------- two-panel figures
from panel_font import panel_font
F = panel_font()
def compose(parts, out):
    ims = [Image.open(p).convert('RGB') for p in parts]
    W = max(i.width for i in ims); pad = 70
    H = sum(i.height + pad for i in ims)
    c = Image.new('RGB', (W, H), 'white'); y = 0; dr = ImageDraw.Draw(c)
    for k, im in enumerate(ims):
        dr.text((0, y + 5), f'({"ab"[k]})', font=F, fill='black')
        c.paste(im, ((W - im.width) // 2, y + pad)); y += im.height + pad
    c.save(os.path.join(OUT, out), dpi=(400, 400)); print(out, c.size)
o = PANELS + '/'
compose([o+'true_T_K_real_00008.png', o+'diff_T_K_real_00008.png'], 'fig8.png')
compose([o+'true_p_MPa_real_00008.png', o+'diff_p_MPa_real_00008.png'], 'fig9.png')
compose([o+'diff_T_K_real_01456.png', o+'diff_p_MPa_real_01456.png'], 'figA2.png')
compose([o+'diff_T_K_real_01528.png', o+'diff_p_MPa_real_01528.png'], 'figA3.png')
