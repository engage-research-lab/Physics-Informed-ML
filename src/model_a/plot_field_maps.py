"""
3D reservoir temperature field maps -- predicted vs. true, at years 0/20/40,
for one held-out test scenario. Intended for the written report / a
presentation slide, in the same spirit as the reservoir-map figures in the
PCNO paper.

Design:
- Years 0/20/40 fall exactly on the model's own 9-point uniform time grid
  (0,5,...,40), at indices 0/4/8 -- no interpolation needed, and this is
  also exactly the grid ComsolFnoDataset already returns per __getitem__,
  so no separate native-resolution (11-point) indexing is needed either.
- Renders each snapshot as several semi-transparent horizontal slices
  stacked through the reservoir's actual depth (not just one slice dressed
  up as a 3D block) -- built with Axes3D.contourf's zdir/offset mechanism,
  which projects a 2D color slice onto an arbitrary plane of a 3D axes,
  called once per depth layer. This was verified as necessary, not just
  nicer to look at: temperature genuinely varies WITH depth even at a
  single instant (e.g. at year 0, before any injection has occurred,
  T ranges ~383-391 K across depth at the same x,y -- confirmed directly
  against the raw data, not assumed), so a single depth slice hides real
  structure, not just an uninteresting flat plane. Both wells' locations
  come directly from the well_map channel (+1 = injector, -1 = producer --
  see build_well_location_channels in the Phase 1 prep script), not
  assumed, and are drawn as vertical lines spanning the plotted depth range.
- True vs. predicted share one color scale per figure (computed from the
  true field's own range across all 3 years), so the comparison is fair
  and the cooling trend over time is visually honest.
- A default scenario (index 0 of the test split) is used unless overridden
  -- deliberately NOT the known wells-pressure outlier scenario (real_01051),
  since that's a different failure mode and would misrepresent typical
  temperature-field behavior.
- Saves a companion metadata file (JSON + a plain-text version) with the
  scenario's identifying info and scalar parameters, so the exact reservoir
  being shown is easy to reference later for a report or slide without
  re-running this script.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import figure_style
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from comsol_fno_model import ComsolFnoModel
from comsol_fno_train import ComsolFnoDataset, denormalize, PREPARED_DATA_PATH, CHECKPOINT_DIR, UNIFORM_TIME_YEARS

BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')

# "X-ray" difference colormap (chosen 2026-07-30, picked from a real side-by-
# side comparison against 3 other candidates using actual model output, not
# guessed): black at zero so a correct/near-zero cell reads as empty space
# exactly like film with nothing to absorb, with glowing cyan for
# under-prediction and glowing orange/red for over-prediction so errors pop
# out of the darkness instead of blending into a washed-out pale color the
# way the old white-background coolwarm scheme did. Paired with a black axes
# background (dark_bg=True below) so "no error" truly disappears.
XRAY_GLOW_CMAP = LinearSegmentedColormap.from_list(
    'xray_glow', ['#00e5ff', '#00e5ff', '#000000', '#000000', '#ff8c00', '#ff3b00'], N=256)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'figures', 'field_maps')
YEARS_TO_PLOT = [0, 20, 40]


def load_scenario(scenario_idx):
    """Returns everything needed for one test scenario: model-ready tensors
    (from ComsolFnoDataset, so normalization matches training exactly) plus
    raw metadata (sample_id, params, grid coordinates) pulled directly from
    the prepared .npz, indexed the same way ComsolFnoDataset does."""
    test_dataset = ComsolFnoDataset(PREPARED_DATA_PATH, split='test')
    item = test_dataset[scenario_idx]

    raw = np.load(PREPARED_DATA_PATH, allow_pickle=True)
    test_idx = raw['test_idx']
    raw_idx = test_idx[scenario_idx]

    metadata = {
        'sample_id': str(raw['sample_id'][raw_idx]),
        'design_focus': str(raw['design_focus'][raw_idx]),
        'params': {name: float(raw['x_params'][raw_idx, i])
                   for i, name in enumerate(raw['parameter_names'])},
    }
    grid = {'x_m': raw['x_m'], 'y_m': raw['y_m'], 'z_m': raw['z_m']}
    return test_dataset, item, metadata, grid


FIELD_INFO = {
    'T_K': {'mask_key': 'mask_T', 'label': 'Temperature (degrees C)', 'diff_label': 'Predicted - True (degrees C)'},
    'p_MPa': {'mask_key': 'mask_P', 'label': 'Pressure (MPa)', 'diff_label': 'Predicted - True (MPa)'},
}

KELVIN_TO_CELSIUS_OFFSET = 273.15


def predict_and_extract(test_dataset, item, ckpt_path, device, field='T_K'):
    """Runs the model on one scenario and returns true/predicted `field`
    ('T_K' in Celsius, despite the channel's own Kelvin-based name; see
    below, or 'p_MPa' in real MPa, NaN outside the valid domain) at years
    0/20/40, plus the injector/producer grid cell indices."""
    model = ComsolFnoModel(width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
                            unet_base_channels=4, n_layers=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    x_static = torch.from_numpy(item['x_static']).unsqueeze(0).to(device)
    params = torch.from_numpy(item['params']).unsqueeze(0).to(device)
    well_map = torch.from_numpy(item['well_map']).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x_static, params, well_map)

    mean, std = test_dataset.stats[f'field__{field}__mean'], test_dataset.stats[f'field__{field}__std']
    pred_f = denormalize(out[field][0], mean, std).cpu().numpy()          # (60,30,20,9)
    true_f = denormalize(torch.from_numpy(item[field]), mean, std).numpy()  # (60,30,20,9)
    mask = item[FIELD_INFO[field]['mask_key']] > 0.5  # (60,30,20,9)

    # Converted here, once, at the source, rather than in each plotting
    # function -- the model/dataset still store and denormalize to Kelvin
    # (T_K is the real channel name throughout the codebase), but every
    # figure this module produces displays Celsius. Subtracting the same
    # constant offset from both pred_f and true_f before any diff is taken
    # elsewhere leaves diff values unchanged (a Celsius difference equals
    # the same Kelvin difference), so this one conversion is sufficient for
    # both the absolute-value and the difference panels.
    if field == 'T_K':
        pred_f = pred_f - KELVIN_TO_CELSIUS_OFFSET
        true_f = true_f - KELVIN_TO_CELSIUS_OFFSET

    pred_f = np.where(mask, pred_f, np.nan)
    true_f = np.where(mask, true_f, np.nan)

    year_idx = [UNIFORM_TIME_YEARS.index(y) for y in YEARS_TO_PLOT]
    pred_by_year = [pred_f[..., i] for i in year_idx]  # each (60,30,20)
    true_by_year = [true_f[..., i] for i in year_idx]

    well_map_raw = item['well_map']  # (60,30,20), +1 injector / -1 producer
    inj_pos = np.argwhere(well_map_raw > 0.5)[0]   # (ix, iy, iz)
    prod_pos = np.argwhere(well_map_raw < -0.5)[0]

    return true_by_year, pred_by_year, inj_pos, prod_pos


N_DEPTH_LAYERS = 6


def _draw_box(ax, field, x_m, y_m, z_m, inj_pos, prod_pos, levels, cmap, dark_bg=False, extend='neither',
              show_contour_lines=True):
    """Draws one field snapshot as several semi-transparent horizontal
    slices stacked through the reservoir's actual valid depth range (not
    a single slice dressed up as a solid block) -- so depth-wise structure
    is genuinely visible, not hidden behind one plane. Confirmed this
    matters, not just looks nicer: temperature varies substantially with
    depth even at a fixed instant (~383-391 K across depth at year 0, at
    the same x,y -- see module docstring), so a single slice was throwing
    away real information the user specifically asked to see.

    dark_bg=True switches to the "x-ray" presentation used for the
    Predicted-True difference panel (2026-07-30): black axes background so
    a correct/near-zero cell truly disappears instead of reading as a pale
    washed-out color, paired with XRAY_GLOW_CMAP. The cosmetic side walls
    also switch to a thin dark outline instead of the light-gray fill used
    for the True/Predicted panels, so they don't reintroduce a hazy gray
    plane across the black backdrop."""
    valid_z = np.where(~np.isnan(field[inj_pos[0], inj_pos[1], :]))[0]
    shallow_idx, deep_idx = valid_z[np.argmax(z_m[valid_z])], valid_z[np.argmin(z_m[valid_z])]
    layer_indices = np.unique(np.linspace(deep_idx, shallow_idx, N_DEPTH_LAYERS).round().astype(int))

    z_top_val, z_bot_val = z_m[shallow_idx], z_m[deep_idx]
    x0, x1 = x_m[0], x_m[-1]
    y0, y1 = y_m[0], y_m[-1]

    if dark_bg:
        ax.set_facecolor('black')

    # Plain side walls, purely cosmetic (not real data) -- just enough to
    # read as a bounded 3D block around the real, data-bearing layers.
    side_faces = [
        [(x0, y0, z_top_val), (x1, y0, z_top_val), (x1, y0, z_bot_val), (x0, y0, z_bot_val)],  # front
        [(x0, y1, z_top_val), (x1, y1, z_top_val), (x1, y1, z_bot_val), (x0, y1, z_bot_val)],  # back
        [(x0, y0, z_top_val), (x0, y1, z_top_val), (x0, y1, z_bot_val), (x0, y0, z_bot_val)],  # left
        [(x1, y0, z_top_val), (x1, y1, z_top_val), (x1, y1, z_bot_val), (x1, y0, z_bot_val)],  # right
    ]
    if dark_bg:
        ax.add_collection3d(Poly3DCollection(side_faces, facecolor='black', edgecolor='#555555',
                                              linewidth=0.5, alpha=1.0))
    else:
        ax.add_collection3d(Poly3DCollection(side_faces, facecolor='lightgray', edgecolor='gray',
                                              linewidth=0.5, alpha=0.15))

    # `levels` is passed as an explicit array (np.linspace(vmin, vmax, N)),
    # not an integer count -- verified this matters, not just style: with
    # an integer count alongside vmin/vmax, matplotlib can still derive the
    # actual level boundaries from each call's own LOCAL data range rather
    # than the shared global range, silently producing a different,
    # inconsistent color scale per layer/year/panel despite vmin/vmax being
    # set "correctly". An explicit levels array removes that ambiguity.
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    cs = None
    # Alpha for the dark-bg "x-ray" panel (2026-07-30, revised): originally
    # set to 0.85 to make the glow colors pop, but this was found to badly
    # occlude depth -- verified directly (per-layer 2D checks, no 3D
    # compositing) that EVERY sampled depth genuinely has real, similarly-
    # sized structure, yet the stacked 3D box only ever showed the
    # shallowest layer. At alpha=0.85, after even 2 stacked layers only
    # (1-0.85)^2 ~= 2% of what's underneath survives -- by the 3rd-4th
    # layer the deeper structure is fully painted over. Dropping to 0.35
    # lets layers actually blend (still an imperfect, blended composite,
    # not true occlusion-free volumetric rendering -- see
    # plot_diff_depth_gallery for the clean, non-composited per-depth view).
    layer_alpha = 0.35 if dark_bg else 0.6
    # Contour-line overlay : the filled bands'
    # boundaries alone are colormap-on-colormap, so the concentric rings
    # around each well (the actual signal of interest -- how far the
    # pressure/temperature disturbance reaches, and how that changes across
    # years) are hard to trace by eye, especially where two adjacent bands
    # have similar luminance. Neon lime green is maximally distant (in hue)
    # from inferno/magma's black->purple->red/orange->yellow range, and
    # green sits at the peak of human luminance sensitivity, so it reads as
    # "bright" even at a thin linewidth -- picked after cyan (#00e5ff) was
    # tried and judged not bright enough against the fill.
    #
    # show_contour_lines=False for the diff panel specifically: it already
    # has its own diverging coolwarm scheme for a different purpose (error,
    # not the field value itself), and the same green lines drawn on top of
    # THAT scale were found to look like unrelated clutter, not a helpful
    # ring -- confirmed by rendering it and comparing before/after.
    line_color = '#39ff14' if not dark_bg else '#ffffff'
    for zi in layer_indices:
        data_layer = field[:, :, zi]
        cs = ax.contourf(X, Y, data_layer, zdir='z', offset=z_m[zi], levels=levels,
                          cmap=cmap, alpha=layer_alpha, extend=extend)

    # Line pass done AFTER every layer's fill, and NOT via ax.contour()'s
    # batched Line3DCollection (tried first, both interleaved and as its
    # own separate pass, including an explicit numeric zorder override --
    # none of it worked: mplot3d sorts artists by an approximate mean-depth
    # heuristic, not true per-pixel occlusion, and a Line3DCollection kept
    # getting painted back over by a later layer's semi-transparent fill,
    # muting the color toward gray/olive regardless of draw order or
    # zorder -- confirmed by sampling actual rendered pixel color, not just
    # eyeballing a thumbnail, and confirmed the color itself was correct in
    # an isolated single-layer test).
    #
    # The fix: extract each contour ring's path (via a throwaway, never-
    # rendered ax.contour() call's .allsegs) and redraw it as individual
    # ax.plot3D() calls instead -- the same artist type already used for
    # the well trajectories below, which DO render at full brightness in
    # this exact stacked-layer scene. mplot3d evidently treats a plain
    # Line3D more favorably in its depth-sort than a Line3DCollection.
    if show_contour_lines:
        for zi in layer_indices:
            data_layer = field[:, :, zi]
            cs_lines = ax.contour(X, Y, data_layer, zdir='z', offset=z_m[zi], levels=levels)
            cs_lines.remove()
            for level_segs in cs_lines.allsegs:
                for seg in level_segs:
                    if len(seg) < 2:
                        continue
                    ax.plot3D(seg[:, 0], seg[:, 1], z_m[zi], color=line_color,
                              linewidth=0.9, zorder=1000 + zi)

    # Well trajectories -- vertical lines spanning the plotted depth range
    # (both wells are vertical in this dataset). Injector was originally
    # white (made sense back when this panel had a black x-ray background),
    # but every panel here is light-background now -- white-on-white made
    # the injector invisible. Switched to black for real contrast.
    #
    # Halo added : even after that switch, a
    # plain linewidth=2 line in either color still got lost against this
    # box's own busy interior -- black crossing a dark/purple colormap
    # region, or red crossing a warm red/orange one, both effectively
    # vanish, confirmed directly from a user screenshot where the wells
    # were "barely distinguishable" against the green contour rings and
    # stacked semi-transparent layers. Fixed the same way as the
    # depth-gallery point markers: draw a wide white halo first, then the
    # real black/red line on top, narrower -- so the well reads as a bold
    # line against every part of the colormap, not just the parts it
    # happens to contrast with on its own.
    for pos, color, name in [(inj_pos, 'black', 'Injector'), (prod_pos, 'red', 'Producer')]:
        xw, yw = x_m[pos[0]], y_m[pos[1]]
        ax.plot3D([xw, xw], [yw, yw], [z_top_val, z_bot_val], color='white',
                   linewidth=6, solid_capstyle='round', zorder=2000)
        ax.plot3D([xw, xw], [yw, yw], [z_top_val, z_bot_val], color=color,
                   linewidth=3, solid_capstyle='round', label=name, zorder=2001)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z_bot_val, z_top_val)
    ax.set_axis_off()  # matching the reference figure's clean, label-free box style
    ax.view_init(elev=22, azim=-60)
    return cs


def _compute_diff_scale(true_by_year, pred_by_year):
    """Predicted-True for each plotted year, plus a shared symmetric "x-ray"
    color scale bound on the 99th percentile of |diff| (not the literal
    max) -- factored out so the 3D box figure and the depth-slice gallery
    always use an IDENTICAL scale, never two independently-computed ones
    that could silently drift apart.

    Bound choice: the literal max|diff| was tried first and found to be a
    real problem, not just a style nitpick -- checked directly against a
    real scenario's error distribution and found the true max is a
    single-cell outlier (~45K for T_K) while the 99th percentile is much
    smaller (~2-8K depending on year); using the literal max as the
    symmetric scale bound compressed all the genuinely-informative moderate
    errors (hundreds of cells in the 5-20K range) into an indistinguishable
    near-black band, defeating the whole point of the x-ray scheme. Using
    the 99th percentile instead stretches that real structure across the
    visible range; genuinely extreme cells beyond it simply clip to the
    cmap's most saturated color via extend='both' (passed through to
    contourf by callers), which is itself informative -- fully-saturated
    color reads as "at or beyond the 99th percentile", not a plotting bug."""
    diff_by_year = [p - t for p, t in zip(pred_by_year, true_by_year)]
    all_abs_diffs = np.concatenate([np.abs(d[~np.isnan(d)]) for d in diff_by_year])
    diff_absmax = np.percentile(all_abs_diffs, 99)
    diff_levels = np.linspace(-diff_absmax, diff_absmax, 17)
    return diff_by_year, diff_levels


def plot_3d_boxes(true_by_year, pred_by_year, inj_pos, prod_pos, grid, metadata, out_dir, model_label, field='T_K'):
    """Builds and saves the main 3x3 field-comparison figure for one
    scenario: rows are years 0/20/40, columns are True / Predicted /
    Predicted-True, each panel a stacked-depth-slice 3D box (via
    `_draw_box`) for the given `field` ('T_K' in degrees C or 'p_MPa' in
    MPa).

    Args:
        true_by_year, pred_by_year: lists of 3 (60,30,20) arrays (one per
            plotted year), the true/predicted field values with cap-rock
            cells as NaN, as returned by `predict_and_extract`.
        inj_pos, prod_pos: (ix, iy, iz) grid-cell indices of the injector
            and producer wells, as returned by `predict_and_extract`.
        grid: dict with 'x_m'/'y_m'/'z_m' 1D coordinate arrays.
        metadata: dict with the scenario's 'sample_id', 'design_focus', and
            'params' (as returned by `load_scenario`), used for the title.
        out_dir: directory the PNG is saved into.
        model_label: short label (e.g. 'ModelA') used in the output filename.
        field: which output channel is being plotted, 'T_K' or 'p_MPa'.

    Saves the figure to `<out_dir>/field_map_3d_<model_label>_<field>_<sample_id>.png`
    and returns nothing.
    """
    x_m, y_m, z_m = grid['x_m'], grid['y_m'], grid['z_m']
    # Pressure previously used 'viridis' --
    # a cool blue-green-yellow palette from a different perceptual family
    # than temperature's warm 'inferno', which made the two fields hard to
    # read consistently side by side. 'magma' is in the same warm,
    # perceptually-uniform family as inferno (both black->purple->red/orange
    # ->yellow) while remaining visually distinct from it.
    cmap = 'inferno' if field == 'T_K' else 'magma'

    vmin = np.nanmin([np.nanmin(f) for f in true_by_year])
    vmax = np.nanmax([np.nanmax(f) for f in true_by_year])
    levels = np.linspace(vmin, vmax, 16)

    # Difference = Predicted - True. The "x-ray" black-bg/glow treatment was
    # tried here too (2026-07-30) but reverted: this stacked-slice 3D box
    # was found to be unreliable for showing real depth structure regardless
    # of color scheme or alpha (a matplotlib mplot3d limitation -- it has no
    # true per-pixel depth compositing for overlapping semi-transparent
    # surfaces, confirmed by comparing against plot_diff_depth_gallery's
    # clean, non-composited per-depth panels, which revealed substantial
    # error the box was hiding entirely). Since this panel can't be trusted
    # for depth-specific detail either way, kept as the plain, familiar
    # coolwarm-on-white scheme -- simpler, and doesn't imply a false
    # precision the dark "x-ray" treatment's dramatic look would suggest.
    # The 99th-percentile scale bound (see _compute_diff_scale) is kept
    # regardless of color scheme -- that fix was about the scale itself,
    # not the colors, and remains necessary either way.
    diff_by_year, diff_levels = _compute_diff_scale(true_by_year, pred_by_year)
    diff_cmap = 'coolwarm'

    # Prefer a few directly interpretable params for the figure title; the
    # full parameter list is always in the saved metadata file regardless.
    # phi0 and c_compaction added (2026-07-31) alongside the original 4 --
    # both are among the parameters already found to correlate with
    # field-level prediction bias, so having them on-screen is directly
    # useful for explaining scenario-to-scenario differences, not just
    # decorative context.
    headline_params = ['well_spacing', 'inj_rate', 'inj_temp_C', 'H', 'phi0', 'c_compaction']
    shown = [p for p in headline_params if p in metadata['params']] or list(metadata['params'])[:4]
    title_bits = ", ".join(f"{k}={metadata['params'][k]:.3g}" for k in shown)

    fig = plt.figure(figsize=(15, 4 * len(YEARS_TO_PLOT)))
    cs_main, cs_diff = None, None
    main_axes, diff_axes = [], []
    panels = [("True", true_by_year, levels, cmap), ("Predicted", pred_by_year, levels, cmap),
              ("Pred. - True", diff_by_year, diff_levels, diff_cmap)]
    for row, year in enumerate(YEARS_TO_PLOT):
        for col, (label, by_year, lvls, cm) in enumerate(panels):
            ax = fig.add_subplot(len(YEARS_TO_PLOT), 3, row * 3 + col + 1, projection='3d')
            # Contour-line overlay requested for the pressure box specifically
            # (2026-08-23) -- tried on temperature too, but with temperature's
            # much larger, more irregular cooling-front shapes the same lines
            # made the box look busy/cluttered rather than clearer, and the
            # user asked to revert T_K to its original (no-overlay) look
            # while keeping it for p_MPa.
            cs = _draw_box(ax, by_year[row], x_m, y_m, z_m, inj_pos, prod_pos, lvls, cm,
                            dark_bg=False, extend=('both' if col == 2 else 'neither'),
                            show_contour_lines=(col < 2 and field == 'p_MPa'))
            ax.set_title(f"Year {year} — {label}", fontsize=12)
            if col < 2:
                cs_main, main_axes = cs, main_axes + [ax]
            else:
                cs_diff, diff_axes = cs, diff_axes + [ax]
            if row == 0 and col == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.colorbar(cs_main, ax=main_axes, label=FIELD_INFO[field]['label'], shrink=0.6, pad=0.05)
    fig.colorbar(cs_diff, ax=diff_axes, label=FIELD_INFO[field]['diff_label'], shrink=0.6, pad=0.12)
    fig.suptitle(f"{metadata['sample_id']} ({model_label}, {field})\n{title_bits}", y=0.99)
    # Legend sits below the boxes so it never covers a panel. The stacked-slice
    # diff panel is only an approximate composite across depth; that caveat
    # belongs in the figure caption (per-depth detail: plot_diff_depth_gallery).
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.45, 0.06), frameon=False)
    out_path = os.path.join(out_dir, f"field_map_3d_{model_label}_{field}_{metadata['sample_id']}.png")
    fig.savefig(out_path, dpi=figure_style.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diff_depth_gallery(true_by_year, pred_by_year, inj_pos, prod_pos, grid, metadata, out_dir, model_label, field='T_K'):
    """Companion figure to plot_3d_boxes' diff panel (2026-07-30): one clean
    2D slice per depth layer, per year, with zero compositing/occlusion.

    Added because the 3D box's stacked semi-transparent layers, even after
    lowering alpha, only ever show a BLENDED composite of all depths, not
    each depth's own true pattern -- verified directly that every sampled
    depth in a real scenario has real, non-trivial structure the 3D box was
    hiding. This is the unambiguous "what's
    really happening at each depth" view; the 3D box is the "big picture at
    a glance" view. Same XRAY_GLOW_CMAP and the same 99th-percentile scale
    as plot_3d_boxes' diff panel (via _compute_diff_scale), so the two
    figures are directly comparable, never independently-scaled."""
    x_m, y_m, z_m = grid['x_m'], grid['y_m'], grid['z_m']
    diff_by_year, diff_levels = _compute_diff_scale(true_by_year, pred_by_year)

    valid_z = np.where(~np.isnan(diff_by_year[0][inj_pos[0], inj_pos[1], :]))[0]
    shallow_idx, deep_idx = valid_z[np.argmax(z_m[valid_z])], valid_z[np.argmin(z_m[valid_z])]
    layer_indices = np.unique(np.linspace(deep_idx, shallow_idx, N_DEPTH_LAYERS).round().astype(int))
    n_layers = len(layer_indices)

    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    fig, axes = plt.subplots(len(YEARS_TO_PLOT), n_layers,
                              figsize=(2.6 * n_layers, 2.6 * len(YEARS_TO_PLOT)),
                              squeeze=False)
    cs = None
    for row, (year, diff_f) in enumerate(zip(YEARS_TO_PLOT, diff_by_year)):
        for col, zi in enumerate(layer_indices):
            ax = axes[row][col]
            ax.set_facecolor('black')
            cs = ax.contourf(X, Y, diff_f[:, :, zi], levels=diff_levels, cmap=XRAY_GLOW_CMAP, extend='both')
            # Markers given a bold black edge and doubled size: at the original markersize=4/no-edge styling,
            # the red producer marker routinely disappeared into this
            # colormap's own red/orange positive-diff fill, and the plain
            # white injector marker had no guaranteed contrast either --
            # confirmed by inspecting real_01528's pressure gallery
            # specifically, where the producer marker sat directly inside
            # an orange contour band. A black edge reads against both the
            # warm (orange/red) and cool (cyan) halves of this diverging
            # colormap and against the black background alike, so it works
            # regardless of which fill color a given well happens to land on.
            ax.plot(x_m[inj_pos[0]], y_m[inj_pos[1]], marker='o', color='white', markersize=8,
                    markeredgecolor='black', markeredgewidth=1.4, zorder=10)
            ax.plot(x_m[prod_pos[0]], y_m[prod_pos[1]], marker='o', color='red', markersize=8,
                    markeredgecolor='black', markeredgewidth=1.4, zorder=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f"z={z_m[zi]:.0f} m", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Year {year}", fontsize=12)

    fig.colorbar(cs, ax=axes, label=FIELD_INFO[field]['diff_label'], shrink=0.7, pad=0.02)
    fig.suptitle(f"{metadata['sample_id']} ({model_label}, {field}) — Predicted-True by depth layer\n"
                 f"(deepest -> shallowest, left to right; white=Injector, red=Producer)", y=0.995)
    out_path = os.path.join(out_dir, f"field_map_depth_gallery_{model_label}_{field}_{metadata['sample_id']}.png")
    fig.savefig(out_path, dpi=figure_style.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_field_depth_gallery(true_by_year, inj_pos, prod_pos, grid, metadata, out_dir, model_label, field='T_K'):
    """True-field companion to plot_diff_depth_gallery (2026-08-23), same
    motivation: the 3D box's stacked semi-transparent layers can't show a
    given depth's own contour rings cleanly, because mplot3d has no true
    per-pixel depth compositing for overlapping 3D artists -- confirmed
    directly this time for the contour-LINE overlay specifically (added to
    _draw_box to make the well-influence rings easier to trace year over
    year): the requested bright line color rendered correctly in an
    isolated single-layer test, but got muted toward gray/olive by
    mplot3d's automatic depth-sort once multiple stacked layers were
    present, regardless of draw order or an explicit zorder override. This
    plain-2D gallery has no such compositing to fight -- each panel is a
    single flat contourf/contour with nothing else drawn behind it, so the
    line color renders exactly as specified. Shows the True field only
    (this is about seeing the real ring pattern evolve, not verifying
    prediction accuracy -- see plot_diff_depth_gallery for that)."""
    x_m, y_m, z_m = grid['x_m'], grid['y_m'], grid['z_m']
    cmap = 'inferno' if field == 'T_K' else 'magma'

    vmin = np.nanmin([np.nanmin(f) for f in true_by_year])
    vmax = np.nanmax([np.nanmax(f) for f in true_by_year])
    levels = np.linspace(vmin, vmax, 16)

    valid_z = np.where(~np.isnan(true_by_year[0][inj_pos[0], inj_pos[1], :]))[0]
    shallow_idx, deep_idx = valid_z[np.argmax(z_m[valid_z])], valid_z[np.argmin(z_m[valid_z])]
    layer_indices = np.unique(np.linspace(deep_idx, shallow_idx, N_DEPTH_LAYERS).round().astype(int))
    n_layers = len(layer_indices)

    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    fig, axes = plt.subplots(len(YEARS_TO_PLOT), n_layers,
                              figsize=(2.6 * n_layers, 2.6 * len(YEARS_TO_PLOT)),
                              squeeze=False)
    cs = None
    for row, (year, true_f) in enumerate(zip(YEARS_TO_PLOT, true_by_year)):
        for col, zi in enumerate(layer_indices):
            ax = axes[row][col]
            data_layer = true_f[:, :, zi]
            cs = ax.contourf(X, Y, data_layer, levels=levels, cmap=cmap, extend='neither')
            # Green contour-line overlay restricted to pressure only
            # (2026-08-28), matching the same field-specific reversal
            # already applied to plot_3d_boxes: temperature reverted to its
            # original look (filled contourf only, no line overlay) and
            # pressure keeps the bright overlay.
            if field == 'p_MPa':
                ax.contour(X, Y, data_layer, levels=levels, colors='#39ff14', linewidths=0.9)
            # Same marker-visibility fix as plot_diff_depth_gallery, edge
            # color flipped to white here since this panel's own background
            # (inferno/magma, dark-to-bright) is the opposite tonal range.
            ax.plot(x_m[inj_pos[0]], y_m[inj_pos[1]], marker='o', color='black', markersize=8,
                    markeredgecolor='white', markeredgewidth=1.4, zorder=10)
            ax.plot(x_m[prod_pos[0]], y_m[prod_pos[1]], marker='o', color='red', markersize=8,
                    markeredgecolor='white', markeredgewidth=1.4, zorder=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f"z={z_m[zi]:.0f} m", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Year {year}", fontsize=12)

    fig.colorbar(cs, ax=axes, label=FIELD_INFO[field]['label'], shrink=0.7, pad=0.02)
    fig.suptitle(f"{metadata['sample_id']} ({model_label}, {field}) — True field by depth layer\n"
                 f"(deepest -> shallowest, left to right; black=Injector, red=Producer)", y=0.995)
    out_path = os.path.join(out_dir, f"field_map_true_depth_gallery_{model_label}_{field}_{metadata['sample_id']}.png")
    fig.savefig(out_path, dpi=figure_style.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_depth_change_profile(true_by_year, inj_pos, prod_pos, grid, metadata, out_dir, model_label, field='T_K'):
    """Summary companion to plot_diff_depth_gallery (2026-08-18): a single
    change-magnitude-vs-depth profile, so "which depths actually change" is
    visible in one glance instead of having to eyeball every panel of the
    depth gallery and do the comparison by hand.

    Motivation: for a real scenario (real_00008), only ~6 of the reservoir's
    12 vertical layers show ANY true field change over the 40-year run --
    the rest (more than 3 layers above/below the wells' own completion
    depth) are flat to within numerical noise, because advection (the fast
    heat/pressure transport mechanism) only acts along the flow path near
    the wells' own depth; layers the flowing water never reaches only see
    slow conduction, which 40 years isn't enough time to move meaningfully.
    This was previously only discoverable by manually diffing year-0 vs.
    year-40 arrays per z-layer -- this plot makes it a first-class, visible
    part of the standard output instead.

    Uses TRUE (not predicted) values, since the question this answers is
    about real reservoir physics, not model accuracy -- see
    plot_diff_depth_gallery for the prediction-error view, which is a
    related but different question (and, not coincidentally, tends to peak
    at the same depths this plot identifies as "active").

    change metric = max(|value(year=40) - value(year=0)|) across all valid
    X,Y cells at that depth -- deliberately max, not mean, so a layer with
    a small but real hot/cold pocket doesn't get averaged away to look flat.
    """
    x_m, y_m, z_m = grid['x_m'], grid['y_m'], grid['z_m']
    year0, year_last = true_by_year[0], true_by_year[-1]
    well_z = z_m[inj_pos[2]]  # inj_pos/prod_pos share a z-index (see build_well_location_channels)

    max_change, mean_change, has_data = [], [], []
    for zi in range(len(z_m)):
        delta = year_last[:, :, zi] - year0[:, :, zi]
        valid = np.isfinite(delta)
        has_data.append(bool(valid.any()))
        if valid.any():
            max_change.append(float(np.max(np.abs(delta[valid]))))
            mean_change.append(float(np.mean(np.abs(delta[valid]))))
        else:
            max_change.append(np.nan)
            mean_change.append(np.nan)
    max_change, mean_change, has_data = np.array(max_change), np.array(mean_change), np.array(has_data)

    fig, ax = plt.subplots(figsize=(6, 7))
    reservoir = has_data  # cap-rock layers (no data at all) excluded from the bars entirely
    dz = abs(z_m[1] - z_m[0])

    # Shade the full simulated-reservoir extent first, so "valid data, zero
    # change" (shaded, empty) reads differently from "cap rock, no data at
    # all" (unshaded white) -- without this, both look identical (blank).
    if reservoir.any():
        z_top = z_m[reservoir].max() + dz / 2
        z_bot = z_m[reservoir].min() - dz / 2
        ax.axhspan(z_bot, z_top, color='#cfd8dc', alpha=0.4, zorder=0,
                   label='Simulated reservoir extent (has data)')

    ax.barh(z_m[reservoir], max_change[reservoir], height=dz * 0.85, zorder=2,
            color='#e85d04', alpha=0.85, label='Max |change|, year 0->40')
    ax.barh(z_m[reservoir], mean_change[reservoir], height=dz * 0.5, zorder=3,
            color='#03045e', alpha=0.9, label='Mean |change|, year 0->40')
    ax.axhline(well_z, color='black', linestyle='--', linewidth=1.5, zorder=5,
               label=f'Wells\' completion depth (z={well_z:.0f} m)')

    unit = FIELD_INFO[field]['label'].split('(')[-1].rstrip(')')
    ax.set_xlabel(f'|Change|, year 0 -> 40 ({unit})')
    ax.set_ylabel('Depth, z (m)')
    ax.set_title(f"{metadata['sample_id']} ({field}) — where does the reservoir actually change?\n"
                 f"Cap-rock layers (no simulated data) excluded from the axis entirely", fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"field_map_depth_change_profile_{model_label}_{field}_{metadata['sample_id']}.png")
    fig.savefig(out_path, dpi=figure_style.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def save_metadata(metadata, ckpt_path, out_dir, model_label):
    """Saves the scenario's identifying info and scalar parameters as both
    JSON (for scripts/charts) and a plain-text table (for quickly pasting
    into a report or slide notes without needing to parse JSON)."""
    metadata_full = {**metadata, 'checkpoint': ckpt_path, 'years_plotted': YEARS_TO_PLOT}
    json_path = os.path.join(out_dir, f"field_map_metadata_{model_label}_{metadata['sample_id']}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_full, f, indent=2)

    txt_path = os.path.join(out_dir, f"field_map_metadata_{model_label}_{metadata['sample_id']}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Scenario: {metadata['sample_id']}  (design_focus={metadata['design_focus']})\n")
        f.write(f"Model checkpoint: {ckpt_path}\n")
        f.write(f"Years plotted: {YEARS_TO_PLOT}\n\n")
        f.write("Scalar parameters:\n")
        for name, val in metadata['params'].items():
            f.write(f"  {name}: {val:.4g}\n")
    print(f"Saved {json_path}\nSaved {txt_path}")


def main():
    """CLI entry point: loads one test scenario, runs the model on it, and
    generates all four field-map figures (3D box comparison, diff depth
    gallery, true-field depth gallery, depth change profile) plus the
    companion metadata file, for the requested checkpoint/scenario/field.
    Takes no arguments (reads `--ckpt-path`, `--model-label`,
    `--scenario-idx`, `--field` from the CLI) and returns nothing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=BEST_CKPT_PATH,
                         help='Path to the checkpoint to visualize (default: Model A\'s best.pth)')
    parser.add_argument('--model-label', default='ModelA',
                         help='Short label used in output filenames (e.g. ModelA, ModelB)')
    parser.add_argument('--scenario-idx', type=int, default=0,
                         help='Index within the test split to visualize (default 0, an ordinary scenario -- '
                              'deliberately not the known wells-pressure outlier scenario)')
    parser.add_argument('--field', default='T_K', choices=['T_K', 'p_MPa'],
                         help='Which field to visualize (default T_K)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.ckpt_path}')

    test_dataset, item, metadata, grid = load_scenario(args.scenario_idx)
    print(f"Scenario: {metadata['sample_id']}  (design_focus={metadata['design_focus']})")

    true_by_year, pred_by_year, inj_pos, prod_pos = predict_and_extract(
        test_dataset, item, args.ckpt_path, device, field=args.field)

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_3d_boxes(true_by_year, pred_by_year, inj_pos, prod_pos, grid, metadata, OUT_DIR, args.model_label, field=args.field)
    plot_diff_depth_gallery(true_by_year, pred_by_year, inj_pos, prod_pos, grid, metadata, OUT_DIR, args.model_label, field=args.field)
    plot_field_depth_gallery(true_by_year, inj_pos, prod_pos, grid, metadata, OUT_DIR, args.model_label, field=args.field)
    plot_depth_change_profile(true_by_year, inj_pos, prod_pos, grid, metadata, OUT_DIR, args.model_label, field=args.field)
    save_metadata(metadata, args.ckpt_path, OUT_DIR, args.model_label)


if __name__ == '__main__':
    main()
