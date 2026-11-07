"""
Line plots of the AVERAGE (across all 232 test scenarios) well-output
trajectory -- True vs Predicted -- for each of the 3 predicted well
channels (T_prod_K, P_inj_MPa, P_prod_MPa), over the full 201-step/40-year
well-time grid.

Companion to plot_field_maps.py, which shows one scenario's 3D field
snapshots at a time (years 0/20/40, coarse 9-point grid). This script
instead aggregates ACROSS scenarios to show the overall shape of the
model's fit at the well-output's full time resolution -- the gap between
the True and Predicted lines at any point in time is the average error at
that point, directly visualizing what evaluate_test_set.py's aggregate
RE% numbers summarize into a single figure per channel.

Averaging across scenarios (rather than picking one) means each
channel's trend reflects the TYPICAL scenario, not one that could be an
outlier -- important here since real_01051's late-time P_prod_MPa
collapse is a documented, extreme per-scenario failure (RE_wells=315% for
that one scenario alone) that would dominate and misrepresent a
single-scenario plot; averaging dilutes it into the ensemble the way the
project's own aggregate accuracy numbers already do.
"""
import argparse
import os

import matplotlib.pyplot as plt
import figure_style
import numpy as np
import torch
from torch.utils.data import DataLoader

from comsol_fno_model import ComsolFnoModel
from comsol_fno_train import ComsolFnoDataset, collate, denormalize, PREPARED_DATA_PATH, CHECKPOINT_DIR

BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'figures', 'field_maps')

# (column name, plot label, index into the 3-channel wells tensor -- matches
# WellOutputHead's documented channel order: [T_prod_K, P_inj_MPa, P_prod_MPa])
CHANNELS = [
    ('T_prod_K', 'Production Temperature (K)', 0),
    ('P_inj_MPa', 'Injection Pressure (MPa)', 1),
    ('P_prod_MPa', 'Production Pressure (MPa)', 2),
]


def compute_avg_trajectories(ckpt_path, batch_size, device):
    """Returns (avg_pred, avg_true), each (201, 3) in real physical units --
    the mean over all 232 test scenarios at every well-timestep, computed
    as a running sum/count rather than concatenating all scenarios in
    memory first (only matters at larger scale, but costs nothing here)."""
    test_dataset = ComsolFnoDataset(PREPARED_DATA_PATH, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = ComsolFnoModel(width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
                            unet_base_channels=4, n_layers=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    stats = test_dataset.stats
    well_mean_std = [
        (stats['well__T_prod_K__mean'], stats['well__T_prod_K__std']),
        (stats['well__P_inj_MPa__mean'], stats['well__P_inj_MPa__std']),
        (stats['well__P_prod_MPa__mean'], stats['well__P_prod_MPa__std']),
    ]

    sum_pred, sum_true, n = None, None, 0
    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['x_static'].to(device)
            params = batch['params'].to(device)
            well_map = batch['well_map'].to(device)
            target_wells = batch['wells'].to(device)

            out = model(x_static, params, well_map)
            pred_wells_real = torch.stack(
                [denormalize(out['wells'][:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
            target_wells_real = torch.stack(
                [denormalize(target_wells[:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)

            batch_pred_sum = pred_wells_real.sum(dim=0).cpu().numpy()  # (201, 3)
            batch_true_sum = target_wells_real.sum(dim=0).cpu().numpy()
            sum_pred = batch_pred_sum if sum_pred is None else sum_pred + batch_pred_sum
            sum_true = batch_true_sum if sum_true is None else sum_true + batch_true_sum
            n += pred_wells_real.shape[0]

    print(f"Averaged over {n} test scenarios")
    return sum_pred / n, sum_true / n


def plot_channel(well_time_years, avg_true, avg_pred, channel_idx, name, label, out_dir, model_label):
    """Draws the True-vs-Predicted average trajectory line plot for one
    well-output channel over the full well-time grid, and saves it to disk
    as a PNG.

    well_time_years: (201,) array of well-timestep ages in years, spanning
    the 40-year horizon. avg_true, avg_pred: (201, 3) arrays of the
    across-scenario mean true/predicted well outputs in real physical
    units, as returned by compute_avg_trajectories(). channel_idx: index
    (0, 1, or 2) into the 3-channel wells axis, matching CHANNELS'
    [T_prod_K, P_inj_MPa, P_prod_MPa] order. name: short channel name used
    in the output filename (e.g. 'T_prod_K'). label: human-readable
    channel label with units, used for the y-axis and title. out_dir:
    directory the PNG is saved into. model_label: short model identifier
    (e.g. 'ModelA') included in the title and filename. Returns nothing.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(well_time_years, avg_true[:, channel_idx], color='#1f77b4', linewidth=2, label='True (avg)')
    ax.plot(well_time_years, avg_pred[:, channel_idx], color='#d62728', linewidth=2, linestyle='--',
             label='Predicted (avg)')
    ax.set_xlabel('Year')
    ax.set_ylabel(label)
    ax.set_title(f'Average {label} over Time -- True vs Predicted\n'
                 f'(mean across all 232 test scenarios, {model_label})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'avg_trajectory_{name}_{model_label}.png')
    fig.savefig(out_path, dpi=figure_style.DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    """CLI entry point: parses --ckpt-path/--model-label/--batch-size, loads
    the well-time grid from the prepared dataset, runs
    compute_avg_trajectories() over the test split, and calls plot_channel()
    for each of the 3 well-output channels to save one figure per channel
    into OUT_DIR. Takes no arguments (reads sys.argv) and returns nothing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=BEST_CKPT_PATH)
    parser.add_argument('--model-label', default='ModelA')
    parser.add_argument('--batch-size', type=int, default=8,
                         help='Drop to 2 on a 4GB local GPU (8 OOM\'d previously); fine at 8 on ARC.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.ckpt_path}')

    raw = np.load(PREPARED_DATA_PATH, allow_pickle=True)
    well_time_years = raw['well_time_years']
    del raw

    avg_pred, avg_true = compute_avg_trajectories(args.ckpt_path, args.batch_size, device)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, label, idx in CHANNELS:
        plot_channel(well_time_years, avg_true, avg_pred, idx, name, label, OUT_DIR, args.model_label)


if __name__ == '__main__':
    main()
