"""
Final test-set evaluation -- the honest, unbiased accuracy number for the
written report.

test_idx (232 scenarios) has NOT been touched by anything so far: all 4
training runs and the per-scenario val diagnostic only ever used
train_idx/val_idx. val was repeatedly used to compare 4 different
configurations (baseline, weight_decay, well-head fix, n_layers=8), so it's
no longer a fully unbiased estimate -- test exists precisely to give one
number that wasn't used to pick the winning configuration.

Only run this once the model design is actually finalized -- if this result
leads to another architecture change, test stops being a clean final check
and the whole point of holding it back is lost.

Pure inference against an existing checkpoint -- no training. Reuses
ComsolFnoDataset/collate/denormalize from comsol_fno_train.py, and the same
masked_relative_error/relative_error/well_output_mse conventions used during
training, so these numbers are directly comparable to the val_re_*/val_loss_*
figures already reported.

IMPORTANT: the architecture kwargs passed to ComsolFnoModel below MUST match
whichever checkpoint is being loaded (width/modes/n_layers/etc. are not
stored in the checkpoint file itself, and are unchanged between Model A and
Model B -- only the training loss differs). Both n_layers=8 and modes4=5
were tried as capacity/resolution increases and found to be clean negatives
(no accuracy gain, modes4=5 even slightly worse on RE_wells).

Checkpoint selection (2026-07-21): after finding the T+P-only and then the
summed (T+P+wells) selection criteria both had real failure modes (see
comsol_fno_train.py's own comments), the worst-case (max) criterion was
adopted, and both models were retrained across 2 seeds to check reliability.
Seed 0 was picked as the canonical reported checkpoint for both models (not
because it scored better -- it's the project's standing default seed,
matching PCNO's own torch.manual_seed(0) convention); seed 1's results are
documented as a robustness/limitation note rather than folded into the
headline number. Run this script with --ckpt-path/--out-name to evaluate
either model's checkpoint without duplicating this file.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from comsol_fno_model import ComsolFnoModel, masked_relative_error, relative_error, _read_at_well
from comsol_fno_train import ComsolFnoDataset, collate, denormalize, PREPARED_DATA_PATH, CHECKPOINT_DIR, UNIFORM_TIME_YEARS

BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'metrics')


def masked_relative_error_per_sample(pred, target, mask, eps=1e-8):
    """Per-scenario version of masked_relative_error -- one RE value per
    batch element instead of one aggregate value for the whole batch."""
    B = pred.shape[0]
    mask = mask.float()
    rel = torch.abs(pred - target) / (torch.abs(target) + eps)
    numerator = (rel * mask).reshape(B, -1).sum(dim=1)
    denominator = mask.reshape(B, -1).sum(dim=1).clamp(min=1.0)
    return (numerator / denominator) * 100.0


def relative_error_per_sample(pred, target, eps=1e-8):
    """Per-scenario version of relative_error, for the well outputs (100%
    finite, no masking needed)."""
    B = pred.shape[0]
    rel = torch.abs(pred - target) / (torch.abs(target) + eps)
    return rel.reshape(B, -1).mean(dim=1) * 100.0


def summarize(name, values):
    """Prints summary statistics (mean, median, std, min, max, p10, p90, all
    in %) for a list of per-scenario relative-error values, labeled `name`."""
    values = np.asarray(values)
    print(f'\n{name}: n={len(values)}')
    print(f'  mean={values.mean():.3f}%  median={np.median(values):.3f}%  '
          f'std={values.std():.3f}%  min={values.min():.3f}%  max={values.max():.3f}%')
    print(f'  p10={np.percentile(values, 10):.3f}%  p90={np.percentile(values, 90):.3f}%')


def main():
    """CLI entry point: runs the model against the held-out test split
    (232 scenarios) and reports the final, unbiased RE_T/RE_P/RE_wells
    accuracy numbers.

    Loads the checkpoint at `--ckpt-path`, evaluates every test scenario in
    batches of `--batch-size`, and prints the aggregate RE (matching the
    training-time convention: one mean over all valid cells/timesteps across
    the whole split, not a mean of per-scenario means) for temperature,
    pressure, and each well-output channel (T_prod_K, P_inj_MPa, P_prod_MPa),
    plus per-scenario distribution summaries, a diagnostic breakdown of the
    single worst and top-10 worst wells scenarios, a design_focus breakdown,
    and correlations between per-scenario RE and each scalar parameter.
    Saves the full set of results as JSON to `results/metrics/<--out-name>`.
    Takes no arguments and returns nothing; all inputs come from the parsed
    CLI args.
    """
    # --ckpt-path lets this same script evaluate Model B (physics-constrained,
    # checkpoints/model_b/best.pth) without duplicating the file --
    # architecture is identical for both models, only the training loss
    # differed, so nothing else here needs to change. --out-name keeps the
    # two models' results from overwriting each other in results/metrics/.
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=BEST_CKPT_PATH,
                         help='Path to the checkpoint to evaluate (default: Model A\'s best.pth)')
    parser.add_argument('--out-name', default='model_a_test_results.json',
                         help='Filename (not full path) for the results JSON, saved under results/metrics/')
    parser.add_argument('--batch-size', type=int, default=8,
                         help='DataLoader batch size (default 8, fine on ARC A100s -- drop to 2 '
                              'on a 4GB local GPU, which OOM\'d at 8 previously)')
    args = parser.parse_args()
    results_json_path = os.path.join(RESULTS_DIR, args.out_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.ckpt_path}')

    test_dataset = ComsolFnoDataset(PREPARED_DATA_PATH, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f'Test: {len(test_dataset)} scenarios')

    # sample_id isn't carried by ComsolFnoDataset itself -- load it directly,
    # indexed by test_idx to stay aligned with test_dataset's own row order
    # (ComsolFnoDataset subsets every array by data[f'{split}_idx'] the same way).
    raw = np.load(PREPARED_DATA_PATH, allow_pickle=True)
    test_idx = raw['test_idx']
    sample_ids = raw['sample_id'][test_idx]
    design_focus = raw['design_focus'][test_idx]
    well_time_years = raw['well_time_years']
    field_time_years = UNIFORM_TIME_YEARS  # the 9 field snapshots the model actually uses (see comsol_fno_train.py)

    model = ComsolFnoModel(width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
                            unet_base_channels=4, n_layers=4).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    stats = test_dataset.stats
    t_mean, t_std = stats['field__T_K__mean'], stats['field__T_K__std']
    p_mean, p_std = stats['field__p_MPa__mean'], stats['field__p_MPa__std']
    well_mean_std = [
        (stats['well__T_prod_K__mean'], stats['well__T_prod_K__std']),
        (stats['well__P_inj_MPa__mean'], stats['well__P_inj_MPa__std']),
        (stats['well__P_prod_MPa__mean'], stats['well__P_prod_MPa__std']),
    ]

    # Aggregate metrics -- mirrors run_epoch's convention exactly (mean over
    # ALL valid cells across the whole test set, not a mean-of-per-scenario-
    # means), so these are the numbers directly comparable to the val_re_*
    # figures already reported for the 4 training runs.
    agg_re_t_num, agg_re_t_den = 0.0, 0.0
    agg_re_p_num, agg_re_p_den = 0.0, 0.0
    agg_re_wells_sum, agg_re_wells_n = 0.0, 0

    per_scenario_re_t, per_scenario_re_p, per_scenario_re_wells = [], [], []
    all_pred_wells, all_target_wells = [], []  # kept for diagnosing the worst-case scenario below
    all_field_p_prod_pred, all_field_p_prod_target = [], []  # field-level p_MPa at the producer cell (9 field timesteps) -- checks whether the late-time bias exists BEFORE WellOutputHead's upsampling

    # Per-channel breakdown of the 3 well outputs -- the combined RE_wells
    # number hides whether error is spread evenly across T_prod_K/P_inj_MPa/
    # P_prod_MPa or concentrated in one of them (the worst-case scenario
    # diagnostic suggested P_prod_MPa specifically).
    channel_names = ['T_prod_K', 'P_inj_MPa', 'P_prod_MPa']
    agg_re_chan_sum = [0.0, 0.0, 0.0]
    agg_re_chan_n = [0, 0, 0]
    per_scenario_re_chan = [[], [], []]

    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['x_static'].to(device)
            params = batch['params'].to(device)
            well_map = batch['well_map'].to(device)
            target_T = batch['T_K'].to(device)
            target_P = batch['p_MPa'].to(device)
            mask_T = batch['mask_T'].to(device)
            mask_P = batch['mask_P'].to(device)
            target_wells = batch['wells'].to(device)

            out = model(x_static, params, well_map)

            pred_T_real = denormalize(out['T_K'], t_mean, t_std)
            target_T_real = denormalize(target_T, t_mean, t_std)
            pred_P_real = denormalize(out['p_MPa'], p_mean, p_std)
            target_P_real = denormalize(target_P, p_mean, p_std)

            pred_wells_real = torch.stack(
                [denormalize(out['wells'][:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
            target_wells_real = torch.stack(
                [denormalize(target_wells[:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)

            # Field-level p_MPa read off at the producer cell, at the 9 field
            # timesteps -- this is exactly what WellOutputHead receives as
            # input BEFORE its own MLP upsamples to 201 well-timesteps.
            prod_mask = (well_map < -0.5)
            field_p_prod_pred = _read_at_well(pred_P_real, prod_mask)      # (B, 9)
            field_p_prod_target = _read_at_well(target_P_real, prod_mask)  # (B, 9)
            all_field_p_prod_pred.append(field_p_prod_pred.cpu().numpy())
            all_field_p_prod_target.append(field_p_prod_target.cpu().numpy())

            # aggregate accumulation (same convention as masked_relative_error/relative_error)
            m_t = mask_T.float()
            rel_t = torch.abs(pred_T_real - target_T_real) / (torch.abs(target_T_real) + 1e-8)
            agg_re_t_num += (rel_t * m_t).sum().item()
            agg_re_t_den += m_t.sum().item()

            m_p = mask_P.float()
            rel_p = torch.abs(pred_P_real - target_P_real) / (torch.abs(target_P_real) + 1e-8)
            agg_re_p_num += (rel_p * m_p).sum().item()
            agg_re_p_den += m_p.sum().item()

            rel_wells = torch.abs(pred_wells_real - target_wells_real) / (torch.abs(target_wells_real) + 1e-8)
            agg_re_wells_sum += rel_wells.mean().item() * pred_wells_real.shape[0]
            agg_re_wells_n += pred_wells_real.shape[0]

            # same rel_wells tensor, split out per channel (dim -1 is [T_prod_K, P_inj_MPa, P_prod_MPa])
            for c in range(3):
                rel_c = rel_wells[:, :, c]  # (B, 201)
                agg_re_chan_sum[c] += rel_c.mean().item() * pred_wells_real.shape[0]
                agg_re_chan_n[c] += pred_wells_real.shape[0]
                per_scenario_re_chan[c].extend((rel_c.mean(dim=1) * 100.0).cpu().tolist())

            per_scenario_re_t.extend(masked_relative_error_per_sample(pred_T_real, target_T_real, mask_T).cpu().tolist())
            per_scenario_re_p.extend(masked_relative_error_per_sample(pred_P_real, target_P_real, mask_P).cpu().tolist())
            per_scenario_re_wells.extend(relative_error_per_sample(pred_wells_real, target_wells_real).cpu().tolist())

            all_pred_wells.append(pred_wells_real.cpu().numpy())
            all_target_wells.append(target_wells_real.cpu().numpy())

    print('\n=== FINAL TEST-SET ACCURACY (aggregate, matches training convention) ===')
    print(f'RE_T:     {100.0 * agg_re_t_num / agg_re_t_den:.3f}%')
    print(f'RE_P:     {100.0 * agg_re_p_num / agg_re_p_den:.3f}%')
    print(f'RE_wells: {100.0 * agg_re_wells_sum / agg_re_wells_n:.3f}%')
    print(f'\n(Project target: <= 2% RE for each)')

    print('\n=== RE_wells broken down by channel ===')
    for c, name in enumerate(channel_names):
        print(f'{name}: aggregate={100.0 * agg_re_chan_sum[c] / agg_re_chan_n[c]:.3f}%')

    summarize('Per-scenario RE_T', per_scenario_re_t)
    summarize('Per-scenario RE_P', per_scenario_re_p)
    summarize('Per-scenario RE_wells', per_scenario_re_wells)
    for c, name in enumerate(channel_names):
        summarize(f'Per-scenario RE_{name}', per_scenario_re_chan[c])

    # --- Diagnose the worst-case wells scenario -----------------------------
    # Checks whether an extreme per-scenario RE_wells is a genuine prediction
    # failure or a metric artifact (relative error blows up near a target
    # value close to zero, even for a small absolute error).
    per_scenario_re_wells_arr = np.array(per_scenario_re_wells)
    pred_wells_all = np.concatenate(all_pred_wells, axis=0)      # (N, 201, 3)
    target_wells_all = np.concatenate(all_target_wells, axis=0)  # (N, 201, 3)
    field_p_prod_pred_all = np.concatenate(all_field_p_prod_pred, axis=0)      # (N, 9)
    field_p_prod_target_all = np.concatenate(all_field_p_prod_target, axis=0)  # (N, 9)

    worst_idx = int(np.argmax(per_scenario_re_wells_arr))
    print(f"\n=== Worst wells scenario: sample_id={sample_ids[worst_idx]}  "
          f"RE_wells={per_scenario_re_wells_arr[worst_idx]:.3f}% ===")

    pred_w = pred_wells_all[worst_idx]      # (201, 3)
    target_w = target_wells_all[worst_idx]  # (201, 3)
    for c, name in enumerate(channel_names):
        t_min, t_max, t_mean_c = target_w[:, c].min(), target_w[:, c].max(), target_w[:, c].mean()
        print(f'\n  {name}: target range [{t_min:.4f}, {t_max:.4f}], mean={t_mean_c:.4f}')
        rel_c = np.abs(pred_w[:, c] - target_w[:, c]) / (np.abs(target_w[:, c]) + 1e-8)
        worst_t_idx = np.argsort(-rel_c)[:5]
        print(f'  Top 5 worst timesteps for {name}:')
        for ti in worst_t_idx:
            print(f'    year={well_time_years[ti]:.2f}  pred={pred_w[ti, c]:.4f}  '
                  f'target={target_w[ti, c]:.4f}  abs_diff={abs(pred_w[ti, c] - target_w[ti, c]):.4f}  '
                  f'rel_error={rel_c[ti] * 100:.1f}%')

    # --- Check whether the tail of bad wells scenarios shares a common trait -
    # Does the same late-time P_prod_MPa collapse repeat across the worst
    # scenarios (not just #1), and does it correlate with any scalar param
    # or design_focus category? Answers whether this is a specific,
    # avoidable design/geology regime or a scattered rare event.
    N_WORST = 10
    worst_order = np.argsort(-per_scenario_re_wells_arr)[:N_WORST]
    print(f'\n=== Top {N_WORST} worst wells scenarios: per-channel breakdown + late-time P_prod_MPa check ===')
    prod_c = channel_names.index('P_prod_MPa')
    for i in worst_order:
        target_prod = target_wells_all[i, :, prod_c]
        pred_prod = pred_wells_all[i, :, prod_c]
        min_idx = int(np.argmin(target_prod))
        print(f'  sample_id={sample_ids[i]}  design_focus={design_focus[i]}  '
              f'RE_wells={per_scenario_re_wells_arr[i]:.3f}%  '
              f'RE_T_prod_K={per_scenario_re_chan[0][i]:.3f}%  '
              f'RE_P_inj_MPa={per_scenario_re_chan[1][i]:.3f}%  '
              f'RE_P_prod_MPa={per_scenario_re_chan[2][i]:.3f}%')
        print(f'    P_prod_MPa min target={target_prod[min_idx]:.4f} at year={well_time_years[min_idx]:.2f}  '
              f'(model predicted {pred_prod[min_idx]:.4f} there)')
        # Field-level p_MPa at the producer cell, at the last 3 field
        # timesteps (years 30, 35, 40) -- this is what WellOutputHead
        # receives as input, BEFORE its own MLP upsamples to 201 steps.
        # If this already shows the same over-prediction, the bias
        # originates in the FieldNet; if not, WellOutputHead introduces it.
        print(f'    Field-level p_MPa at producer cell (input to WellOutputHead), last 3 field snapshots:')
        for fi in [6, 7, 8]:
            print(f'      year={field_time_years[fi]}  pred={field_p_prod_pred_all[i, fi]:.4f}  '
                  f'target={field_p_prod_target_all[i, fi]:.4f}  '
                  f'abs_diff={abs(field_p_prod_pred_all[i, fi] - field_p_prod_target_all[i, fi]):.4f}')

    print('\n=== design_focus breakdown of RE_wells (all 232 test scenarios) ===')
    for focus in np.unique(design_focus):
        m = design_focus == focus
        print(f'  {focus}: n={m.sum()}  mean RE_wells={per_scenario_re_wells_arr[m].mean():.3f}%  '
              f'median={np.median(per_scenario_re_wells_arr[m]):.3f}%  max={per_scenario_re_wells_arr[m].max():.3f}%')

    # Correlation of RE_wells (combined + P_prod_MPa specifically) with each
    # scalar param -- looks for a specific design/geology regime associated
    # with the tail failures, the same way the earlier temperature diagnostic did.
    param_names = test_dataset.param_names
    params_raw = raw['x_params'][test_idx][:, test_dataset.param_keep_idx]
    per_scenario_re_prod = np.array(per_scenario_re_chan[2])

    print('\n=== Correlation of RE_wells (combined) with each scalar param ===')
    correlations = [(name, np.corrcoef(per_scenario_re_wells_arr, params_raw[:, i])[0, 1])
                    for i, name in enumerate(param_names)]
    for name, r in sorted(correlations, key=lambda x: -abs(x[1])):
        print(f'  {name}: r={r:+.3f}')

    print('\n=== Correlation of RE_P_prod_MPa specifically with each scalar param ===')
    correlations_prod = [(name, np.corrcoef(per_scenario_re_prod, params_raw[:, i])[0, 1])
                          for i, name in enumerate(param_names)]
    for name, r in sorted(correlations_prod, key=lambda x: -abs(x[1])):
        print(f'  {name}: r={r:+.3f}')

    # Save results to JSON -- printed output above is not persisted anywhere
    # otherwise, making it impossible to build report_charts.py's final-RE
    # and per-scenario-distribution charts from a real (non-hardcoded) run.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        'checkpoint_epoch': ckpt['epoch'],
        'n_scenarios': len(per_scenario_re_t),
        'aggregate': {
            're_t': 100.0 * agg_re_t_num / agg_re_t_den,
            're_p': 100.0 * agg_re_p_num / agg_re_p_den,
            're_wells': 100.0 * agg_re_wells_sum / agg_re_wells_n,
            're_wells_by_channel': {
                name: 100.0 * agg_re_chan_sum[c] / agg_re_chan_n[c]
                for c, name in enumerate(channel_names)
            },
        },
        'per_scenario': {
            'sample_id': [str(s) for s in sample_ids],
            'design_focus': [str(d) for d in design_focus],
            're_t': per_scenario_re_t,
            're_p': per_scenario_re_p,
            're_wells': per_scenario_re_wells,
            **{f're_{name}': per_scenario_re_chan[c] for c, name in enumerate(channel_names)},
        },
        'param_names': list(param_names),
        'param_correlations_re_wells': {name: r for name, r in correlations},
        'param_correlations_re_p_prod_mpa': {name: r for name, r in correlations_prod},
    }
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved test-set results to {results_json_path}')


if __name__ == '__main__':
    main()
