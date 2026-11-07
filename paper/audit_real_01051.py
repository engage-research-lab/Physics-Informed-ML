"""Audit of the extreme production-pressure failure (real_01051), Section 3.2.

Checks, in order:
  1. Alignment of the prepared data with the raw COMSOL export (well curves and
     fields), if --raw-dir is given.
  2. Phase consistency: well and cell pressures (gauge + 1 atm) against the
     IAPWS-IF97 saturation pressure at the simulated local temperature.
  3. Absolute production-pressure errors of Models A and B on the test set.
  4. Near-wellbore drawdown (field pressure at the producer cells minus the well
     pressure) and its relation to the near-producer permeability.
  5. How rare low initial production pressures are in training, and how the
     production-pressure error grows as the initial pressure falls.
  6. Sharp curvature in the simulated production-pressure curve.
  7. Fine-scale vs coarse permeability around the producer, if --realizations-dir is given.

Usage: python audit_real_01051.py [--raw-dir DIR] [--realizations-dir DIR]
Results are printed and saved to results/metrics/audit_real_01051.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import common as C

SCENARIO = 'real_01051'


def producer_columns(well_map):
    """(x, y) indices of the producer cells (well_map = -1) in a (X, Y, Z) map."""
    xs, ys, _ = np.where(well_map < -0.5)
    return sorted(set(zip(xs.tolist(), ys.tolist())))


def alignment(raw, raw_dir):
    """Max |prepared - raw| for the well curves (CSV) and the fields (HDF5) of SCENARIO."""
    import h5py
    s = list(raw['sample_id']).index(SCENARIO)
    csv = np.genfromtxt(Path(raw_dir) / 'wells' / f'{SCENARIO}_well_outputs.csv', delimiter=',', comments='%')
    wells = raw['y_wells'][s]
    out = {'T_prod_K': float(np.abs(csv[:, 1] - wells[:, 0]).max()),
           'P_inj_MPa': float(np.abs(csv[:, 2] / 1e6 - wells[:, 1]).max()),
           'P_prod_MPa': float(np.abs(csv[:, 3] / 1e6 - wells[:, 2]).max())}
    with h5py.File(Path(raw_dir) / 'fields' / f'{SCENARIO}_fields.h5', 'r') as f:
        T = np.transpose(f['T_K'][:], (3, 2, 1, 0))
        P = np.transpose(f['p_Pa'][:], (3, 2, 1, 0)) / 1e6
    out['field_T_K'] = float(np.abs(T - raw['T_K'][s])[raw['mask_T'][s]].max())
    out['field_p_MPa'] = float(np.abs(P - raw['p_MPa'][s])[raw['mask_P'][s]].max())
    return out


def phase_check(raw):
    """Margin above saturation (MPa) at the producer of SCENARIO and across the dataset."""
    yw, t = raw['y_wells'], raw['well_time_years'].astype(float)
    margin = (yw[:, :, 2] + C.ATM_MPA) - C.psat_mpa(yw[:, :, 0])
    s = list(raw['sample_id']).index(SCENARIO)
    by_year = {}
    for y in (0, 10, 20, 30, 35, 40):
        j = int(np.argmin(np.abs(t - y)))
        by_year[f'{y}yr'] = {'P_abs_MPa': float(yw[s, j, 2] + C.ATM_MPA), 'T_C': float(yw[s, j, 0] - 273.15),
                             'Psat_MPa': float(C.psat_mpa(yw[s, j, 0])), 'margin_MPa': float(margin[s, j])}
    worst = np.argsort(margin.min(1))[:5]
    cell_min = []
    for k in range(len(raw['sample_id'])):
        m = raw['mask_T'][k] & raw['mask_P'][k]
        if m.any():
            cell_min.append(float(((raw['p_MPa'][k][m] + C.ATM_MPA) - C.psat_mpa(raw['T_K'][k][m])).min()))
    return {'producer_by_year': by_year,
            'lowest_well_margins': [(str(raw['sample_id'][i]), float(margin[i].min())) for i in worst],
            'scenarios_with_well_below_saturation': int((margin.min(1) < 0).sum()),
            'min_cell_margin_any_scenario_MPa': float(np.min(cell_min))}


def absolute_errors(device):
    """Per-scenario MAE of the production pressure for Models A and B on the test set."""
    raw = C.load_prepared()
    ids = list(raw['sample_id'][raw['test_idx']])
    ds = C.ComsolFnoDataset(C.PREPARED_DATA_PATH, split='test')
    out = {}
    for which in ('A', 'B'):
        model = C.load_model(which, device)
        pred, true = [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=2, shuffle=False, collate_fn=C.collate):
                o = model(b['x_static'].to(device), b['params'].to(device), b['well_map'].to(device))
                pred.append(C.denorm_wells(o['wells'], ds.stats)[:, :, 2].cpu().numpy())
                true.append(C.denorm_wells(b['wells'].to(device), ds.stats)[:, :, 2].cpu().numpy())
        P, T = np.concatenate(pred), np.concatenate(true)
        mae = np.abs(P - T).mean(1)
        i = ids.index(SCENARIO)
        out[f'Model {which}'] = {'MAE_MPa': float(mae[i]), 'test_median_MAE_MPa': float(np.median(mae)),
                                 'rank_of_232': int((mae > mae[i]).sum()) + 1,
                                 'pred_y0_y40': [float(P[i, 0]), float(P[i, -1])],
                                 'true_y0_y40': [float(T[i, 0]), float(T[i, -1])],
                                 'abs_error_y0': float(abs(P[i, 0] - T[i, 0]))}
    return out


def drawdown(raw):
    """Field pressure at the producer cells minus the well pressure (year 0), and its link to permeability."""
    names = list(raw['static_channel_names'])
    X = raw['x_static_fields']
    wm_all, lk_all = X[:, names.index('well_map')], X[:, names.index('log10_perm')]
    ids = list(raw['sample_id'])
    dd, k_cells, k_window, p0 = [], [], [], raw['y_wells'][:, 0, 2]
    for s in range(len(ids)):
        prod = wm_all[s] < -0.5
        valid = prod & raw['mask_P'][s][..., 0]
        dd.append(raw['p_MPa'][s][..., 0][valid].mean() - p0[s])
        k_cells.append(lk_all[s][prod].mean())
        zs = np.where(prod.any(axis=(0, 1)))[0]
        win = []
        for (i, j) in producer_columns(wm_all[s]):
            sl = (slice(max(i - 2, 0), i + 3), slice(max(j - 2, 0), j + 3))
            for z in zs:
                m = raw['mask_P'][s][sl + (z, 0)]
                win.extend(lk_all[s][sl + (z,)][m].tolist())
        k_window.append(np.mean(win))
    dd, k_cells, k_window = map(np.array, (dd, k_cells, k_window))
    s = ids.index(SCENARIO)
    return {'drawdown_y0_MPa': float(dd[s]), 'drawdown_percentile': float(100 * (dd < dd[s]).mean()),
            'drawdown_median_MPa': float(np.median(dd)),
            'corr_drawdown_vs_inverse_perm_producer_cells': float(np.corrcoef(dd, 10 ** (-k_cells))[0, 1]),
            'near_producer_perm_5x5_percentile': float(100 * (k_window < k_window[s]).mean()),
            'corr_P0_vs_near_producer_perm_5x5': float(np.corrcoef(p0, k_window)[0, 1])}


def rarity_and_trend(raw):
    """Training scenarios with low initial production pressure and the test error trend."""
    p0 = raw['y_wells'][:, 0, 2]
    train_p0 = p0[raw['train_idx']]
    ev = json.loads((C.METRICS_DIR / 'model_a_test_results.json').read_text())['per_scenario']
    ids = list(raw['sample_id'])
    x = np.array([p0[ids.index(s)] for s in ev['sample_id']])
    re = np.array(ev['re_P_prod_MPa'])
    order = np.argsort(x)
    return {'P0_rank_of_1535': int((p0 < p0[ids.index(SCENARIO)]).sum()) + 1,
            'P0_median_MPa': float(np.median(p0)),
            'train_below_5MPa': int((train_p0 < 5).sum()), 'train_below_7.5MPa': int((train_p0 < 7.5).sum()),
            'test_lowest10_median_RE_pct': float(np.median(re[order[:10]])),
            'test_rest_median_RE_pct': float(np.median(re[order[10:]])),
            'corr_logRE_vs_P0': float(np.corrcoef(np.log10(re), x)[0, 1])}


def curvature(raw):
    """Largest second differences of the simulated production pressure of SCENARIO."""
    s = list(raw['sample_id']).index(SCENARIO)
    p, t = raw['y_wells'][s, :, 2], raw['well_time_years'].astype(float)
    d2 = np.diff(p, 2)
    top = np.argsort(np.abs(d2))[::-1][:4]
    return {'largest': [(float(t[i + 1]), float(d2[i])) for i in sorted(top)],
            'median_abs_d2_all_scenarios': float(np.median(np.abs(np.diff(raw['y_wells'][:, :, 2], 2, axis=1))))}


def fine_vs_coarse(raw, real_dir, radius=40.0):
    """Mean log10 permeability within `radius` m of the producer: fine-scale realization vs model input."""
    names = list(raw['static_channel_names'])
    s = list(raw['sample_id']).index(SCENARIO)
    wm, lk = raw['x_static_fields'][s, names.index('well_map')], raw['x_static_fields'][s, names.index('log10_perm')]
    (i, j), = producer_columns(wm)[:1]
    zs = np.where(raw['mask_P'][s][i, j, :, 0])[0]
    z_lo, z_hi = raw['z_m'][zs].min(), raw['z_m'][zs].max()
    fine = np.genfromtxt(Path(real_dir) / f'{SCENARIO}.csv', delimiter=',', names=True)
    near = (np.hypot(fine['x'] - raw['x_m'][i], fine['y'] - raw['y_m'][j]) <= radius) & (fine['z'] >= z_lo) & (fine['z'] <= z_hi)
    return {'fine_log10k_near_producer': float(np.log10(fine['perm'][near]).mean()), 'n_fine_cells': int(near.sum()),
            'coarse_log10k_producer_column': float(lk[i, j, zs].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw-dir', help='folder with wells/ and fields/ raw COMSOL exports')
    parser.add_argument('--realizations-dir', help='folder with the fine-scale real_XXXXX.csv realizations')
    args = parser.parse_args()
    raw = C.load_prepared()
    result = {}
    if args.raw_dir:
        result['alignment_max_abs_diff'] = alignment(raw, args.raw_dir)
    result['phase'] = phase_check(raw)
    result['absolute_errors'] = absolute_errors(C.get_device())
    result['drawdown'] = drawdown(raw)
    result['rarity_and_trend'] = rarity_and_trend(raw)
    result['curvature'] = curvature(raw)
    if args.realizations_dir:
        result['fine_vs_coarse_permeability'] = fine_vs_coarse(raw, args.realizations_dir)
    path = C.out_path('audit_real_01051.json')
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print('saved', path)


if __name__ == '__main__':
    main()
