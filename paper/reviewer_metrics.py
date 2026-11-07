"""Computes the quantities added to the manuscript in response to the reviewer.

Sections (each written to results/metrics/reviewer_metrics.json):
  test_set        Model A errors in physical units on the 232 test scenarios
                  (Section 3.1), the persistence baseline (Section 3.3) and the
                  physics residuals of COMSOL, Model A and Model B (Section 2.3).
  ensemble        Surrogate vs COMSOL indicators over all 1,535 scenarios:
                  thermal capacity, cumulative energy, temperature decline in %,
                  in K and relative to the usable temperature difference,
                  screening agreement, apparent warming and the correlation of the
                  decline with each input (Section 3.5).
  stabilization   How quickly the simulated well pressures settle, from the
                  201-point COMSOL well curves (Section 3.3).
  mask_erosion    Share of valid cells kept by the physics-loss mask erosion,
                  by reservoir thickness (Section 2.3).
  timing          Model A forward-pass time per scenario on the local GPU (Section 4.2).

Usage: python reviewer_metrics.py [--skip-timing]
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

import common as C
from comsol_fno_train_pinn import ComsolFnoDatasetPINN, DT_SECONDS
from comsol_fno_physics import build_physics_loss, erode_mask_3d


def test_set_section(device):
    """Physical-unit errors, persistence baseline and physics residuals on the test set."""
    raw = C.load_prepared()
    static_names = list(raw['static_channel_names'])
    param_names = [n for n in raw['parameter_names'] if n != 'well_radius']
    A, B = C.load_model('A', device), C.load_model('B', device)
    ds = ComsolFnoDatasetPINN(C.PREPARED_DATA_PATH, split='test')
    st = ds.stats
    tm, tsd = st['field__T_K__mean'], st['field__T_K__std']
    pm, psd = st['field__p_MPa__mean'], st['field__p_MPa__std']
    qi, ti = param_names.index('inj_rate'), param_names.index('inj_temp')

    acc = {}

    def add(key, value):
        acc[key] = acc.get(key, 0.0) + float(value)

    residuals = {'COMSOL': ([], []), 'Model A': ([], []), 'Model B': ([], [])}
    pw, tw, q, tin = [], [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=2, shuffle=False, collate_fn=C.collate):
            xs, prm, wm = b['x_static'].to(device), b['params'].to(device), b['well_map'].to(device)
            mT, mP = b['mask_T'].to(device).float(), b['mask_P'].to(device).float()
            tT, tP = C.denormalize(b['T_K'].to(device), tm, tsd), C.denormalize(b['p_MPa'].to(device), pm, psd)
            oA, oB = A(xs, prm, wm), B(xs, prm, wm)
            pT, pP = C.denormalize(oA['T_K'], tm, tsd), C.denormalize(oA['p_MPa'], pm, psd)

            later_T, later_P = mT.clone(), mP.clone()
            later_T[..., 0] = 0
            later_P[..., 0] = 0
            affected = ((tT - tT[..., :1]).abs() > 1.0).float() * later_T   # cells cooled by > 1 K after year 0
            eT, eP = (pT - tT).abs() * mT, (pP - tP).abs() * mP
            persist_T = (tT[..., :1] - tT).abs() * mT                      # repeat the true year-0 field
            add('T_abs', eT.sum()); add('T_sq', (((pT - tT) ** 2) * mT).sum()); add('T_n', mT.sum())
            add('P_abs', eP.sum()); add('P_sq', (((pP - tP) ** 2) * mP).sum()); add('P_n', mP.sum())
            add('aff_err', (eT * affected).sum()); add('aff_n', affected.sum())
            add('aff_signal', ((tT - tT[..., :1]).abs() * affected).sum())
            add('T_err_later', (eT * later_T).sum())
            add('dP_err', (((pP - pP[..., :1]) - (tP - tP[..., :1])).abs() * later_P).sum())
            add('dP_signal', ((tP - tP[..., :1]).abs() * later_P).sum()); add('dP_n', later_P.sum())
            add('persist_T', persist_T.sum()); add('persist_aff', (persist_T * affected).sum())
            add('persist_T_later', (persist_T * later_T).sum())
            add('persist_P', ((tP[..., :1] - tP).abs() * mP).sum())

            phys = build_physics_loss(x_static_raw=b['x_static_raw'].to(device), params_raw=b['params_raw'].to(device),
                                      static_channel_names=static_names, param_names=param_names,
                                      z_m=raw['z_m'], x_m=raw['x_m'], y_m=raw['y_m'], dt_seconds=DT_SECONDS)
            eroded = erode_mask_3d(mT * mP)
            fields = {'COMSOL': (tP, tT), 'Model A': (pP, pT),
                      'Model B': (C.denormalize(oB['p_MPa'], pm, psd), C.denormalize(oB['T_K'], tm, tsd))}
            for name, (P_, T_) in fields.items():
                mass, energy = phys(P_, T_ - 273.15, valid_mask=eroded)   # the loss expects degrees C
                residuals[name][0].append(float(mass))
                residuals[name][1].append(float(energy))

            pw.append(C.denorm_wells(oA['wells'], st).cpu().numpy())
            tw.append(C.denorm_wells(b['wells'].to(device), st).cpu().numpy())
            q.append(b['params_raw'][:, qi].numpy())
            tin.append(b['params_raw'][:, ti].numpy())

    P, T = np.concatenate(pw), np.concatenate(tw)
    q, tin = np.concatenate(q), np.concatenate(tin)
    err = np.abs(P - T)
    test_ids = list(raw['sample_id'][raw['test_idx']])
    i_out = test_ids.index('real_01051')
    cool_true, cool_pred = T[:, 0, 0] - T[:, -1, 0], P[:, 0, 0] - P[:, -1, 0]
    dp_true, dp_pred = T[:, :, 1] - T[:, :, 2], P[:, :, 1] - P[:, :, 2]
    q_true, q_pred = C.thermal_power_mw(T[:, :, 0], q, tin), C.thermal_power_mw(P[:, :, 0], q, tin)
    return {
        'relative_error_epsilon': '1e-8 in the unit of each quantity',
        'field': {
            'T_MAE_K': acc['T_abs'] / acc['T_n'], 'T_RMSE_K': (acc['T_sq'] / acc['T_n']) ** 0.5,
            'P_MAE_MPa': acc['P_abs'] / acc['P_n'], 'P_RMSE_MPa': (acc['P_sq'] / acc['P_n']) ** 0.5,
            'cooled_cells_pct_of_valid': 100 * acc['aff_n'] / acc['T_n'],
            'T_MAE_cooled_cells_K': acc['aff_err'] / acc['aff_n'],
            'T_error_pct_of_cooling_signal': 100 * acc['aff_err'] / acc['aff_signal'],
            'P_change_MAE_MPa': acc['dP_err'] / acc['dP_n'],
            'P_change_error_pct_of_change': 100 * acc['dP_err'] / acc['dP_signal'],
        },
        'persistence_baseline': {
            'T_MAE_K': acc['persist_T'] / acc['T_n'],
            'T_MAE_cooled_cells_K': acc['persist_aff'] / acc['aff_n'],
            'P_MAE_MPa': acc['persist_P'] / acc['P_n'],
            'P_change_MAE_MPa': acc['dP_signal'] / acc['dP_n'],
            'share_of_later_T_error_in_cooled_cells_pct': 100 * acc['persist_aff'] / acc['persist_T_later'],
            'ModelA_share_of_later_T_error_in_cooled_cells_pct': 100 * acc['aff_err'] / acc['T_err_later'],
        },
        'wells': {
            'Tprod_MAE_K': err[:, :, 0].mean(), 'Tprod_worst_K': err[:, :, 0].max(),
            'Pinj_MAE_MPa': err[:, :, 1].mean(), 'Pinj_worst_MPa': err[:, :, 1].max(),
            'Pprod_MAE_MPa': err[:, :, 2].mean(), 'Pprod_worst_MPa': err[:, :, 2].max(),
            'Pprod_MAE_excl_real_01051_MPa': np.delete(err[:, :, 2], i_out, 0).mean(),
            'dP_MAE_MPa': np.abs(dp_pred - dp_true).mean(),
            'cooling_40yr_true_median_K': np.median(cool_true),
            'cooling_40yr_MAE_K': np.abs(cool_pred - cool_true).mean(),
            'thermal_power_MAE_MW': np.abs(q_pred - q_true).mean(),
            'thermal_power_error_pct': 100 * np.abs(q_pred - q_true).sum() / q_true.sum(),
        },
        'physics_residuals_mean': {k: {'mass': float(np.mean(v[0])), 'energy': float(np.mean(v[1]))}
                                   for k, v in residuals.items()},
    }


def indicators(W, inj_rate, inj_temp, t_years):
    """Thermal and pressure indicators for well curves W of shape (N, 201, >=3)."""
    T0, T40 = W[:, 0, 0], W[:, -1, 0]
    mw = C.thermal_power_mw(W[:, :, 0], inj_rate, inj_temp)
    return {'decline_pct': 100 * (T0 - T40) / T0, 'decline_K': T0 - T40,
            'decline_usable_pct': 100 * (T0 - T40) / (T0 - inj_temp),
            'capacity_y40_MW': mw[:, -1],
            'cum_energy_GJ': (mw[:, :-1] * np.diff(t_years)[None] * 31557.6).sum(1),
            'dP_y40_MPa': W[:, -1, 1] - W[:, -1, 2]}


def ensemble_section(device):
    """Surrogate vs COMSOL indicators over all 1,535 scenarios."""
    raw = C.load_prepared()
    names = list(raw['parameter_names'])
    X = raw['x_params']
    q, tin = X[:, names.index('inj_rate')], X[:, names.index('inj_temp')]
    t = raw['well_time_years'].astype(float)
    pred = C.cached_wells_model_a(device)
    true = raw['y_wells'][:, :, :3]
    ks, kc = indicators(pred, q, tin, t), indicators(true, q, tin, t)
    is_test = np.zeros(len(true), bool)
    is_test[raw['test_idx']] = True
    out = {}
    for k in kc:
        out[k] = {'COMSOL_P10_P50_P90': np.percentile(kc[k], [10, 50, 90]).tolist(),
                  'surrogate_P10_P50_P90': np.percentile(ks[k], [10, 50, 90]).tolist(),
                  'COMSOL_max': float(kc[k].max()), 'surrogate_max': float(ks[k].max()),
                  'per_scenario_MAE_all': float(np.abs(ks[k] - kc[k]).mean()),
                  'per_scenario_MAE_test': float(np.abs(ks[k] - kc[k])[is_test].mean())}
    d = ks['decline_pct']
    out['surrogate_decline_classes'] = {'below_2': int((d < 2).sum()), '2_to_4': int(((d >= 2) & (d < 4)).sum()),
                                        '4_to_6': int(((d >= 4) & (d < 6)).sum()), '6_or_more': int((d >= 6).sum())}
    out['screening'] = {'decline_over_10pct': {'COMSOL': int((kc['decline_pct'] > 10).sum()),
                                               'surrogate': int((d > 10).sum())}}
    for thr in (10, 20):
        c_ex, s_ex = kc['decline_usable_pct'] > thr, ks['decline_usable_pct'] > thr
        out['screening'][f'usable_decline_over_{thr}pct'] = {'COMSOL': int(c_ex.sum()), 'surrogate': int(s_ex.sum()),
                                                              'misclassified': int((c_ex != s_ex).sum())}
    neg = np.where(d < 0)[0]
    out['apparent_warming'] = {'surrogate_count': int(len(neg)), 'COMSOL_count': int((kc['decline_pct'] < 0).sum()),
                               'surrogate_min_pct': float(d.min()), 'COMSOL_min_pct': float(kc['decline_pct'].min()),
                               'max_endpoint_error_K_in_those': float(np.abs(ks['decline_K'] - kc['decline_K'])[neg].max())}
    out['decline_correlation_with_inputs'] = {
        src: {n: float(np.corrcoef(k['decline_pct'], X[:, names.index(n)])[0, 1])
              for n in names if n not in ('well_radius', 'inj_temp_C')}
        for src, k in (('COMSOL', kc), ('surrogate', ks))}
    return out


def stabilization_section():
    """Time for the simulated well pressures to settle, from the 201-point COMSOL curves."""
    raw = C.load_prepared()
    yw, t = raw['y_wells'], raw['well_time_years'].astype(float)
    out = {}
    for c, name in ((1, 'P_inj'), (2, 'P_prod')):
        total = yw[:, -1, c] - yw[:, 0, c]
        total = np.where(np.abs(total) > 1e-6, total, np.nan)
        reached = {f'{y}yr': float(np.nanmedian((yw[:, int(np.argmin(np.abs(t - y))), c] - yw[:, 0, c]) / total))
                   for y in (0.2, 1.0, 2.0, 5.0)}
        settle = []
        for i in range(yw.shape[0]):
            ok = np.abs(yw[i, :, c] - yw[i, -1, c]) < 0.5
            settle.append(t[next(j for j in range(len(ok)) if ok[j:].all())])
        out[name] = {'median_fraction_of_40yr_change_reached': reached,
                     'years_to_stay_within_0.5MPa_median_P90': np.percentile(settle, [50, 90]).tolist()}
    return out


def mask_erosion_section():
    """Share of valid cells kept after the one-cell erosion used by the physics loss."""
    raw = C.load_prepared()
    H = raw['x_params'][:, list(raw['parameter_names']).index('H')]
    struct = ndimage.generate_binary_structure(3, 1)
    kept = []
    for s in range(len(H)):
        m = raw['mask_T'][s][..., 0] & raw['mask_P'][s][..., 0]
        kept.append(100 * ndimage.binary_erosion(m, structure=struct, border_value=1).sum() / max(m.sum(), 1))
    kept = np.array(kept)
    bins = [100, 150, 200, 250, 301]
    return {'all_median_min_pct': [float(np.median(kept)), float(kept.min())],
            'by_thickness_median_min_pct': {f'{bins[i]}-{bins[i + 1]} m': [float(np.median(kept[(H >= bins[i]) & (H < bins[i + 1])])),
                                                                            float(kept[(H >= bins[i]) & (H < bins[i + 1])].min())]
                                            for i in range(len(bins) - 1)}}


def timing_section(device):
    """Model A forward-pass time per scenario: inputs on the GPU, 5 warm-up batches, 3 timed passes."""
    model = C.load_model('A', device)
    ds = C.ComsolFnoDataset(C.PREPARED_DATA_PATH, split='test')
    out = {'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}
    for bs in (1, 2):
        batches = [(b['x_static'].to(device), b['params'].to(device), b['well_map'].to(device))
                   for b in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=C.collate)]
        reps = []
        with torch.no_grad():
            for xb in batches[:5]:
                model(*xb)
            for _ in range(3):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for xb in batches:
                    model(*xb)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                reps.append((time.perf_counter() - t0) / len(ds) * 1000)
        out[f'batch{bs}_ms_per_scenario_mean_min_max'] = [float(np.mean(reps)), float(min(reps)), float(max(reps))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--skip-timing', action='store_true', help='skip the GPU timing section')
    args = parser.parse_args()
    device = C.get_device()
    result = {'test_set': test_set_section(device)}
    torch.cuda.empty_cache()
    result['ensemble'] = ensemble_section(device)
    result['stabilization'] = stabilization_section()
    result['mask_erosion'] = mask_erosion_section()
    if not args.skip_timing:
        result['timing'] = timing_section(device)
    path = C.out_path('reviewer_metrics.json')
    path.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    print('saved', path)


if __name__ == '__main__':
    main()
