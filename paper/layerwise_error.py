"""Layer-wise mean absolute error of Model A for single test scenarios (Appendix A).

For each depth layer, prints the MAE of temperature (degrees C) and pressure (MPa)
over the valid cells, averaged over all nine snapshots and over years 0, 20 and 40.

Usage: python layerwise_error.py [scenario ...]   (default: real_01456 real_01528)
"""
import argparse
import json

import numpy as np
import torch

import common as C

YEARS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def layer_mae(scenario, model, ds, test_ids, z_m):
    """Per-layer MAE for one test scenario, as {field: [(z, MAE all snapshots, MAE years 0/20/40)]}."""
    item = ds[test_ids.index(scenario)]
    with torch.no_grad():
        out = model(torch.from_numpy(item['x_static'])[None], torch.from_numpy(item['params'])[None],
                    torch.from_numpy(item['well_map'])[None])
    st, sel = ds.stats, [YEARS.index(y) for y in (0, 20, 40)]
    result = {}
    for key, mask_key in (('T_K', 'mask_T'), ('p_MPa', 'mask_P')):
        mu, sd = st[f'field__{key}__mean'], st[f'field__{key}__std']
        err = (C.denormalize(out[key][0], mu, sd) - C.denormalize(torch.from_numpy(item[key]), mu, sd)).abs().numpy()
        mask = item[mask_key].astype(bool)
        rows = []
        for zi in range(len(z_m)):
            every, three = err[:, :, zi, :][mask[:, :, zi, :]], err[:, :, zi, sel][mask[:, :, zi, sel]]
            if every.size:
                rows.append((round(float(z_m[zi])), float(every.mean()), float(three.mean())))
        result[key] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('scenarios', nargs='*', default=['real_01456', 'real_01528'])
    args = parser.parse_args()
    raw = C.load_prepared()
    model = C.load_model('A', torch.device('cpu'))
    ds = C.ComsolFnoDataset(C.PREPARED_DATA_PATH, split='test')
    test_ids = list(raw['sample_id'][raw['test_idx']])
    result = {s: layer_mae(s, model, ds, test_ids, raw['z_m']) for s in args.scenarios}
    for s, fields in result.items():
        for key, rows in fields.items():
            print(s, key, '(z m, MAE all snapshots, MAE years 0/20/40)')
            for z, a, b in rows:
                print(f'   {z:6d}  {a:.3f}  {b:.3f}')
    path = C.out_path('layerwise_error.json')
    path.write_text(json.dumps(result, indent=1))
    print('saved', path)


if __name__ == '__main__':
    main()
