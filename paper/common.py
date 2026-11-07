"""Shared paths, model loading and helpers for the paper's analyses and figures.

Every script in this folder imports from here so that checkpoints, the prepared
dataset and the output folder are resolved relative to the repository rather
than to the current working directory.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / 'src' / 'model_a'
PINN_DIR = ROOT / 'src' / 'model_b'
METRICS_DIR = ROOT / 'results' / 'metrics'
OUT_DIR = ROOT / 'results' / 'figures'
sys.path.insert(0, str(PINN_DIR))
sys.path.insert(0, str(MODEL_DIR))

from comsol_fno_model import ComsolFnoModel  # noqa: E402
from comsol_fno_train import (ComsolFnoDataset, collate, denormalize,  # noqa: E402
                              PREPARED_DATA_PATH, CHECKPOINT_DIR)

CHECKPOINTS = {
    'A': Path(CHECKPOINT_DIR) / 'best.pth',
    'B': ROOT / 'checkpoints' / 'model_b' / 'best.pth',
}
MODEL_KWARGS = dict(width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
                    unet_base_channels=4, n_layers=4)
CP_WATER = 4184.0          # J/(kg K), same constant as uq_analysis_mc_v2.py
ATM_MPA = 0.101325         # COMSOL exports gauge pressure relative to 1 atm
SPLITS = ('train', 'val', 'test')
WELL_CHANNELS = ('T_prod_K', 'P_inj_MPa', 'P_prod_MPa')


def out_path(name):
    """Returns where to save `name`: images in results/figures, everything else in results/metrics."""
    folder = OUT_DIR if name.endswith('.png') else METRICS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def get_device():
    """CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(which, device):
    """Loads Model A or Model B ('A' or 'B') from its best checkpoint, in eval mode."""
    model = ComsolFnoModel(**MODEL_KWARGS).to(device)
    ckpt = torch.load(CHECKPOINTS[which], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


class _CachedNpz:
    """Read-once view of the prepared .npz: each array is decompressed on first access
    and kept, because indexing an NpzFile re-reads the whole array every time."""

    def __init__(self, path):
        self._npz = np.load(path, allow_pickle=True)
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._npz[key]
        return self._cache[key]


_PREPARED = None


def load_prepared():
    """The prepared dataset (.npz), shared by every caller in the process."""
    global _PREPARED
    if _PREPARED is None:
        _PREPARED = _CachedNpz(PREPARED_DATA_PATH)
    return _PREPARED


def well_mean_std(stats):
    """Per-channel (mean, std) pairs used to denormalize the three well outputs."""
    return [(stats[f'well__{c}__mean'], stats[f'well__{c}__std']) for c in WELL_CHANNELS]


def denorm_wells(wells, stats):
    """Denormalizes a (B, 201, 3) well tensor to physical units (K, MPa, MPa)."""
    ms = well_mean_std(stats)
    return torch.stack([denormalize(wells[:, :, c], *ms[c]) for c in range(3)], dim=-1)


def predict_wells_all(model, device, batch_size=2):
    """Predicts the well outputs of every scenario with `model`.

    Returns a (N, 201, 3) array in the dataset's original scenario order
    (the order of `sample_id` in the prepared .npz), in physical units.
    """
    raw = load_prepared()
    pred = np.zeros((len(raw['sample_id']), 201, 3), np.float32)
    with torch.no_grad():
        for split in SPLITS:
            ds = ComsolFnoDataset(PREPARED_DATA_PATH, split=split)
            out = []
            for b in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate):
                o = model(b['x_static'].to(device), b['params'].to(device), b['well_map'].to(device))
                out.append(denorm_wells(o['wells'], ds.stats).cpu().numpy())
            pred[raw[f'{split}_idx']] = np.concatenate(out)
            del ds
    return pred


def cached_wells_model_a(device=None):
    """Model A well predictions for all scenarios, cached in results/metrics/ after the first call."""
    path = out_path('all_wells_modelA.npy')
    if path.exists():
        return np.load(path)
    pred = predict_wells_all(load_model('A', device or get_device()), device or get_device())
    np.save(path, pred)
    return pred


def psat_mpa(T_K):
    """Saturation pressure of water (MPa) at temperature T_K, IAPWS-IF97 region 4."""
    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5,
         -0.32325550322333e7, 0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6,
         -0.23855557567849, 0.65017534844798e3]
    T = np.asarray(T_K, dtype=np.float64)
    th = T + n[8] / (T - n[9])
    A = th ** 2 + n[0] * th + n[1]
    B = n[2] * th ** 2 + n[3] * th + n[4]
    C = n[5] * th ** 2 + n[6] * th + n[7]
    return (2 * C / (-B + np.sqrt(B ** 2 - 4 * A * C))) ** 4


def thermal_power_mw(T_prod_K, inj_rate, inj_temp_K):
    """Reservoir-side gross thermal power (MW) over time: q cp max(T_prod - T_inj, 0)."""
    return inj_rate[:, None] * CP_WATER * np.maximum(T_prod_K - inj_temp_K[:, None], 0) / 1e6
