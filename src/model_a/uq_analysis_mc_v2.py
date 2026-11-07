"""
uq_analysis_mc_v2.py
────────────────────────────────────────────────────────────────────────────
Monte Carlo (MC) Forward Propagation UQ Engine — Version 2
COMSOL/FNO 3D Geothermal Surrogate (ComsolFnoModel)

════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT IS (AND IS NOT)
════════════════════════════════════════════════════════════════════════════
THIS IS:   A pure UQ engine. It propagates operational and geological input
           uncertainty through the trained surrogate to build empirical
           output probability distributions for T_prod, BHP_inj, and dP.

THIS IS NOT:  A model accuracy checker. Relative Error (RE%) measures model
              performance vs. ground truth — that is the job of
              evaluate_test_set.py. RE% does NOT appear here.

════════════════════════════════════════════════════════════════════════════
METHODOLOGY
════════════════════════════════════════════════════════════════════════════
Reference approach: Xue et al., Nature Communications 2026,
  "A robust physics-constrained neural operator framework for efficient
   geothermal resource development" (DOI: 10.1038/s41467-026-73183-0)

OUR IMPLEMENTATION (truthfully described):
  - Dataset:  N = 1,535 LHS (Latin Hypercube Sampling) scenarios from
              fno_lhs_v1_prepared_phase1.npz, spanning a 14-dimensional
              operational x geological parameter space.
  - Sampling: LHS -- a space-filling design that guarantees uniform
              marginal coverage of every dimension. Superior to independent
              uniform random for high-dimensional parameter spaces.
  - NOT the reference paper's method: the paper uses independent uniform
              random sampling, ~1,000 scenarios per pre-designed reservoir
              configuration (2,000 total). We have one continuous LHS space.
  - Reservoir grouping: K-Means clustering on 10 geological parameters
              produces K=3 emergent reservoir archetypes (A, B, C). Operational
              parameters still vary within each cluster.

STAGE 1:  Schema Discovery and KPI Derivability Check
STAGE 2:  Geological Reservoir Clustering (K-Means, K=3)
STAGE 3:  Monte Carlo Forward Ensemble (all 1,535 scenarios through trained model)
STAGE 4:  Visualization Suite (5+ figures mirroring Figs 4 and 5 of ref paper)
STAGE 5:  Markdown Report (no RE%, includes methodology and cluster tables)

════════════════════════════════════════════════════════════════════════════
OUTPUTS  ->  ./uq_outputs/
════════════════════════════════════════════════════════════════════════════
  strategy_field_slices.png            (Ref Fig. 4 style)
  mc_percentile_fan_chart.png          (Pooled ensemble, 2x2 panel)
  mc_percentile_fan_chart_by_reservoir.png  (Per-cluster, 3x3 panel)
  kpi_distributions_boxplots.png       (Ref Fig. 5a-d style)
  production_temp_decline_dual_axis.png  (Ref Fig. 5e/g style)
  stratified_capacity_kde.png          (Ref Fig. 5i-l style)
  uq_summary_report.md
  uq_results_v2.json
"""

# ─────────────────────────────────────────────────────────────────────────────
# HEADLESS DISPLAY GUARD -- must precede any matplotlib import
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import random
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Read-only imports from existing codebase ──────────────────────────────────
try:
    from comsol_fno_model import ComsolFnoModel
    from comsol_fno_train import (
        ComsolFnoDataset,
        collate,
        denormalize,
        UNIFORM_TIME_YEARS,  # 9 field snapshot years: [0,5,10,15,20,25,30,35,40]
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Cannot import from codebase: {e}\n"
        "Run from the same directory as comsol_fno_model.py and comsol_fno_train.py."
    )

# ── Headless matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available -- all figures will be skipped.")

# ── SciPy (for KDE) ──────────────────────────────────────────────────────────
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available -- KDE figure will use histograms.")

# ── Scikit-learn (for K-Means clustering) ────────────────────────────────────
# Required: without K-Means the scenario clusters would not match the reported results.
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DATA_FILENAME = "fno_lhs_v1_prepared_phase1.npz"
_CKPT_FILENAME = "best.pth"

_DATA_SEARCH_DIRS = [os.path.join(_REPO_ROOT, "data"), os.getcwd()]
_CKPT_SEARCH_DIRS = [os.path.join(_REPO_ROOT, "checkpoints", "model_a")]


def _find_file(filename, search_dirs):
    """Searches `search_dirs` in order for a file named `filename` and
    returns the first matching path found, or None if it isn't present in
    any of them."""
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


PREPARED_DATA_PATH = (
    os.environ.get("GEOTHERMAL_DATA_PATH")
    or _find_file(_DATA_FILENAME, _DATA_SEARCH_DIRS)
    or os.path.join(_REPO_ROOT, "data", _DATA_FILENAME)
)
BEST_CKPT_PATH = (
    _find_file(_CKPT_FILENAME, _CKPT_SEARCH_DIRS)
    or os.path.join(_REPO_ROOT, "checkpoints", "model_a", _CKPT_FILENAME)
)
UQ_OUT_DIR = os.path.join(_REPO_ROOT, "results", "uq")

# ── Model architecture kwargs (must match trained checkpoint exactly) ─────────
MODEL_KWARGS = dict(
    width=8, width2=32,
    modes1=4, modes2=4, modes3=4, modes4=3,
    unet_base_channels=4, n_layers=4,
)

# ── Inference constants ───────────────────────────────────────────────────────
ALL_BATCH_SIZE   = 16    # full-ensemble batches (~4-16 GB RAM on ARC)
FIELD_BATCH_SIZE = 6     # for spatial field collection (6 representative scenarios)
K_CLUSTERS       = 3     # geological reservoir archetypes: A, B, C
_GLOBAL_SEED     = 0


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER CLASSIFICATION
# Names match data["parameter_names"] in the prepared .npz exactly.
# ─────────────────────────────────────────────────────────────────────────────
GEOLOGICAL_PARAM_NAMES = [
    "variance",       # Permeability field log-normal variance
    "corr_x",         # Correlation length, X-direction [m]
    "corr_y",         # Correlation length, Y-direction [m]
    "corr_z",         # Correlation length, Z-direction [m]
    "nugget",         # Nugget effect (uncorrelated variance fraction)
    "phi0",           # Base porosity
    "c_compaction",   # Compressibility coefficient
    "grad_C_per_km",  # Geothermal gradient [deg C/km]
    "cp_rock",        # Rock specific heat capacity [J/(kg*K)]
    "H",              # Reservoir thickness [m]
]
OPERATIONAL_PARAM_NAMES = [
    "inj_rate",       # Injection flow rate [kg/s]
    "inj_temp",       # Injection temperature [K]
    "well_spacing",   # Injector-producer distance [m]
    # "inj_temp_C" intentionally excluded -- redundant with inj_temp (same value in deg C)
]
CONSTANT_PARAM_NAMES = ["well_radius"]  # constant 0.085m across all 1,535 scenarios

# ── Cluster display configuration ─────────────────────────────────────────────
CLUSTER_COLORS  = {0: "#4361ee", 1: "#f72585", 2: "#7209b7"}
CLUSTER_LABELS  = {0: "Scenario cluster A", 1: "Scenario cluster B", 2: "Scenario cluster C"}
CLUSTER_MARKERS = {0: "o", 1: "s", 2: "^"}


# ─────────────────────────────────────────────────────────────────────────────
# VARIABLE ABBREVIATION DICTIONARY
# ALL plot labels, console output, and report text must resolve through this.
# ─────────────────────────────────────────────────────────────────────────────
ABBREV_DICT = {
    "T_prod_K":      {"full": "Production Well Temperature",                       "unit": "K"},
    "P_inj_MPa":     {"full": "Injection-Well Pressure",                          "unit": "MPa"},
    "P_prod_MPa":    {"full": "Production-Well Pressure",
                      "unit": "MPa"},
    "dP_MPa":        {"full": "Differential Pressure (Injection minus Production)", "unit": "MPa"},
    "T_K":           {"full": "Subsurface Temperature Field",                      "unit": "K"},
    "p_MPa":         {"full": "Subsurface Pressure Field",                         "unit": "MPa"},
    "MW_th":         {"full": "Thermal Power Generation Capacity",                 "unit": "MWth"},
    "Cum_Energy":    {"full": "Cumulative Thermal Energy Extracted",               "unit": "GJ"},
    "Delta_T_pct":   {"full": "Production Temperature Decline",                   "unit": "%"},
    "P10":           {"full": "10th Percentile -- Lower Bound",                    "unit": ""},
    "P25":           {"full": "25th Percentile -- IQR Lower Bound",               "unit": ""},
    "P50":           {"full": "50th Percentile -- Median Forecast",               "unit": ""},
    "P75":           {"full": "75th Percentile -- IQR Upper Bound",               "unit": ""},
    "P90":           {"full": "90th Percentile -- Upper Bound",                   "unit": ""},
    "LHS":           {"full": "Latin Hypercube Sampling",                         "unit": ""},
    "MC":            {"full": "Monte Carlo Forward Propagation",                  "unit": ""},
    "inj_rate":      {"full": "Injection Flow Rate",                              "unit": "kg/s"},
    "inj_temp":      {"full": "Injection Temperature",                            "unit": "K"},
    "well_spacing":  {"full": "Injector-Producer Well Spacing",                   "unit": "m"},
    "H":             {"full": "Reservoir Thickness",                               "unit": "m"},
    "grad_C_per_km": {"full": "Geothermal Gradient",                             "unit": "deg C/km"},
}


def get_label(key: str, mode: str = "full_unit") -> str:
    """Resolve an abbreviation key to a formatted descriptive label.

    Modes:
      'full_unit' -> 'Production Well Temperature [K]'
      'full'      -> 'Production Well Temperature'
      'unit_only' -> '[K]'
    """
    entry = ABBREV_DICT.get(key, {"full": key, "unit": ""})
    full = entry["full"]
    unit = entry.get("unit", "")
    if mode == "full_unit":
        return f"{full} [{unit}]" if unit else full
    if mode == "unit_only":
        return f"[{unit}]" if unit else ""
    return full   # 'full' or 'title'


def get_color(key: str, fallback: str = "#6366f1") -> str:
    """Return a canonical plotting colour for a channel key."""
    COLOR_MAP = {
        "T_prod_K":    "#e63946",
        "P_inj_MPa":  "#457b9d",
        "P_prod_MPa": "#a8dadc",
        "dP_MPa":     "#f4a261",
        "T_K":        "#e63946",
        "p_MPa":      "#457b9d",
        "MW_th":      "#2a9d8f",
        "Cum_Energy": "#e9c46a",
        "Delta_T_pct": "#e76f51",
    }
    return COLOR_MAP.get(key, fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_console_column_legend():
    """Print a self-contained ASCII legend for all variable abbreviations."""
    print("\n" + "=" * 80)
    print("  VARIABLE ABBREVIATION DICTIONARY")
    print("  " + "-" * 76)
    for key, info in ABBREV_DICT.items():
        unit_str = f" [{info['unit']}]" if info.get("unit") else ""
        print(f"  * {key:<16} ->  {info['full']}{unit_str}")
    print("=" * 80 + "\n")


def _section_banner(title: str):
    """Prints `title` as a bannered console section header (a dashed rule
    above and below the text) to mark the start of a new pipeline stage."""
    bar = "-" * 72
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{bar}", flush=True)


def _save_fig(fig, fname: str):
    """Saves a matplotlib Figure `fig` to `fname` inside UQ_OUT_DIR at
    300 dpi with a tight bounding box, closes it to free memory, and
    prints the path it was saved to."""
    path = os.path.join(UQ_OUT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight validation
# ─────────────────────────────────────────────────────────────────────────────
def _validate_paths():
    """Checks that the prepared dataset .npz and the model checkpoint both
    exist at their resolved paths (PREPARED_DATA_PATH, BEST_CKPT_PATH),
    printing a description of any that's missing and exiting the process
    if either is absent."""
    errors = []
    if not os.path.isfile(PREPARED_DATA_PATH):
        errors.append(
            f"  MISSING DATASET: {PREPARED_DATA_PATH}\n"
            f"  Expected: {_DATA_FILENAME} (~1.86 GB) in script directory."
        )
    if not os.path.isfile(BEST_CKPT_PATH):
        errors.append(
            f"  MISSING CHECKPOINT: {BEST_CKPT_PATH}\n"
            f"  Expected: checkpoints/best.pth or best.pth alongside this script."
        )
    if errors:
        print("\n[PRE-FLIGHT FAILURE] Required files are missing:\n")
        for msg in errors:
            print(msg)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1A: Dataset Class -- all 1,535 LHS scenarios
# ─────────────────────────────────────────────────────────────────────────────
class _ComsolFnoDatasetAll(Dataset):
    """All 1,535 LHS scenarios for the MC forward ensemble.

    Critical design note:
      self.x_params_raw  -- unnormalised physical-unit parameters (N, 14)
                            Used for KPI computation (inj_rate, inj_temp).
      __getitem__ returns normalised params for model inference.

    The original _ComsolFnoDatasetAll from v1 is reproduced here so this
    script remains self-contained and does not depend on v1.
    """

    def __init__(self, npz_path: str):
        """Loads every scenario from the prepared dataset .npz at
        `npz_path` (no train/val/test split filtering), keeping both the
        raw physical-unit parameters (used for KPI computation) and the
        static geology fields, well outputs, field validity masks, and
        normalization stats needed for model inference. Drops the constant
        'well_radius' parameter column."""
        data    = np.load(npz_path, allow_pickle=True)
        n_total = data["x_static_fields"].shape[0]
        idx     = np.arange(n_total)
        self.n  = len(idx)

        field_time_years = list(data["field_time_years"])
        self.time_idx    = [field_time_years.index(y) for y in UNIFORM_TIME_YEARS]

        self.x_static_fields      = data["x_static_fields"][idx]
        self.static_channel_names = list(data["static_channel_names"])

        param_names         = list(data["parameter_names"])
        self.param_keep_idx = [i for i, n in enumerate(param_names) if n != "well_radius"]
        self.param_names    = [param_names[i] for i in self.param_keep_idx]

        # Raw (un-normalised) parameters in physical units -- used for KPI computation
        self.x_params_raw = data["x_params"][idx][:, self.param_keep_idx].astype(np.float32)

        self.T_K     = data["T_K"][idx]
        self.p_MPa   = data["p_MPa"][idx]
        self.mask_T  = data["mask_T"][idx]
        self.mask_P  = data["mask_P"][idx]
        self.y_wells = data["y_wells"][idx]

        stats      = dict(zip(data["stats_keys"], data["stats_values"]))
        self.stats = {k: float(v) for k, v in stats.items()}

        self.well_map_idx    = self.static_channel_names.index("well_map")
        self.sample_id       = data["sample_id"][idx]
        self.well_time_years = data["well_time_years"]

    def __len__(self) -> int:
        """Returns the number of LHS scenarios in the ensemble (N = 1,535)."""
        return self.n

    def _norm_static(self, arr):
        """Z-score normalises one scenario's static geology/well-location
        field channels using the per-channel mean/std stored in `self.stats`.

        Args:
          arr: (C, X, Y, Z) static field channels in physical units.
        Returns:
          (C, X, Y, Z) array normalised channel-by-channel.
        """
        out = np.empty_like(arr)
        for ci, name in enumerate(self.static_channel_names):
            mean = self.stats[f"static__{name}__mean"]
            std  = self.stats[f"static__{name}__std"]
            out[ci] = (arr[ci] - mean) / (std if std > 1e-6 else 1.0)
        return out

    def _norm_params(self, arr):
        """Z-score normalises one scenario's scalar parameter vector using
        the per-parameter mean/std stored in `self.stats`.

        Args:
          arr: (14,) raw physical-unit parameter values.
        Returns:
          (14,) array normalised parameter-by-parameter.
        """
        out = np.empty_like(arr)
        for i, name in enumerate(self.param_names):
            mean = self.stats[f"param__{name}__mean"]
            std  = self.stats[f"param__{name}__std"]
            out[i] = (arr[i] - mean) / (std if std > 1e-6 else 1.0)
        return out

    def __getitem__(self, idx):
        """Assembles and normalises one scenario's static fields, scalar
        parameters, well-location map, target T_K/p_MPa fields, field
        validity masks, and well outputs for model inference.

        Args:
          idx: Scenario index into the loaded ensemble.
        Returns:
          dict with keys x_static, params, well_map, T_K, p_MPa, mask_T,
          mask_P, wells -- all normalised (except well_map and masks) and
          cast to float32.
        """
        x_static = self._norm_static(self.x_static_fields[idx]).astype(np.float32)
        params   = self._norm_params(self.x_params_raw[idx]).astype(np.float32)
        well_map = self.x_static_fields[idx, self.well_map_idx].astype(np.float32)
        T_K      = self.T_K[idx][..., self.time_idx]
        p_MPa_f  = self.p_MPa[idx][..., self.time_idx]
        mask_T   = self.mask_T[idx][..., self.time_idx]
        mask_P   = self.mask_P[idx][..., self.time_idx]
        wells    = self.y_wells[idx, :, :3].astype(np.float32)

        t_mean, t_std = self.stats["field__T_K__mean"],   self.stats["field__T_K__std"]
        p_mean, p_std = self.stats["field__p_MPa__mean"], self.stats["field__p_MPa__std"]
        wn = [
            (self.stats["well__T_prod_K__mean"],  self.stats["well__T_prod_K__std"]),
            (self.stats["well__P_inj_MPa__mean"],  self.stats["well__P_inj_MPa__std"]),
            (self.stats["well__P_prod_MPa__mean"], self.stats["well__P_prod_MPa__std"]),
        ]
        T_K_norm   = ((T_K     - t_mean) / t_std).astype(np.float32)
        p_MPa_norm = ((p_MPa_f - p_mean) / p_std).astype(np.float32)
        wells_norm = np.stack(
            [(wells[:, c] - m) / s for c, (m, s) in enumerate(wn)], axis=1
        ).astype(np.float32)

        return {
            "x_static": x_static,
            "params":   params,
            "well_map": well_map,
            "T_K":      T_K_norm,
            "p_MPa":    p_MPa_norm,
            "mask_T":   mask_T.astype(np.float32),
            "mask_P":   mask_P.astype(np.float32),
            "wells":    wells_norm,
        }


class _IndexedSubset(Dataset):
    """Index-based subset of _ComsolFnoDatasetAll for field collection."""

    def __init__(self, full_ds: _ComsolFnoDatasetAll, indices):
        """Wraps `full_ds` to expose only the scenarios at `indices`.

        Args:
          full_ds: The full 1,535-scenario _ComsolFnoDatasetAll.
          indices: Sequence of integer indices into `full_ds` to expose.
        """
        self.full_ds = full_ds
        self.indices = np.asarray(indices, dtype=int)
        self.stats   = full_ds.stats

    def __len__(self) -> int:
        """Returns the number of scenarios in this subset."""
        return len(self.indices)

    def __getitem__(self, i: int):
        """Returns the `i`-th scenario of the subset, mapped through
        `self.indices` to the corresponding item in `full_ds`."""
        return self.full_ds[int(self.indices[i])]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1B: Schema Inspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_parameters(param_names, x_params_raw):
    """Print parameter classification table and physical range summary."""
    print("\n  PARAMETER CLASSIFICATION TABLE:")
    print(f"  {'#':<4} {'Name':<18} {'Type':<14} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print(f"  {'='*68}")
    for i, name in enumerate(param_names):
        col = x_params_raw[:, i]
        if name in GEOLOGICAL_PARAM_NAMES:
            ptype = "GEOLOGICAL  "
        elif name in OPERATIONAL_PARAM_NAMES:
            ptype = "OPERATIONAL "
        elif name == "inj_temp_C":
            ptype = "REDUNDANT   "
        else:
            ptype = "OTHER       "
        print(f"  {i:<4} {name:<18} {ptype:<14} "
              f"{col.min():>10.3g} {col.max():>10.3g} {col.mean():>10.3g}")


def verify_bhp_prod_status(pred_wells_sample):
    """Check whether P_prod_MPa behaves as a quasi-constant control setting.

    Computes std(P_prod at Year 40) / std(P_inj at Year 40).
    Ratio < 0.4 indicates P_prod is quasi-controlled (low inter-scenario variability).

    Returns: (is_quasi_controlled: bool, ratio: float)
    """
    std_p_inj  = float(np.std(pred_wells_sample[:, -1, 1]))  # P_inj at Year 40
    std_p_prod = float(np.std(pred_wells_sample[:, -1, 2]))  # P_prod at Year 40
    ratio = std_p_prod / (std_p_inj + 1e-8)
    is_quasi = ratio < 0.4
    print(f"\n  BHP_prod (P_prod_MPa) variability check:")
    print(f"    std(P_inj  @ Year 40) = {std_p_inj:.4f} MPa")
    print(f"    std(P_prod @ Year 40) = {std_p_prod:.4f} MPa")
    print(f"    Variability ratio     = {ratio:.3f}")
    if is_quasi:
        print("    -> P_prod is quasi-controlled: variability ratio < 0.4.")
        print("       Figures label it as approximately tracking the input control setting.")
    else:
        print("    -> P_prod shows genuine variability (ratio >= 0.4): full model output.")
    return is_quasi, ratio


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Geological Clustering
# ─────────────────────────────────────────────────────────────────────────────
def cluster_geological_groups(x_params_raw, param_names, k=3):
    """Partition scenarios into K geological reservoir archetypes.

    Strategy: K-Means on the 10 geological parameters (StandardScaler-normalised).

    Clusters are relabelled so that 0 = lowest median H (thinnest reservoir),
    1 = medium, 2 = highest (thickest reservoir).

    Returns:
      cluster_labels: np.ndarray (N,) with values in {0, ..., k-1}
      method: str description of clustering method used
    """
    geo_indices = [i for i, n in enumerate(param_names) if n in GEOLOGICAL_PARAM_NAMES]
    if not geo_indices:
        warnings.warn("No geological parameters found -- using single cluster.")
        return np.zeros(len(x_params_raw), dtype=int), "single_cluster"

    G = x_params_raw[:, geo_indices]   # (N, n_geo)

    scaler   = StandardScaler()
    G_scaled = scaler.fit_transform(G)
    kmeans   = KMeans(n_clusters=k, n_init=20, random_state=_GLOBAL_SEED)
    labels   = kmeans.fit_predict(G_scaled)
    print(f"\n  K-Means clustering: K={k}, n_geo_params={len(geo_indices)}, "
          f"inertia={kmeans.inertia_:.2f}")
    method = f"KMeans(K={k}, n_geo_params={len(geo_indices)}, n_init=20, seed={_GLOBAL_SEED})"

    # Remap so cluster 0 = lowest median H
    h_col = next((i for i, n in enumerate(param_names) if n == "H"), None)
    if h_col is not None:
        med_H  = [np.median(x_params_raw[labels == c, h_col]) for c in range(k)]
        remap  = {int(old): int(new) for new, old in enumerate(np.argsort(med_H))}
        labels = np.array([remap[l] for l in labels], dtype=int)

    return labels, method


def print_cluster_characterization(cluster_labels, x_params_raw, param_names, k=3):
    """Print a characterisation table for each geological cluster."""

    def _get(name):
        """Looks up the column of `x_params_raw` for parameter `name`.

        Returns the (N,) array of raw physical-unit values, or None if
        `name` is not present in `param_names`.
        """
        idx = next((i for i, n in enumerate(param_names) if n == name), None)
        return x_params_raw[:, idx] if idx is not None else None

    H_arr   = _get("H")
    gr_arr  = _get("grad_C_per_km")
    q_arr   = _get("inj_rate")
    ws_arr  = _get("well_spacing")

    print("\n  GEOLOGICAL RESERVOIR ARCHETYPES:")
    hdr = (f"  {'Cluster':<14} {'n':>5}  {'H_median [m]':>13}  "
           f"{'H_range [m]':>15}  {'Grad [dC/km]':>13}  "
           f"{'inj_rate [kg/s]':>16}  {'spacing [m]':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for c in range(k):
        mask = cluster_labels == c
        n_c  = mask.sum()

        def _med(arr):
            """Formats the median of `arr` restricted to cluster `c` as a
            1-decimal string, or "N/A" if `arr` is None or the cluster is
            empty."""
            if arr is None or n_c == 0:
                return "N/A"
            return f"{np.median(arr[mask]):.1f}"

        def _rng(arr):
            """Formats the min-max range of `arr` restricted to cluster `c`
            as a "min-max" string, or "N/A" if `arr` is None or the cluster
            is empty."""
            if arr is None or n_c == 0:
                return "N/A"
            return f"{arr[mask].min():.0f}-{arr[mask].max():.0f}"

        print(f"  {CLUSTER_LABELS[c]:<14} {n_c:>5}  "
              f"{_med(H_arr):>13}  {_rng(H_arr):>15}  "
              f"{_med(gr_arr):>13}  {_med(q_arr):>16}  {_med(ws_arr):>12}")

    print("\n  NOTE: Operational params per cluster reflect LHS co-variation, not")
    print("  controlled isolation of geological vs. operational effects.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3A: MC Forward Pass
# ─────────────────────────────────────────────────────────────────────────────
def run_mc_forward_pass(model, dataset, device, batch_size=16, collect_fields=False):
    """Run all scenarios in dataset through model; return physical-unit predictions.

    Args:
      model:          Trained ComsolFnoModel in eval() mode.
      dataset:        _ComsolFnoDatasetAll or _IndexedSubset.
      collect_fields: If True, also collect 3D spatial field predictions.

    Returns dict:
      pred_wells     (N, 201, 3)   T_prod_K, P_inj_MPa, P_prod_MPa [physical units]
      pred_wells_dP  (N, 201)      dP_MPa (derived)
      pred_T_field   (N, X, Y, Z, T_field)  -- only if collect_fields=True
      pred_P_field   (N, X, Y, Z, T_field)  -- only if collect_fields=True
      mask_T         (N, X, Y, Z, T_field)  -- validity mask
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    stats  = dataset.stats

    t_mean, t_std = stats["field__T_K__mean"],    stats["field__T_K__std"]
    p_mean, p_std = stats["field__p_MPa__mean"],  stats["field__p_MPa__std"]
    wms = [
        (stats["well__T_prod_K__mean"],  stats["well__T_prod_K__std"]),
        (stats["well__P_inj_MPa__mean"],  stats["well__P_inj_MPa__std"]),
        (stats["well__P_prod_MPa__mean"], stats["well__P_prod_MPa__std"]),
    ]

    pred_w_list, pred_T_list, pred_P_list, mask_T_list = [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_static = batch["x_static"].to(device)
            params   = batch["params"].to(device)
            well_map = batch["well_map"].to(device)

            out = model(x_static, params, well_map)

            pw = torch.stack(
                [denormalize(out["wells"][:, :, c], *wms[c]) for c in range(3)], dim=-1
            )
            pred_w_list.append(pw.cpu().numpy())

            if collect_fields:
                pred_T_list.append(denormalize(out["T_K"],   t_mean, t_std).cpu().numpy())
                pred_P_list.append(denormalize(out["p_MPa"], p_mean, p_std).cpu().numpy())
                mask_T_list.append(batch["mask_T"].numpy())

    pred_wells = np.concatenate(pred_w_list, axis=0)   # (N, 201, 3)
    result = {
        "pred_wells":    pred_wells,
        "pred_wells_dP": pred_wells[:, :, 1] - pred_wells[:, :, 2],
    }
    if collect_fields and pred_T_list:
        result["pred_T_field"] = np.concatenate(pred_T_list, axis=0)
        result["pred_P_field"] = np.concatenate(pred_P_list, axis=0)
        result["mask_T"]       = np.concatenate(mask_T_list, axis=0).astype(bool)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3B: Percentile Band Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_mc_percentile_bands(pred_wells, well_time_years):
    """Compute P10/P25/P50/P75/P90 percentile bands across the ensemble.

    Args:
      pred_wells: (N, 201, 3) -- T_prod_K, P_inj_MPa, P_prod_MPa
    Returns:
      dict with keys T_prod_K, P_inj_MPa, P_prod_MPa, dP_MPa
      each: {p10, p25, p50, p75, p90, mean, iqr_width, spread_width}
    """
    dP       = pred_wells[:, :, 1] - pred_wells[:, :, 2]
    ch_names = ["T_prod_K", "P_inj_MPa", "P_prod_MPa", "dP_MPa"]
    ensemble = np.concatenate([pred_wells, dP[:, :, np.newaxis]], axis=2)  # (N,201,4)

    bands = {}
    for c, name in enumerate(ch_names):
        s   = ensemble[:, :, c]
        p10 = np.percentile(s, 10, axis=0)
        p25 = np.percentile(s, 25, axis=0)
        p50 = np.percentile(s, 50, axis=0)
        p75 = np.percentile(s, 75, axis=0)
        p90 = np.percentile(s, 90, axis=0)
        mn  = s.mean(axis=0)
        bands[name] = {
            "p10":          p10.tolist(),
            "p25":          p25.tolist(),
            "p50":          p50.tolist(),
            "p75":          p75.tolist(),
            "p90":          p90.tolist(),
            "mean":         mn.tolist(),
            "iqr_width":    (p75 - p25).tolist(),
            "spread_width": (p90 - p10).tolist(),
        }
    return bands


def compute_all_bands(pred_wells_full, cluster_labels, well_time_years, k=3):
    """Compute pooled bands + per-cluster bands.

    Returns:
      pooled_bands:       dict  (all N scenarios)
      per_cluster_bands:  list[dict | None] of length k
    """
    pooled_bands = compute_mc_percentile_bands(pred_wells_full, well_time_years)
    per_cluster_bands = []
    for c in range(k):
        mask = cluster_labels == c
        if mask.sum() > 1:
            per_cluster_bands.append(
                compute_mc_percentile_bands(pred_wells_full[mask], well_time_years)
            )
        else:
            per_cluster_bands.append(None)
            warnings.warn(f"Cluster {c} has < 2 scenarios -- band skipped.")
    return pooled_bands, per_cluster_bands


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3C: Derived KPI Computation (BUG FIXED -- uses 'inj_rate' not 'flow_rate')
# ─────────────────────────────────────────────────────────────────────────────
def compute_derived_kpis(pred_wells, x_params_raw, param_names, well_time_years):
    """Compute engineering KPIs from the MC ensemble predictions.

    All parameter lookups use name-matching, not hardcoded indices.

    Formulas:
      Delta_T_pct(i) = (T_prod_i(0) - T_prod_i(end)) / T_prod_i(0) * 100  [%]
      MW_th(i, t)    = inj_rate_i * c_p * max(T_prod_i(t) - T_inj_i, 0) / 1e6  [MW]
      Cum_Energy(i)  = sum_t MW_th(i,t) * dt_t * 31557.6  [GJ]
                       (31557.6 GJ per MW*yr = 1e6 W * 365.25 * 86400 s / 1e9)

    Args:
      pred_wells:      (N, 201, 3)   T_prod_K, P_inj_MPa, P_prod_MPa
      x_params_raw:    (N, n_params) unnormalised physical parameters
      param_names:     list[str]     names matching x_params_raw columns
      well_time_years: (201,)        simulation years

    Returns dict: Delta_T_pct (N,), MW_th (N,), MW_th_ts (N,201), Cum_Energy (N,)
    """
    kpis = {}

    # ── Temperature Decline ────────────────────────────────────────────────────
    T_prod_0   = pred_wells[:, 0,  0]    # Production temp at t=0   (K)
    T_prod_end = pred_wells[:, -1, 0]    # Production temp at t=40  (K)
    kpis["Delta_T_pct"] = (
        (T_prod_0 - T_prod_end) / np.maximum(T_prod_0, 1e-6) * 100.0
    )
    print(f"  Delta_T_pct: range [{kpis['Delta_T_pct'].min():.2f}, "
          f"{kpis['Delta_T_pct'].max():.2f}] %")

    # ── Thermal Power and Cumulative Energy ────────────────────────────────────
    # Name-based index lookup -- robust against parameter list reordering
    q_idx    = next((i for i, n in enumerate(param_names) if n == "inj_rate"),   None)
    tinj_idx = next((i for i, n in enumerate(param_names) if n == "inj_temp"),   None)

    if q_idx is not None and tinj_idx is not None:
        Q_kg_s  = x_params_raw[:, q_idx]     # [kg/s]  (N,)
        T_inj_K = x_params_raw[:, tinj_idx]  # [K]     (N,)
        c_p     = 4184.0                       # specific heat of water [J/(kg*K)]

        T_prod_ts   = pred_wells[:, :, 0]           # (N, 201) [K]
        T_inj_broad = T_inj_K[:, np.newaxis]        # (N, 1) broadcast
        Q_broad      = Q_kg_s[:, np.newaxis]         # (N, 1) broadcast

        delta_T_pos = np.maximum(T_prod_ts - T_inj_broad, 0.0)  # (N, 201)
        mw_th_ts    = Q_broad * c_p * delta_T_pos / 1e6          # (N, 201) [MW]

        kpis["MW_th_ts"] = mw_th_ts
        kpis["MW_th"]    = mw_th_ts[:, -1]   # Final-year thermal power capacity [MW]

        # Cumulative energy via Riemann left-endpoint sum
        dt_years = np.diff(np.asarray(well_time_years, dtype=np.float64))  # (200,)
        dt_broad = dt_years[np.newaxis, :]                                   # (1, 200)
        kpis["Cum_Energy"] = (mw_th_ts[:, :-1] * dt_broad * 31557.6).sum(axis=1)  # (N,) [GJ]

        print(f"  MW_th:      range [{kpis['MW_th'].min():.3f}, {kpis['MW_th'].max():.3f}] MWth")
        print(f"  Cum_Energy: range [{kpis['Cum_Energy'].min():.0f}, "
              f"{kpis['Cum_Energy'].max():.0f}] GJ")
    else:
        warnings.warn(
            f"Could not locate 'inj_rate' (found: {q_idx}) or "
            f"'inj_temp' (found: {tinj_idx}) in param_names={param_names}. "
            "MW_th and Cum_Energy will NOT be computed."
        )

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3D: Representative Field Collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_representative_fields(model, full_dataset, cluster_labels,
                                  x_params_raw, param_names, device, k=3):
    """Select 2 operationally-contrasting scenarios per cluster; run model with fields.

    Contrast criterion: lowest vs. highest injection rate within each geological cluster.
    This gives "low flow" and "high flow" strategies for the same geological context.

    Returns:
      field_data:    dict with pred_T_field (n, X, Y, Z, T), pred_P_field, mask_T
      scenario_info: list[dict] of length n with cluster/label/operational params
    """
    q_idx  = next((i for i, n in enumerate(param_names) if n == "inj_rate"),    None)
    ws_idx = next((i for i, n in enumerate(param_names) if n == "well_spacing"), None)

    selected_indices = []
    scenario_info    = []

    for c in range(k):
        cluster_mask = np.where(cluster_labels == c)[0]
        if len(cluster_mask) == 0:
            continue

        if q_idx is not None and len(cluster_mask) >= 2:
            inj_rates = x_params_raw[cluster_mask, q_idx]
            lo_local  = int(np.argmin(inj_rates))
            hi_local  = int(np.argmax(inj_rates))
            if lo_local == hi_local:
                hi_local = (lo_local + 1) % len(cluster_mask)
            chosen   = [cluster_mask[lo_local], cluster_mask[hi_local]]
            suffixes = ["Low Flow", "High Flow"]
        else:
            chosen   = [cluster_mask[0], cluster_mask[min(1, len(cluster_mask) - 1)]]
            suffixes = ["Strategy 1", "Strategy 2"]

        for global_idx, suffix in zip(chosen, suffixes):
            selected_indices.append(int(global_idx))
            info = {
                "cluster":       c,
                "cluster_label": CLUSTER_LABELS[c],
                "suffix":        suffix,
                "global_idx":    int(global_idx),
                "sample_id":     str(full_dataset.sample_id[global_idx]),
            }
            if q_idx is not None:
                info["inj_rate_kgs"]  = float(x_params_raw[global_idx, q_idx])
            if ws_idx is not None:
                info["well_spacing_m"] = float(x_params_raw[global_idx, ws_idx])
            scenario_info.append(info)

    subset_ds  = _IndexedSubset(full_dataset, selected_indices)
    field_data = run_mc_forward_pass(model, subset_ds, device,
                                     batch_size=FIELD_BATCH_SIZE, collect_fields=True)
    return field_data, scenario_info


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Visualization Suite
# ─────────────────────────────────────────────────────────────────────────────

def _add_lhs_annotation(ax, n_total, n_cluster=None, fontsize=9):
    """Add a truthful LHS annotation box. Explicitly NOT labelled as ref paper method."""
    if n_cluster is not None:
        text = (
            f"LHS Ensemble\n"
            f"Cluster n = {n_cluster:,}\n"
            f"Total N = {n_total:,}"
        )
    else:
        text = (
            f"Latin Hypercube Sampling (LHS)\n"
            f"Space-filling, 14-dim param space\n"
            f"N = {n_total:,} scenarios"
        )
    ax.text(0.02, 0.03, text, transform=ax.transAxes,
            fontsize=fontsize, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                      edgecolor="#adb5bd", alpha=0.90))


# ── Figure 1: Strategy Field Slices ──────────────────────────────────────────
def plot_strategy_field_slices(field_data, scenario_info):
    """Figure 1 (Ref Fig. 4 Style): Mid-Z field slices at Year 40.

    Layout: n_scenarios rows x 2 columns (T_K, p_MPa).
      Rows 0-1: Reservoir A (low/high injection rate)
      Rows 2-3: Reservoir B
      Rows 4-5: Reservoir C
    """
    if not HAS_MATPLOTLIB:
        return
    pred_T = field_data.get("pred_T_field")
    pred_P = field_data.get("pred_P_field")
    mask_T = field_data.get("mask_T")
    if pred_T is None or pred_P is None:
        print("  [SKIP] strategy_field_slices.png -- field data not collected.")
        return

    n_sc  = pred_T.shape[0]
    Z     = pred_T.shape[3]   # 20 z-layers
    z_mid = Z // 2            # mid-z index = 10
    t_end = pred_T.shape[4] - 1   # last field snapshot = Year 40

    fig, axes = plt.subplots(n_sc, 2, figsize=(10, 2.9 * n_sc),
                              gridspec_kw={"wspace": 0.40, "hspace": 0.55})
    if n_sc == 1:
        axes = axes[np.newaxis, :]

    for row, info in enumerate(scenario_info):
        ax_T = axes[row, 0]
        ax_P = axes[row, 1]

        # T_K: apply validity mask (cap-rock cells -> NaN)
        T_slice = pred_T[row, :, :, z_mid, t_end]
        if mask_T is not None:
            T_plot = np.where(mask_T[row, :, :, z_mid, t_end], T_slice, np.nan)
        else:
            T_plot = T_slice

        im_T = ax_T.imshow(T_plot.T, cmap="inferno", origin="lower",
                           aspect="auto", interpolation="nearest")
        cb_T = fig.colorbar(im_T, ax=ax_T, fraction=0.046, pad=0.04)
        cb_T.set_label(get_label("T_K", "unit_only"), fontsize=9)

        # p_MPa
        P_slice = pred_P[row, :, :, z_mid, t_end]
        im_P = ax_P.imshow(P_slice.T, cmap="viridis", origin="lower",
                           aspect="auto", interpolation="nearest")
        cb_P = fig.colorbar(im_P, ax=ax_P, fraction=0.046, pad=0.04)
        cb_P.set_label(get_label("p_MPa", "unit_only"), fontsize=9)

        # Subtitle with operational context
        c_lbl = info["cluster_label"]
        sfx   = info["suffix"]
        q_s   = (f"  Q={info['inj_rate_kgs']:.1f} kg/s"
                 if "inj_rate_kgs" in info else "")
        ws_s  = (f"  d={info['well_spacing_m']:.0f} m"
                 if "well_spacing_m" in info else "")
        subtitle = f"{c_lbl} -- {sfx}{q_s}{ws_s}"

        ax_T.set_title(f"{get_label('T_K', 'title')}\n{subtitle}", fontsize=9.5, fontweight="bold")
        ax_P.set_title(f"{get_label('p_MPa', 'title')}\n{subtitle}", fontsize=9.5, fontweight="bold")

        for ax in [ax_T, ax_P]:
            ax.set_xlabel("X grid cell", fontsize=9)
            ax.tick_params(labelsize=6)
        ax_T.set_ylabel(f"{c_lbl}\n{sfx}\nY grid cell", fontsize=9, fontweight="bold")
        ax_P.set_ylabel("Y grid cell", fontsize=9)

    fig.suptitle(
        "Illustrative Scenarios -- Subsurface Fields at Year 40, Mid-Depth Plane\n"
        "Low- and high-flow examples per scenario cluster (other operating conditions also differ)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _save_fig(fig, "strategy_field_slices.png")


# ── Figure 2a: Pooled MC Fan Chart ───────────────────────────────────────────
def plot_mc_percentile_fan(well_time_years, pooled_bands, n_total):
    """Figure 2a: Pooled 2x2 MC percentile fan chart.

    Panels: T_prod_K, P_inj_MPa, dP_MPa, P_prod_MPa.
    """
    if not HAS_MATPLOTLIB:
        return

    CHANNELS = ["T_prod_K", "P_inj_MPa", "dP_MPa", "P_prod_MPa"]
    t = np.array(well_time_years)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    for c, name in enumerate(CHANNELS):
        ax    = axes[c]
        color = get_color(name)
        bd    = pooled_bands.get(name)
        if bd is None:
            ax.set_visible(False)
            continue

        p10 = np.array(bd["p10"]);  p25 = np.array(bd["p25"])
        p50 = np.array(bd["p50"]);  p75 = np.array(bd["p75"])
        p90 = np.array(bd["p90"]);  mn  = np.array(bd["mean"])

        # Two-tier fill: outer P10-P90, inner IQR P25-P75
        ax.fill_between(t, p10, p90, alpha=0.12, color=color, label="P10-P90 Spread (80%)")
        ax.fill_between(t, p25, p75, alpha=0.38, color=color, label="P25-P75 IQR (50%)")
        ax.plot(t, p50, color=color, lw=2.2, zorder=5, label="P50 Median")
        ax.plot(t, mn,  color=color, lw=1.2, ls="--", alpha=0.70, label="Ensemble Mean")
        ax.plot(t, p10, color=color, lw=0.7, ls=":", alpha=0.50, label="P10")
        ax.plot(t, p90, color=color, lw=0.7, ls=":", alpha=0.50, label="P90")

        unit = get_label(name, "unit_only")
        full = get_label(name, "full")
        ax.set_xlabel("Simulation Year", fontsize=11)
        ax.set_ylabel(f"{full} {unit}", fontsize=11)
        title = full
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=9, loc="best", framealpha=0.85, ncol=1)
        ax.grid(True, alpha=0.22, ls=":")
        ax.set_xlim(0, 40)

        if c == 0:
            _add_lhs_annotation(ax, n_total)

    fig.suptitle(
        f"Ensemble scenario analysis: empirical percentile bands of surrogate predictions (N = {n_total:,})",
        fontsize=11.5, fontweight="bold",
    )
    _save_fig(fig, "mc_percentile_fan_chart.png")


# ── Figure 2b: Per-Reservoir Fan Chart ───────────────────────────────────────
def plot_mc_percentile_fan_by_reservoir(well_time_years, per_cluster_bands,
                                         cluster_labels, n_total, k=3):
    """Figure 2b: 3x3 fan chart (3 reservoir archetypes x 3 key channels).

    Shows how uncertainty bands differ across geological environments.
    """
    if not HAS_MATPLOTLIB:
        return

    CHANNELS = ["T_prod_K", "P_inj_MPa", "dP_MPa"]
    t = np.array(well_time_years)
    fig, axes = plt.subplots(len(CHANNELS), k,
                              figsize=(5.5 * k, 4.5 * len(CHANNELS)),
                              constrained_layout=True)

    for row, ch_name in enumerate(CHANNELS):
        for col in range(k):
            ax  = axes[row, col]
            bd  = per_cluster_bands[col]
            n_c = int((cluster_labels == col).sum())
            c_color = CLUSTER_COLORS[col]

            if bd is None or ch_name not in bd:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            p10 = np.array(bd[ch_name]["p10"])
            p25 = np.array(bd[ch_name]["p25"])
            p50 = np.array(bd[ch_name]["p50"])
            p75 = np.array(bd[ch_name]["p75"])
            p90 = np.array(bd[ch_name]["p90"])

            ax.fill_between(t, p10, p90, alpha=0.12, color=c_color)
            ax.fill_between(t, p25, p75, alpha=0.38, color=c_color)
            ax.plot(t, p50, color=c_color, lw=2.0, zorder=5, label="P50 Median")
            ax.plot(t, p10, color=c_color, lw=0.7, ls=":", alpha=0.50, label="P10")
            ax.plot(t, p90, color=c_color, lw=0.7, ls=":", alpha=0.50, label="P90")

            if row == 0:
                ax.set_title(
                    f"{CLUSTER_LABELS[col]}\n(n = {n_c:,} scenarios)",
                    fontsize=12, fontweight="bold", color=c_color,
                )
            if col == 0:
                ax.set_ylabel(get_label(ch_name, "full_unit"), fontsize=10.5)
            ax.set_xlabel("Year", fontsize=10)
            ax.grid(True, alpha=0.20, ls=":")
            ax.set_xlim(0, 40)

            if row == 0 and col == 0:
                _add_lhs_annotation(ax, n_total, n_cluster=n_c, fontsize=9)

    fig.suptitle(
        f"Empirical percentile bands by scenario cluster (K-means, K = 3; N = {n_total:,})",
        fontsize=12.5, fontweight="bold",
    )
    _save_fig(fig, "mc_percentile_fan_chart_by_reservoir.png")


# ── Figure 3: KPI Distributions Boxplots ─────────────────────────────────────
def plot_kpi_distributions_boxplots(kpis, cluster_labels, k=3):
    """Figure 3 (Ref Fig. 5a-d Style): Grouped boxplots for KPIs by reservoir archetype.

    Box: IQR (25th-75th pct)  |  Whiskers: 1.5xIQR  |  D = mean  |  line = median
    """
    if not HAS_MATPLOTLIB:
        return

    keys_to_plot = [kk for kk in ["MW_th", "Cum_Energy", "Delta_T_pct"] if kk in kpis]
    if not keys_to_plot:
        print("  [SKIP] kpi_distributions_boxplots.png -- no KPI data.")
        return

    fig, axes = plt.subplots(1, len(keys_to_plot),
                              figsize=(5.5 * len(keys_to_plot), 6.5),
                              constrained_layout=True)
    if len(keys_to_plot) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys_to_plot):
        data_per_cluster  = []
        labels_per_cluster = []
        for c in range(k):
            mask = cluster_labels == c
            if mask.sum() > 0:
                data_per_cluster.append(kpis[key][mask])
                labels_per_cluster.append(f"{CLUSTER_LABELS[c]}\n(n={mask.sum():,})")

        bp = ax.boxplot(
            data_per_cluster,
            patch_artist=True,
            showmeans=True,
            widths=0.5,
            meanprops=dict(marker="D", markerfacecolor="black",
                           markeredgecolor="black", markersize=6.5),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.45),
        )
        for patch, c in zip(bp["boxes"], range(k)):
            patch.set_facecolor(CLUSTER_COLORS[c])
            patch.set_alpha(0.75)

        ax.set_xticklabels(labels_per_cluster, fontsize=10.5)
        ax.set_title(get_label(key, "full"), fontsize=12.5, fontweight="bold")
        ax.set_ylabel(get_label(key, "full_unit"), fontsize=11)
        ax.grid(True, axis="y", alpha=0.25, ls=":")

        # P10/P50/P90 annotation per box
        y_top = ax.get_ylim()[1]
        for c_idx, data_c in enumerate(data_per_cluster):
            ax.text(c_idx + 1, y_top * 0.98,
                    f"P10={np.percentile(data_c,10):.1f}\n"
                    f"P50={np.percentile(data_c,50):.1f}\n"
                    f"P90={np.percentile(data_c,90):.1f}",
                    ha="center", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#cccccc", alpha=0.82))

    # Shared cluster colour legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_COLORS[c], alpha=0.75)
               for c in range(k)]
    fig.legend(handles, [CLUSTER_LABELS[c] for c in range(k)],
               loc="upper center", ncol=k, fontsize=11,
               bbox_to_anchor=(0.5, 1.09), framealpha=0.90)

    fig.suptitle(
        "Derived indicators by scenario cluster\n"
        "Box = IQR (25th-75th pct)  |  Whiskers = 1.5xIQR  |  D = Mean  |  -- = Median",
        fontsize=11.5, fontweight="bold", y=1.15,
    )
    _save_fig(fig, "kpi_distributions_boxplots.png")


# ── Figure 4: Production Temp Decline Dual-Axis ──────────────────────────────
def plot_production_temp_decline_dual_axis(pred_wells, kpis, cluster_labels, k=3):
    """Figure 4 (Ref Fig. 5e/g Adaptation): scatter-density dual-axis.

    X-axis: scenarios sorted by Delta_T_pct (low to high).
    Left Y: T_prod at Year 40 [K].  Right Y: Delta_T_pct [%].
    Colour-coded by geological cluster.
    """
    if not HAS_MATPLOTLIB:
        return
    if "Delta_T_pct" not in kpis:
        print("  [SKIP] production_temp_decline_dual_axis.png -- Delta_T_pct missing.")
        return

    T_end    = pred_wells[:, -1, 0]
    dT_pct   = kpis["Delta_T_pct"]
    sort_idx = np.argsort(dT_pct)

    fig, ax1 = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax2 = ax1.twinx()

    x_pos = np.arange(len(sort_idx))

    for c in range(k):
        cluster_sorted_mask = cluster_labels[sort_idx] == c
        x_c = x_pos[cluster_sorted_mask]
        ax1.scatter(x_c, T_end[sort_idx][cluster_sorted_mask],
                    c=CLUSTER_COLORS[c], s=8, alpha=0.55,
                    marker=CLUSTER_MARKERS[c], zorder=3)
        ax2.scatter(x_c, dT_pct[sort_idx][cluster_sorted_mask],
                    c=CLUSTER_COLORS[c], s=8, alpha=0.30,
                    marker=CLUSTER_MARKERS[c], zorder=2)

    # Median reference lines
    p50_T  = float(np.median(T_end))
    p50_dT = float(np.median(dT_pct))
    ax1.axhline(p50_T,  color="#1d3557", lw=1.5, ls="--", alpha=0.80)
    ax2.axhline(p50_dT, color="#e63946", lw=1.5, ls="-.", alpha=0.80)

    ax1.set_xlabel(
        "Scenarios sorted by Temperature Decline (Delta_T%) -- Low to High",
        fontsize=11.5
    )
    ax1.set_ylabel(get_label("T_prod_K", "full_unit"), fontsize=11.5, color="#1d3557")
    ax2.set_ylabel(get_label("Delta_T_pct", "full_unit"), fontsize=11.5, color="#e63946")
    ax1.tick_params(axis="y", labelcolor="#1d3557")
    ax2.tick_params(axis="y", labelcolor="#e63946")
    ax1.grid(True, alpha=0.20, ls=":")

    # Cluster legend
    cluster_handles = [
        Line2D([0], [0], marker=CLUSTER_MARKERS[c], color=CLUSTER_COLORS[c],
               linestyle="None", markersize=7, label=CLUSTER_LABELS[c])
        for c in range(k)
    ]
    # Reference line legend
    ref_handles = [
        Line2D([0], [0], color="#1d3557", lw=1.5, ls="--",
               label=f"Median T_prod = {p50_T:.1f} K"),
        Line2D([0], [0], color="#e63946", lw=1.5, ls="-.",
               label=f"Median DeltaT% = {p50_dT:.1f}%"),
    ]
    ax1.legend(handles=cluster_handles + ref_handles,
               loc="upper left", fontsize=10, framealpha=0.88, ncol=2)

    ax1.set_title(
        "Production Temperature at Year 40 vs. 40-Year Temperature Decline\n"
        "Colour = Scenario Cluster",
        fontsize=12.5, fontweight="bold",
    )
    _save_fig(fig, "production_temp_decline_dual_axis.png")


# ── Figure 5: Stratified Capacity KDE ────────────────────────────────────────
def plot_stratified_capacity_kde(kpis, n_total):
    """Figure 5 (Ref Fig. 5i-l Adaptation): KDE capacity curves stratified by DeltaT%.

    Top strip: rug plot showing individual scenario density.
    Falls back to normalised histograms if scipy is unavailable.
    """
    if not HAS_MATPLOTLIB:
        return
    if "MW_th" not in kpis or "Delta_T_pct" not in kpis:
        print("  [SKIP] stratified_capacity_kde.png -- MW_th or Delta_T_pct missing.")
        return

    mw_all = kpis["MW_th"]
    dt_all = kpis["Delta_T_pct"]

    # Decline classes used in the manuscript (Section 3.5); every scenario is
    # far below the 10% screening criterion, so finer classes are informative.
    bins = [
        (">= 6%",   dt_all >= 6,                     "#d62828"),
        ("4-6%",    (dt_all >= 4) & (dt_all < 6),   "#2d6a4f"),
        ("2-4%",    (dt_all >= 2) & (dt_all < 4),   "#f77f00"),
        ("< 2%",    dt_all < 2,                     "#4361ee"),
    ]

    fig = plt.figure(figsize=(10, 7.0), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 8], figure=fig)
    ax_rug  = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    x_margin = 0.06 * (mw_all.max() - mw_all.min())
    x_lo = mw_all.min() - x_margin
    x_hi = mw_all.max() + x_margin
    x_grid = np.linspace(x_lo, x_hi, 500)

    legend_handles = []
    for label, mask, color in bins:
        subset = mw_all[mask]
        n_sub  = int(mask.sum())
        if n_sub < 2:
            continue

        lbl_str = f"DeltaT {label}  (n={n_sub:,})"

        if HAS_SCIPY:
            kde  = gaussian_kde(subset, bw_method="scott")
            y    = kde(x_grid)
            ax_main.plot(x_grid, y, color=color, lw=2.2, label=lbl_str)
            ax_main.fill_between(x_grid, y, alpha=0.18, color=color)
        else:
            cnts, edges = np.histogram(subset, bins=25, density=True)
            ctrs = 0.5 * (edges[:-1] + edges[1:])
            ax_main.plot(ctrs, cnts, color=color, lw=2.2, label=lbl_str)
            ax_main.fill_between(ctrs, cnts, alpha=0.18, color=color)

        # Rug: individual scenario marks on top strip
        ax_rug.plot(subset, np.zeros_like(subset), "|",
                    color=color, alpha=0.55, markersize=7)
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=lbl_str))

    # Dataset median reference
    p50_mw = float(np.median(mw_all))
    ax_main.axvline(p50_mw, color="#1d3557", lw=1.5, ls="--")
    legend_handles.append(
        Line2D([0], [0], color="#1d3557", lw=1.5, ls="--",
               label=f"Dataset Median = {p50_mw:.2f} MWth")
    )

    ax_main.set_xlabel(get_label("MW_th", "full_unit"), fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Probability Density", fontsize=12)
    ax_main.legend(handles=legend_handles,
                   title="Temperature Decline Bins",
                   fontsize=10.5, title_fontsize=11, framealpha=0.88)
    ax_main.grid(True, alpha=0.22, ls=":")
    ax_main.set_xlim(x_lo, x_hi)

    ax_rug.set_xlim(ax_main.get_xlim())
    ax_rug.set_yticks([])
    ax_rug.set_xticks([])
    ax_rug.set_ylabel("Rug", fontsize=9, rotation=0, labelpad=25)
    for spine in ["top", "right", "left", "bottom"]:
        ax_rug.spines[spine].set_visible(spine == "bottom")
    ax_rug.spines["bottom"].set_visible(False)

    fig.suptitle(
        f"Year-40 Thermal Capacity by 40-Year Temperature-Decline Class\n"
        f"KDE curves  |  N = {n_total:,} scenarios",
        fontsize=12.5, fontweight="bold",
    )
    _save_fig(fig, "stratified_capacity_kde.png")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: Markdown Report
# ─────────────────────────────────────────────────────────────────────────────
def write_markdown_report(
    ckpt_epoch, n_total, cluster_labels, cluster_stats,
    pooled_bands, per_cluster_bands, kpis,
    well_time_years, param_names,
    bhp_prod_quasi, clustering_method, k=3,
):
    """Write structured UQ report. NO RE% section -- accuracy belongs in evaluate_test_set.py."""
    t_arr    = np.array(well_time_years)
    idx_yr0  = 0
    idx_yr20 = int(np.argmin(np.abs(t_arr - 20.0)))
    idx_yr40 = len(t_arr) - 1

    lines = []
    a = lines.append

    a("# Forward Monte Carlo UQ Report")
    a("")
    a(f"**Model:** ComsolFnoModel (COMSOL/FNO 3D+Time) | Checkpoint epoch {ckpt_epoch}  ")
    a(f"**Ensemble:** N = {n_total:,} LHS scenarios | "
      f"K = {k} geological archetypes (K-Means)  ")
    a("> Reference: Xue et al., Nature Communications 2026, DOI: 10.1038/s41467-026-73183-0")
    a("")
    a("---")
    a("")

    a("## 1. Methodology & Dataset Mapping Notes")
    a("")
    a("### 1.1 What This Report Is (and Is Not)")
    a("")
    a("This report presents **Uncertainty Quantification (UQ)** results: how physical")
    a("outputs vary across the 14-dimensional LHS input space. It does NOT contain")
    a("Relative Error (RE%) accuracy metrics. RE% belongs in `evaluate_test_set.py`.")
    a("")
    a("### 1.2 Our Dataset vs. Reference Paper")
    a("")
    a("| Feature | Reference Paper | This Work |")
    a("|---------|----------------|-----------|")
    a("| Sampling | Independent uniform random | Latin Hypercube Sampling (LHS) |")
    a("| N | ~1,000/reservoir (2,000 total) | 1,535 total |")
    a("| Why LHS | N/A | Uniform marginal coverage in high-dim space |")
    a("| Reservoir types | 2 pre-designed templates | K=3 emergent K-Means clusters |")
    a("")
    a("### 1.3 Geological Reservoir Grouping")
    a("")
    a(f"Method: `{clustering_method}`")
    a("")
    a("K-Means clusters the 10 geological parameters only. Operational parameters")
    a("(inj_rate, inj_temp, well_spacing) still vary continuously within each cluster.")
    a("")
    a("### 1.4 P_prod_MPa (BHP_prod) Status")
    a("")
    if bhp_prod_quasi:
        a("**Quasi-controlled**: variability ratio std(P_prod)/std(P_inj) < 0.4.")
        a("In COMSOL pressure-controlled producer wells, P_prod approximately tracks")
        a("the imposed boundary condition. Shown in figures with warning label.")
    else:
        a("**Genuine model output**: variability ratio >= 0.4. Treated as full prediction.")
    a("")
    a("---")
    a("")

    a("## 2. Variable Abbreviation Dictionary")
    a("")
    a("| Abbreviation | Description | Unit |")
    a("|:-------------|:-----------|:-----|")
    for key, info in ABBREV_DICT.items():
        unit = f"`{info['unit']}`" if info.get("unit") else "--"
        a(f"| `{key}` | {info['full']} | {unit} |")
    a("")
    a("---")
    a("")

    a("## 3. Geological Reservoir Cluster Characterisation")
    a("")
    a("| Cluster | n | Median H [m] | H Range [m] | Median Grad [dC/km] | Median inj_rate [kg/s] |")
    a("|:--------|:--|:------------|:------------|:--------------------|:----------------------|")
    for c in range(k):
        cs = cluster_stats.get(c, {})
        a(f"| {CLUSTER_LABELS[c]} | {cs.get('n', '?'):,} | "
          f"{cs.get('median_H', 0):.1f} | "
          f"{cs.get('min_H', 0):.0f}-{cs.get('max_H', 0):.0f} | "
          f"{cs.get('median_grad', 0):.2f} | "
          f"{cs.get('median_inj_rate', 0):.1f} |")
    a("")
    a("> Operational values per cluster reflect LHS co-variation, not controlled isolation.")
    a("")
    a("---")
    a("")

    a("## 4. Derived KPI Formulas")
    a("")
    a("**Temperature Decline:**")
    a("  DeltaT% = (T_prod(t=0) - T_prod(t=40yr)) / T_prod(t=0) * 100  [%]")
    a("")
    a("**Thermal Power:**")
    a("  MW_th(t) = inj_rate * c_p * max(T_prod(t) - T_inj, 0) / 1e6  [MW]")
    a("  c_p = 4184 J/(kg*K), specific heat of water")
    a("")
    a("**Cumulative Energy:**")
    a("  Cum_Energy = sum_t MW_th(t) * dt [yr] * 31557.6  [GJ]")
    a("  31557.6 GJ/(MW*yr) = 1e6 W * 365.25 * 86400 s / 1e9")
    a("")
    a("---")
    a("")

    a("## 5. MC Percentile Band Summary")
    a("")
    a(f"Year 0 (idx {idx_yr0}), Year 20 (idx {idx_yr20}), Year 40 (idx {idx_yr40})")
    a("")

    def _band_table(bd_dict, section_label, n_sc):
        """Appends a Markdown subsection for one band group (pooled or a
        single cluster), with one percentile table per well channel.

        Args:
          bd_dict:       Band dict as returned by compute_mc_percentile_bands
                          (keyed by channel name).
          section_label:  Heading text for this subsection (e.g. cluster name).
          n_sc:           Scenario count for this group, shown in the heading.
        """
        a(f"### {section_label}  (n = {n_sc:,})")
        a("")
        for ch in ["T_prod_K", "P_inj_MPa", "dP_MPa", "P_prod_MPa"]:
            bd = bd_dict.get(ch)
            if bd is None:
                continue
            unit = get_label(ch, "unit_only")
            a(f"**{get_label(ch, 'full')} {unit}**")
            a("")
            a(f"| | Year 0 | Year 20 | Year 40 |")
            a("|:--|:-------|:--------|:--------|")
            for pkey, plbl in [("p10","P10"),("p25","P25"),("p50","P50 Median"),
                                ("p75","P75"),("p90","P90")]:
                arr = np.array(bd[pkey])
                a(f"| {plbl} | {arr[idx_yr0]:.3f} | {arr[idx_yr20]:.3f} | {arr[idx_yr40]:.3f} |")
            iqr = np.array(bd["iqr_width"])
            spd = np.array(bd["spread_width"])
            a(f"| IQR (P75-P25) | {iqr[idx_yr0]:.3f} | {iqr[idx_yr20]:.3f} | {iqr[idx_yr40]:.3f} |")
            a(f"| Spread (P90-P10) | {spd[idx_yr0]:.3f} | {spd[idx_yr20]:.3f} | {spd[idx_yr40]:.3f} |")
            a("")

    if pooled_bands:
        _band_table(pooled_bands, "Pooled Ensemble", n_total)
    for c in range(k):
        if per_cluster_bands[c] is not None:
            n_c = int((cluster_labels == c).sum())
            _band_table(per_cluster_bands[c], CLUSTER_LABELS[c], n_c)

    a("---")
    a("")

    a("## 6. KPI Distribution Statistics")
    a("")
    a("| KPI | Unit | Mean | P10 | P50 | P90 | Std |")
    a("|:----|:-----|:-----|:----|:----|:----|:----|")
    for kpi_key in ["Delta_T_pct", "MW_th", "Cum_Energy"]:
        if kpi_key not in kpis:
            continue
        arr  = kpis[kpi_key]
        unit = get_label(kpi_key, "unit_only")
        a(f"| {get_label(kpi_key, 'full')} | {unit} | "
          f"{np.mean(arr):.2f} | {np.percentile(arr,10):.2f} | "
          f"{np.percentile(arr,50):.2f} | {np.percentile(arr,90):.2f} | "
          f"{np.std(arr):.2f} |")
    a("")
    a("---")
    a("")

    a("## 7. Output Figures")
    a("")
    a("| Figure | Reference | Description |")
    a("|:-------|:----------|:------------|")
    a("| `strategy_field_slices.png` | Fig. 4 | Mid-Z T_K and p_MPa at Year 40 for "
      "2 injection strategies x 3 geological archetypes. |")
    a("| `mc_percentile_fan_chart.png` | Fig. 4/5 | Pooled 2x2 fan chart: "
      "P10/P25/P50/P75/P90 over 40-year horizon. |")
    a("| `mc_percentile_fan_chart_by_reservoir.png` | Fig. 4 | Per-cluster 3x3 fan chart. |")
    a("| `kpi_distributions_boxplots.png` | Fig. 5a-d | Grouped boxplots: MW_th, "
      "Cum_Energy, DeltaT%. |")
    a("| `production_temp_decline_dual_axis.png` | Fig. 5e/g | Scatter: T_prod(Year 40) "
      "vs DeltaT%, coloured by cluster. |")
    a("| `stratified_capacity_kde.png` | Fig. 5i-l | KDE of MW_th by DeltaT% bins "
      "+ rug plot. |")
    a("")
    a("---")
    a(f"*Generated by uq_analysis_mc_v2.py | "
      f"N = {n_total:,} LHS realizations | Zero fabricated numbers.*")

    report_path = os.path.join(UQ_OUT_DIR, "uq_summary_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved -> {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Runs the full v2 Monte Carlo UQ pipeline end to end: loads the
    prepared 1,535-scenario ensemble and trained checkpoint, forward-passes
    every scenario through the model, clusters scenarios into geological
    archetypes, computes percentile bands and derived KPIs, collects
    representative spatial fields, renders all figures, and writes the
    Markdown report and JSON results to UQ_OUT_DIR. Takes no arguments and
    returns nothing; all outputs are written to disk."""
    # SLURM stdout line-buffering fix
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)

    os.makedirs(UQ_OUT_DIR, exist_ok=True)

    print(f"\n{'='*72}")
    print("  UQ Analysis v2 -- Pure Monte Carlo Forward Propagation Engine")
    print("  ComsolFnoModel: COMSOL/FNO 3D+Time Geothermal Surrogate")
    print(f"  Output -> {UQ_OUT_DIR}")
    print(f"{'='*72}\n")
    print_console_column_legend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _validate_paths()
    print(f"\n[PRE-FLIGHT PASSED]  Dataset: {PREPARED_DATA_PATH}")
    print(f"                     Checkpoint: {BEST_CKPT_PATH}")

    # ═══════════════════════════════════════════════════════════════════════════
    _section_banner("STAGE 1/5 -- Schema Discovery")
    # ═══════════════════════════════════════════════════════════════════════════
    all_dataset = _ComsolFnoDatasetAll(PREPARED_DATA_PATH)
    print(f"  Loaded: {all_dataset.n:,} LHS scenarios")
    print(f"  Well time series: {len(all_dataset.well_time_years)} steps  "
          f"({all_dataset.well_time_years[0]:.1f} to {all_dataset.well_time_years[-1]:.1f} yr)")
    print(f"  Active parameters: {len(all_dataset.param_names)} "
          f"(well_radius dropped -- constant 0.085m)")
    print(f"  Field snapshots: {len(UNIFORM_TIME_YEARS)}  years={UNIFORM_TIME_YEARS}")

    inspect_parameters(all_dataset.param_names, all_dataset.x_params_raw)

    # ═══════════════════════════════════════════════════════════════════════════
    _section_banner("STAGE 2/5 -- Loading Model Checkpoint (read-only)")
    # ═══════════════════════════════════════════════════════════════════════════
    model          = ComsolFnoModel(**MODEL_KWARGS).to(device)
    ckpt           = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ckpt_epoch     = int(ckpt["epoch"])
    n_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Checkpoint epoch     : {ckpt_epoch}")
    print(f"  Trainable parameters : {n_params_model:,}")
    print(f"  Architecture         : ComsolFnoModel {MODEL_KWARGS}")

    # ═══════════════════════════════════════════════════════════════════════════
    _section_banner("STAGE 3/5 -- Monte Carlo Forward Ensemble")
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"  Running {all_dataset.n:,} LHS scenarios through ComsolFnoModel...")
    print("  (Requires ~4-16 GB RAM; falls back to test split on OOM.)")

    pred_wells = None
    n_total    = 0

    try:
        mc_result  = run_mc_forward_pass(model, all_dataset, device, ALL_BATCH_SIZE)
        pred_wells = mc_result["pred_wells"]   # (1535, 201, 3)
        n_total    = pred_wells.shape[0]
        print(f"  OK MC forward passes: {n_total:,} scenarios")
    except (RuntimeError, MemoryError) as e:
        print(f"  [WARNING] Full-ensemble OOM ({type(e).__name__}): {e}")
        print("  Falling back to test split...")
        fallback_ds = ComsolFnoDataset(PREPARED_DATA_PATH, split="test")
        fb_res      = run_mc_forward_pass(model, fallback_ds, device, ALL_BATCH_SIZE)
        pred_wells  = fb_res["pred_wells"]
        n_total     = pred_wells.shape[0]
        print(f"  Fallback ensemble: {n_total} scenarios (test split only)")

    well_time_years = all_dataset.well_time_years

    # BHP_prod status check
    bhp_prod_quasi, bhp_ratio = verify_bhp_prod_status(pred_wells)

    # Geological clustering
    _section_banner("  -> Geological Reservoir Clustering (K-Means K=3)")
    cluster_labels, clustering_method = cluster_geological_groups(
        all_dataset.x_params_raw, all_dataset.param_names, k=K_CLUSTERS
    )
    print_cluster_characterization(
        cluster_labels, all_dataset.x_params_raw, all_dataset.param_names, k=K_CLUSTERS
    )

    # Percentile bands
    _section_banner("  -> Computing Percentile Bands (pooled + per-cluster)")
    pooled_bands, per_cluster_bands = compute_all_bands(
        pred_wells, cluster_labels, well_time_years, k=K_CLUSTERS
    )
    print(f"  OK: pooled bands for channels {list(pooled_bands.keys())}")

    # Derived KPIs
    _section_banner("  -> Derived KPI Computation")
    kpis = compute_derived_kpis(
        pred_wells,
        all_dataset.x_params_raw,
        all_dataset.param_names,
        well_time_years,
    )
    print(f"  OK: KPIs computed: {[k for k in kpis if not k.endswith('_ts')]}")

    # Cluster summary stats for report
    def _get_param_col(name):
        """Looks up the column of `all_dataset.x_params_raw` for parameter
        `name`.

        Returns the (N,) array of raw physical-unit values, or None if
        `name` is not present in `all_dataset.param_names`.
        """
        idx = next((i for i, n in enumerate(all_dataset.param_names) if n == name), None)
        return all_dataset.x_params_raw[:, idx] if idx is not None else None

    H_col  = _get_param_col("H")
    gr_col = _get_param_col("grad_C_per_km")
    q_col  = _get_param_col("inj_rate")

    cluster_stats = {}
    for c in range(K_CLUSTERS):
        mask = cluster_labels == c
        cs   = {"n": int(mask.sum())}
        for key, col in [("H", H_col), ("grad", gr_col), ("inj_rate", q_col)]:
            if col is not None and mask.sum() > 0:
                sub = col[mask]
                cs[f"median_{key}"] = float(np.median(sub))
                cs[f"min_{key}"]    = float(sub.min())
                cs[f"max_{key}"]    = float(sub.max())
        cluster_stats[c] = cs

    # ═══════════════════════════════════════════════════════════════════════════
    _section_banner("STAGE 4/5 -- Representative Spatial Field Collection")
    # ═══════════════════════════════════════════════════════════════════════════
    field_data    = {}
    scenario_info = []
    try:
        field_data, scenario_info = collect_representative_fields(
            model, all_dataset, cluster_labels,
            all_dataset.x_params_raw, all_dataset.param_names,
            device, k=K_CLUSTERS,
        )
        print(f"  OK: collected fields for {len(scenario_info)} representative scenarios")
    except Exception as exc:
        print(f"  [WARNING] Field collection failed: {exc}")
        print("  strategy_field_slices.png will be skipped.")

    # ═══════════════════════════════════════════════════════════════════════════
    _section_banner("STAGE 5/5 -- Figures, Report, JSON")
    # ═══════════════════════════════════════════════════════════════════════════

    if HAS_MATPLOTLIB:
        print("\n  Figure 1: strategy_field_slices.png")
        plot_strategy_field_slices(field_data, scenario_info)

        print("  Figure 2a: mc_percentile_fan_chart.png")
        plot_mc_percentile_fan(well_time_years, pooled_bands, n_total)

        print("  Figure 2b: mc_percentile_fan_chart_by_reservoir.png")
        plot_mc_percentile_fan_by_reservoir(
            well_time_years, per_cluster_bands, cluster_labels, n_total, k=K_CLUSTERS
        )

        print("  Figure 3: kpi_distributions_boxplots.png")
        plot_kpi_distributions_boxplots(kpis, cluster_labels, k=K_CLUSTERS)

        print("  Figure 4: production_temp_decline_dual_axis.png")
        plot_production_temp_decline_dual_axis(
            pred_wells, kpis, cluster_labels, k=K_CLUSTERS
        )

        print("  Figure 5: stratified_capacity_kde.png")
        plot_stratified_capacity_kde(kpis, n_total)
    else:
        print("  [WARN] matplotlib unavailable -- all figures skipped.")

    # Markdown report
    print("\n  Writing uq_summary_report.md...")
    write_markdown_report(
        ckpt_epoch=ckpt_epoch,
        n_total=n_total,
        cluster_labels=cluster_labels,
        cluster_stats=cluster_stats,
        pooled_bands=pooled_bands,
        per_cluster_bands=per_cluster_bands,
        kpis=kpis,
        well_time_years=well_time_years,
        param_names=all_dataset.param_names,
        bhp_prod_quasi=bhp_prod_quasi,
        clustering_method=clustering_method,
        k=K_CLUSTERS,
    )

    # JSON output
    def _bands_to_json(bd):
        """Converts a percentile-band dict into a plain, JSON-serialisable
        dict, or returns None if `bd` is empty/falsy."""
        return {name: vals for name, vals in bd.items()} if bd else None

    kpi_json = {
        "Delta_T_pct": {
            "mean": float(np.mean(kpis["Delta_T_pct"])),
            "p10":  float(np.percentile(kpis["Delta_T_pct"], 10)),
            "p50":  float(np.percentile(kpis["Delta_T_pct"], 50)),
            "p90":  float(np.percentile(kpis["Delta_T_pct"], 90)),
            "unit": "%",
            "formula": "(T_prod(0) - T_prod(end)) / T_prod(0) * 100",
        }
    }
    if "MW_th" in kpis:
        kpi_json["MW_th"] = {
            "mean": float(np.mean(kpis["MW_th"])),
            "p10":  float(np.percentile(kpis["MW_th"], 10)),
            "p50":  float(np.percentile(kpis["MW_th"], 50)),
            "p90":  float(np.percentile(kpis["MW_th"], 90)),
            "unit": "MWth",
            "formula": "inj_rate * c_p * max(T_prod - T_inj, 0) / 1e6",
            "c_p_J_per_kgK": 4184.0,
        }
        kpi_json["Cum_Energy"] = {
            "mean": float(np.mean(kpis["Cum_Energy"])),
            "p10":  float(np.percentile(kpis["Cum_Energy"], 10)),
            "p50":  float(np.percentile(kpis["Cum_Energy"], 50)),
            "p90":  float(np.percentile(kpis["Cum_Energy"], 90)),
            "unit": "GJ",
            "formula": "sum(MW_th * dt_yr * 31557.6 GJ/MW/yr)",
        }

    json_out = {
        "script":             "uq_analysis_mc_v2.py",
        "methodology":        "Monte Carlo Forward Propagation of LHS Input Uncertainty",
        "reference_paper":    "Nature Comms 2026, DOI: 10.1038/s41467-026-73183-0",
        "sampling_method":    "Latin Hypercube Sampling (LHS) -- space-filling",
        "note_vs_ref_paper":  (
            "Reference paper uses independent uniform random sampling (~1000/reservoir). "
            "This work uses LHS across one continuous 14-dim parameter space."
        ),
        "checkpoint_epoch":       ckpt_epoch,
        "n_mc_realizations":      n_total,
        "k_clusters":             K_CLUSTERS,
        "clustering_method":      clustering_method,
        "cluster_assignments":    cluster_labels.tolist(),
        "cluster_stats":          {CLUSTER_LABELS[c]: cluster_stats.get(c, {}) for c in range(K_CLUSTERS)},
        "bhp_prod_quasi_controlled": bhp_prod_quasi,
        "bhp_prod_variability_ratio": round(bhp_ratio, 4),
        "pooled_bands":           _bands_to_json(pooled_bands),
        "per_cluster_bands": {
            CLUSTER_LABELS[c]: _bands_to_json(per_cluster_bands[c])
            for c in range(K_CLUSTERS)
            if per_cluster_bands[c] is not None
        },
        "kpis":                   kpi_json,
        "well_time_years":        [float(x) for x in well_time_years],
        "abbreviation_dictionary": {
            k: {"full": v["full"], "unit": v["unit"]}
            for k, v in ABBREV_DICT.items()
        },
    }
    json_path = os.path.join(UQ_OUT_DIR, "uq_results_v2.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved -> {json_path}")

    print(f"\n{'='*72}")
    print("  UQ Analysis v2 Complete.")
    print(f"  All outputs -> {UQ_OUT_DIR}/")
    print("\n  Generated files:")
    for fname in sorted(os.listdir(UQ_OUT_DIR)):
        fpath = os.path.join(UQ_OUT_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {fname:<52} {size / 1024:>8.1f} KB")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
