"""
COMSOL/FNO dataset preparation -- single-file version.

Everything needed to go from the raw assembled dataset
(fno_lhs_v1_dataset_cycle_055_1535.npz) to a fully prepared, ready-to-train
dataset, in one file. This is the same logic as the modular
data_prep.py / build_split.py / build_prepared_dataset.py /
verify_prepared_dataset.py files, just combined so it's easier for anyone
on the team to run and debug top to bottom without jumping between files.

Run it with the location of the input files:
    python scripts/prepare_data.py --raw-npz <raw .npz> --summary-csv <summary .csv> --out <prepared .npz>

It will:
  1. Load the raw assembled dataset
  2. Fix issue 1: derive well-location channels (well map + distance-to-well)
  3. Fix issue 3: build per-scenario validity masks (cap-rock domain boundary)
  4. Fix issue 2: detect and repair implausible T_K outlier cells
  5. Compute normalization statistics
  6. Build a stratified train/val/test split
  7. Save one consolidated output file
  8. Reload that saved file fresh and verify it -- because "the script ran
     without errors" is not the same as "the output is correct" (this
     exact gap caught a real bug during development -- see the printed
     verification section, and COMSOL_FNO_Dataset_Findings_and_Model_Plan.md
     / Phase1_Implementation_and_Phase2_Plan.md in the repo root for the
     full story).

See those two .md files for the reasoning behind every decision below.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (set from the command line in the __main__ block) and constants
# ---------------------------------------------------------------------------
RAW_NPZ_PATH = 'fno_lhs_v1_dataset_cycle_055_1535.npz'
SUMMARY_CSV_PATH = 'UQ_summary_fno_lhs.csv'
OUT_PATH = 'fno_lhs_v1_prepared_phase1.npz'

# Issue 2: physically implausible T_K range. Lower bound sits just below the
# coldest injection temperature in the dataset (293.15K) -- nothing in the
# reservoir should ever be colder than the coldest water pumped into it, so
# anything below this is either a glitch or untrustworthy numerical noise.
# Upper bound is comfortably above the hottest legitimate reservoir temp observed (~421K).
T_OUTLIER_MIN = 290.0
T_OUTLIER_MAX = 450.0
MAX_REPAIR_PASSES = 10  # iterative repair -- see step 4 below for why this needs to be iterative

FACE_NEIGHBOR_OFFSETS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

# Split fractions (train/val, test is the remainder), stratified by design_focus
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
RANDOM_SEED = 7


# ---------------------------------------------------------------------------
# Step 1: load the raw assembled dataset
# ---------------------------------------------------------------------------
def load_dataset(npz_path):
    """Loads the raw assembled dataset .npz file (field/param/well arrays,
    coordinate grids, and metadata) into a plain dict keyed by array name.

    npz_path: path to the raw assembled dataset .npz file (e.g.
    fno_lhs_v1_dataset_cycle_055_1535.npz). Returns a dict mapping every
    array name stored in the file to its loaded numpy array.
    """
    raw = np.load(npz_path, allow_pickle=True)
    return {key: raw[key] for key in raw.files}


# ---------------------------------------------------------------------------
# Step 2 (Issue 1): derive well-location channels from well_spacing
# ---------------------------------------------------------------------------
def build_well_location_channels(well_spacing, x_m, y_m, z_m):
    """
    The assembled dataset has no well-location input channel -- PCNO's
    original format explicitly gave the model a well map + distance-to-well
    channels rather than making it infer well position, so we derive the
    same thing here.

    Verified well geometry (from the COMSOL model-building template, and
    confirmed empirically via connected-component analysis on 1,542
    scenarios): exactly one injector and one producer per scenario, at
    x = -well_spacing/2 (injector) and x = +well_spacing/2 (producer), both
    at y = 0, spanning the full z range (vertical wells).

    Returns three (N, nx, ny, nz) float32 arrays: well_map (+1 injector,
    -1 producer, 0 elsewhere), dist_inj, dist_prod (horizontal distance in
    meters from each cell to each well).

    well_map marks exactly ONE cell per well (not the whole vertical
    column), at the middle z-layer. This is deliberate: the real well only
    physically exists over a short (~50m) screen interval near the
    reservoir's mid-depth, not the full exported z-range (much of which is
    cap rock). Marking the whole column would let the model-side well
    read-off pick up invalid/placeholder cap-rock cells whenever a
    thin-reservoir scenario's valid region doesn't reach the column's
    edges. The middle z-layer is a safe, scenario-independent choice: the
    cap-rock margin is always symmetric top/bottom (top_layer_H ==
    bottom_layer_H in the COMSOL template), so the middle layer is
    guaranteed to fall inside the valid reservoir region regardless of
    reservoir thickness H. (dist_inj/dist_prod are still broadcast across
    all z, matching PCNO's own "distance to well" feature convention --
    only the point-selection well_map indicator needed this fix.)
    """
    n = well_spacing.shape[0]
    nx, ny, nz = len(x_m), len(y_m), len(z_m)
    z_mid = nz // 2
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')  # (nx, ny)

    well_map = np.zeros((n, nx, ny, nz), dtype=np.float32)
    dist_inj = np.zeros((n, nx, ny, nz), dtype=np.float32)
    dist_prod = np.zeros((n, nx, ny, nz), dtype=np.float32)

    for i in range(n):
        L = float(well_spacing[i])
        x_inj, x_prod = -L / 2.0, L / 2.0

        d_inj_xy = np.sqrt((X - x_inj) ** 2 + Y ** 2)   # (nx, ny)
        d_prod_xy = np.sqrt((X - x_prod) ** 2 + Y ** 2)  # (nx, ny)

        inj_ix, inj_iy = np.unravel_index(np.argmin(d_inj_xy), d_inj_xy.shape)
        prod_ix, prod_iy = np.unravel_index(np.argmin(d_prod_xy), d_prod_xy.shape)

        well_map[i, inj_ix, inj_iy, z_mid] = 1.0
        well_map[i, prod_ix, prod_iy, z_mid] = -1.0
        dist_inj[i] = d_inj_xy[:, :, None]
        dist_prod[i] = d_prod_xy[:, :, None]

    return well_map, dist_inj, dist_prod


# ---------------------------------------------------------------------------
# Step 3 (Issue 3): validity masks for the cap-rock domain boundary
# ---------------------------------------------------------------------------
def compute_valid_masks(y_fields, output_field_channel_names):
    """
    T_K and p_MPa are only ~73%/~47% finite dataset-wide -- the rest is cap
    rock above/below the reservoir, where COMSOL's flow physics (pressure)
    isn't solved at all, and heat-transfer physics (temperature) is solved
    over a slightly smaller region than the full grid. Confirmed this is a
    real depth/geology effect (correlates ~-0.98/-0.99 with reservoir
    thickness H), not noise or "distance from well" relevance.

    mask = True means "trust this cell for loss/metrics", False means
    "cap rock / outside the solved domain, exclude entirely".
    """
    t_idx = list(output_field_channel_names).index('T_K')
    p_idx = list(output_field_channel_names).index('p_MPa')
    mask_t = np.isfinite(y_fields[:, t_idx])
    mask_p = np.isfinite(y_fields[:, p_idx])
    return mask_t, mask_p


# ---------------------------------------------------------------------------
# Step 4 (Issue 2): detect and repair implausible T_K outlier cells
# ---------------------------------------------------------------------------
def repair_temperature_outliers(T, mask_t, t_min=T_OUTLIER_MIN, t_max=T_OUTLIER_MAX, max_passes=MAX_REPAIR_PASSES):
    """
    A small number of cells (found: 2,913 out of ~442 million) have
    physically impossible T_K values (as extreme as -462.65K). Each one
    checked by hand was an isolated glitch surrounded by smooth, valid
    data, so we repair (average from neighbors) rather than mask them --
    unlike the cap-rock cells above, there IS good data right next to these
    to draw a reliable estimate from.

    IMPORTANT: this must run in multiple passes, not just one. Some outlier
    cells are directly adjacent to *other* outlier cells (confirmed by
    checking actual coordinates during development), so a cell surrounded
    entirely by other bad cells has zero valid neighbors on the first pass.
    Each pass, newly-repaired cells become valid neighbors for the next
    pass, so clusters resolve from the outside in. Anything still
    unresolved after max_passes is returned separately so the caller can
    exclude it from the mask instead of keeping a corrupted value.

    Returns (repaired_T, n_outliers_found, unresolved_locations).
    """
    T = T.copy()
    outlier = mask_t & ((T < t_min) | (T > t_max))
    n_outliers = int(outlier.sum())

    nx, ny, nz = T.shape[1], T.shape[2], T.shape[3]
    remaining = outlier.copy()

    for _ in range(max_passes):
        locations = np.argwhere(remaining)
        if len(locations) == 0:
            break
        fixed_any = False
        for si, xi, yi, zi, ti in locations:
            neighbor_vals = []
            for dx, dy, dz in FACE_NEIGHBOR_OFFSETS:
                nxi, nyi, nzi = xi + dx, yi + dy, zi + dz
                if 0 <= nxi < nx and 0 <= nyi < ny and 0 <= nzi < nz:
                    if mask_t[si, nxi, nyi, nzi, ti] and not remaining[si, nxi, nyi, nzi, ti]:
                        neighbor_vals.append(T[si, nxi, nyi, nzi, ti])
            if neighbor_vals:
                T[si, xi, yi, zi, ti] = np.mean(neighbor_vals)
                remaining[si, xi, yi, zi, ti] = False
                fixed_any = True
        if not fixed_any:
            break  # no progress possible -- remaining cells are fully isolated

    unresolved_locations = np.argwhere(remaining)
    return T, n_outliers, unresolved_locations


def fill_invalid_with_placeholder(field, mask, placeholder=0.0):
    """Replace invalid cells with a harmless placeholder so tensor math
    doesn't break downstream. These cells are excluded from loss via
    `mask` regardless -- the placeholder value itself is never scored."""
    return np.where(mask, field, placeholder).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: normalization statistics
# ---------------------------------------------------------------------------
def compute_normalization_stats(data, mask_t, mask_p, repaired_T, well_map, dist_inj, dist_prod):
    """Per-channel mean/std. Static inputs and scalar params are 100%
    finite already (no masking needed); field outputs use the masks so
    cap-rock placeholder cells don't skew the statistics.

    Also computes stats for the 3 derived well-location channels
    (well_map/dist_inj/dist_prod), from their own individual arrays rather
    than the concatenated 8-channel combined_static_fields array -- building
    that combined array before this point adds a large extra allocation
    right when T_K's std() computation needs its own temporary memory too,
    which caused a real MemoryError during development. Using the smaller
    individual arrays avoids that."""
    stats = {}

    field_channel_names = list(data['field_channel_names'])
    for ci, name in enumerate(field_channel_names):
        ch = data['x_static_fields'][:, ci]
        stats[f'static__{name}__mean'] = float(ch.mean())
        stats[f'static__{name}__std'] = float(ch.std())

    for name, arr in [('well_map', well_map), ('dist_inj', dist_inj), ('dist_prod', dist_prod)]:
        stats[f'static__{name}__mean'] = float(arr.mean())
        stats[f'static__{name}__std'] = float(arr.std())

    param_names = list(data['parameter_names'])
    for ci, name in enumerate(param_names):
        col = data['x_params'][:, ci]
        stats[f'param__{name}__mean'] = float(col.mean())
        stats[f'param__{name}__std'] = float(col.std())

    p_idx = list(data['output_field_channel_names']).index('p_MPa')
    P = data['y_fields'][:, p_idx]

    t_vals = repaired_T[mask_t]
    p_vals = P[mask_p]
    stats['field__T_K__mean'] = float(t_vals.mean())
    stats['field__T_K__std'] = float(t_vals.std())
    stats['field__p_MPa__mean'] = float(p_vals.mean())
    stats['field__p_MPa__std'] = float(p_vals.std())

    well_target_names = list(data['well_target_names'])
    for ci, name in enumerate(well_target_names):
        col = data['y_wells'][:, :, ci]
        stats[f'well__{name}__mean'] = float(col.mean())
        stats[f'well__{name}__std'] = float(col.std())

    return stats


# ---------------------------------------------------------------------------
# Step 6: stratified train/val/test split
# ---------------------------------------------------------------------------
def load_design_focus(sample_ids, summary_csv_path=None):
    """Look up each scenario's design_focus label (none / anomaly_targeted /
    lhs_explore) from the original summary CSV, joined on real_id."""
    df = pd.read_csv(summary_csv_path or SUMMARY_CSV_PATH)
    df['sample_id'] = df['real_id'].apply(lambda i: f'real_{int(i):05d}')
    focus_map = dict(zip(df['sample_id'], df['design_focus'].fillna('none')))

    missing = [sid for sid in sample_ids if sid not in focus_map]
    if missing:
        raise ValueError(f'{len(missing)} sample_ids not found in summary CSV, e.g. {missing[:5]}')

    return np.array([focus_map[sid] for sid in sample_ids])


def build_stratified_split(sample_ids, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    """
    Fixed, reproducible split, stratified by design_focus rather than a
    plain random split. The dataset was built iteratively -- later
    scenarios were deliberately sampled near parameter-space "anomalies"
    flagged by an active-learning process, not drawn uniformly at random
    like the earlier scenarios. Stratifying keeps each split proportionally
    representative of all design phases, not skewed toward whichever
    happened to fall in random shuffling.
    """
    focus = load_design_focus(sample_ids)
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for group in np.unique(focus):
        group_idx = np.where(focus == group)[0]
        rng.shuffle(group_idx)
        n = len(group_idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_idx.extend(group_idx[:n_train])
        val_idx.extend(group_idx[n_train:n_train + n_val])
        test_idx.extend(group_idx[n_train + n_val:])

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(sample_ids)

    return train_idx, val_idx, test_idx, focus


# ---------------------------------------------------------------------------
# Step 7 + 8: run everything, save, then reload and verify the SAVED file
# ---------------------------------------------------------------------------
def build_prepared_dataset():
    """Runs the full dataset-preparation pipeline end to end: loads the raw
    assembled dataset, derives well-location channels, builds T_K/p_MPa
    validity masks, repairs T_K outlier cells, computes normalization
    statistics, builds the stratified train/val/test split, and saves the
    consolidated prepared dataset to OUT_PATH.

    Takes no arguments and returns nothing; progress for each step is
    printed to stdout as it runs.
    """
    print('=' * 70)
    print('STEP 1: Loading raw assembled dataset...')
    print('=' * 70)
    data = load_dataset(RAW_NPZ_PATH)
    sample_ids = list(data['sample_id'])
    n = len(sample_ids)
    print(f'  {n} scenarios loaded')

    print()
    print('STEP 2 (Issue 1): deriving well-location channels...')
    param_names = list(data['parameter_names'])
    well_spacing = data['x_params'][:, param_names.index('well_spacing')]
    well_map, dist_inj, dist_prod = build_well_location_channels(
        well_spacing, data['x_m'], data['y_m'], data['z_m'])
    print(f'  well_map, dist_inj, dist_prod: shape {well_map.shape}')

    print()
    print('STEP 3 (Issue 3): building validity masks...')
    mask_t, mask_p = compute_valid_masks(data['y_fields'], data['output_field_channel_names'])
    print(f'  T_K finite: {mask_t.mean()*100:.2f}%   p_MPa finite: {mask_p.mean()*100:.2f}%')

    print()
    print('STEP 4 (Issue 2): detecting and repairing T_K outliers (iterative)...')
    t_idx = list(data['output_field_channel_names']).index('T_K')
    T_raw = data['y_fields'][:, t_idx]
    T_repaired, n_outliers, unresolved = repair_temperature_outliers(T_raw, mask_t)
    n_unresolved = len(unresolved)
    print(f'  {n_outliers} outlier cells found; {n_outliers - n_unresolved} repaired via neighbor averaging')
    if n_unresolved:
        print(f'  {n_unresolved} cells unresolvable even after iterative passes -- excluding from mask_T')
        for si, xi, yi, zi, ti in unresolved:
            mask_t[si, xi, yi, zi, ti] = False

    print()
    print('  Filling remaining invalid (cap-rock + unresolved-outlier) cells with placeholder 0...')
    T_final = fill_invalid_with_placeholder(T_repaired, mask_t, placeholder=0.0)
    p_idx = list(data['output_field_channel_names']).index('p_MPa')
    P_final = fill_invalid_with_placeholder(data['y_fields'][:, p_idx], mask_p, placeholder=0.0)

    print()
    print('STEP 5: computing normalization statistics...')
    stats = compute_normalization_stats(data, mask_t, mask_p, T_repaired, well_map, dist_inj, dist_prod)
    print(f'  {len(stats)} stats computed')

    print()
    print('STEP 6: building stratified train/val/test split...')
    train_idx, val_idx, test_idx, design_focus = build_stratified_split(sample_ids)
    print(f'  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

    combined_static_fields = np.concatenate([
        data['x_static_fields'],
        well_map[:, None], dist_inj[:, None], dist_prod[:, None],
    ], axis=1)  # (N, 5+3=8, nx, ny, nz)
    static_channel_names = list(data['field_channel_names']) + ['well_map', 'dist_inj', 'dist_prod']

    print()
    print(f'STEP 7: saving prepared dataset to {OUT_PATH} ...')
    np.savez_compressed(
        OUT_PATH,
        x_static_fields=combined_static_fields.astype(np.float32),
        static_channel_names=np.array(static_channel_names),
        x_params=data['x_params'],
        parameter_names=data['parameter_names'],
        T_K=T_final, p_MPa=P_final,
        mask_T=mask_t, mask_P=mask_p,
        y_wells=data['y_wells'], well_target_names=data['well_target_names'],
        x_m=data['x_m'], y_m=data['y_m'], z_m=data['z_m'],
        field_time_years=data['field_time_years'], well_time_years=data['well_time_years'],
        sample_id=data['sample_id'], design_focus=design_focus,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        n_outliers_repaired=n_outliers,
        stats_keys=np.array(list(stats.keys())),
        stats_values=np.array(list(stats.values())),
    )
    print('  Done.')


def verify_prepared_dataset():
    """
    Reload the file that was just saved to disk -- fresh, independent of
    the in-memory objects from build_prepared_dataset() above -- and check
    it directly. This distinction matters: during development, the build
    script ran without any errors while a real bug (151 unrepaired outlier
    cells) was sitting in the saved output. Only reloading and checking the
    actual file caught it.
    """
    print()
    print('=' * 70)
    print('VERIFICATION: reloading the saved file fresh and checking it...')
    print('=' * 70)
    d = np.load(OUT_PATH, allow_pickle=True)
    n = len(d['sample_id'])

    ok = True

    def check(label, condition):
        """Prints one '[OK] label' or '[FAILED] label' verification line
        depending on whether `condition` is True, and flips the enclosing
        `ok` flag to False on any failure. label: description of the check
        being reported; condition: bool result of that check. Returns
        nothing."""
        nonlocal ok
        status = 'OK' if condition else 'FAILED'
        if not condition:
            ok = False
        print(f'  [{status}] {label}')

    print(f'\n{n} scenarios in saved file.')

    check('No NaNs in T_K', np.isnan(d['T_K']).sum() == 0)
    check('No NaNs in p_MPa', np.isnan(d['p_MPa']).sum() == 0)
    check('No NaNs in x_static_fields', np.isnan(d['x_static_fields']).sum() == 0)
    check('No NaNs in y_wells', np.isnan(d['y_wells']).sum() == 0)

    t_frac = d['mask_T'].mean()
    p_frac = d['mask_P'].mean()
    print(f'  mask_T valid fraction: {t_frac*100:.2f}%  (expect ~72.8%)')
    print(f'  mask_P valid fraction: {p_frac*100:.2f}%  (expect ~47.5%)')
    check('mask_T fraction in expected range', 0.70 <= t_frac <= 0.76)
    check('mask_P fraction in expected range', 0.44 <= p_frac <= 0.50)

    t_placeholder = d['T_K'][~d['mask_T']]
    check('Placeholder cells (T_K) are exactly 0', np.all(t_placeholder == 0.0))

    t_valid = d['T_K'][d['mask_T']]
    n_still_bad = int(((t_valid < T_OUTLIER_MIN) | (t_valid > T_OUTLIER_MAX)).sum())
    print(f'  Cells still outside [{T_OUTLIER_MIN},{T_OUTLIER_MAX}] within valid region: {n_still_bad}')
    check('No remaining T_K outliers in valid region', n_still_bad == 0)

    stats = dict(zip(d['stats_keys'], d['stats_values']))
    static_names = list(d['static_channel_names'])
    missing_stats = [name for name in static_names
                      if f'static__{name}__mean' not in stats or f'static__{name}__std' not in stats]
    check(f'All {len(static_names)} static channels have normalization stats (incl. derived well-location ones)',
          len(missing_stats) == 0)
    if missing_stats:
        print(f'    Missing stats for: {missing_stats}')

    train_idx, val_idx, test_idx = d['train_idx'], d['val_idx'], d['test_idx']
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    check('Split sizes sum to n', len(train_idx) + len(val_idx) + len(test_idx) == n)
    check('Split indices unique (no overlap)', len(np.unique(all_idx)) == n)
    check('Split covers full 0..n-1 range', set(all_idx.tolist()) == set(range(n)))

    print()
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED -- investigate before trusting this file')
    return ok


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build the prepared training dataset from the raw COMSOL dataset.')
    parser.add_argument('--raw-npz', default=RAW_NPZ_PATH, help='raw assembled dataset (.npz)')
    parser.add_argument('--summary-csv', default=SUMMARY_CSV_PATH, help='design summary table (.csv)')
    parser.add_argument('--out', default=OUT_PATH, help='where to write the prepared dataset (.npz)')
    args = parser.parse_args()
    RAW_NPZ_PATH, SUMMARY_CSV_PATH, OUT_PATH = args.raw_npz, args.summary_csv, args.out
    build_prepared_dataset()
    passed = verify_prepared_dataset()
    if not passed:
        raise SystemExit('Verification failed -- see [FAILED] lines above.')
