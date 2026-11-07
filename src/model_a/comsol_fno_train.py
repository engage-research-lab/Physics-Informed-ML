"""
Real training script for ComsolFnoModel -- uses the actual train/val/test
split from Phase 1, a validation loop, checkpointing + resume, incrementally
flushed logging, early stopping, and relative-error (RE) metrics matching
the project's actual accuracy target (<=2% RE).

Mirrors PCNO_Train_4D.py's separation from PCNO_Model_4D.py: this file only
handles data + training mechanics; the architecture itself lives in
comsol_fno_model.py. Checkpoint/resume/logging/early-stopping patterns
follow pcno_lite/train.py's established conventions.

--- Time-fix + FiLM-fix architecture folded in as canonical (2026-07-28) --
This was first trained/validated separately in a
controlled A/B test against the time-fix-only architecture (see
comsol_fno_model.py's docstring for the full root-cause writeup on both
fixes). Result: val_re_t 3.61%->0.25%, val_re_p 0.195%->0.079%, val_re_wells
~unchanged (~3.49%) -- temperature and pressure both now comfortably beat
the <=2% target. Folded in here as the canonical Model A once confirmed.
"""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from comsol_fno_model import ComsolFnoModel, masked_mse, well_output_mse, masked_relative_error, relative_error

# Fixed seed for reproducibility (matching PCNO_Model_4D.py's own
# torch.manual_seed(0)/np.random.seed(0) convention). Overridable via
# TRAIN_SEED env var (default 0, unchanged behavior).
SEED = int(os.environ.get('TRAIN_SEED', 0))
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this script's own location, so the whole
# thing works unchanged regardless of where the folder (this file +
# comsol_fno_model.py + fno_lhs_v1_prepared_phase1.npz, all together) gets
# placed -- e.g. on ARC's filesystem, wherever that ends up being.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Set GEOTHERMAL_DATA_PATH to use a dataset stored somewhere else.
PREPARED_DATA_PATH = os.environ.get('GEOTHERMAL_DATA_PATH') or os.path.join(_REPO_ROOT, 'data', 'fno_lhs_v1_prepared_phase1.npz')
_CKPT_SUBDIR = 'model_a' if SEED == 0 else f'model_a_seed{SEED}'
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, 'checkpoints', _CKPT_SUBDIR)
RESUME_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'resume.pth')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
LOG_PATH = os.path.join(CHECKPOINT_DIR, 'train_log.jsonl')

# Step=5 uniform time grid decision: keep only the field snapshots at years
# 0,5,10,...,40 (a pure subset of the 11 real snapshots -- years 1 and 2 are
# dropped, not interpolated). See findings docs for why.
UNIFORM_TIME_YEARS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


# ---------------------------------------------------------------------------
# Dataset -- uses the REAL train_idx/val_idx/test_idx split from Phase 1
# (the overfit-diagnostic version only ever used indices [0, 1])
# ---------------------------------------------------------------------------
class ComsolFnoDataset(Dataset):
    """
    Wraps the Phase 1 prepared .npz for one split ('train', 'val', or
    'test' -- uses the stratified indices Phase 1 already built, not an
    arbitrary index range). Applies, per sample:
      - selection of the 9 uniform-time-grid field snapshots (dropping
        years 1 and 2, no interpolation -- see UNIFORM_TIME_YEARS above)
      - per-channel normalization using the Phase 1 stats
      - dropping the constant `well_radius` scalar (would divide-by-zero
        during normalization otherwise -- verified std ~= 7e-9)
      - deriving nothing extra for dP_MPa here -- it's just not predicted;
        computed from P_inj_MPa - P_prod_MPa at evaluation time instead
    """
    def __init__(self, npz_path, split):
        """
        IMPORTANT: subsets every per-scenario array down to just this
        split's rows immediately after loading, rather than keeping the
        full 1,535-scenario arrays around and indexing into them lazily.
        An earlier version kept the full arrays in every Dataset instance
        regardless of split -- meaning creating train_ds AND val_ds held
        two full independent ~1.86GB copies of the entire dataset
        simultaneously (train only needs 1,074 of 1,535 scenarios, val
        only 229), which caused sporadic, seemingly-nonsensical small
        allocation failures once memory got fragmented. Caught by actually
        trying to instantiate both at once, not by code review.
        """
        assert split in ('train', 'val', 'test')
        data = np.load(npz_path, allow_pickle=True)
        split_idx = data[f'{split}_idx']
        self.n = len(split_idx)
        self.split = split

        field_time_years = list(data['field_time_years'])
        self.time_idx = [field_time_years.index(y) for y in UNIFORM_TIME_YEARS]

        self.x_static_fields = data['x_static_fields'][split_idx]  # (n_split, 8, 60, 30, 20)
        self.static_channel_names = list(data['static_channel_names'])

        param_names = list(data['parameter_names'])
        self.param_keep_idx = [i for i, n in enumerate(param_names) if n != 'well_radius']
        self.param_names = [param_names[i] for i in self.param_keep_idx]
        self.x_params = data['x_params'][split_idx][:, self.param_keep_idx]

        self.T_K = data['T_K'][split_idx]       # (n_split, 60, 30, 20, 11)
        self.p_MPa = data['p_MPa'][split_idx]
        self.mask_T = data['mask_T'][split_idx]
        self.mask_P = data['mask_P'][split_idx]
        self.y_wells = data['y_wells'][split_idx]  # (n_split, 201, 4) -- only cols 0:3 are predicted targets

        stats = dict(zip(data['stats_keys'], data['stats_values']))
        self.stats = {k: float(v) for k, v in stats.items()}

        self.well_map_idx = self.static_channel_names.index('well_map')

    def __len__(self):
        """Number of scenarios in this split.

        Returns: int -- len(split_idx), i.e. self.n.
        """
        return self.n

    def _norm_static(self, arr):
        """
        Per-channel z-score normalization of one scenario's static spatial
        fields, using the mean/std stored in self.stats (Phase 1's
        precomputed per-channel statistics). A channel whose std is ~0
        (below 1e-6) is left unscaled (divided by 1) rather than blown up.

        arr: (n_static_channels, X, Y, Z) raw static fields for one scenario.
        Returns: array of the same shape, normalized.
        """
        out = np.empty_like(arr)
        for ci, name in enumerate(self.static_channel_names):
            mean = self.stats[f'static__{name}__mean']
            std = self.stats[f'static__{name}__std']
            out[ci] = (arr[ci] - mean) / (std if std > 1e-6 else 1.0)
        return out

    def _norm_params(self, arr):
        """
        Per-parameter z-score normalization of one scenario's scalar
        parameters, using the mean/std stored in self.stats. A parameter
        whose std is ~0 is left unscaled (divided by 1) rather than blown up.

        arr: (n_params,) raw scalar parameters for one scenario.
        Returns: array of the same shape, normalized.
        """
        out = np.empty_like(arr)
        for i, name in enumerate(self.param_names):
            mean = self.stats[f'param__{name}__mean']
            std = self.stats[f'param__{name}__std']
            out[i] = (arr[i] - mean) / (std if std > 1e-6 else 1.0)
        return out

    def __getitem__(self, idx):
        """
        Builds the full training sample for scenario `idx`: normalizes the
        static fields and scalar params, pulls out the well-location
        channel raw (unnormalized, used as a cell selector rather than a
        normalized feature), subsets the field snapshots down to the 9
        uniform-time-grid years, and z-score-normalizes the T_K/p_MPa fields
        and the three predicted well-output series (T_prod_K, P_inj_MPa,
        P_prod_MPa).

        idx: index into this split's already-subsetted arrays.
        Returns: dict with keys 'x_static' (n_static_channels, X, Y, Z),
        'params' (n_params,), 'well_map' (X, Y, Z), 'T_K'/'p_MPa'
        (X, Y, Z, 9) normalized fields, 'mask_T'/'mask_P' (X, Y, Z, 9), and
        'wells' (201, 3) normalized [T_prod_K, P_inj_MPa, P_prod_MPa].
        """
        # `idx` indexes directly into the already-split-subsetted arrays
        # (no remapping through a separate indices array needed anymore).
        x_static = self._norm_static(self.x_static_fields[idx]).astype(np.float32)
        params = self._norm_params(self.x_params[idx]).astype(np.float32)
        well_map = self.x_static_fields[idx, self.well_map_idx].astype(np.float32)  # raw, used as a cell selector, not a normalized feature here

        T_K = self.T_K[idx][..., self.time_idx]
        p_MPa = self.p_MPa[idx][..., self.time_idx]
        mask_T = self.mask_T[idx][..., self.time_idx]
        mask_P = self.mask_P[idx][..., self.time_idx]

        t_mean, t_std = self.stats['field__T_K__mean'], self.stats['field__T_K__std']
        p_mean, p_std = self.stats['field__p_MPa__mean'], self.stats['field__p_MPa__std']
        T_K_norm = ((T_K - t_mean) / t_std).astype(np.float32)
        p_MPa_norm = ((p_MPa - p_mean) / p_std).astype(np.float32)

        wells = self.y_wells[idx, :, :3].astype(np.float32)  # T_prod_K, P_inj_MPa, P_prod_MPa
        wn_mean_std = [(self.stats['well__T_prod_K__mean'], self.stats['well__T_prod_K__std']),
                        (self.stats['well__P_inj_MPa__mean'], self.stats['well__P_inj_MPa__std']),
                        (self.stats['well__P_prod_MPa__mean'], self.stats['well__P_prod_MPa__std'])]
        wells_norm = np.stack([(wells[:, c] - m) / s for c, (m, s) in enumerate(wn_mean_std)], axis=1).astype(np.float32)

        return {
            'x_static': x_static, 'params': params, 'well_map': well_map,
            'T_K': T_K_norm, 'p_MPa': p_MPa_norm,
            'mask_T': mask_T.astype(np.float32), 'mask_P': mask_P.astype(np.float32),
            'wells': wells_norm,
        }


def collate(batch):
    """
    Custom collate_fn for DataLoader: stacks each key of the per-sample
    dicts returned by ComsolFnoDataset.__getitem__ across the batch into a
    single tensor, converting from numpy to torch in the process.

    batch: list of per-sample dicts (as returned by __getitem__).
    Returns: dict with the same keys, each now a batched torch tensor with
    a leading batch dimension.
    """
    return {key: torch.from_numpy(np.stack([b[key] for b in batch])) for key in batch[0]}


def denormalize(x, mean, std):
    """
    Inverts z-score normalization, mapping a normalized tensor back to real
    physical units.

    x: normalized tensor of any shape. mean/std: the scalars used to
    normalize it.
    Returns: x * std + mean, same shape as x.
    """
    return x * std + mean


# ---------------------------------------------------------------------------
# Early stopping (matching pcno_lite/train.py's EarlyStopState)
# ---------------------------------------------------------------------------
@dataclass
class EarlyStopState:
    """Tracks early-stopping state across evaluations: the best validation
    metric seen so far, the epoch it occurred at, and how many evaluations
    have passed since the last improvement. best_val is the lowest val
    metric seen; best_epoch is the epoch it occurred at; evals_since_improve
    is the count of evaluations since the last improvement."""
    best_val: float = float('inf')
    best_epoch: int = -1
    evals_since_improve: int = 0

    def update(self, val_metric, epoch):
        """
        Records one new validation-metric observation: if it improves on
        best_val, updates best_val/best_epoch and resets
        evals_since_improve to 0; otherwise increments
        evals_since_improve.

        val_metric: this evaluation's metric value (lower is better).
        epoch: the epoch it was measured at.
        Returns: bool, True if this was a new best.
        """
        if val_metric < self.best_val:
            self.best_val = val_metric
            self.best_epoch = epoch
            self.evals_since_improve = 0
            return True
        self.evals_since_improve += 1
        return False

    def should_stop(self, patience):
        """
        Whether early stopping should trigger: True once
        evals_since_improve has reached patience.

        patience: number of evaluations without improvement to tolerate
        before stopping.
        Returns: bool.
        """
        return self.evals_since_improve >= patience


@dataclass
class FitResult:
    """Holds the full outcome of one train() run. history is the list of
    per-epoch record dicts (as logged to train_log.jsonl); best_epoch and
    best_val_metric are the epoch/value of the best validation result seen;
    stopped_early records whether early stopping triggered before n_epochs
    completed."""
    history: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_metric: float = float('inf')
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# One epoch (train or eval) -- returns averaged metrics including both the
# normalized-space losses (what's optimized) and denormalized RE% (what
# the project's actual <=2% target is measured in)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, stats, device, optimizer=None):
    """
    Runs one epoch over `loader`: for each batch, runs the model forward,
    computes the masked T_K/p_MPa MSE losses plus the well-output MSE, and
    -- when `optimizer` is given -- backpropagates and steps the optimizer.
    Also computes denormalized relative-error (RE%) metrics for T_K, p_MPa,
    and wells, matching the project's <=2% RE accuracy target. Train vs.
    eval mode is inferred from whether `optimizer` is provided.

    model: ComsolFnoModel. loader: DataLoader yielding collated batches.
    stats: per-channel/per-param mean/std dict (from the dataset). device:
    torch device. optimizer: torch optimizer, or None for a
    validation/eval pass (runs under torch.no_grad() in that case).
    Returns: dict of the epoch's averaged metrics (loss, loss_t, loss_p,
    loss_wells, re_t, re_p, re_wells).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    t_mean, t_std = stats['field__T_K__mean'], stats['field__T_K__std']
    p_mean, p_std = stats['field__p_MPa__mean'], stats['field__p_MPa__std']
    well_mean_std = [
        (stats['well__T_prod_K__mean'], stats['well__T_prod_K__std']),
        (stats['well__P_inj_MPa__mean'], stats['well__P_inj_MPa__std']),
        (stats['well__P_prod_MPa__mean'], stats['well__P_prod_MPa__std']),
    ]

    totals = {'loss': 0.0, 'loss_t': 0.0, 'loss_p': 0.0, 'loss_wells': 0.0,
              're_t': 0.0, 're_p': 0.0, 're_wells': 0.0}
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x_static = batch['x_static'].to(device)
            params = batch['params'].to(device)
            well_map = batch['well_map'].to(device)
            target_T = batch['T_K'].to(device)
            target_P = batch['p_MPa'].to(device)
            mask_T = batch['mask_T'].to(device)
            mask_P = batch['mask_P'].to(device)
            target_wells = batch['wells'].to(device)

            if is_train:
                optimizer.zero_grad()

            out = model(x_static, params, well_map)
            loss_t = masked_mse(out['T_K'], target_T, mask_T)
            loss_p = masked_mse(out['p_MPa'], target_P, mask_P)
            loss_w = well_output_mse(out['wells'], target_wells)
            loss = loss_t + loss_p + loss_w

            if is_train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_T_real = denormalize(out['T_K'], t_mean, t_std)
                target_T_real = denormalize(target_T, t_mean, t_std)
                pred_P_real = denormalize(out['p_MPa'], p_mean, p_std)
                target_P_real = denormalize(target_P, p_mean, p_std)
                re_t = masked_relative_error(pred_T_real, target_T_real, mask_T)
                re_p = masked_relative_error(pred_P_real, target_P_real, mask_P)

                pred_wells_real = torch.stack(
                    [denormalize(out['wells'][:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
                target_wells_real = torch.stack(
                    [denormalize(target_wells[:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
                re_w = relative_error(pred_wells_real, target_wells_real)

            totals['loss'] += loss.item()
            totals['loss_t'] += loss_t.item()
            totals['loss_p'] += loss_p.item()
            totals['loss_wells'] += loss_w.item()
            totals['re_t'] += re_t.item()
            totals['re_p'] += re_p.item()
            totals['re_wells'] += re_w.item()
            n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(model, optimizer, epoch, path, stopper=None):
    """
    stopper: the EarlyStopState to persist alongside model/optimizer weights.
    BUG FIXED 2026-07-19: stopper state was never saved, so every resume
    started a fresh EarlyStopState (best_val=inf) with no memory of the
    true best found before the interruption. A real PINN training run hit
    this: it found its true best at epoch 13 (val_re_combined=3.6456), got
    interrupted, resumed at epoch 31, and because the fresh stopper had
    best_val=inf, epoch 31's WORSE value (3.8047) was treated as a new
    "best" and overwrote the actual best.pth (epoch 13's weights, now
    unrecoverable) -- then trained 20 more wasted epochs chasing that false
    target before patience ran out. Now always saved so a resume can
    restore the true historical best instead of resetting it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if stopper is not None:
        ckpt['stopper'] = {'best_val': stopper.best_val, 'best_epoch': stopper.best_epoch,
                            'evals_since_improve': stopper.evals_since_improve}
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(
    n_epochs=200,
    batch_size=2,  # SAFE DEFAULT for a 4GB GPU (locally, batch_size>2 caused
                   # catastrophic slowdowns from VRAM oversubscription -- see
                   # findings docs). ARC's GPUs likely have far more headroom;
                   # increase this once the actual node's VRAM is confirmed.
    lr=1e-3,
    weight_decay=1e-4,  # matching PCNO's own AdamW(weight_decay=1e-4) -- the
                        # overfit diagnostics intentionally used plain Adam
                        # with no decay (decay works against deliberate
                        # overfitting); real training should regularize.
    width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
    unet_base_channels=None, n_layers=4,
    eval_every=1,
    early_stop_patience=8,
    min_epochs_before_stop=5,
    resume=True,
):
    """
    Runs the full training loop for ComsolFnoModel: builds the train/val
    datasets and loaders, constructs the model and AdamW optimizer,
    optionally resumes from a saved checkpoint, then iterates epochs
    running run_epoch on the training set (every epoch) and the validation
    set (every `eval_every` epochs), saving a resume checkpoint every epoch
    and a best checkpoint whenever validation improves, logging each
    epoch's record to train_log.jsonl, and stopping early once the worst of
    val_re_t/val_re_p/val_re_wells hasn't improved for
    `early_stop_patience` evaluations (after at least
    `min_epochs_before_stop` epochs have run).

    n_epochs: maximum number of epochs to train.
    batch_size: scenarios per batch.
    lr: AdamW learning rate.
    weight_decay: AdamW weight decay.
    width, width2, modes1, modes2, modes3, modes4, unet_base_channels,
    n_layers: ComsolFnoModel architecture hyperparameters, passed straight
    through to its constructor.
    eval_every: run a validation pass every this many epochs (epoch 1 is
    always evaluated regardless).
    early_stop_patience: number of validation evaluations without
    improvement in the combined val metric before stopping.
    min_epochs_before_stop: minimum epoch count before early stopping is
    allowed to trigger.
    resume: if True and a resume checkpoint exists at RESUME_CKPT_PATH,
    continues training from it (including restoring EarlyStopState)
    instead of starting from epoch 1.

    Returns: a FitResult with the full per-epoch history and the best
    epoch/metric found.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_dataset = ComsolFnoDataset(PREPARED_DATA_PATH, split='train')
    val_dataset = ComsolFnoDataset(PREPARED_DATA_PATH, split='val')
    print(f'Train: {len(train_dataset)} scenarios   Val: {len(val_dataset)} scenarios')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = ComsolFnoModel(width=width, width2=width2, modes1=modes1, modes2=modes2,
                            modes3=modes3, modes4=modes4, unet_base_channels=unet_base_channels,
                            n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameter count: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    stopper = EarlyStopState()
    if resume and os.path.exists(RESUME_CKPT_PATH):
        ckpt = torch.load(RESUME_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if 'stopper' in ckpt:
            stopper.best_val = ckpt['stopper']['best_val']
            stopper.best_epoch = ckpt['stopper']['best_epoch']
            stopper.evals_since_improve = ckpt['stopper']['evals_since_improve']
            print(f'Resumed from checkpoint at epoch {ckpt["epoch"]}, continuing from epoch {start_epoch} '
                  f'(early-stop state restored: best={stopper.best_val:.3f} at epoch {stopper.best_epoch}, '
                  f'{stopper.evals_since_improve} evals since improve)')
        else:
            # Checkpoint predates this fix -- no stopper state to restore.
            # Not fatal, but the early-stopping clock effectively resets,
            # same as the bug this fix addresses (see save_checkpoint's
            # docstring) -- flagged loudly so it isn't silently repeated.
            print(f'Resumed from checkpoint at epoch {ckpt["epoch"]}, continuing from epoch {start_epoch} '
                  f'-- WARNING: this checkpoint has no saved early-stop state (predates the 2026-07-19 fix), '
                  f'so best_val/best_epoch cannot be restored and will reset. If a better epoch existed '
                  f'before this checkpoint was written, it will not be recovered.')

    result = FitResult()
    run_start = time.time()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Append mode if resuming, so we don't wipe the log history on restart.
    log_mode = 'a' if (resume and start_epoch > 1) else 'w'
    with open(LOG_PATH, log_mode) as log_file:
        if log_mode == 'w':
            log_file.write(json.dumps({'event': 'start', 'wall_time': run_start, 'n_params': n_params}) + '\n')
            log_file.flush()

        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(model, train_loader, train_dataset.stats, device, optimizer=optimizer)
            epoch_seconds = time.time() - epoch_start

            record = {'epoch': epoch, 'epoch_seconds': round(epoch_seconds, 2),
                      **{f'train_{k}': v for k, v in train_metrics.items()}}

            # Resume checkpoint saved every epoch, independent of validation
            # cadence -- so an interrupted ARC job can always continue from
            # the last completed epoch, not just the last evaluated one.
            save_checkpoint(model, optimizer, epoch, RESUME_CKPT_PATH, stopper=stopper)

            did_eval = epoch % eval_every == 0 or epoch == 1
            if did_eval:
                val_metrics = run_epoch(model, val_loader, val_dataset.stats, device, optimizer=None)
                record.update({f'val_{k}': v for k, v in val_metrics.items()})

                # Switched sum -> max (worst-case) as of 2026-07-20. The sum
                # version (re_t+re_p+re_wells) fixed the original bug (a
                # T+P-only metric could stop early while wells was still
                # improving), but a 2-seed Model A comparison showed the sum
                # has its own side effect: it let temperature drift from its
                # usual ~3.4% floor up to 3.70% in exchange for a wells gain,
                # because wells' improvement in raw percentage points
                # outweighed temperature's regression in the sum. Since the
                # project's actual target is <=2% RE on EACH output
                # independently (not a weighted average), a worst-case metric
                # is the more faithful stand-in: it won't trade one output's
                # regression for another's improvement, it only rewards
                # bringing down whichever output is currently worst. Verified
                # on Model A seed 1's own log: the sum picked epoch 63
                # (re_t=3.70%, re_wells=3.14%, worst=3.70%), but max would
                # have picked epoch 35 (re_t=3.57%, re_wells=3.57%, worst=
                # 3.57%) -- a more balanced point that doesn't let either
                # output drift as far from its own floor.
                val_metric = max(val_metrics['re_t'], val_metrics['re_p'], val_metrics['re_wells'])
                record['val_re_combined'] = val_metric

                is_best = stopper.update(val_metric, epoch)
                record['is_best'] = is_best
                if is_best:
                    save_checkpoint(model, optimizer, epoch, BEST_CKPT_PATH, stopper=stopper)

            result.history.append(record)
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
            print(_format_record(record), flush=True)

            if did_eval and epoch >= min_epochs_before_stop and stopper.should_stop(early_stop_patience):
                msg = (f'Early stopping at epoch {epoch} (no improvement for {early_stop_patience} evals; '
                       f'best={stopper.best_val:.3f} at epoch {stopper.best_epoch})')
                print(msg, flush=True)
                log_file.write(json.dumps({'event': 'early_stop', 'message': msg}) + '\n')
                log_file.flush()
                result.stopped_early = True
                break

        total_seconds = time.time() - run_start
        log_file.write(json.dumps({'event': 'end', 'total_seconds': round(total_seconds, 1)}) + '\n')
        log_file.flush()
        print(f'Total training time: {total_seconds:.1f}s', flush=True)

    result.best_epoch = stopper.best_epoch
    result.best_val_metric = stopper.best_val
    return result


def _format_record(record):
    """
    Formats one epoch's record dict into a single human-readable line for
    console/log output: epoch number, elapsed seconds, then every other key
    as `key=value` (4 decimal places for float values).

    record: the epoch's metrics dict (as appended to FitResult.history).
    Returns: str.
    """
    parts = [f"epoch={record['epoch']:03d}", f"t={record.get('epoch_seconds', 0):.1f}s"]
    for k, v in record.items():
        if k in ('epoch', 'epoch_seconds'):
            continue
        parts.append(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}')
    return ' | '.join(parts)


if __name__ == '__main__':
    # Hyperparameters below are unchanged from the pre-FiLM-fix run (same
    # width/modes/n_layers/weight_decay/patience) -- the ARC A/B test that
    # validated the FiLM fix deliberately kept everything else fixed so the
    # comparison isolated the architecture change. The rationale comments
    # below are carried over from that tuning history and explain why these
    # particular values were chosen; not being re-derived here.
    #
    # REAL ARC RUN: baseline architecture from the overfit diagnostics
    # (capacity increases didn't help there, so start here and revisit
    # width/modes/unet_base_channels/n_layers only if real validation
    # RE_T/RE_P actually warrant it -- not guessed upfront). batch_size=16
    # assumes an A100/H100-class GPU (80GB) -- confirm with nvidia-smi on
    # the allocated node before submitting a long job; drop to 4-8 if the
    # job lands on gpu-v100 (16GB) instead. resume=True means resubmitting
    # this same job after a SLURM time-limit kill continues from the last
    # completed epoch rather than restarting.
    #
    # weight_decay raised 1e-4 -> 1e-3 and early_stop_patience raised 8 -> 20:
    # the first real run (stopped at epoch 29, best epoch 21) showed clear
    # temperature overfitting from epoch ~21 on (train_re_t kept dropping
    # while val_re_t kept rising every epoch) and wells still improving on
    # train when the run stopped -- stronger regularization plus more
    # patience targets both at once. Since a full run only takes ~6 minutes,
    # this is a cheap experiment to try before considering a bigger change
    # like redesigning the well-output head.
    #
    # n_layers: tried raising 4 -> 8 to see if more depth helps temperature
    # (the toy 2-scenario overfit diagnostic never tested depth, only width/
    # modes/unet_base_channels). Real-data result was a clean negative: same
    # ~3.4% val RE_T floor, but temperature overfit FASTER (val_re_t
    # diverged from ~epoch 9 instead of ~22) and each epoch cost ~51% more
    # compute (46.8s vs 31s, matching the ~2x parameter count). Reverted
    # back to 4 -- it's the equally-accurate, cheaper choice, confirmed by
    # a real run, not a guess.
    #
    # modes4: tried raising 3 -> 5 (the maximum possible for our 9-point time
    # grid -- a real FFT of 9 samples yields exactly 5 non-negative frequency
    # bins, so this removed temporal truncation entirely) to test whether it
    # was causing FieldNet's field-level p_MPa prediction at the producer
    # cell to freeze late in the window (found via the test-set per-scenario
    # wells diagnostic: EXACTLY identical to 4 decimals across years
    # 30/35/40 for all 10 worst wells scenarios). Real-data result: a clean
    # negative that actually DISPROVES the hypothesis -- with modes4=5 (zero
    # truncation, maximum available frequency content) the field-level
    # prediction was STILL exactly frozen for every one of those same 10
    # scenarios, and RE_wells/RE_P_prod_MPa were slightly worse, not better
    # (RE_wells 4.855%->5.149%, RE_P_prod_MPa 8.437%->9.953%). Reverted back
    # to 3 -- since the model has full frequency resolution available and
    # still chooses a flat late-time prediction, this looks like a learned
    # behavior (likely: continued-decline scenarios are a minority of
    # training data, so plain MSE loss favors predicting the typical/flat
    # trend) rather than an architecture-capacity limitation. A genuine fix
    # would need a different kind of intervention (e.g. loss reweighting
    # toward declining trajectories), not more spectral/depth capacity.
    # early_stop_patience raised 20 -> 30 alongside the val_metric change
    # above (now 3 terms instead of 2) -- a noisier combined metric warrants
    # a bit more patience so a transient wells wobble doesn't trigger a stop.
    train(n_epochs=200, batch_size=16, width=8, width2=32,
          modes1=4, modes2=4, modes3=4, modes4=3, unet_base_channels=4, n_layers=4,
          weight_decay=1e-3,
          eval_every=1, early_stop_patience=30, min_epochs_before_stop=5, resume=True)
