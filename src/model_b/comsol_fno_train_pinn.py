"""
Physics-constrained training script -- adds mass + energy conservation
residuals (comsol_fno_physics.py) on top of the SAME architecture and data
loss already used by comsol_fno_train.py. Deliberately a controlled
comparison: everything except the loss function is identical to the
current best data-driven model (well-head-fix, n_layers=4, modes4=3,
weight_decay=1e-3, batch_size=16), so any difference in results is
attributable to the physics constraint itself, not a confound from also
changing architecture/hyperparameters at the same time.

Reuses comsol_fno_model.py and comsol_fno_train.py's infrastructure via
import (dataset loading, checkpointing, early stopping, RE metrics) rather
than duplicating it -- only the loss computation is new.

The current best data-driven model (../model_a/) is completely untouched by this file.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_a')
sys.path.insert(0, _MODEL_DIR)

from comsol_fno_model import ComsolFnoModel, masked_mse, well_output_mse, masked_relative_error, relative_error
from comsol_fno_train import (
    ComsolFnoDataset, collate, denormalize, PREPARED_DATA_PATH,
    EarlyStopState, FitResult, save_checkpoint,
)

from comsol_fno_physics import build_physics_loss, erode_mask_3d

# Overridable via TRAIN_SEED env var for a second-seed reproducibility check
# (e.g. confirming the wells-improves-later-than-selected-checkpoint pattern
# isn't specific to seed 0). Default unchanged -- SEED=0 behaves exactly as
# before, including the checkpoint directory name.
SEED = int(os.environ.get('TRAIN_SEED', 0))
torch.manual_seed(SEED)
np.random.seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CKPT_SUBDIR = 'model_b' if SEED == 0 else f'model_b_seed{SEED}'
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', _CKPT_SUBDIR)
RESUME_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'resume.pth')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
LOG_PATH = os.path.join(CHECKPOINT_DIR, 'train_log.jsonl')

DT_SECONDS = 5.0 * 365 * 24 * 3600  # matches the step=5-year uniform field-time grid

# See run_epoch's PHYSICS_LOSS_MAGNITUDE_LIMIT comment below for why this
# exists alongside isfinite() -- healthy mse_mass/mse_energy values observed
# in real training runs stay under ~1 (e.g. mse_mass ranged ~0.15-0.65 in
# the epochs before a run went unstable), so 1e4 leaves generous headroom
# for legitimate variation while still catching genuine blow-ups.
PHYSICS_LOSS_MAGNITUDE_LIMIT = 1e4


# ---------------------------------------------------------------------------
# Dataset -- adds raw (unnormalized) static fields/params alongside
# everything ComsolFnoDataset already provides. The physics loss needs real
# physical values (porosity, permeability, injection rate/temp, etc.), not
# the normalized versions fed to the neural net.
# ---------------------------------------------------------------------------
class ComsolFnoDatasetPINN(ComsolFnoDataset):
    """
    Extends ComsolFnoDataset with the raw (unnormalized) static spatial
    fields and scalar parameters for each scenario, alongside everything
    the base class already provides. The physics loss
    (comsol_fno_physics.py) needs real physical values (porosity,
    permeability, injection rate/temperature, etc.), not the normalized
    versions fed to the neural network.
    """
    def __getitem__(self, idx):
        """
        Builds one scenario's training sample via
        ComsolFnoDataset.__getitem__, then adds the raw (unnormalized)
        static fields and scalar parameters needed by the physics loss.

        idx: index into this split's arrays.
        Returns: dict with all of ComsolFnoDataset.__getitem__'s keys plus
        'x_static_raw' (n_static_channels, X, Y, Z) and 'params_raw'
        (n_params,), both in real physical units.
        """
        item = super().__getitem__(idx)
        item['x_static_raw'] = self.x_static_fields[idx].astype(np.float32)
        item['params_raw'] = self.x_params[idx].astype(np.float32)
        return item


def physics_weight(epoch):
    """
    Staged introduction of the physics loss -- plain data loss only for the
    first few epochs (let the model learn coarse shape before constraining
    still-garbage early predictions with PDE residuals), then ramps in.
    Same rationale as the project's earlier 2D PINN track
    (pcno_lite/pinn_loss.py's staged_weights), but compressed further:
    that schedule assumed 100+ epoch runs: this dataset/architecture
    converges much faster (best epochs 9-26 across 5 real training runs
    on the data-driven model). This exact schedule is a reasonable
    starting point based on that convergence pattern, NOT empirically
    tuned yet for the physics loss specifically -- watch the loss curve
    once real runs start and adjust if physics destabilizes training.
    """
    if epoch < 3:
        return 0.0
    if epoch < 6:
        return 0.05
    return 0.15


def run_epoch(model, loader, stats, device, epoch, static_channel_names, param_names,
              z_m, x_m, y_m, optimizer=None):
    """
    Runs one epoch over `loader`, mirroring comsol_fno_train.py's
    run_epoch (masked T_K/p_MPa MSE, well-output MSE, denormalized RE%
    metrics, backprop + optimizer step when `optimizer` is given), and
    additionally computes a physics-conservation loss (mass + energy
    residuals, via comsol_fno_physics.py) and folds it into the training
    objective, weighted by this epoch's physics_weight(epoch). A batch
    whose physics residual is non-finite or implausibly large has just the
    physics term skipped for that batch; a batch whose backward pass
    produces a non-finite gradient has its entire optimizer step skipped
    instead, so neither corrupts the run.

    model: ComsolFnoModel. loader: DataLoader yielding collated batches
    (including the raw fields/params provided by ComsolFnoDatasetPINN).
    stats: per-channel/per-param mean/std dict. device: torch device.
    epoch: current epoch number, used to look up this epoch's physics-loss
    weight via physics_weight(). static_channel_names, param_names: channel
    and parameter name lists, needed by the physics loss to identify which
    raw channel/param is which. z_m, x_m, y_m: the grid's physical
    coordinate arrays (meters), needed for spatial derivatives in the
    physics residuals. optimizer: torch optimizer, or None for a
    validation/eval pass.

    Returns: dict of the epoch's averaged metrics (loss, loss_t, loss_p,
    loss_wells, loss_mass, loss_energy, re_t, re_p, re_wells), plus this
    epoch's physics_weight and the counts of batches skipped for a bad
    physics residual (n_physics_skipped) or a bad gradient
    (n_bad_grad_skipped).
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

    w_phys = physics_weight(epoch)

    totals = {'loss': 0.0, 'loss_t': 0.0, 'loss_p': 0.0, 'loss_wells': 0.0,
              'loss_mass': 0.0, 'loss_energy': 0.0,
              're_t': 0.0, 're_p': 0.0, 're_wells': 0.0}
    n_batches = 0
    n_physics_skipped = 0
    n_bad_grad_skipped = 0

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
            x_static_raw = batch['x_static_raw'].to(device)
            params_raw = batch['params_raw'].to(device)

            if is_train:
                optimizer.zero_grad()

            out = model(x_static, params, well_map)
            loss_t = masked_mse(out['T_K'], target_T, mask_T)
            loss_p = masked_mse(out['p_MPa'], target_P, mask_P)
            loss_w = well_output_mse(out['wells'], target_wells)
            loss = loss_t + loss_p + loss_w

            loss_mass_val, loss_energy_val = 0.0, 0.0
            if w_phys > 0:
                pred_T_c = denormalize(out['T_K'], t_mean, t_std) - 273.15  # PDE_F expects degrees C
                pred_P_real = denormalize(out['p_MPa'], p_mean, p_std)

                phys_loss_module = build_physics_loss(
                    x_static_raw=x_static_raw, params_raw=params_raw,
                    static_channel_names=static_channel_names, param_names=param_names,
                    z_m=z_m, x_m=x_m, y_m=y_m, dt_seconds=DT_SECONDS,
                )
                eroded_mask = erode_mask_3d(mask_T * mask_P)
                mse_mass, mse_energy = phys_loss_module(pred_P_real, pred_T_c, valid_mask=eroded_mask)

                # PHYSICS_LOSS_MAGNITUDE_LIMIT (2026-07-28): a real ARC run
                # showed the plain isfinite() check isn't sufficient on its
                # own -- it correctly skipped 64 batches at the epoch where
                # training went NaN, yet the model still went NaN right
                # after. That means some batches were passing isfinite()
                # (technically finite) while still being pathologically
                # large -- e.g. a residual of 1e4 against a normal operating
                # range of ~0.1-1 is 10,000x too big, and multiplying that by
                # w_phys and backpropagating is enough to blow up the
                # weights even though nothing was ever literally NaN/inf at
                # this check. Reject those too, not just literal non-finite.
                mass_ok = torch.isfinite(mse_mass) and mse_mass.abs() < PHYSICS_LOSS_MAGNITUDE_LIMIT
                energy_ok = torch.isfinite(mse_energy) and mse_energy.abs() < PHYSICS_LOSS_MAGNITUDE_LIMIT
                if mass_ok and energy_ok:
                    loss = loss + w_phys * (mse_mass + mse_energy)
                    loss_mass_val = mse_mass.item()
                    loss_energy_val = mse_energy.item()
                else:
                    # Defensive: an occasional non-finite OR pathologically
                    # large physics residual (e.g. an extreme early-training
                    # prediction) shouldn't poison the whole batch's gradient
                    # -- skip just the physics term for this batch and log it.
                    n_physics_skipped += 1

            if is_train:
                loss.backward()

                # Gradient-finiteness check (2026-07-29): a real ARC run on
                # the FiLM architecture went NaN at epoch 3, physics_weight
                # only 0.05 (the smallest non-zero value) -- 59 of 67
                # batches were correctly skipped by the magnitude-limit
                # check above, but the remaining 8 (which PASSED that check
                # -- finite, under the threshold) still corrupted the
                # model. That means the forward LOSS VALUE being reasonable
                # doesn't guarantee the GRADIENT is: the same class of bug
                # fixed in PDE_F.vis() (torch.where computing a NaN local
                # gradient for an unselected branch, 0*NaN=NaN) could exist
                # elsewhere in PDE_F/MassEnergyConservationLoss and hasn't
                # necessarily all been found. Also realized: clip_grad_norm_
                # does NOT protect against this -- it computes one global
                # norm across every parameter, so a single NaN gradient
                # poisons that norm, poisons the shared clip coefficient,
                # and then gets multiplied into EVERY parameter's gradient,
                # potentially spreading the corruption rather than
                # containing it. Checking gradient finiteness directly,
                # before clipping/stepping, is the correct layer to catch
                # this -- skip the optimizer step entirely (not just the
                # physics term) for any batch whose gradient isn't clean,
                # rather than letting a corrupted update reach the weights.
                grads_finite = all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if grads_finite:
                    # Gradient clipping (2026-07-28): a general safety net
                    # for large-but-finite gradients (a well-known PINN
                    # failure mode) -- only meaningful once we've already
                    # confirmed there's no NaN/inf to spread.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                else:
                    n_bad_grad_skipped += 1
                optimizer.zero_grad()

            with torch.no_grad():
                pred_T_real = denormalize(out['T_K'], t_mean, t_std)
                target_T_real = denormalize(target_T, t_mean, t_std)
                pred_P_real2 = denormalize(out['p_MPa'], p_mean, p_std)
                target_P_real = denormalize(target_P, p_mean, p_std)
                re_t = masked_relative_error(pred_T_real, target_T_real, mask_T)
                re_p = masked_relative_error(pred_P_real2, target_P_real, mask_P)

                pred_wells_real = torch.stack(
                    [denormalize(out['wells'][:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
                target_wells_real = torch.stack(
                    [denormalize(target_wells[:, :, c], *well_mean_std[c]) for c in range(3)], dim=-1)
                re_w = relative_error(pred_wells_real, target_wells_real)

            # Defensive: only fold this batch's metrics into the epoch
            # average if they're actually finite -- guards the REPORTED/
            # logged numbers the same way the gradient check above guards
            # the WEIGHTS, in case some still-unknown pathway ever produces
            # a genuinely-NaN forward loss (as opposed to a finite loss with
            # a NaN gradient, which is what was actually found and fixed
            # here) despite the isfinite()/magnitude checks already in place.
            if torch.isfinite(loss):
                totals['loss'] += loss.item()
                totals['loss_t'] += loss_t.item()
                totals['loss_p'] += loss_p.item()
                totals['loss_wells'] += loss_w.item()
                totals['loss_mass'] += loss_mass_val
                totals['loss_energy'] += loss_energy_val
                totals['re_t'] += re_t.item()
                totals['re_p'] += re_p.item()
                totals['re_wells'] += re_w.item()
                n_batches += 1

    n_batches = max(n_batches, 1)
    result = {k: v / n_batches for k, v in totals.items()}
    result['physics_weight'] = w_phys
    result['n_physics_skipped'] = n_physics_skipped
    result['n_bad_grad_skipped'] = n_bad_grad_skipped
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


def train(
    n_epochs=200,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-3,  # matches the current best data-driven config
    width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
    unet_base_channels=4, n_layers=4,
    eval_every=1,
    early_stop_patience=20,
    min_epochs_before_stop=5,
    resume=True,
):
    """
    Runs the full training loop for the physics-constrained model: builds
    the PINN train/val datasets and loaders (ComsolFnoDatasetPINN, which
    also carries the raw fields/params the physics loss needs), constructs
    the model and AdamW optimizer, optionally resumes from a saved
    checkpoint, then iterates epochs running run_epoch on the training set
    (every epoch) and the validation set (every `eval_every` epochs). Each
    epoch's physics-loss contribution is weighted according to
    physics_weight(epoch), which ramps the physics term in gradually over
    the first few epochs rather than applying it at full weight from epoch
    1. Saves a resume checkpoint every epoch and a best checkpoint whenever
    validation improves, logs each epoch's record to train_log.jsonl, and
    stops early once the worst of val_re_t/val_re_p/val_re_wells hasn't
    improved for `early_stop_patience` evaluations (after at least
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

    train_dataset = ComsolFnoDatasetPINN(PREPARED_DATA_PATH, split='train')
    val_dataset = ComsolFnoDatasetPINN(PREPARED_DATA_PATH, split='val')
    print(f'Train: {len(train_dataset)} scenarios   Val: {len(val_dataset)} scenarios')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    raw = np.load(PREPARED_DATA_PATH, allow_pickle=True)
    static_channel_names = list(raw['static_channel_names'])
    param_names = [n for n in raw['parameter_names'] if n != 'well_radius']
    z_m, x_m, y_m = raw['z_m'], raw['x_m'], raw['y_m']

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
            # See comsol_fno_train.py's save_checkpoint docstring -- this is
            # the exact bug that lost the true best (epoch 13) of the first
            # real PINN run on 2026-07-19. Fixed going forward; this branch
            # only fires for checkpoints saved before the fix.
            print(f'Resumed from checkpoint at epoch {ckpt["epoch"]}, continuing from epoch {start_epoch} '
                  f'-- WARNING: this checkpoint has no saved early-stop state, so best_val/best_epoch cannot '
                  f'be restored and will reset. If a better epoch existed before this checkpoint was written, '
                  f'it will not be recovered.')

    result = FitResult()
    run_start = time.time()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_mode = 'a' if (resume and start_epoch > 1) else 'w'
    with open(LOG_PATH, log_mode) as log_file:
        if log_mode == 'w':
            log_file.write(json.dumps({'event': 'start', 'wall_time': run_start, 'n_params': n_params}) + '\n')
            log_file.flush()

        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(model, train_loader, train_dataset.stats, device, epoch,
                                       static_channel_names, param_names, z_m, x_m, y_m, optimizer=optimizer)
            epoch_seconds = time.time() - epoch_start

            record = {'epoch': epoch, 'epoch_seconds': round(epoch_seconds, 2),
                      **{f'train_{k}': v for k, v in train_metrics.items()}}

            save_checkpoint(model, optimizer, epoch, RESUME_CKPT_PATH, stopper=stopper)

            did_eval = epoch % eval_every == 0 or epoch == 1
            if did_eval:
                val_metrics = run_epoch(model, val_loader, val_dataset.stats, device, epoch,
                                         static_channel_names, param_names, z_m, x_m, y_m, optimizer=None)
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


if __name__ == '__main__':
    # Same architecture/hyperparameters as the current best data-driven
    # model (n_layers=4, modes4=3, weight_decay=1e-3, batch_size=16) --
    # deliberately unchanged, so this is a controlled comparison of the
    # physics loss specifically, not a confound with other changes.
    # early_stop_patience raised 20 -> 30 alongside the val_metric change
    # above (now 3 terms instead of 2) -- a noisier combined metric warrants
    # a bit more patience so a transient wells wobble doesn't trigger a stop.
    train(n_epochs=200, batch_size=16, width=8, width2=32,
          modes1=4, modes2=4, modes3=4, modes4=3, unet_base_channels=4, n_layers=4,
          weight_decay=1e-3,
          eval_every=1, early_stop_patience=30, min_epochs_before_stop=5, resume=True)
