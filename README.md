# Physics-Informed Machine Learning Surrogate Models for 3D Geothermal Reservoir Simulation

A neural operator surrogate for 3D geothermal reservoir simulation, trained and tested on a COMSOL-generated dataset of 1,535 scenarios. Given a scenario's static geology and 14 scalar parameters, the model predicts how temperature and pressure evolve across the 3D reservoir, and how production temperature, injection pressure and production pressure evolve at the wells, over 40 years.

Two variants are compared:

- **Model A** (data-driven): a 4D Fourier Neural Operator (spectral convolution over x, y, z and t) with FiLM-conditioned parameter fusion and a 3D U-Net.
- **Model B** (physics-constrained): the same architecture, with an added training loss for mass and energy conservation.

## Quick start

Requires Python 3.14. Run these from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/reproduce_figures.py --figures 3 4 10
```

This regenerates Figs. 3, 4 and 10 in `results/figures/` from the saved metrics, in under a minute. To run both trained models on the 3 sample scenarios included in `data/`:

```bash
python scripts/evaluate.py --model A --sample
python scripts/evaluate.py --model B --sample
```

Everything else needs the full prepared dataset, which is available on request (see [data/README.md](data/README.md)):

```bash
python scripts/evaluate.py --model A       # test-set metrics for Model A (or --model B)
python scripts/reproduce_figures.py        # all figures
```

## Results

| Test set (232 scenarios), mean relative error | Temperature | Pressure | Well outputs |
|---|---|---|---|
| Model A | 0.24% | 0.08% | 4.90% (median 2.79%) |
| Model B | 0.24% | 0.19% | 4.74% (median 2.57%) |

- Model A's field errors correspond to mean absolute errors of 0.92 K and 0.019 MPa. In cells that cool by more than 1 K the temperature error is 3.0 K, or 8.3% of the simulated cooling.
- Model B does not improve on Model A overall: temperature is unchanged, field pressure is worse and the wells improve only slightly. The COMSOL fields themselves do not satisfy Model B's discrete conservation residuals, which points to a mismatch between the loss and the simulator.
- Across all 1,535 scenarios, the median 40-year production-temperature decline is 5.0 K for the surrogate and 5.2 K for COMSOL, and no scenario exceeds a 10% decline in either.

## Repository layout

```
scripts/      Start here: evaluate, train, prepare the data, reproduce the figures
src/          model_a (architecture, training, evaluation, ensemble analysis, plotting)
              model_b (physics-constrained training and conservation loss)
              data_preparation (raw COMSOL export -> prepared dataset)
paper/        Analyses behind the paper's numbers, figure scripts, graphical abstract
checkpoints/  Final weights: model_a/best.pth (epoch 109), model_b/best.pth (epoch 99)
data/         3-scenario sample; the full dataset goes here (available on request)
results/      Saved metrics, the training log behind Fig. 2, and the figures
```

## Reproducing the paper

### Figures

`python scripts/reproduce_figures.py --list` prints this table. Each script makes its whole group.

| Figure | Script in `paper/figures/` | Needs |
|---|---|---|
| 2, 7 | `fig02_07_training_and_depth_change.py` | dataset |
| 3, 4, 10 | `fig03_04_10_error_distributions.py` | nothing extra |
| 5a | `fig05a_real_01051.py` | dataset |
| 5b | `fig05b_error_vs_initial_pressure.py` | dataset |
| 6 | `fig06_3d_fields.py` | dataset |
| 8, 9, A2, A3 | `fig08_09_A2_A3_depth_galleries.py` | dataset |
| 11, 12 | `fig11_12_ensemble.py` | dataset, ensemble run |
| 13 | `fig13_controls.py` | dataset, ensemble run |
| A1 | `figA1_well_trajectories.py` | dataset |

Fig. 1 is a schematic with no script. The ensemble run (`fig11_13_capture_ensemble.py`) starts automatically when Figs. 11-13 are requested and its output is missing.

### Numbers

| Result | Command | Output in `results/metrics/` |
|---|---|---|
| Test-set errors, Table and Figs. 3, 4, 10 | `python scripts/evaluate.py --model A` (and `B`) | `model_a_test_results.json`, `model_b_test_results.json` |
| Physical-unit errors, baselines, physics residuals, ensemble indicators, timing | `python paper/reviewer_metrics.py` | `reviewer_metrics.json` |
| Audit of the extreme scenario real_01051 | `python paper/audit_real_01051.py` | `audit_real_01051.json` |
| Error by depth layer (Appendix A) | `python paper/layerwise_error.py` | `layerwise_error.json` |

Each command overwrites the saved file. The saved results were computed on a GPU. On a CPU we get the same values at the precision the paper reports, with differences from about the fifth significant digit, which can also shift a few pixels in Figs. 3, 4 and 10. `git checkout results/metrics` restores the saved files.

### Run times

Measured on a 12-thread CPU, without a GPU:

| Step | Time |
|---|---|
| `scripts/evaluate.py`, per model | 2 min |
| `paper/reviewer_metrics.py` | 15 min |
| `paper/audit_real_01051.py` | 4 min |
| `paper/layerwise_error.py` | 30 s |
| All figures, excluding the ensemble run | 5 min |
| Ensemble run | 10 min |
| `paper/graphical_abstract/prepare_panels.py` (first run) | 8 min |

The graphical abstract panels come from `prepare_panels.py`, and `build_graphical_abstract.js` assembles them into a PowerPoint file (it needs the `pptxgenjs` npm package).

## Training

```bash
python scripts/train.py --model A --seed 1
```

Checkpoints go to `checkpoints/model_a_seed1/`, and rerunning resumes from the last one. Seed 0 is the seed of the published checkpoints and writes to the folder that holds them. A GPU is strongly recommended. Hyperparameters are set at the bottom of `src/model_a/comsol_fno_train.py`.

## Credits and license

The architecture and the physics loss build on PCNO. Parts of its code are reused or adapted here, including `PDE_F` in `src/model_b/comsol_fno_physics.py`, which is copied unchanged:

Z. Xue, A robust physics-constrained neural operator framework for efficient geothermal resource development [source code], Code Ocean, 2026, https://doi.org/10.24433/CO.2608907.v1.

This repository is released under the GNU General Public License v3.0, the license of the PCNO code. See [LICENSE](LICENSE).

## Citation

```
Alvarado, A., Sharma, A., & Moynihan, P. (2026). Physics-Informed-ML: Surrogate models
for 3D geothermal reservoir simulation [Source code]. GitHub.
https://github.com/engage-research-lab/Physics-Informed-ML
```
