# Dataset

## Sample (included)

`sample_scenarios.npz` (5 MB) holds 4 of the 232 test scenarios, in the same format as the full prepared dataset. They are the scenarios shown in the paper's figures:

| Scenario | Figures |
|---|---|
| real_00008 | 6, 7, 8, 9 |
| real_01456 | 7, A2 |
| real_01528 | 7, A3 |
| real_01051 | 5a (the test scenario with the largest well error) |

Each scenario contains the static geology (permeability, porosity and the other input fields on the 60 x 30 x 20 grid), the 15 scalar parameters, the simulated temperature and pressure fields at 11 times, and the well time series. The normalization statistics of the full dataset are included, so the published checkpoints can be applied directly.

To run both models on the sample and compare with the paper:

```bash
python scripts/evaluate.py --model A --sample
python scripts/evaluate.py --model B --sample
python scripts/reproduce_figures.py --sample
```

The first two commands take a few seconds each on a CPU. They print each scenario's temperature, pressure and well errors next to the values from the published test-set run (`results/metrics/model_a_test_results.json` and `model_b_test_results.json`). On a CPU these agree to 3 or 4 significant digits; the published values were computed on a GPU.

The third command takes under a minute on a 12-thread CPU and regenerates Figs. 2, 3, 4, 5a, 6, 7, 8, 9, 10, A2 and A3 in `results/figures/`. We checked that each one comes out identical to the published figure.

The remaining figures (5b, 11, 12, 13, A1) and the test-set totals in the paper use all 232 test scenarios or all 1,535 scenarios, so they need the full dataset.

## Full dataset (on request)

The full prepared dataset (1,535 scenarios, 2 GB) is available from the authors on request. Scripts look for it at:

```
data/fno_lhs_v1_prepared_phase1.npz
```

To keep it somewhere else, set the `GEOTHERMAL_DATA_PATH` environment variable to its full path.

If you have the raw COMSOL export instead, build the prepared file with:

```bash
python scripts/prepare_data.py --raw-npz <raw .npz> --summary-csv <summary .csv>
```
