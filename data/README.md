# Dataset

## Sample (included)

`sample_3_scenarios.npz` (4 MB) holds 3 of the 232 test scenarios, in the same format as the full prepared dataset:

| Scenario | Why it is included |
|---|---|
| real_00386 | Median well error for Model A |
| real_01548 | Upper-quartile well error, from the exploratory part of the design |
| real_01051 | Largest well error in the test set, the case shown in Fig. 5a |

Each scenario contains the static geology (permeability, porosity and the other input fields on the 60 x 30 x 20 grid), the 15 scalar parameters, the simulated temperature and pressure fields at 11 times, and the well time series. The normalization statistics of the full dataset are included, so the published checkpoints can be applied directly.

To evaluate both models on it:

```bash
python scripts/evaluate.py --model A --sample
python scripts/evaluate.py --model B --sample
```

This takes a few seconds on a CPU and writes `results/metrics/model_a_sample_results.json` (and `model_b_...`). The per-scenario errors match those in `model_a_test_results.json` and `model_b_test_results.json`.

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

Without the full dataset you can still evaluate the sample above and regenerate Figs. 3, 4 and 10, which run from the saved metrics in `results/metrics/`.
