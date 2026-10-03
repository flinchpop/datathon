# datathon: Track 1, Campus Building Energy Usage

Predict hourly `energy_usage` for 12 campus buildings from building, calendar, weather, occupancy and previous-hour usage.

**Full write-up (problem understanding, model choice, assumptions): [REPORT.md](REPORT.md)**

## Result

**Final model:** a structured linear regression (building-type daily profiles, type-specific temperature and occupancy slopes, a cooling hinge above 30 °C, previous-hour persistence) plus a shallow LightGBM trained on its residuals. One such model is fitted per pattern of missing sensors.

| Validation view | RMSE | MAE | R² |
|---|---|---|---|
| 5-fold CV × 3 repeats | 3.090 | 2.435 | 0.972 |
| CV with the test set's missing-value patterns | 3.224 | 2.508 | 0.969 |
| ...and re-weighted to the test set's covariate shift | 3.441 | 2.609 | 0.972 |
| *Naive baseline: previous-hour usage* | *6.423* | *4.342* | *0.879* |

Predictions for the 3,000 test rows: [`outputs/submission.csv`](outputs/submission.csv).

## Run

```bash
pip install -r requirements.txt
python -m src.run_cv        # compare candidate models -> outputs/cv_results.csv
python -m src.train         # final model + submission -> outputs/submission.csv, outputs/final_metrics.json
python -m src.diagnostics   # assumption tests + figures -> outputs/assumption_tests.csv, outputs/figures/
```

## Layout

```
data/                      training and test CSVs as provided
src/features.py            loading, calendar features, cooling-degree hinge
src/models.py              structured linear model, LightGBM, hybrid, per-missing-pattern wrapper
src/evaluation.py          metrics, test-like missingness injection, covariate-shift weights, repeated CV
src/run_cv.py              candidate model comparison
src/train.py               final fit + submission
src/diagnostics.py         regression assumption checks and report figures
outputs/                   submission, CV results, assumption tests, coefficients, figures
REPORT.md                  the write-up
```
