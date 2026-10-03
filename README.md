# datathon: Track 2 fraud detection

Fraud classifier for the Track 2 transactions dataset. The final model is a blend of **CatBoost**
and **monotone-constrained XGBoost** on behavioural features, validated for the train-to-test
distribution shift in this data. See **[SUBMISSION_NOTES.md](SUBMISSION_NOTES.md)** for the full
write-up: problem understanding, data findings, model choice, assumptions and results.

| | Original XGBoost | Final model |
|---|---|---|
| CV PR-AUC | 0.205 | **0.226** |
| Shift-weighted CV PR-AUC | 0.321 | **0.356** |
| Test-like hold-out PR-AUC | 0.304 | **0.364** |

## Layout

| File | Purpose |
|---|---|
| `features.py` | Feature engineering (label-free, shared by train and predict) |
| `model.py` | `FraudEnsemble`: CatBoost (3 seeds) + monotone XGBoost, save/load, SHAP importance |
| `validation.py` | Adversarial weights, repeated CV, shift-weighted CV, test-like hold-out, threshold rule |
| `train.py` | Validates, chooses the threshold, fits on all rows, saves `model/` and `reports/` |
| `predict.py` | Scores a CSV and writes `id, fraud_probability, fraud` |
| `compare_models.py` | Baseline vs alternatives under all three validation schemes |
| `analysis.py` | Data findings behind the design (`reports/data_analysis.md`) |
| `baseline/` | The original XGBoost scripts, unchanged, for comparison |
| `predictions.csv` | Predictions for `Track_2_Testing_Dataset.csv` |

## Usage

```bash
pip install -r requirements.txt
# Put Track_2_Training_Dataset.csv and Track_2_Testing_Dataset.csv in data/ (not committed)
python train.py                         # ~2-3 min
python predict.py                       # -> predictions.csv
python predict.py in.csv out.csv --threshold 0.06   # custom file / operating point
```
