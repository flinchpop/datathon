# datathon: Track 2 fraud detection

Fraud classifier for the Track 2 transactions dataset. The final model is a blend of **CatBoost**
and **monotone-constrained XGBoost** on behavioural features, validated for the train-to-test
distribution shift in this data. See **[SUBMISSION_NOTES.md](SUBMISSION_NOTES.md)** for the full
write-up: problem understanding, data findings, model choice, assumptions and results.

| Out-of-fold | Original XGBoost | Final model |
|---|---|---|
| PR-AUC of the probabilities, CV | 0.205 | **0.226** |
| PR-AUC of the probabilities, test-like hold-out | 0.304 | **0.364** |
| PR-AUC of the 0/1 column (what the leaderboard appears to score), CV | 0.104 | **0.117** |

The 0/1 `fraud` column is cut with a plug-in threshold that maximises the expected PR-AUC of the 0/1
column on the file being scored (see SUBMISSION_NOTES.md §5 and §7).

## Layout

| File | Purpose |
|---|---|
| `features.py` | Feature engineering (label-free, shared by train and predict) |
| `model.py` | `FraudEnsemble`: CatBoost (3 seeds) + monotone XGBoost, save/load, SHAP importance |
| `validation.py` | Adversarial weights, repeated CV, shift-weighted CV, test-like hold-out, threshold rule |
| `train.py` | Validates, chooses the threshold, fits on all rows, saves `model/` and `reports/` |
| `predict.py` | Scores a CSV and writes `id, fraud_probability, fraud` (plug-in threshold by default) |
| `rethreshold.py` | Re-cuts the 0/1 column of any existing submission (probabilities unchanged) |
| `leaderboard_check.py` | Which metric explains the leaderboard result, and how noisy it is |
| `compare_models.py` | Baseline vs alternatives under all three validation schemes |
| `analysis.py` | Data findings behind the design (`reports/data_analysis.md`) |
| `baseline/` | The original XGBoost scripts, unchanged, for comparison |
| `predictions.csv` | Predictions for `Track_2_Testing_Dataset.csv` |

## Usage

```bash
pip install -r requirements.txt
# Put Track_2_Training_Dataset.csv and Track_2_Testing_Dataset.csv in data/ (not committed)
python train.py                         # ~2-3 min
python predict.py                       # -> predictions.csv (plug-in threshold)
python predict.py --rule f1             # cut the 0/1 column for F1 instead
python predict.py in.csv out.csv --threshold 0.06   # custom file / fixed threshold
python rethreshold.py team_submission.csv team_recut.csv   # re-cut another submission
```
