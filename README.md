# datathon — Track 2 fraud detection

Predict the probability that a transaction is fraud. Scored on PR-AUC on a hidden test set.

## Final model: XGBoost (`train_xgboost.py`)

| Submission | Leaderboard PR-AUC |
|---|---|
| XGBoost + engineered features, `outputs/submission_xgboost.csv` | **0.19096** |
| Same XGBoost pipeline run on a teammate's machine | 0.19416 |
| CatBoost, raw columns, `outputs/submission_catboost.csv` | 0.18772 |

The two XGBoost rows are the same code; the 0.003 gap comes from library-version
differences, which shows how much the leaderboard moves on small changes.

### Reproduce

```
pip install -r requirements.txt
python train_xgboost.py      # CV + hold-out evaluation, saves model/ and outputs/xgboost/
python predict.py            # scores data/Track_2_Testing_Dataset.csv -> predictions.csv (id,prediction)
python explain_xgboost.py    # column weights / value effects / per-row SHAP -> outputs/xgboost/
```

`export_model_pkl.py` is a Jupyter cell: put the contents of `train_xgboost.py` in the cell
above it, run both, and it writes `model.pkl`: the same model with the feature engineering
built in (`pickle.load(...)` then `model.predict_proba(raw_test_df)[:, 1]`).

`predict.py --model catboost` uses the CatBoost model instead (`train_catboost.py`, `explain_catboost.py`).

### Features the XGBoost model uses (30)

* 10 original columns (categoricals use XGBoost's native categorical splits)
* ratios: `avg_spend_per_txn_24h`, `spike_ratio`, `share_of_24h_spend`, `urgency_ratio`
* risk scores: smoothed historical fraud rate of `merchant_category`, `country`, `transaction_channel`
  (computed out-of-fold on training rows so a row never sees its own label)
* flags: `is_high_risk_merchant` (luxury / cash_transfer / electronics), `is_new_account` (< 30 days)
* 10 missing-value flags

Model: 300 trees, depth 2, learning rate 0.03, subsample/colsample 0.8, min_child_weight 5, lambda 5.

## Files

### Code

| File | What it does |
|---|---|
| `train_xgboost.py` | Feature pipeline, CV, hold-out report, final model, intermediate CSVs |
| `predict.py` | Scores a CSV with the saved model, writes `id,prediction` |
| `explain_xgboost.py` | Column / feature weights (SHAP), value effects, per-row contributions |
| `train_catboost.py`, `explain_catboost.py` | Same for the CatBoost alternative |
| `analysis/common.py` | Shared loaders and model builders for the analysis scripts |
| `analysis/00_xgboost_tuning.py` | Depth / rounds / fraud up-weighting grid (raw columns) |
| `analysis/01_eda.py` | Fraud rates per value, missing values, train-vs-test shift, id drift |
| `analysis/02_adversarial_validation.py` | Train-vs-test classifier and test-likeness weights |
| `analysis/03_feature_ablation.py` | Submitted XGBoost with each engineered feature group removed |
| `analysis/04_model_comparison.py` | XGBoost / LightGBM / spline logistic / CatBoost variants and blends |
| `analysis/05_paired_comparison.py` | 10 paired CV splits: submitted XGBoost vs raw XGBoost vs CatBoost |

Run analysis scripts from the repo root (`python analysis/01_eda.py`); run `02` before `03`–`05`.

### Outputs

| File | Contents |
|---|---|
| `outputs/submission_xgboost.csv` | Submitted XGBoost predictions (`id,prediction`) |
| `outputs/submission_catboost.csv` | CatBoost predictions |
| `outputs/xgboost/cv_fold_scores.csv` | ROC-AUC / PR-AUC per CV fold and overall |
| `outputs/xgboost/oof_predictions.csv` | Out-of-fold prediction for every training row |
| `outputs/xgboost/holdout_predictions.csv` | 20% hold-out predictions and 0/1 flags |
| `outputs/xgboost/train_features.csv` | Full 30-feature matrix the final model was trained on |
| `outputs/xgboost/test_features.csv` | Same features for the test set |
| `outputs/xgboost/risk_scores.csv` | Risk score used for every category value |
| `outputs/xgboost/feature_importance_gain.csv` | XGBoost gain / split counts per feature |
| `outputs/xgboost/feature_weights.csv` | SHAP weight per model feature (train and test) |
| `outputs/xgboost/column_weights.csv` | SHAP weight per original column (engineered features folded in) |
| `outputs/xgboost/value_effects.csv` | Odds multiplier per category / numeric range |
| `outputs/xgboost/test_shap_values.csv` | Per-test-row contribution of every feature (log-odds) |
| `outputs/catboost/*.csv` | Column weights and value effects for the CatBoost model |
| `analysis/results/*.csv`, `*.log` | Output of each analysis script |
