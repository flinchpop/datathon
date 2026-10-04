# Datathon – Track 2 fraud detection

Rank-averaged ensemble (CatBoost ×2, XGBoost, LightGBM, logistic regression) on engineered
account-relative and merchant-relative features. Repeated-CV PR-AUC 0.226, F1 0.31, recall 0.25.

* **Platform upload:** the `submission/` folder – `predict_notebook.ipynb` + `model.pkl` + `requirements.txt`
  (see `submission/README.txt`). The notebook reads `DATATHON_INPUT_PATH`, writes a single `prediction` column
  to `DATATHON_OUTPUT_PATH`, no training cells.
* **Model file:** `model.pkl` – one self-contained object; `pickle.load` (or `joblib.load`) it and call
  `predict_proba(X)[:, 1]` (score for PR-AUC) or `predict(X)` (0/1 at the chosen operating point) on the raw
  test CSV read with pandas (`id` column optional). Needs only numpy, pandas, lightgbm, catboost at prediction
  time – no project code, no xgboost (its trees are evaluated in numpy). Built with `python3 build_model_pkl.py`,
  checked with `python3 verify_model_pkl.py`.
* **Upload file (scores):** `outputs/submission.csv` (`id, fraud` score in [0,1]; `fraud ≥ 0.5` = flagged, 1.5 % of rows).
  Binary version: `outputs/submission_binary.csv`. Alternates at 1.1 % / 2.5 % flagged alongside.
* **Full write-up** (problem understanding, EDA, feature engineering, model choice, assumptions,
  trial-and-error log, results): [`REPORT.md`](REPORT.md).
* **Code:** `src/` (features, CV harness, models, pipeline), `experiments/e0–e4` (ablations), `run_final.py`,
  `make_submissions.py`, `fraud_model.py` (pickle-able ensemble class), `build_model_pkl.py`, `verify_model_pkl.py`. Experiment result tables are in `outputs/e*.csv`.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost scipy
python3 run_final.py --seeds 5      # ~20 min on 4 cores; writes outputs/submission*.csv
```
