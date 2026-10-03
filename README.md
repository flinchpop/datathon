# Datathon Track 2 — Transaction Fraud Detection

An **Explainable Boosting Machine (EBM)** that scores each transaction's fraud probability and flags
it as fraud when that probability reaches a threshold chosen for **F1 and Recall jointly**.

| | |
|---|---|
| Model | Explainable Boosting Machine (glass-box GA²M), `interpret` |
| Cross-validated PR-AUC | **0.224 ± 0.036** (5-fold × 3 repeats; no-skill baseline 0.018) |
| Decision threshold | **0.085**, maximises F<sub>√3</sub> = harmonic mean of F1 and Recall |
| CV metrics at threshold | precision 0.23 · **recall 0.30** · **F1 0.26** |
| Test predictions | [`outputs/submission.csv`](outputs/submission.csv) (`id, fraud_probability, fraud_prediction`) |

**Read next:**
- [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md): the write-up covering the problem statement, why this model, its assumptions and limitations.
- [`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb): the evidence (EDA, drift analysis, benchmark, assumption checks, explanations).

## Repository layout

```
data/                         training + testing CSVs as provided
src/fraud_pipeline.py         features, model zoo, repeated CV, threshold logic, drift diagnostics
src/plots.py                  report figures
train.py                      end-to-end pipeline -> outputs/ and reports/figures/
notebooks/build_notebook.py   source of the analysis notebook
notebooks/fraud_detection.ipynb   executed analysis notebook
outputs/submission.csv        test-set predictions
outputs/cv_results.csv        benchmark of 9 models
outputs/threshold_table.csv   precision/recall/F1/F-beta at every threshold
outputs/metrics.json          headline numbers
outputs/ebm_model.pkl         fitted model + threshold
reports/MODEL_REPORT.md       write-up
reports/figures/              figures
```

## Reproduce

```bash
pip install -r requirements.txt
python train.py                         # ~6 min on 4 cores; add --skip-benchmark for the EBM only (~1.5 min)
python notebooks/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/fraud_detection.ipynb
```

`train.py --beta 1` would instead pick the F1-optimal threshold (≈0.16: F1 0.28, recall 0.22), and
`--beta 2` the F2-optimal one (≈0.055: F1 0.22, recall 0.37).

## Scoring new transactions

```python
import pickle, pandas as pd
bundle = pickle.load(open("outputs/ebm_model.pkl", "rb"))
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]
df = pd.read_csv("new_transactions.csv")
p = model.predict_proba(df[features])[:, 1]
flag = p >= threshold
```
