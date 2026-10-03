"""Adversarial validation: how different is the test set from the training set?

Trains a classifier to tell training rows from test rows. AUC 0.5 = identical
distributions; higher = more shift. Each training row gets a "test-likeness"
weight p(test)/p(train), used by the later scripts to compute a test-weighted
PR-AUC (a rough preview of the leaderboard).

Writes: results/adversarial_summary.csv, results/adversarial_weights.csv,
        results/adversarial_feature_importance.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import RESULTS, raw_features, test, train

X = pd.concat([raw_features(train), raw_features(test)], ignore_index=True)
is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]

p = np.zeros(len(X))
gain = []
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, is_test):
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, enable_categorical=True,
                          tree_method="hist", random_state=0)
    m.fit(X.iloc[a], is_test[a])
    p[b] = m.predict_proba(X.iloc[b])[:, 1]
    gain.append(pd.Series(m.get_booster().get_score(importance_type="gain")))
auc = roc_auc_score(is_test, p)

p_train = np.clip(p[:len(train)], 0.01, 0.99)
w = p_train / (1 - p_train) * len(train) / len(test)
w = w / w.mean()
ess = w.sum() ** 2 / (w ** 2).sum()

pd.DataFrame({"id": train["id"], "p_test": p[:len(train)], "weight": w}).to_csv(
    RESULTS / "adversarial_weights.csv", index=False)
imp = pd.concat(gain, axis=1).mean(axis=1)
(imp / imp.sum() * 100).sort_values(ascending=False).round(1).rename("share_of_gain_%").to_csv(
    RESULTS / "adversarial_feature_importance.csv", index_label="column")
pd.DataFrame([{"adversarial_auc": round(auc, 4), "effective_train_rows": round(ess),
               "weight_median": round(float(np.median(w)), 3), "weight_max": round(float(w.max()), 2)}]).to_csv(
    RESULTS / "adversarial_summary.csv", index=False)
print(f"Train-vs-test AUC {auc:.4f}  effective training rows under test weighting {ess:.0f}")
print("Columns that differ most between train and test (share of gain %):")
print((imp / imp.sum() * 100).sort_values(ascending=False).round(1).to_string())
