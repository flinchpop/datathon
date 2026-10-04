"""Fit the final ensemble and write a self-contained model.pkl; verify it end-to-end.

python3 build_model_pkl.py [--seeds 5] [--flag-share 0.015]
"""
import sys, os, json, time, pickle, argparse, subprocess, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import fraud_model as fm

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--flag-share", type=float, default=0.015)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")

# 0. standalone feature code must equal the experiment code -----------------------------
from src.features import add_basic_features as ab_src, add_merchant_relative_features as am_src, FEATURE_SETS as FS_SRC
a = am_src(ab_src(train), ab_src(train))[FS_SRC["full"]].astype(float)
raw = fm._coerce_input(train)
b = fm.add_merchant_relative_features(fm.add_basic_features(raw), fm.fit_merchant_stats(raw))[fm.FEATURE_SETS["full"]].astype(float)
diff = np.nanmax(np.abs(a.values - b.values)); assert diff < 1e-9 and (np.isnan(a.values) == np.isnan(b.values)).all(), diff
print(f"[0] feature parity with src/features.py OK (max abs diff {diff:.1e})")

# 1. fit ---------------------------------------------------------------------------------
t0 = time.time()
model = fm.FraudEnsemble(seeds=[1000 + i for i in range(args.seeds)], flag_share=args.flag_share,
                         threads=args.threads, source=fm.module_source())
model.fit(train, reference=test)
print(f"[1] fitted {len(fm.MEMBERS)} members x {args.seeds} seeds in {time.time()-t0:.0f}s; "
      f"threshold={model.threshold_:.6f} blend range=[{model.blend_min_:.4f},{model.blend_max_:.4f}]")

# 2. reproduce the submitted files -------------------------------------------------------
full = pd.read_csv("outputs/submission_full.csv"); sub = pd.read_csv("outputs/submission.csv")
mp = model._member_probs(fm._coerce_input(test))
for n, p in mp.items():
    print(f"    member {n:17s} max|Δ| vs run_final = {np.max(np.abs(p - full[f'p_{n}'].values)):.2e}")
score = model.predict_proba(test)[:, 1]; label = model.predict(test)
from scipy.stats import spearmanr
print(f"[2] score vs outputs/submission.csv: spearman={spearmanr(score, sub.fraud.values).correlation:.6f} "
      f"max|Δ|={np.max(np.abs(score - sub.fraud.values)):.2e}; flagged {label.sum()} (submitted {int((sub.fraud>=0.5).sum())})")

# 3. batch independence + input flexibility ----------------------------------------------
s_sub = model.predict_proba(test.iloc[:500])[:, 1]; assert np.allclose(s_sub, score[:500]), "batch dependence!"
s_noid = model.predict_proba(test.drop(columns="id"))[:, 1]; assert np.allclose(s_noid, score)
s_np = model.predict_proba(test.values)[:, 1]; assert np.allclose(s_np, score)
s_shuf = model.predict_proba(test[list(reversed(test.columns))])[:, 1]; assert np.allclose(s_shuf, score)
print("[3] batch-independent; accepts DataFrame with/without id, shuffled columns, and ndarray")

# 4. pickle ------------------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)
print(f"[4] wrote model.pkl ({os.path.getsize('model.pkl')/1e6:.1f} MB)")

# 5. load in a clean interpreter from another directory (no repo modules importable) -----
r = subprocess.run([sys.executable, "verify_model_pkl.py", "model.pkl", "data/test.csv", "outputs/submission.csv"])
if r.returncode != 0: sys.exit("clean-environment load FAILED")
print("[5] clean-environment load + predict reproduces outputs/submission.csv")
json.dump(dict(flag_share=args.flag_share, seeds=model.seeds, threshold=model.threshold_, n_flagged_test=int(label.sum()),
               size_mb=os.path.getsize("model.pkl")/1e6), open("outputs/model_pkl_info.json", "w"), indent=2)
