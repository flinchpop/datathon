"""Load model.pkl in a fresh interpreter from a neutral directory and score the test CSV.

python3 verify_model_pkl.py [model.pkl] [data/test.csv] [outputs/submission.csv]
Proves: plain pickle.load works without any project module on the path, predictions are
batch-independent, and (if a reference submission is given) match it.
"""
import sys, os, subprocess, numpy as np, pandas as pd
pkl = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "model.pkl")
csv = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "data/test.csv")
ref = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else (os.path.abspath("outputs/submission.csv") if os.path.exists("outputs/submission.csv") else "")
code = r'''
import sys, pickle, numpy as np, pandas as pd
assert "fraud_model" not in sys.modules and "src" not in sys.modules
m = pickle.load(open(sys.argv[1], "rb"))
assert "fraud_model" not in sys.modules and "src" not in sys.modules, "pickle imported project modules"
X = pd.read_csv(sys.argv[2])
p = m.predict_proba(X); l = m.predict(X)
assert p.shape == (len(X), 2) and set(np.unique(l)) <= {0, 1}
assert np.allclose(m.predict_proba(X.iloc[:300])[:, 1], p[:300, 1]), "batch dependence"
pickle.loads(pickle.dumps(m)).predict(X.head(3))          # re-pickle round trip
print("loaded OK:", type(m).__name__, "| predict_proba", p.shape, "| flagged", int(l.sum()), f"({l.mean()*100:.2f}%)")
print("libs:", *[f"{k}={sys.modules[k].__version__}" for k in ("numpy","pandas","sklearn","xgboost","lightgbm","catboost") if k in sys.modules])
if sys.argv[3]:
    ref = pd.read_csv(sys.argv[3]); assert (ref.id.values == X.id.values).all()
    d = np.max(np.abs(p[:, 1] - ref.fraud.values)); print(f"max |score - reference submission| = {d:.2e}")
    assert d < 1e-4, "does not reproduce reference submission"
print("VERIFY OK")
'''
r = subprocess.run([sys.executable, "-c", code, pkl, csv, ref], cwd="/tmp", capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr); sys.exit(1)
