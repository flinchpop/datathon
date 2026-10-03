"""Package the saved XGBoost model (model/xgb_fraud.json + model/metadata.json) as model.pkl.

The pickle holds one object that takes the RAW transactions table (same columns as
Track_2_Testing_Dataset.csv) and does the feature engineering itself:

    import pickle, pandas as pd
    model = pickle.load(open("model.pkl", "rb"))
    df = pd.read_csv("Track_2_Testing_Dataset.csv")
    proba = model.predict_proba(df)[:, 1]   # fraud probability
    flags = model.predict(df)               # 0/1 at the saved threshold

The class and feature code are stored inside the pickle (by value), so loading it only
needs numpy, pandas, xgboost and cloudpickle (installed with scikit-learn), not this
repository.

Usage: python export_model_pkl.py   ->  writes model.pkl
"""
import json
import os
import pickle

import cloudpickle
import xgboost as xgb

import train_xgboost as tx

OUT_PATH = "model.pkl"


class FraudModel:
    """Raw transactions DataFrame in, fraud probabilities out."""

    def __init__(self, classifier, encoders, features, threshold):
        self.classifier = classifier
        self.encoders = encoders
        self.features = features
        self.threshold = threshold

    def transform(self, df):
        """The 30 model features, built exactly as in train_xgboost.prepare_features."""
        return tx.prepare_features(df, self.encoders)[self.features]

    def predict_proba(self, df):
        return self.classifier.predict_proba(self.transform(df))

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= self.threshold).astype(int)


def main():
    with open(os.path.join(tx.MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(tx.MODEL_DIR, "xgb_fraud.json"))
    model = FraudModel(clf, meta["encoders"], meta["features"], meta["threshold"])

    # Store this module's class and train_xgboost's feature code inside the pickle.
    cloudpickle.register_pickle_by_value(tx)
    with open(OUT_PATH, "wb") as f:
        cloudpickle.dump(model, f, protocol=pickle.DEFAULT_PROTOCOL)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)  threshold {model.threshold:.3f}")


if __name__ == "__main__":
    main()
