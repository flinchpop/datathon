# Jupyter cell: package the trained XGBoost model as model.pkl.
#
# Run this in the cell BELOW the train_xgboost.py cell. That cell defines
# prepare_features, MODEL_DIR, TARGET, CATEGORICAL, ... in the notebook and its
# main() saves model/xgb_fraud.json and model/metadata.json, which are loaded here.
#
# The pickle holds one object that takes the RAW transactions table (same columns as
# Track_2_Testing_Dataset.csv) and builds the features itself:
#
#     import pickle, pandas as pd
#     model = pickle.load(open("model.pkl", "rb"))
#     df = pd.read_csv("Track_2_Testing_Dataset.csv")
#     proba = model.predict_proba(df)[:, 1]   # fraud probability
#     flags = model.predict(df)               # 0/1 at the saved threshold
#
# cloudpickle stores notebook-defined code (FraudModel, prepare_features and the
# constants it uses) inside the pickle, so loading it needs only numpy, pandas,
# xgboost and cloudpickle (installed with scikit-learn), not the notebook.
import json
import os
import pickle

import cloudpickle
import xgboost as xgb

PKL_PATH = "model.pkl"


class FraudModel:
    """Raw transactions DataFrame in, fraud probabilities out."""

    def __init__(self, classifier, encoders, features, threshold):
        self.classifier = classifier
        self.encoders = encoders
        self.features = features
        self.threshold = threshold

    def transform(self, df):
        """The 30 model features, built by prepare_features from the train_xgboost cell."""
        return prepare_features(df, self.encoders)[self.features]

    def predict_proba(self, df):
        return self.classifier.predict_proba(self.transform(df))

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= self.threshold).astype(int)


with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)
classifier = xgb.XGBClassifier()
classifier.load_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))
fraud_model = FraudModel(classifier, metadata["encoders"], metadata["features"], metadata["threshold"])

with open(PKL_PATH, "wb") as f:
    cloudpickle.dump(fraud_model, f, protocol=pickle.DEFAULT_PROTOCOL)
print(f"Wrote {PKL_PATH} ({os.path.getsize(PKL_PATH) / 1024:.0f} KB)  threshold {fraud_model.threshold:.3f}")
