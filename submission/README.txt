Upload these three files to the platform (same folder, these names):
  predict_notebook.ipynb   prediction notebook (no training; reads DATATHON_INPUT_PATH, writes DATATHON_OUTPUT_PATH)
  model.pkl                self-contained FraudEnsemble (features + 5 members x 5 seeds + blend + operating point)
  requirements.txt         catboost / lightgbm pins, only used if the platform does not pre-install them

Prediction-time libraries: numpy, pandas, lightgbm, catboost (XGBoost trees are evaluated in numpy; sklearn not needed).
Verified (see ../verify_model_pkl.py and REPORT.md 5.6): plain pickle.load / joblib.load from any directory,
shuffled rows, no id column, header-less input, batch independence, byte-identical re-runs, and the same
predictions under numpy 1.26 / pandas 2.2 / lightgbm 4.5 / catboost 1.2.7 as under the training stack.
