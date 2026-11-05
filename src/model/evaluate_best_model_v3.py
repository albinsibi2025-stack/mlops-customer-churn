
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "experimental"
    / "logistic_regression_v3.joblib"
)
METADATA_PATH = (
    ROOT / "models" / "experimental"
    / "logistic_regression_v3_metadata.json"
)
TEST_PATH = (
    ROOT / "data" / "experimental"
    / "features_v3" / "test.csv"
)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = float(metadata.get("threshold", 0.34))

test_df = pd.read_csv(TEST_PATH)
X_test = test_df.drop(columns=["churn"])
y_test = test_df["churn"]

# Align the test features with the model's training feature order.
if hasattr(model, "feature_names_in_"):
    X_test = X_test[list(model.feature_names_in_)]

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nLOGISTIC REGRESSION V3 EVALUATION")
print("=================================")
print(f"Model     : {MODEL_PATH.name}")
print(f"Test data : {TEST_PATH}")
print(f"Test rows : {len(y_test)}")
print(f"Threshold : {threshold:.2f}")
print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision : {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, probabilities):.4f}")

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\nCONFUSION MATRIX COUNTS")
print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")
