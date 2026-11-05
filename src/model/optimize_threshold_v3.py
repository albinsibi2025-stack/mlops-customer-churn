
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experimental" / "features_v3"
OUTPUT_DIR = ROOT / "data" / "experimental"

# Load the existing V3 splits
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_train = train.drop(columns=["churn"])
y_train = train["churn"]

X_val = validation.drop(columns=["churn"])
y_val = validation["churn"]

X_test = test.drop(columns=["churn"])
y_test = test["churn"]

# Use the same Random Forest configuration as V3
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
)

print("Training Random Forest V3...")
model.fit(X_train, y_train)

# Select threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

results = []

for threshold in np.arange(0.10, 0.901, 0.01):
    predictions = (val_probabilities >= threshold).astype(int)

    results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(y_val, predictions, zero_division=0),
    })

results_df = pd.DataFrame(results)

# Highest F1 first; use precision to break F1 ties
ranked = results_df.sort_values(
    ["F1", "Precision"],
    ascending=[False, False],
)

best = ranked.iloc[0]
threshold = float(best["Threshold"])

output_path = OUTPUT_DIR / "v3_threshold_optimization.csv"
ranked.to_csv(output_path, index=False)

print("\nTop 10 validation thresholds ranked by F1:")
print(
    ranked.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(f"\nSelected threshold: {threshold:.2f}")
print(f"Validation F1: {best['F1']:.4f}")
print(f"Validation precision: {best['Precision']:.4f}")
print(f"Validation recall: {best['Recall']:.4f}")
print(f"Threshold results saved to: {output_path}")

# Evaluate the selected threshold on the held-out test set.
# Do not use test results to select another threshold.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

print("\nFINAL HELD-OUT TEST RESULTS")
print("===========================")
print(f"Selected threshold: {threshold:.2f}")
print(f"Accuracy : {accuracy_score(y_test, test_predictions):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(f"F1 score : {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, test_probabilities):.4f}")

print("\nCONFUSION MATRIX COUNTS")
print("=======================")
print(f"True negatives  (correctly predicted stay): {tn}")
print(f"False positives (unnecessary retention):   {fp}")
print(f"False negatives (missed churners):         {fn}")
print(f"True positives  (churners identified):     {tp}")
