
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experimental" / "features_v3"
OUTPUT_DIR = ROOT / "data" / "experimental"

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_train = train.drop(columns=["churn"])
y_train = train["churn"]
X_val = validation.drop(columns=["churn"])
y_val = validation["churn"]
X_test = test.drop(columns=["churn"])
y_test = test["churn"]

# Keep the same model configurations used in V3.
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(random_state=42),
    ),
    "Hist Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(random_state=42),
    ),
}

thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
results = []
fitted_models = {}

print("Comparing models using validation F1...")

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    probabilities = model.predict_proba(X_val)[:, 1]
    candidates = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        candidates.append({
            "Model": name,
            "Threshold": float(threshold),
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_val, probabilities),
        })

    model_results = pd.DataFrame(candidates).sort_values(
        ["F1", "Precision"],
        ascending=[False, False],
    )

    results.append(model_results.iloc[0].to_dict())

results_df = pd.DataFrame(results).sort_values(
    ["F1", "Precision"],
    ascending=[False, False],
).reset_index(drop=True)

print("\nBEST VALIDATION THRESHOLD FOR EACH MODEL")
print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))

results_path = OUTPUT_DIR / "v3_model_f1_comparison.csv"
results_df.to_csv(results_path, index=False)
print(f"\nValidation comparison saved to: {results_path}")

# Choose the winning model using validation F1 only.
winner = results_df.iloc[0]
winner_name = winner["Model"]
winner_threshold = float(winner["Threshold"])
winner_model = fitted_models[winner_name]

print(f"\nSelected model: {winner_name}")
print(f"Selected threshold: {winner_threshold:.2f}")
print(f"Validation F1: {winner['F1']:.4f}")
print(f"Validation precision: {winner['Precision']:.4f}")
print(f"Validation recall: {winner['Recall']:.4f}")

# Final evaluation: test set is not used for model/threshold selection.
test_probabilities = winner_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= winner_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

print("\nFINAL HELD-OUT TEST RESULTS")
print("===========================")
print(f"Model     : {winner_name}")
print(f"Threshold : {winner_threshold:.2f}")
print(f"Accuracy  : {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision : {precision_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F1 score  : {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, test_probabilities):.4f}")

print("\nCONFUSION MATRIX COUNTS")
print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")

print("\nNote: results are from a synthetic customer dataset.")
