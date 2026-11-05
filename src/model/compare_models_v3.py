
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experimental" / "features_v3"

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["churn"])
y_train = train["churn"]

X_val = validation.drop(columns=["churn"])
y_val = validation["churn"]

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

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(y_val, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, probabilities),
    })

results_df = pd.DataFrame(results).sort_values(
    by="ROC_AUC", ascending=False
)

print("\nV3 Validation Results (ranked by ROC AUC)")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

output_path = ROOT / "data" / "experimental" / "v3_model_comparison.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")


# Final evaluation of the selected model on the held-out test set
from sklearn.base import clone

test = pd.read_csv(DATA_DIR / "test.csv")
X_test = test.drop(columns=["churn"])
y_test = test["churn"]

selected_model = clone(models["Random Forest"])
selected_model.fit(X_train, y_train)

test_predictions = selected_model.predict(X_test)
test_probabilities = selected_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Results — Random Forest")
print("----------------------------------")
print(f"Accuracy : {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, test_probabilities):.4f}")


from sklearn.dummy import DummyClassifier

# Baseline: always predict the most common class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nMajority-Class Baseline — Test Results")
print("--------------------------------------")
print(
    f"Accuracy : "
    f"{accuracy_score(y_test, baseline_predictions):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_test, baseline_predictions, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_test, baseline_predictions, zero_division=0):.4f}"
)
print(
    f"F1 Score : "
    f"{f1_score(y_test, baseline_predictions, zero_division=0):.4f}"
)


# Threshold tuning using validation data only
from sklearn.metrics import confusion_matrix

validation_probabilities = selected_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
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
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nRandom Forest — Validation Threshold Comparison")
print(threshold_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

best_row = threshold_df.loc[threshold_df["F1"].idxmax()]

print("\nBest threshold by validation F1:")
print(best_row.to_string())

# Final test evaluation at the selected threshold
# Threshold was selected using validation data only.

test_probabilities = selected_model.predict_proba(X_test)[:, 1]
test_predictions_030 = (test_probabilities >= 0.30).astype(int)

print("\nFinal Test Results — Random Forest (Threshold 0.30)")
print("---------------------------------------------------")
print(f"Accuracy : {accuracy_score(y_test, test_predictions_030):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions_030, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_predictions_030, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, test_predictions_030, zero_division=0):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, test_probabilities):.4f}")