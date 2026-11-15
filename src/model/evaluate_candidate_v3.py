import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "experimental"
    / "features_v3"
)

MODEL_DIR = PROJECT_ROOT / "models" / "experimental"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

CANDIDATE_MODEL = MODEL_DIR / "candidate_v3.joblib"
CANDIDATE_METADATA = MODEL_DIR / "candidate_v3_metadata.json"

BASELINE_MODEL = MODEL_DIR / "logistic_regression_v3.joblib"
BASELINE_METADATA = MODEL_DIR / "logistic_regression_v3_metadata.json"

TARGET = "churn"

TOLERANCE = 1e-9

# Small tolerance for floating-point comparisons.
TOLERANCE = 1e-9


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }


def load_dataset(path, feature_columns):
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing in {path}")

    missing = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features in {path}: {missing}"
        )

    X = df[feature_columns]
    y = df[TARGET].astype(int)

    return X, y


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * 60)

    print(f"Threshold : {metrics['threshold']:.2f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    cm = metrics["confusion_matrix"]

    print("\nConfusion Matrix")
    print(f"TN: {cm['TN']} | FP: {cm['FP']}")
    print(f"FN: {cm['FN']} | TP: {cm['TP']}")


def main():
    print("=" * 65)
    print("CUSTOMER CHURN V3 - CANDIDATE EVALUATION")
    print("=" * 65)

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
        CANDIDATE_MODEL,
        CANDIDATE_METADATA,
        BASELINE_MODEL,
        BASELINE_METADATA,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    candidate_metadata = load_json(CANDIDATE_METADATA)
    baseline_metadata = load_json(BASELINE_METADATA)

    feature_columns = candidate_metadata["feature_columns"]

    candidate_threshold = float(candidate_metadata["threshold"])
    baseline_threshold = float(baseline_metadata["threshold"])

    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Candidate threshold: {candidate_threshold:.2f}")
    print(f"Baseline threshold:  {baseline_threshold:.2f}")

    candidate_model = joblib.load(CANDIDATE_MODEL)
    baseline_model = joblib.load(BASELINE_MODEL)

    # Load datasets using the candidate's recorded feature order.
    X_train, _ = load_dataset(TRAIN_FILE, feature_columns)
    X_val, y_val = load_dataset(VALIDATION_FILE, feature_columns)
    X_test, y_test = load_dataset(TEST_FILE, feature_columns)

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows:       {len(X_test)}")

    # ---------------------------------------------------------
    # 1. Validation comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION SET COMPARISON")
    print("=" * 65)

    baseline_validation = calculate_metrics(
        baseline_model,
        X_val,
        y_val,
        baseline_threshold,
    )

    candidate_validation = calculate_metrics(
        candidate_model,
        X_val,
        y_val,
        candidate_threshold,
    )

    print_metrics("BASELINE - VALIDATION", baseline_validation)
    print_metrics("CANDIDATE - VALIDATION", candidate_validation)

    # Quality gate uses validation metrics only.
    f1_pass = (
        candidate_validation["f1"]
        >= baseline_validation["f1"] - TOLERANCE
    )

    precision_pass = (
        candidate_validation["precision"]
        >= baseline_validation["precision"] - TOLERANCE
    )

    quality_gate_passed = f1_pass and precision_pass

    print("\n" + "=" * 65)
    print("VALIDATION QUALITY GATE")
    print("=" * 65)

    print(
        f"F1 comparison: "
        f"{candidate_validation['f1']:.4f} vs "
        f"{baseline_validation['f1']:.4f} "
        f"-> {'PASS' if f1_pass else 'FAIL'}"
    )

    print(
        f"Precision comparison: "
        f"{candidate_validation['precision']:.4f} vs "
        f"{baseline_validation['precision']:.4f} "
        f"-> {'PASS' if precision_pass else 'FAIL'}"
    )

    print(
        "\nQUALITY GATE:",
        "PASSED" if quality_gate_passed else "FAILED",
    )

    # ---------------------------------------------------------
    # 2. Candidate test evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 65)

    candidate_test = calculate_metrics(
        candidate_model,
        X_test,
        y_test,
        candidate_threshold,
    )

    print_metrics("CANDIDATE - TEST", candidate_test)

    # ---------------------------------------------------------
    # 3. Save evaluation report
    # ---------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "candidate_model": candidate_metadata.get(
            "model_name", "Unknown"
        ),
        "candidate_threshold": candidate_threshold,
        "baseline_threshold": baseline_threshold,
        "quality_gate": {
            "passed": bool(quality_gate_passed),
            "criteria": {
                "validation_f1_not_below_baseline": bool(f1_pass),
                "validation_precision_not_below_baseline": bool(
                    precision_pass
                ),
            },
        },
        "validation": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
        },
        "test": {
            "candidate": candidate_test,
            "note": (
                "Held-out test metrics are reported only. "
                "They are not used by the quality gate."
            ),
        },
    }

    report_path = REPORT_DIR / "candidate_v3_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("EVALUATION REPORT SAVED")
    print("=" * 65)
    print(report_path)

    if quality_gate_passed:
        print("\nCANDIDATE PASSED VALIDATION QUALITY GATE.")
        print("Candidate is eligible for the next registration stage.")
    else:
        print("\nCANDIDATE FAILED VALIDATION QUALITY GATE.")
        print("Do not register this candidate for promotion.")

    print("\nCANDIDATE EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()