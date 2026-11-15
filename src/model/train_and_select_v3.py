
from pathlib import Path
import json

import joblib
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
OUTPUT_DIR = ROOT / "models" / "experimental"

MODEL_PATH = OUTPUT_DIR / "candidate_v3.joblib"
METADATA_PATH = OUTPUT_DIR / "candidate_v3_metadata.json"
COMPARISON_PATH = (
    ROOT / "data" / "experimental" / "v3_model_f1_comparison.csv"
)

THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train.drop(columns=["churn"])
    y_train = train["churn"]

    X_val = validation.drop(columns=["churn"])
    y_val = validation["churn"]

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation features do not match.")

    return X_train, y_train, X_val, y_val


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                random_state=42,
            ),
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


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    print("\nCUSTOMER CHURN V3 - AUTOMATED MODEL SELECTION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Features: {len(X_train.columns)}")

    models = build_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        fitted_models[name] = model

        probabilities = model.predict_proba(X_val)[:, 1]

        best_metrics = None

        for threshold in THRESHOLDS:
            metrics = calculate_metrics(
                y_val,
                probabilities,
                threshold,
            )

            if best_metrics is None or (
                metrics["f1"],
                metrics["precision"],
            ) > (
                best_metrics["f1"],
                best_metrics["precision"],
            ):
                best_metrics = metrics

        results.append({
            "model_name": name,
            **best_metrics,
        })

        print(
            f"Best validation F1: {best_metrics['f1']:.4f} | "
            f"Precision: {best_metrics['precision']:.4f} | "
            f"Recall: {best_metrics['recall']:.4f} | "
            f"ROC-AUC: {best_metrics['roc_auc']:.4f} | "
            f"Threshold: {best_metrics['threshold']:.2f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results_df.to_csv(COMPARISON_PATH, index=False)

    winner = results_df.iloc[0]
    winner_name = winner["model_name"]
    winner_model = fitted_models[winner_name]

    print("\nMODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\nSELECTED CANDIDATE")
    print(f"Model: {winner_name}")
    print(f"Threshold: {winner['threshold']:.2f}")
    print(f"Validation F1: {winner['f1']:.4f}")

    metadata = {
        "model_name": winner_name,
        "model_version": "v3-candidate",
        "threshold": float(winner["threshold"]),
        "selection_metric": "validation_f1",
        "validation_metrics": {
            "accuracy": float(winner["accuracy"]),
            "precision": float(winner["precision"]),
            "recall": float(winner["recall"]),
            "f1": float(winner["f1"]),
            "roc_auc": float(winner["roc_auc"]),
        },
        "feature_columns": list(X_train.columns),
        "number_of_features": len(X_train.columns),
        "target_column": "churn",
        "dataset": "synthetic customer churn V3",
        "note": (
            "Candidate selected using validation data. "
            "Test data has not been used for model selection."
        ),
    }

    joblib.dump(winner_model, MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Reload candidate and verify its prediction interface.
    loaded_model = joblib.load(MODEL_PATH)

    sample_probabilities = loaded_model.predict_proba(
        X_val.head(5)
    )[:, 1]

    sample_predictions = (
        sample_probabilities >= metadata["threshold"]
    ).astype(int)

    print("\nCANDIDATE SAVED AND VERIFIED")
    print(f"Model path: {MODEL_PATH}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Sample probabilities: {sample_probabilities.round(4)}")
    print(f"Sample predictions: {sample_predictions}")

    print("\nTRAINING AND SELECTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
