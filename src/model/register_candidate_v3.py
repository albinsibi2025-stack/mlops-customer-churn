
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "experimental"
DATA_DIR = ROOT / "data" / "experimental" / "features_v3"
REPORT_DIR = ROOT / "reports" / "model_evaluation"

MODEL_PATH = MODEL_DIR / "candidate_v3.joblib"
METADATA_PATH = MODEL_DIR / "candidate_v3_metadata.json"
TRAIN_FILE = DATA_DIR / "train.csv"
EVALUATION_REPORT = REPORT_DIR / "candidate_v3_evaluation.json"

TRACKING_URI = "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")

EXPERIMENT_NAME = "customer-churn-v3"

# Keep the candidate separate from the existing registered model.
MODEL_NAME = "customer-churn-logistic-regression-v3-candidate"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("\nCUSTOMER CHURN V3 - CANDIDATE REGISTRATION")
    print("=" * 65)

    # Verify required files.
    required_files = [
        MODEL_PATH,
        METADATA_PATH,
        TRAIN_FILE,
        EVALUATION_REPORT,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    metadata = load_json(METADATA_PATH)
    evaluation = load_json(EVALUATION_REPORT)

    # Never register a candidate that failed the validation gate.
    if not evaluation["quality_gate"]["passed"]:
        raise RuntimeError(
            "Candidate failed the validation quality gate. "
            "Registration cancelled."
        )

    feature_columns = metadata["feature_columns"]
    threshold = float(metadata["threshold"])

    candidate_validation = evaluation["validation"]["candidate"]
    baseline_validation = evaluation["validation"]["baseline"]
    candidate_test = evaluation["test"]["candidate"]

    print(f"Candidate model: {metadata['model_name']}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Features: {len(feature_columns)}")
    print(f"Validation quality gate: PASSED")

    # Load candidate model.
    model = joblib.load(MODEL_PATH)

    # Use a real representative training row for the model signature.
    train_df = pd.read_csv(TRAIN_FILE)

    missing_features = [
        column for column in feature_columns
        if column not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing training features: {missing_features}"
        )

    input_example = train_df[feature_columns].head(5).copy()

    if input_example.empty:
        raise ValueError("Training dataset is empty.")

    probability_example = model.predict_proba(input_example)[:, 1]

    output_example = pd.DataFrame({
        "churn_probability": probability_example
    })

    signature = infer_signature(
        input_example,
        output_example,
    )

    # Connect to the existing MLflow database and experiment.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nMLflow tracking URI: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Registered model name: {MODEL_NAME}")

    with mlflow.start_run(
        run_name="customer-churn-v3-candidate-registration"
    ) as run:

        # Parameters.
        mlflow.log_param("model_name", metadata["model_name"])
        mlflow.log_param("model_version", metadata["model_version"])
        mlflow.log_param("classification_threshold", threshold)
        mlflow.log_param("number_of_features", len(feature_columns))
        mlflow.log_param("selection_metric", "validation_f1")
        mlflow.log_param("quality_gate", "passed")

        # Candidate validation metrics.
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]:
            mlflow.log_metric(
                f"validation_{metric}",
                float(candidate_validation[metric]),
            )

        # Baseline validation metrics for comparison.
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]:
            mlflow.log_metric(
                f"baseline_validation_{metric}",
                float(baseline_validation[metric]),
            )

        # Candidate held-out test metrics.
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]:
            mlflow.log_metric(
                f"test_{metric}",
                float(candidate_test[metric]),
            )

        # Candidate test confusion matrix.
        for key, value in candidate_test["confusion_matrix"].items():
            mlflow.log_metric(
                f"test_{key}",
                float(value),
            )

        # Log the model with its input/output signature.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="candidate_v3",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["numpy.dtype"],
        )

        # Keep the evaluation report and model metadata with the run.
        mlflow.log_artifact(
            str(EVALUATION_REPORT),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(METADATA_PATH),
            artifact_path="metadata",
        )

        run_id = run.info.run_id

    # Register as a separate model, preserving the original model.
    model_uri = f"runs:/{run_id}/candidate_v3"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n" + "=" * 65)
    print("CANDIDATE REGISTERED SUCCESSFULLY")
    print("=" * 65)

    print(f"Registered model: {MODEL_NAME}")
    print(f"Version: {registered.version}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Validation F1: {candidate_validation['f1']:.4f}")
    print(f"Test F1: {candidate_test['f1']:.4f}")
    print(f"MLflow run ID: {run_id}")

    print("\nOriginal registered model was not modified.")


if __name__ == "__main__":
    main()
