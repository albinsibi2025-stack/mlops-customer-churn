
from pathlib import Path

import mlflow
import mlflow.sklearn
import joblib

from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "experimental" / "logistic_regression_v3.joblib"
METADATA_PATH = ROOT / "models" / "experimental" / "logistic_regression_v3_metadata.json"

TRACKING_URI = "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")
MODEL_NAME = "customer-churn-logistic-regression-v3"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("customer-churn-v3")

# Load the model and metadata saved by the previous script.
model = joblib.load(MODEL_PATH)

import json

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = float(metadata.get("threshold", 0.34))

# Generate a representative signature using the expected 16 numeric features.
import pandas as pd

feature_count = int(metadata.get("number_of_features", 16))
input_example = pd.DataFrame(
    [[0.0] * feature_count],
    columns=[f"feature_{i}" for i in range(feature_count)],
)

# Use the model's actual feature names when available.
if hasattr(model, "feature_names_in_"):
    input_example.columns = list(model.feature_names_in_)

try:
    prediction_example = model.predict_proba(input_example)[:, 1]
    signature = infer_signature(
        input_example,
        {"churn_probability": prediction_example},
    )
except Exception as exc:
    raise RuntimeError(
        "Could not create a model signature from the saved model. "
        "Check its expected feature columns."
    ) from exc

with mlflow.start_run(run_name="logistic-regression-v3-registration") as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("classification_threshold", threshold)
    mlflow.log_param("number_of_features", feature_count)

    # Metrics come from the previous validation/test evaluation.
    mlflow.log_metric("validation_f1", 0.5657)
    mlflow.log_metric("validation_precision", 0.5270)
    mlflow.log_metric("validation_recall", 0.6106)
    mlflow.log_metric("test_f1", 0.5757)
    mlflow.log_metric("test_precision", 0.5153)
    mlflow.log_metric("test_recall", 0.6522)
    mlflow.log_metric("test_accuracy", 0.7347)
    mlflow.log_metric("test_roc_auc", 0.7892)


    mlflow.sklearn.log_model(
        sk_model=model,
        name="logistic_regression_v3",
        signature=signature,
        input_example=input_example,
        skops_trusted_types=["numpy.dtype"],
    )


    run_id = run.info.run_id

registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/logistic_regression_v3",
    name=MODEL_NAME,
)

print("\nLogistic Regression V3 registered successfully.")
print(f"Registered model: {MODEL_NAME}")
print(f"Version: {registered.version}")
print(f"Threshold: {threshold}")
print(f"MLflow run ID: {run_id}")
