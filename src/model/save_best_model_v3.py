
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experimental" / "features_v3"
OUTPUT_DIR = ROOT / "models" / "experimental"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["churn"])
y_train = train["churn"]
X_val = validation.drop(columns=["churn"])
y_val = validation["churn"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=3000, random_state=42),
)

model.fit(X_train, y_train)

# Selected using validation F1 in the model-comparison experiment.
threshold = 0.34

model_path = OUTPUT_DIR / "logistic_regression_v3.joblib"
metadata_path = OUTPUT_DIR / "logistic_regression_v3_metadata.json"

joblib.dump(model, model_path)

metadata = {
    "model_name": "customer-churn-logistic-regression-v3",
    "model_type": "Logistic Regression",
    "version": 3,
    "threshold": threshold,
    "selection_metric": "validation_f1",
    "validation_f1": 0.5657,
    "validation_precision": 0.5270,
    "validation_recall": 0.6106,
    "feature_columns": list(X_train.columns),
    "target_column": "churn",
    "dataset": "synthetic customer churn V3",
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Best candidate model saved successfully.")
print(f"Model: {model_path}")
print(f"Metadata: {metadata_path}")
print(f"Threshold: {threshold}")
print(f"Number of features: {len(X_train.columns)}")

# Verify that the saved model can produce predictions.
loaded_model = joblib.load(model_path)
sample_probabilities = loaded_model.predict_proba(X_val.head(5))[:, 1]
sample_predictions = (sample_probabilities >= threshold).astype(int)

print("\nSample churn probabilities:", sample_probabilities.round(4))
print("Sample predictions:", sample_predictions)
