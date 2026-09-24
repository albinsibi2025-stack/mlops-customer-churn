import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

mlflow_db = os.path.join(PROJECT_ROOT, "mlflow.db")

mlflow.set_tracking_uri(
    "sqlite:///" + mlflow_db.replace("\\", "/")
)

# Load test data
test_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "test.csv"
)

test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["churn"])
y_test = test_df["churn"]

# Load the latest logged model
experiment = mlflow.get_experiment_by_name(
    "customer-churn-prediction"
)

if experiment is None:
    raise RuntimeError("MLflow experiment not found.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError("No MLflow runs found.")

run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/customer_churn_model"

print(f"Loading model from run: {run_id}")

model = mlflow.sklearn.load_model(model_uri)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")