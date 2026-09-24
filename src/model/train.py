import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ============================================================
# MLflow CONFIGURATION
# ============================================================

mlflow_db = os.path.join(PROJECT_ROOT, "mlflow.db")

mlflow.set_tracking_uri(
    "sqlite:///" + mlflow_db.replace("\\", "/")
)

mlflow.set_experiment("customer-churn-prediction")


# ============================================================
# LOAD DATA
# ============================================================

train_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "train.csv"
)

validation_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "validation.csv"
)

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)

X_train = train_df.drop(columns=["churn"])
y_train = train_df["churn"]

X_validation = validation_df.drop(columns=["churn"])
y_validation = validation_df["churn"]


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# BASE MODEL
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"]
}


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


print("\nStarting hyperparameter tuning...")
print("--------------------------------")

random_search.fit(X_train, y_train)


# ============================================================
# BEST MODEL
# ============================================================

best_model = random_search.best_estimator_

best_params = random_search.best_params_
best_cv_score = random_search.best_score_


print("\nHyperparameter tuning completed.")
print("--------------------------------")
print("Best Parameters:")

for parameter, value in best_params.items():
    print(f"{parameter}: {value}")

print(f"\nBest CV ROC AUC: {best_cv_score:.4f}")


# ============================================================
# VALIDATION EVALUATION
# ============================================================

y_pred = best_model.predict(X_validation)
y_prob = best_model.predict_proba(X_validation)[:, 1]

accuracy = accuracy_score(y_validation, y_pred)
precision = precision_score(y_validation, y_pred)
recall = recall_score(y_validation, y_pred)
f1 = f1_score(y_validation, y_pred)
roc_auc = roc_auc_score(y_validation, y_prob)


print("\nValidation Results")
print("------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")


# ============================================================
# MLflow TRACKING
# ============================================================

with mlflow.start_run():

    # Log best hyperparameters
    mlflow.log_params(best_params)

    # Log cross-validation result
    mlflow.log_metric(
        "best_cv_roc_auc",
        best_cv_score
    )

    # Log validation metrics
    mlflow.log_metric("validation_accuracy", accuracy)
    mlflow.log_metric("validation_precision", precision)
    mlflow.log_metric("validation_recall", recall)
    mlflow.log_metric("validation_f1", f1)
    mlflow.log_metric("validation_roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="customer_churn_model"
    )

    print("\nMLflow tracking completed.")
    print("--------------------------------")


print("\nPhase 5 training completed successfully.")