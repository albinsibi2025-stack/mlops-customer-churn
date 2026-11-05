import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data/experimental/processed_v3"
)

FEATURES_DIR = (
    PROJECT_ROOT / "data/experimental/features_v3"
)

FEAST_DATA_DIR = (
    PROJECT_ROOT / "feature_store/feature_repo/data"
)

OUTPUT_FILE = FEAST_DATA_DIR / "customer_features_v3.parquet"


CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]


# --------------------------------------------------
# Load V3 training data
# --------------------------------------------------

print("Loading V3 training data...")

processed_train = pd.read_csv(
    PROCESSED_DIR / "train.csv"
)

engineered_train = pd.read_csv(
    FEATURES_DIR / "train.csv"
)

print("Processed shape:", processed_train.shape)
print("Engineered shape:", engineered_train.shape)


# --------------------------------------------------
# Preserve customer IDs
# --------------------------------------------------

customer_ids = processed_train["customer_id"].copy()

if customer_ids.duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

if len(customer_ids) != len(engineered_train):
    raise ValueError("Processed and engineered row counts differ.")


# --------------------------------------------------
# Reproduce V3 feature engineering
# --------------------------------------------------

df = processed_train.drop(
    columns=["customer_id", "churn"]
).copy()

categories = {
    column: sorted(processed_train[column].dropna().unique())
    for column in CATEGORICAL_FEATURES
}

for column in CATEGORICAL_FEATURES:
    df[column] = pd.Categorical(
        df[column],
        categories=categories[column]
    )

df = pd.get_dummies(
    df,
    columns=CATEGORICAL_FEATURES,
    dtype=int
)

# Ensure feature columns match the V3 training dataset
expected_columns = [
    column for column in engineered_train.columns
    if column != "churn"
]

df = df.reindex(
    columns=expected_columns,
    fill_value=0
)


# --------------------------------------------------
# Verify exact feature alignment
# --------------------------------------------------

expected_values = engineered_train[expected_columns].to_numpy(
    dtype=float
)

actual_values = df[expected_columns].to_numpy(
    dtype=float
)

if not np.allclose(
    actual_values,
    expected_values,
    rtol=1e-5,
    atol=1e-5
):
    raise ValueError(
        "Feature mismatch! Feast dataset was not created."
    )

print("\nFeature alignment verified successfully.")


# --------------------------------------------------
# Build Feast dataset
# --------------------------------------------------

feast_df = df.copy()

feast_df.insert(
    0,
    "customer_id",
    customer_ids.to_numpy()
)

# Use UTC event timestamps
feast_df.insert(
    1,
    "event_timestamp",
    pd.Timestamp.now(tz="UTC")
)

# Exclude target: Feast stores model input features only
FEAST_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

feast_df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print("\nV3 Feast dataset created successfully.")
print("Output:", OUTPUT_FILE)
print("Shape:", feast_df.shape)
print("Unique customers:", feast_df["customer_id"].nunique())
print("Feature count:", len(expected_columns))
print("Target included:", "churn" in feast_df.columns)

print("\nFirst five records:")
print(feast_df.head().to_string(index=False))