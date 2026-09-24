import os
import pandas as pd


# Project paths
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
RAW_FILE = os.path.join(
    PROJECT_ROOT, "data", "raw", "customer_churn.csv"
)

FEAST_DATA_DIR = os.path.join(
    PROJECT_ROOT, "feature_store", "feature_repo", "data"
)

OUTPUT_FILE = os.path.join(
    FEAST_DATA_DIR, "customer_features.parquet"
)


# Create Feast data directory if needed
os.makedirs(FEAST_DATA_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load the engineered training features
# ------------------------------------------------------------

features = pd.read_csv(
    os.path.join(FEATURES_DIR, "train.csv")
)

raw = pd.read_csv(RAW_FILE)


# ------------------------------------------------------------
# Add customer_id
# ------------------------------------------------------------

features.insert(
    0,
    "customer_id",
    raw.loc[features.index, "customer_id"].values
)


# ------------------------------------------------------------
# Add event timestamp
# ------------------------------------------------------------

features["event_timestamp"] = pd.Timestamp.now()


# ------------------------------------------------------------
# Remove target column
# ------------------------------------------------------------

if "churn" in features.columns:
    features = features.drop(columns=["churn"])


# ------------------------------------------------------------
# Reorder columns
# ------------------------------------------------------------

features = features[
    ["customer_id", "event_timestamp"]
    + [
        column
        for column in features.columns
        if column not in ["customer_id", "event_timestamp"]
    ]
]


# ------------------------------------------------------------
# Save as Parquet
# ------------------------------------------------------------

features.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("Feast source data created successfully.")
print(f"Shape: {features.shape}")
print(f"Saved to: {OUTPUT_FILE}")
print("\nColumns:")
print(features.columns.tolist())