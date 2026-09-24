import pandas as pd
from pathlib import Path


FEATURE_DATA_PATH = Path("data/features")

TARGET_COLUMN = "churn"


def validate_features(filename):

    file_path = FEATURE_DATA_PATH / filename

    print(f"\nValidating {filename}...")

    df = pd.read_csv(file_path)

    # Check customer_id was removed
    if "customer_id" in df.columns:
        raise ValueError(
            "customer_id should not be present in feature data."
        )

    print("✓ customer_id removed")

    # Check target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Target column 'churn' is missing."
        )

    print("✓ Target column 'churn' exists")

    # Check missing values
    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Found {missing_values} missing values."
        )

    print("✓ No missing values")

    # Check all feature columns are numeric
    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric features found: {non_numeric}"
        )

    print("✓ All model features are numeric")

    # Check target values
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Target column contains invalid values."
        )

    print("✓ Target values are valid")

    print(f"✓ Shape: {df.shape}")


def main():

    for filename in [
        "train.csv",
        "validation.csv",
        "test.csv"
    ]:
        validate_features(filename)

    print("\nFeature validation completed successfully.")


if __name__ == "__main__":
    main()