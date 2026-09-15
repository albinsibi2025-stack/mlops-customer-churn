import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/customer_churn.csv")


REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "support_calls",
    "payment_method",
    "num_products",
    "churn"
]


def validate_data(df):

    print("Starting data validation...\n")

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("✓ Required columns are present")

    # Check duplicate customer IDs
    duplicate_ids = df["customer_id"].duplicated().sum()

    print(f"✓ Duplicate customer IDs: {duplicate_ids}")

    # Check missing values
    missing_values = df.isnull().sum().sum()

    print(f"✓ Total missing values: {missing_values}")

    # Check target values
    invalid_churn = ~df["churn"].isin([0, 1])

    if invalid_churn.any():
        raise ValueError(
            "Invalid values found in churn column"
        )

    print("✓ Churn values are valid (0/1)")

    # Check numeric columns
    numeric_columns = [
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_calls",
        "num_products"
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} must be numeric"
            )

    print("✓ Numeric columns validated")

    print("\nData validation completed successfully.")


def main():

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}\n")

    validate_data(df)


if __name__ == "__main__":
    main()