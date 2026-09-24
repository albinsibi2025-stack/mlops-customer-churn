import pandas as pd
from pathlib import Path


PROCESSED_DATA_PATH = Path("data/processed")
FEATURE_DATA_PATH = Path("data/features")


NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "num_products"
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method"
]


def create_features(df):
    """
    Create machine-learning-ready features.
    """

    df = df.copy()

    # Remove customer ID from model features
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=int
    )

    return df


def process_dataset(filename):

    input_path = PROCESSED_DATA_PATH / filename
    output_path = FEATURE_DATA_PATH / filename

    print(f"\nProcessing {filename}...")

    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")

    feature_df = create_features(df)

    print(f"Feature shape: {feature_df.shape}")

    FEATURE_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved to: {output_path}")


def main():

    process_dataset("train.csv")
    process_dataset("validation.csv")
    process_dataset("test.csv")

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()