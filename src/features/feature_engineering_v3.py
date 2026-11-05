import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/experimental/processed_v3")
FEATURE_DATA_PATH = Path("data/experimental/features_v3")

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method"
]

FILES = ["train.csv", "validation.csv", "test.csv"]


def main():
    # Load all splits
    datasets = {
        filename: pd.read_csv(PROCESSED_DATA_PATH / filename)
        for filename in FILES
    }

    # Identify categories from training data only
    train_df = datasets["train.csv"]

    categories = {
        column: sorted(train_df[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }

    FEATURE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for filename, df in datasets.items():
        print(f"\nProcessing {filename}...")

        df = df.copy()

        # Keep the target separate from the input features
        target = df["churn"].reset_index(drop=True)

        df = df.drop(columns=["customer_id", "churn"])

        # Apply consistent one-hot encoding based on training categories
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

        # Ensure every split has exactly the training feature columns
        if filename == "train.csv":
            feature_columns = df.columns.tolist()
        else:
            df = df.reindex(columns=feature_columns, fill_value=0)

        df["churn"] = target

        output_path = FEATURE_DATA_PATH / filename
        df.to_csv(output_path, index=False)

        print(f"Feature shape: {df.shape}")
        print(f"Saved to: {output_path}")

    print("\nExperimental feature engineering completed.")


if __name__ == "__main__":
    main()