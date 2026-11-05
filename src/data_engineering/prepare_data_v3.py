import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/experimental/customer_churn_v3.csv")
PROCESSED_DATA_PATH = Path("data/experimental/processed_v3")


def prepare_data():

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    print(f"Cleaned dataset shape: {df.shape}")

    # Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # First split: 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Add target column back
    train_df = X_train.copy()
    train_df["churn"] = y_train

    validation_df = X_val.copy()
    validation_df["churn"] = y_val

    test_df = X_test.copy()
    test_df["churn"] = y_test

    # Create processed directory
    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save datasets
    train_df.to_csv(
        PROCESSED_DATA_PATH / "train.csv",
        index=False
    )

    validation_df.to_csv(
        PROCESSED_DATA_PATH / "validation.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DATA_PATH / "test.csv",
        index=False
    )

    print("\nData preparation completed successfully.")

    print(f"Training data:   {train_df.shape}")
    print(f"Validation data: {validation_df.shape}")
    print(f"Test data:       {test_df.shape}")

    print("\nFiles created:")
    print("✓ data/processed/train.csv")
    print("✓ data/processed/validation.csv")
    print("✓ data/processed/test.csv")


if __name__ == "__main__":
    prepare_data()