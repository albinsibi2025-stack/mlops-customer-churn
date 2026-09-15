import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/customer_churn.csv")


def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("Raw data loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_raw_data()

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())