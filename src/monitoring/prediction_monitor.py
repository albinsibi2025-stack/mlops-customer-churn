
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = ROOT / "logs" / "predictions.csv"


def analyze_predictions():

    if not LOG_FILE.exists():
        print("Prediction log file not found.")
        return

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        print("No prediction records available.")
        return

    total = len(df)

    churn_count = int((df["churn_prediction"] == 1).sum())
    stay_count = int((df["churn_prediction"] == 0).sum())

    churn_percentage = churn_count / total * 100
    stay_percentage = stay_count / total * 100

    average_probability = df["churn_probability"].mean()

    print("\nPREDICTION MONITORING REPORT")
    print("=" * 40)

    print(f"Total predictions       : {total}")
    print(f"Predicted churn         : {churn_count}")
    print(f"Predicted stay          : {stay_count}")

    print(f"Churn percentage        : {churn_percentage:.2f}%")
    print(f"Stay percentage         : {stay_percentage:.2f}%")

    print(f"Average churn probability: {average_probability:.4f}")

    print("\nPrediction distribution:")
    print(df["churn_prediction"].value_counts().sort_index())


if __name__ == "__main__":
    analyze_predictions()
