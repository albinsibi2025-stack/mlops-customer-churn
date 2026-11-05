
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data/experimental/features_v3/test.csv"
OUTPUT_PATH = ROOT / "data/monitoring/current_data.csv"

# Load unseen test customer features
df = pd.read_csv(SOURCE_PATH)

# Remove target because incoming production data has no actual label
if "churn" in df.columns:
    df = df.drop(columns=["churn"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Monitoring dataset created successfully")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Saved to:", OUTPUT_PATH)
