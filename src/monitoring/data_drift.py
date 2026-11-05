
from pathlib import Path
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT / "data/experimental/features_v3/train.csv"
CURRENT_PATH = ROOT / "data/monitoring/current_data.csv"

REPORT_PATH = ROOT / "reports/data_drift_report.html"

def run_data_drift():

    print("Loading reference dataset...")
    reference = pd.read_csv(REFERENCE_PATH)

    print("Loading current dataset...")
    current = pd.read_csv(CURRENT_PATH)

    # Remove target column if present
    reference = reference.drop(columns=["churn"], errors="ignore")

    # Ensure identical feature order
    current = current[reference.columns]

    print("\nReference shape:", reference.shape)
    print("Current shape:", current.shape)

    print("\nRunning Evidently data drift analysis...")

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        reference_data=reference,
        current_data=current
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(REPORT_PATH))

    print("\nDATA DRIFT REPORT GENERATED")
    print("===========================")
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run_data_drift()
