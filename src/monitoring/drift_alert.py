
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT / "data/experimental/features_v3/train.csv"
CURRENT_PATH = ROOT / "data/monitoring/current_data.csv"
LOG_PATH = ROOT / "logs/drift_alerts.csv"

DRIFT_THRESHOLD = 0.5


def check_data_drift():
    print("Loading reference and current datasets...")

    if not REFERENCE_PATH.exists() or not CURRENT_PATH.exists():
        print("ERROR: Monitoring dataset not found.")
        return

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    reference = reference.drop(columns=["churn"], errors="ignore")
    current = current[reference.columns]

    print("Running Evidently drift analysis...")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=current
    )

    results = snapshot.dict()
    metrics = results.get("metrics", [])

    drift_metric = next(
        (
            metric for metric in metrics
            if metric.get("metric_name", "").startswith(
                "DriftedColumnsCount"
            )
        ),
        None
    )

    if drift_metric is None:
        print("ERROR: Drift metric not found.")
        return

    value = drift_metric.get("value", {})
    drifted_count = int(value.get("count", 0))
    drift_share = float(value.get("share", 0))
    total_features = len(reference.columns)

    drift_detected = drift_share >= DRIFT_THRESHOLD

    print("\nDATA DRIFT MONITORING")
    print("=" * 40)
    print(f"Total features  : {total_features}")
    print(f"Drifted features: {drifted_count}")
    print(f"Drift share     : {drift_share:.4f}")
    print(f"Threshold       : {DRIFT_THRESHOLD:.2f}")

    if drift_detected:
        status = "ALERT"
        print("\nALERT: Significant data drift detected!")
    else:
        status = "OK"
        print("\nOK: No significant data drift detected.")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    alert_record = pd.DataFrame([{
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "status": status
    }])

    alert_record.to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),
        index=False
    )

    print(f"\nAlert log saved to: {LOG_PATH}")


if __name__ == "__main__":
    check_data_drift()
