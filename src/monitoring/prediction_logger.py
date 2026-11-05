
import csv
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "predictions.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "timestamp",
    "customer_id",
    "model_name",
    "model_version",
    "churn_probability",
    "churn_prediction",
    "threshold",
    "result"
]

_log_lock = Lock()


def log_prediction(
    prediction: dict,
    customer_id: str = "unknown"
):
    """Save a successful prediction to a CSV monitoring log."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "model_name": "logistic_regression_v3",
        "model_version": "1",
        "churn_probability": prediction["churn_probability"],
        "churn_prediction": prediction["churn_prediction"],
        "threshold": prediction["threshold"],
        "result": prediction["result"]
    }

    with _log_lock:
        file_exists = LOG_FILE.exists()

        with LOG_FILE.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

    return record
