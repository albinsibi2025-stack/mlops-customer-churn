
import csv
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "api_requests.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "timestamp",
    "method",
    "path",
    "status_code",
    "response_time_ms",
    "error"
]

_log_lock = Lock()


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
    error: str = ""
):
    """Record API request metrics in a CSV file."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": round(response_time_ms, 2),
        "error": error
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
