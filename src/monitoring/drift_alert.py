from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
import smtplib
import ssl

import keyring
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT / "data/experimental/features_v3/train.csv"
CURRENT_PATH = ROOT / "data/monitoring/current_data.csv"
LOG_PATH = ROOT / "logs/drift_alerts.csv"

DRIFT_THRESHOLD = 0.5

EMAIL_ACCOUNT = "albinsibi2025@gmail.com"
CREDENTIAL_SERVICE = "customer-churn-monitoring"


def send_drift_notification(
    drifted_count,
    total_features,
    drift_share,
    threshold
):
    password = keyring.get_password(
        CREDENTIAL_SERVICE,
        EMAIL_ACCOUNT
    )

    if not password:
        raise RuntimeError("Gmail credential not found.")

    message = EmailMessage()
    message["Subject"] = "ALERT: Customer Churn Data Drift Detected"
    message["From"] = EMAIL_ACCOUNT
    message["To"] = EMAIL_ACCOUNT

    message.set_content(
        f"""
Customer Churn Monitoring - Data Drift Alert

Significant data drift has been detected.

Drifted features: {drifted_count}
Total features: {total_features}
Drift share: {drift_share:.4f}
Alert threshold: {threshold:.2f}

Detection time:
{datetime.now(timezone.utc).isoformat()}

Please review the monitoring dashboard:
reports/monitoring_dashboard.html

This is an automated monitoring notification.
"""
    )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        context=context,
        timeout=20
    ) as server:
        server.login(EMAIL_ACCOUNT, password)
        server.send_message(message)

    print("Gmail drift notification sent successfully.")


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
    email_status = "NOT_SENT"

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if drift_detected:
        try:
            send_drift_notification(
                drifted_count,
                total_features,
                drift_share,
                DRIFT_THRESHOLD
            )
            email_status = "SENT"

        except Exception as exc:
            email_status = "FAILED"
            print("\nEMAIL NOTIFICATION FAILED")
            print("Error:", type(exc).__name__)

    alert_record = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "status": status,
        "email_status": email_status
    }])

    alert_record.to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),
        index=False
    )

    print(f"\nAlert log saved to: {LOG_PATH}")
    print(f"Email notification status: {email_status}")


if __name__ == "__main__":
    check_data_drift()
