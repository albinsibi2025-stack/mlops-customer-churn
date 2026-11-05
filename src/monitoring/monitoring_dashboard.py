
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import html

ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG = ROOT / "logs" / "predictions.csv"
API_LOG = ROOT / "logs" / "api_requests.csv"
DRIFT_LOG = ROOT / "logs" / "drift_alerts.csv"

REPORT_PATH = ROOT / "reports" / "monitoring_dashboard.html"


def load_log(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def create_dashboard():

    print("Loading monitoring logs...")

    predictions = load_log(PREDICTION_LOG)
    api_requests = load_log(API_LOG)
    drift = load_log(DRIFT_LOG)

    # ----------------------------------
    # Prediction monitoring
    # ----------------------------------

    total_predictions = len(predictions)

    churn_count = 0
    stay_count = 0
    avg_probability = 0.0

    if not predictions.empty:
        churn_count = int(
            (predictions["churn_prediction"] == 1).sum()
        )

        stay_count = int(
            (predictions["churn_prediction"] == 0).sum()
        )

        avg_probability = float(
            predictions["churn_probability"].mean()
        )

    # ----------------------------------
    # API monitoring
    # ----------------------------------

    total_requests = len(api_requests)
    successful_requests = 0
    failed_requests = 0
    avg_response_time = 0.0

    if not api_requests.empty:
        status_codes = pd.to_numeric(
            api_requests["status_code"],
            errors="coerce"
        )

        successful_requests = int(
            status_codes.between(200, 299).sum()
        )

        failed_requests = int(
            (status_codes.isna() | ~status_codes.between(200, 299)).sum()
        )

        response_times = pd.to_numeric(
            api_requests["response_time_ms"],
            errors="coerce"
        )

        avg_response_time = float(
            response_times.mean()
        ) if response_times.notna().any() else 0.0

    # ----------------------------------
    # Data drift monitoring
    # ----------------------------------

    drift_status = "NO DATA"
    drift_share = 0.0
    drifted_features = 0
    total_features = 0

    if not drift.empty:
        latest_drift = drift.iloc[-1]

        drift_status = str(latest_drift["status"])
        drift_share = float(latest_drift["drift_share"])
        drifted_features = int(latest_drift["drifted_features"])
        total_features = int(latest_drift["total_features"])

    # ----------------------------------
    # Summary tables
    # ----------------------------------

    prediction_summary = pd.DataFrame({
        "Metric": [
            "Total predictions",
            "Predicted churn",
            "Predicted stay",
            "Average churn probability"
        ],
        "Value": [
            total_predictions,
            churn_count,
            stay_count,
            f"{avg_probability:.4f}"
        ]
    })

    api_summary = pd.DataFrame({
        "Metric": [
            "Total API requests",
            "Successful requests",
            "Failed requests",
            "Average response time (ms)"
        ],
        "Value": [
            total_requests,
            successful_requests,
            failed_requests,
            f"{avg_response_time:.2f}"
        ]
    })

    drift_summary = pd.DataFrame({
        "Metric": [
            "Monitoring status",
            "Total features",
            "Drifted features",
            "Drift share"
        ],
        "Value": [
            drift_status,
            total_features,
            drifted_features,
            f"{drift_share:.4f}"
        ]
    })

    # ----------------------------------
    # Generate HTML report
    # ----------------------------------

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    drift_color = (
        "#dc2626" if drift_status == "ALERT" else "#15803d"
    )

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Customer Churn Monitoring Dashboard</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 35px;
                background: #f4f6f9;
                color: #222;
            }}

            h1 {{
                color: #1e293b;
            }}

            h2 {{
                color: #334155;
                margin-top: 30px;
            }}

            .card {{
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            }}

            .status {{
                color: {drift_color};
                font-weight: bold;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
            }}

            th, td {{
                padding: 11px;
                border: 1px solid #ddd;
                text-align: left;
            }}

            th {{
                background: #e2e8f0;
            }}

            .note {{
                background: #fff7ed;
                padding: 15px;
                border-left: 4px solid #ea580c;
                margin-top: 25px;
            }}

            footer {{
                margin-top: 30px;
                color: #64748b;
                font-size: 13px;
            }}
        </style>
    </head>

    <body>

        <h1>Customer Churn Monitoring Dashboard</h1>
        <p>Generated: {html.escape(generated_at)}</p>

        <div class="card">
            <h2>Prediction Monitoring</h2>
            {prediction_summary.to_html(index=False, escape=True)}
        </div>

        <div class="card">
            <h2>API Performance Monitoring</h2>
            {api_summary.to_html(index=False, escape=True)}
        </div>

        <div class="card">
            <h2>Data Drift Monitoring</h2>
            <p class="status">
                Status: {html.escape(drift_status)}
            </p>
            {drift_summary.to_html(index=False, escape=True)}
        </div>

        <div class="note">
            <strong>Monitoring limitation:</strong>
            Current monitoring logs contain demonstration requests.
            The drift comparison uses the held-out V3 test dataset,
            not live production customer data. These results should
            not be interpreted as production performance evidence.
        </div>

        <footer>
            Customer Churn Prediction | MLOps Phase 9 - Monitoring
        </footer>

    </body>
    </html>
    """

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_html, encoding="utf-8")

    print("\nMONITORING DASHBOARD GENERATED")
    print("=" * 40)
    print("Total predictions:", total_predictions)
    print("Total API requests:", total_requests)
    print("Successful API requests:", successful_requests)
    print("Failed API requests:", failed_requests)
    print("Average response time:", f"{avg_response_time:.2f} ms")
    print("Drift status:", drift_status)
    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    create_dashboard()
