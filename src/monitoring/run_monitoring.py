import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_script(script_name):
    script_path = PROJECT_ROOT / "src" / "monitoring" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Monitoring script not found: {script_path}"
        )

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True
    )


def main():
    print("\nCUSTOMER CHURN MONITORING PIPELINE")
    print("Starting monitoring execution...")

    stages = [
        ("prepare_monitoring_data.py", "Prepare monitoring data"),
        ("data_drift.py", "Generate data drift report"),
        ("drift_alert.py", "Check data drift and log alerts"),
        ("prediction_monitor.py", "Analyze prediction distribution"),
        ("monitoring_dashboard.py", "Generate monitoring dashboard"),
    ]

    for script_name, description in stages:
        print(f"\n[STAGE] {description}")
        run_script(script_name)

    print("\n" + "=" * 60)
    print("MONITORING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("\nMONITORING PIPELINE FAILED!")
        print("Failed command:", error.cmd)
        sys.exit(error.returncode)
    except Exception as error:
        print("\nMONITORING PIPELINE FAILED!")
        print("Error:", error)
        sys.exit(1)
