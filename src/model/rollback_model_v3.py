import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")

PRODUCTION_MODEL = "customer-churn-logistic-regression-v3"


def main():
    parser = argparse.ArgumentParser(
        description="Restore a previous champion model version."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Previous production model version to restore.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Confirm the rollback operation.",
    )
    args = parser.parse_args()

    if not args.approve:
        raise SystemExit("Rollback blocked. Add --approve to confirm.")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    target = client.get_model_version(
        name=PRODUCTION_MODEL,
        version=args.version,
    )

    if target.status != "READY":
        raise RuntimeError(
            f"Rollback target is not READY: {target.status}"
        )

    # Verify the model can actually be loaded.
    mlflow.sklearn.load_model(
        f"models:/{PRODUCTION_MODEL}/{target.version}"
    )

    current = client.get_model_version_by_alias(
        PRODUCTION_MODEL,
        "champion",
    )

    client.set_registered_model_alias(
        name=PRODUCTION_MODEL,
        alias="champion",
        version=target.version,
    )

    restored = client.get_model_version_by_alias(
        PRODUCTION_MODEL,
        "champion",
    )

    print("\nROLLBACK SUCCESSFUL")
    print("Previous champion:", current.version)
    print("Restored champion:", restored.version)
    print("API deployment has not been changed.")


if __name__ == "__main__":
    main()
