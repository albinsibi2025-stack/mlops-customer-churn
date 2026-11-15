import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")

CANDIDATE_MODEL = "customer-churn-logistic-regression-v3-candidate"
PRODUCTION_MODEL = "customer-churn-logistic-regression-v3"

EVALUATION_REPORT = (
    ROOT / "reports" / "model_evaluation" / "candidate_v3_evaluation.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Promote an approved customer churn candidate."
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicitly approve candidate promotion.",
    )
    args = parser.parse_args()

    if not args.approve:
        raise SystemExit(
            "Promotion blocked. Review the candidate and rerun with --approve."
        )

    if not EVALUATION_REPORT.exists():
        raise FileNotFoundError(EVALUATION_REPORT)

    with open(EVALUATION_REPORT, "r", encoding="utf-8") as file:
        evaluation = json.load(file)

    if not evaluation.get("quality_gate", {}).get("passed", False):
        raise RuntimeError("Quality gate failed. Promotion cancelled.")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    candidate = client.get_model_version_by_alias(
        CANDIDATE_MODEL,
        "candidate",
    )
    # Verify that the registered candidate can be loaded.

    if candidate.status != "READY":
        raise RuntimeError(
            f"Candidate version is not READY: {candidate.status}"
        )

    mlflow.sklearn.load_model(
        f"models:/{CANDIDATE_MODEL}/{candidate.version}"
    )

    # Preserve the existing champion before changing the alias.
    try:
        previous = client.get_model_version_by_alias(
            PRODUCTION_MODEL,
            "champion",
        )
        previous_version = previous.version
    except Exception:
        previous_version = None

    # Record the candidate's promotion details.
    client.set_model_version_tag(
        CANDIDATE_MODEL,
        candidate.version,
        "approval_status",
        "approved",
    )
    client.set_model_version_tag(
        CANDIDATE_MODEL,
        candidate.version,
        "quality_gate",
        "passed",
    )

    client.set_model_version_tag(
        CANDIDATE_MODEL,
        candidate.version,
        "approved_at_utc",
        datetime.now(timezone.utc).isoformat(),
    )

    # Register the candidate under the production model name.
    # This creates a new production model version without deleting
    # or modifying the existing production version.
    model_uri = f"models:/{CANDIDATE_MODEL}/{candidate.version}"

    promoted = mlflow.register_model(
        model_uri=model_uri,
        name=PRODUCTION_MODEL,
    )
    client = MlflowClient()

    promoted = client.get_model_version(
        name=PRODUCTION_MODEL,
        version=promoted.version,
    )

    if promoted.status != "READY":
        raise RuntimeError(
            f"Production model version is not READY: {promoted.status}"
        )

    client.set_model_version_tag(
        PRODUCTION_MODEL,
        promoted.version,
        "source_candidate_model",
        CANDIDATE_MODEL,
    )
    client.set_model_version_tag(
        PRODUCTION_MODEL,
        promoted.version,
        "source_candidate_version",
        candidate.version,
    )
    client.set_model_version_tag(
        PRODUCTION_MODEL,
        promoted.version,
        "approval_status",
        "approved",
    )

    # Update champion only after the new production version is READY.
    client.set_registered_model_alias(
        name=PRODUCTION_MODEL,
        alias="champion",
        version=promoted.version,
    )

    champion = client.get_model_version_by_alias(
        PRODUCTION_MODEL,
        "champion",
    )

    print("\nPROMOTION SUCCESSFUL")
    print("=" * 55)
    print("Previous champion:", previous_version)
    print("New champion:", champion.version)
    print("Candidate version:", candidate.version)
    print("Validation F1:", evaluation["validation"]["candidate"]["f1"])
    print("Validation precision:", evaluation["validation"]["candidate"]["precision"])
    print("Test F1:", evaluation["test"]["candidate"]["f1"])
    print("\nPrevious champion remains registered for rollback.")
    print("Running API deployment has not been changed.")


if __name__ == "__main__":
    main()
