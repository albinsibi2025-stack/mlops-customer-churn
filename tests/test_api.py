import builtins
import importlib
import io
from unittest.mock import patch
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "num_products",
    "contract_type_Month-to-month",
    "contract_type_One year",
    "contract_type_Two year",
    "internet_service_DSL",
    "internet_service_Fiber optic",
    "internet_service_No",
    "payment_method_Bank transfer",
    "payment_method_Credit card",
    "payment_method_Electronic check",
    "payment_method_Mailed check",
]


class FakeModel:
    def predict_proba(self, data):
        return [[0.4, 0.6]]


class FakeService:
    name = "customer_churn_service"


class FakeOnlineResult:
    def to_dict(self):
        return {
            feature: [1.0]
            for feature in FEATURE_COLUMNS
        }


class FakeFeatureStore:
    def __init__(self, repo_path):
        pass

    def get_feature_service(self, name):
        return FakeService()

    def get_online_features(self, features, entity_rows):
        return FakeOnlineResult()


# Mock external files and services before importing the API.
original_open = builtins.open


def mocked_open(file, *args, **kwargs):
    if str(file).endswith("candidate_v3_metadata.json"):
        return io.StringIO(
            json.dumps({
                "threshold": 0.34,
                "feature_columns": FEATURE_COLUMNS
            })
        )

    return original_open(file, *args, **kwargs)

with (
    patch("builtins.open", side_effect=mocked_open),
    patch("mlflow.sklearn.load_model", return_value=FakeModel()),
    patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(columns=FEATURE_COLUMNS + ["churn"]),
    ),
    patch("feast.FeatureStore", FakeFeatureStore),
):
    api = importlib.import_module("src.api.app")


api.log_prediction = lambda *args, **kwargs: None
api.log_api_request = lambda *args, **kwargs: None

client = TestClient(api.app)


@pytest.fixture
def valid_features():
    return {
        feature: 1.0
        for feature in FEATURE_COLUMNS
    }


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "Customer Churn Prediction API" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction(valid_features):
    response = client.post(
        "/predict",
        json={"features": valid_features},
    )

    assert response.status_code == 200
    assert response.json()["churn_prediction"] == 1
    assert response.json()["churn_probability"] == 0.6
    assert response.json()["threshold"] == 0.34


def test_prediction_has_required_fields(valid_features):
    response = client.post(
        "/predict",
        json={"features": valid_features},
    )

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "threshold" in result
    assert "result" in result


def test_incorrect_feature_names():
    response = client.post(
        "/predict",
        json={"features": {"incorrect_feature": 1.0}},
    )

    assert response.status_code == 400


def test_feast_prediction_endpoint():
    response = client.post("/predict/CUST_04600")

    assert response.status_code == 200
    assert response.json()["customer_id"] == "CUST_04600"
    assert response.json()["churn_prediction"] == 1