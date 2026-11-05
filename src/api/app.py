
from pathlib import Path
import json

import joblib
import pandas as pd
from feast import FeatureStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.monitoring.prediction_logger import log_prediction
import time
from src.monitoring.api_logger import log_api_request
# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/experimental/logistic_regression_v3.joblib"
METADATA_PATH = ROOT / "models/experimental/logistic_regression_v3_metadata.json"
TRAIN_PATH = ROOT / "data/experimental/features_v3/train.csv"

FEAST_REPO_PATH = ROOT / "feature_store/feature_repo"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

threshold = float(metadata["threshold"])

feature_columns = [
    col for col in pd.read_csv(TRAIN_PATH, nrows=1).columns
    if col != "churn"
]

# --------------------------------------------------
# INITIALIZE FEAST
# --------------------------------------------------

feature_store = FeatureStore(
    repo_path=str(FEAST_REPO_PATH)
)

customer_service = feature_store.get_feature_service(
    "customer_churn_service"
)

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction using Logistic Regression V3 and Feast Feature Store",
    version="2.0.0"
)

@app.middleware("http")
async def monitor_api_requests(request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    error = ""

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception as e:
        error = str(e)
        raise

    finally:
        response_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        try:
            log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                response_time_ms=response_time_ms,
                error=error
            )
        except Exception as logging_error:
            print(f"API monitoring log failed: {logging_error}")


class CustomerData(BaseModel):
    features: dict[str, float]


# --------------------------------------------------
# COMMON PREDICTION FUNCTION
# --------------------------------------------------

def generate_prediction(supplied):

    if set(supplied.keys()) != set(feature_columns):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Incorrect feature names",
                "expected_features": feature_columns
            }
        )

    try:
        input_data = pd.DataFrame(
            [[float(supplied[col]) for col in feature_columns]],
            columns=feature_columns
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction = int(probability >= threshold)

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "threshold": threshold,
            "result": "Likely to churn" if prediction == 1
                      else "Likely to stay"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "model": MODEL_PATH.name,
        "feature_store": "Feast"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH.name,
        "threshold": threshold,
        "feature_service": customer_service.name
    }


# --------------------------------------------------
# PREDICTION USING DIRECT FEATURES
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):
    prediction = generate_prediction(customer.features)
    log_prediction(prediction)
    return prediction

# --------------------------------------------------
# PREDICTION USING FEAST ONLINE FEATURES
# --------------------------------------------------

@app.post("/predict/{customer_id}")
def predict_customer(customer_id: str):
    try:
        result = feature_store.get_online_features(
            features=customer_service,
            entity_rows=[{"customer_id": customer_id}],
        ).to_dict()

        supplied = {
            feature: result[feature][0]
            for feature in feature_columns
        }

        if any(value is None for value in supplied.values()):
            raise HTTPException(
                status_code=404,
                detail=f"Online features not found for customer: {customer_id}"
            )

        prediction = generate_prediction(supplied)

        # Log successful prediction
        log_prediction(prediction, customer_id)

        return {"customer_id": customer_id, **prediction}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature retrieval or prediction failed: {str(e)}"
        )