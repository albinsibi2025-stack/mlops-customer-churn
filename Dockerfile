FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Copy the selected V3 model and metadata
COPY models/experimental/mlflow_candidate/ ./models/experimental/mlflow_candidate/
COPY models/experimental/candidate_v3_metadata.json ./models/experimental/


# Copy Feast configuration and definitions
COPY feature_store/feature_repo/feature_store.yaml ./feature_store/feature_repo/
COPY feature_store/feature_repo/feature_definitions.py ./feature_store/feature_repo/
COPY feature_store/feature_repo/__init__.py ./feature_store/feature_repo/

# Copy Feast runtime data
COPY feature_store/feature_repo/data/registry.db ./feature_store/feature_repo/data/
COPY feature_store/feature_repo/data/online_store.db ./feature_store/feature_repo/data/
COPY feature_store/feature_repo/data/customer_features_v3.parquet ./feature_store/feature_repo/data/

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
