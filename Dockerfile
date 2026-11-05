FROM python:3.13-slim

WORKDIR /app

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy trained model and metadata
COPY models/ ./models/

# Copy training feature columns
COPY data/experimental/features_v3/train.csv ./data/experimental/features_v3/train.csv

# Copy complete Feast feature repository
COPY feature_store/feature_repo/ ./feature_store/feature_repo/

# API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]