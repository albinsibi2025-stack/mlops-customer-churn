from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64


# ============================================================
# Customer Entity
# ============================================================

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer entity for churn prediction",
)


# ============================================================
# Customer Feature Source
# ============================================================

customer_features_source = FileSource(
    name="customer_features_source",
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)


# ============================================================
# Customer Feature View
# ============================================================

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=365),

    schema=[
        # Numerical features
        Field(name="age", dtype=Int64),
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="support_calls", dtype=Int64),
        Field(name="num_products", dtype=Int64),

        # Contract type
        Field(name="contract_type_Month-to-month", dtype=Int64),
        Field(name="contract_type_One year", dtype=Int64),
        Field(name="contract_type_Two year", dtype=Int64),

        # Internet service
        Field(name="internet_service_DSL", dtype=Int64),
        Field(name="internet_service_Fiber optic", dtype=Int64),
        Field(name="internet_service_No", dtype=Int64),

        # Payment method
        Field(name="payment_method_Bank transfer", dtype=Int64),
        Field(name="payment_method_Credit card", dtype=Int64),
        Field(name="payment_method_Electronic check", dtype=Int64),
        Field(name="payment_method_Mailed check", dtype=Int64),
    ],

    source=customer_features_source,
    online=True,

    tags={
        "project": "customer_churn",
        "team": "mlops",
    },
)