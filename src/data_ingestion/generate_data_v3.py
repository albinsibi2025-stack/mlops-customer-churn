import numpy as np
import pandas as pd
from pathlib import Path


def generate_customer_data(num_records=5000, random_seed=2026):
    np.random.seed(random_seed)

    customer_id = [
        f"CUST_{i:05d}"
        for i in range(1, num_records + 1)
    ]

    age = np.random.randint(18, 75, num_records)

    tenure_months = np.random.randint(1, 73, num_records)

    monthly_charges = np.round(
        np.random.uniform(20, 150, num_records),
        2
    )

    total_charges = np.round(
        monthly_charges
        * tenure_months
        * np.random.uniform(0.85, 1.15, num_records),
        2
    )

    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        num_records,
        p=[0.55, 0.25, 0.20]
    )

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        num_records,
        p=[0.35, 0.45, 0.20]
    )

    support_calls = np.random.randint(0, 11, num_records)

    payment_method = np.random.choice(
        [
            "Electronic check",
            "Credit card",
            "Bank transfer",
            "Mailed check"
        ],
        num_records,
        p=[0.35, 0.25, 0.20, 0.20]
    )

    num_products = np.random.randint(1, 6, num_records)

    # ---------------------------------------------------------
    # Churn risk score
    # ---------------------------------------------------------
    # The score represents realistic relationships between
    # customer behavior and the probability of churn.
    #
    # Positive contribution  -> higher churn risk
    # Negative contribution  -> lower churn risk
    # ---------------------------------------------------------

    churn_score = (
        -1.2
        + 0.10 * support_calls
        + 0.012 * (monthly_charges - 50)
        - 0.035 * tenure_months
        + 0.90 * (contract_type == "Month-to-month")
        - 0.35 * (contract_type == "One year")
        - 0.70 * (contract_type == "Two year")
        + 0.30 * (internet_service == "Fiber optic")
        - 0.20 * (internet_service == "No")
        + 0.35 * (payment_method == "Electronic check")
        - 0.08 * (num_products - 1)
    )

    # Convert risk score to probability.
    churn_probability = 1 / (1 + np.exp(-churn_score))

    # Add natural randomness so the problem is not perfectly predictable.
    churn = (
        np.random.random(num_records)
        < churn_probability
    ).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "support_calls": support_calls,
        "payment_method": payment_method,
        "num_products": num_products,
        "churn": churn
    })

    return df


def main():
    output_path = Path("data/experimental/customer_churn_v3.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_customer_data()

    df.to_csv(output_path, index=False)

    print("Customer churn dataset created successfully.")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()