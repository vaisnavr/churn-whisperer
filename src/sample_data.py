"""
sample_data.py
Generates a small synthetic, Telco-shaped dataset so the app can run anywhere
(e.g. Streamlit Cloud) WITHOUT shipping the real Kaggle CSV — which we don't
redistribute. The shape and columns match the real Telco Customer Churn dataset,
so the same code path works locally (real data) and in the cloud (sample data).
"""
import numpy as np
import pandas as pd


def make_sample(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"S{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No"], n),
        "OnlineBackup": rng.choice(["Yes", "No"], n),
        "TechSupport": rng.choice(["Yes", "No"], n),
        "Contract": rng.choice(
            ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
        ),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges": np.round(rng.uniform(18, 120, n), 2),
    })
    df["TotalCharges"] = np.round(
        df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.9, 1.1, n), 2
    )
    # churn signal: month-to-month, low tenure, high charges, e-check, no support
    score = (
        (df.Contract == "Month-to-month") * 1.2
        + (df.tenure < 6) * 1.1
        + (df.MonthlyCharges > 80) * 0.6
        + (df.PaymentMethod == "Electronic check") * 0.7
        + (df.TechSupport == "No") * 0.4
        + rng.normal(0, 0.6, n)
    )
    df["Churn"] = np.where(score > 1.4, "Yes", "No")
    return df


if __name__ == "__main__":
    make_sample().to_csv("data/telco_churn.csv", index=False)
    print("Wrote sample data -> data/telco_churn.csv")
