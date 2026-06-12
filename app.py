"""
app.py
Churn Whisperer — the product surface.
Pick a customer -> see risk score -> see WHY (SHAP) -> see AI-drafted save email.

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from src.llm_layer import generate_insight

st.set_page_config(page_title="Churn Whisperer", page_icon="🎯", layout="wide")

# ---------- load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("data/churn_model.joblib")
    encoders = joblib.load("data/encoders.joblib")
    feature_cols = joblib.load("data/feature_cols.joblib")
    benchmarks = joblib.load("data/benchmarks.joblib")
    explainer = shap.TreeExplainer(model)
    return model, encoders, feature_cols, explainer, benchmarks


@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


model, encoders, feature_cols, explainer, benchmarks = load_artifacts()
raw_df = load_data()


def encode_row(row):
    enc = {}
    for col in feature_cols:
        val = row[col]
        if col in encoders:
            le = encoders[col]
            # handle unseen labels gracefully
            if val in le.classes_:
                enc[col] = le.transform([val])[0]
            else:
                enc[col] = 0
        else:
            enc[col] = val
    return pd.DataFrame([enc])[feature_cols]


# ---------- UI ----------
st.title("🎯 Churn Whisperer")
st.caption("Predict churn, explain *why*, and draft the save — in one click.")

# pick a customer (default to a high-risk-looking one)
cust_ids = raw_df["customerID"].tolist()
choice = st.selectbox("Select a customer", cust_ids, index=2)
row = raw_df[raw_df["customerID"] == choice].iloc[0]

X_enc = encode_row(row)
risk = model.predict_proba(X_enc)[:, 1][0]

# SHAP drivers
shap_vals = explainer.shap_values(X_enc)
sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
contrib = sorted(
    zip(feature_cols, X_enc.iloc[0].values, sv),
    key=lambda x: abs(x[2]),
    reverse=True,
)
top_drivers = [(name, row[name], float(impact)) for name, _, impact in contrib[:4]]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Churn Risk", f"{risk:.0%}")
    band = "🔴 High" if risk > 0.6 else "🟡 Medium" if risk > 0.3 else "🟢 Low"
    st.write(f"**Risk band:** {band}")
    st.write("**Top drivers** _(vs. all customers)_")
    from src.benchmarks import benchmark_driver
    for name, val, impact in top_drivers:
        arrow = "↑" if impact > 0 else "↓"
        context = benchmark_driver(name, row[name], benchmarks)
        # context already includes the value, so show name + context
        st.write(f"{arrow} **{name}**: {context}")

with col2:
    profile = {name: row[name] for name, _, _ in top_drivers}
    profile["tenure"] = int(row.get("tenure", 0))
    profile["MonthlyCharges"] = float(row.get("MonthlyCharges", 0))

    if st.button("🤖 Generate AI insight + save email", type="primary"):
        with st.spinner("Asking Gemini to read the signals..."):
            try:
                out = generate_insight(profile, top_drivers, risk, benchmarks)
                st.subheader("Why they're at risk")
                st.info(out["explanation"])
                st.subheader("Suggested retention email")
                st.text_area("Draft", out["email"], height=220)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.caption("Check your GEMINI_API_KEY env var.")

st.divider()
st.caption(
    "Built by Vaisnav Roy · churn model (XGBoost) + SHAP explainability + "
    "Gemini for narrative & action. If shipped: cuts manual save-email drafting "
    "and surfaces at-risk revenue before it walks."
)
