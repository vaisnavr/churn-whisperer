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
import os

from src.llm_layer import generate_insight

st.set_page_config(page_title="Churn Whisperer", page_icon="◎", layout="wide")

# ---------- visual identity ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #15233a;
  --slate: #5b6b82;
  --teal: #0d9488;
  --teal-soft: #e6f4f2;
  --danger: #dc5a5a;
  --safe: #2f9e6f;
  --line: #e6e9ef;
  --bg-card: #ffffff;
}

html, body, [class*="css"], .stMarkdown, .stTextArea textarea {
  font-family: 'Inter', -apple-system, sans-serif;
  color: var(--ink);
}

/* hero */
.cw-hero { padding: 0.5rem 0 0.25rem 0; }
.cw-title {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 2.6rem;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
  line-height: 1.05;
}
.cw-title .mark { color: var(--teal); }
.cw-sub { color: var(--slate); font-size: 1rem; margin-top: 0.3rem; }
.cw-rule { height: 3px; width: 56px; background: var(--teal); border-radius: 3px; margin: 0.9rem 0 0.5rem; }

/* score card */
.cw-score-card {
  background: var(--bg-card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.4rem 1.5rem;
  box-shadow: 0 1px 2px rgba(21,35,58,0.04);
}
.cw-score-label { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--slate); font-weight: 600; }
.cw-score-num { font-family: 'Fraunces', serif; font-size: 3.4rem; font-weight: 600; line-height: 1; margin: 0.2rem 0; }
.cw-band { display:inline-block; font-size:0.8rem; font-weight:600; padding: 0.25rem 0.7rem; border-radius: 999px; }

/* driver cards */
.cw-drivers-h { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--slate); font-weight: 600; margin: 1.4rem 0 0.6rem; }
.cw-driver {
  border: 1px solid var(--line);
  border-left: 3px solid var(--line);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  margin-bottom: 0.55rem;
  background: var(--bg-card);
}
.cw-driver.up { border-left-color: var(--danger); }
.cw-driver.down { border-left-color: var(--safe); }
.cw-driver-name { font-weight: 600; font-size: 0.92rem; }
.cw-driver-ctx { color: var(--slate); font-size: 0.84rem; margin-top: 0.1rem; }

/* section headers in right column */
.cw-section { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.35rem; color: var(--ink); margin: 0.5rem 0 0.6rem; }

/* the explanation panel */
.cw-explain {
  background: var(--teal-soft);
  border: 1px solid #cfe9e5;
  border-radius: 12px;
  padding: 1rem 1.15rem;
  color: var(--ink);
  font-size: 0.98rem;
  line-height: 1.55;
}

/* primary button */
.stButton > button {
  background: var(--teal) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.1rem !important;
}
.stButton > button:hover { background: #0b7d73 !important; }

footer, #MainMenu { visibility: hidden; }
.cw-foot { color: var(--slate); font-size: 0.82rem; border-top: 1px solid var(--line); padding-top: 0.9rem; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------- bootstrap: ensure data + model exist (for cloud deploy) ----------
@st.cache_resource
def ensure_artifacts():
    """If running somewhere without the data/model (e.g. Streamlit Cloud),
    generate a synthetic sample dataset and train the model on first launch."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/telco_churn.csv"):
        from src.sample_data import make_sample
        make_sample().to_csv("data/telco_churn.csv", index=False)
    if not os.path.exists("data/churn_model.joblib"):
        import src.train_model as tm
        tm.main()
    return True


ensure_artifacts()


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
st.markdown("""
<div class="cw-hero">
  <h1 class="cw-title">Churn <span class="mark">Whisperer</span></h1>
  <div class="cw-sub">Predict churn, explain why, and draft the save - in one click.</div>
  <div class="cw-rule"></div>
</div>
""", unsafe_allow_html=True)

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

from src.benchmarks import benchmark_driver

col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    if risk > 0.6:
        band_txt, band_bg, band_fg = "High risk", "#fbe6e6", "#c0392b"
        num_color = "var(--danger)"
    elif risk > 0.3:
        band_txt, band_bg, band_fg = "Medium risk", "#fdf3e0", "#b8791b"
        num_color = "#d98324"
    else:
        band_txt, band_bg, band_fg = "Low risk", "#e6f5ee", "#1e7a52"
        num_color = "var(--safe)"

    st.markdown(f"""
    <div class="cw-score-card">
      <div class="cw-score-label">Churn risk</div>
      <div class="cw-score-num" style="color:{num_color}">{risk:.0%}</div>
      <span class="cw-band" style="background:{band_bg};color:{band_fg}">{band_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cw-drivers-h">Top drivers · vs. all customers</div>', unsafe_allow_html=True)
    for name, val, impact in top_drivers:
        direction = "up" if impact > 0 else "down"
        context = benchmark_driver(name, row[name], benchmarks)
        st.markdown(f"""
        <div class="cw-driver {direction}">
          <div class="cw-driver-name">{name}</div>
          <div class="cw-driver-ctx">{context}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    profile = {name: row[name] for name, _, _ in top_drivers}
    profile["tenure"] = int(row.get("tenure", 0))
    profile["MonthlyCharges"] = float(row.get("MonthlyCharges", 0))

    if st.button("Generate insight & draft email", type="primary"):
        with st.spinner("Reading the signals…"):
            try:
                out = generate_insight(profile, top_drivers, risk, benchmarks)
                st.markdown('<div class="cw-section">Why they\'re at risk</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cw-explain">{out["explanation"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="cw-section">Suggested retention email</div>', unsafe_allow_html=True)
                st.text_area("Draft", out["email"], height=240, label_visibility="collapsed")
            except Exception as e:
                st.error(f"Couldn't reach the model: {e}")
                st.caption("If this persists, the demo's API quota may be cooling down - try again shortly.")
    else:
        st.markdown(
            '<div style="color:var(--slate);font-size:0.95rem;padding-top:0.4rem;">'
            'Select a customer and generate a grounded explanation plus a ready-to-send '
            'retention email. Every reason is benchmarked against the full customer base.'
            '</div>', unsafe_allow_html=True)

st.markdown("""
<div class="cw-foot">
Built by Vaisnav Roy · XGBoost scoring + SHAP per-customer explainability + Gemini for narrative & drafting.
Designed to cut manual save-email work and surface at-risk revenue before it walks.
</div>
""", unsafe_allow_html=True)
