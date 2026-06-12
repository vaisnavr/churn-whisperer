"""
llm_layer.py
Takes a customer's top churn drivers (from SHAP) and uses Gemini to:
  1. Write a plain-English risk explanation
  2. Draft a tailored retention email
This is the 'product' layer that turns a score into an action.
"""
import os
import json
from google import genai


def _get_api_key():
    """Read the Gemini key from an env var (local) or Streamlit secrets (cloud)."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


client = genai.Client(api_key=_get_api_key())

MODEL = "gemini-2.5-flash"


def build_driver_text(drivers, benchmarks=None):
    """drivers: list of (feature_name, value, shap_impact) tuples.
    If benchmarks provided, each value is annotated with population context
    so the model grounds words like 'high' in real data instead of guessing."""
    from src.benchmarks import benchmark_driver
    lines = []
    for name, value, impact in drivers:
        direction = "increases" if impact > 0 else "decreases"
        if benchmarks:
            value_str = benchmark_driver(name, value, benchmarks)
        else:
            value_str = str(value)
        lines.append(f"- {name} = {value_str} ({direction} churn risk)")
    return "\n".join(lines)


def generate_insight(customer_profile, drivers, risk_score, benchmarks=None):
    """Returns dict with 'explanation' and 'email' keys."""
    driver_text = build_driver_text(drivers, benchmarks)

    prompt = f"""You are a customer success analyst. A churn model flagged this \
customer with a risk score of {risk_score:.0%}.

Customer profile:
{json.dumps(customer_profile, indent=2)}

Top factors driving the model's prediction (from SHAP), each benchmarked against \
the full customer population:
{driver_text}

IMPORTANT: When you call a value "high", "low", or "short", you MUST justify it \
using the percentile or segment churn rate provided above — do not assert it from \
general intuition. For example, say "in the top 12% of customers" rather than just \
"high".

Produce a JSON object with exactly two keys and nothing else:
1. "explanation": 2-3 sentences in plain English explaining WHY this customer is \
at risk, written for a non-technical CS manager. Ground every claim in the \
benchmark numbers above.
2. "email": a short, warm retention email (under 120 words) addressed to the \
customer that proactively addresses their likely concerns. Do not mention churn \
scores, percentiles, or that they were flagged. Sign as "The [Company] Team".

Return ONLY the raw JSON, no markdown fences, no preamble."""

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    text = resp.text.strip()
    # strip accidental fences
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


if __name__ == "__main__":
    # quick smoke test
    demo_profile = {"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 89.5}
    demo_drivers = [
        ("Contract", "Month-to-month", 0.8),
        ("tenure", 2, 0.6),
        ("MonthlyCharges", 89.5, 0.3),
    ]
    out = generate_insight(demo_profile, demo_drivers, 0.82)
    print(json.dumps(out, indent=2))
