# 🎯 Churn Whisperer

**An AI agent that doesn't just predict churn — it explains *why* a specific customer is at risk and drafts the email to save them.**

Most churn projects stop at a number. A score of `0.82` tells a customer success manager *nothing* about what to do next. Churn Whisperer closes that gap: it pairs a churn model with per-customer explainability and an LLM action layer, turning a prediction into a decision and a draft.

🔗 **[Live demo](#)** · 🎥 **[30-sec walkthrough](#)**

---

## The problem (product framing)

> CS and RevOps teams sit on churn scores they can't act on. By the time a human reads the dashboard, segments the at-risk users, figures out *why* each one is leaving, and writes a personal save email — the customer is already gone.

**Users:** Customer Success Managers, RevOps analysts.
**Job to be done:** "When an account looks shaky, help me understand why and respond — fast — without writing every email from scratch."

## What it does

1. **Scores** churn risk per customer (XGBoost, AUC ~0.87).
2. **Explains** the top drivers for *that individual* customer using SHAP — not global feature importance, but "here's why *this* person specifically is at risk." Each driver is **benchmarked against the full customer population** (e.g. "monthly charges in the 88th percentile" or "this segment churns at 1.5x the average"), so claims like "high" are grounded in data, not guessed.
3. **Acts** — Gemini turns those drivers into a plain-English risk summary for the CSM *and* a tailored, ready-to-send retention email.

## The metric it would move

If shipped, Churn Whisperer is designed to cut the manual time a CSM spends triaging and drafting save outreach (from ~10 min/account to a one-click review), letting teams reach more at-risk accounts before they lapse — directly protecting renewal revenue in the high-risk segment.

## Architecture

```
Customer record
      │
      ▼
 [XGBoost model] ──► risk score
      │
      ▼
   [SHAP] ──► per-customer top drivers
      │
      ▼
  [Gemini 2.5 Pro] ──► risk narrative + retention email
      │
      ▼
 [Streamlit UI] ──► CSM reviews & sends
```

## Stack

| Layer | Tool |
|-------|------|
| Model | XGBoost + scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| AI action layer | Google Gemini API (gemini-2.5-pro) |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |

## Run it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"

# 1. get the data: download the Telco Customer Churn CSV from Kaggle,
#    save it as data/telco_churn.csv
# 2. train
python src/train_model.py
# 3. launch
streamlit run app.py
```

## What I'd build next (roadmap)

- **Batch mode:** rank the full book of business, not one customer at a time.
- **Offer optimization:** A/B test which save offers actually retain, by segment (ties to real uplift testing).
- **Feedback loop:** log which emails landed and feed outcomes back to refine targeting.

---

*Built by Vaisnav Roy — MS Business Analytics, USC Marshall. Background in fintech & SaaS RevOps. [LinkedIn](https://linkedin.com/in/vaisnavroy) · [Portfolio](https://vaisnavroy.com)*
# churn-whisperer
