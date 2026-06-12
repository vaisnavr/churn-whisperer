"""
train_model.py
Trains a churn model on the Telco dataset and saves the model + SHAP explainer.
Run once: python src/train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "data/churn_model.joblib"
ENCODERS_PATH = "data/encoders.joblib"
FEATURES_PATH = "data/feature_cols.joblib"
BENCHMARKS_PATH = "data/benchmarks.joblib"

# Numeric features we want to benchmark a customer against the population for.
NUMERIC_BENCHMARK_COLS = ["MonthlyCharges", "tenure", "TotalCharges"]
# Categorical features where we want the churn rate per category.
CATEGORICAL_BENCHMARK_COLS = ["Contract", "PaymentMethod", "InternetService"]


def compute_benchmarks(df_raw):
    """
    Build a reference table so the app can say *why* a value is high/low
    instead of asserting it. Two kinds:
      - numeric: full distribution (median + percentiles) to locate a customer
      - categorical: actual churn rate per category in THIS dataset
    df_raw is the cleaned df BEFORE encoding (so categories are readable).
    """
    bench = {"numeric": {}, "categorical": {}, "overall_churn_rate": float(df_raw["Churn"].mean())}

    for col in NUMERIC_BENCHMARK_COLS:
        if col in df_raw.columns:
            s = pd.to_numeric(df_raw[col], errors="coerce").dropna()
            bench["numeric"][col] = {
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
                "max": float(s.max()),
                "sorted_values": np.sort(s.values),  # for exact percentile lookup
            }

    for col in CATEGORICAL_BENCHMARK_COLS:
        if col in df_raw.columns:
            rates = df_raw.groupby(col)["Churn"].mean()
            bench["categorical"][col] = {str(k): float(v) for k, v in rates.items()}

    return bench


def load_and_clean(path):
    df = pd.read_csv(path)
    # TotalCharges has blank strings for new customers -> coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop(columns=["customerID"])
    # Target to 0/1
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def encode(df):
    """Label-encode categoricals. Returns encoded df + dict of encoders."""
    encoders = {}
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def main():
    df = load_and_clean(DATA_PATH)
    # benchmarks need the readable (pre-encoding) df
    benchmarks = compute_benchmarks(df)

    df_enc, encoders = encode(df)

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.3f}")
    print(classification_report(y_test, (preds > 0.5).astype(int)))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(benchmarks, BENCHMARKS_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved benchmarks -> {BENCHMARKS_PATH}")
    print(f"  Population median MonthlyCharges: "
          f"${benchmarks['numeric'].get('MonthlyCharges', {}).get('median', 0):.2f}")
    print(f"  Overall churn rate: {benchmarks['overall_churn_rate']:.1%}")


if __name__ == "__main__":
    main()
