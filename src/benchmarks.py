"""
benchmarks.py
Turns a raw feature value into a grounded, defensible statement by comparing it
to the population distribution computed at training time.

e.g. MonthlyCharges = 105.5  ->  "$105.50 (88th percentile; population median $70.35)"
"""
import numpy as np


def percentile_of(value, sorted_values):
    """What % of the population is at or below this value."""
    if sorted_values is None or len(sorted_values) == 0:
        return None
    rank = np.searchsorted(sorted_values, value, side="right")
    return round(100.0 * rank / len(sorted_values))


def describe_numeric(col, value, bench):
    """Return a human + LLM-friendly benchmark string for a numeric feature."""
    info = bench.get("numeric", {}).get(col)
    if not info:
        return f"{value}"
    pct = percentile_of(float(value), info.get("sorted_values"))
    median = info["median"]
    if pct is None:
        return f"{value}"
    # word label
    if pct >= 75:
        label = "high"
    elif pct <= 25:
        label = "low"
    else:
        label = "typical"
    return (f"{value} ({pct}th percentile — {label}; "
            f"population median {median:.2f})")


def describe_categorical(col, value, bench):
    """Return churn-rate context for a categorical feature value."""
    cat = bench.get("categorical", {}).get(col)
    overall = bench.get("overall_churn_rate")
    if not cat or str(value) not in cat:
        return f"{value}"
    rate = cat[str(value)]
    if overall:
        rel = rate / overall if overall else 1
        comp = "above" if rate > overall else "below"
        return (f"{value} (this segment churns at {rate:.0%}, "
                f"{comp} the {overall:.0%} average — {rel:.1f}x)")
    return f"{value} (segment churn rate {rate:.0%})"


def benchmark_driver(col, value, bench):
    """Pick numeric vs categorical benchmarking automatically."""
    if col in bench.get("numeric", {}):
        return describe_numeric(col, value, bench)
    if col in bench.get("categorical", {}):
        return describe_categorical(col, value, bench)
    return f"{value}"
