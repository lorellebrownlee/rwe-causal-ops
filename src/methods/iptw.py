import json
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

COVARIATES = ["age", "sex", "cci", "n_meds", "smoker", "baseline_bp"]
TRUE_ATE = -0.15


def compute_iptw(df, trim_percentile=1):
    # Fit propensity score model
    X = df[COVARIATES]
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, df["treatment"])
    df = df.copy()
    df["ps"] = ps_model.predict_proba(X)[:, 1]

    # Stabilised weights: P(A) / P(A|X)
    p_treated = df["treatment"].mean()
    p_control = 1 - p_treated

    df["sw"] = np.where(
        df["treatment"] == 1,
        p_treated / df["ps"],
        p_control / (1 - df["ps"]),
    )

    # Trim weights at percentile to reduce influence of extreme weights
    lower = np.percentile(df["sw"], trim_percentile)
    upper = np.percentile(df["sw"], 100 - trim_percentile)
    df["sw_trimmed"] = df["sw"].clip(lower, upper)

    # Effective sample size — a balance diagnostic
    ess = (df["sw_trimmed"].sum() ** 2) / (df["sw_trimmed"] ** 2).sum()

    # Weighted outcome model to estimate ATE
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    y1 = np.average(treated["outcome"], weights=treated["sw_trimmed"])
    y0 = np.average(control["outcome"], weights=control["sw_trimmed"])
    ate = y1 - y0

    # SE via bootstrap
    bootstrap_ates = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        idx = rng.integers(0, len(df), len(df))
        b = df.iloc[idx]
        b_treated = b[b["treatment"] == 1]
        b_control = b[b["treatment"] == 0]
        if len(b_treated) == 0 or len(b_control) == 0:
            continue
        b_y1 = np.average(b_treated["outcome"], weights=b_treated["sw_trimmed"])
        b_y0 = np.average(b_control["outcome"], weights=b_control["sw_trimmed"])
        bootstrap_ates.append(b_y1 - b_y0)
    se = np.std(bootstrap_ates)

    weight_variance = df["sw_trimmed"].var()

    return ate, se, ess, weight_variance


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="IPTW"):
        mlflow.set_tag("method", "Inverse Probability of Treatment Weighting")
        mlflow.log_param("weights", "stabilised")
        mlflow.log_param("trim_percentile", 1)

        ate, se, ess, weight_var = compute_iptw(df)
        bias = abs(ate - TRUE_ATE)

        mlflow.log_metrics(
            {
                "ATE_estimate": ate,
                "ATE_SE": se,
                "bias_from_truth": bias,
                "effective_sample_size": ess,
                "weight_variance": weight_var,
            }
        )

        print(f"IPTW ATE:  {ate:.4f} (SE: {se:.4f})")
        print(f"Bias from truth: {bias:.4f}")
        print(f"Effective sample size: {ess:.0f} / {len(df)}")
        print(f"Weight variance: {weight_var:.4f}")

        metrics = {
            "method": "IPTW",
            "ATE": round(ate, 6),
            "SE": round(se, 6),
            "bias": round(bias, 6),
            "true_ATE": TRUE_ATE,
        }

    os.makedirs("results", exist_ok=True)
    with open("results/iptw_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/iptw_metrics.json")
