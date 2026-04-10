import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

COVARIATES = ["age", "sex", "cci", "n_meds", "smoker", "baseline_bp"]
TRUE_ATE = -0.15


def compute_smd(df, col, treatment_col="treatment"):
    """Standardised Mean Difference for a single covariate."""
    treated = df[df[treatment_col] == 1][col]
    control = df[df[treatment_col] == 0][col]
    pooled_std = np.sqrt((treated.std() ** 2 + control.std() ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return abs((treated.mean() - control.mean()) / pooled_std)


def run_psm(df, caliper=0.2):
    # Fit propensity score model
    X = df[COVARIATES]
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, df["treatment"])
    df = df.copy()
    df["ps"] = ps_model.predict_proba(X)[:, 1]

    # 1:1 nearest neighbour matching with caliper
    treated = df[df["treatment"] == 1].reset_index(drop=True)
    control = df[df["treatment"] == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["ps"]])
    distances, indices = nn.kneighbors(treated[["ps"]])

    # Apply caliper — drop pairs where PS difference exceeds caliper * SD(PS)
    caliper_width = caliper * df["ps"].std()
    mask = distances.flatten() <= caliper_width
    treated_matched = treated[mask]
    control_matched = control.iloc[indices.flatten()[mask]]

    matched = pd.concat([treated_matched, control_matched])

    # Estimate ATE on matched sample
    ate = (
        matched[matched["treatment"] == 1]["outcome"].mean()
        - matched[matched["treatment"] == 0]["outcome"].mean()
    )

    # SE via bootstrap
    bootstrap_ates = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        sample = matched.sample(
            len(matched), replace=True, random_state=int(rng.integers(1e6))
        )
        b_ate = (
            sample[sample["treatment"] == 1]["outcome"].mean()
            - sample[sample["treatment"] == 0]["outcome"].mean()
        )
        bootstrap_ates.append(b_ate)
    se = np.std(bootstrap_ates)

    # Post-match SMD for each covariate
    smds = {col: compute_smd(matched, col) for col in COVARIATES}
    max_smd = max(smds.values())

    return ate, se, max_smd, len(treated_matched)


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="PSM"):
        mlflow.set_tag("method", "Propensity Score Matching")
        mlflow.log_param("caliper", 0.2)
        mlflow.log_param("matching", "1:1 nearest neighbour")

        ate, se, max_smd, n_matched = run_psm(df)
        bias = abs(ate - TRUE_ATE)

        mlflow.log_metrics(
            {
                "ATE_estimate": ate,
                "ATE_SE": se,
                "bias_from_truth": bias,
                "max_SMD_post_match": max_smd,
                "n_matched_pairs": n_matched,
            }
        )

        print(f"PSM ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"Bias from truth: {bias:.4f}")
        print(f"Max post-match SMD: {max_smd:.4f} (target < 0.1)")
        print(f"Matched pairs: {n_matched}")

        metrics = {
            "method": "PSM",
            "ATE": round(ate, 6),
            "SE": round(se, 6),
            "bias": round(bias, 6),
            "true_ATE": TRUE_ATE,
        }

    import os

    os.makedirs("results", exist_ok=True)
    with open("results/psm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/psm_metrics.json")
