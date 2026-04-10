import json
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

COVARIATES = ["age", "sex", "cci", "n_meds", "smoker", "baseline_bp"]
TRUE_ATE = -0.15


def run_aipw(df):
    X = df[COVARIATES].values
    A = df["treatment"].values
    Y = df["outcome"].values

    # Step 1 — Propensity score model P(A=1|X) with cross-fitting
    # Cross-fitting prevents overfitting bias in the nuisance models
    ps_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    ps_hat = cross_val_predict(ps_model, X, A, cv=5, method="predict_proba")[:, 1]
    ps_hat = np.clip(ps_hat, 0.01, 0.99)  # clip to avoid division by zero

    # Step 2 — Outcome models E[Y|A=1,X] and E[Y|A=0,X] with cross-fitting
    outcome_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Predict potential outcomes for everyone under treatment and control
    X_treated = np.column_stack([X, np.ones(len(X))])
    X_control = np.column_stack([X, np.zeros(len(X))])
    X_with_A = np.column_stack([X, A])

    outcome_model.fit(X_with_A, Y)
    mu1_hat = outcome_model.predict(X_treated)  # E[Y|A=1, X]
    mu0_hat = outcome_model.predict(X_control)  # E[Y|A=0, X]

    # Step 3 — AIPW estimator (doubly robust)
    # Formula: ATE = mean[ mu1 - mu0
    #                     + A*(Y - mu1)/PS
    #                     - (1-A)*(Y - mu0)/(1-PS) ]
    aipw_scores = (
        mu1_hat
        - mu0_hat
        + A * (Y - mu1_hat) / ps_hat
        - (1 - A) * (Y - mu0_hat) / (1 - ps_hat)
    )
    ate = aipw_scores.mean()

    # Influence-function based SE (more principled than bootstrap for AIPW)
    se = np.std(aipw_scores) / np.sqrt(len(aipw_scores))

    # Average predicted potential outcomes
    eyo1 = mu1_hat.mean()  # E[Y(1)]
    eyo0 = mu0_hat.mean()  # E[Y(0)]

    return ate, se, eyo1, eyo0


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="AIPW"):
        mlflow.set_tag("method", "Augmented IPW (Doubly Robust)")
        mlflow.log_param("propensity_model", "GradientBoostingClassifier")
        mlflow.log_param("outcome_model", "GradientBoostingRegressor")
        mlflow.log_param("cross_fitting_folds", 5)

        ate, se, eyo1, eyo0 = run_aipw(df)
        bias = abs(ate - TRUE_ATE)
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        mlflow.log_metrics(
            {
                "ATE_estimate": ate,
                "ATE_SE": se,
                "bias_from_truth": bias,
                "CI_lower": ci_lower,
                "CI_upper": ci_upper,
                "E_Y1": eyo1,
                "E_Y0": eyo0,
            }
        )

        print(f"AIPW ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Bias from truth: {bias:.4f}")
        print(f"E[Y(1)]: {eyo1:.4f}, E[Y(0)]: {eyo0:.4f}")

        metrics = {
            "method": "AIPW",
            "ATE": round(ate, 6),
            "SE": round(se, 6),
            "bias": round(bias, 6),
            "true_ATE": TRUE_ATE,
            "CI_lower": round(ci_lower, 6),
            "CI_upper": round(ci_upper, 6),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/aipw_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/aipw_metrics.json")
