import json
import os

import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm

TRUE_ATE = -0.15

# ITS requires a time-series structure — we engineer one from the cohort
# by treating the study period as monthly aggregates with an intervention
# at month 12 (the "policy change" point)
INTERVENTION_TIME = 12


def create_time_series(df):
    """
    Aggregate cohort into monthly outcome rates.
    Patients are assigned to months by their observed_time.
    Intervention splits the series at month 12.
    """
    df = df.copy()
    df["month"] = np.floor(df["observed_time"]).clip(1, 36).astype(int)
    monthly = (
        df.groupby("month")
        .agg(outcome_rate=("outcome", "mean"), n=("outcome", "count"))
        .reset_index()
    )
    monthly["time"] = monthly["month"]
    monthly["intervention"] = (monthly["time"] >= INTERVENTION_TIME).astype(int)
    monthly["time_since_intervention"] = (
        monthly["time"] - INTERVENTION_TIME
    ) * monthly["intervention"]
    return monthly


def run_its(monthly):
    """
    Segmented regression (standard ITS model):
    Y = b0 + b1*time + b2*intervention + b3*time_since_intervention + e

    b2 = immediate level change at intervention
    b3 = change in slope after intervention
    """
    X = monthly[["time", "intervention", "time_since_intervention"]]
    X = sm.add_constant(X)
    Y = monthly["outcome_rate"]

    model = sm.OLS(Y, X).fit(cov_type="HC3")  # heteroscedasticity-robust SE

    b2 = model.params["intervention"]  # level change
    b3 = model.params["time_since_intervention"]  # slope change
    b2_pval = model.pvalues["intervention"]
    b3_pval = model.pvalues["time_since_intervention"]
    r_squared = model.rsquared

    return b2, b3, b2_pval, b3_pval, r_squared, model


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="ITS"):
        mlflow.set_tag("method", "Interrupted Time Series")
        mlflow.log_param("intervention_time", INTERVENTION_TIME)
        mlflow.log_param("se_type", "HC3 (heteroscedasticity-robust)")

        monthly = create_time_series(df)
        b2, b3, b2_pval, b3_pval, r2, model = run_its(monthly)

        mlflow.log_metrics(
            {
                "level_change": b2,
                "slope_change": b3,
                "level_change_pval": b2_pval,
                "slope_change_pval": b3_pval,
                "r_squared": r2,
            }
        )

        print(model.summary())
        print(f"\nLevel change at intervention: {b2:.4f} (p={b2_pval:.4f})")
        print(f"Slope change after intervention: {b3:.4f} (p={b3_pval:.4f})")
        print(f"R-squared: {r2:.4f}")

        metrics = {
            "method": "ITS",
            "ATE": round(b2, 6),  # level change is the closest to ATE
            "SE": None,
            "bias": None,
            "true_ATE": TRUE_ATE,
            "slope_change": round(b3, 6),
            "r_squared": round(r2, 6),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/its_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/its_metrics.json")
