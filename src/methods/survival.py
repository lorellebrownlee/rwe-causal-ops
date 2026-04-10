import json
import os

import mlflow
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

TRUE_ATE = -0.15


def run_survival(df):
    # --- Kaplan-Meier ---
    kmf_treated = KaplanMeierFitter()
    kmf_control = KaplanMeierFitter()

    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    kmf_treated.fit(
        treated["observed_time"], event_observed=treated["event"], label="Treated"
    )
    kmf_control.fit(
        control["observed_time"], event_observed=control["event"], label="Control"
    )

    # Survival probability at time t=12 months
    t_eval = 12.0
    s1 = kmf_treated.survival_function_at_times(t_eval).values[0]
    s0 = kmf_control.survival_function_at_times(t_eval).values[0]
    km_risk_diff = s1 - s0  # positive = treatment is protective

    # Log-rank test
    lr = logrank_test(
        treated["observed_time"],
        control["observed_time"],
        event_observed_A=treated["event"],
        event_observed_B=control["event"],
    )

    # --- Cox Proportional Hazards ---
    cox_cols = [
        "observed_time",
        "event",
        "treatment",
        "age",
        "sex",
        "cci",
        "n_meds",
        "smoker",
        "baseline_bp",
    ]
    cox_df = df[cox_cols].copy()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="observed_time", event_col="event")

    hr = cph.hazard_ratios_["treatment"]
    hr_lower = np.exp(cph.confidence_intervals_.loc["treatment", "95% lower-bound"])
    hr_upper = np.exp(cph.confidence_intervals_.loc["treatment", "95% upper-bound"])
    cox_pval = cph.summary.loc["treatment", "p"]

    # C-index — measures discriminative ability of the model
    c_index = concordance_index(
        df["observed_time"], -cph.predict_partial_hazard(cox_df), df["event"]
    )

    # Schoenfeld residual test for proportional hazards assumption
    ph_test = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
    mlflow.log_param("ph_assumption_violated", str(ph_test))

    return {
        "km_survival_t1_treated": s1,
        "km_survival_t1_control": s0,
        "km_risk_diff_t12": km_risk_diff,
        "logrank_pval": lr.p_value,
        "cox_HR": hr,
        "cox_HR_lower": hr_lower,
        "cox_HR_upper": hr_upper,
        "cox_pval": cox_pval,
        "c_index": c_index,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="Survival"):
        mlflow.set_tag("method", "Kaplan-Meier + Cox PH")
        mlflow.log_param("t_eval_months", 12)
        mlflow.log_param("cox_penalizer", 0.01)

        results = run_survival(df)
        mlflow.log_metrics(results)

        print(
            f"KM survival at t=12: Treated={results['km_survival_t1_treated']:.4f}, "
            f"Control={results['km_survival_t1_control']:.4f}"
        )
        print(f"KM risk difference: {results['km_risk_diff_t12']:.4f}")
        print(f"Log-rank p-value: {results['logrank_pval']:.6f}")
        print(
            f"Cox HR: {results['cox_HR']:.4f} "
            f"[{results['cox_HR_lower']:.4f}, {results['cox_HR_upper']:.4f}]"
        )
        print(f"C-index: {results['c_index']:.4f}")

        metrics = {
            "method": "Survival",
            "ATE": round(results["km_risk_diff_t12"], 6),
            "SE": None,
            "bias": None,
            "true_ATE": TRUE_ATE,
            "cox_HR": round(results["cox_HR"], 6),
            "c_index": round(results["c_index"], 6),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/survival_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/survival_metrics.json")
