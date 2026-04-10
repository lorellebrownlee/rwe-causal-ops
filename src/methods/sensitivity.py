import json
import os

import mlflow
import numpy as np
import pandas as pd

TRUE_ATE = -0.15


def risk_ratio_from_rd(rd, p0):
    """
    Convert a risk difference (RD) to a risk ratio (RR).
    RR = (p0 + RD) / p0
    Required because E-value formula operates on the RR scale.
    """
    p1 = p0 + rd
    p1 = np.clip(p1, 0.001, 0.999)
    p0 = np.clip(p0, 0.001, 0.999)
    return p1 / p0


def compute_evalue(rr):
    """
    VanderWeele & Ding (2017) E-value formula.
    For RR >= 1: E = RR + sqrt(RR * (RR - 1))
    For RR < 1: first convert to RR > 1 by taking reciprocal.

    The E-value is the minimum strength of association (on the RR scale)
    that an unmeasured confounder would need to have with BOTH treatment
    and outcome to fully explain away the observed effect.
    """
    if rr < 1:
        rr = 1 / rr  # convert to > 1 scale
    evalue = rr + np.sqrt(rr * (rr - 1))
    return evalue


def compute_evalue_ci(rr, se_log_rr):
    """
    E-value for the confidence limit closest to the null (RR = 1).
    This is the more conservative and more commonly reported E-value.
    """
    ci_bound = np.exp(np.log(rr) - 1.96 * se_log_rr)  # lower CI
    if ci_bound > 1:
        return compute_evalue(ci_bound)
    ci_bound = np.exp(np.log(rr) + 1.96 * se_log_rr)  # upper CI
    return compute_evalue(ci_bound)


if __name__ == "__main__":
    # Load AIPW results — most principled estimate to base sensitivity on
    with open("results/aipw_metrics.json") as f:
        aipw = json.load(f)

    df = pd.read_csv("data/interim/cohort_imputed.csv")
    p0 = df[df["treatment"] == 0]["outcome"].mean()  # baseline risk

    ate = aipw["ATE"]
    se = aipw["SE"]
    rr = risk_ratio_from_rd(ate, p0)
    se_log_rr = se / abs(ate) if ate != 0 else 0.01  # delta method approximation

    evalue_point = compute_evalue(rr)
    evalue_ci = compute_evalue_ci(rr, se_log_rr)

    # True confounder strength from DGP — known because we built it
    # u affects treatment with OR~1.35 and outcome with OR~1.49
    # (derived from DGP coefficients: 0.3 and 0.4 on logit scale)
    true_u_treatment_rr = np.exp(0.3)  # ~1.35
    true_u_outcome_rr = np.exp(0.4)  # ~1.49

    # Bias formula: how much bias does u actually induce?
    # B = (RR_UD * RR_EU) / (RR_UD + RR_EU - 1)
    induced_bias_rr = (true_u_outcome_rr * true_u_treatment_rr) / (
        true_u_outcome_rr + true_u_treatment_rr - 1
    )

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="Sensitivity_EValue"):
        mlflow.set_tag("method", "E-value Sensitivity Analysis")
        mlflow.log_param("base_estimate", "AIPW")
        mlflow.log_param("baseline_risk_p0", round(p0, 4))

        mlflow.log_metrics(
            {
                "observed_RR": rr,
                "evalue_point_estimate": evalue_point,
                "evalue_CI_bound": evalue_ci,
                "true_confounder_treatment_RR": true_u_treatment_rr,
                "true_confounder_outcome_RR": true_u_outcome_rr,
                "induced_bias_RR": induced_bias_rr,
                "evalue_exceeds_true_bias": float(evalue_point > induced_bias_rr),
            }
        )

        print(f"Baseline risk (p0):       {p0:.4f}")
        print(f"AIPW ATE:                 {ate:.4f}")
        print(f"Observed RR:              {rr:.4f}")
        print(f"\nE-value (point):          {evalue_point:.4f}")
        print(f"E-value (CI bound):       {evalue_ci:.4f}")
        print("\nTrue confounder u:")
        print(f"  RR with treatment:      {true_u_treatment_rr:.4f}")
        print(f"  RR with outcome:        {true_u_outcome_rr:.4f}")
        print(f"  Induced bias (RR):      {induced_bias_rr:.4f}")
        print(f"\nE-value > true bias? {evalue_point > induced_bias_rr}")
        print(
            "(If False: the true confounder u is strong enough to explain away the result)"
        )

        metrics = {
            "method": "E-value",
            "ATE": None,
            "SE": None,
            "bias": None,
            "true_ATE": TRUE_ATE,
            "evalue_point": round(evalue_point, 6),
            "evalue_ci": round(evalue_ci, 6),
            "observed_RR": round(rr, 6),
            "induced_bias_RR": round(induced_bias_rr, 6),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/evalue_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/evalue_metrics.json")
