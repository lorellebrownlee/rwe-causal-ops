import json
import os

import mlflow
import pandas as pd
import statsmodels.formula.api as smf

TRUE_ATE = -0.15
INTERVENTION_TIME = 12


def create_panel(df):
    """
    DiD requires a two-period panel: pre and post intervention.
    We split patients into pre/post based on their observed_time
    relative to the intervention point.
    """
    df = df.copy()
    df["post"] = (df["observed_time"] >= INTERVENTION_TIME).astype(int)
    # DiD coefficient is on the interaction: treatment x post
    df["treat_x_post"] = df["treatment"] * df["post"]
    return df


def run_did(df):
    """
    Standard DiD regression:
    Y = b0 + b1*treatment + b2*post + b3*(treatment x post) + covariates + e

    b3 is the DiD estimator — the causal effect of treatment
    b1 captures baseline differences between groups
    b2 captures time trend common to both groups
    """
    panel = create_panel(df)

    # Pre-trend check: outcomes should trend similarly pre-intervention
    pre = panel[panel["post"] == 0]
    pre_treated_mean = pre[pre["treatment"] == 1]["outcome"].mean()
    pre_control_mean = pre[pre["treatment"] == 0]["outcome"].mean()
    pre_diff = abs(pre_treated_mean - pre_control_mean)

    # Main DiD regression with covariates
    formula = (
        "outcome ~ treatment + post + treat_x_post "
        "+ age + sex + cci + n_meds + smoker + baseline_bp"
    )
    model = smf.ols(formula, data=panel).fit(cov_type="HC3")

    did_coef = model.params["treat_x_post"]
    did_se = model.bse["treat_x_post"]
    did_pval = model.pvalues["treat_x_post"]
    ci_lower = model.conf_int().loc["treat_x_post", 0]
    ci_upper = model.conf_int().loc["treat_x_post", 1]
    r2 = model.rsquared

    return did_coef, did_se, did_pval, ci_lower, ci_upper, r2, pre_diff, model


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cohort_imputed.csv")

    mlflow.set_experiment("rwe-causal-ops")

    with mlflow.start_run(run_name="DiD"):
        mlflow.set_tag("method", "Difference-in-Differences")
        mlflow.log_param("intervention_time", INTERVENTION_TIME)
        mlflow.log_param("se_type", "HC3 (heteroscedasticity-robust)")
        mlflow.log_param("covariates_adjusted", True)

        coef, se, pval, ci_lo, ci_hi, r2, pre_diff, model = run_did(df)
        bias = abs(coef - TRUE_ATE)

        mlflow.log_metrics(
            {
                "DiD_estimate": coef,
                "DiD_SE": se,
                "DiD_pval": pval,
                "CI_lower": ci_lo,
                "CI_upper": ci_hi,
                "r_squared": r2,
                "bias_from_truth": bias,
                "pre_period_group_diff": pre_diff,
            }
        )

        print(model.summary())
        print(f"\nDiD estimate: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
        print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"Bias from truth: {bias:.4f}")
        print(f"Pre-period group difference: {pre_diff:.4f} (should be small)")

        metrics = {
            "method": "DiD",
            "ATE": round(coef, 6),
            "SE": round(se, 6),
            "bias": round(bias, 6),
            "true_ATE": TRUE_ATE,
            "CI_lower": round(ci_lo, 6),
            "CI_upper": round(ci_hi, 6),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/did_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: results/did_metrics.json")
