import numpy as np
import pandas as pd


def generate_cohort(n=5000, true_ate=-0.15, seed=42):
    """
    Simulate an observational EHR cohort.
    True ATE is known: treatment reduces event probability by 15pp.
    Confounding is introduced via covariate-driven treatment assignment.
    """
    rng = np.random.default_rng(seed)

    # Baseline covariates
    age = rng.normal(60, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.48, n)  # 1 = female
    cci = rng.poisson(2.5, n).clip(0, 8)  # Charlson Comorbidity Index
    n_meds = rng.poisson(3, n).clip(0, 10)  # number of concomitant meds
    smoker = rng.binomial(1, 0.25, n)
    baseline_bp = rng.normal(135, 18, n).clip(90, 200)

    # Unmeasured confounder (creates residual bias for sensitivity analyses)
    u = rng.normal(0, 1, n)

    # Treatment assignment (propensity driven by age, cci, smoker)
    log_odds_tx = -1.5 + 0.03 * age + 0.25 * cci + 0.4 * smoker - 0.1 * n_meds + 0.3 * u
    ps_true = 1 / (1 + np.exp(-log_odds_tx))
    treatment = rng.binomial(1, ps_true, n)

    # Potential outcomes
    log_odds_y0 = (
        -2.0 + 0.02 * age + 0.3 * cci + 0.5 * smoker + 0.01 * baseline_bp + 0.4 * u
    )
    y0_prob = 1 / (1 + np.exp(-log_odds_y0))
    y1_prob = (y0_prob + true_ate).clip(0.01, 0.99)  # shift by true ATE

    # Observed outcome
    outcome = np.where(
        treatment == 1, rng.binomial(1, y1_prob, n), rng.binomial(1, y0_prob, n)
    )

    # Time-to-event for survival methods (Weibull)
    scale = np.where(treatment == 1, 18, 14)
    tte = rng.weibull(1.5, n) * scale
    censoring_time = rng.uniform(6, 36, n)
    observed_time = np.minimum(tte, censoring_time)
    event = (tte <= censoring_time).astype(int)

    # Introduce 8% MCAR missingness on baseline_bp
    missing_mask = rng.binomial(1, 0.08, n).astype(bool)
    baseline_bp_obs = baseline_bp.copy().astype(float)
    baseline_bp_obs[missing_mask] = np.nan

    df = pd.DataFrame(
        {
            "patient_id": range(n),
            "age": age,
            "sex": sex,
            "cci": cci,
            "n_meds": n_meds,
            "smoker": smoker,
            "baseline_bp": baseline_bp_obs,
            "treatment": treatment,
            "outcome": outcome,
            "observed_time": observed_time,
            "event": event,
            "ps_true": ps_true,
        }
    )
    return df, true_ate


if __name__ == "__main__":
    df, ate = generate_cohort()
    df.to_csv("data/raw/cohort.csv", index=False)
    print(f"Cohort saved. True ATE = {ate}")
