import miceforest as mf
import pandas as pd
import pandera as pa

# --- Schema Validation ---
schema = pa.DataFrameSchema(
    {
        "age": pa.Column(float, pa.Check.between(18, 90)),
        "sex": pa.Column(int, pa.Check.isin([0, 1])),
        "cci": pa.Column(int, pa.Check.between(0, 8)),
        "n_meds": pa.Column(int, pa.Check.between(0, 10)),
        "smoker": pa.Column(int, pa.Check.isin([0, 1])),
        "treatment": pa.Column(int, pa.Check.isin([0, 1])),
        "outcome": pa.Column(int, pa.Check.isin([0, 1])),
        "observed_time": pa.Column(float, pa.Check.greater_than(0)),
        "event": pa.Column(int, pa.Check.isin([0, 1])),
    }
)

df = pd.read_csv("data/raw/cohort.csv")

# Validate raw data
schema.validate(df)
print(f"Validation passed. Shape: {df.shape}")
print(f"Missing baseline_bp: {df['baseline_bp'].isna().sum()} rows")

# Save clean (pre-imputation) version
df.to_csv("data/interim/cohort_clean.csv", index=False)

# --- MICE Imputation ---
# miceforest uses LightGBM to impute missing values
kernel = mf.ImputationKernel(data=df, save_all_iterations_data=True, random_state=42)
kernel.mice(3)  # 3 iterations

df_imputed = kernel.complete_data()

print(f"Missing after imputation: {df_imputed['baseline_bp'].isna().sum()} rows")
df_imputed.to_csv("data/interim/cohort_imputed.csv", index=False)
print("Saved: data/interim/cohort_imputed.csv")
