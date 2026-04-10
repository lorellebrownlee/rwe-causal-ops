import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

TRUE_ATE = -0.15
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    results = []
    for f in glob.glob("results/*_metrics.json"):
        with open(f) as fh:
            results.append(json.load(fh))
    return pd.DataFrame(results)


def make_forest_plot(df, output_path):
    # Only plot methods that have ATE + SE (excludes survival, ITS, E-value)
    plot_df = df.dropna(subset=["ATE", "SE"]).copy()
    plot_df = plot_df.sort_values("bias", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 1.2)))

    colors = plt.cm.tab10.colors
    for i, row in plot_df.iterrows():
        ci_lo = row["ATE"] - 1.96 * row["SE"]
        ci_hi = row["ATE"] + 1.96 * row["SE"]
        ax.errorbar(
            row["ATE"],
            i,
            xerr=[[row["ATE"] - ci_lo], [ci_hi - row["ATE"]]],
            fmt="o",
            color=colors[i % len(colors)],
            capsize=6,
            markersize=8,
            linewidth=2,
            label=row["method"],
        )

    # True ATE reference line
    ax.axvline(
        TRUE_ATE,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"True ATE = {TRUE_ATE}",
    )
    ax.axvline(0, color="grey", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["method"], fontsize=12)
    ax.set_xlabel("Average Treatment Effect (Risk Difference)", fontsize=12)
    ax.set_title(
        "ATE Estimates vs True Effect — Method Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def make_bias_table(df):
    table_df = df[["method", "ATE", "SE", "bias", "true_ATE"]].copy()
    table_df = table_df.sort_values("bias")
    table_df.columns = ["Method", "ATE Estimate", "SE", "Bias from Truth", "True ATE"]
    for col in ["ATE Estimate", "SE", "Bias from Truth", "True ATE"]:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
    return table_df.to_html(index=False, classes="table", border=0)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RWE Causal-Ops — Method Comparison Report</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; max-width: 960px;
           margin: 40px auto; padding: 0 20px; color: #1a1a2e; }
    h1   { color: #01696f; border-bottom: 3px solid #01696f; padding-bottom: 8px; }
    h2   { color: #0c4e54; margin-top: 40px; }
    .table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    .table th { background: #01696f; color: white; padding: 10px 14px; text-align: left; }
    .table td { padding: 9px 14px; border-bottom: 1px solid #dcd9d5; }
    .table tr:nth-child(even) td { background: #f7f6f2; }
    .forest-img { width: 100%; max-width: 900px; margin: 24px 0;
                  border: 1px solid #dcd9d5; border-radius: 8px; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 16px; margin: 24px 0; }
    .metric-card { background: #f7f6f2; border-radius: 8px; padding: 16px 20px;
                   border-left: 4px solid #01696f; }
    .metric-card .label { font-size: 12px; color: #7a7974; text-transform: uppercase;
                          letter-spacing: 0.05em; }
    .metric-card .value { font-size: 24px; font-weight: 700; color: #01696f; }
    .callout { background: #cedcd8; border-radius: 8px;
               padding: 16px 20px; margin: 20px 0; }
    footer   { margin-top: 60px; color: #7a7974; font-size: 13px;
               border-top: 1px solid #dcd9d5; padding-top: 16px; }
  </style>
</head>
<body>
  <h1>RWE Causal-Ops — Statistical Methods Comparison</h1>
  <p>Synthetic observational cohort (n={{ n_patients }}). True ATE = {{ true_ate }}.</p>

  <div class="metric-grid">
    <div class="metric-card">
      <div class="label">Best Method</div>
      <div class="value">{{ best_method }}</div>
    </div>
    <div class="metric-card">
      <div class="label">Lowest Bias</div>
      <div class="value">{{ lowest_bias }}</div>
    </div>
    <div class="metric-card">
      <div class="label">Methods Run</div>
      <div class="value">{{ n_methods }}</div>
    </div>
  </div>

  <h2>Forest Plot — ATE Estimates vs True Effect</h2>
  <img class="forest-img" src="forest_plot.png" alt="Forest plot of ATE estimates">

  <h2>Bias Table</h2>
  {{ bias_table }}

  <h2>E-value Sensitivity Analysis</h2>
  <div class="callout">
    <strong>Observed RR:</strong> {{ evalue_rr }}<br>
    <strong>E-value (point estimate):</strong> {{ evalue_point }}<br>
    <strong>E-value (CI bound):</strong> {{ evalue_ci }}<br>
    <strong>Induced bias from true confounder u:</strong> {{ induced_bias }}<br>
    <br>
    An unmeasured confounder would need to be associated with both treatment
    and outcome by a factor of at least <strong>{{ evalue_point }}</strong>×
    to fully explain away the AIPW result. The true confounder <em>u</em>
    built into the DGP induces a bias of <strong>{{ induced_bias }}</strong>×.
  </div>

  <footer>Generated by rwe-causal-ops reporting pipeline. True ATE = {{ true_ate }}.</footer>
</body>
</html>
"""


if __name__ == "__main__":
    df = load_results()
    forest_path = os.path.join(OUTPUT_DIR, "forest_plot.png")
    make_forest_plot(df, forest_path)

    evalue_row = df[df["method"] == "E-value"]
    evalue_data = evalue_row.iloc[0] if not evalue_row.empty else {}

    best_row = df.dropna(subset=["bias"]).sort_values("bias").iloc[0]

    cohort = pd.read_csv("data/interim/cohort_imputed.csv")

    template = Template(HTML_TEMPLATE)
    html = template.render(
        n_patients=len(cohort),
        true_ate=TRUE_ATE,
        best_method=best_row["method"],
        lowest_bias=f"{best_row['bias']:.4f}",
        n_methods=len(df),
        bias_table=make_bias_table(df),
        evalue_rr=(
            f"{evalue_data.get('observed_RR', '—'):.4f}"
            if isinstance(evalue_data.get("observed_RR"), float)
            else "—"
        ),
        evalue_point=(
            f"{evalue_data.get('evalue_point', '—'):.4f}"
            if isinstance(evalue_data.get("evalue_point"), float)
            else "—"
        ),
        evalue_ci=(
            f"{evalue_data.get('evalue_ci', '—'):.4f}"
            if isinstance(evalue_data.get("evalue_ci"), float)
            else "—"
        ),
        induced_bias=(
            f"{evalue_data.get('induced_bias_RR', '—'):.4f}"
            if isinstance(evalue_data.get("induced_bias_RR"), float)
            else "—"
        ),
    )

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Saved: {report_path}")
