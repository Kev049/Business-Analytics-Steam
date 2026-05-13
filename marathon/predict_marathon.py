"""
predict_marathon.py — Marathon (2026) case study.

Trains the production-equivalent GBR per horizon on Kevin's data, then predicts
Marathon's CCU at 3, 6 and 12 months post-launch using the single day-7 CCU
input from the project's data pipeline (`marathon_features.csv`,
semicolon-delimited; sourced 2026-05-12). The day-7 value (60,335 CCU) is
the exact concurrent-user count from the `players_7days_after_release`
column — dated 2026-03-12, the second week of March 2026, seven days after
the 2026-03-05 launch.

Preprocessing matches `../visualisation/audit_plots.py` exactly:
  - drop rows with missing target
  - apply the >=100 first-week-players cutoff (matches all three Kevin scripts)
  - mean-impute numeric features
  - log1p any column whose name contains "review" or "player"
  - 85/15 random split with seed 42
  - GBR(n=150, depth=3, lr=0.05, seed=42)

Usage:
    python predict_marathon.py

Output:
    figures/07_marathon_case_study.png    (1x2 panel: bars + trajectory)
    stdout: per-horizon prediction table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# === CONFIG ===
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "models-6-5"
MARATHON_CSV = HERE / "marathon_features.csv"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

HORIZONS = {
    "3m": ("three_months_model/three_month_final.csv", "players_month_3_after_release"),
    "6m": ("six_months_model/six_month_final.csv", "players_month_6_after_release"),
    "12m": ("twelve_months_model/twelve_month_final.csv", "players_month_12_after_release"),
}
WEEK1_COL = "players_7days_after_release"
RANDOM_STATE = 42
TEST_SIZE = 0.15
GBR_KWARGS = dict(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)

# Observed Steam trajectory. Sub-monthly granularity near launch (peak +
# day-7 snapshot + launch-month avg) plus calendar-month averages afterward.
# Marathon released 2026-03-05 (Steam App 3065800, Bungie); values sourced
# from SteamCharts on 2026-05-11. The day-7 entry doubles as the model's
# input value; it is plotted as a blue square on the prediction line and
# annotated there (rather than on the observed line) to keep labels readable.
LAUNCH_PEAK_CCU = 77358   # all-time peak on 2026-03-08 (3 days post-launch)
MARCH_AVG_CCU = 35040     # March 2026 monthly average (Mar 5-31)
APRIL_AVG_CCU = 15833     # April 2026 monthly average
MAY_AVG_CCU = 3869        # last-30-days as of mid-May 2026
MARATHON_OBSERVED = [
    (3 / 30.0, LAUNCH_PEAK_CCU, "launch peak (Mar 8)"),
    # day-7 (Mar 12) at month 7/30 ~= 0.23 inserted by render_figure
    (0.5, MARCH_AVG_CCU, "March avg"),
    (1.0, APRIL_AVG_CCU, "April avg"),
    (2.0, MAY_AVG_CCU, "May avg"),
]

# Single week-1 input: the exact day-7 CCU value from the project's data
# pipeline. The CSV (`marathon_features.csv`) provides this at runtime; the
# constant below is kept only as a fallback if the CSV is missing the column.
EMPIRICAL_WEEK1 = 60335  # day-7 CCU snapshot on 2026-03-12 (week 2 of March)


# === DATA LOADING ===

def load_horizon(horizon: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load Kevin's horizon CSV and return (X, y, feature_names) matching audit_plots.py."""
    rel_path, target_col = HORIZONS[horizon]
    df = pd.read_csv(DATA_DIR / rel_path)
    df.columns = df.columns.str.strip().str.replace("﻿", "")
    df = df.dropna(subset=[target_col])
    df = df[df[WEEK1_COL] >= 100]

    drop_cols = [c for c in df.columns if "name" in c.lower() or "app_id" in c.lower()]
    feature_df = df.drop(columns=drop_cols + [target_col], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(feature_df), columns=feature_df.columns)
    for col in X.columns:
        if "review" in col.lower() or "player" in col.lower():
            X[col] = np.log1p(X[col])
    y = np.log1p(df[target_col].values)
    return X, pd.Series(y, name=target_col), list(X.columns)


def load_marathon_features() -> pd.Series:
    """Load Dominik's Marathon feature row from marathon_features.csv (semicolon-delimited)."""
    df = pd.read_csv(MARATHON_CSV, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace("﻿", "")
    return df.iloc[0]


# === PREDICTION ===

def train_horizon_gbr(horizon: str) -> tuple[GradientBoostingRegressor, list[str]]:
    """Train one GBR on Kevin's data for the given horizon and return (model, feature_names)."""
    X, y, feat_names = load_horizon(horizon)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return GradientBoostingRegressor(**GBR_KWARGS).fit(X_tr, y_tr), feat_names


def predict_marathon(gbr: GradientBoostingRegressor, feat_names: list[str],
                     marathon: pd.Series, week1_override: float) -> float:
    """Predict Marathon's CCU at one horizon with the given week-1 input scenario.

    Marathon's feature row is reindexed against the horizon's training columns;
    absent columns default to 0 (correct for unset one-hots). week1_override
    replaces players_7days_after_release for the scenario. The same log1p
    transform load_horizon applied to the training features is applied here.
    """
    row = pd.Series(0.0, index=feat_names)
    for col in feat_names:
        if col in marathon.index:
            try:
                row[col] = float(marathon[col])
            except (ValueError, TypeError):
                row[col] = 0.0
    row[WEEK1_COL] = float(week1_override)
    for c in row.index:
        if "review" in c.lower() or "player" in c.lower():
            row[c] = np.log1p(row[c])
    pred_log = gbr.predict(pd.DataFrame([row]))[0]
    return float(np.expm1(pred_log))


# === FIGURE ===

def render_figure(predictions: dict, marathon: pd.Series, empirical_week1: float) -> Path:
    """1x2 figure: single day-7 input -> three horizon predictions.

    Left panel:  predicted CCU bars at 3, 6, 12 months for the single day-7
                 input (60,335 CCU from the data pipeline, dated 2026-03-12).
    Right panel: observed Steam trajectory (Mar-May 2026) plus the three
                 predicted points (Jun 2026 / Sep 2026 / Mar 2027) plus a
                 naive -55%/mo decay baseline anchored on April.
    """
    horizons = ["3m", "6m", "12m"]
    label = "Empirical (pipeline)"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bar_colors = ["#9ec5e8", "#4d8fd1", "#1f4e79"]  # graded blue across horizons

    # --- LEFT: predicted CCU bars at the 3 horizons (single day-7 input) ---
    ax = axes[0]
    x = np.arange(len(horizons))
    vals = [predictions[(h, label)] for h in horizons]
    bars = ax.bar(x, vals, width=0.55, color=bar_colors,
                  edgecolor="k", linewidth=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                f"{int(v):,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["3-month", "6-month", "12-month"], fontsize=11)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Predicted CCU")
    ax.set_title(f"Predicted CCU per horizon\n(day-7 input: {int(empirical_week1):,} CCU, 2026-03-12)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.18)

    # --- RIGHT: observed trajectory + day-7 input + 3 predicted points + naive baseline ---
    ax = axes[1]

    day7_x = 7 / 30.0  # months: ~0.23
    day7_y = float(empirical_week1)

    # Observed Steam CCU. Includes sub-monthly granularity near launch
    # (peak on Mar 8, day-7 on Mar 12, March avg) plus April / May
    # monthly averages. Day-7 is part of both observed reality and the
    # model's input, so it appears on this line AND as the starting
    # square of the prediction line below.
    observed_points = [
        (MARATHON_OBSERVED[0][0], MARATHON_OBSERVED[0][1]),  # launch peak
        (day7_x, day7_y),                                     # day-7
        (MARATHON_OBSERVED[1][0], MARATHON_OBSERVED[1][1]),  # March avg
        (MARATHON_OBSERVED[2][0], MARATHON_OBSERVED[2][1]),  # April avg
        (MARATHON_OBSERVED[3][0], MARATHON_OBSERVED[3][1]),  # May avg
    ]
    obs_x = [p[0] for p in observed_points]
    obs_y = [p[1] for p in observed_points]
    ax.plot(obs_x, obs_y, "o-", color="black", linewidth=2.2, markersize=8,
            label="Observed Steam CCU (launch peak, day-7, monthly avg)", zorder=10)

    # Annotate observed points (skip day-7 -- the prediction line annotates it).
    observed_annotations = [
        (MARATHON_OBSERVED[0][0], MARATHON_OBSERVED[0][1], MARATHON_OBSERVED[0][2]),
        (MARATHON_OBSERVED[1][0], MARATHON_OBSERVED[1][1], MARATHON_OBSERVED[1][2]),
        (MARATHON_OBSERVED[2][0], MARATHON_OBSERVED[2][1], MARATHON_OBSERVED[2][2]),
        (MARATHON_OBSERVED[3][0], MARATHON_OBSERVED[3][1], MARATHON_OBSERVED[3][2]),
    ]
    for mx, my, atext in observed_annotations:
        ax.annotate(f"{int(my):,}\n{atext}", (mx, my),
                    textcoords="offset points", xytext=(7, 8),
                    fontsize=8, fontweight="bold")

    # Model prediction line: day-7 input -> month 3 -> month 6 -> month 12.
    pred_months = [3, 6, 12]
    pred_y = [predictions[(f"{m}m", label)] for m in pred_months]
    line_x = [day7_x] + pred_months
    line_y = [day7_y] + pred_y
    ax.plot(line_x, line_y, "--", color="#1f4e79", linewidth=1.8,
            marker="s", markersize=10, zorder=11,
            label="Model: day-7 input -> 3/6/12-month predictions")
    ax.annotate(f"day-7 input\n{int(day7_y):,}\n(Mar 12)",
                (day7_x, day7_y), textcoords="offset points",
                xytext=(8, -28), fontsize=8, color="#1f4e79", fontweight="bold")
    for m, y_val in zip(pred_months, pred_y):
        ax.annotate(f"{int(y_val):,}", (m, y_val), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, color="#1f4e79", fontweight="bold")

    # Naive -55%/mo decay anchored on April 2026 (the last full month of decay).
    naive_x = [1, 2, 3, 6, 12]
    base_april = APRIL_AVG_CCU
    decay = 0.45  # i.e. -55% MoM
    naive_y = [base_april * (decay ** (m - 1)) for m in naive_x]
    ax.plot(naive_x, naive_y, ":", color="#888", linewidth=1.5,
            marker="^", markersize=6, label="Naive -55%/mo decay")

    ax.set_xticks([0, 1, 2, 3, 6, 12])
    ax.set_xticklabels(["0\nMar 26", "1\nApr", "2\nMay", "3\nJun",
                         "6\nSep", "12\nMar 27"])
    ax.set_xlabel("Months since launch (2026-03-05)")
    ax.set_ylabel("Average monthly CCU")
    ax.set_title("CCU trajectory: observed (launch peak -> day-7 -> monthly avg) + model predictions")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Case study: Marathon (2026) - Bungie - Steam App 3065800 - released 2026-03-05",
        fontsize=12, y=1.02, fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "07_marathon_case_study.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# === MAIN ===

def main() -> int:
    marathon = load_marathon_features()
    empirical_week1 = float(marathon[WEEK1_COL])

    print("=== Marathon feature snapshot (from data pipeline) ===")
    for key in [
        "app_id", "name", "type", "players_7days_after_release",
        "early_reviews_total", "early_reviews_positive", "early_reviews_negative",
        "early_positive_ratio", "early_review_score", "metacritic_score",
        "achievements_total", "dlc_count", "required_age", "price_final_usd_cents",
    ]:
        val = marathon.get(key, "(absent)")
        print(f"  {key}: {val}")

    label = "Empirical (pipeline)"
    print(f"\n=== Predicting at 3 horizons from single day-7 input ({int(empirical_week1):,} CCU, 2026-03-12) ===")
    predictions: dict[tuple[str, str], float] = {}
    for h in HORIZONS:
        print(f"  training GBR for {h} horizon ...")
        gbr, feat_names = train_horizon_gbr(h)
        predictions[(h, label)] = predict_marathon(gbr, feat_names, marathon, float(empirical_week1))
        print(f"  {h:>3} -> {predictions[(h, label)]:>9,.0f} CCU")

    out = render_figure(predictions, marathon, empirical_week1)
    print(f"\n=== Figure rendered: {out} ===")

    # Paste-friendly summary for §6.5 Marathon prose
    p3 = predictions[("3m", label)]
    p6 = predictions[("6m", label)]
    p12 = predictions[("12m", label)]
    print("\n=== Numbers for report prose (paste-friendly) ===")
    print(f"  day-7 input: {int(empirical_week1):,} CCU (2026-03-12, week 2 of March 2026)")
    print(f"  3-month  prediction: {p3:,.0f} CCU")
    print(f"  6-month  prediction: {p6:,.0f} CCU")
    print(f"  12-month prediction: {p12:,.0f} CCU")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
