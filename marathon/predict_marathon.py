"""
predict_marathon.py — Marathon (2026) case study.

Trains the production-equivalent GBR per horizon on Kevin's data, then predicts
Marathon's average monthly CCU at 3, 6 and 12 months post-launch using the
feature row produced by Dominik's data pipeline (`marathon_features.csv`,
semicolon-delimited; sourced 2026-05-12).

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
    figures/07_marathon_case_study.png    (1x2 panel)
    stdout: scenario × horizon prediction table
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

# Observed Steam trajectory (months-since-launch, raw average CCU, label).
# Marathon released 2026-03-05 (Steam App 3065800, Bungie). Values sourced
# from SteamCharts monthly averages on 2026-05-11.
MARATHON_OBSERVED = [
    (0, 35040, "Mar 2026"),
    (1, 15833, "Apr 2026"),
    (2, 3869, "May 2026"),
]

# Three week-1 input scenarios. Empirical = Dominik's pipeline output, the
# headline value for the report. Low and High retain the sensitivity context:
# Low = March 2026 monthly average (biased downward by post-peak decay);
# High = launch-day peak from Steam launch records (2026-03-08).
EMPIRICAL_WEEK1 = 60335  # overridden by Dominik's CSV at runtime; kept for fallback
LOW_WEEK1 = 35040
HIGH_WEEK1 = 77358


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
    """1x2 figure mirroring the layout established in audit_plots.py figure_7."""
    horizons = ["3m", "6m", "12m"]
    scenarios = [
        ("Low (March 2026 avg)", LOW_WEEK1),
        (f"Empirical (Dominik, first-7-day avg)", int(empirical_week1)),
        ("High (launch-day peak)", HIGH_WEEK1),
    ]
    scenario_labels = [s[0] for s in scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#9ec5e8", "#4d8fd1", "#1f4e79"]  # graded blue: Low / Mid / High

    # --- LEFT: predicted CCU bars per horizon, three scenarios ---
    x = np.arange(len(horizons))
    width = 0.26
    ax = axes[0]
    for i, label in enumerate(scenario_labels):
        vals = [predictions[(h, label)] for h in horizons]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=colors[i],
                      edgecolor="k", linewidth=0.3, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                    f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["3-month", "6-month", "12-month"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Predicted average monthly CCU")
    ax.set_title("Predicted CCU per horizon under three week-1 input scenarios")
    ax.legend(title="Week-1 CCU input", fontsize=9, title_fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # --- RIGHT: observed trajectory + predicted bands + naive baseline ---
    ax = axes[1]
    obs_x = [t[0] for t in MARATHON_OBSERVED]
    obs_y = [t[1] for t in MARATHON_OBSERVED]
    ax.plot(obs_x, obs_y, "o-", color="black", linewidth=2.2, markersize=9,
            label="Observed Steam avg CCU", zorder=10)
    for mx, my, _ in MARATHON_OBSERVED:
        ax.annotate(f"{int(my):,}", (mx, my), textcoords="offset points",
                    xytext=(7, 8), fontsize=8, fontweight="bold")

    pred_months = [3, 6, 12]
    for i, label in enumerate(scenario_labels):
        pred_y = [predictions[(f"{m}m", label)] for m in pred_months]
        connect_x = [obs_x[-1]] + pred_months
        connect_y = [obs_y[-1]] + pred_y
        ax.plot(connect_x, connect_y, "--", color=colors[i], linewidth=1.6,
                marker="s", markersize=7, label=f"Model: {label}")

    # Naive -55%/mo decay anchored on April 2026 (the last full month of decay).
    naive_x = [1, 2, 3, 6, 12]
    base_april = 15833
    decay = 0.45  # i.e. -55% MoM
    naive_y = [base_april * (decay ** (m - 1)) for m in naive_x]
    ax.plot(naive_x, naive_y, ":", color="#888", linewidth=1.5,
            marker="^", markersize=6, label="Naive -55%/mo decay")

    ax.set_xticks([0, 1, 2, 3, 6, 12])
    ax.set_xticklabels(["0\nMar 26", "1\nApr", "2\nMay", "3\nJun",
                         "6\nSep", "12\nMar 27"])
    ax.set_xlabel("Months since launch (2026-03-05)")
    ax.set_ylabel("Average monthly CCU")
    ax.set_title("CCU trajectory: observed Mar-May 2026 vs predicted Jun-2026-Mar-2027")
    ax.legend(fontsize=8, loc="upper right")
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

    print("=== Marathon feature snapshot (from Dominik's pipeline) ===")
    for key in [
        "app_id", "name", "type", "players_7days_after_release",
        "early_reviews_total", "early_reviews_positive", "early_reviews_negative",
        "early_positive_ratio", "early_review_score", "metacritic_score",
        "achievements_total", "dlc_count", "required_age", "price_final_usd_cents",
    ]:
        val = marathon.get(key, "(absent)")
        print(f"  {key}: {val}")

    scenarios = [
        ("Low (March 2026 avg)", LOW_WEEK1),
        (f"Empirical (Dominik, first-7-day avg)", int(empirical_week1)),
        ("High (launch-day peak)", HIGH_WEEK1),
    ]

    print("\n=== Predictions per horizon x scenario (raw CCU, back-transformed from log) ===")
    predictions: dict[tuple[str, str], float] = {}
    for h in HORIZONS:
        print(f"  training GBR for {h} horizon ...")
        gbr, feat_names = train_horizon_gbr(h)
        for label, week1 in scenarios:
            predictions[(h, label)] = predict_marathon(gbr, feat_names, marathon, float(week1))
            print(f"  {h:>3} | {label:<42} (week1={week1:>6,}) -> {predictions[(h, label)]:>9,.0f} CCU")

    out = render_figure(predictions, marathon, empirical_week1)
    print(f"\n=== Figure rendered: {out} ===")

    # Also print absolute numbers in markdown-friendly form for pasting into Overleaf prose
    print("\n=== Numbers for report prose (paste-friendly) ===")
    for h in ["3m", "6m", "12m"]:
        low = predictions[(h, "Low (March 2026 avg)")]
        emp = predictions[(h, "Empirical (Dominik, first-7-day avg)")]
        high = predictions[(h, "High (launch-day peak)")]
        print(f"  {h}: Low={low:,.0f}  Empirical={emp:,.0f}  High={high:,.0f}  (range {min(low,emp,high):,.0f}-{max(low,emp,high):,.0f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
