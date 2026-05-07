"""
audit_plots.py — generate Section 6 figures for the Team 7 final report.

Reads Kevin's v2 CSVs from ../models-6-5/ and emits six PNGs into figures/.
Single-file by design (KISS). Sectioned with `# === FIGURE N: ... ===`.

Usage:
    python audit_plots.py            # generate all six figures
    python audit_plots.py --fig 3    # generate just figure 3
"""

from __future__ import annotations

# === IMPORTS ===
# stdlib
import argparse
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (matplotlib.use must precede pyplot)
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === CONFIG ===
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "models-6-5"
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
RF_KWARGS = dict(n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
LASSO_KWARGS = dict(alpha=0.01)

# === DATA LOADING (Task 2) ===

def load_horizon(horizon: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load one horizon's CSV and return (X, y, feature_names).

    Mirrors Kevin's preprocessing pipeline:
      - drop name / app_id
      - keep only numeric columns
      - mean-impute missing
      - log1p any column whose name contains "review" or "player"
      - log1p the target

    NOTE: pd.get_dummies on genre/categories is omitted because the columns
    arrive already pre-flattened in the CSV (verified during audit).
    """
    rel_path, target_col = HORIZONS[horizon]
    df = pd.read_csv(DATA_DIR / rel_path)
    df.columns = df.columns.str.strip().str.replace("﻿", "")
    df = df.dropna(subset=[target_col])

    drop_cols = [c for c in df.columns if "name" in c.lower() or "app_id" in c.lower()]
    feature_df = df.drop(columns=drop_cols + [target_col], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(
        imputer.fit_transform(feature_df),
        columns=feature_df.columns,
    )
    for col in X.columns:
        if "review" in col.lower() or "player" in col.lower():
            X[col] = np.log1p(X[col])
    y = np.log1p(df[target_col].values)

    return X, pd.Series(y, name=target_col), list(X.columns)


# === MODEL TRAINING + METRICS (Task 2) ===

def train_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> dict:
    """Train one model on a random 85/15 split (seed 42) and return metrics + predictions.

    Returns a dict with keys: model_name, r2, mae, mape, y_test, y_pred, model.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if model_name == "GBR":
        model = GradientBoostingRegressor(**GBR_KWARGS)
    elif model_name == "RF":
        model = RandomForestRegressor(**RF_KWARGS)
    elif model_name == "LR":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = Lasso(**LASSO_KWARGS)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        return _scoring_dict(model_name, y_te, y_pred, model, X.columns)
    else:
        raise ValueError(f"Unknown model {model_name!r}")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return _scoring_dict(model_name, y_te, y_pred, model, X.columns)


def _scoring_dict(name, y_te, y_pred, model, feature_names) -> dict:
    """Compute R² and MAE on log-CCU plus MAPE on raw CCU."""
    y_te_arr = np.asarray(y_te)
    y_te_raw = np.expm1(y_te_arr)
    y_pred_raw = np.expm1(y_pred)
    nonzero = y_te_raw > 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_te_raw[nonzero] - y_pred_raw[nonzero]) / y_te_raw[nonzero])) * 100)
    else:
        mape = float("nan")
    return {
        "model_name": name,
        "r2": float(r2_score(y_te, y_pred)),
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "mape": mape,
        "y_test": y_te_arr,
        "y_pred": np.asarray(y_pred),
        "model": model,
        "feature_names": list(feature_names),
    }


def build_results() -> dict:
    """Run all (horizon × feature_state × model_name) combinations.

    Returns: dict[(horizon, has_feature, model_name)] -> scoring_dict
    """
    out = {}
    for horizon in HORIZONS:
        X_full, y, feat_names = load_horizon(horizon)
        for has_feat in (True, False):
            X = X_full.copy() if has_feat else X_full.drop(columns=[WEEK1_COL], errors="ignore")
            for model_name in ("LR", "RF", "GBR"):
                key = (horizon, has_feat, model_name)
                print(f"  training {model_name:<3} | horizon={horizon} | with_week1={has_feat}")
                out[key] = train_and_score(X, y, model_name)
    return out


# Cache results across figure calls (avoids retraining 18 models per --fig invocation)
_RESULTS_CACHE: dict | None = None


def get_results() -> dict:
    global _RESULTS_CACHE
    if _RESULTS_CACHE is None:
        _RESULTS_CACHE = build_results()
    return _RESULTS_CACHE


# === FIGURE 1: Feature Importance (Task 3) ===

def figure_1_feature_importance() -> None:
    """Top-10 GBR Gini feature importance, horizontal bar chart, 3-month horizon."""
    results = get_results()
    res = results[("3m", True, "GBR")]
    importances = pd.Series(res["model"].feature_importances_, index=res["feature_names"])
    top10 = importances.sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10.index, top10.values, color="steelblue")
    ax.set_xlabel("Gini importance")
    ax.set_title("Top-10 features — Gradient Boosting Regressor (3-month horizon)")
    plt.tight_layout()
    out = FIGURES_DIR / "01_feature_importance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 2: Model Comparison (Task 4) ===

def figure_2_model_comparison() -> None:
    """2×2 grid. Rows = no/with players_7days. Cols = R² / MAE.
    Bars = LR / RF / GBR, grouped by horizon (3m / 6m / 12m).
    """
    results = get_results()
    horizons = ["3m", "6m", "12m"]
    models = ["LR", "RF", "GBR"]
    feature_states = [False, True]
    metrics = [("r2", "R$^2$"), ("mae", "MAE (log-CCU)")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    width = 0.25
    x = np.arange(len(horizons))

    for row, has_feat in enumerate(feature_states):
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for i, model_name in enumerate(models):
                vals = [results[(h, has_feat, model_name)][metric_key] for h in horizons]
                ax.bar(x + i * width - width, vals, width, label=model_name)
            ax.set_xticks(x)
            ax.set_xticklabels(horizons)
            feature_label = "with players_7days" if has_feat else "without players_7days"
            ax.set_title(f"{feature_label} — {metric_label}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric_label)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "02_model_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 3: Predicted vs Actual (Task 5) ===

def figure_3_pred_vs_actual() -> None:
    """1×3 grid: predicted vs actual CCU per horizon, log-log scale, GBR with feature."""
    results = get_results()
    horizons = ["3m", "6m", "12m"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, h in zip(axes, horizons):
        res = results[(h, True, "GBR")]
        # back-transform from log-CCU to raw CCU for interpretability
        y_true_raw = np.expm1(res["y_test"])
        y_pred_raw = np.expm1(res["y_pred"])
        # filter for log scale (need > 0)
        mask = (y_true_raw > 0) & (y_pred_raw > 0)
        ax.scatter(y_true_raw[mask], y_pred_raw[mask], alpha=0.4, s=10, color="steelblue")
        if mask.sum() > 0:
            lo = max(min(y_true_raw[mask].min(), y_pred_raw[mask].min()), 1e-2)
            hi = max(y_true_raw[mask].max(), y_pred_raw[mask].max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("Actual CCU")
        ax.set_ylabel("Predicted CCU")
        ax.set_title(f"{h} horizon (R² = {res['r2']:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "03_pred_vs_actual.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 4: Residuals (Task 6) ===

def figure_4_residuals() -> None:
    """1×3 grid: residuals (observed − predicted, log-CCU) vs predicted (log-CCU)."""
    results = get_results()
    horizons = ["3m", "6m", "12m"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, h in zip(axes, horizons):
        res = results[(h, True, "GBR")]
        residuals = res["y_test"] - res["y_pred"]
        ax.scatter(res["y_pred"], residuals, alpha=0.4, s=10, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted (log-CCU)")
        ax.set_ylabel("Residual (observed − predicted, log-CCU)")
        ax.set_title(f"{h} horizon")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "04_residuals.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 5: Week-1 vs Target (Task 7) ===

def figure_5_week1_vs_target() -> None:
    """3×2 grid. Rows = horizons (3m/6m/12m). Cols = linear / log scale.
    Each panel: scatter of players_7days vs target + linear regression trendline.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row, h in enumerate(["3m", "6m", "12m"]):
        rel_path, target_col = HORIZONS[h]
        df = pd.read_csv(DATA_DIR / rel_path)
        df.columns = df.columns.str.strip().str.replace("﻿", "")
        df = df.dropna(subset=[WEEK1_COL, target_col])
        x = df[WEEK1_COL].values
        y = df[target_col].values

        for col, scale in enumerate(["linear", "log"]):
            ax = axes[row, col]
            ax.scatter(x, y, alpha=0.3, s=8, color="steelblue")
            if scale == "log":
                mask = (x > 0) & (y > 0)
                if mask.sum() > 1:
                    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                    xline = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 100)
                    yline = np.exp(intercept) * xline ** slope
                    label = f"log-fit: slope = {slope:.2f}"
                    ax.plot(xline, yline, "r-", linewidth=1.5, label=label)
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                if len(x) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    xline = np.linspace(x.min(), x.max(), 100)
                    yline = slope * xline + intercept
                    label = f"linear-fit: y = {slope:.2f}x + {intercept:.0f}"
                    ax.plot(xline, yline, "r-", linewidth=1.5, label=label)
            ax.set_xlabel(WEEK1_COL)
            ax.set_ylabel(target_col)
            ax.set_title(f"{h} horizon ({scale} scale)")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "05_week1_vs_target.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 6: Exploration combo (Task 8) ===

def figure_6_exploration() -> None:
    """Two-panel combo. Left: early_positive_ratio vs 3-month CCU scatter.
    Right: 6×6 correlation heatmap.
    """
    rel_path, target_col = HORIZONS["3m"]
    df = pd.read_csv(DATA_DIR / rel_path)
    df.columns = df.columns.str.strip().str.replace("﻿", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: early_positive_ratio scatter ---
    sub = df.dropna(subset=["early_positive_ratio", target_col])
    sub = sub[(sub["early_positive_ratio"] >= 0) & (sub["early_positive_ratio"] <= 1)]
    sub = sub[sub[target_col] > 0]
    ax = axes[0]
    ax.scatter(sub["early_positive_ratio"], sub[target_col], alpha=0.3, s=8, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("early_positive_ratio")
    ax.set_ylabel(f"{target_col} (log scale)")
    ax.set_title("Early positive review ratio vs 3-month CCU")
    ax.grid(True, which="both", alpha=0.3)

    # --- Right panel: 6×6 correlation heatmap ---
    cols = [
        WEEK1_COL,
        "early_reviews_total",
        "early_positive_ratio",
        "metacritic_score",
        "price_final_usd_cents",
        target_col,
    ]
    corr_df = df[cols].dropna()
    # filter sentinel metacritic_score == 0 (76% of rows per audit) for honest correlation
    corr_df = corr_df[corr_df["metacritic_score"] > 0]
    corr = corr_df.corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson r"}, ax=axes[1],
        xticklabels=[c[:24] for c in corr.columns],
        yticklabels=[c[:24] for c in corr.columns],
    )
    axes[1].set_title("Feature correlation heatmap (3-month)")

    plt.tight_layout()
    out = FIGURES_DIR / "06_exploration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === MAIN / CLI ===

FIGURES = {
    1: figure_1_feature_importance,
    2: figure_2_model_comparison,
    3: figure_3_pred_vs_actual,
    4: figure_4_residuals,
    5: figure_5_week1_vs_target,
    6: figure_6_exploration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig", type=int, choices=range(1, 7),
                        help="Generate just one figure (1-6). Omit to generate all.")
    args = parser.parse_args()

    targets = [args.fig] if args.fig else list(FIGURES.keys())
    for n in targets:
        print(f"=== Figure {n}: {FIGURES[n].__name__} ===")
        try:
            FIGURES[n]()  # full pipeline plumbing wired in Tasks 2-8
        except NotImplementedError:
            print(f"  (skipped — figure {n} not implemented yet)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
