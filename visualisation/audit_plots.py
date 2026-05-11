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

# --- Marathon (2026) case study --------------------------------------------------
# Inputs sourced from Wikipedia / Steam store / SteamCharts on 2026-05-11. Marathon
# released 2026-03-05 (Steam App 3065800, Bungie). Used by figure 7. The exact
# `players_7days_after_release` value is not published; figure 7 runs three
# scenarios (March monthly avg / geometric mean / launch-day peak).
MARATHON_FEATURES = {
    "required_age": 17,
    "is_free": 0,
    "coming_soon": 0,
    "price_initial_usd_cents": 3999,
    "price_final_usd_cents": 3999,
    "metacritic_score": 82,
    "achievements_total": 14,
    "dlc_count": 1,
    "supports_windows": 1,
    "supports_mac": 0,
    "supports_linux": 0,
    "early_reviews_window_available": 1,
    "early_reviews_total": 37422,
    "early_reviews_positive": 32557,
    "early_reviews_negative": 4865,
    "early_review_score": 87,
    "early_positive_ratio": 0.87,
    "genre__action": 1,
    "category__multi_player": 1,
    "category__online_pvp": 1,
    "category__pvp": 1,
    "category__cross_platform_multiplayer": 1,
    "category__family_sharing": 1,
    "tag__shooter": 1,
    "tag__fps": 1,
    "tag__pvp": 1,
    "tag__multiplayer": 1,
    "tag__sci_fi": 1,
    "tag__action": 1,
    "tag__first_person": 1,
    "tag__pve": 1,
    "tag__cyberpunk": 1,
    "tag__survival": 1,
    "tag__online_co_op": 1,
    "tag__class_based": 1,
    "tag__inventory_management": 1,
    "tag__character_customization": 1,
    "tag__futuristic": 1,
    "tag__hero_shooter": 1,
}

# Three week-1 input scenarios (raw CCU; will be log1p-transformed at predict-time)
MARATHON_WEEK1_SCENARIOS = [
    ("Low (March 2026 avg)", 35040),
    ("Mid (geom mean avg/peak)", 52063),
    ("High (launch-day peak)", 77358),
]

# Observed Steam trajectory (months-since-launch, average CCU, label)
MARATHON_OBSERVED = [
    (0, 35040, "Mar 2026"),
    (1, 15833, "Apr 2026"),
    (2, 3869, "May 2026"),
]

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


# === FIGURE 0: Top-10 example predictions (test + train) ===

def figure_0_top10_predictions() -> None:
    """Two panels: top-10 highest-actual-CCU titles in TRAIN (left) and TEST (right) sets,
    each panel showing predicted CCU vs actual CCU side-by-side. Uses the 3-month
    Gradient Boosting Regressor with the players_7days_after_release feature.

    Numbers are raw CCU (back-transformed via np.expm1 from log-CCU) so the values are
    directly meaningful to a managerial reader. Log x-axis because CCU spans several
    orders of magnitude.
    """
    rel_path, target_col = HORIZONS["3m"]
    df = pd.read_csv(DATA_DIR / rel_path)
    df.columns = df.columns.str.strip().str.replace("﻿", "")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    names = df["name"].astype(str)

    # Re-derive X, y exactly as load_horizon does (RangeIndex, same row order)
    X, y, _ = load_horizon("3m")

    indices = np.arange(len(X))
    tr_idx, te_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    res = get_results()[("3m", True, "GBR")]
    model = res["model"]

    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    y_tr_pred = model.predict(X_tr)

    X_te = X.iloc[te_idx]
    y_te = y.iloc[te_idx]
    y_te_pred = model.predict(X_te)

    def top10_frame(idx, y_true_log, y_pred_log):
        return (
            pd.DataFrame({
                "name": names.iloc[idx].values,
                "actual": np.expm1(y_true_log.values),
                "predicted": np.expm1(y_pred_log),
            })
            .nlargest(10, "actual")
            .iloc[::-1]  # reverse so largest sits at top of barh
            .reset_index(drop=True)
        )

    tr_top = top10_frame(tr_idx, y_tr, y_tr_pred)
    te_top = top10_frame(te_idx, y_te, y_te_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    bar_h = 0.4
    for ax, label, top in zip(axes, ["train set", "test set"], [tr_top, te_top]):
        y_pos = np.arange(len(top))
        ax.barh(y_pos - bar_h / 2, top["actual"], bar_h, label="actual", color="steelblue")
        ax.barh(y_pos + bar_h / 2, top["predicted"], bar_h, label="predicted", color="orange")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n[:32] for n in top["name"]], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Average monthly CCU at month 3")
        ax.set_title(f"Top-10 by actual CCU — {label}")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="x", which="both", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "00_top10_predictions.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 1: Feature importance per horizon (3m / 6m / 12m) ===

def figure_1_feature_importance() -> None:
    """1×3 grid. One panel per horizon (3m / 6m / 12m). Each panel shows the top-10
    values from the GBR's `feature_importances_` attribute, with the numeric value
    annotated on each bar. Shared x-axis across panels so the importance distributions
    are visually comparable across horizons.
    """
    horizons = ["3m", "6m", "12m"]
    # collect top-10 per horizon and the global max for the shared x-axis
    panels = []
    global_max = 0.0
    for h in horizons:
        res = get_results()[(h, True, "GBR")]
        importances = pd.Series(res["model"].feature_importances_, index=res["feature_names"])
        top10 = importances.sort_values(ascending=True).tail(10)
        panels.append((h, top10))
        global_max = max(global_max, float(top10.values.max()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, (h, top10) in zip(axes, panels):
        bars = ax.barh(top10.index, top10.values, color="steelblue")
        for bar, val in zip(bars, top10.values):
            ax.text(
                bar.get_width() + global_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8,
            )
        ax.set_xlim(0, global_max * 1.18)
        ax.set_xlabel("Feature importance value")
        ax.set_title(f"{h} horizon — top-10 features (GBR)")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Feature importance values from `model.feature_importances_` (sum to 1.0 per panel)", y=1.02, fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / "01_feature_importance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # also print the top-10 per horizon to stdout for the user's reference
    for h, top10 in panels:
        print(f"\n  Top-10 values ({h}):")
        for name, val in top10.iloc[::-1].items():
            print(f"    {val:.4f}  {name}")


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
    short_labels = {
        "players_7days_after_release": "week-1 players",
        "early_reviews_total": "early reviews",
        "early_positive_ratio": "positive ratio",
        "metacritic_score": "metacritic",
        "price_final_usd_cents": "price (¢)",
        "players_month_3_after_release": "3-mo players",
    }
    labels = [short_labels.get(c, c) for c in corr.columns]
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson r"}, ax=axes[1],
        xticklabels=labels,
        yticklabels=labels,
    )
    axes[1].set_title("Feature correlation heatmap (3-month)")

    plt.tight_layout()
    out = FIGURES_DIR / "06_exploration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


# === FIGURE 7: Marathon (2026) case study ===

def _predict_marathon_horizon(horizon_key: str, week1_ccu: float) -> float:
    """Train a GBR on Kevin's data for the given horizon, then predict Marathon's
    average monthly CCU at that horizon for a given week-1 input. Returns raw CCU."""
    X, y, feat_names = load_horizon(horizon_key)
    X_tr, _X_te, y_tr, _y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    gbr = GradientBoostingRegressor(**GBR_KWARGS).fit(X_tr, y_tr)

    # Build Marathon's feature row in the trained column order. Unset features = 0
    # (correct for one-hot categoricals; matches Kevin's CSV convention where a
    # missing tag/category is 0). Numeric features Marathon has are set explicitly.
    row = pd.Series(0.0, index=feat_names)
    payload = {**MARATHON_FEATURES, "players_7days_after_release": week1_ccu}
    for k, v in payload.items():
        if k in row.index:
            row[k] = v
    # Apply the same log1p transform load_horizon already applied to X_tr
    for c in row.index:
        if "review" in c.lower() or "player" in c.lower():
            row[c] = np.log1p(row[c])

    pred_log = gbr.predict(pd.DataFrame([row]))[0]
    return float(np.expm1(pred_log))


def figure_7_marathon() -> None:
    """1×2 panel for Marathon (2026) case study.

    Left: predicted average monthly CCU at the 3/6/12-month horizons under three
    week-1 input scenarios (Low / Mid / High).
    Right: observed Steam trajectory (Mar-May 2026) overlaid with the three
    predicted-trajectory bands and a naive -55%-per-month decay baseline.
    """
    horizons = ["3m", "6m", "12m"]
    scenario_labels = [s[0] for s in MARATHON_WEEK1_SCENARIOS]

    print("  Training GBR per horizon and predicting Marathon CCU ...")
    predictions: dict[tuple[str, str], float] = {}
    for h in horizons:
        for label, week1 in MARATHON_WEEK1_SCENARIOS:
            predictions[(h, label)] = _predict_marathon_horizon(h, float(week1))
            print(f"    {h:>3} | {label:<28} (week1={week1:>6,}) → {predictions[(h, label)]:>9,.0f} CCU")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#9ec5e8", "#4d8fd1", "#1f4e79"]  # graded blue: Low / Mid / High

    # --- LEFT: predicted CCU bars per horizon, three scenarios -----------------
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

    # --- RIGHT: observed trajectory + predicted bands + naive baseline ---------
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

    # Naive -55%/mo decay anchored on April 2026 (the last full month of decay)
    naive_x = [1, 2, 3, 6, 12]
    base_april = 15833
    decay = 0.45  # i.e. -55% MoM
    naive_y = [base_april * (decay ** (m - 1)) for m in naive_x]
    ax.plot(naive_x, naive_y, ":", color="#888", linewidth=1.5,
            marker="^", markersize=6, label="Naive -55%/mo decay")

    ax.set_xticks([0, 1, 2, 3, 6, 12])
    ax.set_xticklabels(["0\nMar‧26", "1\nApr", "2\nMay", "3\nJun",
                         "6\nSep", "12\nMar‧27"])
    ax.set_xlabel("Months since launch (2026-03-05)")
    ax.set_ylabel("Average monthly CCU")
    ax.set_title("CCU trajectory: observed Mar–May 2026 vs predicted Jun-2026–Mar-2027")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Case study: Marathon (2026) — Bungie · Steam App 3065800 · released 2026-03-05",
        fontsize=12, y=1.02, fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "07_marathon_case_study.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# === MAIN / CLI ===

FIGURES = {
    0: figure_0_top10_predictions,
    1: figure_1_feature_importance,
    2: figure_2_model_comparison,
    3: figure_3_pred_vs_actual,
    4: figure_4_residuals,
    5: figure_5_week1_vs_target,
    6: figure_6_exploration,
    7: figure_7_marathon,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig", type=int, choices=range(0, 8),
                        help="Generate just one figure (0-7). Omit to generate all.")
    args = parser.parse_args()

    targets = [args.fig] if args.fig is not None else list(FIGURES.keys())
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
