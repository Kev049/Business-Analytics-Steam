"""
audit_plots.py — generate Section 6 figures for the Team 7 final report.

Reads Kevin's v2 CSVs from ../models-6-5/ and emits six PNGs into figures/.
Single-file by design (KISS). Sectioned with `# === FIGURE N: ... ===`.

Usage:
    python audit_plots.py            # generate all six figures
    python audit_plots.py --fig 3    # generate just figure 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def load_horizon(horizon: str) -> tuple[pd.DataFrame, pd.Series]:
    """Stub — implemented in Task 2."""
    raise NotImplementedError("Task 2")


# === MODEL TRAINING + METRICS (Task 2) ===

def train_and_score(*args, **kwargs):
    """Stub — implemented in Task 2."""
    raise NotImplementedError("Task 2")


# === FIGURE 1: Feature Importance (Task 3) ===

def figure_1_feature_importance(*args, **kwargs):
    raise NotImplementedError("Task 3")


# === FIGURE 2: Model Comparison (Task 4) ===

def figure_2_model_comparison(*args, **kwargs):
    raise NotImplementedError("Task 4")


# === FIGURE 3: Predicted vs Actual (Task 5) ===

def figure_3_pred_vs_actual(*args, **kwargs):
    raise NotImplementedError("Task 5")


# === FIGURE 4: Residuals (Task 6) ===

def figure_4_residuals(*args, **kwargs):
    raise NotImplementedError("Task 6")


# === FIGURE 5: Week-1 vs Target (Task 7) ===

def figure_5_week1_vs_target(*args, **kwargs):
    raise NotImplementedError("Task 7")


# === FIGURE 6: Exploration combo (Task 8) ===

def figure_6_exploration(*args, **kwargs):
    raise NotImplementedError("Task 8")


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
