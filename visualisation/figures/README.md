# figures/

Output PNGs from `../audit_plots.py`. Regenerate by running the script
from the parent folder.

**Last regenerated:** 2026-05-13T13:26Z
**Repo commit at regeneration:** 0d562dbf (Kevin's `Add 100-player cutoff on all models`)
**Source dataset:** `../models-6-5/{three,six,twelve}_months_model/{horizon}_final.csv`
(Kevin's v2; 3,988 rows → 2,900 rows after the >=100 first-week-players filter that
Kevin applies in all three horizon scripts). The auditor pipeline in `../audit_plots.py`
now applies the same filter so the figures reflect Kevin's modeling subset.

## Inventory

| File | What it shows |
|---|---|
| `01_feature_importance.png` | Top-10 Gini importances from the 3-month Gradient Boosting Regressor. |
| `02_model_comparison.png` | 2×2 grid. Rows: with/without `players_7days_after_release` feature. Cols: R² / MAE. Bars = LR / RF / GBR per horizon. |
| `03_pred_vs_actual.png` | 1×3 grid. Predicted vs actual CCU per horizon, log-log, with `y = x` reference. |
| `04_residuals.png` | 1×3 grid. Residuals (observed − predicted, log-CCU) vs predicted, per horizon. |
| `05_week1_vs_target.png` | 3×2 grid. `players_7days_after_release` vs target per horizon × linear/log scale, with regression trendline. |
| `06_exploration.png` | Two-panel. Left: `early_positive_ratio` vs 3-month CCU scatter (log y). Right: 6×6 correlation heatmap. |

(The Marathon (2026) case-study figure was previously `07_marathon_case_study.png`; it has moved to `../../marathon/figures/07_marathon_case_study.png` and is now produced by `../../marathon/predict_marathon.py`.)

All figures use `random_state=42` and a single 85/15 random split (matches
Kevin's setup). All 200 DPI.
