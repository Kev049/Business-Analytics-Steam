# figures/

Output PNGs from `../audit_plots.py`. Regenerate by running the script
from the parent folder.

**Last regenerated:** 2026-05-13T13:55Z
**Repo commit at regeneration:** 7dad22d (post Marathon-folder split)
**Source dataset:** `../models-6-5/three_months_model/three_month_final.csv`
(Kevin's v2; 3,988 rows → 2,900 rows after the >=100 day-7 CCU cutoff that
Kevin applies in his three_months_model script). The auditor pipeline in
`../audit_plots.py` applies the same filter.

**Horizon scope:** the audit visualisations are reported at a single 3-month
horizon. The 6- and 12-month horizons were dropped from the auditor output
because (a) the input `players_7days_after_release` is the exact CCU snapshot
at the seven-days-after-release date — one input observation maps most
defensibly to one target horizon, and (b) the 3-month horizon is the most
defensible window: shortest, highest R² historically, and closest to what
the Marathon case study can actually validate against observed Steam data.

## Inventory

| File | What it shows |
|---|---|
| `00_top10_predictions.png` | Auditor-only sanity check: top-10 actual-CCU titles in train vs test sets, with predicted CCU alongside, at the 3-month horizon. |
| `01_feature_importance.png` | Top-10 GBR feature importances at the 3-month horizon. |
| `02_model_comparison.png` | 2×2 grid at 3-month horizon. Rows: with/without `players_7days_after_release` feature. Cols: R² / MAE. Bars = LR / RF / GBR. |
| `03_pred_vs_actual.png` | Predicted vs actual CCU at 3 months, log-log, with `y = x` reference. |
| `04_residuals.png` | Residuals (observed − predicted, log-CCU) vs predicted at the 3-month horizon. |
| `05_week1_vs_target.png` | 1×2 grid (linear / log). `players_7days_after_release` vs 3-month target, with regression trendline. |
| `06_exploration.png` | Two-panel. Left: `early_positive_ratio` vs 3-month CCU scatter (log y). Right: 6×6 correlation heatmap. |

(The Marathon (2026) case-study figure is `../../marathon/figures/07_marathon_case_study.png`, produced by `../../marathon/predict_marathon.py`.)

All figures use `random_state=42` and a single 85/15 random split. All 200 DPI.
