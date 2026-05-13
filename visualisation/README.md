# visualisation/

Audit-layer visualisation deliverable for the ETH FS26 Business Analytics
group project (Team 7 — Steam CCU prediction for Tencent).

**Owner:** Nick Hulsbergen (model auditor)

**Read-only contract:** This folder does NOT modify `models-6-5/`. The
auditor-independence rule (model-builder and model-auditor are different
people) is preserved by keeping all auditor code in this separate folder.

## What's here
- `audit_plots.py` — single-file script that consumes Kevin's 3-month
  CSV and produces seven PNG figures for the auditor + the report
  (figures 1, 2, 3, 4, 5, 6 are referenced in §5, §6 and the Appendix;
  figure 0 is auditor-only).
- `requirements.txt` — pinned Python dependencies.
- `figures/` — committed output PNGs (200 DPI).

## How to run

```bash
pip install -r requirements.txt
python audit_plots.py            # regenerate all seven figures
python audit_plots.py --fig 3    # regenerate just figure 3
```

The script reads `../models-6-5/three_months_model/three_month_final.csv`.

## Single 3-month horizon

The audit pipeline reports a single 3-month prediction horizon. The
6- and 12-month horizons were dropped because (a) the input
`players_7days_after_release` is the exact CCU snapshot on the
seven-days-after-release date — one input observation maps most
defensibly to one target horizon, and (b) the 3-month horizon is
the most defensible window: shortest, highest R² historically, and
closest to what the Marathon case study (`../marathon/`) can actually
validate against observed Steam data.

## Figure inventory (see `figures/README.md` for details)
0. Top-10 predictions (auditor-only: train vs test, actual vs predicted)
1. Feature importance — top-10 GBR features at 3-month horizon
2. Model comparison — LR / RF / GBR × feature ablation × R² / MAE
3. Predicted vs actual CCU at 3 months (log-log)
4. Residuals vs predicted at 3 months
5. Day-7 CCU vs 3-month target with linear fit (linear / log scale)
6. Exploration: early-positive-ratio scatter + correlation heatmap

## Reproducibility
All models use `random_state=42` and a single 85/15 random split (matches
Kevin's setup). Re-running on the same input CSV reproduces the same PNGs.
