# visualisation/

Audit-layer visualisation deliverable for the ETH FS26 Business Analytics
group project (Team 7 — Steam CCU prediction for Tencent).

**Owner:** Nick Hulsbergen (model auditor)

**Read-only contract:** This folder does NOT modify `models-6-5/`. The
auditor-independence rule (model-builder and model-auditor are different
people) is preserved by keeping all auditor code in this separate folder.

## What's here
- `audit_plots.py` — single-file script that consumes Kevin's v2 CSVs
  and produces the figures used in Section 6 and the Appendix of the
  final report.
- `requirements.txt` — pinned Python dependencies.
- `figures/` — committed output PNGs (200 DPI).

## How to run

```bash
pip install -r requirements.txt
python audit_plots.py            # regenerate all figures
python audit_plots.py --fig 3    # regenerate just figure 3
python audit_plots.py --fig 8    # regenerate just the 6-month top-10
python audit_plots.py --fig 9    # regenerate just the 12-month top-10
```

The script reads `../models-6-5/three_month_final.csv`, `six_month_final.csv`,
and `twelve_month_final.csv` from the sibling folder.

## Figure inventory (see `figures/README.md` for details)
0. Top-10 train vs test predictions at the 3-month horizon (§6.3)
1. Feature importance (GBR top-10, 3-month)
2. Model comparison (LR / RF / GBR × feature ablation × R² / MAE)
3. Predicted vs actual CCU (per horizon, log-log)
4. Residuals vs predicted (per horizon)
5. Week-1 vs target with regression trendline (per horizon × linear/log)
6. Exploration: early-positive-ratio scatter + correlation heatmap
8. Top-10 train vs test predictions at the 6-month horizon (Appendix)
9. Top-10 train vs test predictions at the 12-month horizon (Appendix)

## Reproducibility
All models use `random_state=42` and a single 85/15 random split (matches
Kevin's setup). Re-running on the same input CSVs reproduces the same PNGs.
