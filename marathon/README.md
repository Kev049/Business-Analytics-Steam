# marathon/

Marathon (2026) case study — the standalone home for everything Marathon-specific
in the Team 7 final report (§6.5 of the LaTeX deliverable).

**Why a separate folder?** The case study trains the production-equivalent GBR
per horizon on Kevin's `models-6-5/` data but feeds in a single external feature
row produced by Dominik's pipeline. Keeping the Marathon workflow isolated from
`visualisation/audit_plots.py` makes it easy to re-run on its own when Dominik
refreshes the Marathon feature CSV or when Kevin reruns the training data.

## What's here

- `predict_marathon.py` — single-file script that loads Kevin's three horizon
  CSVs, trains one GBR per horizon (matching audit-plots preprocessing: 100-player
  cutoff, log1p on player/review columns, mean-impute, 85/15 random split with
  seed 42), then predicts Marathon's average monthly CCU at 3/6/12-month horizons
  under three week-1 scenarios (Low / Empirical / High). Renders the report's
  Figure 7 (1×2 panel) into `figures/`.
- `marathon_features.csv` — Marathon's first-7-day feature row from Dominik's
  pipeline (2026-05-12 snapshot, semicolon-delimited, 60 columns, one row).
- `requirements.txt` — pinned dependencies (same set as `../visualisation/`).
- `figures/07_marathon_case_study.png` — rendered figure (200 DPI), shipped
  to the Overleaf project as `Figures/07_marathon_case_study.png`.

## How to run

```bash
pip install -r requirements.txt
python predict_marathon.py
```

The script prints the Marathon feature snapshot, the 3-horizon × 3-scenario
prediction grid, and a paste-friendly summary line per horizon that maps onto
the report's prose in `Text/Machine Learning Prediction.tex`.

## Scenarios

| Scenario | Week-1 input | Source |
|---|---|---|
| Low (March 2026 avg) | 35,040 CCU | SteamCharts monthly average; biased downward by 27 days of post-peak decay |
| Empirical (Dominik) | 60,335 CCU | Dominik's pipeline output, first-7-day average |
| High (launch-day peak) | 77,358 CCU | SteamCharts launch-day peak (2026-03-08) |

The empirical scenario is the headline for the report. Low/High bracket sensitivity.

## Why Dominik's feature row, not the hardcoded dict

The previous version of the case study (`../visualisation/audit_plots.py`,
function `figure_7_marathon` before its removal) used a hardcoded
`MARATHON_FEATURES` dict that Nick populated from the Steam store page on
2026-05-11. Several values drifted from what Dominik's pipeline produces
(notably `metacritic_score`, `early_reviews_total`, `early_review_score`,
and `required_age`). Dominik's CSV is treated as the source of truth for
all non-target Marathon features.

## Schema reconciliation

Dominik's CSV has 60 columns; Kevin's training CSVs have 558. The script
reindexes Marathon's feature row against the horizon's training-time feature
list — columns present in Dominik's row pass through; columns absent default
to 0 (the correct value for unset one-hot category/tag features). All 17
Marathon-defining tags including `extraction_shooter`, `gun_customization`
and `hero_shooter` exist in Kevin's training schema, so no information is
discarded at predict time.
