# marathon/

Marathon (2026) case study — the standalone home for everything Marathon-specific
in the Team 7 final report (§6.5 of the LaTeX deliverable).

**Why a separate folder?** The case study trains the production-equivalent GBR
on Kevin's 3-month dataset under `models-6-5/` but feeds in a single external
feature row produced by the project's data pipeline. Keeping the Marathon
workflow isolated from `visualisation/audit_plots.py` makes it easy to re-run
on its own when the pipeline refreshes the Marathon feature CSV or when Kevin
reruns the training data.

## What's here

- `predict_marathon.py` — single-file script that loads Kevin's 3-month CSV,
  trains one GBR (matching audit-plots preprocessing: >=100 day-7 CCU cutoff,
  log1p on player/review columns, mean-impute, 85/15 random split with seed 42),
  then predicts Marathon's CCU at the 3-month horizon under three week-1
  input scenarios (Low / Empirical / High). Renders the report's Figure 7
  (1×2 panel) into `figures/`.
- `marathon_features.csv` — Marathon's day-7 CCU snapshot + first-week feature
  row from the project's data pipeline (2026-05-12 snapshot, semicolon-delimited,
  60 columns, one row).
- `requirements.txt` — pinned dependencies (same set as `../visualisation/`).
- `figures/07_marathon_case_study.png` — rendered figure (200 DPI), shipped
  to the Overleaf project as `Figures/07_marathon_case_study.png`.

## How to run

```bash
pip install -r requirements.txt
python predict_marathon.py
```

The script prints the Marathon feature snapshot, the three predicted-CCU
values at the 3-month horizon (one per scenario), and a paste-friendly
summary that maps onto the §6.5 prose in `Text/Machine Learning Prediction.tex`.

## Single 3-month horizon

The case study reports a single prediction horizon (3 months post-launch)
because (a) Marathon released 2026-03-05 and only ~2 months of observed
post-launch Steam data is available at submission time, so a 6- or 12-month
out-of-sample comparison would have no observed counterpart, and (b) the
input `players_7days_after_release` is the exact CCU snapshot at the
seven-days-after-release date, and a single input snapshot maps most
defensibly to a single target horizon. The 3-month horizon is the closest
window we can realistically check against observed reality and is also the
horizon at which the GBR has historically shown the strongest fit.

## Scenarios

| Scenario | Week-1 input | Source |
|---|---|---|
| Low (March 2026 avg) | 35,040 CCU | SteamCharts monthly average; biased downward by 27 days of post-peak decay |
| Empirical (pipeline) | 60,335 CCU | Day-7 CCU snapshot from the project's data pipeline (2026-05-12) |
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

The pipeline CSV has 60 columns; Kevin's 3-month training CSV has 558.
The script reindexes Marathon's feature row against the 3-month training-time
feature list — columns present in the pipeline row pass through; columns
absent default to 0 (the correct value for unset one-hot category/tag
features). All 17 Marathon-defining tags including `extraction_shooter`,
`gun_customization` and `hero_shooter` exist in Kevin's training schema,
so no information is discarded at predict time.
