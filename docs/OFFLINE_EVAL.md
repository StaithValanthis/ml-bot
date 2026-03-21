# Offline Model Filter Evaluation

## Overview

The offline evaluator performs **purged cross-validation** and **threshold analysis** on historical labeled candidate data to assess whether the model filter is ready to promote from SHADOW mode to ACTIVE gating.

**No runtime behavior changes** — this is evaluation-only tooling.

## What It Does

1. **Purged Walk-Forward CV** — Time-aware splits with embargo (gap between train and validation) and purging (exclude train samples near validation) to prevent leakage. No random k-fold.

2. **Threshold Analysis** — For each threshold (e.g. 0.3–0.7), computes:
   - Precision, recall, F1, support
   - Retained count, filtered count, retain ratio
   - Win-rate / positive-rate before and after gating (when labels support it)

3. **Shadow vs Baseline** — Compares baseline (all candidates) to model-gated retention:
   - Total candidates, retained, filtered, retain ratio
   - Positive-label rate before/after
   - False negative risk, false positive reduction
   - Expected uplift

4. **Artifacts** — Writes JSON summary, CSV threshold table, per-fold metrics CSV, optional predictions CSV, and a markdown report.

## How to Run

```bash
# Using the installed script (after pip install -e .)
trading-eval --model data/archive/offline_train/model_20250101_120000.pkl

# Or with explicit paths
trading-eval --dataset data/archive/decision_exports/decisions_20250101_120000.json \
  --model data/archive/offline_train/model_20250101_120000.pkl \
  --output-dir data/archive/eval

# With custom threshold grid and CV settings
trading-eval --model model.pkl \
  --thresholds "0.25,0.35,0.45,0.55,0.65,0.75" \
  --n-splits 10 \
  --embargo 600 \
  --purge 300 \
  --rolling
```

### Inputs

| Argument   | Default                          | Description                          |
|-----------|-----------------------------------|--------------------------------------|
| `--dataset` | Latest in `decision_exports/`   | Path to decision export JSON         |
| `--model`   | Required                        | Path to model artifact (.pkl)       |
| `--output-dir` | `TRADING_ARCHIVE_DIR/eval`   | Output directory                     |
| `--thresholds` | `0.3,0.4,0.5,0.6,0.7`       | Comma-separated threshold grid       |
| `--n-splits` | 5                             | Number of purged CV folds            |
| `--embargo` | 300                           | Embargo seconds between train/val    |
| `--purge`   | 300                           | Purge seconds before val window      |
| `--min-train` | 10                          | Minimum train samples per fold       |
| `--min-val`  | 5                            | Minimum validation samples per fold  |
| `--rolling`  | (expanding by default)       | Use rolling instead of expanding     |

## Output Structure

```
data/archive/eval/
├── eval_summary_20250101_120000.json   # Full JSON summary
├── threshold_table_20250101_120000.csv # Threshold vs metrics
├── per_fold_metrics_20250101_120000.csv
├── predictions_20250101_120000.csv     # Per-row (if ≤10k rows)
└── eval_report_20250101_120000.md     # Human-readable report
```

## Interpreting Outputs

### Threshold Table

- **retained_count** — How many candidates would pass the filter at this threshold.
- **filtered_count** — How many would be blocked.
- **retain_ratio** — Fraction of candidates retained.
- **precision** — Of retained, how many are true positives (filled).
- **recall** — Of all filled, how many are retained.
- **f1** — Harmonic mean of precision and recall.
- **win_rate_retained** — When profitable_fill is available, fraction of retained fills that were profitable.

### Shadow vs Baseline

- **false_negative_risk** — Fraction of good trades (filled) that would be filtered out.
- **false_positive_reduction** — How much the filter reduces false positives (non-fills) in the retained set.
- **uplift_positive_rate** — Improvement in positive-label rate when gating vs baseline.

### Promotion Readiness

Use the report to answer:

1. **At threshold X, how many trades would be filtered?** — See `filtered_count` / `retain_ratio`.
2. **Would retained trades be better?** — See `positive_rate_retained`, `win_rate_retained`, `uplift_positive_rate`.
3. **What is the risk of filtering good trades?** — See `false_negative_count`, `false_negative_risk`.
4. **Is the model consistently useful across folds?** — Check per-fold metrics in the CSV.

**Promotion recommendation:** If precision/recall/F1 are stable across folds, uplift is positive, and FN risk is acceptable for your risk appetite, the model may be ready for ACTIVE gating. Start with a conservative threshold and monitor live shadow logs before enabling hard blocking.

## Relationship to SHADOW Mode

- **SHADOW** — Model is evaluated at runtime; decisions are logged and reported but not used to block.
- **Offline eval** — Uses historical data to estimate what would have happened if ACTIVE gating were used.
- **Promotion path** — Run offline eval → verify metrics and FN risk → enable ACTIVE at chosen threshold → monitor.
