# Model Calibration Report - 2026 Season

*Generated: 2026-04-20T14:07:50.230271+00:00*

This report measures three things: whether the Monte Carlo uncertainty
bands are empirically honest, whether the model has systematic directional
bias for specific drivers or teams, and whether it beats a naive baseline.

It is generated from saved prediction artifacts by
`scripts/generate_evaluation_report.py`. Re-run after each race to keep
it current.

---

## Coverage

- Prediction files analyzed: 14
- Latest qualifying checkpoints selected: 3
- Latest race checkpoints selected: 3
- Qualifying races with actuals: 3
- Race results with actuals: 3
- Intermediate qualifying checkpoints ignored in canonical evaluation: 11
- Intermediate race checkpoints ignored in canonical evaluation: 11

---

## 0. Accuracy Overview

| Session | Events | MAE | Exact match | Within-3 | Spearman ρ | Kendall τ |
|---|---|---|---|---|---|---|
| Qualifying | 3 | 2.88 | 21.21% | 74.24% | 0.76 | 0.64 |
| Race | 3 | 3.73 | 9.09% | 59.09% | 0.70 | 0.55 |

### Weekend Format Coverage

| Format | Prediction files | Qualifying pairs | Race pairs |
|---|---|---|---|
| normal | 9 | 8 | 8 |
| sprint | 5 | 2 | 3 |

---

## Selection Policy

Canonical evaluation uses `latest_checkpoint_per_race_and_target` so each race/target contributes at most one scored forecast.

### Selected Checkpoints

| Session | Checkpoint | Count |
|---|---|---|
| qualifying | FP1 | 1 |
| qualifying | FP3 | 2 |
| race | FP3 | 2 |
| race | SQ | 1 |

### Selected Targets

| Session | Target | Count |
|---|---|---|
| qualifying | main_qualifying | 2 |
| qualifying | sprint_qualifying | 1 |
| race | grand_prix_race | 2 |
| race | sprint_race | 1 |

---

## 1. Segmented Performance

### Qualifying

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| normal | 2 | 3.05 | 15.91% | 75.00% | 0.74 |
| sprint | 1 | 2.55 | 31.82% | 72.73% | 0.80 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| dry | 3 | 2.88 | 21.21% | 74.24% | 0.76 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| permanent | 2 | 2.91 | 27.27% | 68.18% | 0.77 |
| street | 1 | 2.82 | 9.09% | 86.36% | 0.74 |

### Race

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| normal | 2 | 4.00 | 6.82% | 56.82% | 0.66 |
| sprint | 1 | 3.18 | 13.64% | 63.64% | 0.78 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| dry | 3 | 3.73 | 9.09% | 59.09% | 0.70 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |
|---|---|---|---|---|---|
| permanent | 2 | 3.50 | 9.09% | 59.09% | 0.74 |
| street | 1 | 4.18 | 9.09% | 59.09% | 0.62 |

---

## 2. Confidence Interval Calibration (Qualifying)

The Monte Carlo simulation produces a p5 - p95 position interval for each
driver. A well-calibrated model should have ~90% of actual outcomes fall
inside that interval.

| Metric | Value |
|---|---|
| Races with interval data | 3 |
| Driver-race predictions covered | 66 |
| Nominal coverage (target) | 90.0% |
| Empirical coverage (actual) | 80.3% |
| Calibration error | -0.10 (-9.7%) |
| Mean interval width | 7.86 positions |

Warning: intervals are too tight - model is overconfident by 9.7pp.

Interpretation: A negative calibration error means intervals are
too tight - the model is more certain than it should be. A positive
error means intervals are too wide.

---

## 3. Error Analysis

### Qualifying

Evaluated 3 event(s).

Worst weekends:
- Japanese Grand Prix (`permanent`, `normal`, `dry`) MAE=3.27, winner=LEC -> ANT
- Australian Grand Prix (`street`, `normal`, `dry`) MAE=2.82, winner=LEC -> RUS
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=2.55, winner=RUS -> RUS

Drivers that show up repeatedly among the largest misses:
- LAW: appearances=2, avg_abs_error=8.00
- VER: appearances=1, avg_abs_error=18.00
- NOR: appearances=1, avg_abs_error=12.00

### Race

Evaluated 3 event(s).

Worst weekends:
- Australian Grand Prix (`street`, `normal`, `dry`) MAE=4.18, winner=LEC -> RUS
- Japanese Grand Prix (`permanent`, `normal`, `dry`) MAE=3.82, winner=LEC -> ANT
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=3.18, winner=RUS -> RUS

Drivers that show up repeatedly among the largest misses:
- HAD: appearances=2, avg_abs_error=10.50
- PIA: appearances=2, avg_abs_error=10.50
- HUL: appearances=2, avg_abs_error=10.00

---

## 4. Systematic Bias

Signed error = predicted position − actual position.
Negative = model predicted *better* than reality (overestimated the driver).
Positive = model predicted *worse* than reality (underestimated the driver).

### Qualifying

Based on 3 races.

Most overestimated teams:
- RB: mean signed error -2.83 (MAE 3.50, n=6)
- Ferrari: mean signed error -1.67 (MAE 2.67, n=6)
- Williams: mean signed error -1.33 (MAE 2.00, n=6)

Most underestimated teams:
- McLaren: mean signed error 4.67 (MAE 5.00, n=6)
- Mercedes: mean signed error 1.83 (MAE 1.83, n=6)
- Cadillac F1: mean signed error 1.00 (MAE 1.67, n=6)

Most overestimated drivers:
- LAW: mean signed error -5.33 (MAE 5.33, n=3)
- VER: mean signed error -4.67 (MAE 8.00, n=3)
- LEC: mean signed error -3.33 (MAE 3.33, n=3)

Most underestimated drivers:
- PIA: mean signed error 4.67 (MAE 4.67, n=3)
- NOR: mean signed error 4.67 (MAE 5.33, n=3)
- HAD: mean signed error 3.33 (MAE 4.67, n=3)

### Race

Based on 3 races.

Most overestimated drivers:
- HUL: mean signed error -8.33 (MAE 8.33, n=3)
- HAD: mean signed error -5.33 (MAE 8.67, n=3)
- LIN: mean signed error -3.00 (MAE 4.33, n=3)

Most underestimated drivers:
- GAS: mean signed error 5.67 (MAE 5.67, n=3)
- PER: mean signed error 4.33 (MAE 4.33, n=3)
- COL: mean signed error 3.33 (MAE 3.33, n=3)

---

## 5. Baseline Comparison

Naive baseline: predict race N using the actual results of race N-1
(previous-race classification). This is a realistic lower bar - it
requires no modelling, just memory of last week.

### Qualifying

Based on 2 races.

| Metric | Model | Naive baseline | Δ |
|---|---|---|---|
| MAE | 2.91 | 2.68 | -0.23 |
| Within-3 rate | 68.2% | 75.0% | - |
| Spearman ρ | 0.77 | 0.83 | -0.06 |
| Kendall τ | 0.62 | 0.66 | -0.04 |

Failed Model does not beat naive baseline on MAE

### Race

Based on 2 races.

| Metric | Model | Naive baseline | Δ |
|---|---|---|---|
| MAE | 3.50 | 3.32 | -0.18 |
| Within-3 rate | 59.1% | 70.5% | - |
| Spearman ρ | 0.74 | 0.71 | 0.03 |
| Kendall τ | 0.58 | 0.54 | 0.04 |

Failed Model does not beat naive baseline on MAE

---

## Notes

- Calibration data populates only for predictions generated after
  p5/p95 interval persistence was added. Older artifacts carry no band data.
- Segment breakdowns slice the same event-level metrics by weekend format, weather,
  and track type using saved metadata and local track characteristics.
- Error analysis highlights the worst weekends and repeat offenders, not just mean scores.
- Systematic bias analysis requires ≥ 2 races with saved actuals.
- Baseline comparison requires ≥ 2 races (first race has no predecessor).
- For known model limitations see [LIMITATIONS.md](../LIMITATIONS.md).
