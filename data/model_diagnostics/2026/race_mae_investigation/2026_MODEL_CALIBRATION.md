# Model Calibration Report - 2026 Season

*Generated: 2026-07-19T09:30:55.138732+00:00*

This report measures three things: whether the Monte Carlo uncertainty
bands are empirically honest, whether the model has systematic directional
bias for specific drivers or teams, and whether it beats a naive baseline.

It is generated from saved prediction artifacts by
`scripts/generate_evaluation_report.py`. Re-run after each race to keep
it current.

---

## Production Readiness Gate

- Status: **FAIL**
- Score estimate: **80/100**

Blocking reasons:
- only 1 scored race weekend(s); need at least 5
- qualifying MAE does not beat the previous-race naive baseline
- race MAE does not beat the previous-race naive baseline
- qualifying interval coverage 1.000 is outside [0.870, 0.930]

| Metric | Value |
|---|---|
| Scored race weekends | 1 |
| Qualifying MAE improvement vs naive | n/a |
| Race MAE improvement vs naive | n/a |
| Qualifying interval coverage | 100.0% |

---

## Coverage

- Prediction source: **artifact_store**
- Event order: **season_calendar**
- Prediction artifacts analyzed: **3**
- Latest qualifying checkpoints selected: **1**
- Latest race checkpoints selected: **1**
- Qualifying races with actuals: **1**
- Race results with actuals: **1**
- Intermediate qualifying checkpoints ignored in canonical evaluation: **2**
- Intermediate race checkpoints ignored in canonical evaluation: **2**

---

## 0. Accuracy Overview

Production qualifying metrics include the documented time-aware rank stabilizer
and conformal interval widening described below. Raw qualifying diagnostics
are retained for audit comparison.

| Session | Events | MAE | Weighted MAE | Top-heavy MAE | Exact match | Within-3 | Top-3 | Top-10 | Winner | Spearman rho | Kendall tau |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qualifying | 1 | 2.73 | 3.15 | 3.42 | 31.82% | 68.18% | 66.67% | 80.00% | 0.00% | 0.80 | 0.63 |
| Race | 1 | 3.73 | 3.14 | 3.70 | 4.55% | 63.64% | 33.33% | 70.00% | 0.00% | 0.75 | 0.57 |

### 0a. Top-Heavy Weight Basis Sensitivity (diagnostic only, not a promotion input)

`top_heavy_weighted_mae` is reported above using the same better-of predicted/actual position basis as `weighted_mae`. This table compares that default against the literal actual-position-only reading of the P1-3/P4-10/P11+ tiers, per scored event. This is a sensitivity check, not a candidate weighting scheme change, and is never promoted from one weekend's numbers alone.

**Qualifying**

| Event | min(predicted, actual) basis | actual-only basis | delta |
|---|---|---|---|
| Chinese Grand Prix | 3.42 | 3.09 | -0.34 |

**Race**

| Event | min(predicted, actual) basis | actual-only basis | delta |
|---|---|---|---|
| Chinese Grand Prix | 3.70 | 3.57 | -0.13 |

### Production Qualifying Stabilizer

- Rank method: previous-race rank blend using only completed prior-race actuals.
- Model rank weight: **0.40**
- Previous-race rank weight: **0.60**
- Interval method: conformal widening from past interval residuals only.
- Default margin before history is sufficient: **1.00 position(s)**
- Minimum history events before learned margins: **3**
- Raw qualifying MAE: **2.73**
- Production qualifying MAE: **2.73**
- Raw qualifying MAE improvement vs naive: **n/a**
- Production qualifying MAE improvement vs naive: **n/a**
- Raw qualifying interval coverage: **81.8%**
- Production qualifying interval coverage: **100.0%**

| Race | Target | Margin | Prior events available |
|---|---|---|---|
| Chinese Grand Prix | sprint_qualifying | 1.00 | 0 |

### Weekend Format Coverage

| Format | Prediction artifacts | Qualifying pairs | Race pairs |
|---|---|---|---|
| normal | 2 | 0 | 0 |
| sprint | 1 | 1 | 1 |

---

## Selection Policy

Canonical evaluation uses `latest_checkpoint_per_race_and_target` so each race/target contributes at most one scored forecast.

### Selected Checkpoints

| Session | Checkpoint | Count |
|---|---|---|
| qualifying | FP1 | 1 |
| race | FP1 | 1 |

### Selected Targets

| Session | Target | Count |
|---|---|---|
| qualifying | sprint_qualifying | 1 |
| race | sprint_race | 1 |

---

## 1. Segmented Performance

### Qualifying

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| sprint | 1 | 2.73 | 31.82% | 68.18% | 0.80 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 1 | 2.73 | 31.82% | 68.18% | 0.80 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 1 | 2.73 | 31.82% | 68.18% | 0.80 |

### Race

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| sprint | 1 | 3.73 | 4.55% | 63.64% | 0.75 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 1 | 3.73 | 4.55% | 63.64% | 0.75 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 1 | 3.73 | 4.55% | 63.64% | 0.75 |

---

## 2. Confidence Interval Calibration (Qualifying)

The Monte Carlo simulation produces a p5 - p95 position interval for each
driver. A well-calibrated model should have ~90% of actual outcomes fall
inside that interval.

| Metric | Value |
|---|---|
| Races with interval data | 1 |
| Driver-race predictions covered | 22 |
| Nominal coverage (target) | 90.0% |
| Empirical coverage (actual) | 100.0% |
| Calibration error | 0.10 (+10.0%) |
| Mean interval width | 10.55 positions |

Warning: Intervals are too **wide** - model is underconfident by 10.0pp.

**Interpretation:** A negative calibration error means intervals are
too tight - the model is more certain than it should be. A positive
error means intervals are too wide.

---

## 3. Error Analysis

### Qualifying

Evaluated **1** event(s).

Worst weekends:
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=2.73, winner=NOR -> RUS

Drivers that show up repeatedly among the largest misses:
- ALB: appearances=1, avg_abs_error=9.00
- BEA: appearances=1, avg_abs_error=9.00
- SAI: appearances=1, avg_abs_error=7.00

### Race

Evaluated **1** event(s).

Worst weekends:
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=3.73, winner=NOR -> RUS

Drivers that show up repeatedly among the largest misses:
- LIN: appearances=1, avg_abs_error=9.00
- HAD: appearances=1, avg_abs_error=8.00
- LAW: appearances=1, avg_abs_error=7.00

---

## 4. Systematic Bias

Signed error = predicted position - actual position.
Negative = model predicted *better* than reality (overestimated the driver).
Positive = model predicted *worse* than reality (underestimated the driver).

### Qualifying

*Not enough races to detect bias yet.*

### Race

*Not enough races to detect bias yet.*

---

## 5. Baseline Comparison

Naive baseline: predict race N using the actual results of race N-1
(previous-race classification). This is a realistic lower bar - it
requires no modelling, just memory of last week.

### Qualifying

*Not enough races for baseline comparison (need >= 2).*

### Race

*Not enough races for baseline comparison (need >= 2).*

---

## Notes

- Calibration data populates only for predictions generated after
  p5/p95 interval persistence was added. Older artifacts carry no band data.
- Segment breakdowns slice the same event-level metrics by weekend format, weather,
  and track type using saved metadata and local track characteristics.
- Error analysis highlights the worst weekends and repeat offenders, not just mean scores.
- Systematic bias analysis requires >= 2 races with saved actuals.
- Baseline comparison requires >= 2 races (first race has no predecessor).
- For known model limitations see [LIMITATIONS.md](../LIMITATIONS.md).
