# Model Calibration Report - 2026 Season

*Generated: 2026-06-16T19:46:42.011345+00:00*

This report measures three things: whether the Monte Carlo uncertainty
bands are empirically honest, whether the model has systematic directional
bias for specific drivers or teams, and whether it beats a naive baseline.

It is generated from saved prediction artifacts by
`scripts/generate_evaluation_report.py`. Re-run after each race to keep
it current.

---

## Production Readiness Gate

- Status: **PASS**
- Score estimate: **95/100**

No blocking reasons.

| Metric | Value |
|---|---|
| Scored race weekends | 7 |
| Qualifying MAE improvement vs naive | 0.12 |
| Race MAE improvement vs naive | 0.52 |
| Qualifying interval coverage | 90.3% |

---

## Coverage

- Prediction source: **artifact_store**
- Event order: **season_calendar**
- Prediction artifacts analyzed: **27**
- Latest qualifying checkpoints selected: **7**
- Latest race checkpoints selected: **7**
- Qualifying races with actuals: **7**
- Race results with actuals: **7**
- Intermediate qualifying checkpoints ignored in canonical evaluation: **20**
- Intermediate race checkpoints ignored in canonical evaluation: **20**

---

## 0. Accuracy Overview

Production qualifying metrics include the documented time-aware rank stabilizer
and conformal interval widening described below. Raw qualifying diagnostics
are retained for audit comparison.

| Session | Events | MAE | Exact match | Within-3 | Spearman rho | Kendall tau |
|---|---|---|---|---|---|---|
| Qualifying | 7 | 2.86 | 14.29% | 70.78% | 0.81 | 0.66 |
| Race | 7 | 4.01 | 11.04% | 57.14% | 0.62 | 0.48 |

### Production Qualifying Stabilizer

- Rank method: previous-race rank blend using only completed prior-race actuals.
- Model rank weight: **0.40**
- Previous-race rank weight: **0.60**
- Interval method: conformal widening from past interval residuals only.
- Default margin before history is sufficient: **1.00 position(s)**
- Minimum history events before learned margins: **3**
- Raw qualifying MAE: **3.21**
- Production qualifying MAE: **2.86**
- Raw qualifying MAE improvement vs naive: **-0.29**
- Production qualifying MAE improvement vs naive: **0.12**
- Raw qualifying interval coverage: **77.3%**
- Production qualifying interval coverage: **90.3%**

| Race | Target | Margin | Prior events available |
|---|---|---|---|
| Australian Grand Prix | main_qualifying | 1.00 | 0 |
| Chinese Grand Prix | sprint_qualifying | 1.00 | 1 |
| Japanese Grand Prix | main_qualifying | 1.00 | 2 |
| Miami Grand Prix | sprint_qualifying | 1.00 | 3 |
| Canadian Grand Prix | sprint_qualifying | 1.00 | 4 |
| Monaco Grand Prix | main_qualifying | 2.00 | 5 |
| Barcelona Grand Prix | main_qualifying | 2.00 | 6 |

### Weekend Format Coverage

| Format | Prediction artifacts | Qualifying pairs | Race pairs |
|---|---|---|---|
| normal | 17 | 14 | 14 |
| sprint | 10 | 6 | 9 |

---

## Selection Policy

Canonical evaluation uses `latest_checkpoint_per_race_and_target` so each race/target contributes at most one scored forecast.

### Selected Checkpoints

| Session | Checkpoint | Count |
|---|---|---|
| qualifying | FP1 | 3 |
| qualifying | FP2 | 1 |
| qualifying | FP3 | 3 |
| race | FP2 | 1 |
| race | FP3 | 3 |
| race | SQ | 3 |

### Selected Targets

| Session | Target | Count |
|---|---|---|
| qualifying | main_qualifying | 4 |
| qualifying | sprint_qualifying | 3 |
| race | grand_prix_race | 4 |
| race | sprint_race | 3 |

---

## 1. Segmented Performance

### Qualifying

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| normal | 4 | 2.84 | 12.50% | 72.73% | 0.81 |
| sprint | 3 | 2.88 | 16.67% | 68.18% | 0.82 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 7 | 2.86 | 14.29% | 70.78% | 0.81 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 3 | 2.82 | 12.12% | 72.73% | 0.83 |
| street | 3 | 2.85 | 19.70% | 71.21% | 0.79 |
| unknown | 1 | 3.00 | 4.55% | 63.64% | 0.84 |

### Race

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| normal | 4 | 4.73 | 7.95% | 50.00% | 0.50 |
| sprint | 3 | 3.06 | 15.15% | 66.67% | 0.78 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 7 | 4.01 | 11.04% | 57.14% | 0.62 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 3 | 3.03 | 13.64% | 69.70% | 0.78 |
| street | 3 | 4.76 | 6.06% | 51.52% | 0.47 |
| unknown | 1 | 4.73 | 18.18% | 36.36% | 0.56 |

---

## 2. Confidence Interval Calibration (Qualifying)

The Monte Carlo simulation produces a p5 - p95 position interval for each
driver. A well-calibrated model should have ~90% of actual outcomes fall
inside that interval.

| Metric | Value |
|---|---|
| Races with interval data | 7 |
| Driver-race predictions covered | 154 |
| Nominal coverage (target) | 90.0% |
| Empirical coverage (actual) | 90.3% |
| Calibration error | 0.00 (+0.3%) |
| Mean interval width | 9.63 positions |

OK Well-calibrated (within 3% of nominal).

**Interpretation:** A negative calibration error means intervals are
too tight - the model is more certain than it should be. A positive
error means intervals are too wide.

---

## 3. Error Analysis

### Qualifying

Evaluated **7** event(s).

Worst weekends:
- Canadian Grand Prix (`permanent`, `sprint`, `dry`) MAE=3.55, winner=NOR -> RUS
- Monaco Grand Prix (`street`, `normal`, `dry`) MAE=3.36, winner=ANT -> ANT
- Barcelona Grand Prix (`unknown`, `normal`, `dry`) MAE=3.00, winner=ANT -> RUS

Drivers that show up repeatedly among the largest misses:
- VER: appearances=3, avg_abs_error=10.00
- LAW: appearances=3, avg_abs_error=9.00
- LIN: appearances=3, avg_abs_error=6.33

### Race

Evaluated **7** event(s).

Worst weekends:
- Monaco Grand Prix (`street`, `normal`, `dry`) MAE=6.91, winner=VER -> ANT
- Barcelona Grand Prix (`unknown`, `normal`, `dry`) MAE=4.73, winner=VER -> HAM
- Australian Grand Prix (`street`, `normal`, `dry`) MAE=4.18, winner=LEC -> RUS

Drivers that show up repeatedly among the largest misses:
- HAD: appearances=3, avg_abs_error=11.33
- HUL: appearances=3, avg_abs_error=9.33
- LAW: appearances=2, avg_abs_error=13.50

---

## 4. Systematic Bias

Signed error = predicted position - actual position.
Negative = model predicted *better* than reality (overestimated the driver).
Positive = model predicted *worse* than reality (underestimated the driver).

### Qualifying

Based on 7 races.

**Most overestimated teams:**
- Williams: mean signed error -1.50 (MAE 4.50, n=14)
- Ferrari: mean signed error -1.29 (MAE 2.14, n=14)
- Red Bull Racing: mean signed error -1.21 (MAE 3.64, n=14)

**Most underestimated teams:**
- Mercedes: mean signed error 1.14 (MAE 2.29, n=14)
- McLaren: mean signed error 1.07 (MAE 2.07, n=14)
- Alpine: mean signed error 1.07 (MAE 3.21, n=14)

**Most overestimated drivers:**
- LEC: mean signed error -2.43 (MAE 2.71, n=7)
- VER: mean signed error -2.29 (MAE 5.71, n=7)
- ALB: mean signed error -2.00 (MAE 4.86, n=7)

**Most underestimated drivers:**
- COL: mean signed error 2.43 (MAE 2.71, n=7)
- ANT: mean signed error 1.43 (MAE 2.00, n=7)
- PER: mean signed error 1.29 (MAE 1.86, n=7)

### Race

Based on 7 races.

**Most overestimated drivers:**
- HUL: mean signed error -5.71 (MAE 6.86, n=7)
- VER: mean signed error -5.43 (MAE 5.43, n=7)
- HAD: mean signed error -4.29 (MAE 5.71, n=7)

**Most underestimated drivers:**
- LAW: mean signed error 5.14 (MAE 6.86, n=7)
- OCO: mean signed error 3.71 (MAE 3.71, n=7)
- GAS: mean signed error 3.71 (MAE 4.29, n=7)

---

## 5. Baseline Comparison

Naive baseline: predict race N using the actual results of race N-1
(previous-race classification). This is a realistic lower bar - it
requires no modelling, just memory of last week.

### Qualifying

Based on 6 races.

| Metric | Model | Naive baseline | Delta |
|---|---|---|---|
| MAE | 2.86 | 2.98 | 0.12 |
| Within-3 rate | 68.2% | 62.9% | - |
| Spearman rho | 0.82 | 0.81 | 0.01 |
| Kendall tau | 0.66 | 0.63 | 0.02 |

OK Model beats naive baseline on MAE

### Race

Based on 6 races.

| Metric | Model | Naive baseline | Delta |
|---|---|---|---|
| MAE | 3.98 | 4.50 | 0.52 |
| Within-3 rate | 56.8% | 56.1% | - |
| Spearman rho | 0.62 | 0.51 | 0.11 |
| Kendall tau | 0.48 | 0.38 | 0.10 |

OK Model beats naive baseline on MAE

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
