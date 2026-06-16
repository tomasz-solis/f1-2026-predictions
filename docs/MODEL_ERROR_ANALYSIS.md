# Model Error Analysis - 2026

*Generated: 2026-06-16T19:46:42.011345+00:00*

This companion note focuses on the failures the model needs to explain,
not the averages it would prefer to show.

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

## Context

- Pair this with [MODEL_CALIBRATION.md](./MODEL_CALIBRATION.md) for calibration and baseline metrics.
- Pair this with [LIMITATIONS.md](../LIMITATIONS.md) for known structural gaps.
