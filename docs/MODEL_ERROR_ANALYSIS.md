# Model Error Analysis - 2026

*Generated: 2026-04-20T14:07:50.230271+00:00*

This companion note focuses on the failures the model needs to explain,
not the averages it would prefer to show.

### Qualifying

Evaluated **3** event(s).

Worst weekends:
- Japanese Grand Prix (`permanent`, `normal`, `dry`) MAE=3.27, winner=LEC -> ANT
- Australian Grand Prix (`street`, `normal`, `dry`) MAE=2.82, winner=LEC -> RUS
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=2.55, winner=RUS -> RUS

Drivers that show up repeatedly among the largest misses:
- LAW: appearances=2, avg_abs_error=8.00
- VER: appearances=1, avg_abs_error=18.00
- NOR: appearances=1, avg_abs_error=12.00

### Race

Evaluated **3** event(s).

Worst weekends:
- Australian Grand Prix (`street`, `normal`, `dry`) MAE=4.18, winner=LEC -> RUS
- Japanese Grand Prix (`permanent`, `normal`, `dry`) MAE=3.82, winner=LEC -> ANT
- Chinese Grand Prix (`permanent`, `sprint`, `dry`) MAE=3.18, winner=RUS -> RUS

Drivers that show up repeatedly among the largest misses:
- HAD: appearances=2, avg_abs_error=10.50
- PIA: appearances=2, avg_abs_error=10.50
- HUL: appearances=2, avg_abs_error=10.00

## Context

- Pair this with [MODEL_CALIBRATION.md](./MODEL_CALIBRATION.md) for calibration and baseline metrics.
- Pair this with [LIMITATIONS.md](../LIMITATIONS.md) for known structural gaps.
