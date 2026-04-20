# Trackside Labs

Production ML system for F1 race weekend predictions under data scarcity.

The 2026 regulation reset is the core constraint: every team's historical performance
baseline became unreliable at once. There is no prior season to learn from. The system
has to start from pre-season testing data — three days of limited running — and improve
its predictions across each weekend as new session data arrives.

That constraint shapes every design decision in here.

---

## The Engineering Problem

Most ML systems for sports prediction start with years of historical data and train a
model. That approach doesn't work here.

At Race 1, you have:
- three days of pre-season testing (limited laps, teams sandbagging)
- no race results at all

By Race 5, you have:
- testing data
- four race weekends of FP1/FP2/FP3/qualifying/race

The model needs to work in both situations and degrade gracefully in between.

The specific challenges this creates:

**Signal scarcity at season start.** Pre-season testing is intentionally misleading.
Teams run different fuel loads, different modes, different programs. The system extracts
directional signals (who improved relative to last year, who looks strong on high-stress
tracks) rather than treating raw lap times as ground truth.

**Progressive trust shifting.** As in-season data accumulates, the weight on testing
signals drops and the weight on actual race performance rises. By Race 3, the system
is running almost entirely on current-season evidence. The weight schedule is
configurable and runs in "extreme" mode for regulation-change seasons.

**Compound data scarcity.** Tire compound characteristics can only be learned
track-by-track. A team's SOFT performance at Monaco tells you nothing useful about
their SOFT performance at Bahrain. The system tracks compound metrics per team per
circuit and only applies compound adjustments when enough laps exist to make them
reliable (minimum 8 laps per compound per team).

**Rookie and missing-driver handling.** New drivers have no performance history.
The system falls back to team baseline with a calibrated rookie uncertainty
adjustment, rather than treating missing data as zero.

**Session-level freshness.** Predictions improve as each session completes. The
system tracks checkpoint state (PRE → FP1 → FP2 → FP3 → Q) and rebuilds predictions
when new session data becomes available, without user-facing latency spikes.

---

## How It Works

### Signal blending

Team strength is built from three signals blended by a race-number-aware weight schedule:

```
blended_strength = w_baseline * baseline
                 + w_testing * testing_modifier
                 + w_current * current_season_mean
```

In regulation-change mode (the 2026 season):

| Race | Baseline | Testing | Current |
|------|----------|---------|---------|
| 1    | 30%      | 20%     | 50%     |
| 2    | 15%      | 10%     | 75%     |
| 3+   | 5%       | 0%      | 95%     |

Before any races exist, `current` falls back to `baseline` rather than zero.

### Qualifying prediction

```
blended team strength (weight schedule)
  → session pace blend from available FP/sprint sessions
  → confidence-scaled blend weight (more data = more session trust)
  → combine team + driver skill
  → Monte Carlo simulations (300 runs)
  → median grid position + confidence band
```

Session blend priority:
- Normal weekend: FP3 + FP2 + FP1 (FP3-weighted)
- Sprint weekend: Sprint Qualifying + FP1 + Sprint

If no session data exists, qualifying runs model-only.

### Race prediction

```
qualifying grid (actual or predicted)
  → per-driver compound strengths from session history
  → Monte Carlo pit strategy generation
  → lap-by-lap simulation (300 runs)
    → tire degradation (compound-specific slopes)
    → fuel load effect
    → traffic-position correction (front: +5% tire life, back: -5%)
    → track-specific pit loss (Monaco: 19s, Singapore: 24s)
    → safety car, lap-1 chaos, DNF probability
  → finish order + strategy distribution + podium probability
```

### Adaptive calibration

After each race, the system updates per-driver EMA error state and teammate-gap
calibration using actual results. These learned adjustments feed into the next
prediction cycle automatically.

---

## System Design

The dashboard request path is intentionally read-only. All prediction artifact
generation happens in background workers:

- `scripts/warmup_precompute.py` — checkpoint-aware warmup, runs every 5 minutes in production
- `scripts/run_session_automation.py` — applies post-session updates, reconciles actuals
- `scripts/update_from_race.py` — manual race update trigger
- `scripts/update_from_testing.py` — manual testing/practice directionality update

This keeps user-facing latency predictable regardless of when sessions complete.

Artifacts are persisted through `ArtifactStore` with configurable storage backends:

| Mode | Behaviour |
|------|-----------|
| `file_only` | Local JSON (default) |
| `fallback` | DB-first read, file fallback |
| `dual_write` | Write both, DB-first read (migration mode) |
| `db_only` | Supabase only |

---

## Accuracy Tracking

The system tracks predictions at each checkpoint and compares them to actuals after
sessions complete. Accuracy is tracked per target (qualifying, race, sprint qualifying,
sprint race) and per checkpoint.

Current season metrics are stored in Supabase and surfaced in the **Prediction
Accuracy** dashboard tab.

The evaluation script writes a fuller review bundle to
[`docs/MODEL_CALIBRATION.md`](docs/MODEL_CALIBRATION.md) and
[`docs/MODEL_ERROR_ANALYSIS.md`](docs/MODEL_ERROR_ANALYSIS.md):

The live predictor also keeps residual history from completed events so it can
apply a learned minimum interval radius when recent Monte Carlo bands have been
too tight.

- **Calibration** — do the p5–p95 Monte Carlo intervals cover ~90% of actual outcomes?
- **Segment breakdowns** — where does accuracy shift by weekend format, weather, and track type?
- **Systematic bias** — which drivers or teams does the model consistently get wrong
  in the same direction across multiple races?
- **Baseline comparison** — does it beat a naive previous-race classifier on MAE and
  rank correlation?
- **Error analysis** — which weekends and drivers keep showing up among the biggest misses?

To regenerate after new races complete:

```bash
make evaluation-report
```

---

## Quick Start

Runtime target: Python 3.11.x

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local development:

```bash
uv sync --extra dev
source .venv/bin/activate
```

Generate predictions for an upcoming weekend:

```bash
python scripts/warmup_precompute.py --year 2026
```

Update from the most recent race:

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

Run the full check pipeline (lint + mypy + tests):

```bash
make check
```

---

## Tests

```bash
.venv/bin/pytest tests/
```

Targeted test suites:

```bash
pytest tests/test_baseline_2026_integration.py
pytest tests/test_tire_degradation.py        # 18 tests
pytest tests/test_pit_strategy.py            # 22 tests
pytest tests/test_dashboard_smoke.py
```

Nightly live FastF1 checks (CI):

```bash
make test-live-fastf1
```

Backtesting against 2025 season data:

```bash
python scripts/backtest_2025_season.py --year 2025 --max-races 6 --evaluation-mode historical --learning-mode both
```

Reviewer-facing artifacts land in `reports/backtest_2025/`:
`evaluation_packet.json`, `REVIEW_PACKET.md`, per-experiment summaries, and
recommendation output for ablations.

---

## Documentation

- `ARCHITECTURE.md` — component map and data flow
- `CONFIGURATION.md` — all tunable parameters
- `LIMITATIONS.md` — known model boundaries, assumptions, and what would fix them
- `docs/MODEL_CALIBRATION.md` — calibration, bias, and baseline comparison (auto-generated)
- `docs/MODEL_ERROR_ANALYSIS.md` — worst weekends, repeat misses, and failure patterns
- `reports/backtest_2025/REVIEW_PACKET.md` — reproducible historical evaluation bundle summary
- `docs/WEIGHT_SCHEDULE_GUIDE.md` — signal blending and trust progression
- `docs/FP_BLENDING_SYSTEM.md` — session blend mechanics
- `docs/COMPOUND_ANALYSIS.md` — tire compound performance system
- `docs/WEEKEND_PREDICTIONS.md` — normal vs sprint weekend cascade
- `docs/PREDICTION_TRACKING.md` — checkpoint prediction storage and accuracy
- `docs/PERSISTENCE_SUPABASE.md` — ArtifactStore modes and migration
- `docs/WARMUP_PRECOMPUTE.md` — background warmup worker
- `docs/DASHBOARD_AUTO_UPDATE.md` — what's automatic vs manual

---

## Stack

Python 3.11 · FastF1 · Streamlit · Supabase · Render · uv · pre-commit · mypy · pytest · GitHub Actions
