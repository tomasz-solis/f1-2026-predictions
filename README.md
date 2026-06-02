# Trackside Labs

Trackside Labs is an F1 race-weekend prediction system for the 2026 regulation
reset. It produces checkpointed qualifying and race forecasts, then updates them
as practice, qualifying, sprint, and race data arrives.

The hard part is not running a simulation. The hard part is deciding how much to
trust each source of evidence when the car concept has just changed and the
season has almost no current data. A 2025 baseline is useful, but stale. Testing
is current, but teams hide pace. Friday practice is fresh, but noisy. The system
is built around that tradeoff.

## Current Model Work

The next planned model change is the driver-rating de-carring work described in
`docs/fixes/`. It separates team strength from driver residuals so driver skill
is not accidentally estimated through car performance.

Planned direction:

- keep active-season learning centred on `team_strength`
- split driver residuals into `race_rating_mu_s` and `quali_rating_mu_s`
- store driver residuals directly in seconds
- make driver ratings slower-moving than team strength during the season
- use separate mappings from team strength to qualifying and race pace
- migrate local JSON artifacts and Supabase rows together
- add dashboard diagnostics for scale drift, residuals, slope, and R-squared

This work is staged because it touches extraction, validation, local artifacts,
Supabase persistence, dashboard diagnostics, and live prediction readers. The
matched-lap extractor comes first; live updater changes come last.

## Prediction Problem

At race 1, the model has three days of pre-season testing and no 2026 race
results. By race 5, it has testing, four race weekends, and several sessions of
practice and qualifying evidence. The model has to behave sensibly at both ends
of that range.

Main modelling constraints:

- Pre-season testing is directional evidence, not ground truth. Fuel load,
  engine mode, run plan, and sandbagging all matter.
- The blend must move quickly from stale priors toward current-season results.
- Compound performance is circuit-specific and is only used once there are
  enough laps for that compound and team.
- Rookie and missing-driver cases fall back to a team baseline with added
  uncertainty.
- Predictions need to refresh after each session without making the dashboard
  slow for users.

## How The Model Works

Team strength is blended from three signals:

```text
blended_strength = w_baseline * baseline
                 + w_testing * testing_modifier
                 + w_current * current_season_mean
```

The active reset-year schedule is intentionally aggressive:

| Race | Baseline | Testing | Current |
|------|----------|---------|---------|
| 1    | 35%      | 20%     | 45%     |
| 2    | 20%      | 10%     | 70%     |
| 3    | 8%       | 5%      | 87%     |
| 4+   | 5%       | 0%      | 95%     |

Before current-season races exist, the current-season term falls back to the
baseline rather than zero.

### Qualifying

Qualifying prediction combines team strength, available session pace, driver
skill, and uncertainty:

```text
team strength
session pace from available FP or sprint sessions
confidence-scaled session weight
driver skill adjustment
Monte Carlo simulation
median grid position and interval
```

Normal weekends weight FP3 most heavily, then FP2 and FP1. Sprint weekends use
sprint qualifying, FP1, and sprint evidence when those sessions exist. If no
session data is available, the model runs from priors only.

### Race

Race prediction starts from the actual or predicted grid, then runs a lap-level
simulation:

```text
grid position
compound strengths from session history
pit strategy sampling
lap-by-lap tire degradation and fuel effect
traffic and clean-air adjustments
track-specific pit loss
safety car, lap-1, and DNF events
finish order and probability table
```

Each race forecast currently uses 300 Monte Carlo runs. Outputs include finish
order, strategy distribution, podium probability, and uncertainty bands.

### Learning

After a race, the system updates driver error state and teammate-gap calibration
from actual results. Those adjustments feed the next prediction cycle.

Learning is gated. Retrospective runs, duplicate run IDs, missing actuals, and
tiny actual overlaps are skipped so the model does not train on partial or
contaminated records.

## Runtime Design

The dashboard request path is read-only. Artifact generation runs in background
workers:

- `scripts/warmup_precompute.py`: checkpoint-aware warmup used in production
- `scripts/run_session_automation.py`: post-session updates and actuals
- `scripts/update_from_race.py`: manual race update trigger
- `scripts/update_from_testing.py`: manual testing and practice directionality update

Artifacts are read and written through `ArtifactStore`.

| Mode | Behaviour |
|------|-----------|
| `file_only` | Local JSON only |
| `fallback` | DB-first read, file fallback |
| `dual_write` | Write file and DB during migration |
| `db_only` | Supabase only |

## Evaluation

The system stores predictions at each checkpoint and compares them with actuals
after sessions complete. Accuracy is tracked by target and checkpoint:

- qualifying
- race
- sprint qualifying
- sprint race

The evaluation bundle is written to:

- `docs/MODEL_CALIBRATION.md`
- `docs/MODEL_ERROR_ANALYSIS.md`

It covers interval calibration, segment breakdowns, systematic bias, baseline
comparison, error analysis, and promotion gates for experimental components.

Regenerate the report after new races complete:

```bash
make evaluation-report
```

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

Run lint, type checks, and tests:

```bash
make check
```
