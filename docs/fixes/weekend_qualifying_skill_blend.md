# Weekend Qualifying Skill Blend

Status: accepted for live 2026 predictions  
Decision date: 2026-05-22

## Decision

Live qualifying prediction blends the static driver `skill_score` toward the
weekend-updated Bayesian driver form after completed race weekends.

The live schedule is:

```text
bayesian_quali_skill_blend_per_race = 0.45
bayesian_quali_skill_blend_cap = 0.90
```

Race skill stays on its existing slower schedule. This change does not hand-edit
driver priors and does not reinterpret the seconds-native driver fields. It
changes how much the live qualifying path trusts the driver form posterior that
the updater already refreshes from completed dry weekends.

## Why

The 2026 driver order cannot wait for a full-season sample before reacting to a
regulation-reset season. Four completed weekends are enough to test whether the
weekend-updated signal helps the live output now, but not enough to claim the
blend rate is final for the full season.

The first check was the RUS/ANT symptom. That was not used as the release test.
The release test was a sequential 2026 prediction replay over Australia, China,
Japan, and Miami against the current baseline.

## Replay Evidence

The qualifying-only fast blend is the accepted candidate:

| Candidate | All-target position MSE | Race-target position MSE |
| --- | ---: | ---: |
| Current baseline | `34.289` | `35.909` |
| Conservative qualifying blend (`0.20`, cap `0.60`) | `34.126` | `35.709` |
| Fast qualifying and race blend (`0.45`, cap `0.90`) | `33.818` | `35.309` |
| Fast qualifying-only blend (`0.45`, cap `0.90`) | `33.770` | `35.227` |

Australia is unchanged because no completed 2026 weekend exists before its
checkpoints. The accepted candidate improves aggregate replay MSE on each
post-update weekend in the current sample:

- China: all-target MSE delta `-1.45%`;
- Japan: all-target MSE delta `-2.23%`;
- Miami: all-target MSE delta `-2.97%`.

Speeding up race skill did not beat the qualifying-only candidate. That is why
race skill is unchanged.

## Whole-Grid Check

The accepted candidate was checked over the full predicted field, not only
RUS/ANT:

- `34` scored checkpoint-target rows;
- `22` drivers in baseline, candidate, and actual rows each time;
- `748` driver-position comparisons;
- no driver-set mismatches.

The candidate improves aggregate field error while making smaller regressions
than the larger corrections it captures:

- field MSE: `34.289 -> 33.770`;
- field MAE: `4.476 -> 4.444`;
- mean-MSE gains across improved drivers sum to about `-21.0`;
- mean-MSE regressions across worsened drivers sum to about `+9.6`.

That supports the field-level objective. It does not claim every driver improves
on a four-weekend sample.

## Monitoring Rule

Re-run the same sequential replay gate after completed weekends. Keep the live
blend only while the aggregate field MSE/MAE remains supported and driver-level
regressions stay explainable as smaller rank nudges rather than a broad error
shift. Treat a future normal-weekend-only consistency check as a separate
diagnostic once the 2026 sample contains enough post-update normal weekends.
