# Driver Seconds And Wet Trace Remediation

Status: in progress  
Started: 2026-05-21

This plan closes two Model Diagnostics coverage gaps:

1. dry leakage is still measured through legacy `bayesian.rating_mu`, which is
   not a seconds field;
2. 2026 replay has no real wet weather-routed sample and no persisted
   session-level driver update trace for proving the wet dry-state invariant.

The fixes are related but not interchangeable. Dry leakage needs the
race/qualifying driver-seconds schema cutover. Wet leakage needs trace evidence
and then real wet replay coverage. UI wording alone does not close either gap.

## Target Conditions

### Dry leakage

The exact diagnostic is:

```text
corr(delta_race_rating_mu_s, delta_team_strength_for_driver_team)
```

It is complete only when persisted driver artifacts contain real seconds-native
race and qualifying fields, the live/replay updater moves those fields, and the
diagnostics builder reports the seconds metric instead of the legacy proxy.

### Wet leakage

Code-level closure means the production updater emits session-level update
trace rows and the diagnostics evaluator proves that a fully wet route applies
zero dry race/qualifying driver-state delta.

2026 coverage closure is stricter: a real 2026 wet weather-routed replay sample
and a matching traced update must exist. Until that happens, the dashboard must
keep saying that 2026 wet replay coverage is absent.

## Work Plan

### 1. Wet trace foundation

- [x] Guard fully wet race, qualifying, and sprint result updates from moving
  the current dry driver-state paths.
- [x] Define a persisted driver update trace row with event/session/weather
  route, before/after dry state, before/after wet skill, and applied-update
  flags.
- [x] Emit trace rows from the production updater boundary without changing
  normal artifact writes.
- [x] Carry trace rows through historical replay and write a replay report.
- [x] Evaluate the fully wet dry-state invariant from trace rows in replay
  diagnostics.
- [x] Add a synthetic failure test for the invariant evaluator.
- [ ] Add historical wet evidence tests for the invariant evaluator.

### 2. Driver-seconds schema migration

- [x] Extend driver artifact validation for `race_rating_*` and
  `quali_rating_*` fields while accepting old-only, mixed, and new-only
  artifacts.
- [x] Add reader/writer helpers that keep legacy rating units separate from
  seconds-native fields.
- [x] Add a migration script that snapshots current local artifacts, seeds
  seconds fields from the teammate-network prior artifact, writes a report, and
  fails on missing active-driver coverage instead of deriving seconds from
  `rating_mu`.
- [x] Verify the same mixed-artifact behavior through the Supabase
  `ArtifactStore` path.

Implementation note (2026-05-21): the first local dry-run exposed that
current-lineup driver `LIN` has no race or qualifying node in the
teammate-network prior. The seed path now handles that specific class of gap
through a generated debut-season rookie fallback. It uses the median of
historical rookie debut-season seconds estimates in the same matched-lap
construct, keeps wide data-derived uncertainty, and records which drivers used
the fallback. The legacy `bayesian.rating_mu` value is not a valid substitute.

The fallback is not a calendar rule. Per session kind, live seconds-state
updates should replace it after at least the teammate-network prior
`min_driver_observations` evidence threshold (`24` in the current artifact) of
construct-aligned driver observations.

Verification note (2026-05-21): after backfilling the updated 2026 driver
artifact, a DB-only predictor load and DB-only warmup both read the driver
payload through `ArtifactStore` and passed driver-characteristics validation.
That covers the mixed legacy-plus-seconds artifact path used by the live app.

### 3. Live seconds-state cutover

- [x] Split live dry driver state into race seconds and qualifying seconds
  update paths.
- [x] Keep race updates from clobbering qualifying state and qualifying updates
  from clobbering race state.
- [x] Make prediction readers prefer seconds fields after cutover while
  retaining the temporary legacy fallback.
- [x] Rebuild 2026 artifacts, replay, warmup, and production storage state
  after artifact fingerprints change.

Implementation note (2026-05-21): the live race boundary now extracts dry
canonical matched-lap aggregates from the main race and qualifying sessions.
Those rows update only `race_rating_*` or only `quali_rating_*`. If FastF1
cannot supply lap/weather inputs for that construct, the seconds update skips
with a warning instead of deriving seconds from result positions. Prediction
keeps using the existing legacy driver signals when a driver artifact has no
complete seconds state.

Sprint path implementation (2026-05-22): sprint learning is live-wired.
`auto_update_from_races()` now calls `update_from_sprint_race()` before the
main-race updater on sprint weekends, and historical replay follows the same
order. The sprint updater extracts construct-aligned `SQ` and `Sprint`
matched-lap aggregates. `SQ` changes only qualifying seconds; `Sprint` changes
only race seconds. Both use the accepted v1 0.5 sprint rule as half aggregate
precision rather than treating sprint matched laps like full main-session
evidence.

Verification note (2026-05-21): the 2026 race artifacts and historical replay
were rebuilt from the seeded driver baseline. File-backed warmup regenerated
the current PRE horizon for Canadian, Monaco, and Barcelona. The updated
artifacts and runtime warmup state were backfilled to Supabase, then DB-only
warmup regenerated the same horizon with read-after-write verification enabled
and no DB verification warnings.

### 4. Exact leakage diagnostics

- [x] Compute dry leakage from `delta_race_rating_mu_s` when seconds field
  coverage exists.
- [x] Keep a blocked state only when seconds fields are genuinely unavailable.
- [x] Regenerate local diagnostics and sync the persisted diagnostics artifact
  to Supabase.

Verification note (2026-05-21): the persisted replay/leakage diagnostics now
report the exact dry driver-seconds metric as measured. The diagnostics builder
was run in DB-backed mode after replay rebuild, so the dashboard-facing
artifact in Supabase matches the local artifact. The wet limitation remains
open because 2026 replay still has no real wet weather-routed sample.

### 5. Real 2026 wet closure

- [ ] Rebuild matched-lap observations when a real 2026 wet event exists.
- [ ] Replay the traced update path over that wet event.
- [ ] Regenerate diagnostics so the wet coverage limitation disappears only
  after a real 2026 wet sample passes the invariant.

## Implementation Order

1. Wet trace foundation.
2. Driver-seconds schema migration.
3. Live seconds-state cutover.
4. Exact seconds diagnostics.
5. Real 2026 wet closure when weather evidence exists.

## Non-Goals

- Do not hide missing evidence by suppressing diagnostics messages.
- Do not infer seconds-native driver fields from legacy `rating_mu`.
- Do not remove legacy reader fallback paths until the K=3 production-weekend
  removal rule in `master_execution_plan.md` is satisfied.
