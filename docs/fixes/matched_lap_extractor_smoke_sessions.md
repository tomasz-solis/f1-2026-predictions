# Matched-Lap Extractor Smoke Sessions

Date: 2026-05-12
Status: locked

This file locks the small set of sessions used to smoke-test the
matched-lap extractor before the 2022-2025 bulk run.

These checks catch extractor bugs. They are not evidence that the prior is
calibrated.

The cells below are locked on the basis of read-only FastF1 inspector
output under `data/diagnostics/smoke_session_inspections/` plus direct
read-only inspection of cached FastF1 lap/stint data where noted. Evidence
observations are factual outputs from the generated files or cache reads.
The ranges are plausibility bands rather than heuristic midpoints. Canonical
skip reasons are separated from filter-diagnostic expectations.

## Candidate Sessions

### Clean Dry Race

- Status: `LOCKED`
- Event: 2024 Bahrain Grand Prix, Race
- Test purpose: baseline dry extraction with normal two-car teams and no
  weather routing complication.
- Expected matched-pair count: at least 80 and no more than 600 across all
  teams.
- Expected delta/sign: n/a, counts and routing only.
- Expected weather buckets: dry.
- Expected skip reasons: none expected.
- Evidence source:
  `data/diagnostics/smoke_session_inspections/2024_clean_dry_race.{json,txt}`

### Wet/Mixed Race

- Status: `LOCKED`
- Event: 2024 British Grand Prix, Race
- Test purpose: lap-level dry/wet routing and exclusion of mixed or
  unreliable laps.
- Expected matched-pair count: at least 20 matched rows with
  `weather_bucket = dry`, non-zero matched rows with `weather_bucket = wet`,
  and mixed/unreliable laps excluded from matched rows and counted
  diagnostically.
- Expected delta/sign: n/a, counts and routing only.
- Expected weather buckets: dry, wet, mixed/unreliable.
- Expected skip reasons:
  - `lap_level_weather_unreliable` for laps where samples are mixed or
    missing.
  - `insufficient_matched_pairs` for any team whose routed subset falls below
    the relevant minimum.
  - `weather_routing_excludes_session` only if lap-interval mapping proves
    unusable for this session.
- Evidence source:
  `data/diagnostics/smoke_session_inspections/2024_wet_mixed_race.{json,txt}`

### Early Teammate DNF

- Status: `LOCKED`
- Event: 2024 Australian Grand Prix, Race
- Test purpose: a teammate retiring early should prevent the surviving
  teammate from receiving a driver-rating update when no valid comparison
  exists.
- Expected matched-pair count: at most 2 raw candidate matched laps for Red
  Bull. VER retired on lap 4; lap 1 is excluded by rule and lap 4 is excluded
  as VER's final classified lap. Remaining teams should produce at least 60
  matched rows.
- Expected delta/sign: n/a, insufficient pairs by design for Red Bull.
- Expected weather buckets: dry.
- Expected skip reasons: `insufficient_matched_pairs` is primary for Red
  Bull. `teammate_dnf_no_matched_laps` is acceptable only if filters strip all
  residual VER/PER laps before pair construction.
- Evidence source:
  `data/diagnostics/smoke_session_inspections/2024_early_teammate_dnf.{json,txt}`

### Strategy-Asymmetric Race

- Status: `LOCKED`
- Event: 2024 Miami Grand Prix, Race
- Test purpose: safety-car/VSC laps and strategy asymmetry should be filtered
  out cleanly. The diagnostic dump should show VSC overlap on laps 22-23, SC
  overlap on laps 28-32, neutralization laps excluded by the green-flag
  track-status filter, post-neutralization stint-outlier filtering, and an
  asymmetric pair distribution where teammate strategies diverged.
- Expected matched-pair count: at least 50 total, visibly lower than Bahrain
  for the same number of two-car teams if SC/VSC exclusion and stint-outlier
  filtering bite.
- Expected delta/sign: n/a, counts and routing only.
- Expected weather buckets: dry.
- Expected skip reasons: `track_status_excluded_all_laps` or
  `all_laps_filtered_out` only if a specific pair's complete comparable
  sample is removed by the filter chain. Routine SC/VSC exclusion and
  stint-outlier filtering are diagnostic counts, not skip reasons.
- Evidence source:
  `data/diagnostics/smoke_session_inspections/2024_strategy_asymmetric_race.{json,txt}`
  plus cached FastF1 lap/stint inspection.

### Representative Qualifying

- Status: `LOCKED`
- Event: 2024 Bahrain Grand Prix, Qualifying
- Test purpose: common-segment qualifying extraction with
  `min_matched_pairs_quali = 3`.
- Branches exercised:
  - both teammates in Q3: Red Bull, Ferrari, McLaren, Mercedes;
  - both teammates in Q2 only: RB;
  - both teammates eliminated in Q1: Alpine, Kick Sauber;
  - split with Q2 as highest common segment: Aston Martin, Haas;
  - split with Q1 as highest common segment: Williams.
- Expected matched-pair count: at least 20 total across all teams.
- Expected delta/sign: n/a, counts and routing only.
- Expected weather buckets: dry.
- Expected skip reasons: none expected at team level.
  `insufficient_matched_pairs` is acceptable only for a team with aborted,
  deleted, or missing push laps in its highest common segment.
  `no_common_quali_segment` is not expected because all teams share Q1; cover
  that branch with a synthetic Phase 3 unit-test fixture.
- Evidence source:
  `data/diagnostics/smoke_session_inspections/2024_representative_qualifying.{json,txt}`

## Lock Decision

Locked on 2026-05-12. The lock criteria applied were:

- verify expected matched-pair count plausibility bands against the
  per-session inspector JSON (the bands above are wide on purpose; tighten
  only if the inspector evidence supports it);
- verify expected weather buckets against the inspector's reported rainfall
  sample counts;
- verify expected skip reasons against the canonical list in
  `docs/fixes/teammate_network_prior.md` Section 4.6;
- add independent source references for any expected delta/sign used as a
  magnitude check (none currently claimed; magnitude validation is Phase 1
  work, not smoke-session work);
- cut or replace any session whose expected behavior cannot be stated before
  coding.

All rows moved from `TODO_COUNTS` to `LOCKED` together with the file-level
status. This smoke set is ready for Phase 3 extractor implementation.

## Session Selection Notes

The candidate selections were verified against FastF1 cache via the Phase 2
read-only inspector. Inspector output is the evidence basis for each row's
expected behavior.

- Clean dry race (2024 Bahrain): needs no rain, no red-flag-truncated
  session, normal pit windows, and at least two two-car teams with full lap
  counts. Inspector confirmed dry weather, no retirements, and ten two-car
  teams.
- Wet/mixed race (2024 British): needs lap-level weather variation
  FastF1 actually records. Inspector confirmed 51 `Rainfall=True` and 96
  `Rainfall=False` samples, about a 35% rain fraction. This is real evidence,
  not edge noise, and lap-interval mapping is expected to produce a mix of
  dry, wet, and mixed/unreliable buckets. The hedged
  `weather_routing_excludes_session` expected skip applies only if
  lap-interval mapping later proves unusable for this session. The extractor
  contract currently defines one output table with a `weather_bucket` column,
  so dry and wet matched rows should be observable in the same diagnostic
  output. If implementation later splits dry-rating and wet-skill diagnostics,
  preserve the same expectation across the two dumps.
- Early teammate DNF (2024 Australian): needs at least one team where one
  driver retired early (lap < 10) and the other completed. Inspector
  confirmed VER retired lap 4, while PER finished. This is a useful boundary
  case: the residual candidate laps are below `min_matched_pairs_race = 8`,
  so the smoke check distinguishes `insufficient_matched_pairs` from
  `teammate_dnf_no_matched_laps` and can catch filter-chain bugs.
- Strategy-asymmetric race (2024 Miami): needs visible mid-race safety-car
  or VSC that split strategies between teammates. Inspector confirmed one SC
  row and two VSC rows, with all teams represented by two drivers and only SAR
  retiring late. Direct cached-lap inspection showed the VSC window overlapping
  laps 22-23 and the SC window overlapping laps 28-32. It also showed
  teammate-divergent stint choices around those windows, for example VER
  stopping before the neutralization while PER stopped during it, NOR stopping
  under SC while PIA had already stopped and later stopped again, and ALO/STR
  splitting VSC/SC-era stops.
- Representative qualifying (2024 Bahrain): needs common-segment logic
  exercised on more than one branch. Inspector confirmed four teams with both
  drivers reaching Q3 (Red Bull, Ferrari, McLaren, Mercedes), one team with
  both drivers reaching Q2 only (RB), two teams where both drivers stopped in
  Q1 (Alpine, Kick Sauber), and split teams such as Williams (ALB Q2, SAR Q1),
  Aston Martin (ALO Q3, STR Q2), and Haas (HUL Q3, MAG Q2). This row tests
  real common-segment processing but does not test `no_common_quali_segment`,
  which is better covered by a synthetic Phase 3 unit test.

### Inspector Evidence Confirmed

These items are recorded so the next reader does not have to re-derive the
reasoning:

- 2024 Bahrain race: 157 dry samples, 0 rain samples, no retirements, ten
  two-car teams.
- 2024 British weather sampling: 51 rain samples and 96 dry samples, about a
  35% rain fraction. Adequate for lap-level routing tests.
- 2024 Australian Red Bull DNF case: VER retired lap 4 while PER finished.
  Red Bull is the affected team that should fail to receive a race-rating
  update.
- 2024 Miami SC/VSC case: one SC row and two VSC rows in the FastF1
  track-status data. Cached lap inspection maps the VSC to lap overlap on
  laps 22-23 and the SC to lap overlap on laps 28-32, with visible teammate
  strategy divergence around those windows.
- 2024 Bahrain qualifying segments: four teams with both drivers in Q3, one
  team with both drivers Q2-only, two teams with both drivers Q1-only, and
  several split teams. All teams shared Q1, so this session cannot test
  `no_common_quali_segment`.

## Review Notes

Five sessions are enough to catch obvious bugs, not systematic bias.
Systematic issues are caught later by the bulk extraction diagnostic dump and
replay diagnostics.

The plausibility bands above are deliberately wide. They are designed to catch
the failure modes the smoke check exists to catch: extractor returns about
zero pairs, extractor returns wildly too many pairs, or extractor returns the
wrong distribution across categories. Tighter bands would require running the
Phase 3 extractor itself, which would be circular. The smoke bands cannot be
derived from extractor output before Phase 4.

Smoke-session expected behavior is grounded in FastF1 inspector output, which
is reproducible from cache. Magnitude validation lives in
`docs/fixes/teammate_network_prior_validation_evidence.md`.
