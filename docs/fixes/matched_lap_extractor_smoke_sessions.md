# Matched-Lap Extractor Smoke Sessions

Date: 2026-05-09
Status: scaffold; not lockable until expected counts are filled

This file locks the small set of sessions used to smoke-test the
matched-lap extractor before the 2022-2025 bulk run.

These checks catch extractor bugs. They are not evidence that the prior is
calibrated.

Expected counts and approximate deltas must be filled before extractor code
is written. If a magnitude cannot be sourced independently, leave the
magnitude blank and use the session only for counts, weather routing, and
skip reasons.

## Candidate Sessions

| Status | Category | Year | Race | Session | What it should test | Expected matched-pair count | Expected delta/sign | Expected weather buckets | Expected skip reasons | Evidence source |
|---|---|---:|---|---|---|---:|---|---|---|---|
| TODO_COUNTS | Clean dry race | 2024 | Bahrain Grand Prix | Race | Baseline dry extraction with normal two-car teams and no weather routing complication. | TODO | TODO if source exists | dry | TODO | TODO |
| TODO_COUNTS | Wet/mixed race | 2024 | British Grand Prix | Race | Lap-level dry/wet routing and exclusion of lap-level mixed/unreliable laps. | TODO | TODO if source exists | dry, wet, mixed/unreliable | `lap_level_weather_unreliable` where applicable | TODO |
| TODO_COUNTS | Early teammate DNF | 2024 | Australian Grand Prix | Race | A teammate retiring early should prevent the surviving teammate from receiving a driver-rating update when no valid comparison exists. | TODO | n/a if insufficient pairs | dry | `teammate_dnf_no_matched_laps` or `insufficient_matched_pairs` for affected team | TODO |
| TODO_COUNTS | Strategy-asymmetric race | 2024 | Miami Grand Prix | Race | Safety-car and strategy asymmetry should be visible in skip reasons and not silently become driver skill. | TODO | TODO if source exists | dry | TODO | TODO |
| TODO_COUNTS | Representative qualifying | 2024 | Bahrain Grand Prix | Qualifying | Common-segment qualifying extraction with `min_matched_pairs_quali = 3`. | TODO | TODO if source exists | dry | `no_common_quali_segment` for any pair without enough comparable laps | TODO |

## Lock Rules

Before extractor implementation starts:

- replace each `TODO_COUNTS` status with `LOCKED`;
- fill expected matched-pair count ranges, not single exact numbers;
- fill expected weather buckets;
- fill expected skip reasons;
- add independent source references for any expected delta/sign used as a
  magnitude check;
- cut or replace any session whose expected behavior cannot be stated
  before coding.

## Session Selection Notes

The candidate selections above are starting points, not locked choices.
Replace or cut any session that does not fit the category once verified
against FastF1 cache:

- **Clean dry race**: needs no rain, no red-flag-truncated session, normal
  pit windows, and at least two two-car teams with full lap counts. 2024
  Bahrain is a reasonable starting candidate.
- **Wet/mixed race**: needs lap-level weather variation FastF1 actually
  records. Verify `session.weather_data` has both `Rainfall == True` and
  `Rainfall == False` samples within the session before locking.
- **Early teammate DNF**: needs at least one team where one driver retired
  early (lap < 10, ideally) and the other completed the race. Verify the
  retirement lap before locking.
- **Strategy-asymmetric race**: needs visible mid-race safety-car or VSC
  that split strategies between teammates. Position-change filter and
  stint-outlier filter behavior should be observable.
- **Representative qualifying**: needs at least one team where both drivers
  reached Q3 and at least one team where both stopped in Q1, so the
  common-segment logic is exercised on more than one branch.

## Review Notes

Five sessions are enough to catch obvious bugs, not systematic bias.
Systematic issues are caught later by the bulk extraction diagnostic dump
and replay diagnostics.

The candidate sessions above are not externally verified. Do not treat any
"expected matched-pair count" as locked until it is filled from a source
that is not this project's own extractor output.
