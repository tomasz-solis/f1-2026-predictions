# Replay And Leakage Diagnostics

- Built at: `2026-05-20T08:58:35.450017+00:00`
- Model version: `2.1`
- Status: `provisional_with_warnings`

## Source State

- Replay races: `4`
- Live artifact races completed: `4.0`
- Replay stale vs live artifact: `False`

## Warnings

- Dry leakage is reported as a legacy rating-mu proxy until schema migration adds seconds fields.
- Regulation-reset scale monitoring is outside the 2024-2025 one-SE band for: qualifying, race.
- Wet-leakage hard invariant is not evaluable without wet routed replay rows.

## Historical Reference

- `race`: slope mean `0.923`, slope SE `0.060`, R² mean `0.470`
- `qualifying`: slope mean `0.653`, slope SE `0.047`, R² mean `0.427`

## Regulation-Reset Monitoring

| Session | State | Rows | Races | Slope | R² | RMSE | Outside 1SE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `race` | `measured` | 61 | 4 | 1.952 | 0.546 | 1.029 | `True` |
| `qualifying` | `measured` | 65 | 4 | 1.116 | 0.671 | 0.575 | `True` |

## Dry Leakage

- State: `measured_legacy_proxy`
- Exact metric state: `schema_blocked_until_race_quali_seconds_fields_exist`
- Correlation: `0.159`
- Drivers: `22`

## Wet Leakage

- State: `not_evaluable_without_weather_routed_wet_replay_rows`
- Hard invariant state: `not_evaluable_from_current_inputs`
