# Replay And Leakage Diagnostics

- Built at: `2026-05-22T07:56:03.985821+00:00`
- Model version: `2.2`
- Status: `provisional_with_limitations`

## Source State

- Replay races: `4`
- Live artifact races completed: `4.0`
- Replay stale vs live artifact: `False`

## Coverage Limitations

- Current replay coverage has no wet weather-routed rows, so the wet-leakage replay invariant has no real 2026 wet sample yet.

## Monitoring Notes

- Reset-year scale differs from the 2024-2025 reference band for qualifying, race. The comparison stays visible for transfer review; it is not a warning by itself.

## Historical Reference

- `race`: slope mean `0.923`, slope SE `0.060`, R² mean `0.470`
- `qualifying`: slope mean `0.653`, slope SE `0.047`, R² mean `0.427`

## Regulation-Reset Monitoring

| Session | State | Rows | Races | Slope | R² | RMSE | Outside 1SE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `race` | `measured` | 61 | 4 | 1.952 | 0.546 | 1.029 | `True` |
| `qualifying` | `measured` | 65 | 4 | 1.116 | 0.671 | 0.575 | `True` |

## Dry Leakage

- State: `measured_seconds`
- Exact metric state: `measured`
- Correlation: `-0.089`
- Drivers: `22`

## Wet Leakage

- State: `not_evaluable_without_weather_routed_wet_replay_rows`
- Hard invariant state: `not_evaluable_without_fully_wet_trace_rows`
