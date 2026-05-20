# Team-Strength Construct-Row Audit

- Built at: `2026-05-20T09:04:34.165292+00:00`
- Model version: `2.1`
- Status: `measured`
- Policy: `same_session_construct`
- Rows: `126`

## Scale Metrics

| Session | Rows | Races | Teams | Drivers | Combined slope | Team-target slope | RMSE | Outside 1SE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `race` | 61 | 4 | 11 | 21 | 1.952 | 2.243 | 1.029 | `True` |
| `qualifying` | 65 | 4 | 10 | 19 | 1.116 | 1.398 | 0.575 | `True` |

## Highest Leave-One-Race Influence

| Session | Omitted | Rows | Slope | Delta | RMSE | Outside 1SE |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `race` | Chinese Grand Prix | 13 | 2.127 | 0.174 | 1.088 | `True` |
| `race` | Australian Grand Prix | 13 | 1.824 | -0.128 | 0.979 | `True` |
| `race` | Miami Grand Prix | 14 | 1.902 | -0.050 | 1.014 | `True` |
| `qualifying` | Japanese Grand Prix | 19 | 0.993 | -0.123 | 0.506 | `True` |
| `qualifying` | Australian Grand Prix | 13 | 1.202 | 0.086 | 0.616 | `True` |
| `qualifying` | Miami Grand Prix | 14 | 1.194 | 0.078 | 0.603 | `True` |

## Highest Leave-One-Team Influence

| Session | Omitted | Rows | Slope | Delta | RMSE | Outside 1SE |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `race` | Cadillac | 8 | 1.692 | -0.260 | 0.791 | `True` |
| `race` | Red Bull Racing | 4 | 2.175 | 0.223 | 1.055 | `True` |
| `race` | Mercedes | 8 | 1.821 | -0.132 | 1.014 | `True` |
| `qualifying` | Aston Martin | 4 | 0.948 | -0.168 | 0.454 | `True` |
| `qualifying` | Audi | 6 | 1.216 | 0.100 | 0.573 | `True` |
| `qualifying` | Ferrari | 8 | 1.212 | 0.096 | 0.598 | `True` |

## Largest Absolute Residual Rows

| Session | Race | Team | Driver | Observed | Predicted | Residual | Laps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `race` | Australian Grand Prix | Cadillac | PER | -3.264 | -0.386 | -2.877 | 9 |
| `race` | Miami Grand Prix | Cadillac | BOT | -3.445 | -0.918 | -2.527 | 10 |
| `race` | Japanese Grand Prix | Cadillac | PER | -2.376 | -0.189 | -2.187 | 19 |
| `race` | Japanese Grand Prix | Aston Martin | ALO | -3.088 | -1.002 | -2.086 | 19 |
| `qualifying` | Japanese Grand Prix | Aston Martin | ALO | -2.921 | -0.917 | -2.004 | 3 |
| `race` | Miami Grand Prix | Cadillac | PER | -2.310 | -0.386 | -1.924 | 10 |
| `race` | Australian Grand Prix | Cadillac | BOT | -2.650 | -0.918 | -1.732 | 9 |
| `race` | Japanese Grand Prix | Aston Martin | STR | -2.993 | -1.294 | -1.699 | 19 |
| `qualifying` | Japanese Grand Prix | Aston Martin | STR | -2.862 | -1.172 | -1.691 | 3 |
| `race` | Chinese Grand Prix | Cadillac | PER | -2.035 | -0.386 | -1.649 | 35 |

## Decision Notes

- This audit is diagnostic-only; it does not retune extractor semantics or priors.
- Rows use the frozen same-session construct and the current teammate-network priors.
- race remains outside the 2024-2025 one-SE slope band in the full sample.
- race top leave-one-race influence is Chinese Grand Prix (slope delta 0.174, outside band after omission: True).
- race top leave-one-team influence is Cadillac (slope delta -0.260, outside band after omission: True).
- qualifying remains outside the 2024-2025 one-SE slope band in the full sample.
- qualifying top leave-one-race influence is Japanese Grand Prix (slope delta -0.123, outside band after omission: True).
- qualifying top leave-one-team influence is Aston Martin (slope delta -0.168, outside band after omission: True).
