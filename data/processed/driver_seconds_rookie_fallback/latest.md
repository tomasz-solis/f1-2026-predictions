# Driver Seconds Rookie Fallback

Built at: `2026-05-21T09:57:55.937553+00:00`
Cohort years: 2022, 2023, 2024, 2025

This artifact gives unseen debut-season drivers a data-derived seconds state when the teammate-network prior has no driver node.

## Fallback values

| Session | Mean (s) | Sigma (s) | Rookies | Implied observations |
| --- | ---: | ---: | ---: | ---: |
| Race | -0.190761 | 0.529271 | 10 | 126 |
| Qualifying | -0.031332 | 0.530421 | 10 | 100 |

## Cohort

| Session | Driver | Debut | Rows | Median implied mean (s) |
| --- | --- | ---: | ---: | ---: |
| Race | `ANT` | 2025 | 22 | -0.364320 |
| Race | `BEA` | 2024 | 2 | -0.308046 |
| Race | `BOR` | 2025 | 19 | -0.314415 |
| Race | `COL` | 2024 | 6 | 0.188093 |
| Race | `DOO` | 2025 | 3 | -0.411377 |
| Race | `HAD` | 2025 | 21 | 0.154915 |
| Race | `LAW` | 2023 | 2 | -0.919622 |
| Race | `PIA` | 2023 | 18 | -0.050273 |
| Race | `SAR` | 2023 | 17 | 0.146593 |
| Race | `ZHO` | 2022 | 16 | -0.073476 |
| Qualifying | `ANT` | 2025 | 19 | 0.188217 |
| Qualifying | `BEA` | 2024 | 2 | -0.199826 |
| Qualifying | `BOR` | 2025 | 10 | -0.551188 |
| Qualifying | `COL` | 2024 | 3 | 0.120845 |
| Qualifying | `DOO` | 2025 | 3 | 0.022781 |
| Qualifying | `HAD` | 2025 | 17 | -0.085190 |
| Qualifying | `LAW` | 2023 | 4 | -0.595767 |
| Qualifying | `PIA` | 2023 | 14 | 0.018259 |
| Qualifying | `SAR` | 2023 | 13 | -0.008655 |
| Qualifying | `ZHO` | 2022 | 15 | -0.054008 |

## Promotion policy

Replace the fallback per session kind after at least `24` construct-aligned driver observations.
