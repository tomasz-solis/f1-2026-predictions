# Qualifying Network Peak Probe

Built at: `2026-05-17T19:44:38.789767+00:00`

## Coverage

| Measure | Value |
| --- | ---: |
| Pair-seasons | 52 |
| Pair-seasons with current rows | 49 |
| Pair-seasons with highest-common-best rows | 52 |
| Phase 5 artifact rows | 627 |
| Current construct rows | 627 |
| Highest-common-best rows | 880 |
| Any-valid-best rows | 881 |
| Peak row gain vs current | 253 |
| Highest-common-best rows with absolute delta above `1s` | 60 |
| Highest-common-best rows with absolute delta above `2s` | 18 |
| Artifact/cache mismatch rows | 0 |

## Pair-Season Dispersion

| Construct | Count | Median SD | P75 SD | Max SD |
| --- | ---: | ---: | ---: | ---: |
| Current construct | 46 | 0.317s | 0.412s | 0.869s |
| Highest-common best | 48 | 0.460s | 0.648s | 1.328s |
| Any-valid best | 48 | 0.854s | 1.307s | 8.907s |

## Pair-Season Mean Shifts

| Measure | Count | Median | P75 | Max |
| --- | ---: | ---: | ---: | ---: |
| Peak minus current equal mean | 49 | 0.001s | 0.097s | 0.490s |
| Phase 5 WLS vs equal absolute shift | 49 | 0.045s | 0.094s | 0.238s |

## Highest Common Segment Mix

| Segment | Rows |
| --- | ---: |
| `Q1` | 357 |
| `Q2` | 248 |
| `Q3` | 275 |

## Largest Peak Row Gains

| Pair-season | Current rows | Peak rows | Gain |
| --- | ---: | ---: | ---: |
| `2025:Aston Martin:ALO-STR` | 11 | 23 | 12 |
| `2022:Williams:ALB-LAT` | 6 | 18 | 12 |
| `2023:Haas F1 Team:HUL-MAG` | 10 | 21 | 11 |
| `2025:Kick Sauber:BOR-HUL` | 10 | 21 | 11 |
| `2025:Alpine:COL-GAS` | 7 | 18 | 11 |
| `2024:Alpine:GAS-OCO` | 12 | 22 | 10 |
| `2024:Kick Sauber:BOT-ZHO` | 11 | 21 | 10 |
| `2025:Haas F1 Team:BEA-OCO` | 14 | 23 | 9 |
| `2023:McLaren:NOR-PIA` | 14 | 22 | 8 |
| `2023:Williams:ALB-SAR` | 13 | 21 | 8 |

## Largest Absolute Peak Session Deltas

| Pair-season | Race | Segment | Current delta | Peak delta |
| --- | --- | --- | ---: | ---: |
| `2024:Alpine:GAS-OCO` | British Grand Prix | `Q1` | - | -5.247s |
| `2023:Red Bull Racing:PER-VER` | Canadian Grand Prix | `Q2` | -0.731s | -4.299s |
| `2023:Haas F1 Team:HUL-MAG` | Belgian Grand Prix | `Q1` | - | -3.908s |
| `2022:Red Bull Racing:PER-VER` | Singapore Grand Prix | `Q3` | 3.185s | 3.734s |
| `2022:Williams:ALB-LAT` | Australian Grand Prix | `Q1` | - | 3.435s |
| `2023:McLaren:NOR-PIA` | Mexico City Grand Prix | `Q1` | - | -3.313s |
| `2025:Alpine:COL-GAS` | Las Vegas Grand Prix | `Q2` | - | -3.076s |
| `2024:Mercedes:HAM-RUS` | Las Vegas Grand Prix | `Q3` | 0.064s | -2.869s |
| `2023:Haas F1 Team:HUL-MAG` | British Grand Prix | `Q1` | - | 2.775s |
| `2023:Alpine:GAS-OCO` | Belgian Grand Prix | `Q2` | -0.495s | 2.701s |

## Notes

- `Current construct` is the existing multi-run median selected by the extractor.
- `Highest-common best` is one best-lap delta from the highest valid common segment.
- Dispersion rows summarize within-pair-season session SDs, not pooled raw deltas.
