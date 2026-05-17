# Qualifying Construct Probe

Built at: `2026-05-17T18:56:37.796551+00:00`

| Check | HARD threshold | Phase 5 WLS | Phase 5 equal mean | Cache current mean | Highest-common best | Any-valid best | Phase 5 rows | Cache rows | Cache mismatch rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `verstappen_perez_quali_2022` | 0.290s | 0.361s | 0.157s | 0.157s | 0.189s | 0.210s | 20 | 20 | 0 |
| `verstappen_perez_quali_2023` | 0.621s | 0.363s | 0.543s | 0.543s | 0.672s | 0.839s | 19 | 19 | 0 |
| `verstappen_perez_quali_2024` | 0.660s | 0.462s | 0.507s | 0.507s | 0.467s | 0.610s | 18 | 18 | 0 |
| `russell_hamilton_quali_2024` | 0.230s | 0.113s | 0.083s | 0.083s | 0.345s | 0.353s | 18 | 18 | 0 |
| `albon_sargeant_quali_2023` | 0.522s | 0.412s | 0.418s | 0.418s | 0.554s | 1.236s | 13 | 13 | 0 |
| `albon_sargeant_quali_2024` | 0.660s | 0.222s | 0.402s | 0.402s | 0.380s | 0.636s | 5 | 5 | 0 |

## Notes

- `Phase 5` columns come from the stored aggregate artifact.
- `Cache` columns are fresh offline FastF1 recomputations using the current extractor.
- `Cache mismatch rows` counts sessions where the fresh current construct does not reproduce the stored Phase 5 delta to within 1ms.
