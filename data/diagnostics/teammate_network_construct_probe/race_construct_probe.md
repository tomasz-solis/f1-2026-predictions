# Race Construct Probe

Built at: `2026-05-17T19:51:52.714682+00:00`

| Check | HARD threshold | Phase 5 WLS | Phase 5 equal mean | Cache current mean | Broad valid-lap median | Broad valid-lap mean | Phase 5 rows | Cache current rows | Broad rows | Cache mismatch rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `verstappen_perez_race_2022` | 0.234s | 0.281s | 0.250s | 0.250s | 0.322s | 0.330s | 19 | 19 | 22 | 0 |
| `verstappen_perez_race_2023` | 0.451s | 0.295s | 0.259s | 0.259s | 0.298s | 0.462s | 20 | 20 | 21 | 0 |
| `verstappen_perez_race_2024` | 0.560s | 0.624s | 0.667s | 0.667s | 0.488s | 0.636s | 21 | 21 | 22 | 0 |
| `alonso_stroll_race_2023` | 0.486s | 0.413s | 0.364s | 0.364s | 0.325s | 0.372s | 20 | 20 | 21 | 0 |
| `alonso_stroll_race_2024` | 0.250s | 0.223s | 0.249s | 0.249s | 0.359s | 0.462s | 20 | 20 | 23 | 0 |
| `albon_sargeant_race_2023` | 0.293s | 0.257s | 0.214s | 0.214s | 0.749s | 0.729s | 17 | 17 | 21 | 0 |
| `albon_sargeant_race_2024` | 0.380s | 0.228s | 0.264s | 0.264s | 0.482s | 0.472s | 13 | 13 | 13 | 0 |

## Notes

- `Phase 5` columns come from the stored aggregate artifact.
- `Cache current mean` is the current paired race residual recomputed from cache.
- `Broad valid-lap` columns keep the same lap-quality filters but remove same-compound and same-stint-lap pairing.
- `Cache mismatch rows` counts sessions where the fresh current construct does not reproduce the stored Phase 5 delta to within 1ms.
