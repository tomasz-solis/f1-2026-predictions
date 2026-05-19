# Team-Strength Seconds Mapping

Built at: `2026-05-19T16:37:02.358552+00:00`
Model version: `2.1`
Policy: `same_session_construct`
Stored state: `single_team_strength_scalar`

Freeze separate race and qualifying seconds mappings over one stored team_strength scalar; do not split short-run and long-run state.

Positive seconds mean a faster-than-field team contribution.

## Frozen mappings

| Session | Intercept (s) | Slope (s/unit) | Training years |
| --- | ---: | ---: | --- |
| `race` | -0.033571 | 1.970772 | 2022, 2023, 2024, 2025 |
| `qualifying` | -0.173160 | 1.774169 | 2022, 2023, 2024, 2025 |

## Validation fold summary

| Session | Holdout season | Rows | RMSE (s) | Prediction slope |
| --- | ---: | ---: | ---: | ---: |
| `race` | 2022 | 376 | 0.921 | 0.993 |
| `race` | 2023 | 384 | 0.943 | 0.791 |
| `race` | 2024 | 418 | 0.537 | 0.863 |
| `race` | 2025 | 426 | 1.076 | 0.982 |
| `qualifying` | 2022 | 304 | 0.726 | 0.969 |
| `qualifying` | 2023 | 334 | 1.503 | 1.450 |
| `qualifying` | 2024 | 332 | 0.462 | 0.701 |
| `qualifying` | 2025 | 316 | 0.541 | 0.606 |
