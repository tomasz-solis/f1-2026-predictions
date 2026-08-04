# Team-Strength Seconds Mapping Candidate Diagnostics

Built at: `2026-08-03T21:40:34.859243+00:00`

## Coverage

| Policy | Session kind | Usable rows | Total rows |
| --- | --- | ---: | ---: |
| `same_session_construct` | `qualifying` | 1467 | 1467 |
| `same_session_construct` | `race` | 1797 | 1797 |
| `race_event_shared_scalar` | `qualifying` | 1300 | 1467 |
| `race_event_shared_scalar` | `race` | 1797 | 1797 |
| `race_season_mean_shared_scalar` | `qualifying` | 1467 | 1467 |
| `race_season_mean_shared_scalar` | `race` | 1797 | 1797 |
| `race_trailing_mean_shared_scalar` | `qualifying` | 1235 | 1467 |
| `race_trailing_mean_shared_scalar` | `race` | 1696 | 1797 |

## `same_session_construct`

| Session kind | Holdout season | Rows | Intercept | Slope | R² | Prediction slope | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `race` | 2022 | 376 | -0.061 | 1.911 | 0.438 | 0.993 | 0.921 |
| `race` | 2023 | 384 | -0.009 | 2.038 | 0.318 | 0.791 | 0.943 |
| `race` | 2024 | 418 | -0.026 | 1.995 | 0.625 | 0.863 | 0.537 |
| `race` | 2025 | 426 | -0.038 | 1.939 | 0.315 | 0.982 | 1.076 |
| `qualifying` | 2022 | 304 | -0.178 | 1.736 | 0.518 | 0.969 | 0.726 |
| `qualifying` | 2023 | 334 | -0.137 | 1.480 | 0.256 | 1.450 | 1.503 |
| `qualifying` | 2024 | 332 | -0.189 | 1.923 | 0.550 | 0.701 | 0.462 |
| `qualifying` | 2025 | 316 | -0.189 | 1.957 | 0.304 | 0.606 | 0.541 |

## `race_event_shared_scalar`

| Session kind | Holdout season | Rows | Intercept | Slope | R² | Prediction slope | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `race` | 2022 | 376 | -0.061 | 1.911 | 0.438 | 0.993 | 0.921 |
| `race` | 2023 | 384 | -0.009 | 2.038 | 0.318 | 0.791 | 0.943 |
| `race` | 2024 | 418 | -0.026 | 1.995 | 0.625 | 0.863 | 0.537 |
| `race` | 2025 | 426 | -0.038 | 1.939 | 0.315 | 0.982 | 1.076 |
| `qualifying` | 2022 | 270 | -0.241 | 1.080 | 0.277 | 0.960 | 0.895 |
| `qualifying` | 2023 | 294 | -0.188 | 0.963 | 0.114 | 1.346 | 1.710 |
| `qualifying` | 2024 | 296 | -0.263 | 1.175 | 0.350 | 0.776 | 0.571 |
| `qualifying` | 2025 | 288 | -0.266 | 1.201 | 0.177 | 0.644 | 0.603 |

## `race_season_mean_shared_scalar`

| Session kind | Holdout season | Rows | Intercept | Slope | R² | Prediction slope | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `race` | 2022 | 376 | -0.061 | 1.678 | 0.280 | 0.972 | 1.043 |
| `race` | 2023 | 384 | -0.009 | 1.893 | 0.126 | 0.681 | 1.067 |
| `race` | 2024 | 418 | -0.026 | 1.728 | 0.432 | 0.882 | 0.661 |
| `race` | 2025 | 426 | -0.038 | 1.702 | 0.166 | 0.962 | 1.187 |
| `qualifying` | 2022 | 304 | -0.238 | 1.525 | 0.305 | 0.878 | 0.872 |
| `qualifying` | 2023 | 334 | -0.197 | 1.265 | 0.137 | 1.327 | 1.618 |
| `qualifying` | 2024 | 332 | -0.263 | 1.718 | 0.287 | 0.662 | 0.582 |
| `qualifying` | 2025 | 316 | -0.257 | 1.620 | 0.233 | 0.655 | 0.567 |

## `race_trailing_mean_shared_scalar`

| Session kind | Holdout season | Rows | Intercept | Slope | R² | Prediction slope | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `race` | 2022 | 356 | -0.076 | 1.520 | 0.243 | 0.974 | 1.088 |
| `race` | 2023 | 364 | -0.033 | 1.717 | 0.110 | 0.672 | 1.097 |
| `race` | 2024 | 398 | -0.037 | 1.519 | 0.416 | 0.925 | 0.676 |
| `race` | 2025 | 406 | -0.054 | 1.537 | 0.146 | 0.967 | 1.227 |
| `qualifying` | 2022 | 258 | -0.278 | 1.605 | 0.289 | 0.867 | 0.902 |
| `qualifying` | 2023 | 282 | -0.216 | 1.262 | 0.144 | 1.507 | 1.712 |
| `qualifying` | 2024 | 284 | -0.304 | 1.811 | 0.287 | 0.654 | 0.606 |
| `qualifying` | 2025 | 274 | -0.293 | 1.708 | 0.231 | 0.654 | 0.595 |
