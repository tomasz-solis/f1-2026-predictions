# Practice Signal Split Diagnostic

- Built at: `2026-05-18T19:02:46.781940+00:00`
- Years: `2022, 2023, 2024, 2025`
- Normal weekends included: `43`
- Weekends excluded: `49`

## Interpretation

- This is a pre-2026 support diagnostic, not the regulation-reset acceptance test.
- The normal-weekend counts below reflect currently cached practice payloads, not the historical season calendar. Low counts in a season can therefore mean missing local FP cache coverage, not only sprint-weekend format.
- On the current cached rows, the split policy does not show a consistent accuracy gain: the row-weighted combined MSE is worse than `shared_long_run`, it wins only two of four combined folds against the best shared policy, and it wins no qualifying fold against the best shared policy.
- Current decision: keep one stored team-strength state in v1 and do not split it into separate short-run and long-run states yet. Separate race and qualifying seconds mappings remain a different, still-valid design choice.
- The decisive transfer-era test is 2026 conventional-weekend evidence under the new regulations. Reopen the state split only if 2026 shows a consistent MSE gain, not a one-off improvement.

## Normal Weekend Coverage By Year

| Year | Normal weekends |
| ---: | ---: |
| 2022 | 18 |
| 2023 | 3 |
| 2024 | 4 |
| 2025 | 18 |

## Coverage Before Common-Row Restriction

| Session kind | Total rows | Balanced | Short run | Long run | Common rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qualifying` | 558 | 558 | 558 | 558 | 558 |
| `race` | 738 | 738 | 738 | 738 | 738 |

## Row-Weighted Policy Summary

| Session kind | Policy | Rows | Weighted MSE | Weighted RMSE |
| --- | --- | ---: | ---: | ---: |
| `combined` | `shared_long_run` | 1296 | 0.5049 | 0.7106 |
| `combined` | `split_short_quali_long_race` | 1296 | 0.5077 | 0.7125 |
| `combined` | `shared_short_run` | 1296 | 0.5235 | 0.7235 |
| `combined` | `shared_balanced` | 1296 | 0.5490 | 0.7410 |
| `qualifying` | `shared_long_run` | 558 | 0.4253 | 0.6522 |
| `qualifying` | `shared_short_run` | 558 | 0.4317 | 0.6570 |
| `qualifying` | `split_short_quali_long_race` | 558 | 0.4317 | 0.6570 |
| `qualifying` | `shared_balanced` | 558 | 0.4594 | 0.6778 |
| `race` | `shared_long_run` | 738 | 0.5651 | 0.7517 |
| `race` | `split_short_quali_long_race` | 738 | 0.5651 | 0.7517 |
| `race` | `shared_short_run` | 738 | 0.5929 | 0.7700 |
| `race` | `shared_balanced` | 738 | 0.6168 | 0.7854 |

## Held-Out Fold Metrics

| Session kind | Holdout | Policy | Profile | Rows | MSE | RMSE | R² |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `combined` | 2022 | `shared_balanced` | `balanced` | 544 | 0.5884 | 0.7671 | 0.3237 |
| `combined` | 2022 | `shared_long_run` | `long_run` | 544 | 0.5525 | 0.7433 | 0.3649 |
| `combined` | 2022 | `shared_short_run` | `short_run` | 544 | 0.5766 | 0.7594 | 0.3372 |
| `combined` | 2022 | `split_short_quali_long_race` | `mixed` | 544 | 0.5568 | 0.7462 | 0.3600 |
| `combined` | 2023 | `shared_balanced` | `balanced` | 94 | 0.3987 | 0.6315 | 0.1982 |
| `combined` | 2023 | `shared_long_run` | `long_run` | 94 | 0.3982 | 0.6310 | 0.1992 |
| `combined` | 2023 | `shared_short_run` | `short_run` | 94 | 0.4143 | 0.6437 | 0.1668 |
| `combined` | 2023 | `split_short_quali_long_race` | `mixed` | 94 | 0.3850 | 0.6205 | 0.2258 |
| `combined` | 2024 | `shared_balanced` | `balanced` | 112 | 0.4316 | 0.6570 | 0.2361 |
| `combined` | 2024 | `shared_long_run` | `long_run` | 112 | 0.3661 | 0.6050 | 0.3521 |
| `combined` | 2024 | `shared_short_run` | `short_run` | 112 | 0.3645 | 0.6038 | 0.3548 |
| `combined` | 2024 | `split_short_quali_long_race` | `mixed` | 112 | 0.3567 | 0.5972 | 0.3687 |
| `combined` | 2025 | `shared_balanced` | `balanced` | 546 | 0.5598 | 0.7482 | 0.1279 |
| `combined` | 2025 | `shared_long_run` | `long_run` | 546 | 0.5043 | 0.7102 | 0.2143 |
| `combined` | 2025 | `shared_short_run` | `short_run` | 546 | 0.5220 | 0.7225 | 0.1868 |
| `combined` | 2025 | `split_short_quali_long_race` | `mixed` | 546 | 0.5108 | 0.7147 | 0.2043 |
| `qualifying` | 2022 | `shared_balanced` | `balanced` | 236 | 0.5151 | 0.7177 | 0.3134 |
| `qualifying` | 2022 | `shared_long_run` | `long_run` | 236 | 0.4827 | 0.6948 | 0.3567 |
| `qualifying` | 2022 | `shared_short_run` | `short_run` | 236 | 0.4926 | 0.7019 | 0.3435 |
| `qualifying` | 2022 | `split_short_quali_long_race` | `short_run` | 236 | 0.4926 | 0.7019 | 0.3435 |
| `qualifying` | 2023 | `shared_balanced` | `balanced` | 42 | 0.3350 | 0.5788 | 0.1068 |
| `qualifying` | 2023 | `shared_long_run` | `long_run` | 42 | 0.3340 | 0.5779 | 0.1095 |
| `qualifying` | 2023 | `shared_short_run` | `short_run` | 42 | 0.3044 | 0.5517 | 0.1884 |
| `qualifying` | 2023 | `split_short_quali_long_race` | `short_run` | 42 | 0.3044 | 0.5517 | 0.1884 |
| `qualifying` | 2024 | `shared_balanced` | `balanced` | 54 | 0.2324 | 0.4821 | 0.3247 |
| `qualifying` | 2024 | `shared_long_run` | `long_run` | 54 | 0.2067 | 0.4546 | 0.3994 |
| `qualifying` | 2024 | `shared_short_run` | `short_run` | 54 | 0.1873 | 0.4327 | 0.4558 |
| `qualifying` | 2024 | `split_short_quali_long_race` | `short_run` | 54 | 0.1873 | 0.4327 | 0.4558 |
| `qualifying` | 2025 | `shared_balanced` | `balanced` | 226 | 0.4784 | 0.6917 | 0.0226 |
| `qualifying` | 2025 | `shared_long_run` | `long_run` | 226 | 0.4346 | 0.6593 | 0.1121 |
| `qualifying` | 2025 | `shared_short_run` | `short_run` | 226 | 0.4502 | 0.6710 | 0.0802 |
| `qualifying` | 2025 | `split_short_quali_long_race` | `short_run` | 226 | 0.4502 | 0.6710 | 0.0802 |
| `race` | 2022 | `shared_balanced` | `balanced` | 308 | 0.6445 | 0.8028 | 0.3213 |
| `race` | 2022 | `shared_long_run` | `long_run` | 308 | 0.6060 | 0.7785 | 0.3618 |
| `race` | 2022 | `shared_short_run` | `short_run` | 308 | 0.6410 | 0.8006 | 0.3250 |
| `race` | 2022 | `split_short_quali_long_race` | `long_run` | 308 | 0.6060 | 0.7785 | 0.3618 |
| `race` | 2023 | `shared_balanced` | `balanced` | 52 | 0.4502 | 0.6710 | 0.2422 |
| `race` | 2023 | `shared_long_run` | `long_run` | 52 | 0.4501 | 0.6709 | 0.2425 |
| `race` | 2023 | `shared_short_run` | `short_run` | 52 | 0.5031 | 0.7093 | 0.1532 |
| `race` | 2023 | `split_short_quali_long_race` | `long_run` | 52 | 0.4501 | 0.6709 | 0.2425 |
| `race` | 2024 | `shared_balanced` | `balanced` | 58 | 0.6170 | 0.7855 | 0.1943 |
| `race` | 2024 | `shared_long_run` | `long_run` | 58 | 0.5144 | 0.7172 | 0.3283 |
| `race` | 2024 | `shared_short_run` | `short_run` | 58 | 0.5296 | 0.7277 | 0.3085 |
| `race` | 2024 | `split_short_quali_long_race` | `long_run` | 58 | 0.5144 | 0.7172 | 0.3283 |
| `race` | 2025 | `shared_balanced` | `balanced` | 320 | 0.6172 | 0.7856 | 0.1719 |
| `race` | 2025 | `shared_long_run` | `long_run` | 320 | 0.5536 | 0.7440 | 0.2573 |
| `race` | 2025 | `shared_short_run` | `short_run` | 320 | 0.5727 | 0.7568 | 0.2316 |
| `race` | 2025 | `split_short_quali_long_race` | `long_run` | 320 | 0.5536 | 0.7440 | 0.2573 |

## Split Versus Best Shared Fold

| Session kind | Holdout | Split MSE | Best shared | Best shared MSE | Delta | Split wins |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `combined` | 2022 | 0.5568 | `shared_long_run` | 0.5525 | 0.0043 | `False` |
| `combined` | 2023 | 0.3850 | `shared_long_run` | 0.3982 | -0.0132 | `True` |
| `combined` | 2024 | 0.3567 | `shared_short_run` | 0.3645 | -0.0078 | `True` |
| `combined` | 2025 | 0.5108 | `shared_long_run` | 0.5043 | 0.0065 | `False` |
| `qualifying` | 2022 | 0.4926 | `shared_long_run` | 0.4827 | 0.0099 | `False` |
| `qualifying` | 2023 | 0.3044 | `shared_short_run` | 0.3044 | 0.0000 | `False` |
| `qualifying` | 2024 | 0.1873 | `shared_short_run` | 0.1873 | 0.0000 | `False` |
| `qualifying` | 2025 | 0.4502 | `shared_long_run` | 0.4346 | 0.0156 | `False` |
| `race` | 2022 | 0.6060 | `shared_long_run` | 0.6060 | 0.0000 | `False` |
| `race` | 2023 | 0.4501 | `shared_long_run` | 0.4501 | 0.0000 | `False` |
| `race` | 2024 | 0.5144 | `shared_long_run` | 0.5144 | 0.0000 | `False` |
| `race` | 2025 | 0.5536 | `shared_long_run` | 0.5536 | 0.0000 | `False` |
