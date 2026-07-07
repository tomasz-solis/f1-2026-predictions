# DNF Calibration Probe - 2026

- Transform: `p' = lambda * p + (1 - lambda) * expanding_prior_base_rate`
- Seed base rate (first event only): **0.1**
- Events scored: **13** (286 driver observations, 11 DNFs)
- Best lambda by pooled Brier: **0.25** (1.00 = current model output, 0.00 = base-rate only)

## Pooled Brier by lambda

| Lambda | Pooled Brier |
|---|---:|
| `0.00` | 0.038354 |
| `0.25` | 0.036979 |
| `0.50` | 0.037744 |
| `0.75` | 0.040647 |
| `1.00` (current) | 0.04569 |

## Per-event Brier

| Race | Target | Checkpoint | Drivers | DNFs | Prior rate | λ=0.00 | λ=0.25 | λ=0.50 | λ=0.75 | λ=1.00 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Australian Grand Prix | grand_prix_race | FP3 | 22 | 0 | 0.1 | 0.01 | 0.010677 | 0.011616 | 0.012818 | 0.014282 |
| Chinese Grand Prix | sprint_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.001223 | 0.004891 | 0.011005 | 0.019564 |
| Chinese Grand Prix | grand_prix_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.001008 | 0.004033 | 0.009074 | 0.016132 |
| Japanese Grand Prix | grand_prix_race | FP2 | 22 | 0 | 0.0 | 0.0 | 0.000948 | 0.003792 | 0.008532 | 0.015168 |
| Miami Grand Prix | sprint_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.000977 | 0.003907 | 0.008791 | 0.015629 |
| Miami Grand Prix | grand_prix_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.000927 | 0.003708 | 0.008343 | 0.014832 |
| Canadian Grand Prix | sprint_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.001154 | 0.004616 | 0.010387 | 0.018465 |
| Canadian Grand Prix | grand_prix_race | SQ | 22 | 0 | 0.0 | 0.0 | 0.001326 | 0.005306 | 0.011938 | 0.021223 |
| Monaco Grand Prix | grand_prix_race | FP3 | 22 | 0 | 0.0 | 0.0 | 0.001442 | 0.005768 | 0.012978 | 0.023072 |
| Barcelona Grand Prix | grand_prix_race | FP3 | 22 | 5 | 0.0 | 0.227273 | 0.207571 | 0.190873 | 0.17718 | 0.166492 |
| Austrian Grand Prix | grand_prix_race | Q | 22 | 4 | 0.022727 | 0.17407 | 0.165936 | 0.160376 | 0.157391 | 0.156981 |
| British Grand Prix | sprint_race | SQ | 22 | 0 | 0.03719 | 0.001383 | 0.004406 | 0.009352 | 0.016223 | 0.025018 |
| British Grand Prix | grand_prix_race | Q | 22 | 2 | 0.034091 | 0.085873 | 0.083138 | 0.082433 | 0.083756 | 0.087108 |

Notes: the expanding base rate uses prior completed events only (leakage-safe), so lambda=0.00 here is a deployable forecast, unlike the per-event oracle baseline in the evaluation report. Changing the emitted probability is an output-layer decision; the Monte Carlo DNF sampling input is out of scope (see docs/MODEL_PROMOTION.md).
