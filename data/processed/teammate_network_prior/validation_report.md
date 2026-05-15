# Teammate-Network Prior Validation Report

Built at: `2026-05-14T08:27:37.532779+00:00`

## Summary

- HARD race checks: 2/7
- HARD qualifying checks: 1/6
- All HARD checks passed: `false`
- Failed HARD checks: `verstappen_perez_race_2023`, `verstappen_perez_race_2024`, `alonso_stroll_race_2023`, `albon_sargeant_race_2023`, `albon_sargeant_race_2024`, `verstappen_perez_quali_2023`, `verstappen_perez_quali_2024`, `russell_hamilton_quali_2024`, `albon_sargeant_quali_2023`, `albon_sargeant_quali_2024`

## HARD Race Checks

| Check | Source | Threshold | Fitted delta | Scope direct | All-year direct | Status | Diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `verstappen_perez_race_2022` | Motorsport.com / PACETEQ Perez trend | 0.234s | 0.421s | 0.281s | 0.421s | PASS | passed |
| `verstappen_perez_race_2023` | Motorsport.com / PACETEQ 2023 review | 0.451s | 0.421s | 0.295s | 0.421s | FAIL | matched-lap direct delta below source threshold |
| `verstappen_perez_race_2024` | Motorsport-Total / PACETEQ Red Bull duel | 0.560s | 0.421s | 0.624s | 0.421s | FAIL | pooled prior below source-scope direct delta |
| `alonso_stroll_race_2023` | Motorsport.com / PACETEQ 2023 review | 0.486s | 0.291s | 0.413s | 0.296s | FAIL | matched-lap direct delta below source threshold |
| `alonso_stroll_race_2024` | Motorsport-Total / PACETEQ Aston Martin duel | 0.250s | 0.291s | 0.223s | 0.296s | PASS | passed |
| `albon_sargeant_race_2023` | Motorsport.com / PACETEQ 2023 review | 0.293s | 0.249s | 0.257s | 0.249s | FAIL | matched-lap direct delta below source threshold |
| `albon_sargeant_race_2024` | Motorsport-Total / PACETEQ Williams duel | 0.380s | 0.249s | 0.228s | 0.249s | FAIL | matched-lap direct delta below source threshold |

## HARD Qualifying Checks

| Check | Source | Threshold | Fitted delta | Scope direct | All-year direct | Status | Diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `verstappen_perez_quali_2022` | Motorsport.com / PACETEQ Perez trend | 0.290s | 0.396s | 0.361s | 0.396s | PASS | passed |
| `verstappen_perez_quali_2023` | Motorsport.com / PACETEQ 2023 review | 0.621s | 0.396s | 0.363s | 0.396s | FAIL | matched-lap direct delta below source threshold |
| `verstappen_perez_quali_2024` | Motorsport-Total / PACETEQ Red Bull duel | 0.660s | 0.396s | 0.462s | 0.396s | FAIL | matched-lap direct delta below source threshold |
| `russell_hamilton_quali_2024` | Motorsport-Total / PACETEQ Mercedes duel | 0.230s | 0.058s | 0.113s | 0.058s | FAIL | matched-lap direct delta below source threshold |
| `albon_sargeant_quali_2023` | Motorsport.com / PACETEQ 2023 review | 0.522s | 0.356s | 0.412s | 0.356s | FAIL | matched-lap direct delta below source threshold |
| `albon_sargeant_quali_2024` | Motorsport-Total / PACETEQ Williams duel | 0.660s | 0.356s | 0.222s | 0.356s | FAIL | matched-lap direct delta below source threshold |

## Supplemental Checks

| Check | Source | Threshold | Fitted delta | Scope direct | All-year direct | Status | Diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `bottas_zhou_race_2024` | Motorsport-Total / PACETEQ Sauber duel | 0.010s | 0.203s | 0.184s | 0.203s | PASS | passed |

## Cut Checks

- `russell_latifi_race_2022`: CUT_IMPOSSIBLE_PAIRING_YEAR - Russell drove for Mercedes in 2022; Latifi's Williams teammate was Albon.
- `bottas_zhou_race_2022`: CUT_NO_NUMERIC_RACE_SOURCE - No defensible numeric race-pace source was found.
- `bottas_zhou_race_2023`: CUT_DIRECTION_CONFLICT - Accepted source reports Zhou slightly faster, conflicting with the candidate.
- `tsunoda_devries_race_2023`: SMOKE_ONLY - Sample too thin for source-backed validation.
- `leclerc_sainz_quali_2022_2024`: SMOKE_ONLY - Contested and hedged source base; excluded from HARD validation.

## Notes

- Qualifying validation has fewer HARD rows than race validation; the artifact keeps a wider qualifying sigma floor and later replay diagnostics must stay strict.
- Direction-only smoke checks are excluded from the HARD pass count and belong in tests.
- Direct-pair diagnostics use the same aggregate matched-lap rows and WLS weights as the prior fit. They are diagnostic only; the locked pass rule is still the fitted driver-prior delta.
