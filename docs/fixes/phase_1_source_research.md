# Phase 1 Source Research Notes

Date: 2026-05-12
Status: research pass complete; scratch notes only; not validation evidence

This file collects candidate links and reading notes from the Phase 1 source
research pass.
It is not a validation artifact. Nothing here counts toward the hard evidence
table in `docs/fixes/teammate_network_prior_validation_evidence.md`.

Outcome note: on 2026-05-12, Motorsport.com / Motorsport-Total PACETEQ
teammate pace articles were accepted as countable source-family evidence, with
the limits recorded in
`docs/fixes/teammate_network_prior_validation_evidence.md`.

Rules for this scratch file:

- candidate links may be surfaced during research;
- hard validation rows must live in the validation evidence doc;
- F1Metrics-style sources are supplemental only, never hard evidence;
- rejected links should stay here with a short reason, so the same source is
  not re-checked later;
- downstream docs, fit artifacts, and replay output must not cite this file as
  validation evidence.

## Candidate Rows To Research

### Race Pace

- `verstappen_perez_race_2022`: Red Bull 2022 race pace, VER over PER.
  Candidate source: Motorsport.com / PACETEQ Perez trend. Candidate value:
  VER +0.234s/lap.
- `verstappen_perez_race_2023`: Red Bull 2023 race pace, VER over PER.
  Candidate source: Motorsport.com / PACETEQ 2023 review. Candidate value:
  VER +0.451s/lap.
- `verstappen_perez_race_2024`: Red Bull 2024 race pace, VER over PER.
  Candidate source: Motorsport-Total / PACETEQ Red Bull duel. Candidate
  value: VER +0.56s/lap.
- `alonso_stroll_race_2023`: Aston Martin 2023 race pace, ALO over STR.
  Candidate source: Motorsport.com / PACETEQ 2023 review. Candidate value:
  ALO +0.486s/lap.
- `alonso_stroll_race_2024`: Aston Martin 2024 race pace, ALO over STR.
  Candidate source: Motorsport-Total / PACETEQ Aston Martin duel. Candidate
  value: ALO +0.25s/lap.
- `albon_sargeant_race_2023_2024`: split recommended. Candidate split values:
  2023 ALB +0.293s/lap and 2024 ALB +0.38s/lap.
- `russell_latifi_race_2022`: cut candidate. The row is impossible as written
  because Russell drove for Mercedes in 2022.
- `bottas_zhou_race_2022`: no clean numeric race-pace source found.
- `bottas_zhou_race_2023`: counterevidence found. Motorsport.com / PACETEQ
  reports ZHO +0.013s/lap.
- `bottas_zhou_race_2024`: supplemental only. Motorsport-Total / PACETEQ
  reports BOT +0.01s/lap, which is near noise.

### Qualifying Pace

- `verstappen_perez_quali_2022`: Red Bull 2022 qualifying, VER over PER.
  Candidate source: Motorsport.com / PACETEQ Perez trend. Candidate value:
  VER +0.290s.
- `verstappen_perez_quali_2023`: Red Bull 2023 qualifying, VER over PER.
  Candidate sources: Motorsport.com / PACETEQ 2023 review and RacingNews365.
  Values disagree: PACETEQ 0.621s and RacingNews365 0.495s.
- `verstappen_perez_quali_2024`: Red Bull 2024 qualifying, VER over PER.
  Candidate source: Motorsport-Total / PACETEQ Red Bull duel. Candidate
  full-season value: VER +0.66s. Motor Sport Magazine mid-season value:
  0.486s.
- `russell_hamilton_quali_2024`: Mercedes 2024 qualifying, RUS over HAM.
  Candidate source: Motorsport-Total / PACETEQ Mercedes duel. Candidate
  full-season value: RUS +0.23s. Motor Sport Magazine mid-season value:
  0.098s.
- `albon_sargeant_quali_2023_2024`: split recommended. Candidate split
  values: 2023 ALB +0.522s and 2024 ALB +0.66s.

## Researched Candidate Links

### Motorsport.com / PACETEQ 2023 Team-Mate Pace Review

Source:
https://lat.motorsport.com/f1/news/verstappen-checo-perez-diferencia-f1-2023/10561671/

Potential rows:

- `verstappen_perez_race_2023`
- `verstappen_perez_quali_2023`
- `alonso_stroll_race_2023`
- `albon_sargeant_race_2023_2024` as partial 2023 evidence only
- `bottas_zhou_race_2023` as a likely counterexample to the current expected
  relationship

Notes:

- The article says Motorsport.com used its data partner to compile qualifying
  speed and race pace for team-mates.
- It states only representative laps were included, with obvious invalid cases
  removed.
- Reported 2023 qualifying gaps include VER over PER by 0.621s, ALO over STR by
  0.746s, ALB over SAR by 0.522s, and BOT over ZHO by 0.392s.
- Reported 2023 race-pace gaps include VER over PER by 0.451s, ALO over STR by
  0.486s, and ALB over SAR by 0.293s.
- Important counterpoint: it reports ZHO over BOT by 0.013s in 2023 race pace.
  That conflicts with the validation doc's current expected relationship for
  `bottas_zhou_race_2023`.
- Preliminary assessment: accepted later as HARD-capable under the
  Motorsport / PACETEQ source-family rule.

### Motorsport.com / PACETEQ Perez vs Verstappen 2021-2024 Trend

Source:
https://lat.motorsport.com/f1/news/checo-perez-diferencia-verstappen-f1-2024/10627633/

Potential rows:

- `verstappen_perez_race_2022`
- `verstappen_perez_race_2023`
- `verstappen_perez_race_2024` as partial 2024 evidence only
- `verstappen_perez_quali_2022`
- `verstappen_perez_quali_2023`
- `verstappen_perez_quali_2024` as partial 2024 evidence only

Notes:

- Reports Perez deficits to Verstappen by year: 2022 qualifying 0.290s and race
  0.234s; 2023 qualifying 0.621s and race 0.451s; 2024 through the Spanish GP
  qualifying 0.609s and race 0.471s.
- Also reports early-2024 all-team mate gaps through that point, including STR
  behind ALO by 0.20s qualifying / 0.29s race, SAR behind ALB by 0.56s
  qualifying / 0.40s race, and ZHO behind BOT by 0.51s qualifying / 0.06s race.
- Preliminary assessment: useful for candidate discovery. The 2024 rows are
  not full-season evidence.

### Motorsport.com / PACETEQ 2024 Team-Mate Pace Review

Source:
https://nl.motorsport.com/f1/news/dit-zeggen-data-over-max-verstappen-sergio-perez-teamgenoten-f1-2024/10685457/

Potential rows:

- `verstappen_perez_race_2024`
- `verstappen_perez_quali_2024`
- `alonso_stroll_race_2024` if the underlying chart exposes exact values
- `bottas_zhou_race_2024` if the underlying chart exposes exact values
- `russell_hamilton_quali_2024`
- `albon_sargeant_quali_2023_2024` as partial 2024 evidence only

Notes:

- The article says the chart covers the full 2024 season and shows both
  qualifying averages and race-lap averages.
- Exact values visible in text: PER behind VER by 0.66s in qualifying and
  0.56s in race pace; SAR behind ALB by 0.66s in qualifying; HAM behind RUS by
  0.23s in qualifying.
- It says ZHO had almost no race deficit to BOT, but the exact full-season race
  value is not visible in the text extract.
- Preliminary assessment: promising, but do not use values that are only in an
  image unless the chart value is directly recorded.

### Motorsport-Total / PACETEQ 2024 Red Bull Team Duel

Source:
https://www.motorsport-total.com/formel-1/news/maximal-ueberlegen-wie-verstappen-perez-2024-in-grund-und-boden-fuhr-24122902

Potential rows:

- `verstappen_perez_race_2024`
- `verstappen_perez_quali_2024`

Notes:

- Reports VER vs PER full-season 2024 qualifying duel 23-1 and qualifying pace
  -0.66s from Verstappen's perspective.
- Reports race duel 23-1 and race pace -0.56s per lap from Verstappen's
  perspective.
- Method text says the article uses PACETEQ data to compare F1 team-mates.
- Preliminary assessment: better than the earlier all-team chart source for
  Red Bull 2024 because exact values are in article text.

### Motorsport-Total / PACETEQ 2024 Aston Martin Team Duel

Source:
https://www.motorsport-total.com/formel-1/news/analyse-ist-lance-stroll-wirklich-zu-langsam-24122701

Potential rows:

- `alonso_stroll_race_2024`

Notes:

- Reports ALO vs STR full-season 2024 qualifying duel 19-5 and qualifying pace
  -0.35s from Alonso's perspective.
- Reports race duel 18-6 and race pace -0.25s per lap from Alonso's
  perspective.
- Also reports tire degradation: Alonso 0.076s/lap, Stroll 0.089s/lap.
- Preliminary assessment: usable candidate for a 2024 Alonso-Stroll row if
  PACETEQ source quality is accepted. It also suggests the race threshold should
  account for tire degradation being part of the reported race-pace gap.

### Motorsport-Total / PACETEQ 2024 Mercedes Team Duel

Source:
https://www.motorsport-total.com/formel-1/news/mercedes-fahrer-analysiert-hat-lewis-hamilton-seine-qualifyingpace-verloren-24122802

Potential rows:

- `russell_hamilton_quali_2024`

Notes:

- Reports RUS vs HAM full-season 2024 qualifying duel 19-5 and qualifying pace
  -0.23s from Russell's perspective.
- Reports race duel 15-9 and race pace -0.09s per lap from Russell's
  perspective.
- Reports tire degradation: Russell 0.070s/lap, Hamilton 0.074s/lap.
- Preliminary assessment: strong candidate for the `russell_hamilton_quali_2024`
  row if PACETEQ is accepted. It also gives a race value, but the current
  validation table only has a qualifying row for this pairing.

### Motorsport-Total / PACETEQ 2024 Williams Team Duel

Source:
https://www.motorsport-total.com/formel-1/news/nach-sargeant-rauswurf-so-viel-schneller-war-franco-colapinto-wirklich-24122307

Potential rows:

- `albon_sargeant_quali_2023_2024` as partial 2024 evidence only
- Possible replacement row: `albon_sargeant_race_2024`

Notes:

- Reports ALB vs SAR through Sargeant's 15-race 2024 sample: qualifying duel
  13-0, qualifying pace -0.66s, race duel 11-2, race pace -0.38s per lap.
- Also reports ALB vs COL after the driver change: qualifying pace -0.11s, race
  pace +0.02s per lap.
- Preliminary assessment: good exact source for 2024 Albon-Sargeant, but it
  does not support the current two-season `albon_sargeant_quali_2023_2024` row
  by itself. A cleaner validation row may be 2024-only.

### Motorsport-Total / PACETEQ 2024 Sauber Team Duel

Source:
https://www.motorsport-total.com/formel-1/news/sauber-duell-das-war-2024-die-ganz-grosse-schwaeche-von-valtteri-bottas-24122202

Potential rows:

- `bottas_zhou_race_2024`

Notes:

- Reports ZHO vs BOT full-season 2024 qualifying duel 3-21 and qualifying pace
  +0.63s from Zhou's perspective, meaning Bottas was faster by 0.63s.
- Reports race duel 10-14 and race pace +0.01s per lap from Zhou's perspective,
  meaning Bottas was faster by only 0.01s/lap.
- Reports tire degradation: Bottas 0.095s/lap, Zhou 0.078s/lap.
- Preliminary assessment: supports the direction of `bottas_zhou_race_2024`,
  but the race threshold would need to be tiny. This is a weak hard-check
  candidate because a 0.01s/lap threshold is close to noise and depends heavily
  on method.

### Sports of the Day / PACETEQ 2024 Red Bull Duel

Source:
https://www.sportsoftheday.com/maximum-advantage-how-verstappen-outdrove-perez-in-2024/

Potential rows:

- `verstappen_perez_race_2024`
- `verstappen_perez_quali_2024`

Notes:

- Reports VER vs PER qualifying duel 23-1, qualifying pace -0.66s, race duel
  23-1, and race pace -0.56s per lap.
- The article attributes the numbers to PACETEQ.
- Preliminary assessment: corroborative only. Motorsport.com appears
  preferable for the same PACETEQ numbers.

### Sports of the Day / PACETEQ 2024 Sauber Duel

Source:
https://www.sportsoftheday.com/sauber-duel-that-was-valtteri-bottass-big-weakness-in-2024/

Potential rows:

- `bottas_zhou_race_2024`

Notes:

- Reports BOT over ZHO by 0.63s in qualifying pace and 0.01s per lap in race
  pace.
- The article also says BOT's race pace was nearly equal to ZHO because of
  higher tire wear.
- Preliminary assessment: useful warning that any 2024 BOT-ZHO hard threshold
  should be very small if accepted at all.

### Motor Sport Magazine Mid-2024 Comparable Qualifying Gaps

Source:
https://www.motorsportmagazine.com/articles/single-seaters/f1/mph-the-surprise-findings-when-you-compare-f1-team-mates-qualifying-times/

Potential rows:

- `verstappen_perez_quali_2024` as partial 2024 evidence only
- `russell_hamilton_quali_2024` as partial 2024 evidence only
- `albon_sargeant_quali_2023_2024` as partial 2024 evidence only

Notes:

- Published after the 2024 Hungarian GP, so it is a mid-season source, not a
  full-season source.
- Reports comparable-session qualifying gaps: VER over PER by 0.486s, BOT over
  ZHO by 0.403s, ALB over SAR by 0.257s, ALO over STR by 0.199s, and RUS over
  HAM by 0.098s.
- Method notes are stronger than many sources: it excludes sessions with car
  spec differences, wet randomness, and cases where team-mates are compared
  across different qualifying stages.
- Preliminary assessment: good methodology candidate, but partial-season scope
  limits its direct use for full-season rows.

### RacingNews365 2023 Qualifying Gaps

Source:
https://racingnews365.com/qualifying-guns-verstappen-slaughters-perez-hamilton-and-russell-match-up

Potential rows:

- `verstappen_perez_quali_2023`
- `albon_sargeant_quali_2023_2024` as partial 2023 evidence only

Notes:

- Reports 2023 average qualifying gaps including VER over PER by 0.495s, ALO
  over STR by 0.440s, ALB over SAR by 0.560s, and HAM over RUS by 0.017s.
- These differ from Motorsport.com / PACETEQ for several pairings, especially
  VER-PER and ALO-STR.
- Preliminary assessment: useful corroboration for direction, but probably not
  HARD without clearer filtering rules. The disagreement itself should be
  preserved as a warning before thresholds are chosen.

### ESPN 2022 Driver Rankings

Source:
https://www.espn.com/f1/story/_/id/35162875

Potential rows:

- Context only for 2022 qualifying relationships.
- Possible replacement context for `albon_sargeant_race_2023_2024` if split to
  Albon-Latifi or Albon-Sargeant qualitative checks, but not hard race evidence.

Notes:

- Reports ALB over LAT by 0.615s average qualifying advantage in 2022.
- Reports BOT vs ZHO qualifying record 14-8 but does not give a Bottas-Zhou
  seconds gap in the visible text.
- Does not publish teammate race-pace seconds deltas.
- Preliminary assessment: not enough for hard race validation. It is useful
  mainly because it confirms the `russell_latifi_race_2022` row is impossible:
  Latifi's 2022 comparison was Albon, not Russell.

### RacingNews365 2021 Russell-Latifi Qualifying Gap

Source:
https://racingnews365.com/find-out-who-topped-the-head-to-head-qualifying-battles-in-2021

Potential rows:

- Possible replacement row only, if the project wants a Russell-Latifi
  qualifying check from 2021 rather than the impossible 2022 race row.

Notes:

- Reports RUS over LAT in 2021 qualifying by 0.393s on average, with a 20-2
  head-to-head.
- This does not support the current `russell_latifi_race_2022` row because it
  is qualifying, not race pace, and it is 2021, not 2022.
- Preliminary assessment: useful if replacing the impossible row, but not a
  drop-in substitute.

### Motor Sport Magazine 2021 Team-Mate Battles

Source:
https://www.motorsportmagazine.com/articles/single-seaters/f1/f1-2021-team-mate-battles-star-drivers-underperformers-and-ones-to-watch-in-2022/

Potential rows:

- Possible replacement context for a Russell-Latifi 2021 race or qualifying
  discussion.

Notes:

- Says Russell outperformed Latifi in both race and qualifying trim, but the
  race gap was less pronounced than qualifying.
- Does not provide a numeric seconds-per-lap race threshold in the visible
  text.
- Preliminary assessment: qualitative context only; not hard evidence.

### Formula1.com Verstappen Team-Mate Head-To-Head Context

Source:
https://www.formula1.com/en/latest/article/in-numbers-how-verstappens-team-mates-fared-against-him-with-lawson-the.2A7zHcadkyis7TvNGLnbmm

Potential rows:

- Context only for Verstappen-Perez rows.

Notes:

- Gives official head-to-head, points, poles, podiums, and wins for Verstappen
  and Perez across 2022 and 2023.
- Does not publish seconds-per-lap deltas.
- Preliminary assessment: not usable as a hard threshold source, but useful
  context for sanity-checking direction and sample size.

## Current Blocking Findings

- `russell_latifi_race_2022` should be cut or replaced. It is not merely
  unsourced; it is factually impossible because Russell and Latifi were not
  team-mates in 2022.
- `bottas_zhou_race_2023` should not be assumed directionally safe. The
  Motorsport.com / PACETEQ 2023 article reports Zhou ahead by 0.013s/lap in
  race pace.
- `bottas_zhou_race_2024` is sourceable in the expected direction, but the
  reported race-pace gap is only 0.01s/lap, so it may be too small for a useful
  hard validation threshold.

## Final Row Disposition

This is the end state of the Phase 1 research pass for the current candidate
list. The validation evidence doc is the source of truth; this scratch file is
research history only.

### Race Pace Rows

- HARD rows promoted in the validation doc:
  `verstappen_perez_race_2022`, `verstappen_perez_race_2023`,
  `verstappen_perez_race_2024`, `alonso_stroll_race_2023`,
  `alonso_stroll_race_2024`, `albon_sargeant_race_2023`,
  `albon_sargeant_race_2024`.
- Replaced by season-specific rows:
  `albon_sargeant_race_2023_2024`.
- Cut:
  `russell_latifi_race_2022`, `bottas_zhou_race_2022`,
  `bottas_zhou_race_2023`.
- Supplemental only:
  `bottas_zhou_race_2024`, because the source reports only BOT +0.01s/lap.

### Qualifying Pace Rows

- HARD rows promoted in the validation doc:
  `verstappen_perez_quali_2022`, `verstappen_perez_quali_2023`,
  `verstappen_perez_quali_2024`, `russell_hamilton_quali_2024`,
  `albon_sargeant_quali_2023`, `albon_sargeant_quali_2024`.
- Replaced by season-specific rows:
  `albon_sargeant_quali_2023_2024`.

## Research Completion Standard

The current Phase 1 source-discovery pass is complete for the candidate list
above. Every row now has one of these outcomes:

- a numeric source candidate that was promoted after source-family acceptance;
- a specific split/replacement recommendation;
- a cut recommendation because the row is factually wrong, directionally
  contradicted, too close to zero, or unsupported by a defensible numeric
  source.

No further web research is required for the current candidate list. The
remaining work is validation execution, not source discovery.

With PACETEQ accepted for HARD evidence, the hard set from this pass is:

- Race pace: VER-PER 2022, VER-PER 2023, VER-PER 2024, ALO-STR 2023, ALO-STR
  2024.
- Qualifying pace: VER-PER 2022, VER-PER 2023, VER-PER 2024, RUS-HAM 2024.

## Source Quality Notes

PACETEQ-backed Motorsport.com / Motorsport-Total articles are the strongest
single source family found so far for race-pace numbers. They publish numeric
seconds deltas and say they use representative laps, but they do not expose the
full lap list, stint filters, or code. The validation doc records the accepted
limits of using this source family.

Motor Sport Magazine's 2024 qualifying comparison has clearer filtering rules:
it excludes wet sessions, obvious non-representative laps, car-spec differences,
and cross-segment comparisons where track evolution would skew the gap. The
limitation is scope: it was published after Hungary 2024, so it is a
partial-season source.

RacingNews365's 2023 qualifying article is useful corroboration for direction,
but the visible methodology is thinner than Motor Sport Magazine's. It also
disagrees materially with PACETEQ for some gaps, especially Verstappen-Perez and
Alonso-Stroll qualifying in 2023.

Formula1.com and ESPN context pieces help sanity-check team-mate relationships,
but they do not publish race-pace seconds-per-lap thresholds. They should not be
promoted as hard validation evidence.

## Promoted Candidates

These rows were promoted because they have numeric values, match the intended
construct, and are not near zero:

- `verstappen_perez_race_2022`: VER +0.234s/lap.
- `verstappen_perez_quali_2022`: VER +0.290s.
- `verstappen_perez_race_2023`: VER +0.451s/lap.
- `verstappen_perez_quali_2023`: VER +0.621s, with RacingNews365 conflict at
  0.495s noted before threshold choice.
- `verstappen_perez_race_2024`: VER +0.56s/lap.
- `verstappen_perez_quali_2024`: VER +0.66s.
- `alonso_stroll_race_2023`: ALO +0.486s/lap.
- `alonso_stroll_race_2024`: ALO +0.25s/lap.
- `russell_hamilton_quali_2024`: RUS +0.23s, with Motor Sport Magazine
  mid-season corroboration at 0.098s.

These needed row changes before promotion:

- Split `albon_sargeant_race_2023_2024` into season-specific rows. Candidate
  values found: 2023 ALB +0.293s/lap, 2024 ALB +0.38s/lap.
- Split `albon_sargeant_quali_2023_2024` into season-specific rows. Candidate
  values found: 2023 ALB +0.522s, 2024 ALB +0.66s.
- Cut `russell_latifi_race_2022` or replace it with a correct pairing/year.
- Cut or reframe `bottas_zhou_race_2023`.
- Keep `bottas_zhou_race_2024` supplemental only.

## Working Coverage Snapshot

The strongest rows from this scratch pass are now recorded in
`docs/fixes/teammate_network_prior_validation_evidence.md` Section 3.

Rows still needing better independent evidence:

- `bottas_zhou_race_2022`: no clean numeric race-pace source found in the
  current pass.
- `russell_latifi_race_2022`: cannot be fixed by sourcing; the row is wrong.
- two-season Albon-Sargeant rows: separate 2023/2024 values are easier to
  support than the current aggregate.

No open source-research decisions remain for the current validation set.
