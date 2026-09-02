> **Status: shelved research, not production.** This documents the challenger
> framework that was shelved on 2026-07-29. The implementation it references —
> `src/analysis/challenger_*`, `src/models/qualifying_practice_*`,
> `scripts/run_challenger_*`, and their tests — lives on the branch
> `shelved/challenger-research`, not on `master`, so the paths below only
> resolve there. The framework does not run against current production code
> (`predict_qualifying` no longer accepts `include_grid_scenarios`). Measured
> verdicts live in `docs/MODEL_LEDGER.md`. Kept for the methodology.
>
> **Config note (2026-09-02):** the keys this document sets —
> `baseline_predictor.model_variant` and
> `baseline_predictor.qualifying.practice_challenger.launch_envelope_path` —
> were removed from `config/default.yaml` and `src/utils/config_schema.py` as
> dead configuration. No code on `master` ever read either one. Because the
> schema uses `extra="forbid"`, pasting the YAML examples below into
> `config/default.yaml` as-is will now fail validation at startup; restoring
> this workflow means restoring those schema fields alongside the branch code.

# Qualifying and Race Challenger Workflow

This workflow keeps the served predictor on `baseline_predictor.model_variant: champion`.
Q0, Q1, R0, R1, and R2 are research variants; creating a model, manifest, replay, or
sidecar does not activate one.

## Safety boundary

- Do not change `config/production_config.json`, champion weights, or active model
  artifacts during a race weekend.
- Freeze a complete manifest and immutable, scrubbed champion/challenger forecast pair
  before qualifying for a run to count as a preregistered shadow. A manifest alone, or
  a later run, is retrospective evidence.
- Store raw practice evidence and joint grid permutations only below
  `data/model_diagnostics/challenger_research/` or the dedicated challenger model
  directory. Served results contain only counts and stable digests.
- Wet, mixed, missing-artifact, wrong-weekend, and insufficient-evidence Q1/R0 inputs
  fail closed to the champion calculation.
- `run_challenger_walk_forward` (`src/analysis/challenger_walk_forward.py`) accepts
  an optional `research_gate_relaxation` mapping (e.g. `{"q1": 4}`) that lowers the
  Q1/R2-source-anchor minimum-training-event gate for one research run only, floor-
  clamped (Q1 >= 4, R2 source anchor >= 3) so it can never be relaxed to near-zero
  evidence. Production defaults (`MINIMUM_Q1_TRAINING_EVENTS`,
  `MINIMUM_R2_TRAINING_EVENTS`) are untouched, the override is only ever applied when
  a caller explicitly passes it, and the manifest must independently disclose the
  same relaxation in `metadata.research_gate_relaxation` or the runner refuses.
  `evaluate_release_readiness` always rejects a manifest carrying that marker
  (`no_research_gate_relaxation` check).
- The Q1 launch envelope is a live-shadow design: `created_at` is always real
  wall-clock time and can never be backdated, and resolution requires
  `created_at <= inference_cutoff`. A bundle fit *today* against an
  already-completed historical fold (offline walk-forward research, not live
  serving) can therefore never resolve without an explicit opt-in.
  `resolve_qualifying_practice_launch_envelope` /
  `resolve_qualifying_practice_bundle` (`src/models/qualifying_practice_bundle.py`)
  and `predict_qualifying` / `_run_qualifying_practice_challenger`
  (`src/predictors/baseline/qualifying_mixin.py`) accept a keyword-only
  `retrospective_diagnostic` flag, **default `False` on every call path with no
  behavior change when omitted** (champion never reaches this code at all; live and
  preregistered-shadow Q1 runs are unaffected -- see
  `tests/test_qualifying_practice_integration.py::test_champion_path_is_byte_identical_regardless_of_retrospective_diagnostic_flag`
  and `::test_live_q1_path_is_byte_identical_whether_the_flag_is_omitted_or_explicitly_false`).
  When explicitly set `True`, it relaxes **only** the artifact/bundle/envelope
  creation-time-vs-cutoff comparison; every leakage-relevant check (input strictly
  before cutoff, calibration disjointness, schema, hashes) stays fully enforced.
  Every resolution performed with it set carries `retrospective_diagnostic: true` in
  its diagnostics, and the manifest that produced it must carry the same marker in
  `metadata.retrospective_diagnostic`. `evaluate_release_readiness`
  (`src/analysis/challenger_release.py`) always rejects a manifest carrying that
  marker (`no_retrospective_diagnostic` check, same pattern as
  `no_research_gate_relaxation` below), and `build_frozen_forecast_pair` /
  `classify_shadow_registration` refuse to freeze or preregister one at all -- a
  retrospective-diagnostic Q1 result can only ever be an offline research finding,
  never a promotion candidate or a shadow forecast.

## Candidate registry

The explicit registry is in `src/models/challenger_variants.py`.

| Variant | Components | Purpose |
| --- | --- | --- |
| `q0_driver_state` | Q0 | Remove legacy form reuse when seconds-native state exists |
| `q1_qualifying_practice` | Q1 | Practice-to-qualifying potential |
| `r0_long_run` | R0 | Comparable long-run pace and tyre degradation |
| `r1_joint_grid` | R1 | Preserve coherent qualifying permutations in the race |
| `r2_no_anchor` | R2 | Disable the second, post-simulation grid anchor |
| `r2_source_anchor` | R2 | Use an anchor calibrated for the grid source |
| `r1_r2_no_anchor` | R1 + R2 | Joint-grid/no-anchor ablation |
| `r1_r2_source_anchor` | R1 + R2 | Joint-grid/source-anchor ablation |

The registry contains all 48 valid component sets, including `champion`. Every subset
of Q0, Q1, R0, and R1 is available with either no R2 mode, `r2_no_anchor`, or
`r2_source_anchor`; the two R2 modes are mutually exclusive. Descriptive IDs above
remain stable, while other combinations receive deterministic component-based IDs.
Configuration loading and manifest validation reject IDs outside the registry.

## Q1 data, fitting, and runtime bundle

`src/models/qualifying_practice_evidence.py` is the canonical dry-practice extractor
for Q1. It classifies compatible runs, learns compound/age/evolution normalization
from same-driver comparisons, preserves S1/S2/S3 as sector identities, and emits the
versioned `qualifying_practice_evidence` sidecar. Version 2 includes compatible-run
feature candidates for run-level bootstrap; those values never enter a served payload.

Prepared normalization input must contain `session_kind` and `track_class`; model and
calibration input must contain `session_kind`. Every prepared input must contain a
timezone-aware `event_start_at` strictly before the cutoff. Training and calibration
events must be disjoint. The fitter filters to the requested checkpoint, main or sprint
path, and track class instead of pooling unlike rows.

Fit each artifact into one canonical candidate directory. This example creates
`normalizations/main/fp2/high_downforce.json` and `models/main/fp2.json`:

```powershell
uv run python scripts/fit_qualifying_practice_challenger.py normalization `
  --input data/model_diagnostics/q1/comparisons.csv `
  --candidate-root data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1 `
  --candidate-id q1-practice-v1 --input-snapshot-id SNAPSHOT_ID `
  --checkpoint FP2 --session-kind main --track-class high_downforce `
  --cutoff 2026-07-18T12:00:00Z
```

```powershell
uv run python scripts/fit_qualifying_practice_challenger.py model `
  --input data/model_diagnostics/q1/train.csv `
  --calibration-input data/model_diagnostics/q1/calibration.csv `
  --candidate-root data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1 `
  --candidate-id q1-practice-v1 --input-snapshot-id SNAPSHOT_ID `
  --checkpoint FP2 --session-kind main --cutoff 2026-07-18T12:00:00Z
```

Repeat this for every preregistered checkpoint, session kind, and supported track
class. PRE has a checkpoint model but no practice normalization. The CLI requires 30
main-qualifying training events or eight sprint-qualifying training events and writes
atomically only to challenger paths. PRE, FP1, FP2, and FP3 remain separate models.

Create a standalone semantic candidate definition before freezing provenance, for
example `config/research/q1-practice-v1.yaml`:

```yaml
artifact_type: qualifying_practice_candidate_definition
schema_version: 1
model_variant: q1_qualifying_practice
candidate_id: q1-practice-v1
bundle_path: data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1/bundle.json
launch_envelope_path: data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1/launch.json
track_class_by_event:
  "2026:Example Grand Prix": high_downforce
uncertainty_scale: 1.0
```

This file is the semantic candidate definition: it contains behavioral settings and
stable future artifact locations, but no generated digest. It is provenance input,
not an application config file; do not put it into the production config chain.

Freeze provenance with source and dirty-diff hashes, snapshot IDs, cutoff, simulation
counts, fixed seeds `17`, `42`, and `91`, and every effective config input. The manifest
builder always hashes `config/default.yaml` and `config/production_config.json`; pass
the candidate overlay explicitly so an in-memory override cannot escape provenance:

```powershell
uv run python scripts/build_challenger_manifest.py `
  --candidate-id q1-practice-v1 --variant-id q1_qualifying_practice `
  --feature-schema qualifying-practice-v2 `
  --input-snapshot-id SNAPSHOT_ID --cutoff 2026-07-18T12:00:00Z `
  --simulation-count qualifying=5000 --simulation-count race=3000 `
  --config-path config/research/q1-practice-v1.yaml
```

`cutoff_at` is the last input-information timestamp and cannot be later than manifest
creation. Snapshot IDs cannot be empty. `MANIFEST_PATH` below is the immutable path
printed by this command.

The fitter records `generated_at` as the actual UTC creation time, separately from the
information cutoff. Bundle assembly requires
`max_input_timestamp < cutoff_timestamp <= generated_at <= manifest.created_at`.
Never rewrite `generated_at` to make a retrospective artifact appear preregistered.

After every candidate artifact and the manifest are frozen, bind them into one runtime
bundle:

```powershell
uv run python scripts/fit_qualifying_practice_challenger.py bundle `
  --candidate-id q1-practice-v1 --variant-id q1_qualifying_practice `
  --manifest MANIFEST_PATH `
  --candidate-root data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1
```

Finally, bind the already-hashed semantic definition, manifest, and bundle into an
immutable launch envelope:

```powershell
uv run python scripts/fit_qualifying_practice_challenger.py launch `
  --candidate-id q1-practice-v1 --variant-id q1_qualifying_practice `
  --manifest MANIFEST_PATH `
  --semantic-config config/research/q1-practice-v1.yaml `
  --candidate-root data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1
```

This acyclic order is deliberate:

`semantic config + fitted artifacts -> manifest -> bundle -> launch envelope`

The runtime config needs only the selected `model_variant` and stable launch pointer:

```yaml
baseline_predictor:
  model_variant: q1_qualifying_practice
  qualifying:
    practice_challenger:
      launch_envelope_path: data/processed/model_artifacts/qualifying_practice/challengers/q1-practice-v1/launch.json
```

There are no runtime bundle, candidate, track-map, uncertainty, or generated-digest
pins outside the launch envelope. The envelope binds the exact semantic config digest,
candidate, variant, manifest digest, bundle path/digest, and artifact layout. Resolution then
requires an exact checkpoint, `main` or `sprint`, and track-class match and revalidates
schema, hashes, and input/cutoff/creation chronology. Any missing or changed binding
fails closed. The returned diagnostics expose stable launch, bundle, manifest, and
semantic-config digests, and `run_challenger_pipeline()` refuses to freeze or persist
Q1 under a different candidate, variant, or manifest.

Track class is an explicit event binding, not a runtime guess. A missing event mapping
falls back to champion. The selected model comes from
`models/{main|sprint}/{checkpoint}.json`; FP checkpoints additionally resolve
`normalizations/{main|sprint}/{checkpoint}/{track_class}.json`.

Use `run_challenger_pipeline()` in `src/analysis/challenger_orchestration.py` for an
isolated shadow. It privately passes Q1 evidence, R1 scenarios, and R0 evidence;
optionally persists immutable manifest-linked sidecars; and removes raw evidence and
scenarios from the returned public result. For preregistration, supply the scrubbed
champion forecast and forecast event/freeze timestamp so the pipeline persists a
manifest-linked `frozen_forecasts` sidecar. Naked prediction hashes are not proof that
a forecast existed before qualifying. Historical replay consumes these immutable
frozen forecasts; it does not weaken the live resolver or pretend that a later-created
artifact existed at an earlier checkpoint.

## Walk-forward replay contract

`scripts/run_challenger_walk_forward.py` evaluates an immutable event catalog. Its
`FrozenPredictionBundleBackend` does not fit models, fetch live data, or construct new
predictions; it only scores role-and-seed outputs already frozen in each checkpoint.
Use a separate research backend to fit folds from raw historical data.

Each event-catalog row must provide:

- unique `event_id`, timezone-aware `event_start_at` and `qualifying_start_at`, with
  the latter strictly later;
- `session_kind` (`main` or `sprint`), boolean `is_dry`, and unique
  `input_snapshot_ids` contained in the manifest;
- complete, sequential `actual_qualifying_grid`, plus an optional complete
  `actual_race_finish_order` over the same drivers;
- one or more chronological `checkpoint_payloads` keyed by PRE, FP1, FP2, or FP3,
  each with `information_cutoff_at` inside the pre-qualifying window.

For the frozen backend, a checkpoint contains only `information_cutoff_at` and a
`forecast_reference`. Build that reference with
`freeze_checkpoint_forecast_bundle()` from
`src/analysis/challenger_walk_forward.py`. The referenced immutable research sidecar
contains champion and challenger predictions keyed by seeds `17`, `42`, and `91`, plus
the optional race views and fold artifacts; actual results remain only in the event
catalog. Resolution recomputes the sidecar, envelope, and bundle digests and verifies
the manifest, event, checkpoint, information cutoff, and freeze timestamp.

A bundle frozen before qualifying under a pre-existing manifest is recorded as
`preregistered_shadow`. Historical bundles frozen later remain valid replay inputs but
are explicitly recorded as `retrospective_diagnostic`; the latter cannot masquerade as
pre-qualifying evidence. Each qualifying prediction must contain intervals for every
driver and the complete teammate-pair set. Main Q1 needs 30 earlier training events
plus a later calibration holdout before an event is scoreable; sprint Q1 needs eight
plus a calibration holdout.

Run the replay and selected gates with:

```powershell
uv run python scripts/run_challenger_walk_forward.py `
  --manifest MANIFEST_PATH `
  --event-catalog data/model_diagnostics/q1/frozen_event_catalog.json `
  --movement-reviews data/model_diagnostics/q1/movement_reviews.json `
  --qualifying-target main_qualifying `
  --race-target grand_prix_race
```

The command writes an immutable `walk_forward_replay` research sidecar. It rejects
nonstandard seeds, snapshots absent from the manifest, future fold inputs, incomplete
grids, wrong variant/source labels, and catalogs with no eligible scored event.

## Race evaluation

Use `fetch_official_race_grid_for_research()` from
`src/analysis/challenger_orchestration.py` after qualifying. It wraps the lower-level
`fetch_official_starting_grid()` and returns `(grid, "actual_starting_grid")` ready for
the research orchestrator. The grid distinguishes grid and pit-lane starts; multiple
pit-lane starters require the qualifying classification for a non-leaky rear order.
Observed official grids take precedence and disable joint predicted scenarios.

`run_dual_race_replay()` reports both required views with matched seeds:

- `conditional_actual_grid`, using the official starting grid and no marginal noise;
- `end_to_end_predicted_grid`, using complete joint qualifying scenarios only when R1
  is active and the existing marginal predicted-grid fallback for Q-only candidates.

R2 source weights are fitted offline with equal weekend weight. Grid-anchor calibration
schema v2 requires `event_at` on every row and an explicit training `cutoff_at`, records
the selected training event IDs/timestamps, and must pass
`validate_grid_anchor_event_separation()` against replay evaluation events. Missing,
malformed, future, or overlapping source calibration fails closed to the champion
anchor.

Race promotion semantics follow component scope. An R0 or R1
`race_input_or_grid_propagation` candidate must improve end-to-end finisher MAE by at
least 0.10 positions while conditional-grid MAE may regress by at most 0.05. Only an
R2 anchor or race-physics candidate must improve both race views by at least 0.10.

## Promotion and release

Promotion thresholds are implemented in `src/analysis/challenger_governance.py`. Gate
replays must carry explicit event identities, fixed seeds, exact manifest simulation
counts, dry-only provenance, and populated checkpoint accounting. The full-field
movement audit flags every change strictly above two positions or ten H2H percentage
points. `src/analysis/challenger_release.py` records preregistration, attaches actuals
to immutable champion/challenger digests, and refuses release unless:

- independent candidates and their passing combination pass;
- evaluation, candidate, shadow, movement, promotion, and leakage audits pass;
- release is on a weekday;
- `champion` remains the immediate configuration rollback;
- the old champion runs in shadow for at least three weekends.

These helpers produce research decisions only. Promotion remains a separate reviewed
configuration change.
