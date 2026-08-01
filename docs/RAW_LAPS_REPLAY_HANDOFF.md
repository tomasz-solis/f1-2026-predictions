> **Status: shelved research, not production.** Handoff notes for the raw-laps
> replay mode that would let the Q1 and R0 challengers actually activate. The
> code this refers to is on the branch `shelved/challenger-research`, not on
> `master`. See `docs/MODEL_LEDGER.md` for why the work is blocked.

# Handoff: raw-laps replay fidelity upgrade

## Mission

Make the historical replay harness (`src/analysis/challenger_research_backend.py`)
feed each checkpoint the same **raw per-lap practice telemetry** the live weekend
predictor receives, instead of the current `practice_signal_mode="stored_profiles"`
aggregates. This is the single change that makes practice-driven variants genuinely
testable: today both Q1 (practice→qualifying potential) and R0 (long-run pace /
tyre degradation) silently degrade to champion-identical predictions in replay
because their runtime guards require raw laps that the harness never loads
(`session_laps_by_type = {}` by construction; Q1 discloses
`fallback_reason: "no_raw_practice_laps"`).

## Why this is the priority (decided 2026-07-19)

- The 2026 walk-forward comparison (see
  `data/model_diagnostics/2026/race_mae_investigation/2026_WALK_FORWARD_VARIANT_COMPARISON_v2.md`
  and the v3 revision) could only genuinely differentiate grid-plumbing variants
  (r1/r2). The two variants matching the core modeling thesis — one-lap pace for
  qualifying, long-run pace for the race — were structurally untestable.
- Replay/live parity is a systemic health property: backtests must exercise the
  same input path as the served weekend forecast, or shadow evidence
  systematically diverges from live behavior.
- Value compounds: every completed weekend adds raw-lap data; Q1 eligibility
  (street/permanent classes, research floor 4) grows through the season; the
  deferred track-similarity roadmap (corner-speed distributions, abrasiveness
  proxies) needs per-lap data too.

## Non-negotiable boundaries (unchanged from the race-MAE handoff)

- `config/production_config.json` untouched (sha256
  `c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1`);
  `config/default.yaml` keeps `baseline_predictor.model_variant: champion`.
- No champion weights, active artifacts, prediction artifacts, or served
  forecasts modified. No commits/pushes unless explicitly requested.
- Leakage discipline: a checkpoint may load raw laps ONLY from sessions whose
  data existed before that checkpoint's `information_cutoff_at`. Existing
  leakage tests must keep passing; add raw-laps-specific ones.
- Fail closed per event-checkpoint with a recorded refusal (the proven
  `CheckpointInputUnavailable` pattern) when telemetry is thin — the Barcelona
  FP1 case (`teams=1 mapped=0 selected_laps=0`) is the canonical fixture.
- Research outputs below `data/historical_replay/` / `data/model_diagnostics/`.

## Current state pointers

- Backend: `src/analysis/challenger_research_backend.py` — `predict_qualifying`
  / `predict_race_views` call the live predictor with
  `practice_signal_mode="stored_profiles"`; `_r0_evidence` extracts long-run
  evidence but the runtime path never activates on aggregates.
- Q1 guard: `src/predictors/baseline/qualifying_mixin.py` (raw-lap requirement;
  carries the authorized `retrospective_diagnostic` flag — preserve it).
- R0 evidence: `src/features/race_practice_evidence.py`.
- Memory hazard: FastF1 session loading has caused memory blowups before — see
  commits `eac843c2` ("isolate FastF1 sessions to cap memory") and `bea5e2c6`.
  Load sessions per checkpoint, release before the next; never hold a full
  season of laps resident.
- Prediction cache: `data/historical_replay/2026/prediction_cache/` keyed by
  (event, checkpoint, variant, seed) + source digest. Raw-laps mode MUST use a
  distinct cache dimension (e.g. `practice_signal_mode` in the key) so old
  stored-profiles results are never silently mixed with new ones.

## Acceptance criteria

1. A replayed FP checkpoint feeds the predictor raw per-lap data equivalent to
   the live path (same loader/normalization code, not a reimplementation).
2. R0 and Q1 produce predictions that differ from champion on at least one
   event-checkpoint-seed — or refuse with a specific recorded reason. The
   generalized identity guard must show zero undisclosed champion-identical
   challenger rows.
3. Champion replay results under raw-laps mode are regenerated and compared
   against the stored-profiles baseline: differences reported per checkpoint
   (this measures the fidelity gap itself — a finding, not an error).
4. Thin-telemetry events refuse per-checkpoint; no whole-variant voiding.
5. Runtime cost measured on one event before the full campaign; simulation
   counts and mode recorded in the run manifest.
6. Walk-forward rerun of champion + q0 + r0 + r1 (and Q1 via the retrospective
   path where eligible) at FP checkpoints, 500/500 sims, seeds 17/42/91, new
   run_tag `raw_laps`; consolidated report revision comparing raw-laps vs
   stored-profiles results side by side.
7. Standard verification: focused tests (leakage, cache-key separation, memory
   isolation smoke), full relevant pytest set, ruff, mypy, config sha256, git
   status. Work log updated every step (`RESEARCH_WORK_LOG.md`).

## Recommended parallel habit (no code)

Preregister r1_joint_grid shadow runs each race weekend (frozen manifest +
scrubbed forecast pair before qualifying, per
`docs/QUALIFYING_RACE_CHALLENGER.md`). r1 is the only variant with a positive
signal (end-to-end finisher MAE 4.27 vs champion 4.34; winner 19.0% vs 14.3%);
preregistered shadows build promotion-grade evidence that no retrospective
replay can.
