"""Research backend that fits and predicts each fold from real 2026 repo data.

``FrozenPredictionBundleBackend`` (see ``challenger_walk_forward.py``) only *scores*
predictions that were frozen elsewhere.  This module is the leakage-safe research
adapter it defers to: it drives the existing champion predictor pipeline
(``src.predictors.baseline``) and the existing challenger component flags
(``src.models.challenger_variants``) against real season data, with a strict cutoff
contract enforced by construction rather than convention:

* Season "fitting" (team/driver state) for event N replays only events strictly
  before N -- the same ``update_from_race`` / ``update_from_testing_sessions``
  machinery ``run_historical_checkpoint_replay`` already uses in production, applied
  to an explicit, caller-supplied event catalog instead of the full hardcoded season
  plan.
* A checkpoint's own prediction additionally sees event N's *own* practice sessions
  that are available strictly before that checkpoint's ``information_cutoff_at`` (the
  checkpoint-overlay mechanism already used by the dashboard), never the target
  session or anything later.
* No physics, simulator, or challenger-component logic is reimplemented here; this
  module only decides *which real data* the existing predictor is allowed to see for
  each fold, and adapts its outputs to the ``WalkForwardBackend`` protocol.

State is fitted **once per (event, checkpoint)**, independent of which challenger
variant or seed is being scored -- component flags (Q0/Q1/R0/R1/R2) only change
*prediction-time* behaviour in the champion predictor, never how team/driver state is
fit from actual results. Fitted processed-data directories and raw prediction outputs
are both cached to disk under ``state_root`` / ``cache_root`` so a partially completed
run resumes instead of recomputing already-fitted folds or already-scored predictions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml

from src.analysis.challenger_governance import DEFAULT_CONFIG_PATHS, build_challenger_manifest
from src.analysis.challenger_walk_forward import CheckpointInputUnavailable, ReplayEvent
from src.analysis.grid_anchor_calibration import fit_source_specific_grid_anchors
from src.analysis.race_view_replay import run_dual_race_replay
from src.dashboard.checkpoint_predictor import build_checkpoint_overlay_predictor
from src.features.race_practice_evidence import build_race_practice_evidence
from src.models.challenger_variants import CHAMPION_VARIANT, VARIANT_COMPONENTS
from src.models.qualifying_practice_bundle import (
    build_qualifying_practice_bundle,
    build_qualifying_practice_launch_envelope,
    resolve_qualifying_practice_launch_envelope,
)
from src.models.qualifying_practice_challenger import (
    DEFAULT_FEATURE_COLUMNS,
    fit_bradley_terry_model,
)
from src.models.qualifying_practice_evidence import fit_practice_normalization
from src.models.qualifying_practice_runtime import (
    build_qualifying_practice_feature_rows,
    build_weekend_qualifying_practice_evidence,
)
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline_2026 import Baseline2026Predictor
from src.systems.updater import update_from_race
from src.utils.config_loader import Config
from src.utils.historical_replay import (
    ReplayConfigOverride,
    _apply_session_update,
    _force_file_only_storage,
    _reset_replay_artifacts,
)

logger = logging.getLogger(__name__)

DEFAULT_QUALIFYING_SIMULATIONS = 60
DEFAULT_RACE_SIMULATIONS = 60
# r2_source_anchor research calibration: fixed coarse-sampling seed/sim-count for
# the internal r2_no_anchor calibration-row extraction (not a scored output), and
# the production same-format training-event floor used as the shrinkage-to-1.0
# denominator (matches challenger_walk_forward.MINIMUM_R2_TRAINING_EVENTS).
DEFAULT_CALIBRATION_SEED = 42
CALIBRATION_SIMULATIONS = 100
MINIMUM_R2_TRAINING_EVENTS_PRODUCTION = 8

_PRACTICE_SESSION_TO_CHECKPOINT = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}
_SESSION_LABEL_TO_CODE = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Race": "R",
}
# Cumulative practice-session labels available at each Q1 checkpoint -- fixed by
# the FP1/FP2/FP3 ordering itself, independent of any one event's catalog filter.
_PRACTICE_SESSIONS_FOR_CHECKPOINT = {
    "FP1": ["Practice 1"],
    "FP2": ["Practice 1", "Practice 2"],
    "FP3": ["Practice 1", "Practice 2", "Practice 3"],
}


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capped_session_weights(
    n_sessions: int, ceiling: float, *, base_weight: float = 0.25
) -> list[float]:
    """Round-9 fix candidate A (research-only): per-session EWMA weights so the
    CUMULATIVE pull toward this weekend's own practice sessions never exceeds
    ``ceiling``, however many sessions get applied in sequence.

    Greedy: session i gets the normal ``base_weight`` unless that would push
    cumulative pull ``p`` (starting at 0, since ``_checkpoint_state_for`` always
    rebuilds from the season prefix before applying this event's own sessions)
    past ``ceiling`` -- then it gets just enough to land exactly on the ceiling,
    and every later session in the same checkpoint gets 0. No knowledge of the
    eventual session count is needed: FP1 uses weights[:1], FP2 weights[:2], FP3
    weights[:3], all drawn from the same precomputed sequence.
    """
    weights: list[float] = []
    pull = 0.0
    for _ in range(n_sessions):
        headroom = max(0.0, ceiling - pull)
        remaining = 1.0 - pull
        weight = min(base_weight, headroom / remaining) if remaining > 0 else 0.0
        weights.append(weight)
        pull += (1.0 - pull) * weight
    return weights


class ProductionReplayBackend:
    """Drive the real champion/challenger predictor pipeline for walk-forward replay.

    ``events`` must be the exact chronological event catalog the walk-forward runner
    was given (as raw catalog rows, i.e. the same mappings passed to
    ``run_challenger_walk_forward``). This backend derives "events strictly before N"
    from that catalog directly, since the ``WalkForwardBackend`` protocol's
    ``predict_qualifying`` / ``predict_race_views`` do not receive the runner's
    training-event list (only ``fit_fold`` does, and it is only invoked for variants
    with a nonzero minimum-training requirement).
    """

    def __init__(
        self,
        *,
        events: Sequence[Mapping[str, Any]],
        source_processed_dir: str | Path = "data/processed",
        state_root: str | Path = "data/historical_replay/2026/research_backend_state",
        prediction_cache_root: str | Path = "data/historical_replay/2026/prediction_cache",
        track_class_binding_path: str
        | Path = "data/historical_replay/2026/track_class_by_event.json",
        qualifying_simulations: int = DEFAULT_QUALIFYING_SIMULATIONS,
        race_simulations: int = DEFAULT_RACE_SIMULATIONS,
        research_cumulative_pull_cap: float | None = None,
        research_min_field_coverage: float | None = None,
    ) -> None:
        self._source_processed_dir = Path(source_processed_dir)
        self._state_root = Path(state_root)
        self._prediction_cache_root = Path(prediction_cache_root)
        self._track_class_binding_path = Path(track_class_binding_path)
        self._track_class_binding_cache: Mapping[str, Any] | None = None
        self._q1_research_root = Path("data/historical_replay/2026/research_q1")
        self._qualifying_simulations = int(qualifying_simulations)
        self._race_simulations = int(race_simulations)
        # Research-only, default-OFF: caps the cumulative EWMA pull toward this
        # weekend's own practice sessions (see docs/RESEARCH_WORK_LOG.md round 9 --
        # backtests the round-8 practice-overlay-degradation fix candidate). None
        # means byte-identical to today's stored_profiles behavior; only
        # `_checkpoint_state_for` (this event's OWN checkpoint session replay)
        # consumes it -- season-prefix state (`_commit_event`, prior events already
        # "in the books") is intentionally untouched, that is a separate question.
        self._research_cumulative_pull_cap = (
            None if research_cumulative_pull_cap is None else float(research_cumulative_pull_cap)
        )
        # Research-only, default-OFF: round-9 Fix B. A practice session only
        # moves car characteristics when at least this fraction of the field has
        # robust evidence (mirrors r0's MIN_R0_TEAM_COVERAGE gate); below
        # threshold the session contributes nothing (see
        # `_characteristics_update_paths`'s snapshot/restore in
        # `_apply_session_update`). Mutually independent of the pull cap above --
        # both may be set at once for an A+B backtest.
        self._research_min_field_coverage = (
            None if research_min_field_coverage is None else float(research_min_field_coverage)
        )

        rows = sorted(
            (dict(row) for row in events),
            key=lambda row: (str(row["event_start_at"]), str(row["event_id"])),
        )
        self._chronology: list[str] = [str(row["event_id"]) for row in rows]
        self._raw_by_id: dict[str, dict[str, Any]] = {str(row["event_id"]): row for row in rows}
        # Season/checkpoint *state* (team/driver fitting) never depends on simulation
        # counts, only on which real events and sessions were replayed -- keep it on
        # its own digest so raising sim counts reuses already-fitted state directories
        # instead of re-copying and re-replaying every event from scratch.
        _state_digest_payload: dict[str, Any] = {
            "events": [
                {
                    "event_id": row["event_id"],
                    "event_start_at": row["event_start_at"],
                    "actual_qualifying_grid_sha256": _digest(row["actual_qualifying_grid"]),
                    "actual_race_finish_order_sha256": _digest(row.get("actual_race_finish_order")),
                }
                for row in rows
            ],
        }
        if self._research_cumulative_pull_cap is not None:
            # Only added when set, so the default (None/off) digest is BYTE-IDENTICAL
            # to before this flag existed -- the existing season-state cache (hours of
            # real replay work) stays valid for every ordinary run; only an explicit
            # capped run gets its own, separate state-directory tree (and therefore
            # prediction cache, which keys off _source_digest -> _state_digest) --
            # the distinct-cache-dimension requirement, without disturbing anything else.
            _state_digest_payload["research_cumulative_pull_cap"] = (
                self._research_cumulative_pull_cap
            )
        if self._research_min_field_coverage is not None:
            _state_digest_payload["research_min_field_coverage"] = self._research_min_field_coverage
        self._state_digest = _digest(_state_digest_payload)
        self._source_digest = _digest(
            {
                "state_digest": self._state_digest,
                "qualifying_simulations": self._qualifying_simulations,
                "race_simulations": self._race_simulations,
            }
        )
        self._prefix_state_dir: dict[int, Path] = {}
        self._checkpoint_state_dir: dict[tuple[str, str], Path] = {}
        self._prediction_cache: dict[str, Any] = {}
        # Sessions skipped while replaying a *training precursor* event (not the
        # scored checkpoint itself) because their real data was unusable. Exposed
        # for transparency; never silently invented, never blocks later events.
        self.degraded_training_sessions: list[dict[str, str]] = []

    # -- season/checkpoint state construction -------------------------------------

    def _prefix_dir_root(self) -> Path:
        return self._state_root / f"prefix-{self._state_digest[:16]}"

    def _reset_state(self) -> Path:
        """Return the pristine pre-season state, building it once and caching it."""
        target = self._prefix_dir_root() / "index_0" / "processed"
        marker = target.parent / "_complete.json"
        if marker.is_file():
            return target
        if target.parent.exists():
            shutil.rmtree(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._source_processed_dir, target)
        _reset_replay_artifacts(target, year=2026)
        marker.write_text(json.dumps({"index": 0}), encoding="utf-8")
        return target

    def _commit_event(self, *, processed_dir: Path, event_id: str) -> None:
        """Replay one training event's own sessions into ``processed_dir`` in place."""
        raw = self._raw_by_id[event_id]
        race_name = str(raw["race_name"])
        cache_dir = str(raw["fastf1_cache_dir"])
        year = int(raw.get("year", 2026))
        # PRE has no sessions_available; FP3 (or the last recorded checkpoint) carries
        # the full cumulative practice-session list for the weekend.
        practice_sessions: list[str] = []
        for checkpoint in ("FP1", "FP2", "FP3"):
            payload = raw["checkpoint_payloads"].get(checkpoint)
            if payload is not None:
                practice_sessions = list(payload["sessions_available"])
        session_labels = [*practice_sessions, "Qualifying", "Race"]
        with _force_file_only_storage():
            for label in session_labels:
                try:
                    _apply_session_update(
                        year=year,
                        event_name=race_name,
                        session_name=_SESSION_LABEL_TO_CODE[label],
                        cache_dirs=[cache_dir],
                        processed_dir=processed_dir,
                    )
                except ValueError as exc:
                    # This event is being replayed only as a *training precursor* for
                    # a later target event -- one unusable session (e.g. too little
                    # completed running to extract team telemetry) must not cascade
                    # into refusing every later event's checkpoints. Season state
                    # proceeds without this session's contribution; the target
                    # event's *own* checkpoint build (_checkpoint_state_for) still
                    # fails closed and is reported when this same gap affects it.
                    logger.warning(
                        "Skipping %s %s in season-state training replay: %s",
                        race_name,
                        label,
                        exc,
                    )
                    self.degraded_training_sessions.append(
                        {"event_id": event_id, "session": label, "reason": str(exc)}
                    )
            # ponytail: sprint-format weekends also ran SQ/Sprint sessions; those are
            # not replayed into season state here (only main Q/R feed team/driver
            # fitting). Upgrade path: call update_from_sprint_race too if a later
            # accuracy check shows sprint-only signal materially changes fits.
            update_from_race(year, race_name, str(processed_dir), trace_rows=None)

    def _prefix_state_for(self, target_event_id: str) -> Path:
        """Return (building/caching as needed) state fit on events strictly before N."""
        index = self._chronology.index(target_event_id)
        cached = self._prefix_state_dir.get(index)
        if cached is not None:
            return cached
        if index == 0:
            target = self._reset_state()
            self._prefix_state_dir[index] = target
            return target
        target = self._prefix_dir_root() / f"index_{index}" / "processed"
        marker = target.parent / "_complete.json"
        if marker.is_file():
            self._prefix_state_dir[index] = target
            return target
        source = self._prefix_state_for(self._chronology[index - 1])
        if target.parent.exists():
            shutil.rmtree(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        self._commit_event(processed_dir=target, event_id=self._chronology[index - 1])
        marker.write_text(json.dumps({"index": index}), encoding="utf-8")
        self._prefix_state_dir[index] = target
        return target

    def _checkpoint_state_for(self, *, event_id: str, checkpoint: str) -> Path:
        """Return prefix state plus this event's own pre-cutoff practice sessions."""
        cache_key = (event_id, checkpoint)
        cached = self._checkpoint_state_dir.get(cache_key)
        if cached is not None:
            return cached
        raw = self._raw_by_id[event_id]
        target = self._prefix_dir_root() / f"checkpoint-{event_id}-{checkpoint}" / "processed"
        marker = target.parent / "_complete.json"
        prefix = self._prefix_state_for(event_id)
        if marker.is_file():
            self._checkpoint_state_dir[cache_key] = target
            return target
        if target.parent.exists():
            shutil.rmtree(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prefix, target)
        checkpoint_payload = raw["checkpoint_payloads"].get(checkpoint, {"sessions_available": []})
        sessions_available = list(checkpoint_payload.get("sessions_available", []))
        if sessions_available:
            year = int(raw.get("year", 2026))
            race_name = str(raw["race_name"])
            cache_dir = str(raw["fastf1_cache_dir"])
            session_weights: list[float | None] = (
                cast(
                    list["float | None"],
                    _capped_session_weights(
                        len(sessions_available), self._research_cumulative_pull_cap
                    ),
                )
                if self._research_cumulative_pull_cap is not None
                else [None] * len(sessions_available)
            )
            with _force_file_only_storage():
                for label, new_weight in zip(sessions_available, session_weights, strict=True):
                    try:
                        _apply_session_update(
                            year=year,
                            event_name=race_name,
                            session_name=_SESSION_LABEL_TO_CODE[label],
                            cache_dirs=[cache_dir],
                            processed_dir=target,
                            new_weight=new_weight,
                            min_field_coverage=self._research_min_field_coverage,
                        )
                    except ValueError as exc:
                        # This IS the event/checkpoint being scored: its own stated
                        # pre-cutoff practice session could not be extracted from
                        # real data, so fail closed on this event-checkpoint only
                        # (the runner records it and keeps scoring every other
                        # eligible event-checkpoint) rather than inventing a grid.
                        raise CheckpointInputUnavailable(
                            f"{event_id} {checkpoint}: required session {label!r} "
                            f"could not be extracted: {exc}"
                        ) from exc
        marker.write_text(
            json.dumps({"event_id": event_id, "checkpoint": checkpoint}), encoding="utf-8"
        )
        self._checkpoint_state_dir[cache_key] = target
        return target

    # -- predictor construction ----------------------------------------------------

    def _config_for(
        self, variant_id: str, *, extra_overrides: Mapping[str, Any] | None = None
    ) -> ReplayConfigOverride:
        overrides: dict[str, Any] = {
            "baseline_predictor.model_variant": variant_id,
            "baseline_predictor.current_season_form.infer_from_saved_actuals": False,
        }
        if extra_overrides:
            overrides.update(extra_overrides)
        return ReplayConfigOverride(base_config=Config(), overrides=overrides)

    def _predictor(
        self,
        *,
        state_dir: Path,
        variant_id: str,
        seed: int,
        event: ReplayEvent,
        checkpoint: str,
        extra_config_overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        raw = self._raw_by_id[event.event_id]
        year = int(raw.get("year", 2026))
        race_name = str(raw["race_name"])
        artifact_store = ArtifactStore(data_root=state_dir.parent)
        base = Baseline2026Predictor(
            data_dir=str(state_dir),
            seed=int(seed),
            season_year=year,
            artifact_store=artifact_store,
            config=cast(
                Config, self._config_for(variant_id, extra_overrides=extra_config_overrides)
            ),
        )
        return build_checkpoint_overlay_predictor(
            base_predictor=base,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint,
            is_sprint=(event.session_kind == "sprint"),
        )

    def _historical_context(self, *, event: ReplayEvent, target_session: str) -> Any:
        from src.utils.prediction_context import build_historical_prediction_context

        raw = self._raw_by_id[event.event_id]
        return build_historical_prediction_context(
            year=int(raw.get("year", 2026)),
            race_name=str(raw["race_name"]),
            target_session_name=target_session,
        )

    # -- research-gate-relaxation: Q1 fold fit and R2 source-anchor calibration -----

    def _track_class_bindings(self) -> Mapping[str, Any]:
        """Load (and cache) the curated street/permanent binding file.

        Track class is documented as "an explicit event binding, not a runtime
        guess" (see docs/QUALIFYING_RACE_CHALLENGER.md); qualifying_mixin.py reads
        it from a launch envelope's ``track_class_by_event`` map, never derives it
        from track_characteristics directly. The binding is curated offline by
        ``scripts/build_track_class_binding.py`` (user-approved taxonomy: exactly
        the repo's existing track_type classification, resolved through
        ``src.data.circuit_registry`` so a migrating GP name like the 2026 Barcelona
        GP still finds its real circuit's classification) and fails closed there --
        this method never invents or pools a missing binding.
        """
        if self._track_class_binding_cache is None:
            if not self._track_class_binding_path.is_file():
                self._track_class_binding_cache = {}
            else:
                payload = json.loads(self._track_class_binding_path.read_text(encoding="utf-8"))
                self._track_class_binding_cache = payload.get("bindings", {})
        return self._track_class_binding_cache

    def _research_track_class(self, event_id: str) -> str | None:
        """Return the curated track class for one event, or None if unbound."""
        binding = self._track_class_bindings().get(event_id)
        if not isinstance(binding, Mapping):
            return None
        return cast(str | None, binding.get("track_class"))

    def _q1_track_class_eligibility(
        self, target_event: ReplayEvent, *, relaxed_floor: int
    ) -> tuple[list[str], str | None]:
        """Cross-format count of prior dry same-track-class events vs the floor.

        Pooled across session_kind (main + sprint) deliberately: track class is a
        physical-circuit property, not a weekend-format one, and the walk-forward
        runner's own ``training_events`` are already restricted to the target's
        session_kind -- restricting *again* by track class inside that set would
        undercount every fold. Returns (matching prior dry event_ids in chronological
        order, refusal reason or None).
        """
        target_class = self._research_track_class(target_event.event_id)
        if target_class is None:
            return [], (
                f"no curated track_class binding for {target_event.event_id} "
                f"(see {self._track_class_binding_path})"
            )
        matching = [
            event_id
            for event_id in self._chronology
            if event_id != target_event.event_id
            and self._raw_by_id[event_id]["event_start_at"]
            < self._raw_by_id[target_event.event_id]["event_start_at"]
            and bool(self._raw_by_id[event_id]["is_dry"])
            and self._research_track_class(event_id) == target_class
        ]
        if len(matching) < relaxed_floor:
            return matching, (
                f"only {len(matching)} prior dry {target_class!r}-class event(s) "
                f"< required {relaxed_floor} (research floor) for {target_event.event_id}"
            )
        return matching, None

    def _r2_no_anchor_calibration_rows(
        self, *, training_event: ReplayEvent
    ) -> list[dict[str, Any]]:
        """Real (simulated_position, grid_position, actual_position) rows for one
        already-completed event, recovered without touching simulator internals.

        ``r2_no_anchor`` sets the post-simulation grid-anchor weight to exactly
        0.0, so its predicted position *is* the pre-anchor simulated position --
        the one field ``fit_source_specific_grid_anchors`` needs that predict_race's
        public payload never returns directly. Cached per training event (fold-
        independent: it only depends on that event's own prior state and actuals).
        """
        cache_key = {
            "kind": "r2_no_anchor_calibration_rows",
            "source_digest": self._source_digest,
            "event_id": training_event.event_id,
        }
        cached = self._cached(cache_key)
        if cached is not None:
            return cast(list[dict[str, Any]], cached["rows"])

        raw = self._raw_by_id[training_event.event_id]
        state_dir = self._checkpoint_state_for(event_id=training_event.event_id, checkpoint="PRE")
        predictor = self._predictor(
            state_dir=state_dir,
            variant_id="r2_no_anchor",
            seed=DEFAULT_CALIBRATION_SEED,
            event=training_event,
            checkpoint="PRE",
        )
        # ponytail: coarse calibration sampling, not a scored/reported output -- a
        # small fixed sim count keeps N-training-event calibration bounded even
        # when the campaign's own sim count is high.
        calibration_sims = min(self._race_simulations, CALIBRATION_SIMULATIONS)
        prediction = predictor.predict_race(
            qualifying_grid=list(raw["actual_starting_grid"]),
            weather="dry",
            race_name=str(raw["race_name"]),
            year=int(raw.get("year", 2026)),
            n_simulations=calibration_sims,
            is_sprint=training_event.session_kind == "sprint",
            grid_source_detail="actual_starting_grid",
            prediction_context=self._historical_context(event=training_event, target_session="R"),
        )
        simulated_by_driver = {row["driver"]: row["position"] for row in prediction["finish_order"]}
        grid_by_driver = {row["driver"]: row["position"] for row in raw["actual_starting_grid"]}
        actual_by_driver = {
            row["driver"]: row["position"] for row in raw["actual_race_finish_order"]
        }
        rows = [
            {
                "event_id": training_event.event_id,
                "event_at": raw["event_start_at"],
                "grid_source_detail": "actual_starting_grid",
                "driver_id": driver,
                "simulated_position": float(simulated_by_driver[driver]),
                "grid_position": float(grid_by_driver[driver]),
                "actual_position": float(actual_by_driver[driver]),
            }
            for driver in simulated_by_driver
            if driver in grid_by_driver and driver in actual_by_driver
        ]
        self._store(cache_key, {"rows": rows})
        return rows

    def _fit_r2_source_anchor(
        self,
        *,
        training_events: Sequence[ReplayEvent],
        target_event: ReplayEvent,
        relaxed_floor: int,
    ) -> dict[str, Any]:
        """Fit a research-only calibrated anchor weight from real prior events."""
        rows: list[dict[str, Any]] = []
        for training_event in training_events:
            rows.extend(self._r2_no_anchor_calibration_rows(training_event=training_event))
        calibration = fit_source_specific_grid_anchors(
            rows,
            cutoff_at=target_event.event_start_at,
            min_events=relaxed_floor,
            allowed_source_details=("actual_starting_grid",),
        )
        diagnostics = calibration["diagnostics"]["actual_starting_grid"]
        if diagnostics["status"] != "fitted":
            return {
                "status": diagnostics["status"],
                "n_training_events": len(training_events),
                "reason": (
                    f"only {diagnostics['event_count']} of {relaxed_floor} required "
                    "same-format prior events produced usable calibration rows"
                ),
            }
        shrinkage_weight = min(
            1.0, len(training_events) / float(MINIMUM_R2_TRAINING_EVENTS_PRODUCTION)
        )
        return {
            "status": "fitted",
            "calibrated_weight": float(diagnostics["selected_weight"]),
            "shrinkage_weight": shrinkage_weight,
            "n_training_events": len(training_events),
            "n_calibration_rows": diagnostics["row_count"],
        }

    def _q1_prior_event_feature_rows(
        self, *, event_id: str, session_labels: list[str], track_class: str
    ) -> pd.DataFrame | None:
        """Real per-driver Q1 feature rows for one prior (already-qualified) event.

        Cached per (event_id, session_labels, track_class): fold-independent, since
        it only depends on that prior event's own real practice laps and actual
        qualifying result, never on the target fold.
        """
        cache_key = {
            "kind": "q1_prior_feature_rows",
            "source_digest": self._source_digest,
            "event_id": event_id,
            "session_labels": session_labels,
            "track_class": track_class,
        }
        cached = self._cached(cache_key)
        if cached is not None:
            return pd.DataFrame(cached["rows"]) if cached["rows"] else None

        raw = self._raw_by_id[event_id]
        try:
            import fastf1

            fastf1.Cache.enable_cache(str(raw["fastf1_cache_dir"]))
            laps_by_session: dict[str, pd.DataFrame] = {}
            for label in session_labels:
                session = fastf1.get_session(
                    int(raw.get("year", 2026)), str(raw["race_name"]), label
                )
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                laps_by_session[_SESSION_LABEL_TO_CODE[label]] = session.laps
        except Exception:  # noqa: BLE001 - fail closed: no rows rather than invented ones
            logger.warning("q1 prior-event evidence extraction failed closed for %s", event_id)
            self._store(cache_key, {"rows": []})
            return None

        # ponytail: no real same-driver "comparisons" dataset is built here (that is
        # its own substantial extraction task); fit_practice_normalization's own
        # documented fallback for comparisons=None is a valid prior-based
        # normalization (real code path, not a special case invented for research
        # mode) -- disclosed in the fold's recorded status as normalization_source.
        normalization = fit_practice_normalization(None)
        evidence_by_session = build_weekend_qualifying_practice_evidence(
            laps_by_session,
            normalization=normalization,
            track_name=str(raw["race_name"]),
            track_class=track_class,
        )
        all_drivers = [
            {"driver": row["driver"], "team": row["team"]} for row in raw["actual_qualifying_grid"]
        ]
        feature_rows, evidence_summary = build_qualifying_practice_feature_rows(
            evidence_by_session, all_drivers=all_drivers
        )
        if not bool(evidence_summary.get("eligible", False)) or feature_rows.empty:
            self._store(cache_key, {"rows": []})
            return None

        actual_by_driver = {row["driver"]: row["position"] for row in raw["actual_qualifying_grid"]}
        feature_rows = feature_rows.copy()
        feature_rows["event_id"] = event_id
        feature_rows["actual_position"] = feature_rows["driver"].map(actual_by_driver)
        feature_rows = feature_rows.dropna(subset=["actual_position"])
        self._store(cache_key, {"rows": json.loads(feature_rows.to_json(orient="records"))})
        return feature_rows if not feature_rows.empty else None

    def _fit_q1_research_launch(
        self,
        *,
        target_event: ReplayEvent,
        checkpoint: str,
        matching_event_ids: list[str],
        relaxed_floor: int,
        track_class: str,
    ) -> tuple[Path | None, dict[str, Any]]:
        """Fit a real, research-labeled Q1 launch envelope for one eligible fold.

        Real practice evidence from every prior same-class dry event -> the
        production Bradley-Terry fitter (no CLI-imposed 30/8-event floor inside the
        fit function itself) -> shrinkage toward champion via ``regularization_c``
        scaled by ``n_training_events/30`` (thinner evidence = stronger shrinkage
        toward a flat, champion-equivalent utility) -> a real bundle + launch
        envelope the Q1 runtime resolves like any other candidate. Refuses (never
        fabricates) with the exact row/event counts if the fitter itself has too
        little usable evidence.
        """
        session_labels = _PRACTICE_SESSIONS_FOR_CHECKPOINT[checkpoint]
        frames: list[pd.DataFrame] = []
        used_events: list[str] = []
        for event_id in matching_event_ids:
            rows = self._q1_prior_event_feature_rows(
                event_id=event_id, session_labels=session_labels, track_class=track_class
            )
            if rows is not None:
                frames.append(rows)
                used_events.append(event_id)

        if not frames:
            return None, {
                "status": "no_usable_prior_event_evidence",
                "prior_events_attempted": matching_event_ids,
            }

        dataset = pd.concat(frames, ignore_index=True)
        n_training_events = len(used_events)
        regularization_c = max(1e-3, n_training_events / 30.0)
        try:
            model = fit_bradley_terry_model(
                dataset,
                checkpoint=checkpoint,
                feature_columns=DEFAULT_FEATURE_COLUMNS,
                regularization_c=regularization_c,
            )
        except ValueError as exc:
            return None, {
                "status": "fitter_refused",
                "reason": str(exc),
                "rows_available": int(len(dataset)),
                "events_used": used_events,
                "feature_columns_required": list(DEFAULT_FEATURE_COLUMNS),
            }

        raw_target = self._raw_by_id[target_event.event_id]
        checkpoint_payload = target_event.checkpoint_payloads[checkpoint]
        cutoff_text = str(checkpoint_payload["information_cutoff_at"])
        cutoff_dt = datetime.fromisoformat(cutoff_text.replace("Z", "+00:00"))
        max_input_dt = max(
            datetime.fromisoformat(
                str(self._raw_by_id[event_id]["event_start_at"]).replace("Z", "+00:00")
            )
            for event_id in used_events
        )
        if max_input_dt >= cutoff_dt:
            # Cannot happen for a genuinely prior event, but never trust silently.
            return None, {
                "status": "leakage_guard_rejected",
                "reason": f"prior-event input {max_input_dt.isoformat()} is not strictly "
                f"before the fold cutoff {cutoff_dt.isoformat()}",
            }

        variant_id = "q1_qualifying_practice"
        candidate_id = f"research-q1-{target_event.event_id}-{checkpoint.lower()}"
        candidate_root = self._q1_research_root / f"{target_event.event_id}_{checkpoint}"
        if candidate_root.exists():
            shutil.rmtree(candidate_root)
        candidate_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)

        semantic_config_path = candidate_root / "candidate.yaml"
        semantic_config = {
            "artifact_type": "qualifying_practice_candidate_definition",
            "schema_version": 1,
            "model_variant": variant_id,
            "candidate_id": candidate_id,
            "bundle_path": str((candidate_root / "bundle.json").as_posix()),
            "launch_envelope_path": str((candidate_root / "launch.json").as_posix()),
            "track_class_by_event": {
                f"{int(raw_target.get('year', 2026))}:{raw_target['race_name']}": track_class
            },
            "uncertainty_scale": 1.0,
        }
        semantic_config_path.write_text(
            yaml.safe_dump(semantic_config, sort_keys=True), encoding="utf-8"
        )

        manifest = build_challenger_manifest(
            repo_root=Path.cwd(),
            candidate_id=candidate_id,
            variant_id=variant_id,
            feature_schema=f"q1-research-{target_event.event_id}-{checkpoint}-v1",
            input_snapshot_ids=[f"q1_research::{event_id}" for event_id in used_events],
            cutoff_at=cutoff_dt,
            simulation_counts={
                "qualifying": self._qualifying_simulations,
                "race": self._race_simulations,
            },
            config_paths=[*DEFAULT_CONFIG_PATHS, semantic_config_path],
            metadata={
                "research_gate_relaxation": {"q1": relaxed_floor},
                "retrospective_diagnostic": True,
            },
        )

        model_metadata = {
            "candidate_id": candidate_id,
            "checkpoint": checkpoint,
            "session_kind": target_event.session_kind,
            "dry_only": True,
            "cutoff_timestamp": cutoff_text,
            "max_input_timestamp": max_input_dt.isoformat().replace("+00:00", "Z"),
        }
        model_payload = {
            **model.to_dict(),
            "training_metadata": model_metadata,
            "generated_at": now.isoformat(),
        }
        model_path = (
            candidate_root / "models" / target_event.session_kind / f"{checkpoint.lower()}.json"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8")

        normalization_metadata = {**model_metadata, "track_class": track_class}
        # Same prior-fallback normalization used to build every prior event's
        # feature rows above (see _q1_prior_event_feature_rows) -- recomputed here
        # (cheap, deterministic, no fitted state) rather than threaded through.
        normalization_payload = {
            "artifact_type": "qualifying_practice_normalization",
            "schema_version": 1,
            "normalization": fit_practice_normalization(None).to_dict(),
            "training_metadata": normalization_metadata,
            "generated_at": now.isoformat(),
        }
        normalization_path = (
            candidate_root
            / "normalizations"
            / target_event.session_kind
            / checkpoint.lower()
            / f"{track_class}.json"
        )
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        normalization_path.write_text(
            json.dumps(normalization_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        bundle = build_qualifying_practice_bundle(
            candidate_id=candidate_id,
            variant_id=variant_id,
            manifest=manifest,
            bundle_directory=candidate_root,
            model_paths=[model_path],
            normalization_paths=[normalization_path],
        )
        bundle_path = candidate_root / "bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

        launch_envelope = build_qualifying_practice_launch_envelope(
            candidate_id=candidate_id,
            variant_id=variant_id,
            manifest=manifest,
            bundle_path=bundle_path,
            launch_directory=candidate_root,
            semantic_config_path=semantic_config_path,
            repo_root=Path.cwd(),
        )
        launch_path = candidate_root / "launch.json"
        launch_path.write_text(
            json.dumps(launch_envelope, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Dry-run exactly what predict_qualifying will do at runtime: resolve this
        # envelope with the same inference_cutoff qualifying_mixin.py will derive
        # from the historical prediction_context. Never claim "fitted" for an
        # envelope that would silently fail closed to champion at predict time.
        # retrospective_diagnostic=True is the user-authorized escape hatch for
        # exactly this case (a bundle built today for an already-completed fold);
        # every other check (leakage, hashes, disjointness) stays fully enforced,
        # and the manifest metadata above already carries the permanent label that
        # governance/preregistration/frozen-forecast paths all reject.
        resolution_cutoff = self._q1_runtime_inference_cutoff(target_event)
        try:
            resolve_qualifying_practice_launch_envelope(
                launch_path,
                expected_variant_id=variant_id,
                event_year=int(raw_target.get("year", 2026)),
                race_name=str(raw_target["race_name"]),
                checkpoint=checkpoint,
                session_kind=target_event.session_kind,
                inference_cutoff=resolution_cutoff,
                require_normalization=True,
                retrospective_diagnostic=True,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return None, {
                "status": "runtime_resolution_failed",
                "reason": str(exc),
                "resolution_cutoff": resolution_cutoff.isoformat(),
                "envelope_created_at": launch_envelope["created_at"],
                "n_training_events": n_training_events,
                "rows": int(len(dataset)),
                "note": (
                    "the fit itself succeeded (see rows/n_training_events); the "
                    "production Q1 launch envelope is a live-shadow design that "
                    "requires created_at <= inference_cutoff and never allows "
                    "backdating created_at, so a bundle built today can never "
                    "resolve against a fold whose own historical cutoff has "
                    "already passed -- a structural mismatch with backtesting, "
                    "not a data or fitting limitation"
                ),
            }

        return launch_path, {
            "status": "fitted",
            "n_training_events": n_training_events,
            "events_used": used_events,
            "rows": int(len(dataset)),
            "regularization_c": regularization_c,
            "shrinkage_scaled_by": "n_training_events/30",
            "launch_envelope_path": str(launch_path),
            "runtime_resolution_verified": True,
        }

    def _q1_runtime_inference_cutoff(self, event: ReplayEvent) -> datetime:
        """Return the exact inference_cutoff qualifying_mixin.py will derive for Q1.

        Mirrors ``q1_inference_cutoff`` in ``qualifying_mixin.py``: the historical
        prediction context's ``as_of_datetime`` (defaulting to ``target_session_
        datetime``) for the main-qualifying target session.
        """
        context = self._historical_context(event=event, target_session="Q").normalized()
        cutoff = context.as_of_datetime or context.target_session_datetime
        if cutoff is None:
            raise ValueError(f"no resolvable Q session date for {event.event_id}")
        return cutoff

    # -- disk-backed prediction cache (resumability) --------------------------------

    def _cache_path(self, key: dict[str, Any]) -> Path:
        digest = _digest(key)
        return self._prediction_cache_root / f"{digest[:2]}" / f"{digest}.json"

    def _cached(self, key: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("key") != key:
            return None
        return cast(dict[str, Any], payload["value"])

    def _store(self, key: dict[str, Any], value: Mapping[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"key": key, "value": value}, sort_keys=True, default=str), encoding="utf-8"
        )

    # -- WalkForwardBackend protocol --------------------------------------------

    def fit_fold(
        self,
        *,
        training_events: Sequence[ReplayEvent],
        calibration_events: Sequence[ReplayEvent],
        target_event: ReplayEvent,
        checkpoint: str,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        variant_id = str(manifest["variant_id"]).strip().lower()
        components = VARIANT_COMPONENTS[variant_id]
        # Force-build (and cache) the fitted state now so predict_* calls are pure
        # reads; this is also where a real train/calibration split would be consumed
        # for a variant that needs one (e.g. Q1's practice model). No currently-run
        # variant needs a train/calibration-dependent fit beyond team/driver state,
        # which uses every prior event regardless of the runner's holdout split.
        self._checkpoint_state_for(event_id=target_event.event_id, checkpoint=checkpoint)
        checkpoint_payload = target_event.checkpoint_payloads[checkpoint]
        cutoff_at = checkpoint_payload["information_cutoff_at"]
        max_input = (
            training_events[-1].event_start_at.isoformat()
            if training_events
            else (target_event.event_start_at.isoformat())
        )
        if training_events and training_events[-1].event_start_at >= target_event.event_start_at:
            raise ValueError("fit_fold received a non-chronological training event")

        relaxation = cast(Mapping[str, Any], manifest.get("metadata", {})).get(
            "research_gate_relaxation"
        )

        q1_launch: dict[str, Any] | None = None
        if "q1" in components and isinstance(relaxation, Mapping) and "q1" in relaxation:
            # Real attempt, real refusal: the fitter's own contract filters to an
            # exact (checkpoint, session_kind, track_class) before fitting anything
            # (see docs/QUALIFYING_RACE_CHALLENGER.md) rather than pooling unlike
            # tracks. Uses the curated data/historical_replay/2026/track_class_by_
            # event.json binding (scripts/build_track_class_binding.py); an unbound
            # event, or too few prior same-class dry events at the research floor,
            # fails this event-checkpoint closed rather than pooling unlike tracks.
            matching_events, eligibility_reason = self._q1_track_class_eligibility(
                target_event, relaxed_floor=int(relaxation["q1"])
            )
            if eligibility_reason is not None:
                raise CheckpointInputUnavailable(
                    f"{target_event.event_id} {checkpoint}: q1 research fit refused -- "
                    f"{eligibility_reason}"
                )
            if checkpoint not in _PRACTICE_SESSIONS_FOR_CHECKPOINT:
                raise CheckpointInputUnavailable(
                    f"{target_event.event_id} {checkpoint}: q1 research fit refused -- "
                    "the practice-comparison fitter needs a real FP checkpoint "
                    "(PRE has no practice evidence to fit from in this research pass)"
                )
            track_class = self._research_track_class(target_event.event_id)
            assert track_class is not None  # guaranteed by the eligibility check above
            launch_path, fit_diagnostics = self._fit_q1_research_launch(
                target_event=target_event,
                checkpoint=checkpoint,
                matching_event_ids=matching_events,
                relaxed_floor=int(relaxation["q1"]),
                track_class=track_class,
            )
            if launch_path is None:
                raise CheckpointInputUnavailable(
                    f"{target_event.event_id} {checkpoint}: q1 research fit refused -- "
                    f"eligible by track_class/count ({len(matching_events)} prior same-class "
                    f"dry events: {matching_events}) but the fitter itself could not "
                    f"produce a model: {fit_diagnostics}"
                )
            q1_launch = {"launch_envelope_path": str(launch_path), **fit_diagnostics}

        anchor_calibration: dict[str, Any] | None = None
        if (
            "r2_source_anchor" in components
            and isinstance(relaxation, Mapping)
            and "r2_source_anchor" in relaxation
        ):
            anchor_calibration = self._fit_r2_source_anchor(
                training_events=training_events,
                target_event=target_event,
                relaxed_floor=int(relaxation["r2_source_anchor"]),
            )

        fold_artifacts = {
            "variant_id": variant_id,
            "checkpoint": checkpoint,
            "session_kind": target_event.session_kind,
            "target_event_id": target_event.event_id,
            "training_event_ids": [event.event_id for event in training_events],
            "calibration_event_ids": [event.event_id for event in calibration_events],
            "cutoff_at": cutoff_at,
            "max_input_timestamp": max_input,
            "anchor_calibration": anchor_calibration,
            "q1_launch": q1_launch,
        }
        # run_challenger_walk_forward's own output only keeps a sha256 digest of
        # fold_artifacts (tamper-evidence, not readability); persist the full
        # research-only detail (calibrated weight, shrinkage, row counts) so the
        # report can show real numbers instead of an opaque hash.
        self._store(
            {
                "kind": "fold_artifacts",
                "source_digest": self._source_digest,
                "variant_id": variant_id,
                "checkpoint": checkpoint,
                "target_event_id": target_event.event_id,
            },
            fold_artifacts,
        )
        return fold_artifacts

    def predict_qualifying(
        self,
        *,
        role: Literal["champion", "challenger"],
        seed: int,
        event: ReplayEvent,
        checkpoint: str,
        checkpoint_payload: Mapping[str, Any],
        fold_artifacts: Mapping[str, Any] | None,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        variant_id = (
            CHAMPION_VARIANT if role == "champion" else str(manifest["variant_id"]).strip().lower()
        )
        components = VARIANT_COMPONENTS[variant_id]
        cache_key = {
            "kind": "qualifying",
            "source_digest": self._source_digest,
            "event_id": event.event_id,
            "checkpoint": checkpoint,
            "variant_id": variant_id,
            "seed": int(seed),
        }
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        state_dir = self._checkpoint_state_for(event_id=event.event_id, checkpoint=checkpoint)
        q1_launch = (fold_artifacts or {}).get("q1_launch") if role == "challenger" else None
        q1_override = (
            {
                "baseline_predictor.qualifying.practice_challenger.launch_envelope_path": (
                    q1_launch["launch_envelope_path"]
                )
            }
            if isinstance(q1_launch, Mapping) and q1_launch.get("launch_envelope_path")
            else None
        )
        predictor = self._predictor(
            state_dir=state_dir,
            variant_id=variant_id,
            seed=seed,
            event=event,
            checkpoint=checkpoint,
            extra_config_overrides=q1_override,
        )
        raw = self._raw_by_id[event.event_id]
        result = predictor.predict_qualifying(
            year=int(raw.get("year", 2026)),
            race_name=str(raw["race_name"]),
            n_simulations=self._qualifying_simulations,
            qualifying_stage="main",
            practice_signal_mode="stored_profiles",
            checkpoint_session_name=checkpoint,
            weather="dry",
            include_grid_scenarios=("r1" in components),
            include_challenger_evidence=("q1" in components),
            prediction_context=self._historical_context(event=event, target_session="Q"),
            q1_retrospective_diagnostic=q1_override is not None,
        )
        payload = deepcopy(dict(result))
        payload.pop("qualifying_practice_evidence", None)
        payload["model_variant"] = variant_id
        self._store(cache_key, payload)
        return payload

    def _r0_evidence(self, *, event: ReplayEvent, checkpoint: str) -> Mapping[str, Any] | None:
        """Extract long-run practice evidence for this event's pre-cutoff sessions."""
        raw = self._raw_by_id[event.event_id]
        checkpoint_payload = raw["checkpoint_payloads"].get(checkpoint, {})
        sessions_available = list(checkpoint_payload.get("sessions_available", []))
        if not sessions_available or not event.is_dry:
            return None
        try:
            import fastf1

            fastf1.Cache.enable_cache(str(raw["fastf1_cache_dir"]))
            frames = {}
            for label in sessions_available:
                session = fastf1.get_session(
                    int(raw.get("year", 2026)), str(raw["race_name"]), label
                )
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                frames[_SESSION_LABEL_TO_CODE[label]] = session.laps
        except Exception:  # noqa: BLE001 - fail closed: no evidence rather than invented evidence
            logger.warning(
                "r0 evidence extraction failed closed for %s %s", event.event_id, checkpoint
            )
            return None
        return build_race_practice_evidence(
            frames,
            year=int(raw.get("year", 2026)),
            event_name=str(raw["race_name"]),
            checkpoint=checkpoint,
            weather="dry",
        )

    def predict_race_views(
        self,
        *,
        role: Literal["champion", "challenger"],
        seed: int,
        event: ReplayEvent,
        checkpoint: str,
        checkpoint_payload: Mapping[str, Any],
        qualifying_prediction: Mapping[str, Any],
        fold_artifacts: Mapping[str, Any] | None,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]:
        variant_id = (
            CHAMPION_VARIANT if role == "champion" else str(manifest["variant_id"]).strip().lower()
        )
        components = VARIANT_COMPONENTS[variant_id]
        cache_key = {
            "kind": "race_views",
            "source_digest": self._source_digest,
            "event_id": event.event_id,
            "checkpoint": checkpoint,
            "variant_id": variant_id,
            "seed": int(seed),
        }
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        raw = self._raw_by_id[event.event_id]
        year = int(raw.get("year", 2026))
        race_name = str(raw["race_name"])
        state_dir = self._checkpoint_state_for(event_id=event.event_id, checkpoint=checkpoint)
        race_practice_evidence = (
            self._r0_evidence(event=event, checkpoint=checkpoint) if "r0" in components else None
        )

        anchor_override: dict[str, Any] | None = None
        anchor_calibration = (
            (fold_artifacts or {}).get("anchor_calibration") if role == "challenger" else None
        )
        if (
            "r2_source_anchor" in components
            and isinstance(anchor_calibration, Mapping)
            and anchor_calibration.get("status") == "fitted"
        ):
            champion_cache_key = {**cache_key, "variant_id": CHAMPION_VARIANT}
            champion_cached = self._cached(champion_cache_key)
            champion_weight = None
            if champion_cached is not None:
                champion_weight = (
                    champion_cached.get("conditional_actual_grid", {})
                    .get("grid_anchor_diagnostics", {})
                    .get("champion_weight")
                )
            if champion_weight is not None:
                shrinkage = float(anchor_calibration["shrinkage_weight"])
                shrunk_weight = (
                    float(champion_weight) * (1.0 - shrinkage)
                    + float(anchor_calibration["calibrated_weight"]) * shrinkage
                )
                anchor_override = {
                    "baseline_predictor.race.grid_anchor.source_calibrated.actual_starting_grid": (
                        shrunk_weight
                    )
                }
            # else: champion's own race view for this exact (event, checkpoint,
            # seed) has not been cached yet -- run_challenger_walk_forward always
            # scores champion's race views before the challenger's for the same
            # seed, so this should not happen; if it ever does, no override is
            # injected and r2_source_anchor legitimately falls back to champion's
            # own weight, which the identity guard below will flag honestly.

        def factory(inner_seed: int) -> Any:
            return self._predictor(
                state_dir=state_dir,
                variant_id=variant_id,
                seed=inner_seed,
                event=event,
                checkpoint=checkpoint,
                extra_config_overrides=anchor_override,
            )

        predicted_grid = list(qualifying_prediction["grid"])
        predicted_scenarios = (
            qualifying_prediction.get("grid_scenarios") if "r1" in components else None
        )
        race_kwargs: dict[str, Any] = {
            "year": year,
            "race_name": race_name,
            "weather": "dry",
            "n_simulations": self._race_simulations,
            "is_sprint": event.session_kind == "sprint",
            "prediction_context": self._historical_context(event=event, target_session="R"),
        }
        if race_practice_evidence is not None:
            race_kwargs["race_practice_evidence"] = race_practice_evidence

        replay = run_dual_race_replay(
            predictor_factory=factory,
            actual_starting_grid=raw["actual_starting_grid"],
            predicted_qualifying_grid=predicted_grid,
            predicted_grid_scenarios=predicted_scenarios,
            race_kwargs=race_kwargs,
            seeds=[int(seed)],
        )
        views: dict[str, Any] = {}
        for view_name in ("conditional_actual_grid", "end_to_end_predicted_grid"):
            prediction = dict(replay["views"][view_name]["runs"][0]["prediction"])
            prediction["model_variant"] = variant_id
            views[view_name] = prediction
        if role == "challenger":
            self._apply_identity_guard(views=views, components=components, cache_key=cache_key)
        self._store(cache_key, views)
        return views

    # r1 alone only ever changes the end-to-end (predicted-grid) view by design (see
    # race_view_replay.py); every other race-affecting component can plausibly move
    # either view.
    _CONDITIONAL_VIEW_COMPONENTS = frozenset({"r0", "r2_no_anchor", "r2_source_anchor"})
    _END_TO_END_VIEW_COMPONENTS = frozenset({"r0", "r1", "r2_no_anchor", "r2_source_anchor"})

    def _apply_identity_guard(
        self,
        *,
        views: dict[str, Any],
        components: frozenset[str],
        cache_key: dict[str, Any],
    ) -> None:
        """Flag (never silently score) a challenger view that claims a race-side
        component but produced champion-identical predictions for this fold."""
        expectations = {
            "conditional_actual_grid": components & self._CONDITIONAL_VIEW_COMPONENTS,
            "end_to_end_predicted_grid": components & self._END_TO_END_VIEW_COMPONENTS,
        }
        champion_cached = self._cached({**cache_key, "variant_id": CHAMPION_VARIANT})
        if champion_cached is None:
            return
        for view_name, expected_components in expectations.items():
            if not expected_components:
                continue
            challenger_positions = {
                row["driver"]: row["position"] for row in views[view_name]["finish_order"]
            }
            champion_positions = {
                row["driver"]: row["position"] for row in champion_cached[view_name]["finish_order"]
            }
            if challenger_positions == champion_positions:
                views[view_name]["ineffective_for_fold"] = True
                views[view_name]["ineffective_reason"] = (
                    f"{sorted(expected_components)} claimed to affect {view_name} for this "
                    "fold but the finish order is byte-identical to champion's; not scored "
                    "as a differentiated research result"
                )
