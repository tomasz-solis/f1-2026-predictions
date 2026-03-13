"""Background warmup orchestration for precomputing dashboard predictions."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import fastf1

from src.dashboard.cache import get_artifact_versions, get_predictor
from src.dashboard.precomputed_predictions import (
    build_precomputed_base_features_key,
    build_precomputed_prediction_key,
    compute_artifact_hash,
    get_prediction_precompute_config,
    load_precomputed_base_features,
    load_precomputed_prediction,
    save_precompute_horizon_index,
    save_precomputed_base_features,
    save_precomputed_prediction,
)
from src.dashboard.prediction_flow import (
    _derive_race_input_confidence,
    _predict_race_with_optional_confidence,
    _predict_sprint_race_with_optional_confidence,
    build_actual_qualifying_section,
    build_actual_race_section,
    build_starting_grid_note,
    fetch_actual_competitive_results_if_completed,
    fetch_grid_if_available,
)
from src.dashboard.update_flow import (
    _boundary_signature,
    _build_event_boundary_snapshot,
    auto_update_practice_characteristics_if_needed,
)
from src.persistence.config import should_read_db_first, should_write_to_db
from src.persistence.runtime_state_store import RuntimeStateStore
from src.utils.session_detector import SessionDetector
from src.utils.weekend import is_sprint_weekend

logger = logging.getLogger(__name__)

_NOT_READY_STATUS_NAMESPACE = "prediction_precompute_not_ready"
_NOT_READY_MIN_WRITE_INTERVAL = timedelta(minutes=45)
_STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS = "precomputed_predictions"
_STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES = "precomputed_prediction_base_features"
_STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX = "prediction_precompute_horizon_index"
_WARMUP_LOCK_KEY_PREFIX = "prediction_precompute_warmup"
_WARMUP_LOCK_TTL_SECONDS = 5400
_DEFAULT_HORIZON_RACES = 3
_DEFAULT_WEATHER_SCENARIOS = ("dry", "mixed", "rain")
_VALID_WEATHER_SCENARIOS = frozenset(_DEFAULT_WEATHER_SCENARIOS)


@dataclass(frozen=True)
class WarmupTargets:
    """Anchor race and horizon races selected for a warmup cycle."""

    anchor_race_name: str
    anchor_is_sprint: bool
    target_races: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointContext:
    """Checkpoint readiness state derived from schedule plus data validation."""

    checkpoint: str
    expected_checkpoint: str
    latest_ready_checkpoint: str
    checkpoint_ready: bool
    reason: str
    boundary_signature: str


@dataclass
class WarmupSummary:
    """Structured summary returned by one warmup precompute cycle."""

    year: int
    status: str
    dry_run: bool = False
    anchor_race_name: str = ""
    checkpoint: str = "PRE"
    expected_checkpoint: str = "PRE"
    latest_ready_checkpoint: str = "PRE"
    reason: str = ""
    target_races: list[str] = field(default_factory=list)
    weather_scenarios: list[str] = field(default_factory=list)
    base_generated: int = 0
    base_reused: int = 0
    predictions_generated: int = 0
    predictions_reused: int = 0
    ready_races: list[str] = field(default_factory=list)
    practice_updated: bool = False
    practice_completed_sessions: list[str] = field(default_factory=list)
    practice_teams_updated: int = 0
    practice_retried_events: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    target_contexts: list[dict[str, Any]] = field(default_factory=list)
    db_verification_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary for logs and CLI output."""
        return {
            "year": self.year,
            "status": self.status,
            "dry_run": self.dry_run,
            "anchor_race_name": self.anchor_race_name,
            "checkpoint": self.checkpoint,
            "expected_checkpoint": self.expected_checkpoint,
            "latest_ready_checkpoint": self.latest_ready_checkpoint,
            "reason": self.reason,
            "target_races": list(self.target_races),
            "weather_scenarios": list(self.weather_scenarios),
            "base_generated": int(self.base_generated),
            "base_reused": int(self.base_reused),
            "predictions_generated": int(self.predictions_generated),
            "predictions_reused": int(self.predictions_reused),
            "ready_races": list(self.ready_races),
            "practice_updated": bool(self.practice_updated),
            "practice_completed_sessions": list(self.practice_completed_sessions),
            "practice_teams_updated": int(self.practice_teams_updated),
            "practice_retried_events": list(self.practice_retried_events),
            "errors": list(self.errors),
            "target_contexts": list(self.target_contexts),
            "db_verification_warnings": list(self.db_verification_warnings),
        }


def _coerce_utc_datetime(value: Any) -> datetime | None:
    """Normalize FastF1 datetime-like values to UTC-aware datetimes."""
    if value is None:
        return None

    candidate = value
    if hasattr(candidate, "to_pydatetime"):
        try:
            candidate = candidate.to_pydatetime()
        except Exception:
            return None

    if not isinstance(candidate, datetime):
        return None

    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=UTC)

    return candidate.astimezone(UTC)


def _parse_schedule_datetime(value: Any) -> datetime | None:
    """Parse serialized schedule timestamps produced by boundary snapshots."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_competitive_event(event_name: str, event_format: str) -> bool:
    """Return True when schedule row represents a race weekend."""
    normalized_name = str(event_name).strip().lower()
    normalized_format = str(event_format).strip().lower()
    if not normalized_name:
        return False
    if "testing" in normalized_name or "testing" in normalized_format:
        return False
    return True


def _normalize_weather_scenarios(raw_weather: Any) -> list[str]:
    """Normalize weather list to valid unique scenarios in deterministic order."""
    if not isinstance(raw_weather, list):
        return list(_DEFAULT_WEATHER_SCENARIOS)

    normalized: list[str] = []
    for item in raw_weather:
        value = str(item).strip().lower()
        if value in _VALID_WEATHER_SCENARIOS and value not in normalized:
            normalized.append(value)
    if normalized:
        return normalized
    return list(_DEFAULT_WEATHER_SCENARIOS)


def _resolve_warmup_targets(
    year: int, *, now_utc: datetime, horizon_races: int
) -> WarmupTargets | None:
    """Find anchor race (next upcoming) and horizon races from the current schedule."""
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:
        logger.warning("Could not load FastF1 schedule for warmup (%s): %s", year, exc)
        return None

    events: list[tuple[str, str, datetime]] = []
    for _, event in schedule.iterrows():
        event_name = str(event.get("EventName", "")).strip()
        event_format = str(event.get("EventFormat", "")).strip()
        if not _is_competitive_event(event_name, event_format):
            continue
        event_date = _coerce_utc_datetime(event.get("EventDate"))
        if event_date is None:
            continue
        events.append((event_name, event_format, event_date))

    if not events:
        return None

    anchor_index: int | None = None
    for index, (_, _, event_date) in enumerate(events):
        if event_date >= now_utc:
            anchor_index = index
            break
    if anchor_index is None:
        return None

    horizon_size = max(1, int(horizon_races))
    selected = events[anchor_index : anchor_index + horizon_size]
    if not selected:
        return None

    anchor_name, anchor_format, _ = selected[0]
    target_races = tuple(event_name for event_name, _, _ in selected if event_name)
    return WarmupTargets(
        anchor_race_name=anchor_name,
        anchor_is_sprint="sprint" in anchor_format.lower(),
        target_races=target_races,
    )


def _checkpoint_sessions(is_sprint: bool) -> tuple[tuple[str, str], ...]:
    """Return checkpoint and FastF1 session bindings for the weekend type."""
    if is_sprint:
        return (("FP1", "FP1"), ("SQ", "SQ"), ("Sprint", "Sprint"), ("Q", "Q"))
    return (("FP1", "FP1"), ("FP2", "FP2"), ("FP3", "FP3"), ("Q", "Q"))


def _resolve_checkpoint_context(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    now_utc: datetime,
    session_detector: SessionDetector,
) -> CheckpointContext:
    """Resolve expected and ready checkpoint labels using schedule and session readiness."""
    snapshot = _build_event_boundary_snapshot(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        session_detector=session_detector,
        now_utc=now_utc,
    )
    has_schedule_data = bool(snapshot.get("has_schedule_data"))
    boundary_signature = _boundary_signature(snapshot) if has_schedule_data else ""

    expected_checkpoint = "PRE"
    for checkpoint, session_name in _checkpoint_sessions(is_sprint):
        session_schedule = _parse_schedule_datetime(
            snapshot.get("session_schedule", {}).get(session_name, "")
        )
        if session_schedule is not None and now_utc >= session_schedule:
            expected_checkpoint = checkpoint

    latest_ready = "PRE"
    if not has_schedule_data:
        return CheckpointContext(
            checkpoint=expected_checkpoint,
            expected_checkpoint=expected_checkpoint,
            latest_ready_checkpoint=latest_ready,
            checkpoint_ready=False,
            reason="schedule_unavailable",
            boundary_signature=boundary_signature,
        )

    session_completion_raw = snapshot.get("session_completion", {})
    session_completion = session_completion_raw if isinstance(session_completion_raw, dict) else {}
    if expected_checkpoint == "PRE":
        return CheckpointContext(
            checkpoint=expected_checkpoint,
            expected_checkpoint=expected_checkpoint,
            latest_ready_checkpoint=latest_ready,
            checkpoint_ready=True,
            reason="ready",
            boundary_signature=boundary_signature,
        )

    for checkpoint, session_name in _checkpoint_sessions(is_sprint):
        is_completed = bool(session_completion.get(session_name, False))
        if not is_completed:
            return CheckpointContext(
                checkpoint=expected_checkpoint,
                expected_checkpoint=expected_checkpoint,
                latest_ready_checkpoint=latest_ready,
                checkpoint_ready=False,
                reason=f"{session_name}_not_ready",
                boundary_signature=boundary_signature,
            )
        latest_ready = checkpoint
        if checkpoint == expected_checkpoint:
            break

    is_ready = expected_checkpoint == latest_ready
    return CheckpointContext(
        checkpoint=expected_checkpoint,
        expected_checkpoint=expected_checkpoint,
        latest_ready_checkpoint=latest_ready,
        checkpoint_ready=is_ready,
        reason="ready" if is_ready else "checkpoint_not_ready",
        boundary_signature=boundary_signature,
    )


def _load_predictor(artifact_versions: dict[str, tuple[int, str]], *, year: int) -> Any:
    """Load predictor with compatibility fallback for older call signatures."""
    try:
        return get_predictor(artifact_versions, year=year)
    except TypeError:
        return get_predictor(artifact_versions)


def _refresh_anchor_practice_characteristics(
    *,
    year: int,
    targets: WarmupTargets,
    session_detector: SessionDetector,
) -> dict[str, Any]:
    """Refresh practice-derived car characteristics for the anchor race when available."""
    practice_update = auto_update_practice_characteristics_if_needed(
        year=int(year),
        race_name=targets.anchor_race_name,
        is_sprint=targets.anchor_is_sprint,
        force_recheck=False,
        session_detector=session_detector,
    )
    return practice_update if isinstance(practice_update, dict) else {}


def compute_base_features(
    year: int,
    target_race: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
    *,
    predictor: Any,
    is_sprint: bool,
) -> dict[str, Any]:
    """Load race inputs once and serialize weather-invariant features for reuse."""
    logger.debug(
        "Computing base features: year=%s race=%s checkpoint=%s artifact_hash=%s boundary=%s sprint=%s",
        int(year),
        str(target_race),
        str(checkpoint),
        str(artifact_hash),
        str(boundary_signature),
        bool(is_sprint),
    )
    base_start = time.time()

    if is_sprint:
        sprint_quali_start = time.time()
        sprint_actual_results, sprint_grid_source = fetch_actual_competitive_results_if_completed(
            year=year,
            race_name=target_race,
            session_name="SQ",
        )
        if sprint_actual_results is not None:
            sprint_grid = sprint_actual_results
            sprint_quali_payload = build_actual_qualifying_section(sprint_grid, session_name="SQ")
        else:
            sprint_quali = predictor.predict_qualifying(
                year=year,
                race_name=target_race,
                qualifying_stage="sprint",
            )
            sprint_grid, sprint_grid_source = fetch_grid_if_available(
                year,
                target_race,
                "SQ",
                sprint_quali["grid"],
            )
            if sprint_grid_source == "ACTUAL":
                sprint_quali_payload = build_actual_qualifying_section(
                    sprint_grid,
                    session_name="SQ",
                )
            else:
                sprint_quali_payload = deepcopy(sprint_quali)
                sprint_quali_payload["grid_source"] = sprint_grid_source
        sprint_quali_elapsed = time.time() - sprint_quali_start
        sprint_input_confidence = _derive_race_input_confidence(
            sprint_quali_payload,
            grid_source=sprint_grid_source,
        )

        main_quali_start = time.time()
        main_actual_results, main_grid_source = fetch_actual_competitive_results_if_completed(
            year=year,
            race_name=target_race,
            session_name="Q",
        )
        if main_actual_results is not None:
            main_grid = main_actual_results
            main_quali_payload = build_actual_qualifying_section(main_grid, session_name="Q")
        else:
            main_quali = predictor.predict_qualifying(
                year=year,
                race_name=target_race,
                qualifying_stage="main",
            )
            main_grid, main_grid_source = fetch_grid_if_available(
                year,
                target_race,
                "Q",
                main_quali["grid"],
            )
            if main_grid_source == "ACTUAL":
                main_quali_payload = build_actual_qualifying_section(
                    main_grid,
                    session_name="Q",
                )
            else:
                main_quali_payload = deepcopy(main_quali)
                main_quali_payload["grid_source"] = main_grid_source
        main_quali_elapsed = time.time() - main_quali_start
        main_input_confidence = _derive_race_input_confidence(
            main_quali_payload,
            grid_source=main_grid_source,
        )

        return {
            "is_sprint": True,
            "sprint_quali": sprint_quali_payload,
            "sprint_grid_for_race": sprint_grid,
            "sprint_race_input_confidence": float(sprint_input_confidence),
            "main_quali": main_quali_payload,
            "main_grid_for_race": main_grid,
            "main_race_input_confidence": float(main_input_confidence),
            "timing": {
                "sprint_quali": float(sprint_quali_elapsed),
                "main_quali": float(main_quali_elapsed),
                "base_total": float(time.time() - base_start),
            },
        }

    qualifying_start = time.time()
    actual_qualifying, grid_source = fetch_actual_competitive_results_if_completed(
        year=year,
        race_name=target_race,
        session_name="Q",
    )
    if actual_qualifying is not None:
        qualifying_grid = actual_qualifying
        qualifying_payload = build_actual_qualifying_section(qualifying_grid, session_name="Q")
    else:
        qualifying = predictor.predict_qualifying(
            year=year,
            race_name=target_race,
            qualifying_stage="main",
        )
        qualifying_grid, grid_source = fetch_grid_if_available(
            year,
            target_race,
            "Q",
            qualifying["grid"],
        )
        if grid_source == "ACTUAL":
            qualifying_payload = build_actual_qualifying_section(
                qualifying_grid,
                session_name="Q",
            )
        else:
            qualifying_payload = deepcopy(qualifying)
            qualifying_payload["grid_source"] = grid_source
    qualifying_elapsed = time.time() - qualifying_start
    race_input_confidence = _derive_race_input_confidence(
        qualifying_payload,
        grid_source=grid_source,
    )

    return {
        "is_sprint": False,
        "qualifying": qualifying_payload,
        "qualifying_grid_for_race": qualifying_grid,
        "race_input_confidence": float(race_input_confidence),
        "timing": {
            "qualifying": float(qualifying_elapsed),
            "base_total": float(time.time() - base_start),
        },
    }


def compute_weather_predictions(
    base_features: dict[str, Any],
    weather: str,
    *,
    predictor: Any,
    year: int,
    target_race: str,
) -> dict[str, Any]:
    """Apply weather overlay to base features and run weather-specific inference."""
    normalized_weather = str(weather).strip().lower()
    if normalized_weather not in _VALID_WEATHER_SCENARIOS:
        raise ValueError(
            f"Invalid weather scenario '{weather}'. Expected one of: {sorted(_VALID_WEATHER_SCENARIOS)}"
        )

    section_timing: dict[str, float] = {}
    if bool(base_features.get("is_sprint")):
        sprint_race_start = time.time()
        sprint_actual_results, _ = fetch_actual_competitive_results_if_completed(
            year=year,
            race_name=target_race,
            session_name="Sprint",
        )
        if sprint_actual_results is not None:
            sprint_race = build_actual_race_section(sprint_actual_results, session_name="Sprint")
        else:
            sprint_race = _predict_sprint_race_with_optional_confidence(
                predictor,
                sprint_quali_grid=deepcopy(base_features["sprint_grid_for_race"]),
                weather=normalized_weather,
                race_name=target_race,
                input_confidence=float(base_features.get("sprint_race_input_confidence", 0.0)),
            )
        sprint_race_elapsed = time.time() - sprint_race_start
        sprint_input_confidence = float(base_features.get("sprint_race_input_confidence", 0.0))
        sprint_grid_source = (
            str(base_features.get("sprint_quali", {}).get("grid_source", "PREDICTED"))
            .strip()
            .upper()
            or "PREDICTED"
        )
        sprint_race["grid_source"] = sprint_grid_source
        if sprint_grid_source == "ACTUAL":
            sprint_race["starting_grid_note"] = build_starting_grid_note("SQ")
        if str(sprint_race.get("result_mode", "")).upper() != "ACTUAL":
            sprint_race["input_confidence"] = round(sprint_input_confidence, 3)

        main_race_start = time.time()
        main_actual_results, _ = fetch_actual_competitive_results_if_completed(
            year=year,
            race_name=target_race,
            session_name="R",
        )
        if main_actual_results is not None:
            main_race = build_actual_race_section(main_actual_results, session_name="R")
        else:
            main_race = _predict_race_with_optional_confidence(
                predictor,
                qualifying_grid=deepcopy(base_features["main_grid_for_race"]),
                weather=normalized_weather,
                race_name=target_race,
                year=year,
                input_confidence=float(base_features.get("main_race_input_confidence", 0.0)),
            )
        main_race_elapsed = time.time() - main_race_start
        main_input_confidence = float(base_features.get("main_race_input_confidence", 0.0))
        main_grid_source = str(base_features.get("main_quali", {}).get("grid_source", "PREDICTED"))
        main_grid_source = main_grid_source.strip().upper() or "PREDICTED"
        main_race["grid_source"] = main_grid_source
        if main_grid_source == "ACTUAL":
            main_race["starting_grid_note"] = build_starting_grid_note("Q")
        if str(main_race.get("result_mode", "")).upper() != "ACTUAL":
            main_race["input_confidence"] = round(main_input_confidence, 3)

        section_timing = {
            "sprint_quali": float(base_features.get("timing", {}).get("sprint_quali", 0.0)),
            "sprint_race": float(sprint_race_elapsed),
            "main_quali": float(base_features.get("timing", {}).get("main_quali", 0.0)),
            "main_race": float(main_race_elapsed),
        }
        section_timing["total"] = (
            section_timing["sprint_quali"]
            + section_timing["sprint_race"]
            + section_timing["main_quali"]
            + section_timing["main_race"]
        )

        prediction_results = {
            "sprint_quali": deepcopy(base_features["sprint_quali"]),
            "sprint_race": sprint_race,
            "main_quali": deepcopy(base_features["main_quali"]),
            "main_race": main_race,
        }
        for section in prediction_results.values():
            if isinstance(section, dict):
                existing_timing = section.get("timing", {})
                merged_timing = dict(existing_timing) if isinstance(existing_timing, dict) else {}
                merged_timing.update(section_timing)
                section["timing"] = merged_timing
        return prediction_results

    race_start = time.time()
    actual_race_results, _ = fetch_actual_competitive_results_if_completed(
        year=year,
        race_name=target_race,
        session_name="R",
    )
    if actual_race_results is not None:
        race = build_actual_race_section(actual_race_results, session_name="R")
    else:
        race = _predict_race_with_optional_confidence(
            predictor,
            qualifying_grid=deepcopy(base_features["qualifying_grid_for_race"]),
            weather=normalized_weather,
            race_name=target_race,
            year=year,
            input_confidence=float(base_features.get("race_input_confidence", 0.0)),
        )
    race_elapsed = time.time() - race_start
    race_input_confidence = float(base_features.get("race_input_confidence", 0.0))
    race_grid_source = str(base_features.get("qualifying", {}).get("grid_source", "PREDICTED"))
    race_grid_source = race_grid_source.strip().upper() or "PREDICTED"
    race["grid_source"] = race_grid_source
    if race_grid_source == "ACTUAL":
        race["starting_grid_note"] = build_starting_grid_note("Q")
    if str(race.get("result_mode", "")).upper() != "ACTUAL":
        race["input_confidence"] = round(race_input_confidence, 3)

    section_timing = {
        "qualifying": float(base_features.get("timing", {}).get("qualifying", 0.0)),
        "race": float(race_elapsed),
    }
    section_timing["total"] = section_timing["qualifying"] + section_timing["race"]

    prediction_results = {
        "qualifying": deepcopy(base_features["qualifying"]),
        "race": race,
    }
    for section in prediction_results.values():
        if isinstance(section, dict):
            existing_timing = section.get("timing", {})
            merged_timing = dict(existing_timing) if isinstance(existing_timing, dict) else {}
            merged_timing.update(section_timing)
            section["timing"] = merged_timing
    return prediction_results


def _record_not_ready_status(
    *,
    year: int,
    anchor_race_name: str,
    context: CheckpointContext,
    now_utc: datetime,
) -> None:
    """Persist a throttled not-ready heartbeat so scheduler runs are observable."""
    state_key = f"{int(year)}::{anchor_race_name}"
    store = RuntimeStateStore()

    try:
        existing = store.get_record(_NOT_READY_STATUS_NAMESPACE, state_key)
    except Exception as exc:
        logger.debug("Could not read existing warmup status heartbeat: %s", exc)
        existing = None

    if isinstance(existing, dict):
        same_reason = str(existing.get("reason", "")).strip() == context.reason
        same_checkpoint = (
            str(existing.get("expected_checkpoint", "")).strip() == context.expected_checkpoint
        )
        if same_reason and same_checkpoint:
            updated_at = _parse_schedule_datetime(existing.get("updated_at"))
            if updated_at is not None:
                try:
                    if now_utc - updated_at < _NOT_READY_MIN_WRITE_INTERVAL:
                        return
                except TypeError:
                    logger.debug(
                        "Skipping not-ready throttle due to invalid timezone value for updated_at=%s",
                        existing.get("updated_at"),
                    )

    payload = {
        "year": int(year),
        "anchor_race_name": str(anchor_race_name).strip(),
        "expected_checkpoint": context.expected_checkpoint,
        "latest_ready_checkpoint": context.latest_ready_checkpoint,
        "reason": context.reason,
        "updated_at": now_utc.isoformat(),
    }
    try:
        store.upsert_record(_NOT_READY_STATUS_NAMESPACE, state_key, payload)
    except Exception as exc:
        logger.debug("Could not write warmup status heartbeat: %s", exc)


def _can_verify_db_writes() -> bool:
    """Return True when DB read-after-write verification is supported in current mode."""
    return should_write_to_db() and should_read_db_first()


def _verify_runtime_state_record(namespace: str, state_key: str) -> bool:
    """Verify that a runtime-state record exists immediately after write."""
    try:
        payload = RuntimeStateStore().get_record(namespace, state_key)
    except Exception:
        return False
    return isinstance(payload, dict)


def run_warmup_precompute_cycle(
    year: int,
    *,
    now_utc: datetime | None = None,
    dry_run: bool = False,
    verify_db_writes: bool = True,
) -> WarmupSummary:
    """Run one warmup cycle that precomputes horizon predictions outside Streamlit requests."""
    run_now = now_utc or datetime.now(UTC)
    summary = WarmupSummary(year=int(year), status="success", dry_run=bool(dry_run))

    db_verification_enabled = bool(verify_db_writes and _can_verify_db_writes())
    if verify_db_writes and not db_verification_enabled:
        summary.db_verification_warnings.append(
            "DB write verification is disabled because storage mode does not support DB read-after-write."
        )

    settings_raw = get_prediction_precompute_config()
    settings = settings_raw if isinstance(settings_raw, dict) else {}
    horizon_races = max(1, int(settings.get("horizon_races", _DEFAULT_HORIZON_RACES)))
    weather_scenarios = _normalize_weather_scenarios(
        settings.get("weather_scenarios", list(_DEFAULT_WEATHER_SCENARIOS))
    )
    summary.weather_scenarios = list(weather_scenarios)

    targets = _resolve_warmup_targets(
        int(year),
        now_utc=run_now,
        horizon_races=horizon_races,
    )
    if targets is None:
        summary.status = "nothing_to_do"
        summary.reason = "no_upcoming_races"
        return summary

    summary.anchor_race_name = targets.anchor_race_name
    summary.target_races = list(targets.target_races)

    detector = SessionDetector()
    checkpoint_context = _resolve_checkpoint_context(
        year=int(year),
        race_name=targets.anchor_race_name,
        is_sprint=targets.anchor_is_sprint,
        now_utc=run_now,
        session_detector=detector,
    )
    summary.checkpoint = checkpoint_context.checkpoint
    summary.expected_checkpoint = checkpoint_context.expected_checkpoint
    summary.latest_ready_checkpoint = checkpoint_context.latest_ready_checkpoint
    summary.reason = checkpoint_context.reason

    if not checkpoint_context.checkpoint_ready:
        summary.status = "not_ready"
        if not dry_run:
            _record_not_ready_status(
                year=int(year),
                anchor_race_name=targets.anchor_race_name,
                context=checkpoint_context,
                now_utc=run_now,
            )
        logger.info(
            "Warmup skipped: anchor=%s expected=%s ready=%s reason=%s",
            targets.anchor_race_name,
            checkpoint_context.expected_checkpoint,
            checkpoint_context.latest_ready_checkpoint,
            checkpoint_context.reason,
        )
        return summary

    lock_key = f"{_WARMUP_LOCK_KEY_PREFIX}::{int(year)}"
    lock_owner = uuid4().hex
    state_store = RuntimeStateStore()
    lock_acquired = True
    if should_write_to_db():
        lock_acquired = state_store.acquire_lock(
            lock_key,
            lock_owner,
            ttl_seconds=_WARMUP_LOCK_TTL_SECONDS,
        )
        if not lock_acquired:
            summary.status = "locked"
            summary.reason = "another_worker_holds_lock"
            logger.info("Warmup skipped: lock already held for key=%s", lock_key)
            return summary

    try:
        if not dry_run:
            practice_update = _refresh_anchor_practice_characteristics(
                year=int(year),
                targets=targets,
                session_detector=detector,
            )
            summary.practice_updated = bool(practice_update.get("updated"))
            completed_sessions = practice_update.get("completed_fp_sessions", [])
            if isinstance(completed_sessions, list):
                summary.practice_completed_sessions = [
                    str(session).strip() for session in completed_sessions if str(session).strip()
                ]
            summary.practice_teams_updated = int(practice_update.get("teams_updated", 0) or 0)
            retried_events = practice_update.get("retried_events", [])
            if isinstance(retried_events, list):
                summary.practice_retried_events = [
                    str(event_name).strip()
                    for event_name in retried_events
                    if str(event_name).strip()
                ]

        artifact_versions = get_artifact_versions(year=int(year))
        artifact_hash = compute_artifact_hash(artifact_versions)
        predictor = _load_predictor(artifact_versions, year=int(year))
        max_file_entries = max(16, int(settings.get("max_file_entries", 2048)))
        boundary_signature = checkpoint_context.boundary_signature
        race_boundaries: dict[str, str] = {}

        race_weather_coverage: dict[str, set[str]] = {
            race_name: set() for race_name in summary.target_races if race_name
        }
        for target_race in summary.target_races:
            try:
                target_is_sprint = bool(is_sprint_weekend(int(year), target_race))
            except Exception as exc:
                logger.warning(
                    "Could not determine weekend format for %s %s, defaulting to conventional: %s",
                    int(year),
                    target_race,
                    exc,
                )
                target_is_sprint = False

            if target_race == targets.anchor_race_name:
                target_checkpoint_context = checkpoint_context
            else:
                target_checkpoint_context = _resolve_checkpoint_context(
                    year=int(year),
                    race_name=target_race,
                    is_sprint=target_is_sprint,
                    now_utc=run_now,
                    session_detector=detector,
                )

            if not target_checkpoint_context.checkpoint_ready:
                logger.info(
                    "Warmup skipped race target due to checkpoint readiness: race=%s expected=%s ready=%s reason=%s",
                    target_race,
                    target_checkpoint_context.expected_checkpoint,
                    target_checkpoint_context.latest_ready_checkpoint,
                    target_checkpoint_context.reason,
                )
                continue

            target_boundary_signature = target_checkpoint_context.boundary_signature
            target_checkpoint = target_checkpoint_context.checkpoint
            race_boundaries[target_race] = target_boundary_signature
            summary.target_contexts.append(
                {
                    "race_name": target_race,
                    "is_sprint": bool(target_is_sprint),
                    "checkpoint": target_checkpoint,
                    "boundary_signature": target_boundary_signature,
                }
            )
            base_features = load_precomputed_base_features(
                year=int(year),
                race_name=target_race,
                checkpoint=target_checkpoint,
                artifact_hash=artifact_hash,
                boundary_signature=target_boundary_signature,
            )
            if base_features is None:
                if dry_run:
                    summary.base_generated += 1
                    base_features = {
                        "is_sprint": bool(target_is_sprint),
                        "timing": {},
                    }
                else:
                    try:
                        base_features = compute_base_features(
                            int(year),
                            target_race,
                            target_checkpoint,
                            artifact_hash,
                            target_boundary_signature,
                            predictor=predictor,
                            is_sprint=target_is_sprint,
                        )
                        save_precomputed_base_features(
                            year=int(year),
                            race_name=target_race,
                            checkpoint=target_checkpoint,
                            artifact_hash=artifact_hash,
                            boundary_signature=target_boundary_signature,
                            is_sprint=target_is_sprint,
                            base_features=base_features,
                            metadata={
                                "source_race_name": targets.anchor_race_name,
                                "boundary_session_name": target_checkpoint,
                            },
                            max_file_entries=max_file_entries,
                        )
                        if db_verification_enabled:
                            base_state_key = build_precomputed_base_features_key(
                                year=int(year),
                                race_name=target_race,
                                checkpoint=target_checkpoint,
                                artifact_hash=artifact_hash,
                                boundary_signature=target_boundary_signature,
                            )
                            verified = _verify_runtime_state_record(
                                _STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES,
                                base_state_key,
                            )
                            if not verified:
                                summary.db_verification_warnings.append(
                                    f"Base-feature write could not be verified in DB for {target_race} ({target_checkpoint})."
                                )
                        summary.base_generated += 1
                    except Exception as exc:
                        error_message = f"{target_race} [base_features]: {exc}"
                        summary.errors.append(error_message)
                        logger.warning("Warmup base-feature compute failed: %s", error_message)
                        continue
            else:
                summary.base_reused += 1

            for target_weather in weather_scenarios:
                persisted_prediction = load_precomputed_prediction(
                    year=int(year),
                    race_name=target_race,
                    weather=target_weather,
                    artifact_hash=artifact_hash,
                    boundary_signature=target_boundary_signature,
                )
                if persisted_prediction is not None:
                    summary.predictions_reused += 1
                    race_weather_coverage.setdefault(target_race, set()).add(target_weather)
                    continue

                if dry_run:
                    summary.predictions_generated += 1
                    race_weather_coverage.setdefault(target_race, set()).add(target_weather)
                    continue

                try:
                    prediction_results = compute_weather_predictions(
                        base_features,
                        target_weather,
                        predictor=predictor,
                        year=int(year),
                        target_race=target_race,
                    )
                    save_precomputed_prediction(
                        year=int(year),
                        race_name=target_race,
                        weather=target_weather,
                        artifact_hash=artifact_hash,
                        boundary_signature=target_boundary_signature,
                        is_sprint=target_is_sprint,
                        prediction_results=prediction_results,
                        metadata={
                            "source_race_name": targets.anchor_race_name,
                            "boundary_session_name": target_checkpoint,
                            "computed_from_base_features": True,
                        },
                        max_file_entries=max_file_entries,
                    )
                    if db_verification_enabled:
                        prediction_state_key = build_precomputed_prediction_key(
                            year=int(year),
                            race_name=target_race,
                            weather=target_weather,
                            artifact_hash=artifact_hash,
                            boundary_signature=target_boundary_signature,
                        )
                        verified = _verify_runtime_state_record(
                            _STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS,
                            prediction_state_key,
                        )
                        if not verified:
                            summary.db_verification_warnings.append(
                                f"Prediction write could not be verified in DB for {target_race} [{target_weather}]."
                            )
                    summary.predictions_generated += 1
                    race_weather_coverage.setdefault(target_race, set()).add(target_weather)
                except Exception as exc:
                    error_message = f"{target_race} [{target_weather}]: {exc}"
                    summary.errors.append(error_message)
                    logger.warning("Warmup weather precompute failed: %s", error_message)

        expected_weather = set(weather_scenarios)
        summary.ready_races = [
            race_name
            for race_name in summary.target_races
            if expected_weather.issubset(race_weather_coverage.get(race_name, set()))
        ]

        if not dry_run:
            try:
                save_precompute_horizon_index(
                    year=int(year),
                    artifact_hash=artifact_hash,
                    boundary_signature=boundary_signature,
                    anchor_race_name=targets.anchor_race_name,
                    anchor_session_name=checkpoint_context.checkpoint,
                    expected_targets=summary.target_races,
                    ready_races=summary.ready_races,
                    weather_scenarios=weather_scenarios,
                    race_boundaries=race_boundaries,
                )
                if db_verification_enabled:
                    horizon_state_key = f"{int(year)}::{str(artifact_hash).strip()}"
                    verified = _verify_runtime_state_record(
                        _STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX,
                        horizon_state_key,
                    )
                    if not verified:
                        summary.db_verification_warnings.append(
                            "Horizon-index write could not be verified in DB."
                        )
            except Exception as exc:
                error_message = f"horizon_index: {exc}"
                summary.errors.append(error_message)
                logger.warning("Could not persist warmup horizon index: %s", exc)

        if summary.errors:
            summary.status = "partial_success"
        elif dry_run:
            summary.status = "dry_run"

        return summary
    finally:
        if should_write_to_db() and lock_acquired:
            try:
                state_store.release_lock(lock_key, lock_owner)
            except Exception as exc:
                logger.warning("Could not release warmup lock %s: %s", lock_key, exc)
