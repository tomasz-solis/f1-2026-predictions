"""Background session automation for post-session updates and predictions."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fastf1

from src.dashboard.cache import get_artifact_versions
from src.dashboard.live_prediction_flow import (
    prediction_payload_for_session,
    prediction_targets_for_checkpoint,
)
from src.dashboard.prediction_flow import run_prediction
from src.dashboard.update_flow import (
    auto_update_practice_characteristics_if_needed,
    detect_event_boundary_refresh_if_needed,
)
from src.persistence.artifact_store import ArtifactStore
from src.persistence.config import should_read_db_first, should_write_to_db, should_write_to_file
from src.persistence.runtime_state_store import RuntimeStateStore
from src.utils.accuracy_snapshots import build_accuracy_snapshot_records
from src.utils.accuracy_targets import (
    explicit_target_predictions,
    fastf1_session_name,
    legacy_target_keys_for_prediction,
    synthesize_legacy_targets,
    weekend_format_name,
)
from src.utils.actual_results_fetcher import fetch_actual_session_results
from src.utils.auto_updater import auto_update_from_races, needs_update
from src.utils.prediction_logger import PredictionLogger
from src.utils.prediction_metrics import PredictionMetrics
from src.utils.session_detector import SessionDetector
from src.utils.weekend import is_sprint_weekend

logger = logging.getLogger(__name__)

_SCHEDULE_NAMESPACE = "session_automation_schedule"
_SCHEDULE_STATE_FILE = Path("data/systems/session_automation_schedule.json")
ActualResultRows = Sequence[Mapping[str, Any]]


@dataclass
class SessionAutomationConfig:
    """Runtime configuration for the background session automation worker."""

    year: int
    enabled: bool = True
    auto_predict: bool = True
    weather: str = "dry"
    lookback_days: int = 14
    lookahead_days: int = 2
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_payload(cls, year: int, payload: dict[str, Any]) -> SessionAutomationConfig:
        """Build typed config from an untrusted payload."""
        weather_value = str(payload.get("weather", "dry")).strip().lower()
        if weather_value not in {"dry", "rain", "mixed"}:
            weather_value = "dry"
        return cls(
            year=int(year),
            enabled=bool(payload.get("enabled", True)),
            auto_predict=bool(payload.get("auto_predict", True)),
            weather=weather_value,
            lookback_days=max(1, int(payload.get("lookback_days", 14))),
            lookahead_days=max(0, int(payload.get("lookahead_days", 2))),
            updated_at=str(payload.get("updated_at", datetime.now(UTC).isoformat())),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize config for runtime-state persistence."""
        payload = asdict(self)
        payload["updated_at"] = datetime.now(UTC).isoformat()
        return payload


@dataclass
class SessionAutomationSummary:
    """Execution summary returned by one automation cycle."""

    year: int
    schedule_enabled: bool
    checked_events: int = 0
    learned_races: int = 0
    updated_practice_events: list[str] = field(default_factory=list)
    generated_predictions: list[str] = field(default_factory=list)
    skipped_predictions: list[str] = field(default_factory=list)
    reconciled_actuals: list[str] = field(default_factory=list)
    accuracy_snapshots: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for logging and CLI output."""
        return asdict(self)


def _runtime_state_store() -> RuntimeStateStore:
    """Create a runtime-state store instance for this process."""
    return RuntimeStateStore()


def _load_schedule_from_file(year: int) -> SessionAutomationConfig | None:
    """Load schedule config from local file fallback."""
    if not _SCHEDULE_STATE_FILE.exists():
        return None
    try:
        with open(_SCHEDULE_STATE_FILE) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    year_payload = payload.get(str(year))
    if not isinstance(year_payload, dict):
        return None
    return SessionAutomationConfig.from_payload(year, year_payload)


def _save_schedule_to_file(config: SessionAutomationConfig) -> None:
    """Persist schedule config to local file fallback."""
    if not should_write_to_file():
        return

    existing: dict[str, Any] = {}
    if _SCHEDULE_STATE_FILE.exists():
        try:
            with open(_SCHEDULE_STATE_FILE) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}

    existing[str(config.year)] = config.to_payload()
    _SCHEDULE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _SCHEDULE_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
    tmp_path.replace(_SCHEDULE_STATE_FILE)


def load_session_automation_config(year: int) -> SessionAutomationConfig:
    """Load schedule config from Supabase runtime state, then file fallback."""
    if should_read_db_first():
        try:
            record = _runtime_state_store().get_record(_SCHEDULE_NAMESPACE, str(year))
            if isinstance(record, dict):
                return SessionAutomationConfig.from_payload(year, record)
        except Exception as exc:
            logger.warning("Could not load session-automation config from DB for %s: %s", year, exc)

    file_config = _load_schedule_from_file(year)
    if file_config is not None:
        return file_config
    return SessionAutomationConfig(year=int(year))


def save_session_automation_config(config: SessionAutomationConfig) -> None:
    """Persist schedule config to configured backends."""
    payload = config.to_payload()
    if should_write_to_db():
        _runtime_state_store().upsert_record(_SCHEDULE_NAMESPACE, str(config.year), payload)
    _save_schedule_to_file(config)


def ensure_session_automation_config(
    year: int,
    *,
    enabled: bool = True,
    auto_predict: bool = True,
    weather: str | None = None,
    lookback_days: int = 14,
    lookahead_days: int = 2,
) -> SessionAutomationConfig:
    """Create or update schedule config for a season and return the stored values."""
    config = load_session_automation_config(year)
    config.enabled = bool(enabled)
    config.auto_predict = bool(auto_predict)
    if weather is not None:
        config.weather = weather if weather in {"dry", "rain", "mixed"} else "dry"
    config.lookback_days = max(1, int(lookback_days))
    config.lookahead_days = max(0, int(lookahead_days))
    save_session_automation_config(config)
    return config


def _coerce_utc_datetime(value: Any) -> datetime | None:
    """Convert FastF1 datetime-like values to UTC-aware datetimes."""
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


def _is_competitive_event(event_name: str, event_format: str) -> bool:
    """Return True when schedule row represents a race weekend."""
    normalized_name = str(event_name).strip().lower()
    normalized_format = str(event_format).strip().lower()
    if not normalized_name:
        return False
    if "testing" in normalized_name or "testing" in normalized_format:
        return False
    return True


def _iter_candidate_events(
    year: int,
    *,
    lookback_days: int,
    lookahead_days: int,
) -> list[tuple[str, bool]]:
    """
    Return race events likely to require session automation updates.

    The worker intentionally limits checks to a rolling window around "now" to
    avoid full-season polling every run.
    """
    now_utc = datetime.now(UTC)
    min_date = now_utc - timedelta(days=max(1, lookback_days))
    max_date = now_utc + timedelta(days=max(0, lookahead_days))
    candidates: list[tuple[str, bool]] = []
    seen: set[str] = set()

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:
        logger.warning("Could not load FastF1 schedule for session automation (%s): %s", year, exc)
        return candidates

    for _, event in schedule.iterrows():
        event_name = str(event.get("EventName", "")).strip()
        event_format = str(event.get("EventFormat", "")).strip()
        if not _is_competitive_event(event_name, event_format):
            continue

        event_date = _coerce_utc_datetime(event.get("EventDate"))
        if event_date is not None and not (min_date <= event_date <= max_date):
            continue

        if event_name in seen:
            continue
        seen.add(event_name)
        candidates.append((event_name, "sprint" in event_format.lower()))

    return candidates


def _generate_prediction_for_latest_session(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    latest_session: str,
    weather: str,
    prediction_logger: PredictionLogger,
) -> bool:
    """Generate and save prediction for the latest completed session."""
    if prediction_logger.has_prediction_for_session(year, race_name, latest_session):
        return False

    artifact_versions = get_artifact_versions(year=year)
    prediction_results = run_prediction(
        race_name,
        weather,
        artifact_versions,
        is_sprint=is_sprint,
        year=year,
    )
    qualifying_prediction, race_prediction, fp_blend_info = prediction_payload_for_session(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=latest_session,
    )
    target_predictions = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=latest_session,
    )
    if not target_predictions:
        return False
    qualifying_target, race_target = legacy_target_keys_for_prediction(
        latest_session,
        is_sprint=is_sprint,
    )
    prediction_logger.save_prediction(
        year=year,
        race_name=race_name,
        session_name=latest_session,
        qualifying_prediction=qualifying_prediction,
        race_prediction=race_prediction,
        weather=weather,
        fp_blend_info=fp_blend_info,
        target_predictions=target_predictions,
        metadata={
            "source": "session_automation",
            "latest_session": latest_session,
            "weekend_format": weekend_format_name(is_sprint),
            "top_level_qualifying_target": qualifying_target,
            "top_level_race_target": race_target,
            "top_level_qualifying_eligible_at_save": qualifying_target in target_predictions,
            "top_level_race_eligible_at_save": race_target in target_predictions,
        },
    )
    return True


def _actual_sessions_for_prediction(
    *,
    checkpoint_session: str,
    is_sprint: bool,
) -> tuple[str, str]:
    """
    Select qualifying/race actual sessions used for a checkpoint prediction.

    Sprint early-stage checkpoints are evaluated against sprint sessions, while
    later checkpoints are evaluated against main qualifying and race.
    """
    checkpoint_upper = str(checkpoint_session).strip().upper()
    if is_sprint and checkpoint_upper in {"FP1", "SQ", "SPRINT"}:
        return "SQ", "SPRINT"
    return "Q", "R"


def _store_accuracy_snapshot(
    *,
    year: int,
    race_name: str,
    session_name: str,
    is_sprint: bool,
    prediction_logger: PredictionLogger,
    metrics_calculator: PredictionMetrics,
    artifact_store: ArtifactStore,
) -> int:
    """Compute and persist accuracy snapshots for one prediction checkpoint."""
    prediction = prediction_logger.load_prediction(year, race_name, session_name)
    if prediction is None:
        return 0

    snapshot_records = build_accuracy_snapshot_records(
        prediction_data=prediction,
        is_sprint=is_sprint,
        metrics_calculator=metrics_calculator,
        generated_by="session_automation",
    )
    for record in snapshot_records:
        artifact_store.save_artifact(
            artifact_type="accuracy_snapshot",
            artifact_key=record["artifact_key"],
            version=1,
            data=record["data"],
        )
    return len(snapshot_records)


def _reconcile_prediction_actuals(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    prediction_logger: PredictionLogger,
    metrics_calculator: PredictionMetrics,
    artifact_store: ArtifactStore,
) -> tuple[int, int]:
    """
    Attach actuals to saved predictions for a completed race weekend.

    Returns:
        Tuple ``(predictions_reconciled, accuracy_snapshots_written)``.
    """
    actual_cache: dict[str, ActualResultRows | None] = {}
    reconciled = 0
    snapshots = 0

    race_predictions = [
        prediction
        for prediction in prediction_logger.get_all_predictions(year)
        if str((prediction.get("metadata") or {}).get("race_name", "")).strip() == race_name
    ]

    for prediction in race_predictions:
        metadata = prediction.get("metadata", {})
        session_name = str(metadata.get("session_name", "")).strip().upper()
        if not session_name:
            continue

        qualifying_actual_session, race_actual_session = _actual_sessions_for_prediction(
            checkpoint_session=session_name,
            is_sprint=is_sprint,
        )
        for actual_session in {qualifying_actual_session, race_actual_session}:
            if actual_session not in actual_cache:
                actual_cache[actual_session] = fetch_actual_session_results(
                    year,
                    race_name,
                    fastf1_session_name(actual_session),
                )

        target_predictions = explicit_target_predictions(prediction)
        if not target_predictions:
            target_predictions = synthesize_legacy_targets(prediction, is_sprint=is_sprint)
        target_actual_results: dict[str, ActualResultRows | None] = {}
        for target_key, payload in target_predictions.items():
            target_session = str(payload.get("target_session", ""))
            if not target_session:
                continue
            if target_session not in actual_cache:
                actual_cache[target_session] = fetch_actual_session_results(
                    year,
                    race_name,
                    fastf1_session_name(target_session),
                )
            target_actual_results[target_key] = actual_cache.get(target_session)

        qualifying_results = actual_cache.get(qualifying_actual_session)
        race_results = actual_cache.get(race_actual_session)
        if (
            qualifying_results is None
            and race_results is None
            and not any(rows is not None for rows in target_actual_results.values())
        ):
            continue

        updated = prediction_logger.update_actuals(
            year=year,
            race_name=race_name,
            session_name=session_name,
            qualifying_results=qualifying_results,
            race_results=race_results,
            target_actual_results=target_actual_results,
        )
        if not updated:
            continue

        reconciled += 1
        snapshots += _store_accuracy_snapshot(
            year=year,
            race_name=race_name,
            session_name=session_name,
            is_sprint=is_sprint,
            prediction_logger=prediction_logger,
            metrics_calculator=metrics_calculator,
            artifact_store=artifact_store,
        )

    return reconciled, snapshots


def run_session_automation_cycle(
    *,
    year: int,
    weather: str | None = None,
    auto_predict: bool | None = None,
    force_recheck: bool = False,
    reconcile_actuals: bool = True,
) -> SessionAutomationSummary:
    """
    Execute one end-to-end automation cycle independent of dashboard clicks.

    This function is intended for scheduled execution (cron/worker) and keeps
    writes idempotent via existing runtime-state and prediction dedupe logic.
    """
    config = load_session_automation_config(year)
    effective_weather = weather if weather in {"dry", "rain", "mixed"} else config.weather
    effective_auto_predict = config.auto_predict if auto_predict is None else bool(auto_predict)

    summary = SessionAutomationSummary(year=year, schedule_enabled=bool(config.enabled))
    if not config.enabled:
        logger.info("Session automation is disabled for %s; skipping cycle.", year)
        return summary

    try:
        has_new_races, race_candidates = needs_update(year=year, force_recheck=force_recheck)
    except TypeError:
        has_new_races, race_candidates = needs_update(force_recheck=force_recheck)

    if has_new_races:
        summary.learned_races = auto_update_from_races(
            races_to_update=race_candidates,
            year=year,
        )

    detector = SessionDetector()
    prediction_logger = PredictionLogger()
    metrics_calculator = PredictionMetrics()
    artifact_store = ArtifactStore(data_root="data")

    for race_name, race_is_sprint in _iter_candidate_events(
        year,
        lookback_days=config.lookback_days,
        lookahead_days=config.lookahead_days,
    ):
        summary.checked_events += 1

        is_sprint = bool(race_is_sprint)
        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except ValueError as exc:
            logger.warning(
                "Could not refresh weekend format for %s %s; using schedule-window value %s: %s",
                year,
                race_name,
                race_is_sprint,
                exc,
            )

        boundary_refresh = detect_event_boundary_refresh_if_needed(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            session_detector=detector,
        )
        latest_session = boundary_refresh.get("latest_elapsed_session")
        refresh_needed = bool(boundary_refresh.get("refresh_needed")) or force_recheck

        if refresh_needed and latest_session:
            practice_update = auto_update_practice_characteristics_if_needed(
                year=year,
                race_name=race_name,
                is_sprint=is_sprint,
                force_recheck=force_recheck,
                session_detector=detector,
            )
            if practice_update.get("updated"):
                summary.updated_practice_events.append(race_name)

        if latest_session and effective_auto_predict:
            generated = _generate_prediction_for_latest_session(
                year=year,
                race_name=race_name,
                is_sprint=is_sprint,
                latest_session=str(latest_session),
                weather=effective_weather,
                prediction_logger=prediction_logger,
            )
            session_label = f"{race_name}::{latest_session}"
            if generated:
                summary.generated_predictions.append(session_label)
            else:
                summary.skipped_predictions.append(session_label)

        if not reconcile_actuals:
            continue

        race_state = detector.get_session_completion_state(year, race_name, "R")
        if race_state != "completed":
            continue

        reconciled, snapshots = _reconcile_prediction_actuals(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            prediction_logger=prediction_logger,
            metrics_calculator=metrics_calculator,
            artifact_store=artifact_store,
        )
        if reconciled > 0:
            summary.reconciled_actuals.append(f"{race_name}::{reconciled}")
        summary.accuracy_snapshots += snapshots

    return summary
