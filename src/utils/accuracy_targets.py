"""Shared target-mapping helpers for prediction accuracy tracking."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.utils.data_paths import resolve_repo_data_path

TARGET_MAIN_QUALIFYING = "main_qualifying"
TARGET_GRAND_PRIX_RACE = "grand_prix_race"
TARGET_SPRINT_QUALIFYING = "sprint_qualifying"
TARGET_SPRINT_RACE = "sprint_race"

PRIMARY_TARGET_KEYS = (
    TARGET_MAIN_QUALIFYING,
    TARGET_GRAND_PRIX_RACE,
)
SECONDARY_SPRINT_TARGET_KEYS = (
    TARGET_SPRINT_QUALIFYING,
    TARGET_SPRINT_RACE,
)
ALL_TARGET_KEYS = PRIMARY_TARGET_KEYS + SECONDARY_SPRINT_TARGET_KEYS

TARGET_LABELS = {
    TARGET_MAIN_QUALIFYING: "Main Qualifying",
    TARGET_GRAND_PRIX_RACE: "Grand Prix Race",
    TARGET_SPRINT_QUALIFYING: "Sprint Qualifying",
    TARGET_SPRINT_RACE: "Sprint Race",
}
TARGET_SESSION_BY_KEY = {
    TARGET_MAIN_QUALIFYING: "Q",
    TARGET_GRAND_PRIX_RACE: "R",
    TARGET_SPRINT_QUALIFYING: "SQ",
    TARGET_SPRINT_RACE: "SPRINT",
}
CHECKPOINT_ORDER = {
    "PRE": 0,
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "Q": 6,
    "R": 7,
}
TARGET_CHECKPOINTS = {
    ("normal", TARGET_MAIN_QUALIFYING): ("PRE", "FP1", "FP2", "FP3"),
    ("normal", TARGET_GRAND_PRIX_RACE): ("PRE", "FP1", "FP2", "FP3", "Q"),
    ("sprint", TARGET_SPRINT_QUALIFYING): ("PRE", "FP1"),
    ("sprint", TARGET_SPRINT_RACE): ("PRE", "FP1", "SQ"),
    ("sprint", TARGET_MAIN_QUALIFYING): ("PRE", "FP1", "SQ"),
    ("sprint", TARGET_GRAND_PRIX_RACE): ("PRE", "FP1", "SQ", "Q"),
}
_EVENT_BOUNDARY_STATE_PATH = resolve_repo_data_path(
    "data/systems/event_boundary_refresh_state.json"
)


def normalize_checkpoint_session(session_name: str | None) -> str:
    """Normalize a checkpoint label into the stored uppercase form."""
    return str(session_name or "").strip().upper()


def weekend_format_name(is_sprint: bool) -> str:
    """Return the canonical weekend-format label stored in payloads."""
    return "sprint" if bool(is_sprint) else "normal"


def target_label(target_key: str) -> str:
    """Return the human-readable label for an accuracy target."""
    return TARGET_LABELS.get(str(target_key), str(target_key).replace("_", " ").title())


def target_session_name(target_key: str) -> str:
    """Return the competitive session code that resolves a target."""
    return TARGET_SESSION_BY_KEY[str(target_key)]


def fastf1_session_name(session_name: str) -> str:
    """Return the FastF1-compatible label for a stored session code."""
    normalized = normalize_checkpoint_session(session_name)
    if normalized == "SPRINT":
        return "Sprint"
    return normalized


def _parse_saved_datetime(value: Any) -> datetime | None:
    """Parse saved timestamps into UTC-aware datetimes."""
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


def _prediction_reference_datetime(metadata: dict[str, Any]) -> datetime | None:
    """Return the timestamp used to judge whether a forecast is contaminated."""
    information_cutoff = _parse_saved_datetime(metadata.get("information_cutoff_at"))
    if information_cutoff is not None:
        return information_cutoff
    return _parse_saved_datetime(metadata.get("predicted_at"))


def _prediction_is_sprint_weekend(prediction_data: dict[str, Any]) -> bool:
    """Infer weekend format from metadata or explicit sprint targets."""
    metadata = prediction_data.get("metadata", {})
    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format in {"normal", "sprint"}:
        return weekend_format == "sprint"

    explicit_targets = prediction_data.get("targets", {})
    if isinstance(explicit_targets, dict) and any(
        "sprint" in str(target_key) for target_key in explicit_targets
    ):
        return True

    checkpoint_session = normalize_checkpoint_session(metadata.get("session_name"))
    return checkpoint_session in {"SQ", "SPRINT"}


def _load_event_boundary_state() -> dict[str, Any]:
    """Load the latest known event-boundary schedule snapshot from local storage."""
    if not _EVENT_BOUNDARY_STATE_PATH.exists():
        return {}
    try:
        with _EVENT_BOUNDARY_STATE_PATH.open() as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scheduled_session_start(
    *,
    year: int,
    race_name: str,
    session_name: str,
) -> datetime | None:
    """Return scheduled session start from the locally persisted boundary snapshot."""
    state = _load_event_boundary_state()
    races = state.get("races", {})
    if not isinstance(races, dict):
        return None

    race_state = races.get(f"{int(year)}::{str(race_name).strip()}")
    if not isinstance(race_state, dict):
        return None

    session_schedule = race_state.get("session_schedule", {})
    if not isinstance(session_schedule, dict):
        return None

    fastf1_name = fastf1_session_name(session_name)
    raw_schedule = session_schedule.get(fastf1_name)
    return _parse_saved_datetime(raw_schedule)


def target_deadline_session(
    target_key: str,
    weekend_format: str,
    checkpoint_session: str,
) -> str | None:
    """
    Return the first session that would make a checkpoint-target forecast stale.

    For example, a sprint-weekend Grand Prix race forecast saved at `SPRINT`
    becomes contaminated once `Q` starts, while the same target saved at `Q`
    remains valid until `R` starts.
    """
    checkpoints = target_checkpoint_sequence(target_key, weekend_format)
    checkpoint = normalize_checkpoint_session(checkpoint_session)
    if checkpoint not in checkpoints:
        return None

    checkpoint_index = checkpoints.index(checkpoint)
    if checkpoint_index + 1 < len(checkpoints):
        return checkpoints[checkpoint_index + 1]
    return target_session_name(target_key)


def timing_eligible_target(
    prediction_data: dict[str, Any],
    *,
    target_key: str,
    is_sprint: bool,
) -> bool | None:
    """
    Return timestamp-based target eligibility when schedule data is available.

    A checkpoint prediction is considered valid only if it was saved before the
    next session that would reveal newer competitive information for that target.
    """
    metadata = prediction_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    try:
        year = int(metadata.get("year", 0) or 0)
    except (TypeError, ValueError):
        return None

    race_name = str(metadata.get("race_name", "")).strip()
    checkpoint_session = normalize_checkpoint_session(metadata.get("session_name"))
    predicted_at = _prediction_reference_datetime(metadata)
    if year <= 0 or not race_name or not checkpoint_session or predicted_at is None:
        return None

    weekend_format = weekend_format_name(is_sprint)
    deadline_session = target_deadline_session(
        target_key,
        weekend_format,
        checkpoint_session,
    )
    if not deadline_session:
        return None

    deadline_start = _scheduled_session_start(
        year=year,
        race_name=race_name,
        session_name=deadline_session,
    )
    if deadline_start is None:
        return None

    return predicted_at < deadline_start


def resolve_target_eligibility(
    prediction_data: dict[str, Any],
    *,
    target_key: str,
    stored_eligible: bool,
    is_sprint: bool,
) -> bool:
    """Combine stored eligibility with checkpoint-policy and timing-based checks."""
    metadata = prediction_data.get("metadata", {})
    checkpoint_session = normalize_checkpoint_session(
        metadata.get("session_name") if isinstance(metadata, dict) else None
    )
    weekend_format = weekend_format_name(is_sprint)
    if checkpoint_session and checkpoint_session not in target_checkpoint_sequence(
        target_key,
        weekend_format,
    ):
        return False

    time_eligible = timing_eligible_target(
        prediction_data,
        target_key=target_key,
        is_sprint=is_sprint,
    )
    if time_eligible is None:
        return bool(stored_eligible)
    return bool(stored_eligible) and bool(time_eligible)


def eligible_target_keys(checkpoint_session: str, is_sprint: bool) -> tuple[str, ...]:
    """Return the forecast targets that are still valid at a checkpoint."""
    checkpoint = normalize_checkpoint_session(checkpoint_session)
    weekend_format = weekend_format_name(is_sprint)
    return tuple(
        target_key
        for target_key in ALL_TARGET_KEYS
        if checkpoint in TARGET_CHECKPOINTS.get((weekend_format, target_key), ())
    )


def target_checkpoint_sequence(target_key: str, weekend_format: str) -> tuple[str, ...]:
    """Return the ordered checkpoints that can score a target."""
    return TARGET_CHECKPOINTS.get((str(weekend_format), str(target_key)), ())


def target_checkpoint_index(target_key: str, weekend_format: str, checkpoint_session: str) -> int:
    """Return the target-specific checkpoint order index used by charts."""
    checkpoints = target_checkpoint_sequence(target_key, weekend_format)
    checkpoint = normalize_checkpoint_session(checkpoint_session)
    try:
        return checkpoints.index(checkpoint)
    except ValueError:
        return CHECKPOINT_ORDER.get(checkpoint, 99)


def legacy_target_keys_for_prediction(
    checkpoint_session: str,
    *,
    is_sprint: bool,
) -> tuple[str | None, str | None]:
    """Map a legacy top-level qualifying and race pair to canonical targets."""
    checkpoint = normalize_checkpoint_session(checkpoint_session)
    if is_sprint and checkpoint in {"PRE", "FP1", "SQ", "SPRINT"}:
        return TARGET_SPRINT_QUALIFYING, TARGET_SPRINT_RACE
    return TARGET_MAIN_QUALIFYING, TARGET_GRAND_PRIX_RACE


def sanitize_prediction_rows(rows: Any) -> list[dict[str, Any]]:
    """Return stored prediction rows in a stable list-of-dicts form."""
    if not isinstance(rows, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        entry = dict(row)
        entry["driver"] = str(row.get("driver", "")).strip()
        entry["team"] = str(row.get("team", "")).strip()
        raw_position = row.get("position", index)
        try:
            entry["position"] = int(raw_position)
        except (TypeError, ValueError):
            entry["position"] = index
        if entry["driver"] and entry["team"]:
            sanitized.append(entry)
    return sanitized


def sanitize_actual_rows(rows: Any) -> list[dict[str, Any]]:
    """Return stored actual rows with only accuracy-relevant fields.

    A DNF signal is preserved when the source carries one (``dnf`` flag,
    ``status`` string, or ``classified`` boolean) so finisher-only and DNF
    calibration metrics can use it. Position-only actuals are unaffected.
    """
    sanitized_rows = sanitize_prediction_rows(rows)
    result: list[dict[str, Any]] = []
    for row in sanitized_rows:
        entry: dict[str, Any] = {
            "position": row["position"],
            "driver": row["driver"],
            "team": row["team"],
        }
        if "dnf" in row:
            entry["dnf"] = bool(row.get("dnf"))
        elif row.get("status"):
            entry["status"] = str(row.get("status"))
        elif "classified" in row and row.get("classified") is not None:
            entry["classified"] = bool(row.get("classified"))
        result.append(entry)
    return result


def mean_confidence_from_rows(rows: Any) -> float | None:
    """Return the mean row confidence when the payload exposes one."""
    if not isinstance(rows, list):
        return None
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_value = row.get("confidence")
        if raw_value is None:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return float(sum(values) / len(values))


def explicit_target_predictions(prediction_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return explicit target predictions when the payload already has them."""
    targets = prediction_data.get("targets")
    if not isinstance(targets, dict):
        return {}
    is_sprint = _prediction_is_sprint_weekend(prediction_data)

    normalized: dict[str, dict[str, Any]] = {}
    for target_key, payload in targets.items():
        if target_key not in ALL_TARGET_KEYS or not isinstance(payload, dict):
            continue
        rows = sanitize_prediction_rows(payload.get("predicted_order"))
        if not rows:
            continue
        normalized[target_key] = {
            "target_session": normalize_checkpoint_session(
                payload.get("target_session", target_session_name(target_key))
            ),
            "predicted_order": rows,
            "result_mode": str(payload.get("result_mode", "PREDICTED")).strip().upper(),
            "grid_source": str(payload.get("grid_source", "PREDICTED")).strip().upper(),
            "fp_blend_info": (
                payload.get("fp_blend_info")
                if isinstance(payload.get("fp_blend_info"), dict)
                else {}
            ),
            "mean_confidence": payload.get("mean_confidence"),
            "eligible_at_save": resolve_target_eligibility(
                prediction_data,
                target_key=target_key,
                stored_eligible=bool(payload.get("eligible_at_save", True)),
                is_sprint=is_sprint,
            ),
        }
    return normalized


def explicit_target_actuals(prediction_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return explicit target actuals when the payload already has them."""
    actuals = prediction_data.get("actuals")
    if not isinstance(actuals, dict):
        return {}
    targets = actuals.get("targets")
    if not isinstance(targets, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for target_key, rows in targets.items():
        if target_key not in ALL_TARGET_KEYS:
            continue
        sanitized = sanitize_actual_rows(rows)
        if sanitized:
            normalized[target_key] = sanitized
    return normalized


def synthesize_legacy_targets(
    prediction_data: dict[str, Any],
    *,
    is_sprint: bool,
) -> dict[str, dict[str, Any]]:
    """Build canonical targets from legacy top-level prediction fields."""
    metadata = prediction_data.get("metadata", {})
    checkpoint_session = normalize_checkpoint_session(metadata.get("session_name"))
    qualifying_target, race_target = legacy_target_keys_for_prediction(
        checkpoint_session,
        is_sprint=is_sprint,
    )

    normalized: dict[str, dict[str, Any]] = {}
    if qualifying_target is not None:
        qualifying_rows = sanitize_prediction_rows(
            (prediction_data.get("qualifying") or {}).get("predicted_grid")
        )
        if qualifying_rows:
            stored_eligible = bool(metadata.get("top_level_qualifying_eligible_at_save", True))
            normalized[qualifying_target] = {
                "target_session": target_session_name(qualifying_target),
                "predicted_order": qualifying_rows,
                "result_mode": str(metadata.get("top_level_qualifying_result_mode", "PREDICTED"))
                .strip()
                .upper(),
                "grid_source": str(metadata.get("top_level_qualifying_grid_source", "PREDICTED"))
                .strip()
                .upper(),
                "fp_blend_info": (
                    metadata.get("fp_blend_info")
                    if isinstance(metadata.get("fp_blend_info"), dict)
                    else {}
                ),
                "mean_confidence": mean_confidence_from_rows(qualifying_rows),
                "eligible_at_save": resolve_target_eligibility(
                    prediction_data,
                    target_key=qualifying_target,
                    stored_eligible=stored_eligible,
                    is_sprint=is_sprint,
                ),
            }

    if race_target is not None:
        race_rows = sanitize_prediction_rows(
            (prediction_data.get("race") or {}).get("predicted_results")
        )
        if race_rows:
            stored_eligible = bool(metadata.get("top_level_race_eligible_at_save", True))
            normalized[race_target] = {
                "target_session": target_session_name(race_target),
                "predicted_order": race_rows,
                "result_mode": str(metadata.get("top_level_race_result_mode", "PREDICTED"))
                .strip()
                .upper(),
                "grid_source": str(metadata.get("top_level_race_grid_source", "PREDICTED"))
                .strip()
                .upper(),
                "fp_blend_info": {},
                "mean_confidence": mean_confidence_from_rows(race_rows),
                "eligible_at_save": resolve_target_eligibility(
                    prediction_data,
                    target_key=race_target,
                    stored_eligible=stored_eligible,
                    is_sprint=is_sprint,
                ),
            }
    return normalized


def synthesize_legacy_actuals(
    prediction_data: dict[str, Any],
    *,
    is_sprint: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Build canonical target actuals from legacy top-level actual fields."""
    metadata = prediction_data.get("metadata", {})
    checkpoint_session = normalize_checkpoint_session(metadata.get("session_name"))
    qualifying_target, race_target = legacy_target_keys_for_prediction(
        checkpoint_session,
        is_sprint=is_sprint,
    )

    actuals = prediction_data.get("actuals", {})
    normalized: dict[str, list[dict[str, Any]]] = {}

    if qualifying_target is not None:
        qualifying_rows = sanitize_actual_rows(actuals.get("qualifying"))
        if qualifying_rows:
            normalized[qualifying_target] = qualifying_rows

    if race_target is not None:
        race_rows = sanitize_actual_rows(actuals.get("race"))
        if race_rows:
            normalized[race_target] = race_rows

    return normalized
