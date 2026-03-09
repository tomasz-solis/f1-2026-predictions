"""Shared target-mapping helpers for prediction accuracy tracking."""

from __future__ import annotations

from typing import Any

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
    ("sprint", TARGET_MAIN_QUALIFYING): ("PRE", "FP1", "SQ", "SPRINT"),
    ("sprint", TARGET_GRAND_PRIX_RACE): ("PRE", "FP1", "SQ", "SPRINT", "Q"),
}


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
    """Return stored actual rows with only accuracy-relevant fields."""
    sanitized_rows = sanitize_prediction_rows(rows)
    return [
        {
            "position": row["position"],
            "driver": row["driver"],
            "team": row["team"],
        }
        for row in sanitized_rows
    ]


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
            "eligible_at_save": bool(payload.get("eligible_at_save", True)),
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
                "eligible_at_save": bool(
                    metadata.get("top_level_qualifying_eligible_at_save", True)
                ),
            }

    if race_target is not None:
        race_rows = sanitize_prediction_rows(
            (prediction_data.get("race") or {}).get("predicted_results")
        )
        if race_rows:
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
                "eligible_at_save": bool(metadata.get("top_level_race_eligible_at_save", True)),
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
