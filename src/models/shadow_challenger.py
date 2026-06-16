"""Background challenger predictions for target-specific model monitoring.

The challenger is intentionally simple and deterministic: it blends the served
champion rank with prior completed actual ranks for the same target. It is not
used for dashboard display; it is persisted only so future actuals can score
whether a target-specific current-form layer deserves promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from src.utils.accuracy_targets import (
    ALL_TARGET_KEYS,
    TARGET_GRAND_PRIX_RACE,
    TARGET_MAIN_QUALIFYING,
    TARGET_SPRINT_QUALIFYING,
    TARGET_SPRINT_RACE,
    explicit_target_actuals,
    explicit_target_predictions,
    sanitize_actual_rows,
    sanitize_prediction_rows,
    target_session_name,
)

SHADOW_CHALLENGER_SCHEMA_VERSION = 1
SHADOW_CHALLENGER_VERSION = "target_form_blend_v1"


@dataclass(frozen=True)
class ShadowChallengerRule:
    """Target-specific background challenger policy."""

    target_key: str
    challenger_name: str
    model_weight: float
    history_window: int
    min_history_events: int
    rationale: str


TARGET_SHADOW_RULES: dict[str, ShadowChallengerRule] = {
    TARGET_MAIN_QUALIFYING: ShadowChallengerRule(
        target_key=TARGET_MAIN_QUALIFYING,
        challenger_name="main_qualifying_form_blend_v1",
        model_weight=0.40,
        history_window=2,
        min_history_events=1,
        rationale="Qualifying challenger follows the audit signal that recent actual form is "
        "material, while retaining a model anchor.",
    ),
    TARGET_SPRINT_QUALIFYING: ShadowChallengerRule(
        target_key=TARGET_SPRINT_QUALIFYING,
        challenger_name="sprint_qualifying_form_blend_v1",
        model_weight=0.55,
        history_window=2,
        min_history_events=1,
        rationale="Sprint qualifying has fewer samples and less practice data, so the challenger "
        "keeps a stronger model anchor than main qualifying.",
    ),
    TARGET_GRAND_PRIX_RACE: ShadowChallengerRule(
        target_key=TARGET_GRAND_PRIX_RACE,
        challenger_name="grand_prix_race_form_blend_v1",
        model_weight=0.80,
        history_window=1,
        min_history_events=1,
        rationale="Grand Prix race audit favours a light model-aware previous-race correction.",
    ),
    TARGET_SPRINT_RACE: ShadowChallengerRule(
        target_key=TARGET_SPRINT_RACE,
        challenger_name="sprint_race_form_blend_v1",
        model_weight=0.75,
        history_window=2,
        min_history_events=1,
        rationale="Sprint race stays model-led but monitors a short recent-form correction.",
    ),
}


def build_shadow_challengers_for_prediction(
    prediction_data: dict[str, Any],
    *,
    historical_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build target-specific shadow challenger payloads for one prediction artifact."""
    target_predictions = explicit_target_predictions(prediction_data)
    if not target_predictions:
        return {}

    metadata = prediction_data.get("metadata", {})
    current_race = str(metadata.get("race_name", "")).strip()
    year = _coerce_int(metadata.get("year"))
    order_map = _calendar_order_map(year)
    current_order = order_map.get(_normalize_label(current_race))
    current_time = _parse_saved_datetime(
        metadata.get("information_cutoff_at") or metadata.get("predicted_at")
    )

    challengers: dict[str, Any] = {}
    for target_key in ALL_TARGET_KEYS:
        rule = TARGET_SHADOW_RULES.get(target_key)
        target_payload = target_predictions.get(target_key)
        if rule is None or not isinstance(target_payload, dict):
            continue

        champion_rows = sanitize_prediction_rows(target_payload.get("predicted_order"))
        if not champion_rows:
            continue

        history = _prior_target_actual_history(
            historical_predictions,
            target_key=target_key,
            current_race=current_race,
            current_order=current_order,
            current_time=current_time,
            order_map=order_map,
        )
        base_payload = {
            "schema_version": SHADOW_CHALLENGER_SCHEMA_VERSION,
            "challenger_version": SHADOW_CHALLENGER_VERSION,
            "challenger_name": rule.challenger_name,
            "target_key": target_key,
            "target_session": str(
                target_payload.get("target_session", target_session_name(target_key))
            )
            .strip()
            .upper(),
            "rule": asdict(rule),
            "history_events": len(history),
            "uses_current_event_actuals": False,
        }

        if len(history) < rule.min_history_events:
            challengers[target_key] = {
                **base_payload,
                "status": "insufficient_history",
                "predicted_order": [],
            }
            continue

        challengers[target_key] = {
            **base_payload,
            "status": "active",
            "predicted_order": _blend_rank_with_history(
                champion_rows,
                history,
                rule=rule,
            ),
        }

    return challengers


def build_shadow_challenger_for_target(
    champion_rows: list[dict[str, Any]],
    actual_history: list[list[dict[str, Any]]],
    *,
    target_key: str,
) -> list[dict[str, Any]]:
    """Build one target's challenger rows from explicit historical actuals."""
    rule = TARGET_SHADOW_RULES[target_key]
    return _blend_rank_with_history(champion_rows, actual_history, rule=rule)


def _prior_target_actual_history(
    historical_predictions: list[dict[str, Any]],
    *,
    target_key: str,
    current_race: str,
    current_order: int | None,
    current_time: datetime | None,
    order_map: dict[str, int],
) -> list[list[dict[str, Any]]]:
    """Return prior completed actual rows for one target without current-event leakage."""
    rows_by_race: dict[str, tuple[tuple[int, datetime], list[dict[str, Any]]]] = {}
    current_label = _normalize_label(current_race)

    for historical in historical_predictions:
        metadata = historical.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        race_name = str(metadata.get("race_name", "")).strip()
        race_label = _normalize_label(race_name)
        if not race_label or race_label == current_label:
            continue

        race_order = order_map.get(race_label)
        prediction_time = _parse_saved_datetime(
            metadata.get("information_cutoff_at") or metadata.get("predicted_at")
        )
        if current_order is not None and race_order is not None:
            if race_order >= current_order:
                continue
        elif current_time is not None and prediction_time is not None:
            if prediction_time >= current_time:
                continue

        actual_rows = sanitize_actual_rows(explicit_target_actuals(historical).get(target_key))
        if not actual_rows:
            continue

        sort_time = prediction_time or datetime.min.replace(tzinfo=UTC)
        sort_order = race_order if race_order is not None else 10_000
        existing = rows_by_race.get(race_label)
        if existing is None or (sort_order, sort_time) >= existing[0]:
            rows_by_race[race_label] = ((sort_order, sort_time), actual_rows)

    ordered = sorted(rows_by_race.values(), key=lambda item: item[0])
    return [actual_rows for _, actual_rows in ordered]


def _blend_rank_with_history(
    champion_rows: list[dict[str, Any]],
    actual_history: list[list[dict[str, Any]]],
    *,
    rule: ShadowChallengerRule,
) -> list[dict[str, Any]]:
    """Blend champion rank with recent same-target actual ranks."""
    recent_history = actual_history[-max(1, int(rule.history_window)) :]
    history_positions_by_driver: dict[str, list[int]] = {}
    for actual_rows in recent_history:
        for driver, position in _positions_by_driver(actual_rows).items():
            history_positions_by_driver.setdefault(driver, []).append(position)

    scored_rows: list[dict[str, Any]] = []
    champion_positions = _positions_by_driver(champion_rows)
    for default_position, row in enumerate(champion_rows, start=1):
        driver = str(row.get("driver", "")).strip()
        if not driver:
            continue
        champion_position = champion_positions.get(driver, default_position)
        history_positions = history_positions_by_driver.get(driver, [])
        history_position = (
            sum(history_positions) / len(history_positions)
            if history_positions
            else float(champion_position)
        )
        score = (float(rule.model_weight) * champion_position) + (
            (1.0 - float(rule.model_weight)) * history_position
        )
        scored = dict(row)
        scored["_shadow_score"] = float(score)
        scored["_champion_position"] = int(champion_position)
        scored_rows.append(scored)

    scored_rows.sort(
        key=lambda row: (
            float(row.get("_shadow_score", row.get("position", 999))),
            int(row.get("_champion_position", row.get("position", 999))),
            str(row.get("driver", "")),
        )
    )

    ranked: list[dict[str, Any]] = []
    for position, row in enumerate(scored_rows, start=1):
        champion_position = int(row.pop("_champion_position"))
        score = float(row.pop("_shadow_score"))
        adjusted = dict(row)
        adjusted["position"] = position
        adjusted["champion_position"] = champion_position
        adjusted["shadow_score"] = round(score, 4)
        adjusted["shadow_adjustment_positions"] = position - champion_position
        ranked.append(adjusted)
    return ranked


def _positions_by_driver(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return driver positions keyed by driver code."""
    positions: dict[str, int] = {}
    for default_position, row in enumerate(rows, start=1):
        driver = str(row.get("driver", "")).strip()
        if not driver:
            continue
        positions[driver] = _coerce_int(row.get("position")) or default_position
    return positions


def _calendar_order_map(year: int | None) -> dict[str, int]:
    """Return normalized race-name to calendar order when schedule data is available."""
    if year is None:
        return {}
    try:
        from src.utils.weekend import get_schedule_rows

        rows = get_schedule_rows(int(year))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    return {_normalize_label(race_name): index for index, (race_name, _) in enumerate(rows)}


def _parse_saved_datetime(value: Any) -> datetime | None:
    """Parse a saved ISO timestamp as UTC."""
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


def _normalize_label(value: Any) -> str:
    """Normalize race labels for matching."""
    return " ".join(str(value).split()).strip().casefold()


def _coerce_int(value: Any) -> int | None:
    """Return an int when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
