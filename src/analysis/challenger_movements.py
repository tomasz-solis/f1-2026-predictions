"""Full-field movement audit for qualifying challenger comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.analysis.challenger_governance import stable_json_sha256
from src.utils.grid_validation import validate_qualifying_grid

GRID_MOVEMENT_REVIEW_THRESHOLD = 2
H2H_MOVEMENT_REVIEW_THRESHOLD_PP = 10.0


def _grid_positions(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, int]:
    try:
        grid = validate_qualifying_grid(rows, require_sequential_positions=True)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    return {str(row["driver"]): int(row["position"]) for row in grid}


def _h2h_probabilities(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, str, str], float]:
    probabilities: dict[tuple[str, str, str], float] = {}
    for index, row in enumerate(rows):
        team = str(row.get("team", "")).strip()
        driver_a = str(row.get("driver_a", "")).strip()
        driver_b = str(row.get("driver_b", "")).strip()
        if not team or not driver_a or not driver_b or driver_a == driver_b:
            raise ValueError(f"{label}[{index}] has invalid teammate identities")
        first, second = sorted((driver_a, driver_b))
        raw_probability: Any = (
            row.get("p_driver_a_ahead") if first == driver_a else row.get("p_driver_b_ahead")
        )
        if raw_probability is None and first != driver_a:
            raw_a: Any = row.get("p_driver_a_ahead")
            raw_probability = None if raw_a is None else 1.0 - float(raw_a)
        if raw_probability is None:
            raise ValueError(f"{label}[{index}] has no numeric H2H probability")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] has no numeric H2H probability") from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{label}[{index}] H2H probability must be in [0, 1]")
        key = (team, first, second)
        if key in probabilities:
            raise ValueError(f"{label} contains duplicate teammate pair {key}")
        probabilities[key] = probability
    return probabilities


def build_full_field_movement_audit(
    *,
    champion_grid: Sequence[Mapping[str, Any]],
    challenger_grid: Sequence[Mapping[str, Any]],
    champion_teammate_h2h: Sequence[Mapping[str, Any]],
    challenger_teammate_h2h: Sequence[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the mandatory >2-position / >10-point review queue.

    Every common driver and teammate matchup is included, even when its movement
    is below the review threshold. A review counts only when its entry has a
    non-empty ``decision``; the caller owns the human judgment itself.
    """

    champion_positions = _grid_positions(champion_grid, label="champion_grid")
    challenger_positions = _grid_positions(challenger_grid, label="challenger_grid")
    if set(champion_positions) != set(challenger_positions):
        raise ValueError("champion and challenger grids must contain the same driver set")

    review_payloads = reviews or {}
    grid_rows: list[dict[str, Any]] = []
    for driver in sorted(champion_positions):
        champion_position = champion_positions[driver]
        challenger_position = challenger_positions[driver]
        delta = champion_position - challenger_position
        audit_id = f"grid:{driver}"
        requires_review = abs(delta) > GRID_MOVEMENT_REVIEW_THRESHOLD
        review = dict(review_payloads.get(audit_id, {}))
        reviewed = requires_review and bool(str(review.get("decision", "")).strip())
        grid_rows.append(
            {
                "audit_id": audit_id,
                "driver": driver,
                "champion_position": champion_position,
                "challenger_position": challenger_position,
                "position_change": delta,
                "absolute_position_change": abs(delta),
                "requires_review": requires_review,
                "reviewed": reviewed,
                "review": review if reviewed else None,
            }
        )

    champion_h2h = _h2h_probabilities(champion_teammate_h2h, label="champion_teammate_h2h")
    challenger_h2h = _h2h_probabilities(
        challenger_teammate_h2h,
        label="challenger_teammate_h2h",
    )
    if set(champion_h2h) != set(challenger_h2h):
        raise ValueError("champion and challenger H2H payloads must contain the same pairs")

    h2h_rows: list[dict[str, Any]] = []
    for team, driver_a, driver_b in sorted(champion_h2h):
        champion_probability = champion_h2h[(team, driver_a, driver_b)]
        challenger_probability = challenger_h2h[(team, driver_a, driver_b)]
        change_pp = (challenger_probability - champion_probability) * 100.0
        audit_id = f"h2h:{team}:{driver_a}:{driver_b}"
        requires_review = abs(change_pp) > H2H_MOVEMENT_REVIEW_THRESHOLD_PP
        review = dict(review_payloads.get(audit_id, {}))
        reviewed = requires_review and bool(str(review.get("decision", "")).strip())
        h2h_rows.append(
            {
                "audit_id": audit_id,
                "team": team,
                "driver_a": driver_a,
                "driver_b": driver_b,
                "probability_driver_a_champion": champion_probability,
                "probability_driver_a_challenger": challenger_probability,
                "probability_change_pp": change_pp,
                "absolute_probability_change_pp": abs(change_pp),
                "requires_review": requires_review,
                "reviewed": reviewed,
                "review": review if reviewed else None,
            }
        )

    material_rows = [row for row in (*grid_rows, *h2h_rows) if bool(row["requires_review"])]
    reviewed_count = sum(bool(row["reviewed"]) for row in material_rows)
    payload: dict[str, Any] = {
        "artifact_type": "qualifying_challenger_movement_audit",
        "schema_version": 1,
        "thresholds": {
            "grid_positions_strictly_above": GRID_MOVEMENT_REVIEW_THRESHOLD,
            "h2h_percentage_points_strictly_above": H2H_MOVEMENT_REVIEW_THRESHOLD_PP,
        },
        "field_size": len(grid_rows),
        "grid_movements": grid_rows,
        "teammate_h2h_movements": h2h_rows,
        "review_required_count": len(material_rows),
        "reviewed_count": reviewed_count,
        "review_complete": reviewed_count == len(material_rows),
    }
    payload["audit_sha256"] = stable_json_sha256(payload)
    return payload
