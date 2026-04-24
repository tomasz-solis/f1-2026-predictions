"""Shared payload helpers for baseline predictor data loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from src.utils.accuracy_targets import explicit_target_actuals, synthesize_legacy_actuals
from src.utils.team_mapping import map_team_to_characteristics


def driver_characteristics_fallback_paths(data_dir: Path, year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallback candidates."""
    candidates: list[Path] = [
        data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json"
    ]
    nearest = nearest_season_payload_path(
        data_dir / "driver_characteristics",
        suffix="driver_characteristics",
        target_year=year,
    )
    if nearest is not None:
        _, nearest_path = nearest
        if nearest_path not in candidates:
            candidates.append(nearest_path)
    candidates.append(data_dir / "driver_characteristics.json")
    return tuple(candidates)


def nearest_season_payload_path(
    directory: Path,
    *,
    suffix: str,
    target_year: int,
) -> tuple[int, Path] | None:
    """Return the closest season-scoped payload file under one directory."""
    exact_path = directory / f"{target_year}_{suffix}.json"
    if exact_path.exists():
        return target_year, exact_path

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"*_{suffix}.json"):
        prefix = path.name.split("_", 1)[0].strip()
        if prefix.isdigit():
            candidates.append((int(prefix), path))

    if not candidates:
        return None

    return min(candidates, key=lambda item: (abs(item[0] - target_year), item[0]))


def infer_payload_year_from_path(path: Path, *, suffix: str) -> int | None:
    """Extract the season year from a `YYYY_<suffix>.json` filename."""
    prefix = path.name.removesuffix(f"_{suffix}.json").split("_", 1)[0].strip()
    if prefix.isdigit():
        return int(prefix)
    return None


def coerce_non_negative_int(value: object) -> int | None:
    """Convert an int-like value into a non-negative integer when possible."""
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int | float | np.integer | np.floating):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, str | bytes | bytearray):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return max(parsed, 0)


def sanitize_performance_observations(observations: object) -> list[float]:
    """Return a finite 0-1 performance series from a raw observations payload."""
    if not isinstance(observations, list):
        return []

    sanitized: list[float] = []
    for value in observations:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        sanitized.append(float(np.clip(numeric_value, 0.0, 1.0)))
    return sanitized


def extract_target_actual_rows(
    prediction_data: dict[str, object],
    *,
    target_key: str,
) -> list[dict[str, object]]:
    """Return canonical actual rows for one target from a saved prediction payload."""
    explicit_rows = explicit_target_actuals(prediction_data).get(target_key)
    if explicit_rows:
        return explicit_rows

    metadata = prediction_data.get("metadata", {})
    weekend_format = ""
    if isinstance(metadata, dict):
        weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    synthesized_targets = synthesize_legacy_actuals(
        prediction_data,
        is_sprint=weekend_format == "sprint",
    )
    return synthesized_targets.get(target_key, [])


def score_teams_from_actual_rows(
    actual_rows: list[dict[str, object]],
    *,
    known_teams: set[str],
) -> dict[str, float]:
    """Convert classified positions into rank-based team-form scores."""
    team_positions: dict[str, list[int]] = {}

    for row in actual_rows:
        raw_team = row.get("team")
        if not isinstance(raw_team, str) or not raw_team.strip():
            continue

        canonical_team = map_team_to_characteristics(raw_team, known_teams=known_teams)
        team_name = canonical_team if canonical_team else raw_team.strip()
        position = coerce_non_negative_int(row.get("position"))
        if position is None or position < 1:
            continue

        team_positions.setdefault(team_name, []).append(position)

    if not team_positions:
        return {}
    if len(team_positions) == 1:
        team_name = next(iter(team_positions))
        return {team_name: 0.5}

    team_avg = {
        team_name: float(np.mean(positions))
        for team_name, positions in team_positions.items()
        if positions
    }
    if not team_avg:
        return {}

    sorted_teams = sorted(team_avg, key=lambda team_name: team_avg[team_name])
    team_count = len(sorted_teams)
    scored_teams: dict[str, float] = {}
    for rank_index, team_name in enumerate(sorted_teams):
        scored_teams[team_name] = float(1.0 - (rank_index / max(team_count - 1, 1)))

    return scored_teams


def canonicalize_team_payload_keys(teams_payload: dict[str, object]) -> dict[str, dict]:
    """Canonicalize team payload keys and safely merge overlapping aliases."""
    canonical_payload: dict[str, dict] = {}
    for raw_team_name, raw_team_data in teams_payload.items():
        if not isinstance(raw_team_data, dict):
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name))
        team_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )

        existing = canonical_payload.get(team_name)
        if existing is None:
            canonical_payload[team_name] = deepcopy(raw_team_data)
            continue

        merged = _merge_team_payload(existing, raw_team_data)
        canonical_payload[team_name] = merged if isinstance(merged, dict) else existing

    return canonical_payload


def _is_missing_payload_value(value: object) -> bool:
    """Return True when payload value should be treated as missing during merge."""
    if value is None:
        return True
    if isinstance(value, float):
        return not np.isfinite(value)
    return False


def _merge_team_payload(existing: object, incoming: object) -> object:
    """Merge team payload fragments while preserving existing non-missing values."""
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = deepcopy(existing)
        for key, incoming_value in incoming.items():
            if key not in merged:
                merged[key] = deepcopy(incoming_value)
                continue
            merged[key] = _merge_team_payload(merged[key], incoming_value)
        return merged

    if _is_missing_payload_value(existing) and not _is_missing_payload_value(incoming):
        return deepcopy(incoming)
    return deepcopy(existing)
