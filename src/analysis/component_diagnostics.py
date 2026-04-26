"""Diagnostics for how challenger components move predictions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _coerce_int(value: Any) -> int | None:
    """Return an integer position when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _driver_positions(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return ranked positions keyed by driver code."""
    positions: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        driver = str(row.get("driver", "")).strip()
        position = _coerce_int(row.get("position"))
        if driver and position is not None:
            positions[driver] = position
    return positions


def _empty_session_summary() -> dict[str, Any]:
    """Return the no-data shape for one movement summary."""
    return {
        "rows": 0,
        "closer_count": 0,
        "farther_count": 0,
        "unchanged_count": 0,
        "mean_movement": None,
        "mean_absolute_movement": None,
        "mae_before": None,
        "mae_after": None,
        "mean_error_delta": None,
        "mean_reported_adjustment": None,
        "mean_absolute_reported_adjustment": None,
        "sample_rows": [],
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize challenger movement rows for one session or bucket."""
    if not rows:
        return _empty_session_summary()

    movement_values = [float(row["movement"]) for row in rows]
    abs_movement_values = [abs(value) for value in movement_values]
    before_errors = [float(row["before_error"]) for row in rows]
    after_errors = [float(row["after_error"]) for row in rows]
    error_deltas = [float(row["error_delta"]) for row in rows]
    reported_adjustments = [
        float(row["reported_adjustment"])
        for row in rows
        if row.get("reported_adjustment") is not None
    ]

    summary = {
        "rows": len(rows),
        "closer_count": sum(1 for row in rows if row["direction"] == "closer"),
        "farther_count": sum(1 for row in rows if row["direction"] == "farther"),
        "unchanged_count": sum(1 for row in rows if row["direction"] == "unchanged"),
        "mean_movement": sum(movement_values) / len(movement_values),
        "mean_absolute_movement": sum(abs_movement_values) / len(abs_movement_values),
        "mae_before": sum(before_errors) / len(before_errors),
        "mae_after": sum(after_errors) / len(after_errors),
        "mean_error_delta": sum(error_deltas) / len(error_deltas),
        "sample_rows": sorted(
            rows,
            key=lambda row: (row["error_delta"], abs(row["movement"])),
            reverse=True,
        )[:10],
    }
    if reported_adjustments:
        summary["mean_reported_adjustment"] = sum(reported_adjustments) / len(reported_adjustments)
        summary["mean_absolute_reported_adjustment"] = sum(
            abs(value) for value in reported_adjustments
        ) / len(reported_adjustments)
    else:
        summary["mean_reported_adjustment"] = None
        summary["mean_absolute_reported_adjustment"] = None
    return summary


def _movement_rows_for_session(
    *,
    champion_rows: Iterable[dict[str, Any]],
    challenger_rows: Iterable[dict[str, Any]],
    actual_rows: Iterable[dict[str, Any]],
    session: str,
    race_name: str,
    regime: str | None,
) -> list[dict[str, Any]]:
    """Build per-driver movement rows for one race/session."""
    champion_positions = _driver_positions(champion_rows)
    challenger_positions = _driver_positions(challenger_rows)
    actual_positions = _driver_positions(actual_rows)
    challenger_by_driver = {
        str(row.get("driver", "")).strip(): row
        for row in challenger_rows
        if isinstance(row, dict) and str(row.get("driver", "")).strip()
    }
    drivers = sorted(
        set(champion_positions).intersection(challenger_positions).intersection(actual_positions)
    )

    rows: list[dict[str, Any]] = []
    for driver in drivers:
        champion_position = champion_positions[driver]
        challenger_position = challenger_positions[driver]
        actual_position = actual_positions[driver]
        before_error = abs(champion_position - actual_position)
        after_error = abs(challenger_position - actual_position)
        error_delta = after_error - before_error
        if error_delta < 0:
            direction = "closer"
        elif error_delta > 0:
            direction = "farther"
        else:
            direction = "unchanged"

        rows.append(
            {
                "race_name": race_name,
                "session": session,
                "regime": regime,
                "driver": driver,
                "champion_position": champion_position,
                "challenger_position": challenger_position,
                "actual_position": actual_position,
                "movement": challenger_position - champion_position,
                "before_error": before_error,
                "after_error": after_error,
                "error_delta": error_delta,
                "direction": direction,
                "reported_adjustment": _reported_adjustment(
                    challenger_by_driver.get(driver, {}),
                    session=session,
                ),
            }
        )
    return rows


def _reported_adjustment(row: dict[str, Any], *, session: str) -> float | None:
    """Return a model-reported adjustment when the prediction row exposes one."""
    keys = (
        ("qualifying_residual_adjustment", "learned_position_adjustment")
        if session == "qualifying"
        else ("race_residual_adjustment", "learned_position_adjustment")
    )
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def build_component_movement_diagnostics(
    *,
    champion_report: dict[str, Any],
    challenger_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare champion and challenger movements against actual results.

    Negative ``mean_error_delta`` is good: the challenger moved predictions
    closer to actual finishing positions on average.
    """
    champion_by_race = {
        row.get("race_name"): row
        for row in champion_report.get("race_results", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    }
    qualifying_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []

    for challenger_row in challenger_report.get("race_results", []):
        if not isinstance(challenger_row, dict) or challenger_row.get("status") != "ok":
            continue
        race_name = str(challenger_row.get("race_name", "")).strip()
        champion_row = champion_by_race.get(race_name)
        if not isinstance(champion_row, dict):
            continue

        qualifying_rows.extend(
            _movement_rows_for_session(
                champion_rows=champion_row.get("qualifying_prediction_rows", []),
                challenger_rows=challenger_row.get("qualifying_prediction_rows", []),
                actual_rows=challenger_row.get("qualifying_actual_rows", []),
                session="qualifying",
                race_name=race_name,
                regime=str(challenger_row.get("qualifying_regime") or "").strip() or None,
            )
        )
        race_rows.extend(
            _movement_rows_for_session(
                champion_rows=champion_row.get("race_prediction_rows", []),
                challenger_rows=challenger_row.get("race_prediction_rows", []),
                actual_rows=challenger_row.get("race_actual_rows", []),
                session="race",
                race_name=race_name,
                regime=str(challenger_row.get("race_regime") or "").strip() or None,
            )
        )

    return {
        "qualifying": _summarize_rows(qualifying_rows),
        "race": _summarize_rows(race_rows),
    }
