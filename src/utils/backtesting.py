"""Backtesting and ablation helpers for season-level evaluation."""

from __future__ import annotations

import copy
import csv
import json
import logging
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import fastf1
import yaml  # type: ignore[import-untyped,unused-ignore]

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.data.track_data_loader import KNOWN_MAIN_RACE_LAPS
from src.utils.prediction_metrics import PredictionMetrics

logger = logging.getLogger(__name__)


class NestedDictConfig:
    """Minimal config adapter exposing dot-path `get` for predictor injection."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._data
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


def load_config_dict(config_path: str = "config/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file into a plain dictionary."""
    with open(config_path) as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must deserialize to a dictionary")
    return loaded


def parse_override_value(raw_value: str) -> Any:
    """Parse override value from CLI into a typed Python value."""
    stripped = raw_value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None

    try:
        # Supports numeric values and JSON literals like arrays/objects.
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _split_assignments(raw_assignments: str) -> list[str]:
    """Split comma-delimited assignments while respecting []/{} and quotes."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char: str | None = None

    for char in raw_assignments:
        if quote_char is not None:
            current.append(char)
            if char == quote_char:
                quote_char = None
            continue

        if char in {"'", '"'}:
            quote_char = char
            current.append(char)
            continue

        if char in {"[", "{", "("}:
            depth += 1
            current.append(char)
            continue
        if char in {"]", "}", ")"}:
            depth = max(0, depth - 1)
            current.append(char)
            continue

        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        tokens.append(tail)
    return tokens


def parse_experiment_spec(raw_spec: str) -> tuple[str, dict[str, Any]]:
    """Parse experiment CLI input in the form `name:key=value,key2=value2`."""
    if ":" not in raw_spec:
        name = raw_spec.strip()
        if not name:
            raise ValueError("Experiment name cannot be empty")
        return name, {}

    name, assignments = raw_spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid experiment spec '{raw_spec}': missing experiment name")

    overrides: dict[str, Any] = {}
    for assignment in _split_assignments(assignments):
        if "=" not in assignment:
            raise ValueError(
                f"Invalid assignment '{assignment}' in experiment '{name}'. "
                "Expected format key=value."
            )
        dotted_key, raw_value = assignment.split("=", 1)
        dotted_key = dotted_key.strip()
        if not dotted_key:
            raise ValueError(f"Invalid assignment '{assignment}' in experiment '{name}': empty key")
        overrides[dotted_key] = parse_override_value(raw_value)

    return name, overrides


def _set_nested_value(config_data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set `a.b.c` value in nested dictionaries, creating intermediate dicts."""
    parts = dotted_key.split(".")
    cursor = config_data
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def apply_config_overrides(
    base_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a deep-copied config dictionary with dotted-key overrides applied."""
    merged = copy.deepcopy(dict(base_config))
    if not overrides:
        return merged

    for dotted_key, value in overrides.items():
        _set_nested_value(merged, dotted_key, value)
    return merged


def get_races_for_year(year: int, max_races: int | None = None) -> list[str]:
    """Get race names for a season from FastF1 schedule with deterministic fallback."""
    races: list[str] = []

    try:
        schedule = fastf1.get_event_schedule(year)
        if "EventName" in schedule.columns:
            for _, row in schedule.iterrows():
                event_name = str(row.get("EventName", "")).strip()
                event_format = str(row.get("EventFormat", "")).strip().lower()
                if not event_name:
                    continue
                if event_format == "testing":
                    continue
                races.append(event_name)
    except (
        AttributeError,
        ConnectionError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(f"Could not load FastF1 schedule for {year}: {exc}")

    if not races:
        # Conservative deterministic fallback for offline use.
        races = list(KNOWN_MAIN_RACE_LAPS.keys())
        logger.warning(
            "Falling back to known race list from track metadata; schedule may be incomplete."
        )

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(races))
    if max_races is not None and max_races > 0:
        return deduped[:max_races]
    return deduped


def warm_fastf1_results_cache(
    *,
    year: int,
    race_names: Iterable[str],
    session_names: Iterable[str] = ("Q", "R"),
) -> list[dict[str, Any]]:
    """Attempt to prefetch result sessions into the local FastF1 cache.

    This is mainly for season backtests where we would rather fail early on one
    race than discover half a season later that the cache never filled.
    """
    reports: list[dict[str, Any]] = []
    for race_name in race_names:
        for session_name in session_names:
            try:
                session = fastf1.get_session(year, race_name, session_name)
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                results = getattr(session, "results", None)
                row_count = len(results) if results is not None else 0
                reports.append(
                    {
                        "race_name": race_name,
                        "session_name": session_name,
                        "status": "ok",
                        "rows_loaded": row_count,
                    }
                )
            except (
                AttributeError,
                ConnectionError,
                FileNotFoundError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                reports.append(
                    {
                        "race_name": race_name,
                        "session_name": session_name,
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return reports


def _normalize_ranked_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize result rows to include required fields for metric computation."""
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(entries, start=1):
        position = row.get("position", index)
        try:
            position_int = int(position)
        except (TypeError, ValueError):
            position_int = index
        normalized.append(
            {
                "position": position_int,
                "driver": str(row.get("driver", "")),
                "team": str(row.get("team", "")),
            }
        )
    return normalized


def _top_n_entries(entries: Iterable[dict[str, Any]], *, n: int = 10) -> list[dict[str, Any]]:
    """Return the first ``n`` normalized rows for human-readable backtest output."""
    return _normalize_ranked_entries(list(entries))[:n]


def _compute_ranked_metrics(
    predicted_entries: Iterable[dict[str, Any]],
    actual_entries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compute ranking metrics for one predicted classification."""
    predicted_norm = _normalize_ranked_entries(predicted_entries)
    actual_norm = _normalize_ranked_entries(actual_entries)
    podium = PredictionMetrics.podium_accuracy(predicted_norm, actual_norm)
    winner_correct = PredictionMetrics.winner_accuracy(predicted_norm, actual_norm)

    return {
        "mae": float(PredictionMetrics.mean_absolute_error(predicted_norm, actual_norm)),
        "exact_accuracy": float(PredictionMetrics.position_accuracy(predicted_norm, actual_norm)),
        "within_3": float(PredictionMetrics.within_n_positions(predicted_norm, actual_norm, n=3)),
        "top3_accuracy": float(podium["accuracy"]),
        "winner_correct": bool(winner_correct),
    }


def run_single_race_backtest(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    results_fetcher: Any = fetch_actual_session_results,
) -> dict[str, Any]:
    """Execute one race backtest and return metric payload or skip reason."""
    qualifying_actual = results_fetcher(year, race_name, "Q")
    race_actual = results_fetcher(year, race_name, "R")

    if not qualifying_actual or not race_actual:
        return {
            "race_name": race_name,
            "status": "skipped",
            "reason": "missing_actual_results",
        }

    try:
        qualifying_prediction = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            n_simulations=qualifying_simulations,
        )
        race_prediction = predictor.predict_race(
            qualifying_grid=qualifying_prediction["grid"],
            weather=weather,
            race_name=race_name,
            n_simulations=race_simulations,
        )
        qualifying_metrics = _compute_ranked_metrics(
            qualifying_prediction["grid"],
            qualifying_actual,
        )
        race_metrics = _compute_ranked_metrics(
            race_prediction["finish_order"],
            race_actual,
        )

        return {
            "race_name": race_name,
            "status": "ok",
            "qualifying_mae": qualifying_metrics["mae"],
            "qualifying_exact_accuracy": qualifying_metrics["exact_accuracy"],
            "qualifying_predicted_top10": _top_n_entries(qualifying_prediction["grid"]),
            "qualifying_actual_top10": _top_n_entries(qualifying_actual),
            "race_mae": race_metrics["mae"],
            "race_exact_accuracy": race_metrics["exact_accuracy"],
            "race_within_3": race_metrics["within_3"],
            "top3_accuracy": race_metrics["top3_accuracy"],
            "winner_correct": race_metrics["winner_correct"],
            "race_predicted_top10": _top_n_entries(race_prediction["finish_order"]),
            "race_actual_top10": _top_n_entries(race_actual),
        }
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(f"Backtest failed for {race_name}: {exc}")
        return {
            "race_name": race_name,
            "status": "skipped",
            "reason": f"prediction_error:{type(exc).__name__}",
        }


def run_previous_race_naive_backtest(
    *,
    year: int,
    race_names: Iterable[str],
    results_fetcher: Any = fetch_actual_session_results,
) -> dict[str, Any]:
    """Evaluate a previous-race classification baseline across an ordered schedule."""
    race_results: list[dict[str, Any]] = []
    previous_qualifying_actual: list[dict[str, Any]] | None = None
    previous_race_actual: list[dict[str, Any]] | None = None
    previous_race_name: str | None = None

    for race_name in race_names:
        qualifying_actual = results_fetcher(year, race_name, "Q")
        race_actual = results_fetcher(year, race_name, "R")

        if not qualifying_actual or not race_actual:
            race_results.append(
                {
                    "race_name": race_name,
                    "status": "skipped",
                    "reason": "missing_actual_results",
                }
            )
            previous_qualifying_actual = None
            previous_race_actual = None
            previous_race_name = None
            continue

        if previous_qualifying_actual is None or previous_race_actual is None:
            race_results.append(
                {
                    "race_name": race_name,
                    "status": "skipped",
                    "reason": "missing_previous_race_results",
                }
            )
        else:
            qualifying_metrics = _compute_ranked_metrics(
                previous_qualifying_actual, qualifying_actual
            )
            race_metrics = _compute_ranked_metrics(previous_race_actual, race_actual)
            race_results.append(
                {
                    "race_name": race_name,
                    "status": "ok",
                    "baseline_name": "previous_race_classification",
                    "predicted_from_race": previous_race_name,
                    "qualifying_mae": qualifying_metrics["mae"],
                    "qualifying_exact_accuracy": qualifying_metrics["exact_accuracy"],
                    "qualifying_predicted_top10": _top_n_entries(previous_qualifying_actual),
                    "qualifying_actual_top10": _top_n_entries(qualifying_actual),
                    "race_mae": race_metrics["mae"],
                    "race_exact_accuracy": race_metrics["exact_accuracy"],
                    "race_within_3": race_metrics["within_3"],
                    "top3_accuracy": race_metrics["top3_accuracy"],
                    "winner_correct": race_metrics["winner_correct"],
                    "race_predicted_top10": _top_n_entries(previous_race_actual),
                    "race_actual_top10": _top_n_entries(race_actual),
                }
            )

        previous_qualifying_actual = list(qualifying_actual)
        previous_race_actual = list(race_actual)
        previous_race_name = race_name

    return {
        "name": "previous_race_classification",
        "summary": aggregate_race_metrics(race_results),
        "race_results": race_results,
    }


def aggregate_race_metrics(race_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate race-level metrics into season summary statistics."""
    successful = [row for row in race_results if row.get("status") == "ok"]
    skipped = [row for row in race_results if row.get("status") != "ok"]

    summary: dict[str, Any] = {
        "races_total": len(race_results),
        "races_evaluated": len(successful),
        "races_skipped": len(skipped),
        "skipped_races": [
            {"race_name": row.get("race_name"), "reason": row.get("reason", "unknown")}
            for row in skipped
        ],
    }
    if not successful:
        return summary

    def _mean(key: str) -> float | None:
        values = [float(row[key]) for row in successful if key in row and row.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    winner_accuracy = (
        sum(1 for row in successful if row.get("winner_correct", False)) / len(successful)
    ) * 100.0

    summary.update(
        {
            "qualifying_mae_mean": _mean("qualifying_mae"),
            "qualifying_exact_accuracy_mean": _mean("qualifying_exact_accuracy"),
            "race_mae_mean": _mean("race_mae"),
            "race_exact_accuracy_mean": _mean("race_exact_accuracy"),
            "race_within_3_mean": _mean("race_within_3"),
            "top3_accuracy_mean": _mean("top3_accuracy"),
            "winner_accuracy_percent": winner_accuracy,
        }
    )
    return summary


def build_overlap_comparison(
    *,
    model_race_results: list[dict[str, Any]],
    naive_race_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare model and naive summaries on the races both could score."""
    model_by_race = {
        str(row.get("race_name")): row
        for row in model_race_results
        if row.get("status") == "ok" and row.get("race_name")
    }
    naive_by_race = {
        str(row.get("race_name")): row
        for row in naive_race_results
        if row.get("status") == "ok" and row.get("race_name")
    }
    shared_races = [race_name for race_name in model_by_race if race_name in naive_by_race]
    model_overlap = [model_by_race[race_name] for race_name in shared_races]
    naive_overlap = [naive_by_race[race_name] for race_name in shared_races]
    model_summary = aggregate_race_metrics(model_overlap)
    naive_summary = aggregate_race_metrics(naive_overlap)

    def _improvement(metric_key: str) -> float | None:
        model_metric = model_summary.get(metric_key)
        naive_metric = naive_summary.get(metric_key)
        if model_metric is None or naive_metric is None:
            return None
        return float(naive_metric) - float(model_metric)

    return {
        "model": model_summary,
        "naive": naive_summary,
        "qualifying_mae_improvement": _improvement("qualifying_mae_mean"),
        "race_mae_improvement": _improvement("race_mae_mean"),
        "races_evaluated": len(shared_races),
        "shared_races": shared_races,
    }


def build_checked_backtest_summary(
    *,
    year: int,
    baseline_report: dict[str, Any],
    naive_report: dict[str, Any],
    overlap_comparison: dict[str, Any],
    reports_dir: str,
) -> dict[str, Any]:
    """Build the checked-in season summary with both aggregate and race-level detail."""
    baseline_summary = dict(baseline_report.get("summary", {}))
    model_summary = {
        "name": str(baseline_report.get("name", "baseline")),
        **baseline_summary,
    }

    return {
        "season": int(year),
        "reports_dir": reports_dir,
        "model": model_summary,
        "baseline_report": dict(baseline_report),
        "naive_previous_race_baseline": dict(naive_report),
        "overlap_comparison": dict(overlap_comparison),
        "notes": [
            "baseline_report mirrors the baseline summary written under reports/backtest_2025/.",
            "Per-race rows keep predicted and actual top-10 classifications for qualifying and race results.",
            "Overlap comparison only scores races where both the model and naive baseline produced valid metrics.",
        ],
    }


def split_train_test_results(
    race_results: list[dict[str, Any]],
    train_fraction: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split successful race results into train/test slices for overfitting checks."""
    successful = [row for row in race_results if row.get("status") == "ok"]
    if len(successful) <= 1:
        return successful, []

    clamped_fraction = min(max(train_fraction, 0.1), 0.9)
    shuffled = successful.copy()
    random.Random(seed).shuffle(shuffled)

    train_size = int(round(len(shuffled) * clamped_fraction))
    train_size = max(1, min(train_size, len(shuffled) - 1))

    return shuffled[:train_size], shuffled[train_size:]


def summarize_generalization(
    race_results: list[dict[str, Any]],
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict[str, Any]:
    """Return train/test summary plus generalization gap."""
    train_rows, test_rows = split_train_test_results(
        race_results=race_results,
        train_fraction=train_fraction,
        seed=seed,
    )

    train_summary = aggregate_race_metrics(train_rows)
    test_summary = aggregate_race_metrics(test_rows)

    train_race_mae = train_summary.get("race_mae_mean")
    test_race_mae = test_summary.get("race_mae_mean")
    if train_race_mae is None or test_race_mae is None:
        gap = None
    else:
        gap = float(test_race_mae) - float(train_race_mae)

    return {
        "train": train_summary,
        "test": test_summary,
        "generalization_gap_race_mae": gap,
    }


def rank_experiments_for_generalization(
    experiment_reports: list[dict[str, Any]],
    min_test_race_mae_improvement: float = 0.10,
    max_generalization_gap: float = 0.35,
) -> list[dict[str, Any]]:
    """Score and rank experiments against baseline with overfitting guardrails."""
    if not experiment_reports:
        return []

    baseline = next(
        (report for report in experiment_reports if report.get("name") == "baseline"),
        experiment_reports[0],
    )
    baseline_test_mae = baseline.get("generalization", {}).get("test", {}).get("race_mae_mean")

    ranked: list[dict[str, Any]] = []
    for report in experiment_reports:
        name = report.get("name", "unknown")
        generalization = report.get("generalization", {})
        train_mae = generalization.get("train", {}).get("race_mae_mean")
        test_mae = generalization.get("test", {}).get("race_mae_mean")
        gap = generalization.get("generalization_gap_race_mae")

        if baseline_test_mae is None or test_mae is None:
            improvement = None
            recommended = False
        else:
            improvement = float(baseline_test_mae) - float(test_mae)
            recommended = (
                name != baseline.get("name")
                and improvement >= min_test_race_mae_improvement
                and (gap is None or float(gap) <= max_generalization_gap)
            )

        ranked.append(
            {
                "name": name,
                "overrides": report.get("overrides", {}),
                "train_race_mae": train_mae,
                "test_race_mae": test_mae,
                "generalization_gap_race_mae": gap,
                "test_race_mae_improvement_vs_baseline": improvement,
                "recommended": recommended,
            }
        )

    return sorted(
        ranked,
        key=lambda item: (
            not bool(item["recommended"]),
            float("inf") if item["test_race_mae"] is None else float(item["test_race_mae"]),
        ),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write CSV table with explicit column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
