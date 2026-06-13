"""Backtesting and ablation helpers for season-level evaluation."""

from __future__ import annotations

import copy
import csv
import json
import logging
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import fastf1
import yaml  # type: ignore[import-untyped,unused-ignore]

from src.analysis.model_evaluation import compute_calibration_metrics
from src.data.actual_results_fetcher import fetch_actual_session_results
from src.models.order_confidence import within_tolerance
from src.utils import config_loader
from src.utils.prediction_context import PredictionContext, build_historical_prediction_context
from src.utils.prediction_metrics import PredictionMetrics
from src.utils.weekend import get_weekend_type

logger = logging.getLogger(__name__)

EvaluationMode = Literal["historical", "live"]
LearningMode = Literal["adaptive", "static"]
_TRACK_CHARACTERISTICS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "track_characteristics"
)
_FALLBACK_SEASON_CALENDARS: dict[int, tuple[str, ...]] = {
    2022: (
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Emilia Romagna Grand Prix",
        "Miami Grand Prix",
        "Spanish Grand Prix",
        "Monaco Grand Prix",
        "Azerbaijan Grand Prix",
        "Canadian Grand Prix",
        "British Grand Prix",
        "Austrian Grand Prix",
        "French Grand Prix",
        "Hungarian Grand Prix",
        "Belgian Grand Prix",
        "Dutch Grand Prix",
        "Italian Grand Prix",
        "Singapore Grand Prix",
        "Japanese Grand Prix",
        "United States Grand Prix",
        "Mexico City Grand Prix",
        "São Paulo Grand Prix",
        "Abu Dhabi Grand Prix",
    ),
    2023: (
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Azerbaijan Grand Prix",
        "Miami Grand Prix",
        "Monaco Grand Prix",
        "Spanish Grand Prix",
        "Canadian Grand Prix",
        "Austrian Grand Prix",
        "British Grand Prix",
        "Hungarian Grand Prix",
        "Belgian Grand Prix",
        "Dutch Grand Prix",
        "Italian Grand Prix",
        "Singapore Grand Prix",
        "Japanese Grand Prix",
        "Qatar Grand Prix",
        "United States Grand Prix",
        "Mexico City Grand Prix",
        "São Paulo Grand Prix",
        "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix",
    ),
    2024: (
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Japanese Grand Prix",
        "Chinese Grand Prix",
        "Miami Grand Prix",
        "Emilia Romagna Grand Prix",
        "Monaco Grand Prix",
        "Canadian Grand Prix",
        "Spanish Grand Prix",
        "Austrian Grand Prix",
        "British Grand Prix",
        "Hungarian Grand Prix",
        "Belgian Grand Prix",
        "Dutch Grand Prix",
        "Italian Grand Prix",
        "Azerbaijan Grand Prix",
        "Singapore Grand Prix",
        "United States Grand Prix",
        "Mexico City Grand Prix",
        "São Paulo Grand Prix",
        "Las Vegas Grand Prix",
        "Qatar Grand Prix",
        "Abu Dhabi Grand Prix",
    ),
}


class NestedDictConfig:
    """Minimal config adapter exposing dot-path `get` for predictor injection."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by dot-notation key, returning default if not found."""
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
        logger.warning("Could not load FastF1 schedule for %s: %s", year, exc)

    if not races:
        races = list(_FALLBACK_SEASON_CALENDARS.get(int(year), ()))
        logger.warning(
            "Falling back to static %s season calendar; schedule may be incomplete.",
            year,
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


def _normalize_ranked_entries(
    entries: Iterable[dict[str, Any]],
    *,
    preserve_interval_fields: bool = False,
) -> list[dict[str, Any]]:
    """Normalize result rows to include stable ranking fields.

    When ``preserve_interval_fields`` is true, the returned rows also keep the
    optional interval metadata needed by adaptive calibration replay.
    """
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(entries, start=1):
        position = row.get("position", index)
        try:
            position_int = int(position)
        except (TypeError, ValueError):
            position_int = index
        normalized_row = {
            "position": position_int,
            "driver": str(row.get("driver", "")),
            "team": str(row.get("team", "")),
        }
        if preserve_interval_fields:
            for key in ("median_position", "p5", "p95"):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    normalized_row[key] = int(value)
                except (TypeError, ValueError):
                    continue
            for key in (
                "qualifying_residual_adjustment",
                "race_residual_adjustment",
                "learned_position_adjustment",
            ):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    normalized_row[key] = float(value)
                except (TypeError, ValueError):
                    continue
        normalized.append(normalized_row)
    return normalized


def _top_n_entries(entries: Iterable[dict[str, Any]], *, n: int = 10) -> list[dict[str, Any]]:
    """Return the first ``n`` normalized rows for human-readable backtest output."""
    return _normalize_ranked_entries(list(entries))[:n]


def _coerce_float(value: object) -> float | None:
    """Convert a scalar value to ``float`` when possible."""
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_track_metadata_map(year: int) -> dict[str, dict[str, Any]]:
    """Load local track metadata used for segmented backtest diagnostics."""
    candidate_years = [int(year)]
    if int(year) != 2026:
        candidate_years.append(2026)

    for candidate_year in candidate_years:
        path = _TRACK_CHARACTERISTICS_DIR / f"{candidate_year}_track_characteristics.json"
        if not path.exists():
            continue
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read track metadata from %s: %s", path, exc)
            continue

        tracks = payload.get("tracks", {})
        if isinstance(tracks, dict):
            return {
                str(track_name): track_payload
                for track_name, track_payload in tracks.items()
                if isinstance(track_payload, dict)
            }

    return {}


def _resolve_backtest_race_metadata(
    *,
    year: int,
    race_name: str,
    weather: str,
) -> dict[str, Any]:
    """Resolve stable race metadata for review-friendly segmented summaries."""
    weekend_format = "unknown"
    try:
        weekend_type = get_weekend_type(year, race_name)
        weekend_format = "sprint" if weekend_type == "sprint" else "normal"
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Could not resolve weekend format for %s %s: %s", year, race_name, exc)

    track_payload = _load_track_metadata_map(year).get(race_name, {})
    track_type = str(track_payload.get("type", "unknown") or "unknown").strip().lower()
    track_overtaking = _coerce_float(track_payload.get("overtaking_difficulty"))
    safety_car_probability = _coerce_float(track_payload.get("safety_car_prob"))

    return {
        "weather": str(weather or "unknown").strip().lower() or "unknown",
        "weekend_format": weekend_format,
        "track_type": track_type or "unknown",
        "track_overtaking_difficulty": track_overtaking,
        "safety_car_probability": safety_car_probability,
    }


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


def _compute_interval_metrics(
    predicted_entries: Iterable[dict[str, Any]],
    actual_entries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Measure empirical coverage for any rows that carry p5-p95 intervals."""
    actual_positions_by_driver: dict[str, int] = {}
    for row in actual_entries:
        driver_code = str(row.get("driver", "")).strip()
        raw_position: object = row.get("position")
        if not isinstance(raw_position, int | float | str):
            continue
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if driver_code:
            actual_positions_by_driver[driver_code] = position

    confidence_bands: list[tuple[float, float]] = []
    aligned_actual_positions: list[int] = []
    interval_hits = 0

    for row in predicted_entries:
        driver_code = str(row.get("driver", "")).strip()
        if not driver_code or driver_code not in actual_positions_by_driver:
            continue

        lower = _coerce_float(row.get("p5"))
        upper = _coerce_float(row.get("p95"))
        if lower is None or upper is None:
            continue

        actual_position = actual_positions_by_driver[driver_code]
        lower_bound = min(lower, upper)
        upper_bound = max(lower, upper)
        confidence_bands.append((lower_bound, upper_bound))
        aligned_actual_positions.append(actual_position)
        if lower_bound <= actual_position <= upper_bound:
            interval_hits += 1

    calibration = compute_calibration_metrics(confidence_bands, aligned_actual_positions)
    interval_count = int(calibration["interval_count"])
    if interval_count == 0:
        return {
            "interval_count": 0,
            "interval_hits": 0,
            "empirical_coverage": None,
            "nominal_coverage": None,
            "calibration_error": None,
            "mean_interval_width": None,
            "average_miss_distance": None,
        }

    return {
        "interval_count": interval_count,
        "interval_hits": interval_hits,
        "empirical_coverage": float(calibration["empirical_coverage"]),
        "nominal_coverage": float(calibration["nominal_coverage"]),
        "calibration_error": float(calibration["calibration_error"]),
        "mean_interval_width": float(calibration["mean_interval_width"]),
        "average_miss_distance": float(calibration["average_miss_distance"]),
    }


def _compute_order_confidence_metrics(
    predicted_entries: Iterable[dict[str, Any]],
    actual_entries: Iterable[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Measure calibration of the published ``order_confidence`` probability.

    For every predicted row that carries an ``order_confidence`` and matches an
    actual classification, we compare the stated probability against whether the
    driver actually finished within ``tolerance`` places of the predicted slot.
    ``calibration_error`` is ``mean_predicted - empirical_hit_rate`` (positive ⇒
    overconfident); driving it toward zero is what ``spread_inflation`` tunes.
    """
    actual_positions_by_driver: dict[str, int] = {}
    for row in actual_entries:
        driver_code = str(row.get("driver", "")).strip()
        raw_position: object = row.get("position")
        if not isinstance(raw_position, int | float | str):
            continue
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if driver_code:
            actual_positions_by_driver[driver_code] = position

    predicted_confidences: list[float] = []
    within_tolerance_hits: list[float] = []
    for row in predicted_entries:
        driver_code = str(row.get("driver", "")).strip()
        if not driver_code or driver_code not in actual_positions_by_driver:
            continue
        confidence = _coerce_float(row.get("order_confidence"))
        predicted_position = _coerce_float(row.get("position"))
        if confidence is None or predicted_position is None:
            continue
        predicted_confidences.append(confidence)
        within_tolerance_hits.append(
            1.0
            if within_tolerance(
                predicted_position=predicted_position,
                actual_position=actual_positions_by_driver[driver_code],
                tolerance=tolerance,
            )
            else 0.0
        )

    count = len(predicted_confidences)
    if count == 0:
        return {
            "order_confidence_count": 0,
            "order_confidence_mean": None,
            "order_confidence_empirical_within_tolerance": None,
            "order_confidence_calibration_error": None,
            "order_confidence_tolerance": float(tolerance),
        }

    mean_confidence = sum(predicted_confidences) / count
    empirical_within_tolerance = (sum(within_tolerance_hits) / count) * 100.0
    return {
        "order_confidence_count": count,
        "order_confidence_mean": float(mean_confidence),
        "order_confidence_empirical_within_tolerance": float(empirical_within_tolerance),
        "order_confidence_calibration_error": float(mean_confidence - empirical_within_tolerance),
        "order_confidence_tolerance": float(tolerance),
    }


def _resolve_prediction_context(
    *,
    evaluation_mode: EvaluationMode,
    year: int,
    race_name: str,
    session_name: str,
    seed: int | None,
) -> PredictionContext | None:
    """Resolve prediction context for one backtest session."""
    if evaluation_mode != "historical":
        return None
    return build_historical_prediction_context(
        year=year,
        race_name=race_name,
        target_session_name=session_name,
        seed=seed,
    )


def _build_backtest_prediction_record(
    *,
    year: int,
    race_name: str,
    qualifying_prediction: dict[str, Any],
    race_prediction: dict[str, Any],
    qualifying_actual: list[dict[str, Any]],
    race_actual: list[dict[str, Any]],
    race_context: PredictionContext | None,
) -> dict[str, Any]:
    """Build prediction-record payload compatible with adaptive learning updates."""
    if race_context is not None:
        predicted_at = race_context.reference_now().isoformat()
    else:
        predicted_at = None

    return {
        "metadata": {
            "run_id": f"backtest:{year}:{race_name}",
            "race_name": race_name,
            "session_name": "R",
            "source": "backtest",
            "generated_by": "season_backtest",
            "predicted_at": predicted_at,
            "information_cutoff_at": predicted_at,
        },
        "qualifying": {
            "predicted_grid": _normalize_ranked_entries(
                qualifying_prediction.get("grid", []),
                preserve_interval_fields=True,
            ),
        },
        "race": {
            "predicted_results": _normalize_ranked_entries(
                race_prediction.get("finish_order", []),
                preserve_interval_fields=True,
            ),
        },
        "actuals": {
            "qualifying": _normalize_ranked_entries(qualifying_actual),
            "race": _normalize_ranked_entries(race_actual),
        },
    }


def _replay_completed_weekend_actuals(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    qualifying_actual: list[dict[str, Any]],
    race_actual: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Record completed-weekend team form when the predictor exposes replay hooks."""
    replay_fn = getattr(predictor, "record_completed_weekend_actuals", None)
    if not callable(replay_fn):
        return None
    return replay_fn(
        year=year,
        race_name=race_name,
        qualifying_actual=qualifying_actual,
        race_actual=race_actual,
    )


def run_single_race_backtest(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    evaluation_mode: EvaluationMode = "historical",
    learning_mode: LearningMode = "adaptive",
    results_fetcher: Any = fetch_actual_session_results,
    include_prediction_payloads: bool = False,
) -> dict[str, Any]:
    """Execute one race backtest and return metric payload or skip reason."""
    race_metadata = _resolve_backtest_race_metadata(
        year=year,
        race_name=race_name,
        weather=weather,
    )
    qualifying_actual = results_fetcher(year, race_name, "Q")
    race_actual = results_fetcher(year, race_name, "R")

    if not qualifying_actual or not race_actual:
        return {
            "race_name": race_name,
            "status": "skipped",
            **race_metadata,
            "reason": "missing_actual_results",
        }

    try:
        predictor_seed = getattr(predictor, "seed", None)
        qualifying_context = _resolve_prediction_context(
            evaluation_mode=evaluation_mode,
            year=year,
            race_name=race_name,
            session_name="Q",
            seed=(int(predictor_seed) if isinstance(predictor_seed, int) else None),
        )
        race_context = _resolve_prediction_context(
            evaluation_mode=evaluation_mode,
            year=year,
            race_name=race_name,
            session_name="R",
            seed=(int(predictor_seed) if isinstance(predictor_seed, int) else None),
        )
        try:
            qualifying_prediction = predictor.predict_qualifying(
                year=year,
                race_name=race_name,
                n_simulations=qualifying_simulations,
                prediction_context=qualifying_context,
            )
        except TypeError:
            qualifying_prediction = predictor.predict_qualifying(
                year=year,
                race_name=race_name,
                n_simulations=qualifying_simulations,
            )

        try:
            race_prediction = predictor.predict_race(
                qualifying_grid=qualifying_prediction["grid"],
                weather=weather,
                race_name=race_name,
                n_simulations=race_simulations,
                year=year,
                prediction_context=race_context,
            )
        except TypeError:
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
        qualifying_interval_metrics = _compute_interval_metrics(
            qualifying_prediction["grid"],
            qualifying_actual,
        )
        race_interval_metrics = _compute_interval_metrics(
            race_prediction["finish_order"],
            race_actual,
        )
        qualifying_oc_tolerance = float(
            config_loader.get("baseline_predictor.qualifying.order_confidence.tolerance", 1.0)
        )
        race_oc_tolerance = float(
            config_loader.get("baseline_predictor.race.order_confidence.tolerance", 1.0)
        )
        qualifying_order_confidence_metrics = _compute_order_confidence_metrics(
            qualifying_prediction["grid"],
            qualifying_actual,
            tolerance=qualifying_oc_tolerance,
        )
        race_order_confidence_metrics = _compute_order_confidence_metrics(
            race_prediction["finish_order"],
            race_actual,
            tolerance=race_oc_tolerance,
        )
        learning_summary: dict[str, Any] | None = None
        if learning_mode == "adaptive":
            learning_system = getattr(predictor, "calibration_system", None)
            update_fn = getattr(learning_system, "update_from_prediction_record", None)
            if callable(update_fn):
                prediction_record = _build_backtest_prediction_record(
                    year=year,
                    race_name=race_name,
                    qualifying_prediction=qualifying_prediction,
                    race_prediction=race_prediction,
                    qualifying_actual=list(qualifying_actual),
                    race_actual=list(race_actual),
                    race_context=race_context,
                )
                learning_summary = update_fn(prediction_record)
        _replay_completed_weekend_actuals(
            predictor=predictor,
            year=year,
            race_name=race_name,
            qualifying_actual=list(qualifying_actual),
            race_actual=list(race_actual),
        )

        result = {
            "race_name": race_name,
            "status": "ok",
            "evaluation_mode": evaluation_mode,
            "learning_mode": learning_mode,
            **race_metadata,
            "qualifying_mae": qualifying_metrics["mae"],
            "qualifying_exact_accuracy": qualifying_metrics["exact_accuracy"],
            "qualifying_interval_count": qualifying_interval_metrics["interval_count"],
            "qualifying_interval_hits": qualifying_interval_metrics["interval_hits"],
            "qualifying_interval_empirical_coverage": qualifying_interval_metrics[
                "empirical_coverage"
            ],
            "qualifying_interval_nominal_coverage": qualifying_interval_metrics["nominal_coverage"],
            "qualifying_interval_calibration_error": qualifying_interval_metrics[
                "calibration_error"
            ],
            "qualifying_interval_width_mean": qualifying_interval_metrics["mean_interval_width"],
            "qualifying_interval_average_miss_distance": qualifying_interval_metrics[
                "average_miss_distance"
            ],
            "qualifying_order_confidence_count": qualifying_order_confidence_metrics[
                "order_confidence_count"
            ],
            "qualifying_order_confidence_mean": qualifying_order_confidence_metrics[
                "order_confidence_mean"
            ],
            "qualifying_order_confidence_empirical_within_tolerance": (
                qualifying_order_confidence_metrics["order_confidence_empirical_within_tolerance"]
            ),
            "qualifying_order_confidence_calibration_error": qualifying_order_confidence_metrics[
                "order_confidence_calibration_error"
            ],
            "qualifying_predicted_top10": _top_n_entries(qualifying_prediction["grid"]),
            "qualifying_actual_top10": _top_n_entries(qualifying_actual),
            "race_mae": race_metrics["mae"],
            "race_exact_accuracy": race_metrics["exact_accuracy"],
            "race_within_3": race_metrics["within_3"],
            "top3_accuracy": race_metrics["top3_accuracy"],
            "winner_correct": race_metrics["winner_correct"],
            "race_interval_count": race_interval_metrics["interval_count"],
            "race_interval_hits": race_interval_metrics["interval_hits"],
            "race_interval_empirical_coverage": race_interval_metrics["empirical_coverage"],
            "race_interval_nominal_coverage": race_interval_metrics["nominal_coverage"],
            "race_interval_calibration_error": race_interval_metrics["calibration_error"],
            "race_interval_width_mean": race_interval_metrics["mean_interval_width"],
            "race_interval_average_miss_distance": race_interval_metrics["average_miss_distance"],
            "race_order_confidence_count": race_order_confidence_metrics["order_confidence_count"],
            "race_order_confidence_mean": race_order_confidence_metrics["order_confidence_mean"],
            "race_order_confidence_empirical_within_tolerance": race_order_confidence_metrics[
                "order_confidence_empirical_within_tolerance"
            ],
            "race_order_confidence_calibration_error": race_order_confidence_metrics[
                "order_confidence_calibration_error"
            ],
            "race_predicted_top10": _top_n_entries(race_prediction["finish_order"]),
            "race_actual_top10": _top_n_entries(race_actual),
            "adaptive_learning": learning_summary,
        }
        if include_prediction_payloads:
            result.update(
                {
                    "qualifying_regime": qualifying_prediction.get("data_regime"),
                    "race_regime": race_prediction.get("data_regime"),
                    "qualifying_prediction_rows": _normalize_ranked_entries(
                        qualifying_prediction["grid"],
                        preserve_interval_fields=True,
                    ),
                    "qualifying_actual_rows": _normalize_ranked_entries(
                        qualifying_actual,
                        preserve_interval_fields=False,
                    ),
                    "race_prediction_rows": _normalize_ranked_entries(
                        race_prediction["finish_order"],
                        preserve_interval_fields=True,
                    ),
                    "race_actual_rows": _normalize_ranked_entries(
                        race_actual,
                        preserve_interval_fields=False,
                    ),
                }
            )
        return result
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Backtest failed for %s: %s", race_name, exc)
        return {
            "race_name": race_name,
            "status": "skipped",
            "evaluation_mode": evaluation_mode,
            "learning_mode": learning_mode,
            **race_metadata,
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

    for session_prefix in ("qualifying", "race"):
        interval_rows = [
            row for row in successful if int(row.get(f"{session_prefix}_interval_count") or 0) > 0
        ]
        if not interval_rows:
            continue

        total_count = sum(
            int(row.get(f"{session_prefix}_interval_count") or 0) for row in interval_rows
        )
        total_hits = sum(
            int(row.get(f"{session_prefix}_interval_hits") or 0) for row in interval_rows
        )
        if total_count <= 0:
            continue

        width_numerator = sum(
            int(row.get(f"{session_prefix}_interval_count") or 0)
            * float(row.get(f"{session_prefix}_interval_width_mean") or 0.0)
            for row in interval_rows
        )
        miss_distance_numerator = sum(
            int(row.get(f"{session_prefix}_interval_count") or 0)
            * float(row.get(f"{session_prefix}_interval_average_miss_distance") or 0.0)
            for row in interval_rows
        )
        empirical_coverage = total_hits / total_count
        nominal_coverage = 0.90
        summary.update(
            {
                f"{session_prefix}_interval_races": len(interval_rows),
                f"{session_prefix}_interval_count": total_count,
                f"{session_prefix}_interval_empirical_coverage": empirical_coverage,
                f"{session_prefix}_interval_nominal_coverage": nominal_coverage,
                f"{session_prefix}_interval_calibration_error": (
                    empirical_coverage - nominal_coverage
                ),
                f"{session_prefix}_interval_width_mean": width_numerator / total_count,
                f"{session_prefix}_interval_average_miss_distance": (
                    miss_distance_numerator / total_count
                ),
            }
        )

    for session_prefix in ("qualifying", "race"):
        oc_rows = [
            row
            for row in successful
            if int(row.get(f"{session_prefix}_order_confidence_count") or 0) > 0
            and row.get(f"{session_prefix}_order_confidence_mean") is not None
        ]
        if not oc_rows:
            continue
        oc_total = sum(
            int(row.get(f"{session_prefix}_order_confidence_count") or 0) for row in oc_rows
        )
        if oc_total <= 0:
            continue
        mean_numerator = sum(
            int(row.get(f"{session_prefix}_order_confidence_count") or 0)
            * float(row.get(f"{session_prefix}_order_confidence_mean") or 0.0)
            for row in oc_rows
        )
        empirical_numerator = sum(
            int(row.get(f"{session_prefix}_order_confidence_count") or 0)
            * float(row.get(f"{session_prefix}_order_confidence_empirical_within_tolerance") or 0.0)
            for row in oc_rows
        )
        mean_predicted = mean_numerator / oc_total
        empirical_within_tolerance = empirical_numerator / oc_total
        summary.update(
            {
                f"{session_prefix}_order_confidence_races": len(oc_rows),
                f"{session_prefix}_order_confidence_count": oc_total,
                f"{session_prefix}_order_confidence_mean": mean_predicted,
                f"{session_prefix}_order_confidence_empirical_within_tolerance": (
                    empirical_within_tolerance
                ),
                # Positive => the published confidence is overstated vs reality.
                f"{session_prefix}_order_confidence_calibration_error": (
                    mean_predicted - empirical_within_tolerance
                ),
            }
        )
    return summary


def build_segment_breakdown(
    race_results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate successful backtest rows by key review segments.

    The output is shaped for machine-readable artifacts and reviewer-facing
    markdown. Each bucket reuses ``aggregate_race_metrics`` so segment views
    stay consistent with the season-wide summary.
    """
    successful = [row for row in race_results if row.get("status") == "ok"]
    if not successful:
        return {}

    breakdown: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in ("weekend_format", "track_type", "weather"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in successful:
            bucket_name = str(row.get(dimension, "unknown") or "unknown").strip().lower()
            grouped.setdefault(bucket_name or "unknown", []).append(row)

        breakdown[dimension] = {}
        for bucket_name, rows in sorted(grouped.items()):
            breakdown[dimension][bucket_name] = {
                "events": len(rows),
                **aggregate_race_metrics(rows),
            }

    return breakdown


def _summarize_bucket_frequency(
    rows: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    """Count how often a label appears in a set of race rows."""
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key, "unknown") or "unknown").strip().lower() or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _compact_error_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the high-signal fields for one backtest error-analysis row."""
    return {
        "race_name": row.get("race_name"),
        "weekend_format": row.get("weekend_format"),
        "track_type": row.get("track_type"),
        "weather": row.get("weather"),
        "qualifying_mae": row.get("qualifying_mae"),
        "race_mae": row.get("race_mae"),
        "top3_accuracy": row.get("top3_accuracy"),
        "winner_correct": row.get("winner_correct"),
    }


def build_error_analysis(race_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Highlight the hardest weekends and repeat miss patterns from a backtest."""
    successful = [row for row in race_results if row.get("status") == "ok"]
    if not successful:
        return {
            "races_evaluated": 0,
            "worst_race_events": [],
            "worst_qualifying_events": [],
            "winner_miss_events": [],
            "worst_race_track_types": [],
            "worst_race_weekend_formats": [],
        }

    worst_race_events = sorted(
        successful,
        key=lambda row: float(row.get("race_mae") or 0.0),
        reverse=True,
    )[:5]
    worst_qualifying_events = sorted(
        successful,
        key=lambda row: float(row.get("qualifying_mae") or 0.0),
        reverse=True,
    )[:5]
    winner_miss_events = [row for row in successful if row.get("winner_correct") is False]
    winner_miss_events = sorted(
        winner_miss_events,
        key=lambda row: float(row.get("race_mae") or 0.0),
        reverse=True,
    )[:5]

    return {
        "races_evaluated": len(successful),
        "worst_race_events": [_compact_error_row(row) for row in worst_race_events],
        "worst_qualifying_events": [_compact_error_row(row) for row in worst_qualifying_events],
        "winner_miss_events": [_compact_error_row(row) for row in winner_miss_events],
        "worst_race_track_types": _summarize_bucket_frequency(worst_race_events, key="track_type"),
        "worst_race_weekend_formats": _summarize_bucket_frequency(
            worst_race_events,
            key="weekend_format",
        ),
    }


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
    strategy: Literal["temporal", "random"] = "temporal",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split successful race results into train/test slices for generalization checks.

    The default strategy is ``temporal``: the first ``train_fraction`` of races
    (in the order they appear in ``race_results``) become the training set and
    the remainder become the test set. This is the correct default for a
    predictor that learns from completed races, because shuffling leaks future
    state into the training fold.

    ``random`` replicates the original shuffle behaviour and is provided only
    for comparison. It is NOT appropriate for evaluating a learning system that
    consumes races in calendar order - use it only to measure how much the
    choice of split strategy changes the reported metrics.

    Args:
        race_results: Full list of per-race result dicts as returned by
            ``run_single_race_backtest``. Only rows with ``status == "ok"``
            are included in either split.
        train_fraction: Fraction of successful races assigned to the training
            split. Clamped to [0.1, 0.9].
        seed: Random seed used only when ``strategy="random"``. Ignored for
            temporal splits.
        strategy: ``"temporal"`` (default) preserves calendar order;
            ``"random"`` shuffles before splitting.

    Returns:
        Tuple of (train_rows, test_rows) where each element is a list of
        race-result dicts.
    """
    successful = [row for row in race_results if row.get("status") == "ok"]
    if len(successful) <= 1:
        return successful, []

    clamped_fraction = min(max(train_fraction, 0.1), 0.9)

    if strategy == "random":
        ordered = successful.copy()
        random.Random(seed).shuffle(ordered)
    elif strategy == "temporal":
        # race_results arrives in calendar order from the harness; preserve it.
        # Shuffling would leak future race outcomes into the training fold for a
        # predictor that updates its internal state (EMA errors, teammate gaps)
        # after each completed race.
        ordered = list(successful)
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}. Choose 'temporal' or 'random'.")

    train_size = int(round(len(ordered) * clamped_fraction))
    train_size = max(1, min(train_size, len(ordered) - 1))

    return ordered[:train_size], ordered[train_size:]


def summarize_generalization(
    race_results: list[dict[str, Any]],
    train_fraction: float = 0.7,
    seed: int = 42,
    strategy: Literal["temporal", "random"] = "temporal",
) -> dict[str, Any]:
    """Return train/test metric summaries and the generalization gap.

    Args:
        race_results: Per-race result dicts from the backtest harness.
        train_fraction: Fraction of completed races used for training.
        seed: Random seed, passed through to ``split_train_test_results``;
            only used when ``strategy="random"``.
        strategy: Split strategy passed to ``split_train_test_results``.
            Use the default ``"temporal"`` for all production evaluations.

    Returns:
        Dict with keys ``train``, ``test`` (each an ``aggregate_race_metrics``
        payload) and ``generalization_gap_race_mae`` (test MAE minus train MAE;
        positive means worse on unseen races).
    """
    train_rows, test_rows = split_train_test_results(
        race_results=race_results,
        train_fraction=train_fraction,
        seed=seed,
        strategy=strategy,
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
        "split_strategy": strategy,
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
