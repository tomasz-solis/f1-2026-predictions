"""Audit time-safe rank model candidates against scored prediction artifacts.

The goal is not to tune a champion from seven races by hindsight. This script
compares simple, inspectable alternatives using only information available
before each scored event: the current saved model forecast, previous completed
race actuals, and bias residuals from earlier races.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.generate_evaluation_report import (  # noqa: E402
    _resolve_session_pair,
    _select_latest_predictions,
    _sort_selected_predictions,
)

from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402
from src.utils.env_file import load_env_file as _load_env_file  # noqa: E402
from src.utils.prediction_logger import PredictionLogger  # noqa: E402
from src.utils.weekend import get_schedule_rows  # noqa: E402

BLEND_WEIGHTS = (0.2, 0.4, 0.6, 0.8)
BIAS_ALPHAS = (0.25, 0.5, 0.75, 1.0)
ROLLING_WINDOWS = (2, 3)
PROMOTION_MIN_SCORED_EVENTS = 8
PROMOTION_MIN_IMPROVEMENT_VS_REFERENCE = 0.15
PROMOTION_MAX_FORMAT_REGRESSION = 0.10
REFERENCE_CANDIDATE = "raw_model"


@dataclass(frozen=True)
class CandidateResult:
    """One candidate's score on one event."""

    candidate: str
    session_kind: str
    race_name: str
    weekend_format: str
    mae: float


def _position_by_driver(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return driver -> rank position from rows."""
    positions: dict[str, int] = {}
    for default_position, row in enumerate(rows, start=1):
        driver = str(row.get("driver", "")).strip()
        if not driver:
            continue
        try:
            position = int(row.get("position", default_position))
        except (TypeError, ValueError):
            position = default_position
        positions[driver] = position
    return positions


def _row_position(row: dict[str, Any], default_position: int) -> int:
    """Return one row's integer position."""
    try:
        return int(row.get("position", default_position))
    except (TypeError, ValueError):
        return default_position


def _rank_from_scores(
    base_rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Create ranked rows by sorting driver scores ascending."""
    rows: list[dict[str, Any]] = []
    for default_position, row in enumerate(base_rows, start=1):
        driver = str(row.get("driver", "")).strip()
        adjusted = dict(row)
        adjusted["_candidate_score"] = scores.get(
            driver,
            float(_row_position(row, default_position)),
        )
        rows.append(adjusted)

    rows.sort(
        key=lambda row: (
            float(row["_candidate_score"]),
            _row_position(row, 999),
            str(row.get("driver", "")),
        )
    )
    for position, row in enumerate(rows, start=1):
        row["position"] = position
        row.pop("_candidate_score", None)
    return rows


def _blend_with_previous(
    model_rows: list[dict[str, Any]],
    previous_actual_rows: list[dict[str, Any]],
    *,
    model_weight: float,
) -> list[dict[str, Any]]:
    """Blend model rank with previous-race actual rank."""
    previous_positions = _position_by_driver(previous_actual_rows)
    scores: dict[str, float] = {}
    for default_position, row in enumerate(model_rows, start=1):
        driver = str(row.get("driver", "")).strip()
        model_position = _row_position(row, default_position)
        previous_position = previous_positions.get(driver, model_position)
        scores[driver] = (model_weight * model_position) + (
            (1.0 - model_weight) * previous_position
        )
    return _rank_from_scores(model_rows, scores)


def _rolling_actual_rank(
    model_rows: list[dict[str, Any]],
    actual_history: list[list[dict[str, Any]]],
    *,
    window: int,
) -> list[dict[str, Any]]:
    """Rank drivers by their average actual position over recent completed events."""
    recent_history = actual_history[-int(window) :]
    scores: dict[str, float] = {}
    for default_position, row in enumerate(model_rows, start=1):
        driver = str(row.get("driver", "")).strip()
        observed_positions: list[int] = []
        for actual_rows in recent_history:
            positions = _position_by_driver(actual_rows)
            if driver in positions:
                observed_positions.append(positions[driver])
        scores[driver] = (
            sum(observed_positions) / len(observed_positions)
            if observed_positions
            else float(_row_position(row, default_position))
        )
    return _rank_from_scores(model_rows, scores)


def _bias_corrected_rank(
    model_rows: list[dict[str, Any]],
    prediction_history: list[list[dict[str, Any]]],
    actual_history: list[list[dict[str, Any]]],
    *,
    alpha: float,
    level: str,
) -> list[dict[str, Any]]:
    """Adjust ranks by driver/team signed error learned from prior events."""
    error_sum: dict[str, float] = defaultdict(float)
    error_count: dict[str, int] = defaultdict(int)
    for predicted_rows, actual_rows in zip(prediction_history, actual_history, strict=True):
        actual_positions = _position_by_driver(actual_rows)
        for default_position, row in enumerate(predicted_rows, start=1):
            driver = str(row.get("driver", "")).strip()
            if driver not in actual_positions:
                continue
            team = str(row.get("team", "")).strip()
            key = driver if level == "driver" else team
            if not key:
                continue
            predicted_position = _row_position(row, default_position)
            error_sum[key] += float(predicted_position - actual_positions[driver])
            error_count[key] += 1

    scores: dict[str, float] = {}
    for default_position, row in enumerate(model_rows, start=1):
        driver = str(row.get("driver", "")).strip()
        team = str(row.get("team", "")).strip()
        key = driver if level == "driver" else team
        mean_error = error_sum[key] / error_count[key] if error_count[key] else 0.0
        scores[driver] = float(_row_position(row, default_position)) - (float(alpha) * mean_error)
    return _rank_from_scores(model_rows, scores)


def _candidate_rows(
    model_rows: list[dict[str, Any]],
    *,
    prediction_history: list[list[dict[str, Any]]],
    actual_history: list[list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Build all time-safe candidate rows for one event."""
    candidates = {
        "raw_model": model_rows,
        "previous_race_naive": actual_history[-1],
    }
    for weight in BLEND_WEIGHTS:
        candidates[f"fixed_blend_model_{weight:.1f}"] = _blend_with_previous(
            model_rows,
            actual_history[-1],
            model_weight=weight,
        )
    for window in ROLLING_WINDOWS:
        candidates[f"rolling_actual_{window}"] = _rolling_actual_rank(
            model_rows,
            actual_history,
            window=window,
        )
    for alpha in BIAS_ALPHAS:
        candidates[f"driver_bias_alpha_{alpha:.2f}"] = _bias_corrected_rank(
            model_rows,
            prediction_history,
            actual_history,
            alpha=alpha,
            level="driver",
        )
        candidates[f"team_bias_alpha_{alpha:.2f}"] = _bias_corrected_rank(
            model_rows,
            prediction_history,
            actual_history,
            alpha=alpha,
            level="team",
        )
    return candidates


def _summarize_candidate_results(results: list[CandidateResult]) -> dict[str, Any]:
    """Aggregate candidate rows into overall and format-specific summaries."""
    by_candidate: dict[str, list[CandidateResult]] = defaultdict(list)
    for row in results:
        by_candidate[row.candidate].append(row)

    raw_mean = _mean_mae(by_candidate.get("raw_model", []))
    naive_mean = _mean_mae(by_candidate.get("previous_race_naive", []))
    summary_rows: list[dict[str, Any]] = []
    for candidate, rows in sorted(by_candidate.items()):
        maes = [row.mae for row in rows]
        summary_rows.append(
            {
                "candidate": candidate,
                "events": len(rows),
                "mean_mae": _mean(maes),
                "median_mae": float(median(maes)) if maes else None,
                "mae_improvement_vs_raw": None
                if raw_mean is None
                else float(raw_mean - _mean(maes)),
                "mae_improvement_vs_previous_race_naive": None
                if naive_mean is None
                else float(naive_mean - _mean(maes)),
            }
        )
    summary_rows.sort(key=lambda row: (float(row["mean_mae"]), str(row["candidate"])))

    by_format: dict[str, list[dict[str, Any]]] = {}
    formats = sorted({row.weekend_format for row in results})
    for weekend_format in formats:
        format_rows = [row for row in results if row.weekend_format == weekend_format]
        by_format[weekend_format] = _summarize_candidate_results_shallow(format_rows)

    return {
        "overall": summary_rows,
        "by_format": by_format,
        "raw_model_mean_mae": raw_mean,
        "previous_race_naive_mean_mae": naive_mean,
    }


def _candidate_family(candidate: str) -> str:
    """Return the broad modeling family for one candidate name."""
    if candidate == "raw_model":
        return "model"
    if candidate == "previous_race_naive" or candidate.startswith("rolling_actual_"):
        return "recent_actual_only"
    if candidate.startswith("fixed_blend_"):
        return "model_recent_actual_blend"
    if candidate.startswith("driver_bias_"):
        return "driver_residual"
    if candidate.startswith("team_bias_"):
        return "team_residual"
    return "unknown"


def _row_by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map summary rows by candidate name."""
    return {str(row.get("candidate", "")): row for row in rows}


def _build_promotion_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative promotion recommendation for the best challenger."""
    overall_rows = list(summary.get("overall", []))
    if not overall_rows:
        return {
            "status": "hold",
            "reference_candidate": REFERENCE_CANDIDATE,
            "best_candidate": None,
            "reasons": ["No scored candidate rows were available."],
            "criteria": _promotion_criteria(),
        }

    best = overall_rows[0]
    best_candidate = str(best.get("candidate", ""))
    best_family = _candidate_family(best_candidate)
    by_candidate = _row_by_candidate(overall_rows)
    reference = by_candidate.get(REFERENCE_CANDIDATE)
    reasons: list[str] = []

    scored_events = int(best.get("events", 0) or 0)
    if scored_events < PROMOTION_MIN_SCORED_EVENTS:
        reasons.append(
            f"Only {scored_events} scored challenger events; require "
            f"{PROMOTION_MIN_SCORED_EVENTS} before promotion."
        )

    if reference is None:
        reasons.append(f"Reference candidate `{REFERENCE_CANDIDATE}` is missing.")
        improvement_vs_reference = None
    else:
        improvement_vs_reference = float(reference["mean_mae"]) - float(best["mean_mae"])
        if improvement_vs_reference < PROMOTION_MIN_IMPROVEMENT_VS_REFERENCE:
            reasons.append(
                "Best candidate improvement vs reference is "
                f"{improvement_vs_reference:.3f}; require at least "
                f"{PROMOTION_MIN_IMPROVEMENT_VS_REFERENCE:.3f} MAE."
            )

    format_regressions = _format_regressions_vs_reference(
        summary.get("by_format", {}),
        candidate=best_candidate,
        reference_candidate=REFERENCE_CANDIDATE,
    )
    for regression in format_regressions:
        reasons.append(
            f"{best_candidate} regresses vs {REFERENCE_CANDIDATE} by "
            f"{regression['regression']:.3f} MAE on {regression['weekend_format']} weekends."
        )

    if best_family == "recent_actual_only":
        reasons.append(
            "Best candidate is recent-actual-only; keep it as a challenger until a "
            "model-aware blend or residual layer matches it without dropping feature explainability."
        )

    status = "promote" if not reasons else "hold"
    return {
        "status": status,
        "reference_candidate": REFERENCE_CANDIDATE,
        "best_candidate": best_candidate,
        "best_candidate_family": best_family,
        "best_mean_mae": best.get("mean_mae"),
        "reference_mean_mae": None if reference is None else reference.get("mean_mae"),
        "improvement_vs_reference": improvement_vs_reference,
        "format_regressions": format_regressions,
        "reasons": reasons,
        "criteria": _promotion_criteria(),
    }


def _promotion_criteria() -> dict[str, Any]:
    """Return the candidate-promotion gate thresholds."""
    return {
        "min_scored_events": PROMOTION_MIN_SCORED_EVENTS,
        "min_improvement_vs_reference_mae": PROMOTION_MIN_IMPROVEMENT_VS_REFERENCE,
        "max_format_regression_mae": PROMOTION_MAX_FORMAT_REGRESSION,
    }


def _format_regressions_vs_reference(
    by_format: dict[str, list[dict[str, Any]]],
    *,
    candidate: str,
    reference_candidate: str,
) -> list[dict[str, Any]]:
    """Return format slices where a candidate is materially worse than reference."""
    regressions: list[dict[str, Any]] = []
    for weekend_format, rows in sorted(by_format.items()):
        indexed = _row_by_candidate(rows)
        candidate_row = indexed.get(candidate)
        reference_row = indexed.get(reference_candidate)
        if candidate_row is None or reference_row is None:
            continue
        regression = float(candidate_row["mean_mae"]) - float(reference_row["mean_mae"])
        if regression > PROMOTION_MAX_FORMAT_REGRESSION:
            regressions.append(
                {
                    "weekend_format": weekend_format or "unknown",
                    "candidate_mean_mae": float(candidate_row["mean_mae"]),
                    "reference_mean_mae": float(reference_row["mean_mae"]),
                    "regression": float(regression),
                }
            )
    return regressions


def _diagnostic_coverage(
    selected_predictions: list[dict[str, Any]],
    *,
    session_kind: str,
) -> dict[str, Any]:
    """Summarize saved model-diagnostic metadata coverage for one session kind."""
    metadata_key = f"{session_kind}_model_diagnostics"
    with_diagnostics = 0
    with_profile = 0
    with_residual_flag = 0
    regimes: dict[str, int] = defaultdict(int)
    for prediction in selected_predictions:
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        diagnostics = metadata.get(metadata_key)
        if not isinstance(diagnostics, dict) or not diagnostics:
            continue
        with_diagnostics += 1
        if diagnostics.get("characteristics_profile_used") is not None:
            with_profile += 1
        residual_key = (
            "qualifying_residual_model_used"
            if session_kind == "qualifying"
            else "race_residual_model_used"
        )
        if residual_key in diagnostics:
            with_residual_flag += 1
        regime = str(diagnostics.get("data_regime", "")).strip() or "unknown"
        regimes[regime] += 1

    total = len(selected_predictions)
    return {
        "metadata_key": metadata_key,
        "selected_events": total,
        "events_with_model_diagnostics": with_diagnostics,
        "events_with_characteristics_profile": with_profile,
        "events_with_residual_model_flag": with_residual_flag,
        "data_regime_counts": dict(sorted(regimes.items())),
        "coverage": None if total == 0 else with_diagnostics / total,
    }


def _summarize_candidate_results_shallow(results: list[CandidateResult]) -> list[dict[str, Any]]:
    """Aggregate without nesting to avoid recursion for by-format slices."""
    by_candidate: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_candidate[row.candidate].append(row.mae)
    rows = [
        {
            "candidate": candidate,
            "events": len(values),
            "mean_mae": _mean(values),
            "median_mae": float(median(values)) if values else None,
        }
        for candidate, values in sorted(by_candidate.items())
    ]
    return sorted(rows, key=lambda row: (float(row["mean_mae"]), str(row["candidate"])))


def _mean(values: list[float]) -> float:
    """Return a non-empty mean."""
    return float(sum(values) / len(values)) if values else float("nan")


def _mean_mae(rows: list[CandidateResult]) -> float | None:
    """Return candidate mean MAE or None when no rows exist."""
    if not rows:
        return None
    return _mean([row.mae for row in rows])


def _audit_session(
    predictions: list[dict[str, Any]],
    *,
    year: int,
    session_kind: str,
    schedule_formats: dict[str, str],
) -> dict[str, Any]:
    """Audit all candidates for one session kind."""
    selected = _sort_selected_predictions(
        _select_latest_predictions(predictions, session_kind=session_kind),
        year=year,
    )
    diagnostics_coverage = _diagnostic_coverage(selected, session_kind=session_kind)
    event_rows: list[dict[str, Any]] = []
    results: list[CandidateResult] = []
    prediction_history: list[list[dict[str, Any]]] = []
    actual_history: list[list[dict[str, Any]]] = []

    for event_index, prediction in enumerate(selected):
        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        weekend_format = schedule_formats.get(race_name, str(metadata.get("weekend_format", "")))
        model_rows, actual_rows = _resolve_session_pair(prediction, session_kind=session_kind)
        if not model_rows or not actual_rows:
            continue
        if event_index == 0 or not actual_history:
            prediction_history.append(model_rows)
            actual_history.append(actual_rows)
            continue

        scored_candidates = _candidate_rows(
            model_rows,
            prediction_history=prediction_history,
            actual_history=actual_history,
        )
        event_scores: dict[str, float] = {}
        for candidate, rows in scored_candidates.items():
            mae = float(compute_prediction_accuracy(rows, actual_rows)["mae"])
            event_scores[candidate] = mae
            results.append(
                CandidateResult(
                    candidate=candidate,
                    session_kind=session_kind,
                    race_name=race_name,
                    weekend_format=weekend_format,
                    mae=mae,
                )
            )
        event_rows.append(
            {
                "race_name": race_name,
                "weekend_format": weekend_format,
                "checkpoint": metadata.get("session_name"),
                "target": metadata.get(f"top_level_{session_kind}_target"),
                "candidate_mae": dict(sorted(event_scores.items())),
            }
        )
        prediction_history.append(model_rows)
        actual_history.append(actual_rows)

    summary = _summarize_candidate_results(results)
    promotion_gate = _build_promotion_gate(summary)
    return {
        "session_kind": session_kind,
        "selected_events": len(selected),
        "scored_events": len(event_rows),
        "candidate_summary": summary,
        "promotion_gate": promotion_gate,
        "diagnostic_coverage": diagnostics_coverage,
        "event_scores": event_rows,
    }


def build_candidate_audit(year: int) -> dict[str, Any]:
    """Build the full candidate audit payload."""
    predictions = PredictionLogger().get_all_predictions(year)
    schedule_rows = get_schedule_rows(year)
    schedule_formats = {race_name: event_format for race_name, event_format in schedule_rows}
    sessions = {
        session_kind: _audit_session(
            predictions,
            year=year,
            session_kind=session_kind,
            schedule_formats=schedule_formats,
        )
        for session_kind in ("qualifying", "race")
    }
    return {
        "artifact_type": "model_candidate_audit",
        "schema_version": 1,
        "year": year,
        "candidate_rules": {
            "scoring": "expanding-window; first event seeds history and is not scored",
            "allowed_inputs": [
                "current saved model ranks",
                "previous completed race actual ranks",
                "driver/team residuals from prior scored races",
            ],
            "blend_weights": list(BLEND_WEIGHTS),
            "bias_alphas": list(BIAS_ALPHAS),
            "rolling_windows": list(ROLLING_WINDOWS),
        },
        "prediction_artifacts_loaded": len(predictions),
        "schedule_rows": len(schedule_rows),
        "sessions": sessions,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    """Render the candidate audit as review-friendly markdown."""
    lines = [
        f"# Model Candidate Audit - {audit['year']}",
        "",
        "Candidates are evaluated in expanding-window order. The first scored",
        "event seeds history; each later event can only use previous completed",
        "actuals and prior residuals.",
        "",
        f"- Prediction artifacts loaded: **{audit.get('prediction_artifacts_loaded', 0)}**",
        f"- Schedule rows: **{audit.get('schedule_rows', 0)}**",
        "",
    ]
    for session_kind, section in audit.get("sessions", {}).items():
        lines.extend([f"## {session_kind.title()}", ""])
        lines.extend(
            [
                f"- Selected events: **{section.get('selected_events', 0)}**",
                f"- Scored candidate events: **{section.get('scored_events', 0)}**",
                "",
            ]
        )
        gate = section.get("promotion_gate", {})
        if gate:
            lines.extend(
                [
                    "### Promotion Readout",
                    "",
                    f"- Recommendation: **{str(gate.get('status', 'hold')).upper()}**",
                    f"- Reference: `{gate.get('reference_candidate')}`",
                    f"- Best challenger: `{gate.get('best_candidate')}` "
                    f"({gate.get('best_candidate_family')})",
                    f"- Improvement vs reference: **{_fmt(gate.get('improvement_vs_reference'))} MAE**",
                ]
            )
            reasons = list(gate.get("reasons") or [])
            if reasons:
                lines.append("- Blocking reasons: " + " | ".join(str(reason) for reason in reasons))
            lines.append("")
        diagnostics = section.get("diagnostic_coverage", {})
        if diagnostics:
            coverage = diagnostics.get("coverage")
            coverage_text = "n/a" if coverage is None else f"{float(coverage):.1%}"
            lines.extend(
                [
                    "### Saved Signal Coverage",
                    "",
                    f"- Model-diagnostic metadata coverage: **{coverage_text}** "
                    f"({diagnostics.get('events_with_model_diagnostics', 0)}/"
                    f"{diagnostics.get('selected_events', 0)})",
                    f"- Characteristics-profile evidence: "
                    f"**{diagnostics.get('events_with_characteristics_profile', 0)}** events",
                    f"- Residual-model flag evidence: "
                    f"**{diagnostics.get('events_with_residual_model_flag', 0)}** events",
                    "",
                ]
            )
        lines.extend(
            [
                "| Candidate | Events | Mean MAE | Median MAE | vs raw | vs prev-race naive |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in section.get("candidate_summary", {}).get("overall", [])[:12]:
            lines.append(
                f"| `{row['candidate']}` | {row['events']} | "
                f"{_fmt(row['mean_mae'])} | {_fmt(row['median_mae'])} | "
                f"{_fmt(row['mae_improvement_vs_raw'])} | "
                f"{_fmt(row['mae_improvement_vs_previous_race_naive'])} |"
            )
        lines.append("")
        by_format = section.get("candidate_summary", {}).get("by_format", {})
        for weekend_format, rows in by_format.items():
            lines.extend(
                [
                    f"### {weekend_format or 'unknown'}",
                    "",
                    "| Candidate | Events | Mean MAE |",
                    "|---|---:|---:|",
                ]
            )
            for row in rows[:8]:
                lines.append(
                    f"| `{row['candidate']}` | {row['events']} | {_fmt(row['mean_mae'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format optional float values."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/model_candidate_audit.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/model_candidate_audit.md",
    )
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)

    audit = build_candidate_audit(args.year)
    json_out = args.json_out or Path(
        f"data/model_diagnostics/{args.year}/model_candidate_audit.json"
    )
    md_out = args.md_out or Path(f"data/model_diagnostics/{args.year}/model_candidate_audit.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
