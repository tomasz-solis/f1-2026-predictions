"""Compare full prediction replay outputs for team-strength mapping candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.team_strength_refit_candidate_test import (
    SCALE_ONLY_CANDIDATE,
    _driver_mu_by_kind,
    _policy_column,
    _prepare_observations,
)
from src.models.team_strength_mapping import (
    LinearTeamStrengthMapping,
    build_construct_aligned_driver_observations,
)
from src.utils.driver_name_mapper import DriverNameMapper
from src.utils.model_version import get_model_version

TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE = "model_diagnostics"
TEAM_STRENGTH_PREDICTION_REPLAY_KEY_TEMPLATE = "{year}::team_strength_prediction_replay_test"
TEAM_STRENGTH_PREDICTION_REPLAY_SCHEMA_VERSION = 1
RACE_TARGETS = frozenset({"grand_prix_race", "sprint_race"})


def team_strength_prediction_replay_artifact_key(year: int) -> str:
    """Return the stable artifact key for full prediction replay comparison."""
    return TEAM_STRENGTH_PREDICTION_REPLAY_KEY_TEMPLATE.format(year=int(year))


def load_construct_observations(
    *,
    raw_matched_laps_path: str | Path,
    prior_artifact_path: str | Path,
) -> pd.DataFrame:
    """Build construct-aligned observations from raw matched-lap rows."""
    raw = pd.read_csv(raw_matched_laps_path)
    prior = _read_json(Path(prior_artifact_path))
    return build_construct_aligned_driver_observations(
        raw,
        driver_mu_by_kind=_driver_mu_by_kind(prior),
    )


def build_race_scale_only_mapping_payload(
    *,
    frozen_mapping_payload: Mapping[str, Any],
    frozen_race_mapping: LinearTeamStrengthMapping,
    observations: pd.DataFrame,
    mapping_policy: str,
    year: int,
    holdout_race: str,
) -> dict[str, Any]:
    """Build a race-only scale candidate mapping artifact for one held-out race."""
    policy_column = _policy_column(mapping_policy)
    rows = _prepare_observations(
        observations,
        year=int(year),
        policy_column=policy_column,
    )
    train = rows[rows["session_kind"].eq("race") & ~rows["race_name"].eq(holdout_race)]
    scale, slope = _fit_scale_only_race_slope(
        train=train,
        frozen_race_mapping=frozen_race_mapping,
        policy_column=policy_column,
    )

    payload = json.loads(json.dumps(frozen_mapping_payload))
    payload["artifact_type"] = "team_strength_seconds_mapping_candidate"
    payload["built_at"] = datetime.now(UTC).isoformat()
    payload["candidate"] = SCALE_ONLY_CANDIDATE
    payload["fit_method"] = "race_only_leave_one_race_scale"
    payload["holdout_race"] = str(holdout_race)
    payload["fit_rows"] = int(len(train))
    payload["fit_races"] = int(train["race_name"].nunique())
    payload["scale_multiplier"] = scale
    payload["mappings"]["race"]["slope_s_per_unit"] = slope
    payload["decision"] = {
        "state": "diagnostic_only",
        "note": "Used only for held-out full prediction replay; not a live mapping.",
    }
    return payload


def compare_prediction_replay_summaries(
    *,
    year: int,
    current_summary_path: str | Path,
    candidate_summaries: Mapping[str, str | Path],
    candidate_name: str,
) -> dict[str, Any]:
    """Compare candidate replay metrics against the current replay for holdout races."""
    current_summary = _read_json(Path(current_summary_path))
    current_records = _records_by_key(current_summary)
    rows: list[dict[str, Any]] = []
    for holdout_race, summary_path in candidate_summaries.items():
        candidate_summary = _read_json(Path(summary_path))
        for record in candidate_summary.get("checkpoints", []):
            if not isinstance(record, Mapping) or record.get("race_name") != holdout_race:
                continue
            key = _record_key(record)
            current_record = current_records.get(key)
            if current_record is None:
                continue
            rows.extend(
                _compare_prediction_files(
                    year=int(year),
                    candidate_name=candidate_name,
                    holdout_race=str(holdout_race),
                    current_record=current_record,
                    candidate_record=record,
                )
            )

    frame = pd.DataFrame(rows)
    artifact = {
        "artifact_type": "team_strength_prediction_replay_test",
        "schema_version": TEAM_STRENGTH_PREDICTION_REPLAY_SCHEMA_VERSION,
        "model_version": get_model_version(),
        "built_at": datetime.now(UTC).isoformat(),
        "year": int(year),
        "candidate": candidate_name,
        "status": "measured" if not frame.empty else "not_available",
        "metric_rows": _frame_records(_round_frame(frame)) if not frame.empty else [],
        "aggregate": _aggregate_prediction_rows(frame) if not frame.empty else [],
        "by_target": _aggregate_prediction_rows(frame, group_columns=["target"])
        if not frame.empty
        else [],
        "by_race": _aggregate_prediction_rows(frame, group_columns=["holdout_race"])
        if not frame.empty
        else [],
        "race_target_aggregate": _aggregate_prediction_rows(
            frame[frame["target"].isin(RACE_TARGETS)]
        )
        if not frame.empty
        else [],
        "decision_assessment": _decision_assessment(frame) if not frame.empty else {},
    }
    return _json_safe(artifact)


def format_team_strength_prediction_replay_test_markdown(artifact: Mapping[str, Any]) -> str:
    """Format a full prediction replay comparison as Markdown."""
    lines = [
        "# Team-Strength Prediction Replay Test",
        "",
        f"- Built at: `{artifact.get('built_at')}`",
        f"- Model version: `{artifact.get('model_version')}`",
        f"- Status: `{artifact.get('status')}`",
        f"- Candidate: `{artifact.get('candidate')}`",
        "",
        "## Aggregate",
        "",
    ]
    _append_table(lines, artifact.get("aggregate", []))
    lines.extend(["", "## Race Targets Only", ""])
    _append_table(lines, artifact.get("race_target_aggregate", []))
    lines.extend(["", "## By Target", ""])
    _append_table(lines, artifact.get("by_target", []), label_key="target")
    decision = artifact.get("decision_assessment", {})
    if isinstance(decision, Mapping):
        lines.extend(
            [
                "",
                "## Decision Assessment",
                "",
                f"- State: `{decision.get('state')}`",
                f"- Recommendation: {decision.get('recommendation')}",
            ]
        )
        for reason in decision.get("reasons", []):
            lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _fit_scale_only_race_slope(
    *,
    train: pd.DataFrame,
    frozen_race_mapping: LinearTeamStrengthMapping,
    policy_column: str,
) -> tuple[float, float]:
    """Fit one scale multiplier for race mapping on training rows."""
    centered_strength = train[policy_column].astype(float).to_numpy() - 0.5
    frozen_centered_delta = frozen_race_mapping.slope_s_per_unit * centered_strength
    target = train["team_target_s"].astype(float).to_numpy() - frozen_race_mapping.intercept_s
    denominator = float(np.dot(frozen_centered_delta, frozen_centered_delta))
    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot fit scale-only mapping with zero centered-strength variance.")
    scale = float(np.dot(frozen_centered_delta, target) / denominator)
    if not np.isfinite(scale):
        raise ValueError("Scale-only fit produced a non-finite multiplier.")
    return scale, float(frozen_race_mapping.slope_s_per_unit * scale)


def _compare_prediction_files(
    *,
    year: int,
    candidate_name: str,
    holdout_race: str,
    current_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare prediction files for one matched checkpoint record."""
    current_prediction = _read_json(Path(str(current_record["prediction_path"])))
    candidate_prediction = _read_json(Path(str(candidate_record["prediction_path"])))
    rows: list[dict[str, Any]] = []
    candidate_targets = candidate_prediction.get("targets", {})
    current_targets = current_prediction.get("targets", {})
    actual_targets = candidate_prediction.get("actuals", {}).get("targets", {})
    if not isinstance(candidate_targets, Mapping) or not isinstance(current_targets, Mapping):
        return rows
    if not isinstance(actual_targets, Mapping):
        return rows
    for target, candidate_payload in candidate_targets.items():
        if not isinstance(candidate_payload, Mapping):
            continue
        if not bool(candidate_payload.get("eligible_at_save", True)):
            continue
        current_payload = current_targets.get(target)
        actual_rows = actual_targets.get(target)
        if not isinstance(current_payload, Mapping) or not isinstance(actual_rows, list):
            continue
        candidate_order = candidate_payload.get("predicted_order", [])
        current_order = current_payload.get("predicted_order", [])
        if not isinstance(candidate_order, list) or not isinstance(current_order, list):
            continue
        current_metrics = _position_error_metrics(current_order, actual_rows)
        candidate_metrics = _position_error_metrics(candidate_order, actual_rows)
        if not current_metrics or not candidate_metrics:
            continue
        rows.append(
            {
                "year": int(year),
                "candidate": candidate_name,
                "holdout_race": holdout_race,
                "checkpoint_session": str(candidate_record.get("checkpoint_session")),
                "target": str(target),
                "field_size": candidate_metrics["field_size"],
                "current_mse": current_metrics["mse"],
                "candidate_mse": candidate_metrics["mse"],
                "mse_delta": candidate_metrics["mse"] - current_metrics["mse"],
                "current_mae": current_metrics["mae"],
                "candidate_mae": candidate_metrics["mae"],
                "mae_delta": candidate_metrics["mae"] - current_metrics["mae"],
                "candidate_wins_mse": candidate_metrics["mse"] < current_metrics["mse"],
                "candidate_wins_mae": candidate_metrics["mae"] < current_metrics["mae"],
            }
        )
    return rows


def _position_error_metrics(
    predicted_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Return position MSE and MAE for one predicted and actual order."""
    predicted_norm = DriverNameMapper.normalize_result_list(predicted_rows)
    actual_norm = DriverNameMapper.normalize_result_list(actual_rows)
    actual_positions = {row["driver"]: int(row["position"]) for row in actual_norm}
    errors = [
        int(row["position"]) - actual_positions[row["driver"]]
        for row in predicted_norm
        if row.get("driver") in actual_positions
    ]
    if not errors:
        return None
    error_array = np.asarray(errors, dtype=float)
    return {
        "field_size": float(len(errors)),
        "mse": float(np.mean(np.square(error_array))),
        "mae": float(np.mean(np.abs(error_array))),
    }


def _aggregate_prediction_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate paired current-vs-candidate prediction metric rows."""
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    if not group_columns:
        rows.append(_aggregate_one_prediction_group(frame))
    else:
        grouped = frame.groupby(group_columns, dropna=False)
        for group_value, group in grouped:
            row = _aggregate_one_prediction_group(group)
            values = group_value if isinstance(group_value, tuple) else (group_value,)
            for column, value in zip(group_columns, values, strict=True):
                row[column] = str(value)
            rows.append(row)
    return _frame_records(_round_frame(pd.DataFrame(rows)))


def _aggregate_one_prediction_group(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate one paired metric group."""
    return {
        "n_rows": int(len(frame)),
        "current_mse": float(frame["current_mse"].mean()),
        "candidate_mse": float(frame["candidate_mse"].mean()),
        "mse_delta": float(frame["mse_delta"].mean()),
        "mse_pct_delta": _pct_delta(frame["candidate_mse"].mean(), frame["current_mse"].mean()),
        "current_mae": float(frame["current_mae"].mean()),
        "candidate_mae": float(frame["candidate_mae"].mean()),
        "mae_delta": float(frame["mae_delta"].mean()),
        "candidate_mse_wins": int(frame["candidate_wins_mse"].sum()),
        "candidate_mae_wins": int(frame["candidate_wins_mae"].sum()),
    }


def _decision_assessment(frame: pd.DataFrame) -> dict[str, Any]:
    """Assess whether full prediction replay supports a candidate."""
    race_rows = frame[frame["target"].isin(RACE_TARGETS)]
    race_aggregate = _aggregate_one_prediction_group(race_rows) if not race_rows.empty else {}
    all_aggregate = _aggregate_one_prediction_group(frame)
    reasons: list[str] = []
    state = "not_enough_evidence"
    recommendation = "Do not ship a median-changing race refit from this replay."
    race_pct = _coerce_float(race_aggregate.get("mse_pct_delta")) if race_aggregate else None
    race_wins = int(race_aggregate.get("candidate_mse_wins", 0)) if race_aggregate else 0
    race_total = int(race_aggregate.get("n_rows", 0)) if race_aggregate else 0
    if race_pct is not None:
        reasons.append(f"Race-target position MSE delta: {race_pct:+.1f}%.")
    reasons.append(f"Race-target MSE wins: {race_wins}/{race_total}.")
    all_pct = _coerce_float(all_aggregate.get("mse_pct_delta"))
    if all_pct is not None:
        reasons.append(f"All-target position MSE delta: {all_pct:+.1f}%.")
    if race_pct is not None and race_pct < -5.0 and race_wins > (race_total / 2.0):
        state = "supports_race_only_prediction_replay_candidate"
        recommendation = (
            "Race-only scale candidate improves full prediction replay. "
            "Treat this as release evidence, then decide whether to ship with a model-version bump."
        )
    return {
        "state": state,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def _records_by_key(summary: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index replay checkpoint records by race and checkpoint."""
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in summary.get("checkpoints", []):
        if isinstance(record, Mapping):
            records[_record_key(record)] = record
    return records


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the comparison key for one replay checkpoint record."""
    return (str(record.get("race_name")), str(record.get("checkpoint_session")))


def _append_table(lines: list[str], rows: Any, *, label_key: str | None = None) -> None:
    """Append aggregate replay rows to Markdown."""
    if not isinstance(rows, list) or not rows:
        lines.append("No metrics available.")
        return
    label_header = "Group" if label_key else "Scope"
    lines.extend(
        [
            f"| {label_header} | Rows | Current MSE | Candidate MSE | MSE delta | Current MAE | Candidate MAE | MSE wins |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(rows, start=1):
        label = row.get(label_key) if label_key else ("combined" if index == 1 else str(index))
        lines.append(
            f"| `{label}` | {row.get('n_rows')} | {_fmt(row.get('current_mse'))} | "
            f"{_fmt(row.get('candidate_mse'))} | {_fmt_pct(row.get('mse_pct_delta'))} | "
            f"{_fmt(row.get('current_mae'))} | {_fmt(row.get('candidate_mae'))} | "
            f"{row.get('candidate_mse_wins')} |"
        )


def _pct_delta(value: Any, baseline: Any) -> float | None:
    """Return percentage delta from baseline when both values are numeric."""
    numeric_value = _coerce_float(value)
    numeric_baseline = _coerce_float(baseline)
    if numeric_value is None or numeric_baseline is None or np.isclose(numeric_baseline, 0.0):
        return None
    return float(((numeric_value - numeric_baseline) / numeric_baseline) * 100.0)


def _coerce_float(value: Any) -> float | None:
    """Convert a value to finite float or return None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _fmt(value: Any) -> str:
    """Format optional numbers for Markdown."""
    numeric = _coerce_float(value)
    return " - " if numeric is None else f"{numeric:.3f}"


def _fmt_pct(value: Any) -> str:
    """Format optional percentage deltas for Markdown."""
    numeric = _coerce_float(value)
    return " - " if numeric is None else f"{numeric:+.1f}%"


def _round_frame(frame: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Round float columns for stable JSON and Markdown output."""
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["float", "float64", "float32"]).columns:
        rounded[column] = rounded[column].round(decimals)
    return rounded


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe record dictionaries."""
    return _json_safe(frame.to_dict(orient="records"))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _json_safe(value: Any) -> Any:
    """Convert numpy and pandas values into JSON-compatible Python values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return value
    return value
