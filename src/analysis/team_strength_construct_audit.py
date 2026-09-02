"""Team-strength construct-row audit for 2026 regulation-reset diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.replay_leakage_diagnostics import build_historical_scale_reference
from src.models.team_strength_mapping import (
    TEAM_STRENGTH_POLICY_COLUMNS,
    LinearTeamStrengthMapping,
    build_construct_aligned_driver_observations,
    load_live_team_strength_mappings,
)
from src.utils.json_io import read_json_object as _read_json
from src.utils.model_version import get_model_version

TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE = "model_diagnostics"
TEAM_STRENGTH_CONSTRUCT_AUDIT_KEY_TEMPLATE = "{year}::team_strength_construct_row_audit"
TEAM_STRENGTH_CONSTRUCT_AUDIT_SCHEMA_VERSION = 1


def team_strength_construct_audit_artifact_key(year: int) -> str:
    """Return the stable artifact key for the team-strength construct-row audit."""
    return TEAM_STRENGTH_CONSTRUCT_AUDIT_KEY_TEMPLATE.format(year=int(year))


def build_team_strength_construct_audit(
    *,
    year: int,
    mapping_artifact_path: str | Path = "data/processed/team_strength_seconds_mapping/latest.json",
    candidate_diagnostics_path: str | Path = (
        "data/processed/team_strength_seconds_mapping/candidate_diagnostics.json"
    ),
    prior_artifact_path: str | Path = "data/processed/teammate_network_prior/latest.json",
    observations_path: str | Path | None = None,
    raw_matched_laps_path: str | Path | None = (
        "data/diagnostics/2026_team_strength_matched_laps/raw_matched_laps.csv"
    ),
    max_detail_rows: int = 20,
) -> dict[str, Any]:
    """Build a row-level audit for the measured 2026 construct-aligned rows."""
    season_year = int(year)
    mapping_path = Path(mapping_artifact_path)
    candidate_path = Path(candidate_diagnostics_path)
    prior_path = Path(prior_artifact_path)
    mapping_payload = _read_json(mapping_path)
    candidate_payload = _read_json(candidate_path) if candidate_path.exists() else {}
    mapping_policy = str(mapping_payload.get("policy", "same_session_construct"))
    historical_reference = build_historical_scale_reference(
        mapping_payload=mapping_payload,
        candidate_payload=candidate_payload,
    )
    mappings = load_live_team_strength_mappings(mapping_path)
    observations = _load_observations(
        observations_path=Path(observations_path) if observations_path is not None else None,
        raw_matched_laps_path=(
            Path(raw_matched_laps_path) if raw_matched_laps_path is not None else None
        ),
        prior_artifact_path=prior_path,
    )

    if observations is None or observations.empty:
        return _json_safe(
            _base_artifact(
                year=season_year,
                mapping_policy=mapping_policy,
                historical_reference=historical_reference,
                status="not_available",
                reason="No construct-aligned observations were available for audit.",
            )
        )

    prediction_rows = build_construct_prediction_rows(
        observations=observations,
        mappings=mappings,
        mapping_policy=mapping_policy,
        year=season_year,
    )
    if prediction_rows.empty:
        return _json_safe(
            _base_artifact(
                year=season_year,
                mapping_policy=mapping_policy,
                historical_reference=historical_reference,
                status="not_available",
                reason="No rows had the policy, mapping, and driver-prior fields required.",
            )
        )

    metrics = _metrics_by_session_kind(
        prediction_rows,
        historical_reference=historical_reference,
    )
    leave_one_race = _leave_one_group_diagnostics(
        prediction_rows,
        group_column="race_name",
        historical_reference=historical_reference,
    )
    leave_one_team = _leave_one_group_diagnostics(
        prediction_rows,
        group_column="team",
        historical_reference=historical_reference,
    )
    grouped_residuals = {
        "by_race": _group_residuals(prediction_rows, ["session_kind", "race_name"]),
        "by_team": _group_residuals(prediction_rows, ["session_kind", "team"]),
        "by_driver": _group_residuals(prediction_rows, ["session_kind", "driver_code"]),
        "largest_race_team_residuals": _top_group_residuals(
            prediction_rows,
            ["session_kind", "race_name", "team"],
            limit=max_detail_rows,
        ),
    }

    artifact = _base_artifact(
        year=season_year,
        mapping_policy=mapping_policy,
        historical_reference=historical_reference,
        status="measured",
    )
    artifact.update(
        {
            "metrics_by_session_kind": metrics,
            "leave_one_race": leave_one_race,
            "leave_one_team": leave_one_team,
            "grouped_residuals": grouped_residuals,
            "largest_abs_residual_rows": _top_residual_rows(
                prediction_rows,
                limit=max_detail_rows,
            ),
            "row_count": int(len(prediction_rows)),
            "audit_rows": _frame_records(_round_frame(prediction_rows)),
            "decision_notes": _decision_notes(metrics, leave_one_race, leave_one_team),
        }
    )
    return _json_safe(artifact)


def build_construct_prediction_rows(
    *,
    observations: pd.DataFrame,
    mappings: Mapping[str, LinearTeamStrengthMapping],
    mapping_policy: str,
    year: int,
) -> pd.DataFrame:
    """Attach frozen mapping predictions and residuals to construct rows."""
    policy_column = TEAM_STRENGTH_POLICY_COLUMNS.get(mapping_policy)
    if policy_column is None:
        raise ValueError(f"Unknown team-strength mapping policy: {mapping_policy!r}")
    required_columns = {
        "year",
        "race_name",
        "session_name",
        "session_kind",
        "team",
        "driver_code",
        "observed_driver_to_field_s",
        "driver_rating_mu_s",
        policy_column,
    }
    _require_columns(observations, required_columns, "observations")

    frames: list[pd.DataFrame] = []
    for session_kind in ("race", "qualifying"):
        mapping = mappings.get(session_kind)
        if mapping is None:
            continue
        rows = observations[
            observations["year"].eq(int(year)) & observations["session_kind"].eq(session_kind)
        ].dropna(
            subset=[
                "observed_driver_to_field_s",
                "driver_rating_mu_s",
                policy_column,
            ]
        )
        if rows.empty:
            continue

        enriched = rows.copy()
        policy_values = enriched[policy_column].astype(float)
        predicted_team_s = mapping.predict(policy_values)
        observed_driver_to_field_s = enriched["observed_driver_to_field_s"].astype(float)
        driver_rating_mu_s = enriched["driver_rating_mu_s"].astype(float)
        observed_team_target_s = observed_driver_to_field_s - driver_rating_mu_s
        predicted_driver_to_field_s = predicted_team_s + driver_rating_mu_s.to_numpy()
        residual_s = observed_driver_to_field_s.to_numpy() - predicted_driver_to_field_s

        enriched = _ensure_optional_columns(
            enriched,
            (
                "n_construct_laps",
                "n_field_drivers",
                "n_field_teams",
                "driver_median_s",
                "team_median_s",
            ),
        )
        enriched = enriched.assign(
            team_strength_policy_value=policy_values.to_numpy(),
            team_strength_centered=(policy_values - 0.5).to_numpy(),
            observed_driver_to_field_s=observed_driver_to_field_s.to_numpy(),
            driver_rating_mu_s=driver_rating_mu_s.to_numpy(),
            observed_team_target_s=observed_team_target_s.to_numpy(),
            predicted_team_s=predicted_team_s,
            predicted_driver_to_field_s=predicted_driver_to_field_s,
            residual_s=residual_s,
            abs_residual_s=np.abs(residual_s),
            mapping_intercept_s=mapping.intercept_s,
            mapping_slope_s_per_unit=mapping.slope_s_per_unit,
        )
        frames.append(
            enriched[
                [
                    "year",
                    "race_name",
                    "session_name",
                    "session_kind",
                    "team",
                    "driver_code",
                    "n_construct_laps",
                    "n_field_drivers",
                    "n_field_teams",
                    "driver_median_s",
                    "team_median_s",
                    "observed_driver_to_field_s",
                    "driver_rating_mu_s",
                    "observed_team_target_s",
                    "team_strength_policy_value",
                    "team_strength_centered",
                    "predicted_team_s",
                    "predicted_driver_to_field_s",
                    "residual_s",
                    "abs_residual_s",
                    "mapping_intercept_s",
                    "mapping_slope_s_per_unit",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["session_kind", "race_name", "team", "driver_code"])
        .reset_index(drop=True)
    )


def format_team_strength_construct_audit_markdown(artifact: Mapping[str, Any]) -> str:
    """Format the team-strength construct-row audit as Markdown for review."""
    lines = [
        "# Team-Strength Construct-Row Audit",
        "",
        f"- Built at: `{artifact.get('built_at')}`",
        f"- Model version: `{artifact.get('model_version')}`",
        f"- Status: `{artifact.get('status')}`",
        f"- Policy: `{artifact.get('policy')}`",
        f"- Rows: `{artifact.get('row_count', 0)}`",
        "",
    ]
    if artifact.get("reason"):
        lines.extend([f"Reason: {artifact.get('reason')}", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Scale Metrics",
            "",
            "| Session | Rows | Races | Teams | Drivers | Combined slope | Team-target slope | RMSE | Outside 1SE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    metrics = artifact.get("metrics_by_session_kind", {})
    for session_kind in ("race", "qualifying"):
        row = metrics.get(session_kind, {}) if isinstance(metrics, Mapping) else {}
        lines.append(
            f"| `{session_kind}` | {row.get('n_rows', 0)} | {row.get('n_races', 0)} | "
            f"{row.get('n_teams', 0)} | {row.get('n_drivers', 0)} | "
            f"{_fmt(row.get('prediction_slope'))} | {_fmt(row.get('team_target_slope'))} | "
            f"{_fmt(row.get('rmse_s'))} | `{row.get('outside_historical_one_se_band')}` |"
        )
    lines.append("")

    lines.extend(["## Highest Leave-One-Race Influence", ""])
    _append_leave_one_table(lines, artifact.get("leave_one_race", {}), group_label="race")
    lines.extend(["", "## Highest Leave-One-Team Influence", ""])
    _append_leave_one_table(lines, artifact.get("leave_one_team", {}), group_label="team")

    lines.extend(["", "## Largest Absolute Residual Rows", ""])
    residual_rows = artifact.get("largest_abs_residual_rows", [])
    if isinstance(residual_rows, list) and residual_rows:
        lines.extend(
            [
                "| Session | Race | Team | Driver | Observed | Predicted | Residual | Laps |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in residual_rows[:10]:
            lines.append(
                f"| `{row.get('session_kind')}` | {row.get('race_name')} | {row.get('team')} | "
                f"{row.get('driver_code')} | {_fmt(row.get('observed_driver_to_field_s'))} | "
                f"{_fmt(row.get('predicted_driver_to_field_s'))} | "
                f"{_fmt(row.get('residual_s'))} | {row.get('n_construct_laps')} |"
            )
    else:
        lines.append("No residual rows available.")
    lines.append("")

    decision_notes = artifact.get("decision_notes", [])
    if isinstance(decision_notes, list) and decision_notes:
        lines.extend(["## Decision Notes", ""])
        lines.extend(f"- {note}" for note in decision_notes)
        lines.append("")
    return "\n".join(lines)


def _base_artifact(
    *,
    year: int,
    mapping_policy: str,
    historical_reference: Mapping[str, Any],
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build common artifact fields for available and unavailable audits."""
    return {
        "artifact_type": "team_strength_construct_row_audit",
        "schema_version": TEAM_STRENGTH_CONSTRUCT_AUDIT_SCHEMA_VERSION,
        "model_version": get_model_version(),
        "built_at": datetime.now(UTC).isoformat(),
        "year": int(year),
        "status": status,
        "reason": reason,
        "policy": mapping_policy,
        "historical_scale_reference": historical_reference,
    }


def _load_observations(
    *,
    observations_path: Path | None,
    raw_matched_laps_path: Path | None,
    prior_artifact_path: Path,
) -> pd.DataFrame | None:
    """Load prebuilt observations or rebuild them from raw matched-lap rows."""
    if observations_path is not None and observations_path.exists():
        return pd.read_csv(observations_path)
    if raw_matched_laps_path is None or not raw_matched_laps_path.exists():
        return None
    if not prior_artifact_path.exists():
        return None
    raw = pd.read_csv(raw_matched_laps_path)
    prior = _read_json(prior_artifact_path)
    return build_construct_aligned_driver_observations(
        raw,
        driver_mu_by_kind=_driver_mu_by_kind(prior),
    )


def _metrics_by_session_kind(
    rows: pd.DataFrame,
    *,
    historical_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate full-sample metrics for each audited construct."""
    metrics: dict[str, Any] = {}
    for session_kind in ("race", "qualifying"):
        kind_rows = rows[rows["session_kind"].eq(session_kind)]
        if kind_rows.empty:
            metrics[session_kind] = {"state": "not_available", "reason": "no rows"}
            continue
        numeric = _numeric_metrics(kind_rows)
        reference = _historical_reference_for(historical_reference, session_kind)
        numeric.update(
            {
                "state": "measured",
                "n_rows": int(len(kind_rows)),
                "n_races": int(kind_rows["race_name"].nunique()),
                "n_teams": int(kind_rows["team"].nunique()),
                "n_drivers": int(kind_rows["driver_code"].nunique()),
                "historical_2024_2025_prediction_slope_mean": reference.get(
                    "prediction_slope_mean"
                ),
                "historical_2024_2025_prediction_slope_se": reference.get("prediction_slope_se"),
                "outside_historical_one_se_band": _outside_one_se_band(
                    value=numeric.get("prediction_slope"),
                    mean=reference.get("prediction_slope_mean"),
                    se=reference.get("prediction_slope_se"),
                ),
                "clipped_abs_residual_p90": _clipped_metrics(kind_rows),
            }
        )
        metrics[session_kind] = numeric
    return metrics


def _numeric_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Return slope and residual metrics for one filtered row set."""
    observed = rows["observed_driver_to_field_s"].astype(float).to_numpy()
    predicted = rows["predicted_driver_to_field_s"].astype(float).to_numpy()
    team_target = rows["observed_team_target_s"].astype(float).to_numpy()
    predicted_team = rows["predicted_team_s"].astype(float).to_numpy()
    residual = observed - predicted
    return {
        "prediction_slope": _prediction_slope(observed=observed, predicted=predicted),
        "prediction_intercept_s": _prediction_intercept(
            observed=observed,
            predicted=predicted,
        ),
        "team_target_slope": _prediction_slope(observed=team_target, predicted=predicted_team),
        "team_target_intercept_s": _prediction_intercept(
            observed=team_target,
            predicted=predicted_team,
        ),
        "r_squared": _r_squared(observed=observed, predicted=predicted),
        "team_target_r_squared": _r_squared(observed=team_target, predicted=predicted_team),
        "rmse_s": float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else None,
        "residual_mean_s": float(np.mean(residual)) if len(residual) else None,
        "abs_residual_mean_s": float(np.mean(np.abs(residual))) if len(residual) else None,
        "observed_std_s": _std_or_none(observed),
        "predicted_std_s": _std_or_none(predicted),
        "team_target_std_s": _std_or_none(team_target),
        "predicted_team_std_s": _std_or_none(predicted_team),
    }


def _clipped_metrics(rows: pd.DataFrame) -> dict[str, Any] | None:
    """Return metrics after excluding the largest residual decile."""
    if len(rows) < 10:
        return None
    threshold = float(rows["abs_residual_s"].quantile(0.90))
    clipped = rows[rows["abs_residual_s"].le(threshold)]
    if len(clipped) < 2:
        return None
    metrics = _numeric_metrics(clipped)
    return {
        "n_rows": int(len(clipped)),
        "abs_residual_threshold_s": threshold,
        "prediction_slope": metrics.get("prediction_slope"),
        "team_target_slope": metrics.get("team_target_slope"),
        "rmse_s": metrics.get("rmse_s"),
    }


def _leave_one_group_diagnostics(
    rows: pd.DataFrame,
    *,
    group_column: str,
    historical_reference: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Measure how much each omitted race or team changes the scale metric."""
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for session_kind in ("race", "qualifying"):
        kind_rows = rows[rows["session_kind"].eq(session_kind)]
        if kind_rows.empty:
            diagnostics[session_kind] = []
            continue
        full_metrics = _numeric_metrics(kind_rows)
        full_slope = _coerce_float(full_metrics.get("prediction_slope"))
        full_team_slope = _coerce_float(full_metrics.get("team_target_slope"))
        reference = _historical_reference_for(historical_reference, session_kind)
        group_rows: list[dict[str, Any]] = []
        for group_value in sorted(kind_rows[group_column].dropna().unique()):
            remaining = kind_rows[~kind_rows[group_column].eq(group_value)]
            if len(remaining) < 2:
                continue
            metrics = _numeric_metrics(remaining)
            slope = _coerce_float(metrics.get("prediction_slope"))
            team_slope = _coerce_float(metrics.get("team_target_slope"))
            group_rows.append(
                {
                    f"omitted_{group_column}": str(group_value),
                    "omitted_rows": int(kind_rows[group_column].eq(group_value).sum()),
                    "remaining_rows": int(len(remaining)),
                    "prediction_slope": slope,
                    "prediction_slope_delta_vs_full": _delta(slope, full_slope),
                    "team_target_slope": team_slope,
                    "team_target_slope_delta_vs_full": _delta(team_slope, full_team_slope),
                    "rmse_s": metrics.get("rmse_s"),
                    "outside_historical_one_se_band": _outside_one_se_band(
                        value=slope,
                        mean=reference.get("prediction_slope_mean"),
                        se=reference.get("prediction_slope_se"),
                    ),
                }
            )
        diagnostics[session_kind] = sorted(
            group_rows,
            key=lambda row: abs(float(row.get("prediction_slope_delta_vs_full") or 0.0)),
            reverse=True,
        )
    return diagnostics


def _group_residuals(rows: pd.DataFrame, group_columns: list[str]) -> list[dict[str, Any]]:
    """Summarize residuals for one grouping definition."""
    grouped = (
        rows.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n_rows=("residual_s", "size"),
            n_drivers=("driver_code", "nunique"),
            n_teams=("team", "nunique"),
            residual_mean_s=("residual_s", "mean"),
            abs_residual_mean_s=("abs_residual_s", "mean"),
            rmse_s=("residual_s", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            observed_mean_s=("observed_driver_to_field_s", "mean"),
            predicted_mean_s=("predicted_driver_to_field_s", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return _frame_records(_round_frame(grouped))


def _top_group_residuals(
    rows: pd.DataFrame,
    group_columns: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the largest grouped residual means by absolute size."""
    grouped = pd.DataFrame(_group_residuals(rows, group_columns))
    if grouped.empty:
        return []
    grouped["abs_residual_sort"] = grouped["residual_mean_s"].abs()
    grouped = grouped.sort_values("abs_residual_sort", ascending=False).drop(
        columns=["abs_residual_sort"]
    )
    return _frame_records(grouped.head(int(limit)))


def _top_residual_rows(rows: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    """Return the largest row-level residuals by absolute size."""
    columns = [
        "session_kind",
        "race_name",
        "session_name",
        "team",
        "driver_code",
        "n_construct_laps",
        "observed_driver_to_field_s",
        "driver_rating_mu_s",
        "observed_team_target_s",
        "team_strength_policy_value",
        "predicted_team_s",
        "predicted_driver_to_field_s",
        "residual_s",
        "abs_residual_s",
    ]
    ranked = rows.sort_values("abs_residual_s", ascending=False)[columns].head(int(limit))
    return _frame_records(_round_frame(ranked))


def _decision_notes(
    metrics: Mapping[str, Any],
    leave_one_race: Mapping[str, list[dict[str, Any]]],
    leave_one_team: Mapping[str, list[dict[str, Any]]],
) -> list[str]:
    """Build concise audit notes without making a refit decision."""
    notes = [
        "This audit is diagnostic-only; it does not retune extractor semantics or priors.",
        "Rows use the frozen same-session construct and the current teammate-network priors.",
    ]
    for session_kind in ("race", "qualifying"):
        row = metrics.get(session_kind, {}) if isinstance(metrics, Mapping) else {}
        if not isinstance(row, Mapping) or row.get("state") != "measured":
            continue
        if row.get("outside_historical_one_se_band") is True:
            notes.append(
                f"{session_kind} remains outside the 2024-2025 one-SE slope band in the full sample."
            )
        race_rows = leave_one_race.get(session_kind, [])
        if race_rows:
            top_race = race_rows[0]
            omitted = top_race.get("omitted_race_name")
            delta = _fmt(top_race.get("prediction_slope_delta_vs_full"))
            state = top_race.get("outside_historical_one_se_band")
            notes.append(
                f"{session_kind} top leave-one-race influence is {omitted} "
                f"(slope delta {delta}, outside band after omission: {state})."
            )
        team_rows = leave_one_team.get(session_kind, [])
        if team_rows:
            top_team = team_rows[0]
            omitted = top_team.get("omitted_team")
            delta = _fmt(top_team.get("prediction_slope_delta_vs_full"))
            state = top_team.get("outside_historical_one_se_band")
            notes.append(
                f"{session_kind} top leave-one-team influence is {omitted} "
                f"(slope delta {delta}, outside band after omission: {state})."
            )
    return notes


def _append_leave_one_table(
    lines: list[str],
    payload: Any,
    *,
    group_label: str,
) -> None:
    """Append the highest leave-one rows to an existing Markdown buffer."""
    if not isinstance(payload, Mapping):
        lines.append("No leave-one diagnostics available.")
        return
    lines.extend(
        [
            "| Session | Omitted | Rows | Slope | Delta | RMSE | Outside 1SE |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    had_rows = False
    for session_kind in ("race", "qualifying"):
        rows = payload.get(session_kind, [])
        if not isinstance(rows, list) or not rows:
            continue
        for row in rows[:3]:
            omitted = row.get(f"omitted_{'race_name' if group_label == 'race' else 'team'}")
            lines.append(
                f"| `{session_kind}` | {omitted} | {row.get('omitted_rows')} | "
                f"{_fmt(row.get('prediction_slope'))} | "
                f"{_fmt(row.get('prediction_slope_delta_vs_full'))} | "
                f"{_fmt(row.get('rmse_s'))} | `{row.get('outside_historical_one_se_band')}` |"
            )
            had_rows = True
    if not had_rows:
        lines.append("| - | - | 0 | - | - | - | - |")


def _historical_reference_for(
    historical_reference: Mapping[str, Any],
    session_kind: str,
) -> Mapping[str, Any]:
    """Return the historical reference band for one construct."""
    bands = historical_reference.get("reference_band_2024_2025", {})
    if not isinstance(bands, Mapping):
        return {}
    band = bands.get(session_kind, {})
    return band if isinstance(band, Mapping) else {}


def _driver_mu_by_kind(prior_artifact: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Extract race and qualifying prior means from the teammate-network artifact."""
    return {
        "race": _driver_mu_map(prior_artifact.get("race_network", {})),
        "qualifying": _driver_mu_map(prior_artifact.get("quali_network", {})),
    }


def _driver_mu_map(network_payload: Any) -> dict[str, float]:
    """Extract driver means from one network payload."""
    if not isinstance(network_payload, Mapping):
        return {}
    drivers = network_payload.get("drivers", {})
    if not isinstance(drivers, Mapping):
        return {}
    return {
        str(driver_code): float(payload["mu_s"])
        for driver_code, payload in drivers.items()
        if isinstance(payload, Mapping) and payload.get("mu_s") is not None
    }


def _ensure_optional_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Ensure optional reporting columns exist before selecting audit fields."""
    enriched = frame.copy()
    for column in columns:
        if column not in enriched.columns:
            enriched[column] = None
    return enriched


def _prediction_slope(*, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return the fitted slope of observed values on predicted values."""
    if len(observed) < 2 or np.isclose(float(np.nanstd(predicted)), 0.0):
        return None
    design = np.column_stack([np.ones(len(predicted), dtype=float), predicted])
    return float(np.linalg.lstsq(design, observed, rcond=None)[0][1])


def _prediction_intercept(*, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return the fitted intercept of observed values on predicted values."""
    if len(observed) < 2 or np.isclose(float(np.nanstd(predicted)), 0.0):
        return None
    design = np.column_stack([np.ones(len(predicted), dtype=float), predicted])
    return float(np.linalg.lstsq(design, observed, rcond=None)[0][0])


def _r_squared(*, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return R-squared for one observed/predicted vector pair."""
    if len(observed) == 0:
        return None
    total = float(np.sum(np.square(observed - float(np.mean(observed)))))
    if np.isclose(total, 0.0):
        return None
    residual = float(np.sum(np.square(observed - predicted)))
    return float(1.0 - (residual / total))


def _outside_one_se_band(*, value: Any, mean: Any, se: Any) -> bool | None:
    """Return whether a value sits outside a mean plus or minus one SE."""
    numeric_value = _coerce_float(value)
    numeric_mean = _coerce_float(mean)
    numeric_se = _coerce_float(se)
    if numeric_value is None or numeric_mean is None or numeric_se is None:
        return None
    if numeric_se <= 0.0:
        return bool(not np.isclose(numeric_value, numeric_mean))
    return bool(
        numeric_value < numeric_mean - numeric_se or numeric_value > numeric_mean + numeric_se
    )


def _std_or_none(values: np.ndarray) -> float | None:
    """Return a finite population standard deviation or None."""
    if len(values) == 0:
        return None
    std = float(np.nanstd(values))
    return std if np.isfinite(std) else None


def _delta(value: float | None, baseline: float | None) -> float | None:
    """Return value minus baseline when both values are finite."""
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def _coerce_float(value: Any) -> float | None:
    """Convert a value to a finite float or return None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _round_frame(frame: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Round float columns for stable JSON and Markdown output."""
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["float", "float64", "float32"]).columns:
        rounded[column] = rounded[column].round(decimals)
    return rounded


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe record dictionaries."""
    return _json_safe(frame.to_dict(orient="records"))


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = sorted(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _fmt(value: Any) -> str:
    """Format optional numbers for Markdown."""
    numeric = _coerce_float(value)
    return " - " if numeric is None else f"{numeric:.3f}"


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
    if pd.isna(value):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return value
    return value
