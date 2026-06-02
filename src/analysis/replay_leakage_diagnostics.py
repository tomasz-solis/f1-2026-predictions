"""Replay, leakage, and regulation-reset diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.team_strength_mapping import (
    TEAM_STRENGTH_POLICY_COLUMNS,
    LinearTeamStrengthMapping,
    build_construct_aligned_driver_observations,
    load_live_team_strength_mappings,
)
from src.utils.model_version import get_model_version

REPLAY_LEAKAGE_ARTIFACT_TYPE = "model_diagnostics"
REPLAY_LEAKAGE_ARTIFACT_KEY_TEMPLATE = "{year}::replay_leakage_diagnostics"
REPLAY_LEAKAGE_SCHEMA_VERSION = 2


def build_replay_leakage_diagnostics(
    *,
    year: int,
    replay_root: str | Path = "data/historical_replay",
    mapping_artifact_path: str | Path = "data/processed/team_strength_seconds_mapping/latest.json",
    candidate_diagnostics_path: str | Path = (
        "data/processed/team_strength_seconds_mapping/candidate_diagnostics.json"
    ),
    prior_artifact_path: str | Path = "data/processed/teammate_network_prior/latest.json",
    regulation_reset_observations_path: str | Path | None = None,
    regulation_reset_raw_matched_laps_path: str | Path | None = None,
    baseline_driver_path: str | Path = "data/processed/driver_characteristics.json",
    current_driver_path: str | Path | None = None,
    current_car_path: str | Path | None = None,
    lineup_path: str | Path = "data/current_lineups.json",
) -> dict[str, Any]:
    """Build the replay/leakage diagnostics artifact from persisted replay inputs.

    Dry leakage prefers seconds-native race state. It falls back to a clearly
    labeled legacy ``bayesian.rating_mu`` proxy only while replay inputs still
    lack the migrated baseline/current race seconds fields.
    """
    season_year = int(year)
    replay_root_path = Path(replay_root)
    mapping_path = Path(mapping_artifact_path)
    candidate_path = Path(candidate_diagnostics_path)
    prior_path = Path(prior_artifact_path)
    current_driver = Path(current_driver_path or _season_driver_path(season_year))
    current_car = Path(current_car_path or _season_car_path(season_year))

    mapping_payload = _read_json(mapping_path)
    candidate_payload = _read_json(candidate_path) if candidate_path.exists() else {}
    mappings = load_live_team_strength_mappings(mapping_path)

    replay_summary = _load_replay_summary(replay_root_path)
    historical_reference = build_historical_scale_reference(
        mapping_payload=mapping_payload,
        candidate_payload=candidate_payload,
    )
    regulation_reset = build_regulation_reset_monitoring(
        year=season_year,
        mappings=mappings,
        mapping_policy=str(mapping_payload.get("policy", "same_session_construct")),
        historical_reference=historical_reference,
        prior_artifact_path=prior_path,
        observations_path=(
            Path(regulation_reset_observations_path)
            if regulation_reset_observations_path is not None
            else None
        ),
        raw_matched_laps_path=(
            Path(regulation_reset_raw_matched_laps_path)
            if regulation_reset_raw_matched_laps_path is not None
            else None
        ),
    )
    dry_leakage = build_dry_leakage_status(
        year=season_year,
        race_mapping=mappings.get("race"),
        baseline_driver_path=Path(baseline_driver_path),
        current_driver_path=current_driver,
        current_car_path=current_car,
        lineup_path=Path(lineup_path),
    )
    wet_leakage = build_wet_leakage_status(
        baseline_driver_path=Path(baseline_driver_path),
        current_driver_path=current_driver,
        raw_matched_laps_path=(
            Path(regulation_reset_raw_matched_laps_path)
            if regulation_reset_raw_matched_laps_path is not None
            else None
        ),
        driver_update_trace_path=replay_root_path / "reports" / "driver_update_trace.json",
    )

    source_state = build_source_state(
        year=season_year,
        replay_root=replay_root_path,
        replay_summary=replay_summary,
        current_car_path=current_car,
    )

    warnings = _diagnostic_warnings(
        source_state=source_state,
        regulation_reset=regulation_reset,
    )
    limitations = _diagnostic_limitations(
        dry_leakage=dry_leakage,
        wet_leakage=wet_leakage,
    )
    monitoring_notes = _diagnostic_monitoring_notes(regulation_reset=regulation_reset)

    return _json_safe(
        {
            "artifact_type": "replay_leakage_diagnostics",
            "schema_version": REPLAY_LEAKAGE_SCHEMA_VERSION,
            "model_version": get_model_version(),
            "built_at": datetime.now(UTC).isoformat(),
            "year": season_year,
            "status": _overall_status(warnings=warnings, limitations=limitations),
            "source_state": source_state,
            "historical_scale_reference": historical_reference,
            "regulation_reset_monitoring": regulation_reset,
            "dry_leakage": dry_leakage,
            "wet_leakage": wet_leakage,
            "warnings": warnings,
            "limitations": limitations,
            "monitoring_notes": monitoring_notes,
            "decision_notes": [
                "This artifact is diagnostic-only; it does not retune extractor semantics.",
                (
                    "Dry leakage uses race driver seconds when baseline and current "
                    "driver artifacts both expose that state; otherwise it reports "
                    "the legacy rating-mu proxy."
                ),
            ],
        }
    )


def replay_leakage_artifact_key(year: int) -> str:
    """Return the stable artifact key used by file and DB persistence."""
    return REPLAY_LEAKAGE_ARTIFACT_KEY_TEMPLATE.format(year=int(year))


def build_historical_scale_reference(
    *,
    mapping_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize historical held-out calibration folds and residual means."""
    selected_policy = str(mapping_payload.get("policy", "same_session_construct"))
    folds = list(mapping_payload.get("validation", {}).get("folds", []))
    reference_band = {
        session_kind: _reference_band(folds, session_kind=session_kind)
        for session_kind in ("race", "qualifying")
    }

    residual_rows = (
        candidate_payload.get("policy_evaluations", {})
        .get(selected_policy, {})
        .get("per_driver_residual_means", [])
    )
    residual_rows = residual_rows if isinstance(residual_rows, list) else []
    return {
        "policy": selected_policy,
        "folds": folds,
        "reference_band_2024_2025": reference_band,
        "per_driver_residual_means": _sorted_residual_rows(residual_rows),
        "residual_outliers": _residual_outliers(residual_rows),
    }


def build_regulation_reset_monitoring(
    *,
    year: int,
    mappings: Mapping[str, LinearTeamStrengthMapping],
    mapping_policy: str,
    historical_reference: Mapping[str, Any],
    prior_artifact_path: Path,
    observations_path: Path | None = None,
    raw_matched_laps_path: Path | None = None,
) -> dict[str, Any]:
    """Measure 2026 transfer behavior when construct observations are available."""
    observations = _load_regulation_reset_observations(
        observations_path=observations_path,
        raw_matched_laps_path=raw_matched_laps_path,
        prior_artifact_path=prior_artifact_path,
    )
    if observations is None or observations.empty:
        return {
            "state": "not_available",
            "reason": (
                "No regulation-reset matched-lap observations were supplied. "
                "Run the 2026 matched-lap extraction before treating transfer "
                "monitoring as complete."
            ),
            "metrics_by_session_kind": {},
            "per_driver_residual_means": [],
        }

    policy_column = TEAM_STRENGTH_POLICY_COLUMNS.get(mapping_policy)
    if not policy_column or policy_column not in observations.columns:
        return {
            "state": "not_available",
            "reason": f"Policy column missing for mapping policy {mapping_policy!r}.",
            "metrics_by_session_kind": {},
            "per_driver_residual_means": [],
        }

    metrics: dict[str, Any] = {}
    prediction_rows: list[pd.DataFrame] = []
    for session_kind in ("race", "qualifying"):
        mapping = mappings.get(session_kind)
        if mapping is None:
            metrics[session_kind] = {"state": "not_available", "reason": "mapping missing"}
            continue

        kind_rows = observations[
            observations["session_kind"].eq(session_kind) & observations["year"].eq(int(year))
        ].dropna(
            subset=[
                policy_column,
                "observed_driver_to_field_s",
                "driver_rating_mu_s",
            ]
        )
        if kind_rows.empty:
            metrics[session_kind] = {"state": "not_available", "reason": "no usable rows"}
            continue

        predicted_team_s = mapping.predict(kind_rows[policy_column])
        predicted_driver_to_field_s = (
            predicted_team_s + kind_rows["driver_rating_mu_s"].astype(float).to_numpy()
        )
        observed = kind_rows["observed_driver_to_field_s"].astype(float).to_numpy()
        residual = observed - predicted_driver_to_field_s
        slope = _prediction_slope(observed=observed, predicted=predicted_driver_to_field_s)
        r_squared = _r_squared(observed=observed, predicted=predicted_driver_to_field_s)
        reference = (
            historical_reference.get("reference_band_2024_2025", {}).get(session_kind, {})
            if isinstance(historical_reference.get("reference_band_2024_2025"), Mapping)
            else {}
        )
        metrics[session_kind] = {
            "state": "measured",
            "n_rows": int(len(kind_rows)),
            "n_races": int(kind_rows["race_name"].nunique()),
            "prediction_slope": slope,
            "r_squared": r_squared,
            "rmse_s": float(np.sqrt(np.mean(np.square(residual)))),
            "residual_mean_s": float(np.mean(residual)),
            "historical_2024_2025_prediction_slope_mean": reference.get("prediction_slope_mean"),
            "historical_2024_2025_prediction_slope_se": reference.get("prediction_slope_se"),
            "outside_historical_one_se_band": _outside_one_se_band(
                value=slope,
                mean=reference.get("prediction_slope_mean"),
                se=reference.get("prediction_slope_se"),
            ),
        }
        prediction_rows.append(
            kind_rows[
                [
                    "year",
                    "race_name",
                    "session_name",
                    "session_kind",
                    "team",
                    "driver_code",
                ]
            ]
            .assign(
                observed_driver_to_field_s=observed,
                predicted_driver_to_field_s=predicted_driver_to_field_s,
                residual_s=residual,
            )
            .reset_index(drop=True)
        )

    prediction_frame = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(columns=["session_kind", "driver_code", "residual_s"])
    )
    residual_means = (
        prediction_frame.groupby(["session_kind", "driver_code"], dropna=False, as_index=False)
        .agg(residual_mean_s=("residual_s", "mean"), n_rows=("residual_s", "size"))
        .sort_values(["session_kind", "driver_code"])
        .to_dict(orient="records")
    )
    return {
        "state": "measured"
        if any(row.get("state") == "measured" for row in metrics.values())
        else "not_available",
        "policy": mapping_policy,
        "metrics_by_session_kind": metrics,
        "per_driver_residual_means": _sorted_residual_rows(residual_means),
    }


def build_dry_leakage_status(
    *,
    year: int,
    race_mapping: LinearTeamStrengthMapping | None,
    baseline_driver_path: Path,
    current_driver_path: Path,
    current_car_path: Path,
    lineup_path: Path,
) -> dict[str, Any]:
    """Measure exact dry leakage when available, else report the legacy proxy."""
    if race_mapping is None:
        return {
            "state": "not_available",
            "reason": "race team-strength seconds mapping is missing",
            "rows": [],
        }
    required_paths = [baseline_driver_path, current_driver_path, current_car_path, lineup_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return {"state": "not_available", "reason": f"Missing inputs: {missing}", "rows": []}

    baseline_drivers = _drivers_payload(_read_json(baseline_driver_path))
    current_drivers = _drivers_payload(_read_json(current_driver_path))
    current_cars = _read_json(current_car_path).get("teams", {})
    driver_to_team = _driver_to_team(_read_json(lineup_path))

    exact_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    exact_missing_driver_codes: list[str] = []
    for driver_code, current_payload in sorted(current_drivers.items()):
        team = driver_to_team.get(driver_code)
        if not team or team not in current_cars or driver_code not in baseline_drivers:
            continue
        car_payload = current_cars.get(team, {})
        if not isinstance(car_payload, Mapping):
            continue
        baseline_strength = _coerce_float(car_payload.get("preseason_overall_performance"))
        current_strength = _coerce_float(car_payload.get("overall_performance"))
        if baseline_strength is None or current_strength is None:
            continue

        baseline_team_seconds = race_mapping.predict_delta_one(baseline_strength)
        current_team_seconds = race_mapping.predict_delta_one(current_strength)
        shared_row = {
            "driver_code": driver_code,
            "team": team,
            "baseline_team_strength": baseline_strength,
            "current_team_strength": current_strength,
            "delta_team_strength": float(current_strength - baseline_strength),
            "delta_team_seconds": float(current_team_seconds - baseline_team_seconds),
        }
        baseline_seconds = _driver_seconds_mu(baseline_drivers[driver_code], session_kind="race")
        current_seconds = _driver_seconds_mu(current_payload, session_kind="race")
        if baseline_seconds is None or current_seconds is None:
            exact_missing_driver_codes.append(driver_code)
        else:
            exact_rows.append(
                {
                    **shared_row,
                    "baseline_race_rating_mu_s": baseline_seconds,
                    "current_race_rating_mu_s": current_seconds,
                    "delta_race_rating_mu_s": float(current_seconds - baseline_seconds),
                }
            )

        baseline_mu = _rating_mu(baseline_drivers[driver_code])
        current_mu = _rating_mu(current_payload)
        if baseline_mu is not None and current_mu is not None:
            legacy_rows.append(
                {
                    **shared_row,
                    "baseline_rating_mu": baseline_mu,
                    "current_rating_mu": current_mu,
                    "delta_rating_mu": float(current_mu - baseline_mu),
                }
            )

    if exact_rows and not exact_missing_driver_codes:
        delta_driver_seconds = [row["delta_race_rating_mu_s"] for row in exact_rows]
        delta_team_seconds = [row["delta_team_seconds"] for row in exact_rows]
        return {
            "state": "measured_seconds",
            "year": int(year),
            "exact_metric_state": "measured",
            "driver_field": "bayesian.race_rating_mu_s",
            "driver_unit": "seconds",
            "team_field": "overall_performance - preseason_overall_performance",
            "team_unit": "race_team_strength_seconds_delta",
            "n_drivers": len(exact_rows),
            "correlation": _correlation(delta_driver_seconds, delta_team_seconds),
            "slope_driver_second_per_team_second": _prediction_slope(
                observed=np.asarray(delta_driver_seconds, dtype=float),
                predicted=np.asarray(delta_team_seconds, dtype=float),
            ),
            "rows": exact_rows,
        }

    delta_rating = [row["delta_rating_mu"] for row in legacy_rows]
    delta_team_seconds = [row["delta_team_seconds"] for row in legacy_rows]
    return {
        "state": "measured_legacy_proxy",
        "year": int(year),
        "exact_metric_state": "blocked_missing_race_seconds_state",
        "driver_field": "bayesian.rating_mu",
        "driver_unit": "legacy_grid_rank_mu",
        "team_field": "overall_performance - preseason_overall_performance",
        "team_unit": "race_team_strength_seconds_delta",
        "n_drivers": len(legacy_rows),
        "missing_race_seconds_driver_codes": sorted(set(exact_missing_driver_codes)),
        "correlation": _correlation(delta_rating, delta_team_seconds),
        "slope_rating_mu_per_team_second": _prediction_slope(
            observed=np.asarray(delta_rating, dtype=float),
            predicted=np.asarray(delta_team_seconds, dtype=float),
        ),
        "rows": legacy_rows,
    }


def build_wet_leakage_status(
    *,
    baseline_driver_path: Path,
    current_driver_path: Path,
    raw_matched_laps_path: Path | None,
    driver_update_trace_path: Path | None = None,
) -> dict[str, Any]:
    """Report wet-leakage evaluability without fabricating missing wet evidence."""
    wet_rows = 0
    mixed_rows = 0
    if raw_matched_laps_path is not None and raw_matched_laps_path.exists():
        raw = pd.read_csv(raw_matched_laps_path)
        if "weather_bucket" in raw.columns:
            weather = raw["weather_bucket"].astype(str).str.lower()
            wet_rows = int(weather.eq("wet").sum())
            mixed_rows = int(weather.str.contains("mixed|unreliable", regex=True).sum())

    proxy_corr = None
    if baseline_driver_path.exists() and current_driver_path.exists():
        baseline = _drivers_payload(_read_json(baseline_driver_path))
        current = _drivers_payload(_read_json(current_driver_path))
        rating_deltas: list[float] = []
        wet_deltas: list[float] = []
        for driver_code, current_payload in current.items():
            if driver_code not in baseline:
                continue
            baseline_mu = _rating_mu(baseline[driver_code])
            current_mu = _rating_mu(current_payload)
            baseline_wet = _coerce_float(baseline[driver_code].get("wet_skill"))
            current_wet = _coerce_float(current_payload.get("wet_skill"))
            if (
                baseline_mu is None
                or current_mu is None
                or baseline_wet is None
                or current_wet is None
            ):
                continue
            rating_deltas.append(float(current_mu - baseline_mu))
            wet_deltas.append(float(current_wet - baseline_wet))
        proxy_corr = _correlation(wet_deltas, rating_deltas)

    traced_invariant = evaluate_fully_wet_dry_update_invariant(
        _load_driver_update_trace_rows(driver_update_trace_path)
    )
    traced_invariant["wet_matched_rows"] = wet_rows
    traced_invariant["mixed_or_unreliable_matched_rows"] = mixed_rows
    status = _wet_leakage_state(
        wet_rows=wet_rows,
        mixed_rows=mixed_rows,
        traced_invariant=traced_invariant,
    )
    return {
        "state": status,
        "fully_wet_dry_update_invariant": traced_invariant,
        "legacy_wet_skill_delta_vs_rating_mu_delta_correlation": proxy_corr,
        "note": (
            "The hard wet invariant requires session-level update trace. "
            "Aggregate driver artifacts alone cannot prove that a wet session "
            "did not update dry ratings."
        ),
    }


def evaluate_fully_wet_dry_update_invariant(trace_rows: list[Any]) -> dict[str, Any]:
    """Evaluate whether fully wet traced sessions kept dry driver state still."""
    wet_rows = [
        row
        for row in trace_rows
        if isinstance(row, Mapping) and str(row.get("weather_route", "")).lower() == "rain"
    ]
    if not trace_rows:
        return {
            "state": "not_evaluable_without_session_update_trace",
            "fully_wet_trace_rows": 0,
            "violations": [],
        }
    if not wet_rows:
        return {
            "state": "not_evaluable_without_fully_wet_trace_rows",
            "fully_wet_trace_rows": 0,
            "violations": [],
        }

    violations = [
        violation for row in wet_rows if (violation := _wet_trace_violation(row)) is not None
    ]
    return {
        "state": "failed" if violations else "passed_from_update_trace",
        "fully_wet_trace_rows": len(wet_rows),
        "violations": violations,
    }


def _wet_leakage_state(
    *,
    wet_rows: int,
    mixed_rows: int,
    traced_invariant: Mapping[str, Any],
) -> str:
    """Return a compact wet-leakage state from coverage and trace evidence."""
    invariant_state = str(traced_invariant.get("state", ""))
    if invariant_state == "failed":
        return "failed_fully_wet_dry_update_invariant"
    if wet_rows == 0 and mixed_rows == 0:
        return "not_evaluable_without_weather_routed_wet_replay_rows"
    if invariant_state == "passed_from_update_trace":
        return "evaluated_from_session_update_trace"
    return "weather_rows_present_needs_session_update_trace"


def _wet_trace_violation(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return one violation when a fully wet trace row moved dry state."""
    session_kind = str(row.get("session_kind", "")).lower()
    relevant_flag = (
        "dry_quali_update_applied" if session_kind == "qualifying" else "dry_race_update_applied"
    )
    relevant_seconds_delta = (
        "quali_rating_mu_s_delta" if session_kind == "qualifying" else "race_rating_mu_s_delta"
    )
    moved_fields = [
        field
        for field in ("legacy_rating_mu_delta", relevant_seconds_delta)
        if _non_zero_delta(row.get(field))
    ]
    dry_update_flagged = bool(row.get(relevant_flag))
    if not dry_update_flagged and not moved_fields:
        return None
    return {
        "event_name": row.get("event_name"),
        "session_name": row.get("session_name"),
        "session_kind": row.get("session_kind"),
        "driver_code": row.get("driver_code"),
        "dry_update_flag": relevant_flag if dry_update_flagged else None,
        "moved_fields": moved_fields,
    }


def _non_zero_delta(value: Any) -> bool:
    """Return whether one optional numeric trace delta is measurably non-zero."""
    numeric = _coerce_float(value)
    return numeric is not None and not np.isclose(numeric, 0.0)


def _load_driver_update_trace_rows(path: Path | None) -> list[Any]:
    """Load trace rows from a replay report when it exists."""
    if path is None or not path.exists():
        return []
    payload = _read_json(path)
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def build_source_state(
    *,
    year: int,
    replay_root: Path,
    replay_summary: Mapping[str, Any] | None,
    current_car_path: Path,
) -> dict[str, Any]:
    """Summarize whether the replay inputs match the current live artifacts."""
    current_races_completed = None
    if current_car_path.exists():
        current_races_completed = _coerce_float(_read_json(current_car_path).get("races_completed"))
    replay_races = (
        list(replay_summary.get("race_updates", [])) if isinstance(replay_summary, Mapping) else []
    )
    replay_checkpoints = (
        list(replay_summary.get("checkpoints", [])) if isinstance(replay_summary, Mapping) else []
    )
    replay_race_count = len(replay_races)
    is_stale = current_races_completed is not None and replay_race_count < int(
        current_races_completed
    )
    return {
        "year": int(year),
        "replay_root": str(replay_root),
        "replay_summary_found": replay_summary is not None,
        "replay_race_updates": replay_races,
        "replay_race_count": replay_race_count,
        "replay_checkpoint_count": len(replay_checkpoints),
        "live_artifact_races_completed": current_races_completed,
        "replay_stale_vs_live_artifact": bool(is_stale),
    }


def format_replay_leakage_diagnostics_markdown(artifact: Mapping[str, Any]) -> str:
    """Format the replay/leakage diagnostics artifact as concise Markdown."""
    lines = [
        "# Replay And Leakage Diagnostics",
        "",
        f"- Built at: `{artifact.get('built_at')}`",
        f"- Model version: `{artifact.get('model_version')}`",
        f"- Status: `{artifact.get('status')}`",
        "",
        "## Source State",
        "",
    ]
    source = artifact.get("source_state", {})
    lines.extend(
        [
            f"- Replay races: `{source.get('replay_race_count')}`",
            f"- Live artifact races completed: `{source.get('live_artifact_races_completed')}`",
            f"- Replay stale vs live artifact: `{source.get('replay_stale_vs_live_artifact')}`",
            "",
        ]
    )
    warnings = artifact.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    limitations = artifact.get("limitations", [])
    if limitations:
        lines.extend(["## Coverage Limitations", ""])
        for limitation in limitations:
            lines.append(f"- {limitation}")
        lines.append("")

    monitoring_notes = artifact.get("monitoring_notes", [])
    if monitoring_notes:
        lines.extend(["## Monitoring Notes", ""])
        for note in monitoring_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(["## Historical Reference", ""])
    for session_kind, band in (
        artifact.get("historical_scale_reference", {}).get("reference_band_2024_2025", {}).items()
    ):
        lines.append(
            f"- `{session_kind}`: slope mean `{_fmt(band.get('prediction_slope_mean'))}`, "
            f"slope SE `{_fmt(band.get('prediction_slope_se'))}`, "
            f"R² mean `{_fmt(band.get('r_squared_mean'))}`"
        )
    lines.append("")

    lines.extend(
        [
            "## Regulation-Reset Monitoring",
            "",
            "| Session | State | Rows | Races | Slope | R² | RMSE | Outside 1SE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    metrics = artifact.get("regulation_reset_monitoring", {}).get("metrics_by_session_kind", {})
    for session_kind in ("race", "qualifying"):
        row = metrics.get(session_kind, {}) if isinstance(metrics, Mapping) else {}
        lines.append(
            f"| `{session_kind}` | `{row.get('state', 'not_available')}` | "
            f"{row.get('n_rows', 0)} | {row.get('n_races', 0)} | "
            f"{_fmt(row.get('prediction_slope'))} | {_fmt(row.get('r_squared'))} | "
            f"{_fmt(row.get('rmse_s'))} | `{row.get('outside_historical_one_se_band')}` |"
        )
    lines.append("")

    dry = artifact.get("dry_leakage", {})
    lines.extend(
        [
            "## Dry Leakage",
            "",
            f"- State: `{dry.get('state')}`",
            f"- Exact metric state: `{dry.get('exact_metric_state')}`",
            f"- Correlation: `{_fmt(dry.get('correlation'))}`",
            f"- Drivers: `{dry.get('n_drivers', 0)}`",
            "",
        ]
    )

    wet = artifact.get("wet_leakage", {})
    lines.extend(
        [
            "## Wet Leakage",
            "",
            f"- State: `{wet.get('state')}`",
            f"- Hard invariant state: "
            f"`{wet.get('fully_wet_dry_update_invariant', {}).get('state')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _load_replay_summary(replay_root: Path) -> dict[str, Any] | None:
    """Load the historical replay summary if it exists."""
    summary_path = replay_root / "reports" / "summary.json"
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    return payload if isinstance(payload, dict) else None


def _load_regulation_reset_observations(
    *,
    observations_path: Path | None,
    raw_matched_laps_path: Path | None,
    prior_artifact_path: Path,
) -> pd.DataFrame | None:
    """Load or build construct-aligned regulation-reset observations."""
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


def _driver_mu_by_kind(prior_artifact: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Extract race and qualifying prior means from the teammate-network artifact."""
    return {
        "race": {
            str(driver_code): float(payload["mu_s"])
            for driver_code, payload in prior_artifact.get("race_network", {})
            .get("drivers", {})
            .items()
            if isinstance(payload, Mapping) and payload.get("mu_s") is not None
        },
        "qualifying": {
            str(driver_code): float(payload["mu_s"])
            for driver_code, payload in prior_artifact.get("quali_network", {})
            .get("drivers", {})
            .items()
            if isinstance(payload, Mapping) and payload.get("mu_s") is not None
        },
    }


def _reference_band(folds: list[Any], *, session_kind: str) -> dict[str, Any]:
    """Build a compact reference band from 2024-2025 held-out folds."""
    rows: list[Mapping[str, Any]] = []
    for row in folds:
        if not isinstance(row, Mapping) or row.get("session_kind") != session_kind:
            continue
        holdout_year = _coerce_int(row.get("holdout_year"))
        if holdout_year in {2024, 2025}:
            rows.append(row)
    return {
        "fold_years": [
            int(holdout_year)
            for row in rows
            if (holdout_year := _coerce_int(row.get("holdout_year"))) is not None
        ],
        "prediction_slope_mean": _mean(row.get("prediction_slope") for row in rows),
        "prediction_slope_se": _standard_error(row.get("prediction_slope") for row in rows),
        "r_squared_mean": _mean(row.get("r_squared") for row in rows),
        "rmse_s_mean": _mean(row.get("rmse_s") for row in rows),
    }


def _residual_outliers(rows: list[Any], *, threshold_s: float = 0.35) -> list[dict[str, Any]]:
    """Return residual rows whose absolute mean exceeds the review threshold."""
    outliers = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        residual = _coerce_float(row.get("residual_mean_s"))
        if residual is None or abs(residual) < threshold_s:
            continue
        outliers.append(dict(row))
    return _sorted_residual_rows(outliers)


def _sorted_residual_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Sort residual rows by construct and absolute residual size."""
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    return sorted(
        normalized,
        key=lambda row: (
            str(row.get("session_kind", "")),
            -abs(float(row.get("residual_mean_s") or 0.0)),
            str(row.get("driver_code", "")),
        ),
    )


def _diagnostic_warnings(
    *,
    source_state: Mapping[str, Any],
    regulation_reset: Mapping[str, Any],
) -> list[str]:
    """Collect plain-language warnings for reviewers."""
    warnings: list[str] = []
    if source_state.get("replay_stale_vs_live_artifact"):
        warnings.append(
            "Historical replay output is stale relative to the live 2026 artifact race count."
        )
    if regulation_reset.get("state") != "measured":
        warnings.append(
            "Regulation-reset seconds monitoring has no measured 2026 construct rows yet."
        )
    return warnings


def _diagnostic_limitations(
    *,
    dry_leakage: Mapping[str, Any],
    wet_leakage: Mapping[str, Any],
) -> list[str]:
    """Collect expected coverage gaps that are not diagnostic failures."""
    limitations: list[str] = []
    if dry_leakage.get("exact_metric_state") == "blocked_missing_race_seconds_state":
        limitations.append(
            "Exact dry-leakage seconds measurement needs baseline and current race "
            "driver-seconds state; the current rating-mu value is a legacy proxy."
        )

    wet_state = wet_leakage.get("state")
    if wet_state == "not_evaluable_without_weather_routed_wet_replay_rows":
        limitations.append(
            "Current replay coverage has no wet weather-routed rows, so the wet-leakage "
            "replay invariant has no real 2026 wet sample yet."
        )
    elif wet_state == "weather_rows_present_needs_session_update_trace":
        limitations.append(
            "Wet weather-routed rows are present, but the wet-leakage replay invariant "
            "still needs a session-level driver-update trace."
        )
    return limitations


def _diagnostic_monitoring_notes(*, regulation_reset: Mapping[str, Any]) -> list[str]:
    """Collect non-failing context notes for transfer monitoring."""
    if regulation_reset.get("state") != "measured":
        return []

    metrics = regulation_reset.get("metrics_by_session_kind", {})
    if not isinstance(metrics, Mapping):
        return []

    outside = [
        str(session_kind)
        for session_kind, payload in metrics.items()
        if isinstance(payload, Mapping) and payload.get("outside_historical_one_se_band") is True
    ]
    if not outside:
        return []
    return [
        (
            "Reset-year scale differs from the 2024-2025 reference band for "
            f"{', '.join(sorted(outside))}. The comparison stays visible for transfer review; "
            "it is not a warning by itself."
        )
    ]


def _overall_status(*, warnings: list[str], limitations: list[str]) -> str:
    """Return a compact artifact status from warnings and coverage gaps."""
    if warnings:
        return "provisional_with_warnings"
    if limitations:
        return "provisional_with_limitations"
    return "measured"


def _drivers_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the driver dictionary from a characteristics payload."""
    drivers = payload.get("drivers", {})
    return {
        str(driver_code): dict(driver_payload)
        for driver_code, driver_payload in drivers.items()
        if isinstance(driver_payload, Mapping)
    }


def _driver_to_team(lineup_payload: Mapping[str, Any]) -> dict[str, str]:
    """Invert the current-lineups payload into a driver-to-team map."""
    lineups = lineup_payload.get("current_lineups", lineup_payload)
    if not isinstance(lineups, Mapping):
        return {}
    return {
        str(driver_code): str(team_name)
        for team_name, drivers in lineups.items()
        if isinstance(drivers, list)
        for driver_code in drivers
    }


def _rating_mu(driver_payload: Mapping[str, Any]) -> float | None:
    """Read legacy Bayesian rating_mu from a driver payload."""
    bayesian = driver_payload.get("bayesian", {})
    if not isinstance(bayesian, Mapping):
        return None
    return _coerce_float(bayesian.get("rating_mu"))


def _driver_seconds_mu(
    driver_payload: Mapping[str, Any],
    *,
    session_kind: str,
) -> float | None:
    """Read one persisted race or qualifying driver mean in seconds."""
    bayesian = driver_payload.get("bayesian", {})
    if not isinstance(bayesian, Mapping):
        return None
    field = "race_rating_mu_s" if session_kind == "race" else "quali_rating_mu_s"
    return _coerce_float(bayesian.get(field))


def _season_driver_path(year: int) -> Path:
    """Return the default current-season driver artifact path."""
    return (
        Path("data/processed/driver_characteristics") / f"{int(year)}_driver_characteristics.json"
    )


def _season_car_path(year: int) -> Path:
    """Return the default current-season car artifact path."""
    return Path("data/processed/car_characteristics") / f"{int(year)}_car_characteristics.json"


def _prediction_slope(*, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return the fitted slope of observed values on predicted values."""
    if len(observed) < 2 or np.isclose(float(np.nanstd(predicted)), 0.0):
        return None
    design = np.column_stack([np.ones(len(predicted), dtype=float), predicted])
    return float(np.linalg.lstsq(design, observed, rcond=None)[0][1])


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
    """Return whether a value sits outside a mean ± one-SE band."""
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


def _correlation(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float | None:
    """Return Pearson correlation for two numeric vectors."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if len(left_arr) < 2 or len(left_arr) != len(right_arr):
        return None
    if np.isclose(float(np.nanstd(left_arr)), 0.0) or np.isclose(float(np.nanstd(right_arr)), 0.0):
        return None
    corr = np.corrcoef(left_arr, right_arr)[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _mean(values: Any) -> float | None:
    """Return a float mean while ignoring null values."""
    numeric = [_coerce_float(value) for value in values]
    clean = [value for value in numeric if value is not None]
    return float(np.mean(clean)) if clean else None


def _standard_error(values: Any) -> float | None:
    """Return the sample standard error while ignoring null values."""
    numeric = [_coerce_float(value) for value in values]
    clean = [value for value in numeric if value is not None]
    if len(clean) < 2:
        return None
    return float(np.std(clean, ddof=1) / np.sqrt(len(clean)))


def _coerce_float(value: Any) -> float | None:
    """Convert a value to finite float or return None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _coerce_int(value: Any) -> int | None:
    """Convert a value to int or return None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""
    with path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


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
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
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
