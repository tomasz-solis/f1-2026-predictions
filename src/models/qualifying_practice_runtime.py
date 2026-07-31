"""Adapter between qualifying-practice sidecars and the Q1 utility model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.qualifying_practice_challenger import (
    FittedQualifyingPracticeModel,
    simulate_plackett_luce_grids,
)
from src.models.qualifying_practice_evidence import (
    FittedPracticeNormalization,
    build_qualifying_practice_evidence,
)

_PACE_FEATURES = {
    "best_adjusted_lap_s",
    "best2_mean_s",
    "adjusted_q20_s",
    "theoretical_lap_s",
    "execution_loss_s",
    "consistency_mad_s",
    "session_improvement_s",
    "teammate_gap_s",
    "compound_adjustment_se_s",
    "measurement_se_s",
}
_SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3, "SQ": 4, "SPRINT": 5}
MIN_Q1_DRIVER_COVERAGE = 0.50
MIN_Q1_TEAM_COVERAGE = 0.50


def build_weekend_qualifying_practice_evidence(
    session_laps_by_type: Mapping[str, pd.DataFrame | None],
    *,
    normalization: FittedPracticeNormalization,
    track_name: str,
    track_class: str,
) -> dict[str, dict[str, Any]]:
    """Build chronological dry sidecars from already-loaded practice laps."""

    evidence: dict[str, dict[str, Any]] = {}
    for raw_session, laps in sorted(
        session_laps_by_type.items(),
        key=lambda item: (_SESSION_ORDER.get(str(item[0]).strip().upper(), 99), str(item[0])),
    ):
        if not isinstance(laps, pd.DataFrame) or laps.empty:
            continue
        session_code = str(raw_session).strip().upper()
        compounds = (
            set(laps["Compound"].dropna().astype(str).str.upper())
            if "Compound" in laps.columns
            else set()
        )
        session_is_dry = (
            bool(compounds)
            and not bool(compounds.intersection({"INTERMEDIATE", "WET"}))
            and not _session_has_rainfall(laps)
        )
        evidence[session_code] = build_qualifying_practice_evidence(
            laps,
            session_code=session_code,
            session_is_dry=session_is_dry,
            track_name=track_name,
            track_class=track_class,
            normalization=normalization,
        )
    return evidence


def _session_has_rainfall(laps: pd.DataFrame) -> bool:
    """Fail the dry-only Q1 path when any lap records rain explicitly."""

    if "Rainfall" not in laps.columns:
        return False
    for value in laps["Rainfall"].dropna():
        if isinstance(value, (bool, np.bool_)):
            if bool(value):
                return True
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(float(value)) and float(value) != 0.0:
                return True
            continue
        if str(value).strip().lower() in {"1", "true", "yes", "y", "rain", "wet"}:
            return True
    return False


def load_practice_normalization(path: str | Path) -> FittedPracticeNormalization | None:
    """Load the leakage-safe fitted normalization required by Q1 runtime."""

    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("Practice normalization artifact must be an object")
    if payload.get("artifact_type") == "qualifying_practice_normalization":
        raw = payload.get("normalization")
        payload = raw if isinstance(raw, Mapping) else {}
    coverage = payload.get("comparison_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    effects = payload.get("compound_effect_s")
    effects = effects if isinstance(effects, Mapping) else {}
    coefficient_provenance = payload.get("coefficient_provenance")
    coefficient_provenance = (
        coefficient_provenance if isinstance(coefficient_provenance, Mapping) else {}
    )
    return FittedPracticeNormalization(
        reference_compound=str(payload.get("reference_compound", "SOFT")),
        compound_effect_s={str(key): float(value) for key, value in effects.items()},
        tyre_age_effect_s_per_lap=float(payload.get("tyre_age_effect_s_per_lap", 0.0)),
        evolution_effect_s_per_unit=float(payload.get("evolution_effect_s_per_unit", 0.0)),
        uncertainty_s=float(payload.get("measurement_uncertainty_s", 0.5)),
        provenance=str(payload.get("provenance", "unknown")),
        prior_source=str(payload.get("prior_source", "unknown")),
        comparison_count=int(coverage.get("comparisons", 0)),
        driver_count=int(coverage.get("drivers", 0)),
        team_count=int(coverage.get("teams", 0)),
        empirical_weight=float(payload.get("empirical_weight", 0.0)),
        coefficient_provenance={
            str(key): str(value) for key, value in coefficient_provenance.items()
        },
        fallback_reasons=tuple(str(value) for value in payload.get("fallback_reasons", [])),
    )


def build_qualifying_practice_feature_rows(
    evidence_by_session: Mapping[str, Mapping[str, Any]],
    *,
    all_drivers: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flatten chronological sidecars into one full-grid model feature frame."""

    ordered_sessions = sorted(
        (
            (str(session).strip().upper(), payload)
            for session, payload in evidence_by_session.items()
            if isinstance(payload, Mapping)
        ),
        key=lambda item: (_SESSION_ORDER.get(item[0], 99), item[0]),
    )
    rows: list[dict[str, Any]] = []
    eligible_drivers = 0
    uncertainty_by_driver: dict[str, float] = {}
    run_feature_rows_by_driver: dict[str, list[dict[str, Any]]] = {}
    all_driver_lookup = {str(row.get("driver")): row for row in all_drivers}

    session_driver_payloads: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for session_code, payload in ordered_sessions:
        if not bool((payload.get("eligibility") or {}).get("eligible", False)):
            continue
        drivers = payload.get("drivers")
        if not isinstance(drivers, Mapping):
            continue
        for driver, driver_payload in drivers.items():
            if isinstance(driver_payload, Mapping):
                session_driver_payloads.setdefault(str(driver), []).append(
                    (session_code, driver_payload)
                )

    latest_best_by_driver: dict[str, float] = {}
    for driver, session_payloads in session_driver_payloads.items():
        latest_features = session_payloads[-1][1].get("features") or {}
        latest_best = _optional_float(latest_features.get("best_adjusted_lap_s"))
        if latest_best is not None:
            latest_best_by_driver[driver] = latest_best

    teammate_gap_by_driver: dict[str, float] = {}
    team_to_drivers: dict[str, list[str]] = {}
    for driver, info in all_driver_lookup.items():
        team_to_drivers.setdefault(str(info.get("team", "")), []).append(driver)
    for drivers in team_to_drivers.values():
        for driver in drivers:
            driver_best = latest_best_by_driver.get(driver)
            teammate_values = [
                latest_best_by_driver[teammate]
                for teammate in drivers
                if teammate != driver and teammate in latest_best_by_driver
            ]
            if driver_best is not None and teammate_values:
                teammate_gap_by_driver[driver] = driver_best - float(np.median(teammate_values))

    for driver_info in all_drivers:
        driver = str(driver_info["driver"])
        payloads = session_driver_payloads.get(driver, [])
        latest_payload = payloads[-1][1] if payloads else {}
        features = latest_payload.get("features") if isinstance(latest_payload, Mapping) else {}
        features = features if isinstance(features, Mapping) else {}
        counts = latest_payload.get("counts") if isinstance(latest_payload, Mapping) else {}
        counts = counts if isinstance(counts, Mapping) else {}
        run_counts = counts.get("runs") if isinstance(counts, Mapping) else {}
        run_counts = run_counts if isinstance(run_counts, Mapping) else {}
        compounds = counts.get("compounds") if isinstance(counts, Mapping) else []
        compounds = compounds if isinstance(compounds, list) else []

        first_best = None
        latest_best = _optional_float(features.get("best_adjusted_lap_s"))
        if payloads:
            first_features = payloads[0][1].get("features") or {}
            if isinstance(first_features, Mapping):
                first_best = _optional_float(first_features.get("best_adjusted_lap_s"))
        session_improvement = (
            first_best - latest_best
            if first_best is not None and latest_best is not None and len(payloads) > 1
            else 0.0
        )
        measurement_se = _optional_float(features.get("measurement_uncertainty_s"))
        normalization_se = _latest_normalization_uncertainty(
            evidence_by_session,
            payloads[-1][0] if payloads else None,
        )
        effective_laps = _optional_float(features.get("effective_lap_count")) or 0.0
        evidence_quality = (
            (effective_laps / (effective_laps + 4.0))
            * (1.0 / (1.0 + max(0.0, measurement_se or normalization_se or 0.0)))
            if payloads
            else 0.0
        )
        if latest_best is not None:
            eligible_drivers += 1
        uncertainty_by_driver[driver] = max(
            0.0,
            measurement_se if measurement_se is not None else normalization_se,
        )

        feature_row = {
            "driver": driver,
            "team": str(driver_info.get("team", "")),
            "prior_utility": _prior_utility(driver_info),
            "best_adjusted_lap_s": latest_best,
            "best2_mean_s": _optional_float(features.get("best_two_mean_adjusted_lap_s")),
            "adjusted_q20_s": _optional_float(features.get("q20_adjusted_lap_s")),
            "theoretical_lap_s": _optional_float(features.get("theoretical_adjusted_lap_s")),
            "execution_loss_s": _optional_float(features.get("execution_loss_s")),
            "consistency_mad_s": _optional_float(features.get("mad_s")),
            "session_improvement_s": session_improvement,
            "teammate_gap_s": teammate_gap_by_driver.get(driver),
            "compound_adjustment_se_s": normalization_se,
            "measurement_se_s": measurement_se,
            "clean_lap_count": int(counts.get("clean_laps", 0) or 0),
            "quali_run_count": int(run_counts.get("quali_sim", 0) or 0),
            "evidence_session_count": len(payloads),
            "direct_soft_flag": float("SOFT" in {str(value).upper() for value in compounds}),
            "evidence_quality_score": evidence_quality,
        }
        rows.append(feature_row)

        run_rows: list[dict[str, Any]] = []
        for session_code, session_payload in payloads:
            raw_candidates = session_payload.get("run_feature_candidates")
            if not isinstance(raw_candidates, list):
                continue
            session_normalization_se = _latest_normalization_uncertainty(
                evidence_by_session,
                session_code,
            )
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate_best = _optional_float(raw_candidate.get("best_adjusted_lap_s"))
                if candidate_best is None:
                    continue
                candidate_effective_laps = (
                    _optional_float(raw_candidate.get("effective_lap_count")) or 0.0
                )
                candidate_measurement_se = _optional_float(
                    raw_candidate.get("measurement_uncertainty_s")
                )
                candidate_uncertainty = max(
                    0.0,
                    candidate_measurement_se
                    if candidate_measurement_se is not None
                    else session_normalization_se,
                )
                candidate_quality = (
                    (candidate_effective_laps / (candidate_effective_laps + 4.0))
                    * (1.0 / (1.0 + candidate_uncertainty))
                    if candidate_effective_laps > 0.0
                    else 0.0
                )
                candidate_row = dict(feature_row)
                candidate_row.update(
                    {
                        "session_code": session_code,
                        "run_id": str(raw_candidate.get("run_id", "")),
                        "best_adjusted_lap_s": candidate_best,
                        "best2_mean_s": _optional_float(
                            raw_candidate.get("best_two_mean_adjusted_lap_s")
                        ),
                        "adjusted_q20_s": _optional_float(raw_candidate.get("q20_adjusted_lap_s")),
                        "theoretical_lap_s": _optional_float(
                            raw_candidate.get("theoretical_adjusted_lap_s")
                        ),
                        "execution_loss_s": _optional_float(raw_candidate.get("execution_loss_s")),
                        "consistency_mad_s": _optional_float(raw_candidate.get("mad_s")),
                        "session_improvement_s": (
                            first_best - candidate_best if first_best is not None else 0.0
                        ),
                        "compound_adjustment_se_s": session_normalization_se,
                        "measurement_se_s": candidate_measurement_se,
                        "clean_lap_count": int(raw_candidate.get("clean_lap_count", 0) or 0),
                        "quali_run_count": int(
                            str(raw_candidate.get("run_class", "")) == "quali_sim"
                        ),
                        "direct_soft_flag": float(
                            str(raw_candidate.get("compound", "")).upper() == "SOFT"
                        ),
                        "evidence_quality_score": candidate_quality,
                    }
                )
                run_rows.append(candidate_row)
        if run_rows:
            run_feature_rows_by_driver[driver] = run_rows

    field_teams = {
        str(info.get("team", "")).strip()
        for info in all_drivers
        if str(info.get("team", "")).strip()
    }
    eligible_teams = {
        str(all_driver_lookup[driver].get("team", "")).strip()
        for driver in latest_best_by_driver
        if driver in all_driver_lookup and str(all_driver_lookup[driver].get("team", "")).strip()
    }
    field_size = len(all_drivers)
    required_drivers = max(2, int(np.ceil(field_size * MIN_Q1_DRIVER_COVERAGE)))
    required_teams = max(1, int(np.ceil(len(field_teams) * MIN_Q1_TEAM_COVERAGE)))
    coverage_eligible = (
        eligible_drivers >= required_drivers and len(eligible_teams) >= required_teams
    )

    return pd.DataFrame(rows), {
        "eligible": coverage_eligible,
        "eligible_drivers": eligible_drivers,
        "required_eligible_drivers": required_drivers,
        "driver_coverage": eligible_drivers / field_size if field_size else 0.0,
        "eligible_teams": len(eligible_teams),
        "field_teams": len(field_teams),
        "required_eligible_teams": required_teams,
        "team_coverage": len(eligible_teams) / len(field_teams) if field_teams else 0.0,
        "field_size": field_size,
        "sessions_used": [session for session, _ in ordered_sessions],
        "utility_input_uncertainty_s_by_driver": uncertainty_by_driver,
        # Raw run feature rows stay internal to Q1/research callers.  Runtime
        # diagnostics expose counts only, never the values themselves.
        "run_feature_rows_by_driver": run_feature_rows_by_driver,
        "fallback_reason": (None if coverage_eligible else "insufficient_grid_evidence_coverage"),
    }


def predict_q1_position_records(
    *,
    model: FittedQualifyingPracticeModel,
    feature_rows: pd.DataFrame,
    n_simulations: int,
    rng: np.random.Generator,
    evidence_summary: Mapping[str, Any],
    uncertainty_scale: float = 1.0,
) -> tuple[dict[str, list[int]], list[list[str]], dict[str, Any]]:
    """Run Q1 with evidence uncertainty translated into latent utility units."""

    if not bool(evidence_summary.get("eligible", False)):
        raise ValueError(
            str(evidence_summary.get("fallback_reason") or "Q1 evidence is ineligible")
        )
    utilities = model.utilities(feature_rows)
    raw_uncertainty = evidence_summary.get("utility_input_uncertainty_s_by_driver")
    seconds_by_driver = raw_uncertainty if isinstance(raw_uncertainty, Mapping) else {}
    seconds_to_utility = _seconds_to_utility_scale(model)
    utility_sigma = {
        str(driver): max(0.0, float(value)) * seconds_to_utility * max(0.0, uncertainty_scale)
        for driver, value in seconds_by_driver.items()
        if _optional_float(value) is not None
    }
    raw_run_rows = evidence_summary.get("run_feature_rows_by_driver")
    run_rows_by_driver = raw_run_rows if isinstance(raw_run_rows, Mapping) else {}
    run_utility_candidates: dict[str, np.ndarray] = {}
    for raw_driver, raw_rows in run_rows_by_driver.items():
        driver = str(raw_driver)
        if driver not in utilities or not isinstance(raw_rows, list) or not raw_rows:
            continue
        candidate_rows = [row for row in raw_rows if isinstance(row, Mapping)]
        if not candidate_rows:
            continue
        run_utility_candidates[driver] = _candidate_utilities_against_event_field(
            model=model,
            central_rows=feature_rows,
            driver=driver,
            candidate_rows=candidate_rows,
        )
    records, scenarios = simulate_plackett_luce_grids(
        utilities=utilities,
        n_simulations=n_simulations,
        rng=rng,
        temperature=model.temperature,
        utility_sigma_by_driver=utility_sigma,
        utility_candidates_by_driver=run_utility_candidates,
    )
    run_candidate_count = sum(len(values) for values in run_utility_candidates.values())
    return (
        records,
        scenarios,
        {
            "model_checkpoint": model.checkpoint,
            "training_events": model.training_events,
            "temperature": model.temperature,
            "eligible_drivers": int(evidence_summary.get("eligible_drivers", 0)),
            "driver_coverage": float(evidence_summary.get("driver_coverage", 0.0)),
            "eligible_teams": int(evidence_summary.get("eligible_teams", 0)),
            "team_coverage": float(evidence_summary.get("team_coverage", 0.0)),
            "sessions_used": list(evidence_summary.get("sessions_used", [])),
            "mean_utility_sigma": (
                float(np.mean(list(utility_sigma.values()))) if utility_sigma else 0.0
            ),
            "run_bootstrap_driver_count": len(run_utility_candidates),
            "run_bootstrap_candidate_count": int(run_candidate_count),
            "run_bootstrap_mode": (
                "compatible_run_utility" if run_utility_candidates else "central_utility_fallback"
            ),
        },
    )


def _prior_utility(driver_info: Mapping[str, Any]) -> float:
    team_strength = _optional_float(driver_info.get("team_strength")) or 0.5
    quali_pace = _optional_float(driver_info.get("quali_pace")) or 0.5
    skill = _optional_float(driver_info.get("skill")) or 0.5
    driver_signal = (0.70 * quali_pace) + (0.30 * skill)
    return float((0.60 * team_strength) + (0.40 * driver_signal))


def _candidate_utilities_against_event_field(
    *,
    model: FittedQualifyingPracticeModel,
    central_rows: pd.DataFrame,
    driver: str,
    candidate_rows: list[Mapping[str, Any]],
) -> np.ndarray:
    """Score one driver's run candidates against the same full-grid reference.

    Event-relative pace features must not be centred inside a one-driver candidate
    frame: doing so would erase its absolute gap to the rest of this event.  Each
    candidate is substituted into the central full-grid frame before utilities are
    calculated, so missing-driver neutrality and circuit centring remain coherent.
    """

    base = central_rows.reset_index(drop=True).copy()
    matches = base.index[base["driver"].astype(str).eq(driver)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Q1 central feature rows require exactly one row for {driver}")
    driver_index = matches[0]
    central_utilities = model.utilities(base)
    utilities: list[float] = []
    for candidate in candidate_rows:
        scenario = base.copy()
        for feature in model.feature_columns:
            if feature in candidate:
                scenario.at[driver_index, feature] = candidate[feature]
        scenario_utilities = model.utilities(scenario)
        # Recentring after candidate substitution changes every driver's utility
        # by the same arbitrary event-level offset.  Align on the unchanged peers
        # before comparing the candidate with the central-grid utilities.
        peer_offsets = [
            scenario_utilities[peer] - central_utilities[peer]
            for peer in central_utilities
            if peer != driver
        ]
        coordinate_offset = float(np.median(peer_offsets)) if peer_offsets else 0.0
        utilities.append(float(scenario_utilities[driver] - coordinate_offset))
    return np.asarray(utilities, dtype=float)


def _latest_normalization_uncertainty(
    evidence_by_session: Mapping[str, Mapping[str, Any]],
    session_code: str | None,
) -> float:
    if session_code is None:
        return 0.0
    payload = next(
        (
            value
            for key, value in evidence_by_session.items()
            if str(key).strip().upper() == session_code
        ),
        {},
    )
    return _normalization_uncertainty(payload)


def _normalization_uncertainty(payload: Any) -> float:
    normalization = payload.get("normalization") if isinstance(payload, Mapping) else {}
    if not isinstance(normalization, Mapping):
        return 0.0
    return max(0.0, _optional_float(normalization.get("measurement_uncertainty_s")) or 0.0)


def _seconds_to_utility_scale(model: FittedQualifyingPracticeModel) -> float:
    impacts: list[float] = []
    for feature, coefficient, scale in zip(
        model.feature_columns,
        model.coefficients,
        model.feature_scales,
        strict=True,
    ):
        if feature in _PACE_FEATURES:
            impacts.append(abs(float(coefficient)) / max(1e-9, float(scale)))
    return float(np.sqrt(np.sum(np.square(impacts)))) if impacts else 0.0


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
