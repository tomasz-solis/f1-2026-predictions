"""Held-out tests for 2026 team-strength refit candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
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

TEAM_STRENGTH_REFIT_TEST_ARTIFACT_TYPE = "model_diagnostics"
TEAM_STRENGTH_REFIT_TEST_KEY_TEMPLATE = "{year}::team_strength_refit_candidate_test"
TEAM_STRENGTH_REFIT_TEST_SCHEMA_VERSION = 1

CURRENT_CANDIDATE = "current_frozen_mapping"
UNCERTAINTY_ONLY_CANDIDATE = "uncertainty_only_current_medians"
SCALE_ONLY_CANDIDATE = "loo_2026_scale_only"
LINEAR_REFIT_CANDIDATE = "loo_2026_linear_refit"


@dataclass(frozen=True)
class CandidateMapping:
    """One mapping candidate used to score a held-out race."""

    name: str
    intercept_s: float
    slope_s_per_unit: float
    fit_rows: int
    fit_races: int
    scale_multiplier: float | None = None

    def predict(self, team_strength: pd.Series | np.ndarray | float) -> np.ndarray:
        """Convert team-strength values into seconds relative to the field."""
        values = np.asarray(team_strength, dtype=float)
        return self.intercept_s + (self.slope_s_per_unit * (values - 0.5))


def team_strength_refit_test_artifact_key(year: int) -> str:
    """Return the stable artifact key for the refit-candidate test."""
    return TEAM_STRENGTH_REFIT_TEST_KEY_TEMPLATE.format(year=int(year))


def build_team_strength_refit_candidate_test(
    *,
    year: int,
    mapping_artifact_path: str | Path = "data/processed/team_strength_seconds_mapping/latest.json",
    prior_artifact_path: str | Path = "data/processed/teammate_network_prior/latest.json",
    observations_path: str | Path | None = None,
    raw_matched_laps_path: str | Path | None = (
        "data/diagnostics/2026_team_strength_matched_laps/raw_matched_laps.csv"
    ),
) -> dict[str, Any]:
    """Compare frozen and refit mappings with leave-one-race-out 2026 evidence."""
    season_year = int(year)
    mapping_path = Path(mapping_artifact_path)
    prior_path = Path(prior_artifact_path)
    mapping_payload = _read_json(mapping_path)
    mapping_policy = str(mapping_payload.get("policy", "same_session_construct"))
    policy_column = _policy_column(mapping_policy)
    frozen_mappings = load_live_team_strength_mappings(mapping_path)
    observations = _load_observations(
        observations_path=Path(observations_path) if observations_path is not None else None,
        raw_matched_laps_path=(
            Path(raw_matched_laps_path) if raw_matched_laps_path is not None else None
        ),
        prior_artifact_path=prior_path,
    )

    if observations is None or observations.empty:
        return _base_artifact(
            year=season_year,
            mapping_policy=mapping_policy,
            status="not_available",
            reason="No construct-aligned observations were available for the test.",
        )

    observations = _prepare_observations(
        observations, year=season_year, policy_column=policy_column
    )
    if observations.empty:
        return _base_artifact(
            year=season_year,
            mapping_policy=mapping_policy,
            status="not_available",
            reason="No observations had the required policy, target, and driver-prior fields.",
        )

    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []
    for session_kind in ("race", "qualifying"):
        frozen_mapping = frozen_mappings.get(session_kind)
        if frozen_mapping is None:
            continue
        kind_rows = observations[observations["session_kind"].eq(session_kind)].copy()
        if kind_rows.empty:
            continue
        for holdout_race in sorted(kind_rows["race_name"].dropna().unique()):
            train = kind_rows[~kind_rows["race_name"].eq(holdout_race)].copy()
            test = kind_rows[kind_rows["race_name"].eq(holdout_race)].copy()
            if train.empty or test.empty:
                continue
            candidates = _candidate_mappings(
                train=train,
                frozen_mapping=frozen_mapping,
                policy_column=policy_column,
            )
            for candidate in candidates:
                metrics, predictions = _score_candidate(
                    candidate=candidate,
                    test=test,
                    policy_column=policy_column,
                    holdout_race=str(holdout_race),
                )
                fold_rows.append(metrics)
                candidate_rows.append(predictions)

    fold_frame = pd.DataFrame(fold_rows)
    if fold_frame.empty:
        return _base_artifact(
            year=season_year,
            mapping_policy=mapping_policy,
            status="not_available",
            reason="No leave-one-race folds could be evaluated.",
        )

    aggregate = _aggregate_candidate_metrics(fold_frame)
    by_session = {
        session_kind: _aggregate_candidate_metrics(
            fold_frame[fold_frame["session_kind"].eq(session_kind)]
        )
        for session_kind in ("race", "qualifying")
    }
    predictions_frame = (
        pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    )
    artifact = _base_artifact(
        year=season_year,
        mapping_policy=mapping_policy,
        status="measured",
    )
    artifact.update(
        {
            "tested_candidates": _candidate_descriptions(),
            "fold_metrics": _frame_records(_round_frame(fold_frame)),
            "aggregate": aggregate,
            "by_session_kind": by_session,
            "winner_summary": _winner_summary(fold_frame),
            "largest_candidate_deltas": _largest_candidate_deltas(predictions_frame),
            "decision_assessment": _decision_assessment(aggregate, by_session),
        }
    )
    return _json_safe(artifact)


def format_team_strength_refit_candidate_test_markdown(artifact: Mapping[str, Any]) -> str:
    """Format the refit-candidate test artifact as Markdown."""
    lines = [
        "# Team-Strength Refit Candidate Test",
        "",
        f"- Built at: `{artifact.get('built_at')}`",
        f"- Model version: `{artifact.get('model_version')}`",
        f"- Status: `{artifact.get('status')}`",
        f"- Policy: `{artifact.get('policy')}`",
        "",
    ]
    if artifact.get("reason"):
        lines.extend([f"Reason: {artifact.get('reason')}", ""])
        return "\n".join(lines)

    lines.extend(["## Aggregate Held-Out Metrics", ""])
    _append_aggregate_table(lines, artifact.get("aggregate", []))
    lines.append("")

    lines.extend(["## By Construct", ""])
    by_session = artifact.get("by_session_kind", {})
    if isinstance(by_session, Mapping):
        for session_kind in ("race", "qualifying"):
            lines.extend([f"### {session_kind.title()}", ""])
            _append_aggregate_table(lines, by_session.get(session_kind, []))
            lines.append("")

    lines.extend(["## Winner Summary", ""])
    winner_summary = artifact.get("winner_summary", {})
    if isinstance(winner_summary, Mapping):
        for session_kind, rows in winner_summary.items():
            lines.append(f"- `{session_kind}`:")
            if isinstance(rows, list):
                for row in rows:
                    lines.append(
                        f"  - {row.get('candidate')}: {row.get('wins')} wins, "
                        f"{row.get('losses')} losses, {row.get('ties')} ties versus current"
                    )
    lines.append("")

    decision = artifact.get("decision_assessment", {})
    if isinstance(decision, Mapping):
        lines.extend(
            [
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


def _candidate_mappings(
    *,
    train: pd.DataFrame,
    frozen_mapping: LinearTeamStrengthMapping,
    policy_column: str,
) -> list[CandidateMapping]:
    """Fit all candidate mappings that should be tested on one held-out race."""
    candidates = [
        CandidateMapping(
            name=CURRENT_CANDIDATE,
            intercept_s=frozen_mapping.intercept_s,
            slope_s_per_unit=frozen_mapping.slope_s_per_unit,
            fit_rows=0,
            fit_races=0,
        ),
        CandidateMapping(
            name=UNCERTAINTY_ONLY_CANDIDATE,
            intercept_s=frozen_mapping.intercept_s,
            slope_s_per_unit=frozen_mapping.slope_s_per_unit,
            fit_rows=0,
            fit_races=0,
        ),
    ]
    scale_mapping = _fit_scale_only_candidate(
        train=train,
        frozen_mapping=frozen_mapping,
        policy_column=policy_column,
    )
    if scale_mapping is not None:
        candidates.append(scale_mapping)
    linear_mapping = _fit_linear_refit_candidate(train=train, policy_column=policy_column)
    if linear_mapping is not None:
        candidates.append(linear_mapping)
    return candidates


def _fit_scale_only_candidate(
    *,
    train: pd.DataFrame,
    frozen_mapping: LinearTeamStrengthMapping,
    policy_column: str,
) -> CandidateMapping | None:
    """Fit one scale multiplier on the frozen centered slope while keeping intercept fixed."""
    centered_strength = train[policy_column].astype(float).to_numpy() - 0.5
    frozen_centered_delta = frozen_mapping.slope_s_per_unit * centered_strength
    target = train["team_target_s"].astype(float).to_numpy() - frozen_mapping.intercept_s
    denominator = float(np.dot(frozen_centered_delta, frozen_centered_delta))
    if np.isclose(denominator, 0.0):
        return None
    scale = float(np.dot(frozen_centered_delta, target) / denominator)
    if not np.isfinite(scale):
        return None
    return CandidateMapping(
        name=SCALE_ONLY_CANDIDATE,
        intercept_s=frozen_mapping.intercept_s,
        slope_s_per_unit=float(frozen_mapping.slope_s_per_unit * scale),
        fit_rows=int(len(train)),
        fit_races=int(train["race_name"].nunique()),
        scale_multiplier=scale,
    )


def _fit_linear_refit_candidate(
    *,
    train: pd.DataFrame,
    policy_column: str,
) -> CandidateMapping | None:
    """Fit a two-parameter intercept and slope mapping from training races."""
    centered_strength = train[policy_column].astype(float).to_numpy() - 0.5
    target = train["team_target_s"].astype(float).to_numpy()
    if len(target) < 2 or np.isclose(float(np.nanstd(centered_strength)), 0.0):
        return None
    design = np.column_stack([np.ones(len(train), dtype=float), centered_strength])
    intercept_s, slope_s_per_unit = np.linalg.lstsq(design, target, rcond=None)[0]
    if not np.isfinite(intercept_s) or not np.isfinite(slope_s_per_unit):
        return None
    return CandidateMapping(
        name=LINEAR_REFIT_CANDIDATE,
        intercept_s=float(intercept_s),
        slope_s_per_unit=float(slope_s_per_unit),
        fit_rows=int(len(train)),
        fit_races=int(train["race_name"].nunique()),
    )


def _score_candidate(
    *,
    candidate: CandidateMapping,
    test: pd.DataFrame,
    policy_column: str,
    holdout_race: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score one candidate on one held-out race."""
    predicted_team_s = candidate.predict(test[policy_column])
    predicted_driver_s = predicted_team_s + test["driver_rating_mu_s"].astype(float).to_numpy()
    observed = test["observed_driver_to_field_s"].astype(float).to_numpy()
    residual = observed - predicted_driver_s
    sse = float(np.sum(np.square(residual)))
    mae = float(np.mean(np.abs(residual))) if len(residual) else None
    metrics = {
        "candidate": candidate.name,
        "session_kind": str(test["session_kind"].iloc[0]),
        "holdout_race": holdout_race,
        "n_rows": int(len(test)),
        "n_teams": int(test["team"].nunique()),
        "n_drivers": int(test["driver_code"].nunique()),
        "fit_rows": candidate.fit_rows,
        "fit_races": candidate.fit_races,
        "intercept_s": candidate.intercept_s,
        "slope_s_per_unit": candidate.slope_s_per_unit,
        "scale_multiplier": candidate.scale_multiplier,
        "sse_s2": sse,
        "mse_s2": float(sse / len(test)) if len(test) else None,
        "rmse_s": float(np.sqrt(sse / len(test))) if len(test) else None,
        "mae_s": mae,
        "residual_mean_s": float(np.mean(residual)) if len(residual) else None,
    }
    predictions = test[
        [
            "year",
            "race_name",
            "session_name",
            "session_kind",
            "team",
            "driver_code",
            policy_column,
            "observed_driver_to_field_s",
            "driver_rating_mu_s",
            "team_target_s",
        ]
    ].copy()
    predictions = predictions.assign(
        candidate=candidate.name,
        holdout_race=holdout_race,
        predicted_team_s=predicted_team_s,
        predicted_driver_to_field_s=predicted_driver_s,
        residual_s=residual,
        abs_residual_s=np.abs(residual),
    )
    return metrics, predictions


def _aggregate_candidate_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate fold metrics by candidate using row-weighted squared error."""
    if frame.empty:
        return []
    current = _aggregate_one_candidate(frame[frame["candidate"].eq(CURRENT_CANDIDATE)])
    current_mse = current.get("weighted_mse_s2") if current else None
    rows: list[dict[str, Any]] = []
    for candidate, group in frame.groupby("candidate", dropna=False):
        row = _aggregate_one_candidate(group)
        row["candidate"] = str(candidate)
        row["mse_delta_vs_current_s2"] = _delta(row.get("weighted_mse_s2"), current_mse)
        row["mse_pct_delta_vs_current"] = _pct_delta(row.get("weighted_mse_s2"), current_mse)
        rows.append(row)
    return sorted(rows, key=lambda row: float(row.get("weighted_mse_s2") or np.inf))


def _aggregate_one_candidate(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize one candidate across available folds."""
    if frame.empty:
        return {}
    n_rows = int(frame["n_rows"].sum())
    sse = float(frame["sse_s2"].sum())
    weighted_mse = float(sse / n_rows) if n_rows else None
    return {
        "candidate": str(frame["candidate"].iloc[0]),
        "n_folds": int(len(frame)),
        "n_rows": n_rows,
        "weighted_mse_s2": weighted_mse,
        "weighted_rmse_s": float(np.sqrt(weighted_mse)) if weighted_mse is not None else None,
        "mean_fold_mse_s2": float(frame["mse_s2"].mean()),
        "mean_fold_rmse_s": float(frame["rmse_s"].mean()),
    }


def _winner_summary(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Count candidate wins, losses, and ties against the current mapping."""
    summary: dict[str, list[dict[str, Any]]] = {}
    if frame.empty:
        return summary
    for session_kind in ("combined", "race", "qualifying"):
        rows = (
            frame.copy()
            if session_kind == "combined"
            else frame[frame["session_kind"].eq(session_kind)]
        )
        if rows.empty:
            summary[session_kind] = []
            continue
        current_by_fold = {
            (str(row["session_kind"]), str(row["holdout_race"])): float(row["mse_s2"])
            for _, row in rows[rows["candidate"].eq(CURRENT_CANDIDATE)].iterrows()
        }
        candidate_rows: list[dict[str, Any]] = []
        for candidate, group in rows.groupby("candidate", dropna=False):
            if candidate == CURRENT_CANDIDATE:
                continue
            wins = losses = ties = 0
            for _, row in group.iterrows():
                baseline = current_by_fold.get((str(row["session_kind"]), str(row["holdout_race"])))
                if baseline is None:
                    continue
                candidate_mse = float(row["mse_s2"])
                if np.isclose(candidate_mse, baseline):
                    ties += 1
                elif candidate_mse < baseline:
                    wins += 1
                else:
                    losses += 1
            candidate_rows.append(
                {
                    "candidate": str(candidate),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                }
            )
        summary[session_kind] = sorted(
            candidate_rows,
            key=lambda row: (-(int(row["wins"])), int(row["losses"]), str(row["candidate"])),
        )
    return summary


def _largest_candidate_deltas(
    predictions: pd.DataFrame, *, limit: int = 20
) -> list[dict[str, Any]]:
    """Return rows where candidate residuals differ most from the current mapping."""
    if predictions.empty:
        return []
    key_columns = [
        "session_kind",
        "holdout_race",
        "team",
        "driver_code",
        "observed_driver_to_field_s",
    ]
    current = predictions[predictions["candidate"].eq(CURRENT_CANDIDATE)][
        [*key_columns, "abs_residual_s"]
    ].rename(columns={"abs_residual_s": "current_abs_residual_s"})
    rows = predictions[~predictions["candidate"].eq(CURRENT_CANDIDATE)].merge(
        current,
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )
    if rows.empty:
        return []
    rows["abs_residual_delta_vs_current_s"] = (
        rows["abs_residual_s"] - rows["current_abs_residual_s"]
    )
    rows = rows.reindex(
        rows["abs_residual_delta_vs_current_s"].abs().sort_values(ascending=False).index
    )
    columns = [
        "candidate",
        "session_kind",
        "holdout_race",
        "team",
        "driver_code",
        "observed_driver_to_field_s",
        "predicted_driver_to_field_s",
        "residual_s",
        "current_abs_residual_s",
        "abs_residual_s",
        "abs_residual_delta_vs_current_s",
    ]
    return _frame_records(_round_frame(rows[columns].head(limit)))


def _decision_assessment(
    aggregate: list[dict[str, Any]],
    by_session: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assess whether the candidate evidence is strong enough to ship a refit."""
    candidate_rows = {str(row["candidate"]): row for row in aggregate if isinstance(row, Mapping)}
    current = candidate_rows.get(CURRENT_CANDIDATE, {})
    best = aggregate[0] if aggregate else {}
    reasons: list[str] = []
    state = "not_enough_evidence"
    recommendation = (
        "Do not ship a median-changing refit from this diagnostic alone; use it to decide "
        "whether a full prediction replay test is worth running."
    )
    if not current or not best:
        reasons.append("No aggregate comparison was available.")
    else:
        best_name = str(best.get("candidate"))
        pct_delta = _coerce_float(best.get("mse_pct_delta_vs_current"))
        if best_name == CURRENT_CANDIDATE:
            reasons.append("The frozen mapping has the best row-weighted held-out MSE.")
            recommendation = "Keep the current mapping; no refit is supported by construct MSE."
        elif pct_delta is not None and pct_delta < -5.0:
            state = "refit_candidate_worth_full_prediction_replay"
            reasons.append(
                f"{best_name} improves row-weighted held-out MSE by {abs(pct_delta):.1f}% "
                "versus the frozen mapping."
            )
            recommendation = (
                "Run a full prediction replay before shipping. Construct MSE supports more testing, "
                "not an automatic production refit."
            )
        else:
            reasons.append(
                f"The best candidate is {best_name}, but the aggregate MSE gain is below 5%."
            )
    for session_kind in ("race", "qualifying"):
        session_rows = {
            str(row["candidate"]): row
            for row in by_session.get(session_kind, [])
            if isinstance(row, Mapping)
        }
        session_best = by_session.get(session_kind, [{}])[0] if by_session.get(session_kind) else {}
        if session_best:
            reasons.append(
                f"{session_kind} best row-weighted candidate: {session_best.get('candidate')} "
                f"({_fmt_pct(session_best.get('mse_pct_delta_vs_current'))} vs current)."
            )
        if session_rows.get(UNCERTAINTY_ONLY_CANDIDATE, {}).get("mse_delta_vs_current_s2") == 0:
            reasons.append(
                f"{session_kind} uncertainty-only keeps identical medians, so MSE is unchanged."
            )
    return {
        "state": state,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def _candidate_descriptions() -> list[dict[str, str]]:
    """Return plain-language descriptions of each tested candidate."""
    return [
        {
            "candidate": CURRENT_CANDIDATE,
            "description": "Frozen 2022-2025 team-strength seconds mapping currently used live.",
        },
        {
            "candidate": UNCERTAINTY_ONLY_CANDIDATE,
            "description": "Same point predictions as current; represents interval widening only.",
        },
        {
            "candidate": SCALE_ONLY_CANDIDATE,
            "description": "Leave-one-race fit of one slope multiplier on the frozen mapping.",
        },
        {
            "candidate": LINEAR_REFIT_CANDIDATE,
            "description": "Leave-one-race fit of intercept and slope from 2026 construct rows.",
        },
    ]


def _prepare_observations(
    observations: pd.DataFrame,
    *,
    year: int,
    policy_column: str,
) -> pd.DataFrame:
    """Filter observations to rows usable for refit-candidate testing."""
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
    rows = observations[observations["year"].eq(int(year))].copy()
    rows = rows.dropna(
        subset=[
            "observed_driver_to_field_s",
            "driver_rating_mu_s",
            policy_column,
        ]
    )
    rows["team_target_s"] = rows["observed_driver_to_field_s"].astype(float) - rows[
        "driver_rating_mu_s"
    ].astype(float)
    return rows.reset_index(drop=True)


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


def _base_artifact(
    *,
    year: int,
    mapping_policy: str,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build common artifact fields for available and unavailable tests."""
    return {
        "artifact_type": "team_strength_refit_candidate_test",
        "schema_version": TEAM_STRENGTH_REFIT_TEST_SCHEMA_VERSION,
        "model_version": get_model_version(),
        "built_at": datetime.now(UTC).isoformat(),
        "year": int(year),
        "status": status,
        "reason": reason,
        "policy": mapping_policy,
    }


def _append_aggregate_table(lines: list[str], rows: Any) -> None:
    """Append an aggregate metrics table to Markdown lines."""
    if not isinstance(rows, list) or not rows:
        lines.append("No aggregate metrics available.")
        return
    lines.extend(
        [
            "| Candidate | Folds | Rows | Weighted MSE | Weighted RMSE | MSE delta vs current |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.get('candidate')}` | {row.get('n_folds')} | {row.get('n_rows')} | "
            f"{_fmt(row.get('weighted_mse_s2'))} | {_fmt(row.get('weighted_rmse_s'))} | "
            f"{_fmt_pct(row.get('mse_pct_delta_vs_current'))} |"
        )


def _policy_column(policy: str) -> str:
    """Resolve a mapping policy to the concrete observation column."""
    try:
        return TEAM_STRENGTH_POLICY_COLUMNS[policy]
    except KeyError as exc:
        raise ValueError(f"Unknown team-strength policy: {policy!r}") from exc


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


def _delta(value: Any, baseline: Any) -> float | None:
    """Return value minus baseline when both values are numeric."""
    numeric_value = _coerce_float(value)
    numeric_baseline = _coerce_float(baseline)
    if numeric_value is None or numeric_baseline is None:
        return None
    return float(numeric_value - numeric_baseline)


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


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = sorted(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


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
