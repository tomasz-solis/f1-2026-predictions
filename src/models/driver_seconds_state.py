"""Read and preserve seconds-native driver state inside driver artifacts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

DRIVER_SECONDS_FIELDS = (
    "race_rating_mu_s",
    "race_rating_sigma_s",
    "quali_rating_mu_s",
    "quali_rating_sigma_s",
)
DRIVER_SECONDS_OBSERVATION_FIELDS = (
    "race_rating_observations",
    "quali_rating_observations",
)
_DRIVER_SECONDS_SIGMA_FIELDS = frozenset({"race_rating_sigma_s", "quali_rating_sigma_s"})
_SECONDS_UPDATE_COLUMNS = frozenset(
    {
        "reference_driver_code",
        "comparison_driver_code",
        "session_kind",
        "matched_gap_median_s",
        "matched_gap_se_s",
        "n_matched_pairs",
        "weather_bucket",
        "skip_reason",
    }
)

DriverSecondsSessionKind = Literal["race", "qualifying"]


@dataclass(frozen=True)
class DriverSecondsState:
    """Seconds-native race and qualifying residual state for one driver."""

    race_rating_mu_s: float
    race_rating_sigma_s: float
    quali_rating_mu_s: float
    quali_rating_sigma_s: float

    def to_bayesian_fields(self) -> dict[str, float]:
        """Return fields stored alongside legacy Bayesian metadata."""
        return {
            "race_rating_mu_s": float(self.race_rating_mu_s),
            "race_rating_sigma_s": float(self.race_rating_sigma_s),
            "quali_rating_mu_s": float(self.quali_rating_mu_s),
            "quali_rating_sigma_s": float(self.quali_rating_sigma_s),
        }


@dataclass(frozen=True)
class DriverSecondsUpdateSummary:
    """Counts from one construct-aligned driver-seconds update pass."""

    session_kind: DriverSecondsSessionKind
    observations_applied: int
    drivers_touched: int
    rows_skipped_missing_state: int


def center_rating_mu_by_team(
    records: Sequence[MutableMapping[str, Any]],
    *,
    field: str,
    team_key: str = "team",
) -> None:
    """Remove each team's mean seconds rating so only the teammate-relative part remains.

    Groups ``records`` by ``record[team_key]`` and subtracts the team mean of ``field``
    from each record's value, in place. A team with fewer than two finite values for
    ``field`` is left untouched: a driver with no rated teammate has no identifiable
    within-team residual, so zeroing it would delete real signal rather than remove a
    team-level offset.
    """
    values_by_team: dict[Any, list[float]] = {}
    for record in records:
        value = _coerce_float(record.get(field))
        if value is None:
            continue
        values_by_team.setdefault(record.get(team_key), []).append(value)

    team_means = {
        team: sum(values) / len(values)
        for team, values in values_by_team.items()
        if len(values) >= 2
    }
    if not team_means:
        return

    for record in records:
        team_mean = team_means.get(record.get(team_key))
        if team_mean is None:
            continue
        value = _coerce_float(record.get(field))
        if value is None:
            continue
        record[field] = value - team_mean


def read_driver_seconds_state(driver_entry: Mapping[str, Any]) -> DriverSecondsState | None:
    """Return complete seconds-native driver state when all fields are valid."""
    bayesian = driver_entry.get("bayesian")
    if not isinstance(bayesian, Mapping):
        return None

    race_mu = _seconds_field(bayesian, "race_rating_mu_s")
    race_sigma = _seconds_field(bayesian, "race_rating_sigma_s")
    quali_mu = _seconds_field(bayesian, "quali_rating_mu_s")
    quali_sigma = _seconds_field(bayesian, "quali_rating_sigma_s")
    if race_mu is None or race_sigma is None or quali_mu is None or quali_sigma is None:
        return None
    return DriverSecondsState(
        race_rating_mu_s=race_mu,
        race_rating_sigma_s=race_sigma,
        quali_rating_mu_s=quali_mu,
        quali_rating_sigma_s=quali_sigma,
    )


def write_driver_seconds_state(
    driver_entry: MutableMapping[str, Any],
    state: DriverSecondsState,
) -> None:
    """Write seconds-native state without touching legacy rating fields."""
    existing = driver_entry.get("bayesian")
    bayesian = dict(existing) if isinstance(existing, Mapping) else {}
    bayesian.update(state.to_bayesian_fields())
    driver_entry["bayesian"] = bayesian


def read_driver_rating_mu_seconds(
    driver_entry: Mapping[str, Any],
    *,
    session_kind: DriverSecondsSessionKind,
) -> float | None:
    """Read one session-specific seconds mean for prediction use."""
    state = read_driver_seconds_state(driver_entry)
    if state is None:
        return None
    return state.race_rating_mu_s if session_kind == "race" else state.quali_rating_mu_s


def update_driver_seconds_from_teammate_aggregates(
    *,
    drivers_payload: MutableMapping[str, Any],
    aggregate_rows: pd.DataFrame | None,
    session_kind: DriverSecondsSessionKind,
    observation_sigma_floor_s: float = 0.02,
    evidence_scale: float = 1.0,
) -> DriverSecondsUpdateSummary:
    """Update one seconds-state path from canonical teammate aggregate rows.

    Each aggregate row is the constraint
    ``matched_gap_median_s = theta_reference - theta_comparison + error``.
    The pair update changes only the requested race or qualifying seconds path.
    It does not convert result positions or legacy rating units into seconds.
    ``evidence_scale`` scales aggregate precision: values below one make a
    source weaker without changing the aggregate residual itself.
    """
    if aggregate_rows is None or aggregate_rows.empty:
        return DriverSecondsUpdateSummary(
            session_kind=session_kind,
            observations_applied=0,
            drivers_touched=0,
            rows_skipped_missing_state=0,
        )

    missing_columns = sorted(_SECONDS_UPDATE_COLUMNS.difference(aggregate_rows.columns))
    if missing_columns:
        raise ValueError(f"Seconds update rows are missing columns: {missing_columns}")

    rows = _valid_update_rows(aggregate_rows, session_kind=session_kind)
    touched_drivers: set[str] = set()
    observations_applied = 0
    rows_skipped_missing_state = 0
    sigma_floor = max(float(observation_sigma_floor_s), 1e-6)
    precision_scale = _validated_evidence_scale(evidence_scale)

    for row in rows.itertuples(index=False):
        reference_code = str(row.reference_driver_code)
        comparison_code = str(row.comparison_driver_code)
        reference_entry = drivers_payload.get(reference_code)
        comparison_entry = drivers_payload.get(comparison_code)
        if not isinstance(reference_entry, MutableMapping) or not isinstance(
            comparison_entry,
            MutableMapping,
        ):
            rows_skipped_missing_state += 1
            continue

        reference_state = read_driver_seconds_state(reference_entry)
        comparison_state = read_driver_seconds_state(comparison_entry)
        if reference_state is None or comparison_state is None:
            rows_skipped_missing_state += 1
            continue

        gap_s = _coerce_float(row.matched_gap_median_s)
        gap_se_s = _coerce_float(row.matched_gap_se_s)
        if gap_s is None or gap_se_s is None or gap_se_s < 0.0:
            continue

        updated_reference, updated_comparison = _update_pair_constraint(
            reference_state=reference_state,
            comparison_state=comparison_state,
            session_kind=session_kind,
            observed_gap_s=gap_s,
            observation_sigma_s=max(gap_se_s, sigma_floor) / np.sqrt(precision_scale),
        )
        write_driver_seconds_state(reference_entry, updated_reference)
        write_driver_seconds_state(comparison_entry, updated_comparison)
        _increment_observation_count(reference_entry, session_kind=session_kind)
        _increment_observation_count(comparison_entry, session_kind=session_kind)
        touched_drivers.update({reference_code, comparison_code})
        observations_applied += 1

    return DriverSecondsUpdateSummary(
        session_kind=session_kind,
        observations_applied=observations_applied,
        drivers_touched=len(touched_drivers),
        rows_skipped_missing_state=rows_skipped_missing_state,
    )


def preserve_driver_seconds_fields(
    *,
    previous_bayesian: Any,
    updated_bayesian: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry valid seconds fields through legacy Bayesian metadata writes."""
    merged = dict(updated_bayesian)
    if not isinstance(previous_bayesian, Mapping):
        return merged

    for field in DRIVER_SECONDS_FIELDS:
        value = _seconds_field(previous_bayesian, field)
        if value is not None:
            merged[field] = value
    for field in DRIVER_SECONDS_OBSERVATION_FIELDS:
        value = _observation_count(previous_bayesian, field)
        if value is not None:
            merged[field] = value
    return merged


def _update_pair_constraint(
    *,
    reference_state: DriverSecondsState,
    comparison_state: DriverSecondsState,
    session_kind: DriverSecondsSessionKind,
    observed_gap_s: float,
    observation_sigma_s: float,
) -> tuple[DriverSecondsState, DriverSecondsState]:
    """Apply one two-driver Gaussian update to a rating-difference constraint."""
    reference_mu, reference_sigma = _path_state(reference_state, session_kind=session_kind)
    comparison_mu, comparison_sigma = _path_state(comparison_state, session_kind=session_kind)
    reference_var = max(float(reference_sigma) ** 2, 1e-12)
    comparison_var = max(float(comparison_sigma) ** 2, 1e-12)
    observation_var = max(float(observation_sigma_s) ** 2, 1e-12)
    denominator = reference_var + comparison_var + observation_var
    innovation = float(observed_gap_s) - (reference_mu - comparison_mu)

    updated_reference_mu = reference_mu + ((reference_var / denominator) * innovation)
    updated_comparison_mu = comparison_mu - ((comparison_var / denominator) * innovation)
    updated_reference_sigma = np.sqrt(
        max(reference_var - ((reference_var**2) / denominator), 1e-12)
    )
    updated_comparison_sigma = np.sqrt(
        max(comparison_var - ((comparison_var**2) / denominator), 1e-12)
    )
    return (
        _replace_path_state(
            reference_state,
            session_kind=session_kind,
            mu_s=float(updated_reference_mu),
            sigma_s=float(updated_reference_sigma),
        ),
        _replace_path_state(
            comparison_state,
            session_kind=session_kind,
            mu_s=float(updated_comparison_mu),
            sigma_s=float(updated_comparison_sigma),
        ),
    )


def _path_state(
    state: DriverSecondsState,
    *,
    session_kind: DriverSecondsSessionKind,
) -> tuple[float, float]:
    """Return one mean and sigma pair from a complete seconds state."""
    if session_kind == "race":
        return state.race_rating_mu_s, state.race_rating_sigma_s
    return state.quali_rating_mu_s, state.quali_rating_sigma_s


def _replace_path_state(
    state: DriverSecondsState,
    *,
    session_kind: DriverSecondsSessionKind,
    mu_s: float,
    sigma_s: float,
) -> DriverSecondsState:
    """Return a state with one session path replaced."""
    if session_kind == "race":
        return DriverSecondsState(
            race_rating_mu_s=mu_s,
            race_rating_sigma_s=sigma_s,
            quali_rating_mu_s=state.quali_rating_mu_s,
            quali_rating_sigma_s=state.quali_rating_sigma_s,
        )
    return DriverSecondsState(
        race_rating_mu_s=state.race_rating_mu_s,
        race_rating_sigma_s=state.race_rating_sigma_s,
        quali_rating_mu_s=mu_s,
        quali_rating_sigma_s=sigma_s,
    )


def _valid_update_rows(
    aggregate_rows: pd.DataFrame,
    *,
    session_kind: DriverSecondsSessionKind,
) -> pd.DataFrame:
    """Return dry usable rows for one requested seconds-state path."""
    rows = aggregate_rows.copy()
    counts = pd.to_numeric(rows["n_matched_pairs"], errors="coerce").fillna(0)
    rows["matched_gap_median_s"] = pd.to_numeric(rows["matched_gap_median_s"], errors="coerce")
    rows["matched_gap_se_s"] = pd.to_numeric(rows["matched_gap_se_s"], errors="coerce")
    mask = (
        rows["session_kind"].eq(session_kind)
        & rows["weather_bucket"].eq("dry")
        & rows["skip_reason"].isna()
        & counts.gt(0)
        & rows["matched_gap_median_s"].notna()
        & rows["matched_gap_se_s"].notna()
    )
    return rows[mask].copy()


def _increment_observation_count(
    driver_entry: MutableMapping[str, Any],
    *,
    session_kind: DriverSecondsSessionKind,
) -> None:
    """Increment a per-path matched aggregate evidence counter."""
    bayesian = driver_entry.get("bayesian")
    if not isinstance(bayesian, MutableMapping):
        return
    field = "race_rating_observations" if session_kind == "race" else "quali_rating_observations"
    bayesian[field] = (_observation_count(bayesian, field) or 0) + 1


def _observation_count(bayesian: Mapping[str, Any], field: str) -> int | None:
    """Read one non-negative observation count."""
    if field not in bayesian:
        return None
    raw_value = bayesian.get(field)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _seconds_field(bayesian: Mapping[str, Any], field: str) -> float | None:
    """Read one finite seconds field and reject invalid uncertainty."""
    value = _coerce_float(bayesian.get(field))
    if value is None:
        return None
    if field in _DRIVER_SECONDS_SIGMA_FIELDS and value < 0.0:
        return None
    return value


def _coerce_float(value: Any) -> float | None:
    """Convert finite numeric values and reject all other input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _validated_evidence_scale(value: Any) -> float:
    """Return one positive finite observation-precision scale."""
    scale = _coerce_float(value)
    if scale is None or scale <= 0.0:
        raise ValueError("evidence_scale must be a positive finite number")
    return float(scale)
