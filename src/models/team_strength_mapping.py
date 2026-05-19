"""Build Phase 7 team-strength calibration observations and diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MATCHED_PAIR_ROW_TYPE = "matched_pair"
CALIBRATION_SESSION_COLUMNS: tuple[str, ...] = (
    "year",
    "race_name",
    "session_name",
    "session_kind",
)
CALIBRATION_DRIVER_COLUMNS: tuple[str, ...] = (
    *CALIBRATION_SESSION_COLUMNS,
    "team",
    "driver_code",
)

TEAM_STRENGTH_POLICY_COLUMNS: dict[str, str] = {
    "same_session_construct": "team_strength_same_session",
    "race_event_shared_scalar": "team_strength_race_event",
    "race_season_mean_shared_scalar": "team_strength_race_season_mean",
    "race_trailing_mean_shared_scalar": "team_strength_race_trailing_mean",
}
DEFAULT_TEAM_STRENGTH_SECONDS_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "team_strength_seconds_mapping"
    / "latest.json"
)


@dataclass(frozen=True)
class LinearTeamStrengthMapping:
    """One fitted centered linear conversion from team-strength units to seconds."""

    session_kind: str
    policy: str
    intercept_s: float
    slope_s_per_unit: float
    training_years: tuple[int, ...]

    def predict(self, team_strength: pd.Series | np.ndarray | float) -> np.ndarray:
        """Convert team-strength values into predicted seconds relative to the field."""
        values = np.asarray(team_strength, dtype=float)
        return self.intercept_s + (self.slope_s_per_unit * (values - 0.5))

    def predict_one(self, team_strength: float) -> float:
        """Convert one team-strength value into predicted seconds."""
        return float(self.predict(float(team_strength)))

    def predict_delta(self, team_strength: pd.Series | np.ndarray | float) -> np.ndarray:
        """Convert team strength into centered seconds, excluding the shared intercept."""
        values = np.asarray(team_strength, dtype=float)
        return self.slope_s_per_unit * (values - 0.5)

    def predict_delta_one(self, team_strength: float) -> float:
        """Convert one team-strength value into centered seconds."""
        return float(self.predict_delta(float(team_strength)))


def build_construct_aligned_driver_observations(
    raw_matched_laps: pd.DataFrame,
    *,
    driver_mu_by_kind: Mapping[str, Mapping[str, float]] | None = None,
    weather_bucket: str = "dry",
) -> pd.DataFrame:
    """Build driver-to-field observations from the existing matched-lap construct.

    The function deliberately starts from already-selected matched pairs rather
    than widening the extractor. Each driver's session median therefore uses the
    same paired laps that currently feed the teammate-network prior.
    """
    required_columns = {
        "row_type",
        "year",
        "race_name",
        "session_name",
        "session_kind",
        "team",
        "reference_driver_code",
        "comparison_driver_code",
        "reference_lap_time_s",
        "comparison_lap_time_s",
        "weather_bucket",
    }
    _require_columns(raw_matched_laps, required_columns, "raw_matched_laps")

    matched = raw_matched_laps[
        raw_matched_laps["row_type"].eq(MATCHED_PAIR_ROW_TYPE)
        & raw_matched_laps["weather_bucket"].eq(weather_bucket)
    ].copy()
    if matched.empty:
        return _empty_observations()

    reference_rows = matched[
        [
            *CALIBRATION_SESSION_COLUMNS,
            "team",
            "reference_driver_code",
            "reference_lap_time_s",
        ]
    ].rename(
        columns={
            "reference_driver_code": "driver_code",
            "reference_lap_time_s": "lap_time_s",
        }
    )
    comparison_rows = matched[
        [
            *CALIBRATION_SESSION_COLUMNS,
            "team",
            "comparison_driver_code",
            "comparison_lap_time_s",
        ]
    ].rename(
        columns={
            "comparison_driver_code": "driver_code",
            "comparison_lap_time_s": "lap_time_s",
        }
    )
    driver_laps = pd.concat([reference_rows, comparison_rows], ignore_index=True)
    driver_laps["lap_time_s"] = pd.to_numeric(driver_laps["lap_time_s"], errors="coerce")
    driver_laps = driver_laps.dropna(subset=["lap_time_s"])

    observations = (
        driver_laps.groupby(list(CALIBRATION_DRIVER_COLUMNS), dropna=False, as_index=False)
        .agg(
            driver_median_s=("lap_time_s", "median"),
            n_construct_laps=("lap_time_s", "size"),
        )
        .sort_values(list(CALIBRATION_DRIVER_COLUMNS))
        .reset_index(drop=True)
    )
    field_summary = (
        observations.groupby(list(CALIBRATION_SESSION_COLUMNS), dropna=False, as_index=False)
        .agg(
            field_median_s=("driver_median_s", "median"),
            n_field_drivers=("driver_code", "nunique"),
            n_field_teams=("team", "nunique"),
        )
        .reset_index(drop=True)
    )
    observations = observations.merge(
        field_summary,
        on=list(CALIBRATION_SESSION_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    observations["observed_driver_to_field_s"] = (
        observations["field_median_s"] - observations["driver_median_s"]
    )
    observations = attach_team_strength_proxies(observations)

    if driver_mu_by_kind is not None:
        observations = attach_driver_rating_mus(
            observations,
            driver_mu_by_kind=driver_mu_by_kind,
        )
    return observations


def attach_team_strength_proxies(observations: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit candidate team-strength proxies to observation rows."""
    required_columns = {
        *CALIBRATION_DRIVER_COLUMNS,
        "driver_median_s",
    }
    _require_columns(observations, required_columns, "observations")

    enriched = observations.copy()
    team_rows = (
        enriched.groupby(
            [*CALIBRATION_SESSION_COLUMNS, "team"],
            dropna=False,
            as_index=False,
        )
        .agg(team_median_s=("driver_median_s", "median"))
        .reset_index(drop=True)
    )
    team_rows["team_rank"] = team_rows.groupby(
        list(CALIBRATION_SESSION_COLUMNS),
        dropna=False,
    )["team_median_s"].rank(method="average", ascending=True)
    team_rows["team_count"] = team_rows.groupby(
        list(CALIBRATION_SESSION_COLUMNS),
        dropna=False,
    )["team"].transform("count")
    team_rows["team_strength_same_session"] = np.where(
        team_rows["team_count"] > 1,
        1.0 - ((team_rows["team_rank"] - 1.0) / (team_rows["team_count"] - 1.0)),
        0.5,
    )
    enriched = enriched.merge(
        team_rows[
            [
                *CALIBRATION_SESSION_COLUMNS,
                "team",
                "team_median_s",
                "team_strength_same_session",
            ]
        ],
        on=[*CALIBRATION_SESSION_COLUMNS, "team"],
        how="left",
        validate="many_to_one",
    )

    race_team_rows = team_rows[team_rows["session_kind"].eq("race")].copy()
    race_team_rows = race_team_rows.rename(
        columns={"team_strength_same_session": "team_strength_race_event"}
    )
    enriched = enriched.merge(
        race_team_rows[["year", "race_name", "team", "team_strength_race_event"]],
        on=["year", "race_name", "team"],
        how="left",
        validate="many_to_one",
    )

    season_mean = (
        race_team_rows.groupby(["year", "team"], dropna=False, as_index=False)
        .agg(team_strength_race_season_mean=("team_strength_race_event", "mean"))
        .reset_index(drop=True)
    )
    enriched = enriched.merge(
        season_mean,
        on=["year", "team"],
        how="left",
        validate="many_to_one",
    )

    race_order = _race_order_frame(enriched)
    trailing = race_team_rows.merge(
        race_order,
        on=["year", "race_name"],
        how="left",
        validate="many_to_one",
    ).sort_values(["year", "team", "race_order", "race_name"])
    trailing["team_strength_race_trailing_mean"] = trailing.groupby(
        ["year", "team"],
        dropna=False,
    )["team_strength_race_event"].transform(lambda values: values.shift().expanding().mean())
    enriched = enriched.merge(
        trailing[
            [
                "year",
                "race_name",
                "team",
                "team_strength_race_trailing_mean",
            ]
        ],
        on=["year", "race_name", "team"],
        how="left",
        validate="many_to_one",
    )
    return enriched


def attach_driver_rating_mus(
    observations: pd.DataFrame,
    *,
    driver_mu_by_kind: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Attach prior driver means and the team-only calibration target."""
    required_columns = {
        "session_kind",
        "driver_code",
        "observed_driver_to_field_s",
    }
    _require_columns(observations, required_columns, "observations")

    enriched = observations.copy()
    enriched["driver_rating_mu_s"] = enriched.apply(
        lambda row: driver_mu_by_kind.get(str(row["session_kind"]), {}).get(
            str(row["driver_code"])
        ),
        axis=1,
    )
    enriched["team_target_s"] = (
        enriched["observed_driver_to_field_s"] - enriched["driver_rating_mu_s"]
    )
    return enriched


def fit_linear_team_strength_mapping(
    observations: pd.DataFrame,
    *,
    session_kind: str,
    policy: str,
    training_years: tuple[int, ...],
) -> LinearTeamStrengthMapping:
    """Fit one centered linear seconds mapping for a chosen proxy policy."""
    policy_column = _resolve_policy_column(policy)
    required_columns = {"year", "session_kind", "team_target_s", policy_column}
    _require_columns(observations, required_columns, "observations")

    train = observations[
        observations["session_kind"].eq(session_kind) & observations["year"].isin(training_years)
    ].dropna(subset=["team_target_s", policy_column])
    if train.empty:
        raise ValueError(
            f"No calibration rows for session_kind={session_kind!r}, policy={policy!r}"
        )

    centered_strength = train[policy_column].astype(float).to_numpy() - 0.5
    target = train["team_target_s"].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(train), dtype=float), centered_strength])
    intercept_s, slope_s_per_unit = np.linalg.lstsq(design, target, rcond=None)[0]
    return LinearTeamStrengthMapping(
        session_kind=session_kind,
        policy=policy,
        intercept_s=float(intercept_s),
        slope_s_per_unit=float(slope_s_per_unit),
        training_years=tuple(sorted(int(year) for year in training_years)),
    )


def load_live_team_strength_mappings(
    artifact_path: str | Path | None = None,
) -> dict[str, LinearTeamStrengthMapping]:
    """Load the frozen live team-strength seconds mappings.

    Missing artifacts return an empty mapping so prediction code can fall back to
    the older unit-scale path instead of failing a live prediction.
    """
    path = (
        Path(artifact_path)
        if artifact_path is not None
        else DEFAULT_TEAM_STRENGTH_SECONDS_MAPPING_PATH
    )
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return dict(_load_live_team_strength_mappings_cached(str(path)))


def team_strength_seconds_components(
    team_strength: float,
    *,
    session_kind: str,
    artifact_path: str | Path | None = None,
) -> dict[str, float] | None:
    """Return predicted and centered team-strength seconds for one runtime value."""
    try:
        strength = float(team_strength)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(strength):
        return None

    mapping = load_live_team_strength_mappings(artifact_path).get(str(session_kind))
    if mapping is None:
        return None
    return {
        "team_strength_seconds": mapping.predict_one(strength),
        "team_strength_seconds_delta": mapping.predict_delta_one(strength),
    }


@lru_cache(maxsize=8)
def _load_live_team_strength_mappings_cached(path_str: str) -> dict[str, LinearTeamStrengthMapping]:
    """Read and parse a frozen mapping artifact from disk."""
    path = Path(path_str)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings_payload = payload.get("mappings", {})
    if not isinstance(mappings_payload, Mapping):
        raise ValueError(f"Invalid team-strength mapping artifact: {path}")

    mappings: dict[str, LinearTeamStrengthMapping] = {}
    for session_kind, raw_mapping in mappings_payload.items():
        if not isinstance(raw_mapping, Mapping):
            raise ValueError(f"Invalid mapping payload for session_kind={session_kind!r} in {path}")
        training_years = raw_mapping.get("training_years", payload.get("training_years", ()))
        mappings[str(session_kind)] = LinearTeamStrengthMapping(
            session_kind=str(raw_mapping.get("session_kind", session_kind)),
            policy=str(raw_mapping.get("policy", payload.get("policy", ""))),
            intercept_s=float(raw_mapping["intercept_s"]),
            slope_s_per_unit=float(raw_mapping["slope_s_per_unit"]),
            training_years=tuple(int(year) for year in training_years),
        )
    return mappings


def evaluate_policy_folds(
    observations: pd.DataFrame,
    *,
    policy: str,
    fold_years: tuple[int, ...] = (2022, 2023, 2024, 2025),
) -> dict[str, Any]:
    """Evaluate one policy with leave-one-season-out calibration folds."""
    policy_column = _resolve_policy_column(policy)
    required_columns = {
        "year",
        "session_kind",
        "driver_code",
        "observed_driver_to_field_s",
        "driver_rating_mu_s",
        "team_target_s",
        policy_column,
    }
    _require_columns(observations, required_columns, "observations")

    fold_rows: list[dict[str, Any]] = []
    held_out_predictions: list[pd.DataFrame] = []
    for session_kind in ("race", "qualifying"):
        for holdout_year in fold_years:
            training_years = tuple(year for year in fold_years if year != holdout_year)
            mapping = fit_linear_team_strength_mapping(
                observations,
                session_kind=session_kind,
                policy=policy,
                training_years=training_years,
            )
            test = observations[
                observations["session_kind"].eq(session_kind)
                & observations["year"].eq(holdout_year)
            ].dropna(
                subset=[
                    policy_column,
                    "observed_driver_to_field_s",
                    "driver_rating_mu_s",
                ]
            )
            if test.empty:
                fold_rows.append(
                    {
                        "session_kind": session_kind,
                        "holdout_year": int(holdout_year),
                        "n_rows": 0,
                        "intercept_s": mapping.intercept_s,
                        "slope_s_per_unit": mapping.slope_s_per_unit,
                        "r_squared": None,
                        "prediction_slope": None,
                        "rmse_s": None,
                    }
                )
                continue

            predicted_team_s = mapping.predict(test[policy_column])
            predicted_driver_to_field_s = (
                predicted_team_s + test["driver_rating_mu_s"].astype(float).to_numpy()
            )
            observed = test["observed_driver_to_field_s"].astype(float).to_numpy()
            residual = observed - predicted_driver_to_field_s
            prediction_slope = _prediction_slope(
                observed=observed,
                predicted=predicted_driver_to_field_s,
            )
            fold_rows.append(
                {
                    "session_kind": session_kind,
                    "holdout_year": int(holdout_year),
                    "n_rows": int(len(test)),
                    "intercept_s": mapping.intercept_s,
                    "slope_s_per_unit": mapping.slope_s_per_unit,
                    "r_squared": _r_squared(
                        observed=observed, predicted=predicted_driver_to_field_s
                    ),
                    "prediction_slope": prediction_slope,
                    "rmse_s": float(np.sqrt(np.mean(np.square(residual)))),
                }
            )
            held_out_predictions.append(
                test[
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

    predictions = (
        pd.concat(held_out_predictions, ignore_index=True)
        if held_out_predictions
        else pd.DataFrame(
            columns=[
                "year",
                "race_name",
                "session_name",
                "session_kind",
                "team",
                "driver_code",
                "observed_driver_to_field_s",
                "predicted_driver_to_field_s",
                "residual_s",
            ]
        )
    )
    driver_residuals = (
        predictions.groupby(["session_kind", "driver_code"], dropna=False, as_index=False)
        .agg(
            residual_mean_s=("residual_s", "mean"),
            n_rows=("residual_s", "size"),
        )
        .sort_values(["session_kind", "driver_code"])
        .reset_index(drop=True)
    )
    return {
        "policy": policy,
        "policy_column": policy_column,
        "folds": fold_rows,
        "per_driver_residual_means": driver_residuals.to_dict(orient="records"),
        "held_out_predictions": predictions,
    }


def summarize_policy_coverage(observations: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize row coverage for every explicit team-strength proxy policy."""
    required_columns = {"session_kind", *TEAM_STRENGTH_POLICY_COLUMNS.values()}
    _require_columns(observations, required_columns, "observations")

    rows: list[dict[str, Any]] = []
    for policy, column in TEAM_STRENGTH_POLICY_COLUMNS.items():
        for session_kind, group in observations.groupby("session_kind", dropna=False):
            rows.append(
                {
                    "policy": policy,
                    "session_kind": str(session_kind),
                    "usable_rows": int(group[column].notna().sum()),
                    "total_rows": int(len(group)),
                }
            )
    return rows


def _race_order_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Return stable within-season race order from the input observation order."""
    race_rows = observations[observations["session_kind"].eq("race")][["year", "race_name"]]
    order = race_rows.drop_duplicates().reset_index(drop=True)
    order["race_order"] = order.groupby("year", dropna=False).cumcount() + 1
    return order


def _prediction_slope(*, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return the fitted slope of observed seconds on predicted seconds."""
    if len(observed) < 2 or np.isclose(np.nanstd(predicted), 0.0):
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


def _resolve_policy_column(policy: str) -> str:
    """Resolve a public policy name to the concrete observation column."""
    try:
        return TEAM_STRENGTH_POLICY_COLUMNS[policy]
    except KeyError as exc:
        raise ValueError(f"Unknown team-strength policy: {policy!r}") from exc


def _empty_observations() -> pd.DataFrame:
    """Return an empty frame with the core observation columns present."""
    return pd.DataFrame(
        columns=[
            *CALIBRATION_DRIVER_COLUMNS,
            "driver_median_s",
            "n_construct_laps",
            "field_median_s",
            "n_field_drivers",
            "n_field_teams",
            "observed_driver_to_field_s",
            "team_median_s",
            *TEAM_STRENGTH_POLICY_COLUMNS.values(),
        ]
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Raise a clear error when a required input column is missing."""
    missing = sorted(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
