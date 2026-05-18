"""Diagnostics for shared versus split practice-derived team signals."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.systems.testing_updater_metrics import (
    _filter_valid_laps,
    _median_lap_seconds,
    _normalize_lower_better,
    _select_program_aware_laps,
    _session_is_predominantly_wet,
)

PRACTICE_PROFILES: tuple[str, ...] = ("balanced", "short_run", "long_run")
NORMAL_WEEKEND_SESSIONS: tuple[str, ...] = ("FP1", "FP2", "FP3")
PRACTICE_SESSION_WEIGHTS: dict[str, float] = {"FP1": 0.68, "FP2": 0.82, "FP3": 1.0}
PROFILE_SIGNAL_COLUMNS: dict[str, str] = {
    "balanced": "practice_signal_balanced",
    "short_run": "practice_signal_short_run",
    "long_run": "practice_signal_long_run",
}
POLICY_PROFILE_BY_KIND: dict[str, dict[str, str]] = {
    "shared_balanced": {"qualifying": "balanced", "race": "balanced"},
    "shared_short_run": {"qualifying": "short_run", "race": "short_run"},
    "shared_long_run": {"qualifying": "long_run", "race": "long_run"},
    "split_short_quali_long_race": {"qualifying": "short_run", "race": "long_run"},
}
SHARED_POLICIES: tuple[str, ...] = (
    "shared_balanced",
    "shared_short_run",
    "shared_long_run",
)
SPLIT_POLICY = "split_short_quali_long_race"


@dataclass(frozen=True)
class PracticeSignalMapping:
    """One fitted linear conversion from a practice signal into team seconds."""

    session_kind: str
    policy: str
    profile: str
    intercept_s: float
    slope_s_per_unit: float
    training_years: tuple[int, ...]

    def predict(self, signal: pd.Series | np.ndarray | float) -> np.ndarray:
        """Convert centered 0-1 practice scores into predicted team seconds."""
        values = np.asarray(signal, dtype=float)
        return self.intercept_s + (self.slope_s_per_unit * (values - 0.5))


def discover_cached_normal_weekends(
    *,
    events: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    """Return observed events whose cache contains a full conventional weekend.

    The cache check is intentional: this diagnostic is about the situation where
    FP1, FP2, and FP3 were all actually available before qualifying and the race.
    Sprint weekends and incomplete cached weekends are excluded without asking a
    schedule API to infer what should have existed.
    """
    _require_columns(events, {"year", "race_name"}, "events")
    rows: list[dict[str, Any]] = []

    for event in (
        events[["year", "race_name"]]
        .drop_duplicates()
        .sort_values(["year", "race_name"])
        .itertuples(index=False)
    ):
        year = int(event.year)
        race_name = str(event.race_name)
        event_dir = _find_cached_event_dir(cache_dir=cache_dir, year=year, race_name=race_name)
        if event_dir is None:
            rows.append(
                {
                    "year": year,
                    "race_name": race_name,
                    "is_normal_weekend": False,
                    "cache_event_dir": None,
                    "available_sessions": [],
                }
            )
            continue

        available_sessions = sorted(_cached_session_labels(event_dir))
        required_sessions = {"practice1", "practice2", "practice3", "qualifying", "race"}
        rows.append(
            {
                "year": year,
                "race_name": race_name,
                "is_normal_weekend": required_sessions.issubset(available_sessions),
                "cache_event_dir": str(event_dir),
                "available_sessions": available_sessions,
            }
        )

    return pd.DataFrame(rows)


def extract_practice_profile_scores(
    session: Any,
    *,
    profiles: tuple[str, ...] = PRACTICE_PROFILES,
) -> dict[str, dict[str, float]]:
    """Extract normalized team pace scores for the requested practice profiles.

    This reuses the updater's existing lap-selection semantics and keeps the
    diagnostic focused on one question: whether those already-defined profiles
    carry distinct predictive information.
    """
    if _session_is_predominantly_wet(session):
        return {profile: {} for profile in profiles}

    laps = getattr(session, "laps", None)
    if not isinstance(laps, pd.DataFrame) or laps.empty:
        return {profile: {} for profile in profiles}
    _require_columns(laps, {"Team", "LapTime"}, "session.laps")

    scores_by_profile: dict[str, dict[str, float]] = {}
    for profile in profiles:
        lap_pace_seconds: dict[str, float] = {}
        for raw_team in sorted(laps["Team"].dropna().astype(str).unique()):
            team_laps = laps[laps["Team"].astype(str).eq(raw_team)]
            valid_laps = _filter_valid_laps(team_laps)
            if valid_laps.empty:
                continue
            selected_laps = _select_program_aware_laps(valid_laps, run_profile=profile)
            if selected_laps.empty:
                continue
            median_seconds = _median_lap_seconds(selected_laps)
            if median_seconds is None:
                continue
            lap_pace_seconds[raw_team] = float(median_seconds)
        scores_by_profile[profile] = _normalize_lower_better(lap_pace_seconds)

    return scores_by_profile


def aggregate_weekend_profile_scores(
    *,
    year: int,
    race_name: str,
    session_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    session_weights: Mapping[str, float] = PRACTICE_SESSION_WEIGHTS,
) -> pd.DataFrame:
    """Blend FP1-FP3 profile scores into one team-level weekend signal row."""
    rows: list[dict[str, Any]] = []

    for profile in PRACTICE_PROFILES:
        team_values: dict[str, list[tuple[float, float]]] = {}
        for session_code in NORMAL_WEEKEND_SESSIONS:
            profile_scores = session_scores.get(session_code, {}).get(profile, {})
            weight = float(session_weights.get(session_code, 1.0))
            for team_name, score in profile_scores.items():
                team_values.setdefault(str(team_name), []).append((float(score), weight))

        for team_name, values in team_values.items():
            scores = np.asarray([value for value, _weight in values], dtype=float)
            weights = np.asarray([weight for _value, weight in values], dtype=float)
            blended_score = (
                float(np.average(scores, weights=weights))
                if float(weights.sum()) > 0.0
                else float(scores.mean())
            )
            rows.append(
                {
                    "year": int(year),
                    "race_name": str(race_name),
                    "team": team_name,
                    "profile": profile,
                    "practice_signal": blended_score,
                    "n_sessions": int(len(values)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "race_name",
                "team",
                "profile",
                "practice_signal",
                "n_sessions",
            ]
        )
    return pd.DataFrame(rows).sort_values(["year", "race_name", "team", "profile"])


def pivot_weekend_profile_scores(profile_scores: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form profile scores into one row per weekend team."""
    _require_columns(
        profile_scores,
        {"year", "race_name", "team", "profile", "practice_signal", "n_sessions"},
        "profile_scores",
    )
    if profile_scores.empty:
        return pd.DataFrame(columns=["year", "race_name", "team"])

    signals = (
        profile_scores.pivot_table(
            index=["year", "race_name", "team"],
            columns="profile",
            values="practice_signal",
            aggfunc="first",
        )
        .rename(columns=PROFILE_SIGNAL_COLUMNS)
        .reset_index()
    )
    sessions = (
        profile_scores.pivot_table(
            index=["year", "race_name", "team"],
            columns="profile",
            values="n_sessions",
            aggfunc="first",
        )
        .rename(columns={profile: f"n_sessions_{profile}" for profile in PRACTICE_PROFILES})
        .reset_index()
    )
    return signals.merge(
        sessions,
        on=["year", "race_name", "team"],
        how="outer",
        validate="one_to_one",
    )


def attach_practice_signals(
    observations: pd.DataFrame,
    weekend_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach practice-derived team signals to construct-aligned observations."""
    _require_columns(observations, {"year", "race_name", "team"}, "observations")
    _require_columns(weekend_signals, {"year", "race_name", "team"}, "weekend_signals")
    return observations.merge(
        weekend_signals,
        on=["year", "race_name", "team"],
        how="left",
        validate="many_to_one",
    )


def restrict_to_common_signal_rows(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where every shared and split candidate has the same coverage."""
    required = {
        "session_kind",
        "driver_rating_mu_s",
        "team_target_s",
        *PROFILE_SIGNAL_COLUMNS.values(),
    }
    _require_columns(observations, required, "observations")
    usable = observations[observations["session_kind"].isin({"qualifying", "race"})].dropna(
        subset=[*PROFILE_SIGNAL_COLUMNS.values(), "driver_rating_mu_s", "team_target_s"]
    )
    return usable.reset_index(drop=True)


def evaluate_practice_signal_policies(observations: pd.DataFrame) -> dict[str, Any]:
    """Evaluate shared and split practice-signal policies with held-out seasons."""
    usable = restrict_to_common_signal_rows(observations)
    years = tuple(sorted(int(year) for year in usable["year"].dropna().unique()))
    if len(years) < 2:
        raise ValueError("At least two seasons are required for held-out evaluation.")

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout_year in years:
        training_years = tuple(year for year in years if year != holdout_year)
        train = usable[usable["year"].isin(training_years)]
        test = usable[usable["year"].eq(holdout_year)]

        for policy, profiles_by_kind in POLICY_PROFILE_BY_KIND.items():
            policy_predictions: list[pd.DataFrame] = []
            for session_kind, profile in profiles_by_kind.items():
                mapping = fit_practice_signal_mapping(
                    train,
                    session_kind=session_kind,
                    policy=policy,
                    profile=profile,
                    training_years=training_years,
                )
                holdout = test[test["session_kind"].eq(session_kind)].copy()
                signal_column = PROFILE_SIGNAL_COLUMNS[profile]
                holdout["policy"] = policy
                holdout["profile"] = profile
                holdout["predicted_team_s"] = mapping.predict(holdout[signal_column])
                holdout["predicted_driver_to_field_s"] = (
                    holdout["predicted_team_s"] + holdout["driver_rating_mu_s"]
                )
                holdout["prediction_error_s"] = (
                    holdout["predicted_driver_to_field_s"] - holdout["observed_driver_to_field_s"]
                )
                policy_predictions.append(holdout)

                fold_rows.append(
                    _fold_metric_row(
                        predictions=holdout,
                        policy=policy,
                        profile=profile,
                        session_kind=session_kind,
                        holdout_year=holdout_year,
                        mapping=mapping,
                    )
                )

            combined = pd.concat(policy_predictions, ignore_index=True)
            combined["session_kind"] = "combined"
            fold_rows.append(
                _fold_metric_row(
                    predictions=combined,
                    policy=policy,
                    profile="mixed" if policy == SPLIT_POLICY else profiles_by_kind["race"],
                    session_kind="combined",
                    holdout_year=holdout_year,
                    mapping=None,
                )
            )
            prediction_frames.append(pd.concat(policy_predictions, ignore_index=True))

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["session_kind", "holdout_year", "policy"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return {
        "fold_metrics": fold_metrics,
        "held_out_predictions": predictions,
        "comparison_vs_best_shared": compare_split_to_best_shared(fold_metrics),
    }


def fit_practice_signal_mapping(
    observations: pd.DataFrame,
    *,
    session_kind: str,
    policy: str,
    profile: str,
    training_years: tuple[int, ...],
) -> PracticeSignalMapping:
    """Fit one centered linear team-seconds mapping for a practice signal."""
    if profile not in PROFILE_SIGNAL_COLUMNS:
        raise ValueError(f"Unsupported profile: {profile}")
    signal_column = PROFILE_SIGNAL_COLUMNS[profile]
    _require_columns(observations, {"session_kind", "team_target_s", signal_column}, "observations")
    train = observations[observations["session_kind"].eq(session_kind)].dropna(
        subset=[signal_column, "team_target_s"]
    )
    if train.empty:
        raise ValueError(f"No training rows available for {session_kind}/{policy}/{profile}.")

    centered_signal = train[signal_column].astype(float).to_numpy() - 0.5
    target = train["team_target_s"].astype(float).to_numpy()
    design = np.column_stack([np.ones_like(centered_signal), centered_signal])
    intercept_s, slope_s_per_unit = np.linalg.lstsq(design, target, rcond=None)[0]
    return PracticeSignalMapping(
        session_kind=session_kind,
        policy=policy,
        profile=profile,
        intercept_s=float(intercept_s),
        slope_s_per_unit=float(slope_s_per_unit),
        training_years=training_years,
    )


def compare_split_to_best_shared(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare split-policy MSE against the optimistic best shared fold result."""
    _require_columns(
        fold_metrics,
        {"policy", "session_kind", "holdout_year", "mse_s"},
        "fold_metrics",
    )
    rows: list[dict[str, Any]] = []
    for (session_kind, holdout_year), group in fold_metrics.groupby(
        ["session_kind", "holdout_year"],
        dropna=False,
    ):
        split_rows = group[group["policy"].eq(SPLIT_POLICY)]
        shared_rows = group[group["policy"].isin(SHARED_POLICIES)]
        if split_rows.empty or shared_rows.empty:
            continue
        split_row = split_rows.iloc[0]
        best_shared_row = shared_rows.sort_values("mse_s").iloc[0]
        split_mse = float(split_row["mse_s"])
        best_shared_mse = float(best_shared_row["mse_s"])
        rows.append(
            {
                "session_kind": session_kind,
                "holdout_year": int(holdout_year),
                "split_mse_s2": split_mse,
                "best_shared_policy": str(best_shared_row["policy"]),
                "best_shared_mse_s2": best_shared_mse,
                "delta_mse_s2": split_mse - best_shared_mse,
                "split_wins": bool(split_mse < best_shared_mse),
            }
        )
    return pd.DataFrame(rows).sort_values(["session_kind", "holdout_year"]).reset_index(drop=True)


def summarize_signal_coverage(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize profile availability before the common-row restriction."""
    required = {"session_kind", *PROFILE_SIGNAL_COLUMNS.values()}
    _require_columns(observations, required, "observations")
    rows: list[dict[str, Any]] = []
    for session_kind, group in observations.groupby("session_kind", dropna=False):
        row: dict[str, Any] = {
            "session_kind": session_kind,
            "total_rows": int(len(group)),
        }
        for profile, signal_column in PROFILE_SIGNAL_COLUMNS.items():
            row[f"{profile}_rows"] = int(group[signal_column].notna().sum())
        row["common_rows"] = int(
            group[list(PROFILE_SIGNAL_COLUMNS.values())].notna().all(axis=1).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_kind").reset_index(drop=True)


def summarize_weighted_policy_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-weighted MSE and RMSE across held-out seasons."""
    _require_columns(
        fold_metrics,
        {"policy", "session_kind", "n_rows", "mse_s"},
        "fold_metrics",
    )
    rows: list[dict[str, Any]] = []
    for (session_kind, policy), group in fold_metrics.groupby(
        ["session_kind", "policy"],
        dropna=False,
    ):
        weights = group["n_rows"].astype(float).to_numpy()
        mse_values = group["mse_s"].astype(float).to_numpy()
        total_weight = float(weights.sum())
        weighted_mse = (
            float(np.average(mse_values, weights=weights)) if total_weight > 0.0 else float("nan")
        )
        rows.append(
            {
                "session_kind": session_kind,
                "policy": policy,
                "n_rows": int(total_weight),
                "weighted_mse_s2": weighted_mse,
                "weighted_rmse_s": float(np.sqrt(weighted_mse)),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["session_kind", "weighted_mse_s2"]).reset_index(drop=True)
    )


def _fold_metric_row(
    *,
    predictions: pd.DataFrame,
    policy: str,
    profile: str,
    session_kind: str,
    holdout_year: int,
    mapping: PracticeSignalMapping | None,
) -> dict[str, Any]:
    """Build one held-out metric row from prediction residuals."""
    if predictions.empty:
        return {
            "policy": policy,
            "profile": profile,
            "session_kind": session_kind,
            "holdout_year": int(holdout_year),
            "n_rows": 0,
            "intercept_s": np.nan,
            "slope_s_per_unit": np.nan,
            "mse_s": np.nan,
            "rmse_s": np.nan,
            "r_squared": np.nan,
        }

    errors = predictions["prediction_error_s"].astype(float).to_numpy()
    observed = predictions["observed_driver_to_field_s"].astype(float).to_numpy()
    residual_ss = float(np.sum(errors**2))
    centered_observed = observed - float(np.mean(observed))
    total_ss = float(np.sum(centered_observed**2))
    mse_s = float(np.mean(errors**2))
    return {
        "policy": policy,
        "profile": profile,
        "session_kind": session_kind,
        "holdout_year": int(holdout_year),
        "n_rows": int(len(predictions)),
        "intercept_s": None if mapping is None else mapping.intercept_s,
        "slope_s_per_unit": None if mapping is None else mapping.slope_s_per_unit,
        "mse_s": mse_s,
        "rmse_s": float(np.sqrt(mse_s)),
        "r_squared": float(1.0 - (residual_ss / total_ss)) if total_ss > 0 else np.nan,
    }


def _find_cached_event_dir(*, cache_dir: Path, year: int, race_name: str) -> Path | None:
    """Resolve one event directory from the local FastF1 cache."""
    year_dir = cache_dir / str(year)
    if not year_dir.exists():
        return None
    target = _normalize_label(race_name)
    for event_dir in year_dir.iterdir():
        if not event_dir.is_dir() or "_" not in event_dir.name:
            continue
        _date_prefix, raw_event_name = event_dir.name.split("_", 1)
        event_name = raw_event_name.replace("_", " ")
        if _normalize_label(event_name) == target:
            return event_dir
    return None


def _cached_session_labels(event_dir: Path) -> set[str]:
    """Return normalized session names with at least one cached payload file."""
    labels: set[str] = set()
    for session_dir in event_dir.iterdir():
        if not session_dir.is_dir() or "_" not in session_dir.name:
            continue
        try:
            if not any(child.is_file() for child in session_dir.iterdir()):
                continue
        except OSError:
            continue
        _date_prefix, raw_session_name = session_dir.name.split("_", 1)
        labels.add(_normalize_label(raw_session_name.replace("_", " ")))
    return labels


def _normalize_label(value: str) -> str:
    """Normalize cache and event labels for accent-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "".join(char for char in ascii_text.lower() if char.isalnum())


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a useful error when a DataFrame is missing required columns."""
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
