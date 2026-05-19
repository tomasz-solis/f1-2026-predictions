"""Update team and driver characteristics after a completed race."""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from src.models.bayesian import BayesianDriverRanking
from src.persistence.artifact_store import ArtifactStore
from src.systems.compound_analyzer import (
    aggregate_compound_samples,
    extract_compound_metrics,
    normalize_compound_metrics_across_teams,
)
from src.systems.updater_flow import (
    update_team_characteristics_core as _update_team_characteristics_core,
)
from src.utils import config_loader
from src.utils.schema_validation import strip_legacy_bayesian_fields
from src.utils.team_mapping import map_team_to_characteristics
from src.utils.validation_helpers import normalize_weather_key

logger = logging.getLogger(__name__)


def _driver_characteristics_fallback_paths(year: int) -> tuple[Path, ...]:
    """Return the season-scoped driver file first, then the legacy fallback path."""
    processed_root = Path("data/processed")
    return (
        processed_root / "driver_characteristics" / f"{year}_driver_characteristics.json",
        processed_root / "driver_characteristics.json",
    )


def _coerce_season_year(race_results: pd.DataFrame, default_year: int = 2026) -> int:
    """Read the season year from race results."""
    if "year" not in race_results.columns or race_results.empty:
        return default_year
    raw_year = race_results["year"].iloc[0]
    try:
        return int(raw_year)
    except (TypeError, ValueError):
        return default_year


def _load_driver_characteristics_payload(
    store: ArtifactStore,
    year: int,
) -> dict | None:
    """Load driver characteristics from the store or JSON file."""
    artifact_key = f"{year}::driver_characteristics"
    payload = store.load_artifact(
        artifact_type="driver_characteristics",
        artifact_key=artifact_key,
    )
    if payload:
        return payload

    for fallback_file in _driver_characteristics_fallback_paths(year):
        if not fallback_file.exists():
            continue
        try:
            with open(fallback_file) as f:
                fallback_payload = json.load(f)
            logger.info(
                "Loaded driver characteristics fallback from %s for season %s",
                fallback_file,
                year,
            )
            return fallback_payload
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Could not read driver characteristics fallback %s: %s",
                fallback_file,
                exc,
            )
            continue
    return None


def _persist_driver_characteristics_payload(
    store: ArtifactStore,
    payload: dict,
    year: int,
) -> None:
    """Save driver characteristics to the store and JSON file."""
    artifact_key = f"{year}::driver_characteristics"
    fallback_file = _driver_characteristics_fallback_paths(year)[0]
    fallback_file.parent.mkdir(parents=True, exist_ok=True)

    current_version_raw = payload.get("version", 0)
    try:
        current_version = int(current_version_raw)
    except (TypeError, ValueError):
        current_version = 0

    latest_store_version = 0
    try:
        latest_store_version = int(store.get_latest_version("driver_characteristics", artifact_key))
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        latest_store_version = 0
    new_version = max(current_version, latest_store_version) + 1

    payload["version"] = new_version
    payload["last_updated"] = datetime.now().isoformat()
    payload["bayesian_last_updated_year"] = year

    def _write_fallback_file() -> None:
        """Persist the same payload to the season-scoped JSON fallback."""
        with open(fallback_file, "w") as f:
            json.dump(payload, f, indent=2)

    try:
        store.save_artifact(
            artifact_type="driver_characteristics",
            artifact_key=artifact_key,
            data=payload,
            version=new_version,
        )
        _write_fallback_file()
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "ArtifactStore save failed for driver characteristics: %s. "
            "Falling back to season-scoped file %s.",
            exc,
            fallback_file,
        )
        _write_fallback_file()


def _read_saved_bayesian_state(
    bayesian_payload: Any,
) -> tuple[float, float] | None:
    """Return a saved `(mu, sigma)` pair when it is valid."""
    if not isinstance(bayesian_payload, dict):
        return None

    try:
        rating_mu = float(bayesian_payload["rating_mu"])
        rating_sigma = float(bayesian_payload["rating_sigma"])
    except (KeyError, TypeError, ValueError):
        return None

    if not np.isfinite(rating_mu) or not np.isfinite(rating_sigma) or rating_sigma <= 0.0:
        return None

    return float(rating_mu), float(rating_sigma)


def _restore_bayesian_state(
    bayesian: BayesianDriverRanking,
    drivers_payload: dict[str, Any],
) -> int:
    """Restore saved in-season ratings."""
    seeded_drivers = 0
    for driver_code, driver_entry in drivers_payload.items():
        if driver_code not in bayesian.ratings or not isinstance(driver_entry, dict):
            continue

        persisted_state = _read_saved_bayesian_state(driver_entry.get("bayesian"))
        if persisted_state is None:
            continue

        bayesian.ratings[driver_code] = persisted_state
        seeded_drivers += 1

    return seeded_drivers


def _read_sessions_observed(bayesian_payload: Any) -> int:
    """Read a non-negative observed-session count from a Bayesian payload."""
    if not isinstance(bayesian_payload, dict):
        return 0
    try:
        return max(int(bayesian_payload.get("sessions_observed", 0)), 0)
    except (TypeError, ValueError):
        return 0


def _bayesian_history_metadata(
    bayesian: BayesianDriverRanking,
) -> tuple[Counter[str], dict[str, str]]:
    """Summarize update counts and last session names from a Bayesian model."""
    raw_history = getattr(bayesian, "history", [])
    if not isinstance(raw_history, list):
        return Counter(), {}

    update_counts: Counter[str] = Counter()
    last_sessions: dict[str, str] = {}
    for record in raw_history:
        driver_code = str(getattr(record, "driver_number", "")).strip()
        session_name = str(getattr(record, "session_name", "")).strip()
        if not driver_code:
            continue
        update_counts[driver_code] += 1
        if session_name:
            last_sessions[driver_code] = session_name
    return update_counts, last_sessions


def _persist_bayesian_ratings_to_drivers(
    *,
    bayesian: BayesianDriverRanking,
    drivers_payload: dict[str, Any],
    season_year: int,
    fallback_session_name: str,
    updated_at: str,
) -> int:
    """Persist the current Bayesian ratings and observed-session counters."""
    update_counts, last_sessions = _bayesian_history_metadata(bayesian)
    touched = 0

    for driver_code, (mu, sigma) in bayesian.ratings.items():
        driver_entry = drivers_payload.get(driver_code)
        if not isinstance(driver_entry, dict):
            continue

        previous_bayesian = driver_entry.get("bayesian")
        sessions_observed = _read_sessions_observed(previous_bayesian) + int(
            update_counts.get(driver_code, 0)
        )
        seeded_from = (
            previous_bayesian.get("seeded_from") if isinstance(previous_bayesian, dict) else None
        )
        if not seeded_from:
            seeded_from = "inseason_update" if sessions_observed > 0 else "extraction_prior"

        driver_entry["bayesian"] = {
            "rating_mu": float(mu),
            "rating_sigma": float(sigma),
            "sessions_observed": int(sessions_observed),
            "seeded_from": str(seeded_from),
            "last_session": last_sessions.get(driver_code, fallback_session_name),
            "last_updated": updated_at,
            "season_year": season_year,
        }
        touched += 1

    return touched


def load_competitive_session(
    year: int,
    race_name: str,
    session_name: str,
    *,
    load_laps: bool,
) -> tuple[pd.DataFrame, fastf1.core.Session]:
    """Load one competitive FastF1 session and annotate the results table."""
    logger.info("Loading %s %s %s results...", year, race_name, session_name)

    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, race_name, session_name)
    session.load(laps=load_laps, telemetry=False, weather=False)

    results = session.results.copy()
    results["race_name"] = race_name
    results["year"] = year
    results["session_name"] = str(session_name).strip().upper()

    return results, session


def load_race_session(year: int, race_name: str) -> tuple[pd.DataFrame, fastf1.core.Session]:
    """Load race results and session from FastF1."""
    return load_competitive_session(year, race_name, "R", load_laps=True)


def load_qualifying_session(year: int, race_name: str) -> tuple[pd.DataFrame, fastf1.core.Session]:
    """Load qualifying results and session from FastF1."""
    return load_competitive_session(year, race_name, "Q", load_laps=False)


def _build_position_observations(session_results: pd.DataFrame) -> dict[str, int]:
    """Extract clean driver-position observations from one session results table."""
    if session_results.empty:
        return {}
    if "Abbreviation" not in session_results.columns or "Position" not in session_results.columns:
        return {}

    observations: dict[str, int] = {}
    for driver, position in zip(
        session_results["Abbreviation"].tolist(),
        session_results["Position"].tolist(),
        strict=False,
    ):
        if pd.isna(position):
            continue
        try:
            observations[str(driver)] = int(position)
        except (TypeError, ValueError):
            continue
    return observations


def extract_team_performance_from_telemetry(
    session: fastf1.core.Session, team_names: list[str]
) -> dict[str, float]:
    """
    Extract team performance from race telemetry using median lap times.

    Filters:
    - Pit laps excluded
    - Lap 1 and last lap excluded
    - Outliers (>3σ) excluded

    Returns dict of team -> performance (0-1 scale, 1.0 = fastest)
    """
    race_pace = {}

    try:
        laps = session.laps
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Could not access lap data: %s", exc)
        return {}

    if laps is None or laps.empty:
        logger.warning("No lap data available")
        return {}

    known_teams = set(team_names)
    if "Team" not in laps.columns:
        logger.warning("Lap data does not include Team column")
        return {}

    laps = laps.copy()
    laps["_canonical_team"] = laps["Team"].apply(
        lambda raw: map_team_to_characteristics(raw, known_teams=known_teams)
    )

    for team in team_names:
        team_laps = laps[laps["_canonical_team"] == team]

        if len(team_laps) == 0:
            logger.warning("  No laps found for %s", team)
            continue

        mask = team_laps["LapTime"].notna()
        if "PitOutTime" in team_laps.columns:
            mask &= team_laps["PitOutTime"].isna()
        if "PitInTime" in team_laps.columns:
            mask &= team_laps["PitInTime"].isna()
        if "LapNumber" in team_laps.columns:
            mask &= team_laps["LapNumber"] > 1
            mask &= team_laps["LapNumber"] < team_laps["LapNumber"].max()

        valid_laps = team_laps[mask]

        if len(valid_laps) < 5:
            logger.warning("  %s: Only %s valid laps, skipping", team, len(valid_laps))
            continue

        lap_times_seconds = valid_laps["LapTime"].dt.total_seconds()

        mean_time = lap_times_seconds.mean()
        std_time = lap_times_seconds.std()
        clean_times = lap_times_seconds[
            (lap_times_seconds > mean_time - 3 * std_time)
            & (lap_times_seconds < mean_time + 3 * std_time)
        ]

        if len(clean_times) == 0:
            clean_times = lap_times_seconds

        median_time = clean_times.median()
        race_pace[team] = median_time
        logger.debug(
            "  %s: Median lap time %ss (%s laps)",
            team,
            format(median_time, ".3f"),
            len(clean_times),
        )

    if race_pace:
        ranked_items = sorted(race_pace.items(), key=lambda item: item[1])
        team_count = len(ranked_items)
        if team_count < 2:
            for team in race_pace:
                race_pace[team] = 0.5
        else:
            grouped_items: list[tuple[float, list[str]]] = []
            for team, lap_time in ranked_items:
                lap_time_value = float(lap_time)
                if grouped_items and np.isclose(lap_time_value, grouped_items[-1][0]):
                    grouped_items[-1][1].append(team)
                else:
                    grouped_items.append((lap_time_value, [team]))

            rank_cursor = 0
            for _lap_time, tied_teams in grouped_items:
                average_rank = rank_cursor + ((len(tied_teams) - 1) / 2.0)
                normalized_score = float(1.0 - (average_rank / (team_count - 1)))
                for team in tied_teams:
                    race_pace[team] = normalized_score
                rank_cursor += len(tied_teams)

    return race_pace


def update_team_characteristics(
    race_results: pd.DataFrame, session: fastf1.core.Session, characteristics_file: Path
) -> None:
    """Update team performance ratings from race telemetry."""
    _update_team_characteristics_core(
        race_results=race_results,
        session=session,
        characteristics_file=characteristics_file,
        artifact_store_factory=ArtifactStore,
        extract_team_performance_from_telemetry_fn=extract_team_performance_from_telemetry,
        map_team_to_characteristics_fn=map_team_to_characteristics,
        extract_compound_metrics_fn=extract_compound_metrics,
        normalize_compound_metrics_across_teams_fn=normalize_compound_metrics_across_teams,
        aggregate_compound_samples_fn=aggregate_compound_samples,
        config_get_fn=config_loader.get,
        logger=logger,
    )


def _extract_dnf_drivers(race_results: pd.DataFrame) -> set[str]:
    """Return driver codes that retired rather than finishing the race.

    Keeps drivers who finished or were classified as lapped (Status contains
    "Lap") but excludes mechanical retirements, collisions, and other DNFs.
    """
    dnf_drivers: set[str] = set()
    if not isinstance(race_results, pd.DataFrame) or "Status" not in race_results.columns:
        return dnf_drivers

    for _, row in race_results.iterrows():
        status = str(row.get("Status", "")).strip()
        if status and status != "Finished" and "Lap" not in status:
            abbrev = str(row.get("Abbreviation", "")).strip()
            if abbrev:
                dnf_drivers.add(abbrev)
    return dnf_drivers


def _status_observed_driver_codes(session_results: pd.DataFrame) -> set[str]:
    """Return drivers whose result row has an explicit classified status."""
    observed: set[str] = set()
    if not isinstance(session_results, pd.DataFrame):
        return observed
    if "Abbreviation" not in session_results.columns or "Status" not in session_results.columns:
        return observed

    for _, row in session_results.iterrows():
        status = str(row.get("Status", "")).strip()
        driver_code = str(row.get("Abbreviation", "")).strip()
        if status and driver_code:
            observed.add(driver_code)
    return observed


def _update_dnf_rate_ema(
    *,
    session_results: pd.DataFrame,
    drivers_payload: dict[str, Any],
    blend_weight: float,
    floor: float,
    cap: float,
) -> int:
    """Update stored DNF risk from explicit classified/retired statuses."""
    observed_codes = _status_observed_driver_codes(session_results)
    if not observed_codes:
        return 0

    dnf_drivers = _extract_dnf_drivers(session_results)
    safe_blend = float(np.clip(blend_weight, 0.0, 1.0))
    safe_floor = float(np.clip(floor, 0.0, 1.0))
    safe_cap = float(np.clip(max(cap, safe_floor), safe_floor, 1.0))
    touched = 0

    for driver_code in observed_codes:
        driver_entry = drivers_payload.get(driver_code)
        if not isinstance(driver_entry, dict):
            continue
        dnf_payload = driver_entry.get("dnf_risk")
        if not isinstance(dnf_payload, dict):
            dnf_payload = {}
            driver_entry["dnf_risk"] = dnf_payload

        try:
            existing_rate = float(dnf_payload.get("dnf_rate", 0.10))
        except (TypeError, ValueError):
            existing_rate = 0.10
        if not np.isfinite(existing_rate):
            existing_rate = 0.10

        observed_rate = 1.0 if driver_code in dnf_drivers else 0.0
        updated_rate = ((1.0 - safe_blend) * existing_rate) + (safe_blend * observed_rate)
        dnf_payload["dnf_rate"] = round(float(np.clip(updated_rate, safe_floor, safe_cap)), 3)
        touched += 1

    return touched


def _update_teammate_relative_pace_ema(
    *,
    observations: dict[str, int],
    drivers_payload: dict[str, dict],
    driver_to_team: dict[str, str],
    grid_size: int,
    blend_weight: float,
    pace_key: str,
) -> int:
    """Apply teammate-relative EMA update to a driver pace field.

    Normalizes finishing positions into 0-1 pace, removes team-level effects
    by subtracting each team's mean and re-centering on the field mean, then
    blends with the existing pace value via exponential moving average.

    Args:
        observations: driver_code -> finishing position (1-indexed).
        drivers_payload: mutable driver entries; updated in-place.
        driver_to_team: driver_code -> team name mapping.
        grid_size: total grid size for position normalization.
        blend_weight: EMA blend toward new observation (0=ignore, 1=replace).
        pace_key: field name in the pace dict ("quali_pace" or "race_pace").

    Returns:
        Count of drivers whose pace was updated.
    """
    if not observations:
        return 0

    raw_paces: dict[str, float] = {
        dc: float(np.clip(1.0 - ((int(pos) - 1) / max(grid_size - 1, 1)), 0.0, 1.0))
        for dc, pos in observations.items()
    }

    team_paces: dict[str, list[float]] = {}
    for dc, raw_pace in raw_paces.items():
        t = driver_to_team.get(dc)
        if t is not None:
            team_paces.setdefault(t, []).append(raw_pace)

    team_means: dict[str, float] = {
        t: float(np.mean(paces)) for t, paces in team_paces.items() if len(paces) >= 2
    }
    field_mean = float(np.mean(list(raw_paces.values()))) if raw_paces else 0.5

    touched = 0
    for driver_code, raw_pace in raw_paces.items():
        driver_entry = drivers_payload.get(driver_code)
        if not isinstance(driver_entry, dict):
            continue

        pace_payload = driver_entry.get("pace")
        if not isinstance(pace_payload, dict):
            pace_payload = {}
            driver_entry["pace"] = pace_payload

        try:
            existing_pace = float(pace_payload.get(pace_key, 0.5))
        except (TypeError, ValueError):
            existing_pace = 0.5
        existing_pace = float(np.clip(existing_pace, 0.05, 0.95))

        driver_team = driver_to_team.get(driver_code)
        if driver_team is not None and driver_team in team_means:
            observed_pace = raw_pace - team_means[driver_team] + field_mean
        else:
            observed_pace = raw_pace
        observed_pace = float(np.clip(observed_pace, 0.0, 1.0))

        updated_pace = (1.0 - blend_weight) * existing_pace + (blend_weight * observed_pace)
        pace_payload[pace_key] = round(float(np.clip(updated_pace, 0.05, 0.95)), 3)
        touched += 1

    return touched


def update_bayesian_driver_ratings(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame | None = None,
    *,
    data_root: str | Path = "data",
    weather: str = "dry",
) -> None:
    """Update Bayesian driver ratings plus qualifying pace from completed sessions."""
    logger.info("Updating Bayesian driver ratings...")

    from src.models.priors_factory import PriorsFactory

    season_year = _coerce_season_year(race_results)
    factory = PriorsFactory(season_year=season_year)
    priors = factory.create_priors()
    configured_grid_size = int(config_loader.get("grid.size", len(priors) or 22))
    grid_size = max(configured_grid_size, len(priors) or configured_grid_size)

    bayesian = BayesianDriverRanking(priors, grid_size=grid_size)
    store = ArtifactStore(data_root=data_root)
    driver_payload = _load_driver_characteristics_payload(store, season_year)
    drivers_payload = driver_payload.get("drivers") if isinstance(driver_payload, dict) else None
    if isinstance(drivers_payload, dict):
        strip_legacy_bayesian_fields(drivers_payload)
        _restore_bayesian_state(bayesian, drivers_payload)

    all_observations = _build_position_observations(race_results)
    dnf_drivers = _extract_dnf_drivers(race_results)
    observations = {code: pos for code, pos in all_observations.items() if code not in dnf_drivers}
    if dnf_drivers:
        logger.info(
            "Excluded %s DNF drivers from Bayesian update: %s",
            len(dnf_drivers),
            ", ".join(sorted(dnf_drivers)),
        )
    if not observations:
        logger.warning("No valid race positions available for Bayesian rating update")
        return

    session_name = str(race_results.get("race_name", pd.Series(["Race"])).iloc[0])
    teammate_confidence = float(config_loader.get("bayesian.teammate_relative_confidence", 0.35))
    teammate_confidence = float(np.clip(teammate_confidence, 0.05, 1.0))
    from src.utils.lineups import load_current_lineups

    lineups = load_current_lineups()

    # Build driver->team mapping once; shared by both pace update blocks below.
    driver_to_team: dict[str, str] = {}
    for team_name, team_drivers in (lineups or {}).items():
        for d in team_drivers:
            driver_to_team[str(d)] = str(team_name)

    if lineups:
        bayesian.update_teammate_relative(
            observations=observations,
            session_name=session_name,
            lineups=lineups,
            confidence=teammate_confidence,
        )
    else:
        bayesian.update(observations=observations, session_name=session_name, confidence=1.0)

    if not isinstance(driver_payload, dict):
        logger.warning("Could not load driver characteristics payload to persist Bayesian updates")
        return

    if not isinstance(drivers_payload, dict):
        logger.warning(
            "Driver characteristics payload missing 'drivers'; skipping Bayesian persistence"
        )
        return

    # --- Qualifying Bayesian update ---
    # Lower confidence than race (0.15 vs 0.35): qualifying is one lap in controlled
    # conditions, so it's a noisier signal than race finishing position.
    # Runs AFTER the race update so the prior going into prediction already
    # reflects the weekend's race outcome; qualifying nudges it further for
    # drivers whose one-lap pace diverges from their race pace (e.g. Gasly-era
    # pattern of strong qualifying, weaker race management).
    qualifying_observations = (
        _build_position_observations(qualifying_results)
        if isinstance(qualifying_results, pd.DataFrame)
        else {}
    )
    if qualifying_observations:
        quali_bayesian_confidence = float(
            config_loader.get("bayesian.qualifying_update_confidence", 0.15)
        )
        quali_bayesian_confidence = float(np.clip(quali_bayesian_confidence, 0.05, 0.5))
        if lineups:
            bayesian.update_teammate_relative(
                observations=qualifying_observations,
                session_name=f"Qualifying_{session_name}",
                lineups=lineups,
                confidence=quali_bayesian_confidence,
            )
        else:
            bayesian.update(
                observations=qualifying_observations,
                session_name=f"Qualifying_{session_name}",
                confidence=quali_bayesian_confidence,
            )
        logger.info(
            "Qualifying Bayesian update applied for %s drivers (confidence=%.2f)",
            len(qualifying_observations),
            quali_bayesian_confidence,
        )

    # Persist Bayesian state only. Prediction-time blending in
    # _blend_race_skill_with_bayesian_form handles the skill_score adjustment —
    # keeping it in one place avoids double-counting. This must happen after the
    # qualifying update above so the stored posterior reflects both sessions.
    touched_skill_drivers = _persist_bayesian_ratings_to_drivers(
        bayesian=bayesian,
        drivers_payload=drivers_payload,
        season_year=season_year,
        fallback_session_name=session_name,
        updated_at=datetime.now().isoformat(),
    )

    # --- Qualifying pace EMA update (teammate-relative) ---
    quali_pace_blend = float(
        config_loader.get("baseline_predictor.driver_form.quali_pace_update_blend", 0.30)
    )
    quali_pace_blend = float(np.clip(quali_pace_blend, 0.0, 1.0))
    touched_quali_pace_drivers = _update_teammate_relative_pace_ema(
        observations=qualifying_observations,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        grid_size=grid_size,
        blend_weight=quali_pace_blend,
        pace_key="quali_pace",
    )

    # --- Race pace EMA update (teammate-relative, DNF drivers excluded) ---
    race_pace_blend = float(
        config_loader.get("baseline_predictor.driver_form.race_pace_update_blend", 0.25)
    )
    race_pace_blend = float(np.clip(race_pace_blend, 0.0, 1.0))
    race_finish_obs: dict[str, int] = {
        dc: pos for dc, pos in all_observations.items() if dc not in dnf_drivers
    }
    touched_race_pace_drivers = _update_teammate_relative_pace_ema(
        observations=race_finish_obs,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        grid_size=grid_size,
        blend_weight=race_pace_blend,
        pace_key="race_pace",
    )

    # --- Wet-skill EMA update (only in wet/mixed conditions) ---
    weather_key = normalize_weather_key(weather)
    touched_wet_skill_drivers = 0
    if weather_key in {"rain", "mixed"}:
        wet_skill_blend = float(
            config_loader.get("baseline_predictor.driver_form.wet_skill_update_blend", 0.15)
        )
        wet_skill_blend = float(np.clip(wet_skill_blend, 0.0, 0.5))
        if weather_key == "mixed":
            wet_skill_blend *= 0.5

        wet_skill_neutral = float(
            config_loader.get("baseline_predictor.race.lap_time.wet_skill_neutral", 0.70)
        )
        wet_skill_observation_scale = float(
            config_loader.get("baseline_predictor.driver_form.wet_skill_observation_scale", 0.40)
        )

        for driver_code, pos in race_finish_obs.items():
            driver_entry = drivers_payload.get(driver_code)
            if not isinstance(driver_entry, dict):
                continue

            raw_pace = float(np.clip(1.0 - ((int(pos) - 1) / max(grid_size - 1, 1)), 0.0, 1.0))

            driver_team = driver_to_team.get(driver_code)
            team_finish_paces = [
                float(np.clip(1.0 - ((int(tp) - 1) / max(grid_size - 1, 1)), 0.0, 1.0))
                for tdc, tp in race_finish_obs.items()
                if driver_to_team.get(tdc) == driver_team and tdc != driver_code
            ]
            if not team_finish_paces:
                continue

            relative_performance = raw_pace - float(np.mean(team_finish_paces))
            observed_wet_signal = wet_skill_neutral + (
                relative_performance * wet_skill_observation_scale
            )
            observed_wet_signal = float(np.clip(observed_wet_signal, 0.40, 0.95))

            existing = driver_entry.get("wet_skill")
            existing_wet_skill = float(existing if existing is not None else wet_skill_neutral)
            updated = (
                1.0 - wet_skill_blend
            ) * existing_wet_skill + wet_skill_blend * observed_wet_signal
            driver_entry["wet_skill"] = round(updated, 3)
            touched_wet_skill_drivers += 1

        if touched_wet_skill_drivers > 0:
            logger.info("Updated wet_skill for %d drivers", touched_wet_skill_drivers)

    dnf_rate_blend = float(
        config_loader.get("baseline_predictor.driver_form.dnf_rate_update_blend", 0.10)
    )
    touched_dnf_rate_drivers = _update_dnf_rate_ema(
        session_results=race_results,
        drivers_payload=drivers_payload,
        blend_weight=dnf_rate_blend,
        floor=float(config_loader.get("baseline_predictor.driver_form.dnf_rate_floor", 0.02)),
        cap=float(config_loader.get("baseline_predictor.driver_form.dnf_rate_cap", 0.35)),
    )

    if (
        touched_skill_drivers == 0
        and touched_quali_pace_drivers == 0
        and touched_race_pace_drivers == 0
        and touched_wet_skill_drivers == 0
        and touched_dnf_rate_drivers == 0
    ):
        logger.warning("Bayesian update produced no persisted driver changes")
        return

    _persist_driver_characteristics_payload(store, driver_payload, season_year)
    logger.info(
        "Updated Bayesian ratings for %s drivers, blended skill for %s, "
        "qualifying pace for %s, race pace for %s, wet_skill for %s, dnf_rate for %s",
        len(observations),
        touched_skill_drivers,
        touched_quali_pace_drivers,
        touched_race_pace_drivers,
        touched_wet_skill_drivers,
        touched_dnf_rate_drivers,
    )


def update_from_race(year: int, race_name: str, data_dir: str = "data/processed") -> None:
    """
    Main entry point: Update all characteristics after a race.

    Workflow:
    1. Load race results from FastF1
    2. Update team performance from telemetry
    3. Update Bayesian driver ratings
    4. Reduce uncertainty
    """
    logger.info("=" * 60)
    logger.info("Updating from %s %s", year, race_name)
    logger.info("=" * 60)

    try:
        race_results, session = load_race_session(year, race_name)
        logger.info("Loaded results for %s drivers\n", len(race_results))
    except (AttributeError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error("Failed to load race results: %s", e)
        logger.error("Make sure race has completed and data is available via FastF1")
        raise

    try:
        qualifying_results, _qualifying_session = load_qualifying_session(year, race_name)
        logger.info("Loaded qualifying results for %s drivers", len(qualifying_results))
    except (AttributeError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not load qualifying results for %s %s; skipping quali pace refresh (%s)",
            year,
            race_name,
            exc,
        )
        qualifying_results = None

    # Detect race weather from session for wet_skill updating
    race_weather = "dry"
    try:
        weather_data = session.weather_data
        if (
            weather_data is not None
            and not weather_data.empty
            and "Rainfall" in weather_data.columns
        ):
            wet_fraction = float(weather_data["Rainfall"].dropna().astype(bool).mean())
            if wet_fraction > 0.30:
                race_weather = "rain"
            elif wet_fraction > 0.05:
                race_weather = "mixed"
    except Exception:
        race_weather = "dry"

    char_file = Path(data_dir) / "car_characteristics" / f"{year}_car_characteristics.json"
    if char_file.exists():
        update_team_characteristics(race_results, session, char_file)
    else:
        logger.warning("Team characteristics file not found: %s", char_file)

    data_dir_path = Path(data_dir)
    data_root = data_dir_path.parent if data_dir_path.name == "processed" else data_dir_path
    update_bayesian_driver_ratings(
        race_results,
        qualifying_results=qualifying_results,
        data_root=data_root,
        weather=race_weather,
    )

    logger.info("Race update complete for %s %s", year, race_name)


def update_from_sprint_race(
    year: int,
    race_name: str,
    data_root: str = "data",
) -> None:
    """Update driver ratings from a sprint race result.

    Call this after sprint results are available — typically Saturday.
    It runs independently from update_from_race (Sunday) because the two
    sessions have different timing; the pipeline layer decides the order.

    Sprint races are roughly 1/3 race distance with less strategic variation,
    so the Bayesian update uses a lower confidence weight than full races.
    """
    from src.utils.weekend import is_sprint_weekend

    if not is_sprint_weekend(year, race_name):
        logger.info("%s %s is not a sprint weekend; skipping sprint update", year, race_name)
        return

    try:
        sprint_results, _sprint_session = load_competitive_session(
            year, race_name, "Sprint", load_laps=False
        )
        logger.info("Loaded sprint results for %s drivers", len(sprint_results))
    except (AttributeError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not load sprint results for %s %s: %s",
            year,
            race_name,
            exc,
        )
        return

    # Detect sprint weather for wet_skill updating
    sprint_weather = "dry"
    try:
        weather_data = _sprint_session.weather_data
        if (
            weather_data is not None
            and not weather_data.empty
            and "Rainfall" in weather_data.columns
        ):
            wet_fraction = float(weather_data["Rainfall"].dropna().astype(bool).mean())
            if wet_fraction > 0.30:
                sprint_weather = "rain"
            elif wet_fraction > 0.05:
                sprint_weather = "mixed"
    except Exception:
        sprint_weather = "dry"

    sprint_confidence = float(config_loader.get("bayesian.sprint_race_confidence", 0.20))
    sprint_confidence = float(np.clip(sprint_confidence, 0.05, 0.5))

    data_root_path = Path(data_root)
    store = ArtifactStore(data_root=data_root_path)
    season_year = _coerce_season_year(sprint_results, default_year=year)
    driver_payload = _load_driver_characteristics_payload(store, season_year)

    if not isinstance(driver_payload, dict) or not isinstance(driver_payload.get("drivers"), dict):
        logger.warning("No driver characteristics available for sprint update")
        return

    drivers_payload: dict = driver_payload["drivers"]

    from src.models.priors_factory import PriorsFactory

    factory = PriorsFactory(season_year=season_year)
    priors = factory.create_priors()
    configured_grid_size = int(config_loader.get("grid.size", len(priors) or 22))
    grid_size = max(configured_grid_size, len(priors) or configured_grid_size)

    bayesian = BayesianDriverRanking(priors, grid_size=grid_size)
    strip_legacy_bayesian_fields(drivers_payload)
    _restore_bayesian_state(bayesian, drivers_payload)

    all_observations = _build_position_observations(sprint_results)
    dnf_drivers = _extract_dnf_drivers(sprint_results)
    observations = {dc: pos for dc, pos in all_observations.items() if dc not in dnf_drivers}

    if not observations:
        logger.warning("No valid sprint positions for Bayesian update")
        return

    session_name = f"Sprint_{race_name}"
    from src.utils.lineups import load_current_lineups

    lineups = load_current_lineups()
    if lineups:
        bayesian.update_teammate_relative(
            observations=observations,
            session_name=session_name,
            lineups=lineups,
            confidence=sprint_confidence,
        )
    else:
        bayesian.update(
            observations=observations,
            session_name=session_name,
            confidence=sprint_confidence,
        )

    touched_bayesian = _persist_bayesian_ratings_to_drivers(
        bayesian=bayesian,
        drivers_payload=drivers_payload,
        season_year=season_year,
        fallback_session_name=session_name,
        updated_at=datetime.now().isoformat(),
    )

    # Sprint race pace EMA — lower blend than main race because sprint is ~1/3 distance.
    driver_to_team: dict[str, str] = {}
    for team_name, team_drivers in (lineups or {}).items():
        for d in team_drivers:
            driver_to_team[str(d)] = str(team_name)

    sprint_pace_blend = (
        float(config_loader.get("baseline_predictor.driver_form.race_pace_update_blend", 0.25))
        * 0.5
    )  # Half weight for sprint distance
    sprint_pace_blend = float(np.clip(sprint_pace_blend, 0.0, 0.5))
    touched_pace = _update_teammate_relative_pace_ema(
        observations=observations,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        grid_size=grid_size,
        blend_weight=sprint_pace_blend,
        pace_key="race_pace",
    )

    # Sprint wet_skill EMA — quarter weight of main race
    touched_wet = 0
    sprint_weather_key = normalize_weather_key(sprint_weather)
    if sprint_weather_key in {"rain", "mixed"}:
        wet_blend = (
            float(config_loader.get("baseline_predictor.driver_form.wet_skill_update_blend", 0.15))
            * 0.25
        )  # Quarter weight: sprint is shorter and less representative
        wet_blend = float(np.clip(wet_blend, 0.0, 0.25))
        if sprint_weather_key == "mixed":
            wet_blend *= 0.5

        wet_neutral = float(
            config_loader.get("baseline_predictor.race.lap_time.wet_skill_neutral", 0.70)
        )
        wet_obs_scale = float(
            config_loader.get("baseline_predictor.driver_form.wet_skill_observation_scale", 0.40)
        )

        for driver_code, pos in observations.items():
            driver_entry = drivers_payload.get(driver_code)
            if not isinstance(driver_entry, dict):
                continue
            raw_pace = float(np.clip(1.0 - ((int(pos) - 1) / max(grid_size - 1, 1)), 0.0, 1.0))
            driver_team = driver_to_team.get(driver_code)
            team_paces = [
                float(np.clip(1.0 - ((int(tp) - 1) / max(grid_size - 1, 1)), 0.0, 1.0))
                for tdc, tp in observations.items()
                if driver_to_team.get(tdc) == driver_team and tdc != driver_code
            ]
            if not team_paces:
                continue
            relative_performance = raw_pace - float(np.mean(team_paces))
            observed_signal = wet_neutral + (relative_performance * wet_obs_scale)
            observed_signal = float(np.clip(observed_signal, 0.40, 0.95))
            existing = driver_entry.get("wet_skill")
            existing_val = float(existing if existing is not None else wet_neutral)
            updated = (1.0 - wet_blend) * existing_val + wet_blend * observed_signal
            driver_entry["wet_skill"] = round(updated, 3)
            touched_wet += 1

        if touched_wet > 0:
            logger.info("Sprint wet_skill update: %d drivers", touched_wet)

    sprint_dnf_blend = (
        float(config_loader.get("baseline_predictor.driver_form.dnf_rate_update_blend", 0.10)) * 0.5
    )
    touched_dnf_rate = _update_dnf_rate_ema(
        session_results=sprint_results,
        drivers_payload=drivers_payload,
        blend_weight=sprint_dnf_blend,
        floor=float(config_loader.get("baseline_predictor.driver_form.dnf_rate_floor", 0.02)),
        cap=float(config_loader.get("baseline_predictor.driver_form.dnf_rate_cap", 0.35)),
    )

    _persist_driver_characteristics_payload(store, driver_payload, season_year)
    logger.info(
        "Sprint update applied for %s drivers: Bayesian (confidence=%.2f), "
        "persisted ratings (%d), race_pace EMA (%d drivers, blend=%.2f), "
        "wet_skill (%d drivers), dnf_rate (%d drivers)",
        len(observations),
        sprint_confidence,
        touched_bayesian,
        touched_pace,
        sprint_pace_blend,
        touched_wet,
        touched_dnf_rate,
    )
