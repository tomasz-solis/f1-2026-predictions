"""
Race Data Updater System

Adaptive learning after each race:
- Updates team performance from telemetry
- Updates Bayesian driver ratings
- Reduces uncertainty as season progresses
"""

import logging
from datetime import datetime
from pathlib import Path

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
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)


def _driver_characteristics_fallback_paths(year: int) -> tuple[Path, ...]:
    """
    Return fallback file paths for driver characteristics in priority order.

    First preference is season-scoped fallback to avoid cross-year contamination.
    Legacy unscoped path is read-only compatibility.
    """
    processed_root = Path("data/processed")
    return (
        processed_root / "driver_characteristics" / f"{year}_driver_characteristics.json",
        processed_root / "driver_characteristics.json",
    )


def _coerce_season_year(race_results: pd.DataFrame, default_year: int = 2026) -> int:
    """Resolve season year from race results payload with safe fallback."""
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
    """Load driver characteristics from artifact store with file fallback."""
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
            import json

            with open(fallback_file) as f:
                fallback_payload = json.load(f)
            logger.info(
                "Loaded driver characteristics fallback from %s for season %s",
                fallback_file,
                year,
            )
            return fallback_payload
        except Exception as exc:
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
    """Persist updated driver characteristics payload with artifact-store fallback."""
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
    except Exception:
        latest_store_version = 0
    new_version = max(current_version, latest_store_version) + 1

    payload["version"] = new_version
    payload["last_updated"] = datetime.now().isoformat()
    payload["bayesian_last_updated_year"] = year

    def _write_fallback_file() -> None:
        """Persist the same payload to the season-scoped JSON fallback."""
        import json

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
    except Exception as exc:
        logger.warning(
            "ArtifactStore save failed for driver characteristics: %s. "
            "Falling back to season-scoped file %s.",
            exc,
            fallback_file,
        )
        _write_fallback_file()


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
    except Exception as exc:
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
            logger.warning(f"  No laps found for {team}")
            continue

        # Filter valid racing laps
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
            logger.warning(f"  {team}: Only {len(valid_laps)} valid laps, skipping")
            continue

        # Get lap times in seconds
        lap_times_seconds = valid_laps["LapTime"].dt.total_seconds()

        # Remove outliers (>3 std devs)
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
        logger.debug(f"  {team}: Median lap time {median_time:.3f}s ({len(clean_times)} laps)")

    # Convert lap times to rank-based 0-1 performance scale.
    # Rank-based scoring is more stable than min-max because a single outlier
    # cannot stretch the whole field and flatten the midfield signal.
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


def update_bayesian_driver_ratings(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame | None = None,
) -> None:
    """Update Bayesian driver ratings plus qualifying pace from completed sessions."""
    logger.info("Updating Bayesian driver ratings...")

    # Create priors for drivers
    from src.models.priors_factory import PriorsFactory

    season_year = _coerce_season_year(race_results)
    factory = PriorsFactory(season_year=season_year)
    priors = factory.create_priors()
    configured_grid_size = int(config_loader.get("grid.size", len(priors) or 22))
    grid_size = max(configured_grid_size, len(priors) or configured_grid_size)

    bayesian = BayesianDriverRanking(priors, grid_size=grid_size)
    observations = _build_position_observations(race_results)
    if not observations:
        logger.warning("No valid race positions available for Bayesian rating update")
        return

    session_name = str(race_results.get("race_name", pd.Series(["Race"])).iloc[0])
    bayesian.update(observations=observations, session_name=session_name, confidence=1.0)

    store = ArtifactStore(data_root="data")
    driver_payload = _load_driver_characteristics_payload(store, season_year)
    if not isinstance(driver_payload, dict):
        logger.warning("Could not load driver characteristics payload to persist Bayesian updates")
        return

    drivers_payload = driver_payload.get("drivers")
    if not isinstance(drivers_payload, dict):
        logger.warning(
            "Driver characteristics payload missing 'drivers'; skipping Bayesian persistence"
        )
        return

    blend_weight = float(config_loader.get("bayesian.runtime_skill_blend_weight", 0.25))
    blend_weight = float(max(0.0, min(blend_weight, 1.0)))
    touched_skill_drivers = 0
    for driver_code, (mu, sigma) in bayesian.ratings.items():
        driver_entry = drivers_payload.get(driver_code)
        if not isinstance(driver_entry, dict):
            continue

        racecraft_payload = driver_entry.get("racecraft")
        if not isinstance(racecraft_payload, dict):
            racecraft_payload = {}
            driver_entry["racecraft"] = racecraft_payload

        existing_skill_raw = racecraft_payload.get("skill_score", 0.5)
        try:
            existing_skill = float(existing_skill_raw)
        except (TypeError, ValueError):
            existing_skill = 0.5
        existing_skill = float(max(0.0, min(existing_skill, 1.0)))

        bayesian_skill = float(max(0.0, min((float(mu) - 1.0) / max(grid_size - 1, 1), 1.0)))
        blended_skill = ((1.0 - blend_weight) * existing_skill) + (blend_weight * bayesian_skill)
        blended_skill = float(max(0.0, min(blended_skill, 1.0)))
        racecraft_payload["skill_score"] = blended_skill

        driver_entry["bayesian"] = {
            "rating_mu": float(mu),
            "rating_sigma": float(sigma),
            "normalized_skill_score": bayesian_skill,
            "blended_skill_score": blended_skill,
            "blend_weight": blend_weight,
            "last_session": session_name,
            "last_updated": datetime.now().isoformat(),
            "season_year": season_year,
        }
        touched_skill_drivers += 1

    qualifying_observations = (
        _build_position_observations(qualifying_results)
        if isinstance(qualifying_results, pd.DataFrame)
        else {}
    )
    quali_pace_blend = float(
        config_loader.get("baseline_predictor.driver_form.quali_pace_update_blend", 0.30)
    )
    quali_pace_blend = float(np.clip(quali_pace_blend, 0.0, 1.0))
    touched_quali_pace_drivers = 0
    for driver_code, quali_position in qualifying_observations.items():
        driver_entry = drivers_payload.get(driver_code)
        if not isinstance(driver_entry, dict):
            continue

        pace_payload = driver_entry.get("pace")
        if not isinstance(pace_payload, dict):
            pace_payload = {}
            driver_entry["pace"] = pace_payload

        try:
            existing_quali_pace = float(pace_payload.get("quali_pace", 0.5))
        except (TypeError, ValueError):
            existing_quali_pace = 0.5
        existing_quali_pace = float(np.clip(existing_quali_pace, 0.05, 0.99))

        observed_quali_pace = 1.0 - ((int(quali_position) - 1) / max(grid_size - 1, 1))
        observed_quali_pace = float(np.clip(observed_quali_pace, 0.0, 1.0))
        updated_quali_pace = ((1.0 - quali_pace_blend) * existing_quali_pace) + (
            quali_pace_blend * observed_quali_pace
        )
        pace_payload["quali_pace"] = round(float(np.clip(updated_quali_pace, 0.05, 0.99)), 3)
        touched_quali_pace_drivers += 1

    if touched_skill_drivers == 0 and touched_quali_pace_drivers == 0:
        logger.warning("Bayesian update produced no persisted driver changes")
        return

    _persist_driver_characteristics_payload(store, driver_payload, season_year)
    logger.info(
        "Updated Bayesian ratings for %s drivers, blended skill for %s drivers, and refreshed "
        "qualifying pace for %s drivers",
        len(observations),
        touched_skill_drivers,
        touched_quali_pace_drivers,
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
    logger.info(f"Updating from {year} {race_name}")
    logger.info("=" * 60)

    try:
        race_results, session = load_race_session(year, race_name)
        logger.info(f"Loaded results for {len(race_results)} drivers\n")
    except Exception as e:
        logger.error(f"Failed to load race results: {e}")
        logger.error("Make sure race has completed and data is available via FastF1")
        raise

    try:
        qualifying_results, _qualifying_session = load_qualifying_session(year, race_name)
        logger.info("Loaded qualifying results for %s drivers", len(qualifying_results))
    except Exception as exc:
        logger.warning(
            "Could not load qualifying results for %s %s; skipping quali pace refresh (%s)",
            year,
            race_name,
            exc,
        )
        qualifying_results = None

    # Update team characteristics
    char_file = Path(data_dir) / "car_characteristics" / f"{year}_car_characteristics.json"
    if char_file.exists():
        update_team_characteristics(race_results, session, char_file)
    else:
        logger.warning(f"Team characteristics file not found: {char_file}")

    # Update driver ratings
    update_bayesian_driver_ratings(race_results, qualifying_results=qualifying_results)

    logger.info("\n" + "=" * 60)
    logger.info("Race update complete.")
    logger.info("=" * 60)
    logger.info("\nSystem learned from this race:")
    logger.info("- Team performance updated from telemetry")
    logger.info("- Driver skill confidence increased")
    logger.info("- Uncertainty reduced")
    logger.info("- Version incremented\n")
    logger.info("Next prediction will use these updated characteristics.")
