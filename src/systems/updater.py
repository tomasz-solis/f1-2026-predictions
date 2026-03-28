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

    try:
        store.save_artifact(
            artifact_type="driver_characteristics",
            artifact_key=artifact_key,
            data=payload,
            version=new_version,
        )
    except Exception as exc:
        logger.warning(
            "ArtifactStore save failed for driver characteristics: %s. "
            "Falling back to season-scoped file %s.",
            exc,
            fallback_file,
        )
        import json

        with open(fallback_file, "w") as f:
            json.dump(payload, f, indent=2)


def load_race_session(year: int, race_name: str) -> tuple[pd.DataFrame, fastf1.core.Session]:
    """Load race results and session from FastF1."""
    logger.info(f"Loading {year} {race_name} results...")

    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, race_name, "R")
    session.load(laps=True, telemetry=False, weather=False)

    results = session.results
    results["race_name"] = race_name
    results["year"] = year

    return results, session


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

    if not hasattr(session, "laps") or session.laps is None or session.laps.empty:
        logger.warning("No lap data available")
        return {}

    laps = session.laps
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

    # Convert lap times to 0-1 performance scale
    if race_pace:
        fastest_time = min(race_pace.values())
        slowest_time = max(race_pace.values())

        if fastest_time < slowest_time:
            for team in race_pace:
                # Invert: faster time = higher score
                performance = 1.0 - (race_pace[team] - fastest_time) / (slowest_time - fastest_time)
                race_pace[team] = performance
        else:
            # All teams same pace
            for team in race_pace:
                race_pace[team] = 0.5

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


def update_bayesian_driver_ratings(race_results: pd.DataFrame) -> None:
    """Update Bayesian driver skill ratings from race results."""
    logger.info("Updating Bayesian driver ratings...")

    # Create priors for drivers
    from src.models.priors_factory import PriorsFactory

    season_year = _coerce_season_year(race_results)
    factory = PriorsFactory(season_year=season_year)
    priors = factory.create_priors()
    configured_grid_size = int(config_loader.get("grid.size", len(priors) or 22))
    grid_size = max(configured_grid_size, len(priors) or configured_grid_size)

    bayesian = BayesianDriverRanking(priors, grid_size=grid_size)
    observations: dict[str, int] = {}
    for driver, position in zip(
        race_results["Abbreviation"].tolist(),
        race_results["Position"].tolist(),
        strict=False,
    ):
        if pd.notna(position):
            observations[str(driver)] = int(position)

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
    touched_drivers = 0
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
        touched_drivers += 1

    if touched_drivers == 0:
        logger.warning("Bayesian update produced no persisted driver changes")
        return

    _persist_driver_characteristics_payload(store, driver_payload, season_year)
    logger.info(
        "Updated Bayesian ratings for %s drivers and persisted %s driver profile updates",
        len(observations),
        touched_drivers,
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

    # Update team characteristics
    char_file = Path(data_dir) / "car_characteristics" / f"{year}_car_characteristics.json"
    if char_file.exists():
        update_team_characteristics(race_results, session, char_file)
    else:
        logger.warning(f"Team characteristics file not found: {char_file}")

    # Update driver ratings
    update_bayesian_driver_ratings(race_results)

    logger.info("\n" + "=" * 60)
    logger.info("Race update complete.")
    logger.info("=" * 60)
    logger.info("\nSystem learned from this race:")
    logger.info("- Team performance updated from telemetry")
    logger.info("- Driver skill confidence increased")
    logger.info("- Uncertainty reduced")
    logger.info("- Version incremented\n")
    logger.info("Next prediction will use these updated characteristics.")
