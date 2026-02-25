"""Core orchestration helpers for race updater team-characteristics flow."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CharacteristicPayload = dict[str, Any]
RacePaceMap = dict[str, float]
CompoundMetricsByTeam = dict[str, dict[str, Any]]

MapTeamToCharacteristicsFn = Callable[[Any, set[str]], str | None]
ArtifactStoreFactory = Callable[..., Any]
ExtractTeamPerformanceFn = Callable[[Any, list[str]], RacePaceMap]
ExtractCompoundMetricsFn = Callable[[pd.DataFrame, str, str], dict[str, Any] | None]
NormalizeCompoundMetricsFn = Callable[[CompoundMetricsByTeam, str], CompoundMetricsByTeam]
AggregateCompoundSamplesFn = Callable[..., dict[str, Any]]
ConfigGetFn = Callable[[str, Any], Any]


def _load_characteristics_payload(
    *,
    characteristics_file: Path,
    artifact_store_factory: ArtifactStoreFactory,
    logger: logging.Logger,
) -> tuple[Any, str, CharacteristicPayload]:
    """Load characteristics payload from artifact store with file fallback."""
    store = artifact_store_factory(data_root=characteristics_file.parent.parent.parent)
    year = characteristics_file.stem.split("_")[0]
    char_data = store.load_artifact(
        artifact_type="car_characteristics",
        artifact_key=f"{year}::car_characteristics",
    )

    if not char_data:
        logger.warning("DB load failed, falling back to file")
        with open(characteristics_file) as f:
            char_data = json.load(f)

    return store, year, char_data


def _build_position_fallback_race_pace(
    *,
    race_results: pd.DataFrame,
    team_names: list[str],
    map_team_to_characteristics_fn: MapTeamToCharacteristicsFn,
    logger: logging.Logger,
) -> RacePaceMap:
    """Fallback race pace estimate using finishing positions."""
    logger.warning("No telemetry data, using positions as fallback")
    race_pace: dict[str, float] = {}
    known_teams = set(team_names)
    canonical_results = race_results.copy()
    if "TeamName" in canonical_results.columns:
        canonical_results["_canonical_team"] = canonical_results["TeamName"].apply(
            lambda raw: map_team_to_characteristics_fn(raw, known_teams=known_teams)
        )
    else:
        canonical_results["_canonical_team"] = None

    for team in team_names:
        team_results = canonical_results[canonical_results["_canonical_team"] == team]
        if len(team_results) > 0:
            positions = pd.to_numeric(team_results["Position"], errors="coerce").dropna()
            if positions.empty:
                continue
            avg_position = positions.mean()
            race_pace[team] = 1.0 - (avg_position - 1) / 19

    return race_pace


def _apply_team_performance_updates(
    *,
    char_data: CharacteristicPayload,
    race_pace: RacePaceMap,
    logger: logging.Logger,
    now_iso: str,
) -> None:
    """Apply race-performance updates to team payload."""
    for team, new_performance in race_pace.items():
        if team in char_data["teams"]:
            team_data = char_data["teams"][team]

            if "current_season_performance" not in team_data:
                team_data["current_season_performance"] = []

            team_data["current_season_performance"].append(new_performance)

            running_avg = np.mean(team_data["current_season_performance"])
            old_uncertainty = team_data["uncertainty"]
            updated_uncertainty = max(0.10, old_uncertainty * 0.9)

            team_data["uncertainty"] = round(updated_uncertainty, 3)
            team_data["last_updated"] = now_iso
            team_data["races_completed"] = len(team_data["current_season_performance"])

            logger.info(
                f"  {team}: Race {new_performance:.3f} → Avg {running_avg:.3f} "
                f"({team_data['races_completed']} races, uncertainty {old_uncertainty:.2f}→{updated_uncertainty:.2f})"
            )


def _resolve_race_name(session: Any) -> str:
    """Resolve race name from session metadata with safe fallback."""
    try:
        return session.event["EventName"]
    except Exception:
        return getattr(session, "name", None) or "Unknown Race"


def _extract_normalized_compound_metrics(
    *,
    session: Any,
    team_names: list[str],
    race_name: str,
    map_team_to_characteristics_fn: MapTeamToCharacteristicsFn,
    extract_compound_metrics_fn: ExtractCompoundMetricsFn,
    normalize_compound_metrics_across_teams_fn: NormalizeCompoundMetricsFn,
) -> CompoundMetricsByTeam:
    """Extract and normalize compound metrics for teams from race session laps."""
    laps = session.laps
    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    known_teams = set(team_names)
    race_compound_metrics = {}
    raw_teams = laps["Team"].dropna().unique()

    for raw_team in raw_teams:
        canonical_team = map_team_to_characteristics_fn(str(raw_team), known_teams=known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        compound_data = extract_compound_metrics_fn(team_laps, canonical_team, race_name)
        if compound_data:
            race_compound_metrics[canonical_team] = compound_data

    if not race_compound_metrics:
        return {}

    return normalize_compound_metrics_across_teams_fn(race_compound_metrics, race_name)


def _apply_compound_metric_updates(
    *,
    char_data: CharacteristicPayload,
    normalized_compound_metrics: CompoundMetricsByTeam,
    race_name: str,
    now_iso: str,
    config_get_fn: ConfigGetFn,
    aggregate_compound_samples_fn: AggregateCompoundSamplesFn,
    logger: logging.Logger,
) -> None:
    """Blend normalized race compound metrics into persisted team characteristics."""
    if not normalized_compound_metrics:
        return

    for team_name, new_compounds in normalized_compound_metrics.items():
        if team_name in char_data["teams"]:
            team_data = char_data["teams"][team_name]
            existing_compound_chars = team_data.get("compound_characteristics")
            if not isinstance(existing_compound_chars, dict):
                existing_compound_chars = {}

            race_blend_weight = config_get_fn(
                "baseline_predictor.compound_blend_weights.race",
                0.70,
            )
            blended_compounds = aggregate_compound_samples_fn(
                existing_compound_chars,
                new_compounds,
                blend_weight=race_blend_weight,
                race_name=race_name,
            )

            for compound_payload in blended_compounds.values():
                compound_payload["last_updated"] = now_iso

            team_data["compound_characteristics"] = blended_compounds
            logger.info(f"  {team_name}: Updated {len(blended_compounds)} compound characteristics")

    logger.info(f"Updated compound characteristics for {len(normalized_compound_metrics)} teams")


def _create_backup_if_needed(*, characteristics_file: Path, logger: logging.Logger) -> None:
    """Create backup of characteristics file before writing updates."""
    if characteristics_file.exists():
        backup_file = Path(str(characteristics_file) + ".backup")
        shutil.copy2(characteristics_file, backup_file)
        logger.debug(f"Created backup at {backup_file}")


def _save_characteristics_payload(
    *,
    store: Any,
    year: str,
    char_data: CharacteristicPayload,
    characteristics_file: Path,
    logger: logging.Logger,
) -> None:
    """Persist characteristics via artifact store with file fallback."""
    current_version = char_data.get("version", 0)
    new_version = current_version + 1
    char_data["version"] = new_version

    _create_backup_if_needed(characteristics_file=characteristics_file, logger=logger)
    try:
        store.save_artifact(
            artifact_type="car_characteristics",
            artifact_key=f"{year}::car_characteristics",
            data=char_data,
            version=new_version,
        )
        logger.info(f"Updated team characteristics (v{new_version}) via ArtifactStore")
    except Exception as exc:
        logger.error(f"ArtifactStore save failed: {exc}, falling back to file")
        with open(characteristics_file, "w") as f:
            json.dump(char_data, f, indent=2)
        logger.info(f"Updated team characteristics (v{new_version}) in {characteristics_file}")


def update_team_characteristics_core(
    *,
    race_results: pd.DataFrame,
    session: Any,
    characteristics_file: Path,
    artifact_store_factory: ArtifactStoreFactory,
    extract_team_performance_from_telemetry_fn: ExtractTeamPerformanceFn,
    map_team_to_characteristics_fn: MapTeamToCharacteristicsFn,
    extract_compound_metrics_fn: ExtractCompoundMetricsFn,
    normalize_compound_metrics_across_teams_fn: NormalizeCompoundMetricsFn,
    aggregate_compound_samples_fn: AggregateCompoundSamplesFn,
    config_get_fn: ConfigGetFn,
    logger: logging.Logger,
) -> None:
    """Update team performance ratings and compound metrics from race telemetry."""
    logger.info("Updating team characteristics from race telemetry...")
    store, year, char_data = _load_characteristics_payload(
        characteristics_file=characteristics_file,
        artifact_store_factory=artifact_store_factory,
        logger=logger,
    )

    team_names = list(char_data["teams"].keys())
    race_pace = extract_team_performance_from_telemetry_fn(session, team_names)
    if not race_pace:
        race_pace = _build_position_fallback_race_pace(
            race_results=race_results,
            team_names=team_names,
            map_team_to_characteristics_fn=map_team_to_characteristics_fn,
            logger=logger,
        )

    now_iso = datetime.now().isoformat()
    _apply_team_performance_updates(
        char_data=char_data,
        race_pace=race_pace,
        logger=logger,
        now_iso=now_iso,
    )

    race_name = _resolve_race_name(session)
    logger.info("Extracting compound-specific performance from race...")
    try:
        normalized_compound_metrics = _extract_normalized_compound_metrics(
            session=session,
            team_names=team_names,
            race_name=race_name,
            map_team_to_characteristics_fn=map_team_to_characteristics_fn,
            extract_compound_metrics_fn=extract_compound_metrics_fn,
            normalize_compound_metrics_across_teams_fn=normalize_compound_metrics_across_teams_fn,
        )
        _apply_compound_metric_updates(
            char_data=char_data,
            normalized_compound_metrics=normalized_compound_metrics,
            race_name=race_name,
            now_iso=now_iso,
            config_get_fn=config_get_fn,
            aggregate_compound_samples_fn=aggregate_compound_samples_fn,
            logger=logger,
        )
    except Exception as exc:
        logger.warning(f"Failed to extract compound metrics from race: {exc}")

    char_data["last_updated"] = now_iso
    char_data["data_freshness"] = "LIVE_UPDATED"
    char_data["races_completed"] = char_data.get("races_completed", 0) + 1

    _save_characteristics_payload(
        store=store,
        year=year,
        char_data=char_data,
        characteristics_file=characteristics_file,
        logger=logger,
    )
