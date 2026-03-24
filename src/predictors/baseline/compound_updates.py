"""Compound-characteristic update helpers for baseline predictors."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("src.predictors.baseline_2026")


def update_compound_characteristics_from_session(
    *,
    context: Any,
    session_laps: pd.DataFrame,
    race_name: str,
    year: int,
    is_sprint: bool,
    cfg: Any,
) -> None:
    """Extract, blend, cache, and optionally persist session compound characteristics."""
    cache_key = (race_name, year, len(session_laps))
    if cache_key in context._compound_cache:
        logger.debug(f"Using cached compound metrics for {race_name} ({len(session_laps)} laps)")
        cached_compounds = context._compound_cache[cache_key]
        for team_name, compounds in cached_compounds.items():
            if team_name in context.teams:
                context.teams[team_name]["compound_characteristics"] = compounds
        return

    from src.systems.compound_analyzer import (
        aggregate_compound_samples,
        extract_compound_metrics,
        normalize_compound_metrics_across_teams,
    )
    from src.utils.team_mapping import map_team_to_characteristics

    logger.info(f"Extracting compound metrics from session for {race_name}...")

    race_compound_metrics: dict[str, dict[str, Any]] = {}
    known_teams = set(context.teams.keys())

    for raw_team in session_laps["Team"].unique():
        if pd.isna(raw_team):
            continue

        canonical_team = map_team_to_characteristics(str(raw_team), known_teams=known_teams)
        if not canonical_team:
            continue

        team_laps = session_laps[session_laps["Team"] == raw_team]
        compound_data = extract_compound_metrics(team_laps, canonical_team, race_name)
        if compound_data:
            race_compound_metrics[canonical_team] = compound_data

    if not race_compound_metrics:
        logger.debug("No compound metrics extracted from session")
        return

    normalized_compound_metrics = normalize_compound_metrics_across_teams(
        race_compound_metrics,
        race_name,
    )
    if is_sprint:
        blend_weight = cfg.get("baseline_predictor.compound_blend_weights.sprint", 0.50)
    else:
        blend_weight = cfg.get("baseline_predictor.compound_blend_weights.practice", 0.30)

    for team_name, new_compounds in normalized_compound_metrics.items():
        if team_name not in context.teams:
            continue

        existing_compound_chars = context.teams[team_name].get("compound_characteristics", {})
        if not isinstance(existing_compound_chars, dict):
            existing_compound_chars = {}

        context.teams[team_name]["compound_characteristics"] = aggregate_compound_samples(
            existing_compound_chars,
            new_compounds,
            blend_weight=blend_weight,
            race_name=race_name,
        )

    context._compound_cache[cache_key] = {
        team: context.teams[team].get("compound_characteristics", {})
        for team in normalized_compound_metrics
        if team in context.teams
    }

    store = getattr(context, "artifact_store", None)
    storage_mode = getattr(store, "storage_mode", "file_only") if store else "file_only"
    if store and storage_mode in {"db_only", "fallback", "dual_write"}:
        try:
            season_year = int(getattr(context, "season_year", getattr(context, "year", 2026)))
            artifact_key = f"{season_year}::car_characteristics"
            car_data = store.load_artifact("car_characteristics", artifact_key)
            if car_data:
                for team_name in normalized_compound_metrics:
                    if team_name in car_data.get("teams", {}):
                        car_data["teams"][team_name]["compound_characteristics"] = context.teams[
                            team_name
                        ].get("compound_characteristics", {})
                store.save_artifact("car_characteristics", artifact_key, car_data)
                logger.debug(
                    "Persisted compound characteristics for "
                    f"{len(normalized_compound_metrics)} teams to DB"
                )
        except Exception as exc:
            logger.warning(f"Failed to persist compound characteristics to DB: {exc}")
    else:
        logger.debug("Skipping DB persistence (file-only mode or no artifact store)")

    logger.info(
        f"Updated and cached compound characteristics for {len(normalized_compound_metrics)} teams "
        f"(blend_weight={blend_weight:.0%})"
    )
