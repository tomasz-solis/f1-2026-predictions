"""Shared helpers for race updater team-characteristics flow."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Sequence
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


def extract_dnf_drivers(race_results: pd.DataFrame) -> set[str]:
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


def _build_position_fallback_race_pace(
    *,
    race_results: pd.DataFrame,
    team_names: list[str],
    map_team_to_characteristics_fn: MapTeamToCharacteristicsFn,
    logger: logging.Logger,
) -> RacePaceMap:
    """Fallback race pace estimate from where teams finished, keeping the margin.

    Scores each team on the grid scale rather than by rank. Rank rescaled the field
    to a fixed 1.0..0.0 spread every race, so the best car always scored 1.0 and the
    worst always 0.0 no matter how large the gap was. An upgrade that halved a team's
    deficit scored identically to the weekend before unless it changed their rank,
    which is why in-season car improvements could never register.

    DNFs are excluded. They are excluded from the Bayesian driver update for the same
    reason, and margin scoring is far more sensitive to them than rank was: one
    retirement classified near the back drags a front team's mean position by ten
    places, where under rank it cost at most a place or two.
    """
    logger.warning("No telemetry data, using positions as fallback")
    known_teams = set(team_names)
    canonical_results = race_results.copy()
    if "TeamName" in canonical_results.columns:
        canonical_results["_canonical_team"] = canonical_results["TeamName"].apply(
            lambda raw: map_team_to_characteristics_fn(raw, known_teams)
        )
    else:
        canonical_results["_canonical_team"] = None

    dnf_drivers = extract_dnf_drivers(race_results)
    if dnf_drivers and "Abbreviation" in canonical_results.columns:
        classified = canonical_results[~canonical_results["Abbreviation"].isin(dnf_drivers)]
        # Only drop retirements when doing so still leaves a field to score against.
        if not classified.empty:
            canonical_results = classified

    all_positions = pd.to_numeric(canonical_results["Position"], errors="coerce").dropna()
    if all_positions.empty:
        return {}
    # Scale against the cars that started, not the ones still classified. These scores
    # are averaged across a season, so the scale has to mean the same thing every race:
    # scaling by the classified count would stretch a high-attrition race and pin the
    # last surviving car at 0.0, which is the rank flattening this fix removes.
    entered_field = float(len(race_results.index))
    field_size = max(entered_field, float(all_positions.max()), 2.0)

    race_pace: dict[str, float] = {}
    for team in team_names:
        team_results = canonical_results[canonical_results["_canonical_team"] == team]
        if len(team_results) == 0:
            continue
        positions = pd.to_numeric(team_results["Position"], errors="coerce").dropna()
        if positions.empty:
            continue
        mean_position = float(positions.mean())
        race_pace[team] = float(
            np.clip(1.0 - ((mean_position - 1.0) / (field_size - 1.0)), 0.0, 1.0)
        )

    return race_pace


def _recency_weighted_mean(observations: Sequence[float], *, recency_exponent: float) -> float:
    """Average season observations with later races weighted more heavily.

    Matches the weighting the predictor already applies to saved-actual form in
    ``data_mixin._resolve_saved_actual_team_score``: weight race ``i`` (1-based) by
    ``i ** recency_exponent``. A flat mean let five stale rounds outvote the most
    recent weekend, so an in-season car upgrade could never move the baseline.
    """
    values = [float(value) for value in observations]
    if not values:
        return 0.0
    if len(values) == 1 or recency_exponent == 0.0:
        return float(np.mean(values))
    weights = np.power(np.arange(1, len(values) + 1, dtype=float), recency_exponent)
    return float(np.average(values, weights=weights))


def _apply_team_performance_updates(
    *,
    char_data: CharacteristicPayload,
    race_pace: RacePaceMap,
    config_get_fn: ConfigGetFn,
    logger: logging.Logger,
    now_iso: str,
) -> None:
    """Apply race-performance updates to team payload."""
    baseline_learning_rate = float(
        np.clip(
            config_get_fn("baseline_predictor.baseline_learning_rate", 0.3),
            0.0,
            1.0,
        )
    )
    recency_exponent = max(
        0.0,
        float(config_get_fn("baseline_predictor.current_season_form.recency_exponent", 1.5)),
    )
    for team, new_performance in race_pace.items():
        if team in char_data["teams"]:
            team_data = char_data["teams"][team]

            if "current_season_performance" not in team_data:
                team_data["current_season_performance"] = []

            team_data["current_season_performance"].append(new_performance)

            running_avg = _recency_weighted_mean(
                team_data["current_season_performance"],
                recency_exponent=recency_exponent,
            )
            old_uncertainty = team_data["uncertainty"]
            updated_uncertainty = max(0.10, old_uncertainty * 0.9)
            old_baseline = float(team_data.get("overall_performance", 0.5))
            team_data.setdefault("preseason_overall_performance", round(old_baseline, 4))
            updated_baseline = old_baseline + (
                baseline_learning_rate * (float(running_avg) - old_baseline)
            )

            team_data["uncertainty"] = round(updated_uncertainty, 3)
            team_data["overall_performance"] = round(float(updated_baseline), 4)
            team_data["last_updated"] = now_iso
            team_data["races_completed"] = len(team_data["current_season_performance"])

            logger.info(
                "  %s: Race %s → Avg %s (baseline %s→%s, %s races, uncertainty %s→%s)",
                team,
                format(new_performance, ".3f"),
                format(running_avg, ".3f"),
                format(old_baseline, ".3f"),
                format(updated_baseline, ".3f"),
                team_data["races_completed"],
                format(old_uncertainty, ".2f"),
                format(updated_uncertainty, ".2f"),
            )


def _refresh_team_race_count_notes(*, char_data: CharacteristicPayload, year: int) -> None:
    """Keep generated team notes aligned with the stored race-count evidence."""
    teams = char_data.get("teams", {})
    if not isinstance(teams, dict):
        return

    for team_data in teams.values():
        if not isinstance(team_data, dict):
            continue
        note = str(team_data.get("note", "")).strip()
        marker = " updated with "
        if marker not in note or " race(s)" not in note:
            continue
        race_count = len(team_data.get("current_season_performance", []))
        prefix = note.split(marker, 1)[0].strip()
        if not prefix:
            continue
        team_data["note"] = f"{prefix}{marker}{race_count} race(s) of {int(year)} data"


def _resolve_race_name(session: Any) -> str:
    """Resolve race name from session metadata with safe fallback."""
    try:
        return session.event["EventName"]
    except (AttributeError, KeyError, TypeError):
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
        canonical_team = map_team_to_characteristics_fn(str(raw_team), known_teams)
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
            logger.info(
                "  %s: Updated %s compound characteristics", team_name, len(blended_compounds)
            )

    logger.info("Updated compound characteristics for %s teams", len(normalized_compound_metrics))


def _create_backup_if_needed(*, characteristics_file: Path, logger: logging.Logger) -> None:
    """Create backup of characteristics file before writing updates."""
    if characteristics_file.exists():
        backup_file = Path(str(characteristics_file) + ".backup")
        shutil.copy2(characteristics_file, backup_file)
        logger.debug("Created backup at %s", backup_file)


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
        logger.info("Updated team characteristics (v%s) via ArtifactStore", new_version)
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.error("ArtifactStore save failed: %s, falling back to file", exc)
        with open(characteristics_file, "w") as f:
            json.dump(char_data, f, indent=2)
        logger.info("Updated team characteristics (v%s) in %s", new_version, characteristics_file)


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
    if not race_pace or len(race_pace) < len(team_names):
        fallback_race_pace = _build_position_fallback_race_pace(
            race_results=race_results,
            team_names=team_names,
            map_team_to_characteristics_fn=map_team_to_characteristics_fn,
            logger=logger,
        )
        if not race_pace:
            race_pace = fallback_race_pace
        elif len(fallback_race_pace) >= len(race_pace):
            logger.warning(
                "Incomplete telemetry coverage (%s/%s teams); using position fallback for a "
                "consistent full-field ranking",
                len(race_pace),
                len(team_names),
            )
            race_pace = fallback_race_pace

    now_iso = datetime.now().isoformat()
    _apply_team_performance_updates(
        char_data=char_data,
        race_pace=race_pace,
        config_get_fn=config_get_fn,
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
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Failed to extract compound metrics from race: %s", exc)

    char_data["last_updated"] = now_iso
    char_data["data_freshness"] = "LIVE_UPDATED"
    char_data["races_completed"] = max(
        (
            len(team_data.get("current_season_performance", []))
            for team_data in char_data.get("teams", {}).values()
            if isinstance(team_data, dict)
        ),
        default=int(char_data.get("races_completed", 0)),
    )
    _refresh_team_race_count_notes(char_data=char_data, year=int(year))

    _save_characteristics_payload(
        store=store,
        year=year,
        char_data=char_data,
        characteristics_file=characteristics_file,
        logger=logger,
    )
