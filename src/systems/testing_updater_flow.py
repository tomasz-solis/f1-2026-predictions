"""Orchestration helpers for testing updater session collection and persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionCollectionResult:
    """Accumulated extracted metrics across all requested events/sessions."""

    metric_samples: dict[str, dict[str, list[tuple[float, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    profile_metric_samples: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = field(
        default_factory=dict
    )
    team_sessions_used: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    team_profile_sessions_used: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    loaded_sessions: list[str] = field(default_factory=list)
    discovered_sessions: list[str] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)
    extraction_diagnostics: list[str] = field(default_factory=list)
    compound_metrics_by_session: dict[
        str, tuple[str, dict[str, dict[str, dict[str, float | str | None]]]]
    ] = field(default_factory=dict)


def load_characteristics_payload(data_dir: str, target_year: int) -> tuple[Path, dict]:
    """Load characteristics JSON and validate required top-level keys."""
    characteristics_file = (
        Path(data_dir) / "car_characteristics" / f"{target_year}_car_characteristics.json"
    )
    if not characteristics_file.exists():
        raise FileNotFoundError(f"Characteristics file not found: {characteristics_file}")

    with open(characteristics_file) as f:
        characteristics = json.load(f)

    if "teams" not in characteristics:
        raise ValueError(
            f"Invalid characteristics format in {characteristics_file}: missing 'teams'"
        )

    return characteristics_file, characteristics


def validate_update_options(
    session_aggregation: str,
    run_profile: str,
    session_aggregation_modes: tuple[str, ...],
    run_profile_modes: tuple[str, ...],
) -> None:
    """Validate update mode options before any extraction work starts."""
    if session_aggregation not in session_aggregation_modes:
        raise ValueError(
            f"Invalid session aggregation mode. Use one of: {', '.join(session_aggregation_modes)}"
        )
    if run_profile not in run_profile_modes:
        raise ValueError(f"Invalid run profile mode. Use one of: {', '.join(run_profile_modes)}")


def _profiles_to_collect(run_profile: str, profiles_for_storage: tuple[str, ...]) -> list[str]:
    """Resolve run profiles that should be extracted for a session."""
    profiles: list[str] = []
    for profile in (*profiles_for_storage, run_profile):
        if profile not in profiles:
            profiles.append(profile)
    return profiles


def _record_weighted_metrics(
    metric_samples: dict[str, dict[str, list[tuple[float, float]]]],
    team_sessions_used: dict[str, set[str]],
    session_id: str,
    team_metrics: dict[str, dict[str, float]],
    team_weights: dict[str, float],
) -> None:
    """Append weighted metrics for a session into cumulative team samples."""
    for team, metrics in team_metrics.items():
        for metric_name, value in metrics.items():
            metric_samples[team][metric_name].append(
                (float(value), float(team_weights.get(team, 1.0)))
            )
            team_sessions_used[team].add(session_id)


def _record_profile_weighted_metrics(
    profile_metric_samples: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    team_profile_sessions_used: dict[str, dict[str, set[str]]],
    profile: str,
    session_id: str,
    profile_weights: dict[str, float],
    perf_by_profile: dict[str, dict[str, float]],
    tire_by_profile: dict[str, dict[str, float]],
) -> None:
    """Append weighted profile metrics (pace + tire) for storage payloads."""
    for metrics_by_team in (perf_by_profile, tire_by_profile):
        for team, metrics in metrics_by_team.items():
            for metric_name, value in metrics.items():
                profile_metric_samples[profile][team][metric_name].append(
                    (float(value), float(profile_weights.get(team, 1.0)))
                )
                team_profile_sessions_used[team][profile].add(session_id)


def collect_sessions_for_events(
    *,
    year: int,
    events: list[str],
    session_candidates: list[str],
    testing_backends: tuple[str | None, ...],
    known_teams: set[str],
    run_profile: str,
    profiles_for_storage: tuple[str, ...],
    load_sessions_for_event: Callable[..., list[tuple[str, Any]]],
    collect_session_metrics: Callable[
        ..., tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]
    ],
    count_team_selected_laps: Callable[..., dict[str, float]],
    extract_session_compound_metrics: Callable[
        [Any, str, set[str]], dict[str, dict[str, dict[str, float | str | None]]]
    ],
    logger: Any,
) -> SessionCollectionResult:
    """Collect normalized team metrics for all requested events/sessions."""
    result = SessionCollectionResult(
        profile_metric_samples={
            profile: defaultdict(lambda: defaultdict(list)) for profile in profiles_for_storage
        }
    )

    for event_name in events:
        event_sessions = load_sessions_for_event(
            year=year,
            event_name=event_name,
            session_candidates=session_candidates,
            testing_backends=testing_backends,
            error_messages=result.load_errors,
        )
        for session_name, session in event_sessions:
            session_id = f"{event_name}::{session_name}"
            result.discovered_sessions.append(session_id)

            profile_results: dict[
                str, tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]
            ] = {}
            for profile in _profiles_to_collect(run_profile, profiles_for_storage):
                perf_by_profile, tire_by_profile = collect_session_metrics(
                    session=session,
                    session_key=session_name,
                    known_teams=known_teams,
                    run_profile=profile,
                    diagnostics=(result.extraction_diagnostics if profile == run_profile else None),
                )
                profile_results[profile] = (perf_by_profile, tire_by_profile)

                if profile in profiles_for_storage and (perf_by_profile or tire_by_profile):
                    profile_weights = count_team_selected_laps(
                        session=session,
                        known_teams=known_teams,
                        run_profile=profile,
                    )
                    _record_profile_weighted_metrics(
                        profile_metric_samples=result.profile_metric_samples,
                        team_profile_sessions_used=result.team_profile_sessions_used,
                        profile=profile,
                        session_id=session_id,
                        profile_weights=profile_weights,
                        perf_by_profile=perf_by_profile,
                        tire_by_profile=tire_by_profile,
                    )

            normalized_perf, normalized_tire = profile_results.get(run_profile, ({}, {}))
            if not normalized_perf and not normalized_tire:
                continue

            result.loaded_sessions.append(session_id)
            team_lap_weights = count_team_selected_laps(
                session=session,
                known_teams=known_teams,
                run_profile=run_profile,
            )
            _record_weighted_metrics(
                metric_samples=result.metric_samples,
                team_sessions_used=result.team_sessions_used,
                session_id=session_id,
                team_metrics=normalized_perf,
                team_weights=team_lap_weights,
            )
            _record_weighted_metrics(
                metric_samples=result.metric_samples,
                team_sessions_used=result.team_sessions_used,
                session_id=session_id,
                team_metrics=normalized_tire,
                team_weights=team_lap_weights,
            )

            try:
                normalized_compound_metrics = extract_session_compound_metrics(
                    session=session,
                    event_name=event_name,
                    known_teams=known_teams,
                )
                if normalized_compound_metrics:
                    result.compound_metrics_by_session[session_id] = (
                        event_name,
                        normalized_compound_metrics,
                    )
                    logger.debug(
                        "  Extracted compound metrics for %s teams",
                        len(normalized_compound_metrics),
                    )
            except Exception as exc:
                logger.warning(f"  Failed to extract compound metrics from {session_id}: {exc}")

    return result


def raise_if_no_loaded_sessions(
    discovered_sessions: list[str],
    loaded_sessions: list[str],
    extraction_diagnostics: list[str],
    load_errors: list[str],
) -> None:
    """Raise detailed errors when no usable sessions were loaded."""
    if loaded_sessions:
        return

    if discovered_sessions:
        unique_discovered = []
        seen_discovered = set()
        for session_id in discovered_sessions:
            if session_id not in seen_discovered:
                seen_discovered.add(session_id)
                unique_discovered.append(session_id)
            if len(unique_discovered) >= 5:
                break

        raise ValueError(
            "Sessions were found, but no usable team telemetry could be extracted yet. "
            "This usually means the session has too little completed running. "
            f"Detected sessions: {unique_discovered}. "
            f"Extraction diagnostics: {extraction_diagnostics[:3]}"
        )

    unique_errors = []
    seen = set()
    for msg in load_errors:
        if msg not in seen:
            seen.add(msg)
            unique_errors.append(msg)
        if len(unique_errors) >= 3:
            break

    details = f" First errors: {unique_errors}" if unique_errors else ""
    all_data_not_loaded = bool(unique_errors) and all(
        "DataNotLoadedError" in error for error in unique_errors
    )
    cache_hint = ""
    if all_data_not_loaded:
        cache_hint = (
            " Likely cache issue; retry with a fresh cache directory "
            "(e.g. --cache-dir _tmp_fastf1_cache_testing_2026 "
            "--force-renew-cache; it will be created under data/raw)."
        )
    raise ValueError(
        "No loadable sessions found. Verify event names and data availability in FastF1 cache/API."
        + cache_hint
        + details
    )


def _collect_aggregated_metrics(
    samples: dict[str, list[tuple[float, float]]],
    session_aggregation: str,
    aggregate_metric_samples: Callable[[list[tuple[float, float]], str], float | None],
) -> dict[str, float]:
    """Aggregate one team's metric samples using configured strategy."""
    aggregated_metrics: dict[str, float] = {}
    for metric_name, values in samples.items():
        aggregated = aggregate_metric_samples(values, session_aggregation=session_aggregation)
        if aggregated is not None:
            aggregated_metrics[metric_name] = aggregated
    return aggregated_metrics


def apply_team_updates(
    *,
    characteristics: dict,
    metric_samples: dict[str, dict[str, list[tuple[float, float]]]],
    profile_metric_samples: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    team_sessions_used: dict[str, set[str]],
    team_profile_sessions_used: dict[str, dict[str, set[str]]],
    compound_metrics_by_session: dict[
        str, tuple[str, dict[str, dict[str, dict[str, float | str | None]]]]
    ],
    now_iso: str,
    session_aggregation: str,
    run_profile: str,
    directionality_scale: float,
    new_weight: float,
    profiles_for_storage: tuple[str, ...],
    testing_characteristic_metrics: tuple[str, ...],
    aggregate_metric_samples: Callable[[list[tuple[float, float]], str], float | None],
    build_directionality_from_metrics: Callable[[dict[str, float], float], dict[str, float]],
    blend_directionality: Callable[[dict[str, float], dict[str, float], float], dict[str, float]],
    aggregate_compound_samples: Callable[..., dict],
) -> list[str]:
    """Apply extracted metrics back into characteristics payload."""
    updated_teams = []

    for team_name, samples in metric_samples.items():
        if team_name not in characteristics["teams"]:
            continue

        averaged_metrics = _collect_aggregated_metrics(
            samples=samples,
            session_aggregation=session_aggregation,
            aggregate_metric_samples=aggregate_metric_samples,
        )
        if not averaged_metrics:
            continue

        extracted_directionality = build_directionality_from_metrics(
            averaged_metrics,
            directionality_scale=directionality_scale,
        )

        team_data = characteristics["teams"][team_name]
        current_directionality = team_data.get("directionality")
        if not isinstance(current_directionality, dict):
            current_directionality = {}
        blended_directionality = blend_directionality(
            old_directionality=current_directionality,
            new_directionality=extracted_directionality,
            new_weight=new_weight,
        )

        team_data["directionality"] = blended_directionality
        team_data["last_updated"] = now_iso

        testing_characteristics = team_data.get("testing_characteristics")
        if not isinstance(testing_characteristics, dict):
            testing_characteristics = {}
        for metric_name in testing_characteristic_metrics:
            if metric_name in averaged_metrics:
                testing_characteristics[metric_name] = round(
                    float(averaged_metrics[metric_name]), 4
                )

        testing_characteristics["last_updated"] = now_iso
        testing_characteristics["sessions_used"] = len(team_sessions_used.get(team_name, set()))
        testing_characteristics["session_aggregation"] = session_aggregation
        testing_characteristics["run_profile"] = run_profile
        team_data["testing_characteristics"] = testing_characteristics

        existing_profiles = team_data.get("testing_characteristics_profiles")
        if not isinstance(existing_profiles, dict):
            existing_profiles = {}

        for profile in profiles_for_storage:
            profile_samples = profile_metric_samples.get(profile, {}).get(team_name, {})
            profile_metrics = _collect_aggregated_metrics(
                samples=profile_samples,
                session_aggregation=session_aggregation,
                aggregate_metric_samples=aggregate_metric_samples,
            )
            if not profile_metrics:
                continue

            profile_data = existing_profiles.get(profile)
            if not isinstance(profile_data, dict):
                profile_data = {}

            for metric_name in testing_characteristic_metrics:
                if metric_name in profile_metrics:
                    profile_data[metric_name] = round(float(profile_metrics[metric_name]), 4)

            profile_data["last_updated"] = now_iso
            profile_data["sessions_used"] = len(
                team_profile_sessions_used[team_name].get(profile, set())
            )
            profile_data["session_aggregation"] = session_aggregation
            profile_data["run_profile"] = profile
            existing_profiles[profile] = profile_data

        team_data["testing_characteristics_profiles"] = existing_profiles

        existing_compound_chars = team_data.get("compound_characteristics")
        if not isinstance(existing_compound_chars, dict):
            existing_compound_chars = {}

        for _session_id, session_payload in compound_metrics_by_session.items():
            session_event_name, session_compounds = session_payload
            if team_name not in session_compounds:
                continue
            existing_compound_chars = aggregate_compound_samples(
                existing_compound_chars,
                session_compounds[team_name],
                blend_weight=new_weight,
                race_name=session_event_name,
            )

        if existing_compound_chars:
            for compound_data in existing_compound_chars.values():
                compound_data["last_updated"] = now_iso

        team_data["compound_characteristics"] = existing_compound_chars
        updated_teams.append(team_name)

    return updated_teams


def write_characteristics_if_needed(
    characteristics_file: Path,
    characteristics: dict,
    now_iso: str,
    dry_run: bool,
    atomic_json_write: Callable[..., Any],
) -> None:
    """Persist characteristics payload unless dry-run mode is enabled."""
    if dry_run:
        return

    current_version = characteristics.get("version", 0)
    try:
        current_version = int(current_version)
    except (TypeError, ValueError):
        current_version = 0

    characteristics["last_updated"] = now_iso
    characteristics["version"] = current_version + 1
    atomic_json_write(characteristics_file, characteristics, create_backup=True)
