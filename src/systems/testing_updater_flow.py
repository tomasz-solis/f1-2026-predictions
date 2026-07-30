"""Orchestration helpers for testing updater session collection and persistence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.artifact_store import ArtifactStore
from src.utils.car_snapshot_history import (
    merge_snapshot_team_metrics,
    resolve_session_snapshot_metadata,
)


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
    session_snapshot_records: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_characteristics_payload(data_dir: str, target_year: int) -> tuple[Path, dict]:
    """Load characteristics payload and validate required top-level keys.

    Reads through ``ArtifactStore`` so DB-backed deployments see the same state the
    predictor and the race-learning path write. Reading the file directly served the
    copy baked into the deployment image, so every practice capture overwrote the
    accumulated season history with a months-old snapshot.
    """
    characteristics_file = (
        Path(data_dir) / "car_characteristics" / f"{target_year}_car_characteristics.json"
    )
    store = ArtifactStore(data_root=characteristics_file.parent.parent.parent)
    characteristics = store.load_artifact(
        artifact_type="car_characteristics",
        artifact_key=f"{target_year}::car_characteristics",
    )
    if not characteristics:
        raise FileNotFoundError(
            f"Characteristics file not found and no stored artifact for season "
            f"{target_year}: {characteristics_file}"
        )

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


def _build_session_snapshot_record(
    *,
    event_name: str,
    session_name: str,
    session: Any,
    profile_results: dict[str, tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]],
    driver_profile_results: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any] | None:
    """Build a raw session snapshot record from extracted per-profile metrics."""
    team_profiles: dict[str, dict[str, dict[str, float]]] = {}
    team_driver_deltas_seconds: dict[str, dict[str, dict[str, float]]] = {}

    for profile_name, (perf_by_profile, tire_by_profile) in profile_results.items():
        merged_metrics = merge_snapshot_team_metrics(perf_by_profile, tire_by_profile)
        for team_name, team_metrics in merged_metrics.items():
            team_profiles.setdefault(team_name, {})[profile_name] = team_metrics

    if isinstance(driver_profile_results, dict):
        for profile_name, team_driver_deltas in driver_profile_results.items():
            if not isinstance(team_driver_deltas, dict):
                continue
            for team_name, driver_deltas in team_driver_deltas.items():
                if not isinstance(driver_deltas, dict) or not driver_deltas:
                    continue
                team_driver_deltas_seconds.setdefault(team_name, {})[profile_name] = driver_deltas

    if not team_profiles:
        return None

    metadata = resolve_session_snapshot_metadata(session=session, session_name=session_name)
    return {
        "event_name": event_name,
        "session_name": session_name,
        "team_profiles": team_profiles,
        "team_driver_deltas_seconds": team_driver_deltas_seconds,
        **metadata,
    }


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
    extract_session_driver_deltas: Callable[..., dict[str, dict[str, float]]],
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
            driver_profile_results: dict[str, dict[str, dict[str, float]]] = {}
            for profile in _profiles_to_collect(run_profile, profiles_for_storage):
                perf_by_profile, tire_by_profile = collect_session_metrics(
                    session=session,
                    session_key=session_name,
                    known_teams=known_teams,
                    run_profile=profile,
                    diagnostics=(result.extraction_diagnostics if profile == run_profile else None),
                )
                profile_results[profile] = (perf_by_profile, tire_by_profile)
                driver_profile_results[profile] = extract_session_driver_deltas(
                    session=session,
                    known_teams=known_teams,
                    run_profile=profile,
                )

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

            snapshot_record = _build_session_snapshot_record(
                event_name=event_name,
                session_name=session_name,
                session=session,
                profile_results=profile_results,
                driver_profile_results=driver_profile_results,
            )
            if snapshot_record is not None:
                # Record per-team clean short-run lap counts so downstream prediction can
                # gate confidence/blend on data sufficiency (a thin/broken session should
                # not override a team's season prior at full checkpoint weight).
                snapshot_record["team_clean_lap_counts"] = count_team_selected_laps(
                    session=session,
                    known_teams=known_teams,
                    run_profile="short_run",
                )
                result.session_snapshot_records[session_id] = snapshot_record

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
                    session,
                    event_name,
                    known_teams,
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
                logger.warning("  Failed to extract compound metrics from %s: %s", session_id, exc)

    return result


class NoUsableSessionTelemetryError(ValueError):
    """Sessions were discovered but yielded no usable team telemetry yet.

    This is distinct from a hard misconfiguration: it is typically a transient
    data-availability lag during a live session weekend (FastF1 has not yet
    backfilled enough completed laps for the session). Subclasses ``ValueError``
    so existing callers that catch/expect ``ValueError`` keep working, while the
    warmup auto-capture path can catch it specifically and treat it as a
    graceful skip-and-retry rather than a fatal error.
    """


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

        raise NoUsableSessionTelemetryError(
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
        aggregated = aggregate_metric_samples(values, session_aggregation)
        if aggregated is not None:
            aggregated_metrics[metric_name] = aggregated
    return aggregated_metrics


def _coerce_non_negative_int(value: object, default: int = 0) -> int:
    """Parse a non-negative integer with a safe default."""
    if not isinstance(value, int | float | str | bytes | bytearray):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _merge_circuits_observed(
    existing_circuits: object,
    session_ids: set[str],
) -> list[str]:
    """Merge prior circuit provenance with the circuits seen in this update."""
    merged: list[str] = []
    if isinstance(existing_circuits, list):
        for raw_circuit in existing_circuits:
            circuit_name = str(raw_circuit).strip()
            if circuit_name and circuit_name not in merged:
                merged.append(circuit_name)

    for session_id in sorted(session_ids):
        event_name = str(session_id).split("::", 1)[0].strip()
        if event_name and event_name not in merged:
            merged.append(event_name)

    return merged


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
    """Apply extracted metrics back into characteristics payload.

    Practice metrics should evolve cumulatively across the season. A new FP
    session is informative, but it should not wipe out the profile built from
    earlier circuits, especially once several rounds have already contributed
    to the stored car picture.
    """
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
            directionality_scale,
        )

        team_data = characteristics["teams"][team_name]
        current_directionality = team_data.get("directionality")
        if not isinstance(current_directionality, dict):
            current_directionality = {}
        existing_testing_characteristics = team_data.get("testing_characteristics")
        if not isinstance(existing_testing_characteristics, dict):
            existing_testing_characteristics = {}
        previous_sessions_blended = _coerce_non_negative_int(
            existing_testing_characteristics.get("sessions_blended")
            or existing_testing_characteristics.get("sessions_used")
        )
        current_session_ids = team_sessions_used.get(team_name, set())
        current_session_count = len(current_session_ids)
        effective_weight = float(
            np.clip(
                new_weight / max((max(previous_sessions_blended, 0) + 1) ** 0.5, 1.0),
                0.0,
                1.0,
            )
        )
        blended_directionality = blend_directionality(
            current_directionality,
            extracted_directionality,
            effective_weight,
        )

        team_data["directionality"] = blended_directionality
        team_data["last_updated"] = now_iso

        testing_characteristics = existing_testing_characteristics
        for metric_name in testing_characteristic_metrics:
            if metric_name in averaged_metrics:
                new_metric_value = float(averaged_metrics[metric_name])
                existing_metric_value = testing_characteristics.get(metric_name)
                if isinstance(existing_metric_value, int | float):
                    new_metric_value = ((1.0 - effective_weight) * float(existing_metric_value)) + (
                        effective_weight * new_metric_value
                    )
                testing_characteristics[metric_name] = round(float(new_metric_value), 4)

        testing_characteristics["last_updated"] = now_iso
        testing_characteristics["sessions_used"] = previous_sessions_blended + current_session_count
        testing_characteristics["sessions_blended"] = (
            previous_sessions_blended + current_session_count
        )
        testing_characteristics["session_aggregation"] = session_aggregation
        testing_characteristics["run_profile"] = run_profile
        testing_characteristics["effective_blend_weight"] = round(effective_weight, 4)
        testing_characteristics["circuits_observed"] = _merge_circuits_observed(
            testing_characteristics.get("circuits_observed"),
            current_session_ids,
        )
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
            previous_profile_sessions_blended = _coerce_non_negative_int(
                profile_data.get("sessions_blended") or profile_data.get("sessions_used")
            )
            current_profile_session_ids = team_profile_sessions_used[team_name].get(profile, set())
            effective_profile_weight = float(
                np.clip(
                    new_weight / max((max(previous_profile_sessions_blended, 0) + 1) ** 0.5, 1.0),
                    0.0,
                    1.0,
                )
            )

            for metric_name in testing_characteristic_metrics:
                if metric_name in profile_metrics:
                    new_metric_value = float(profile_metrics[metric_name])
                    existing_metric_value = profile_data.get(metric_name)
                    if isinstance(existing_metric_value, int | float):
                        new_metric_value = (
                            (1.0 - effective_profile_weight) * float(existing_metric_value)
                        ) + (effective_profile_weight * new_metric_value)
                    profile_data[metric_name] = round(float(new_metric_value), 4)

            profile_data["last_updated"] = now_iso
            profile_data["sessions_used"] = previous_profile_sessions_blended + len(
                current_profile_session_ids
            )
            profile_data["sessions_blended"] = previous_profile_sessions_blended + len(
                current_profile_session_ids
            )
            profile_data["session_aggregation"] = session_aggregation
            profile_data["run_profile"] = profile
            profile_data["effective_blend_weight"] = round(effective_profile_weight, 4)
            profile_data["circuits_observed"] = _merge_circuits_observed(
                profile_data.get("circuits_observed"),
                current_profile_session_ids,
            )
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
                blend_weight=effective_weight,
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
    latest_known_version: int = 0,
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
    characteristics["version"] = max(current_version, int(latest_known_version)) + 1
    atomic_json_write(characteristics_file, characteristics, create_backup=True)
