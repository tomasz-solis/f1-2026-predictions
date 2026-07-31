"""Research-only orchestration for explicit prediction challenger variants.

This module deliberately sits outside dashboard and production prediction flows.  It
keeps raw practice evidence and coherent grid permutations in process just long
enough to connect qualifying to race, then returns only scrubbed prediction payloads
and stable research metadata.  Optional persistence is restricted to the immutable
``ResearchSidecarStore``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from src.analysis.challenger_governance import (
    stable_json_sha256,
    validate_challenger_manifest,
)
from src.analysis.challenger_release import freeze_forecast_pair
from src.data.actual_results_fetcher import fetch_official_starting_grid
from src.models.challenger_variants import (
    CHAMPION_VARIANT,
    VARIANT_COMPONENTS,
    resolve_model_variant,
)
from src.persistence.research_sidecar import ResearchSidecarStore
from src.utils.grid_scenarios import (
    GRID_SOURCE_DETAILS,
    grid_scenario_digest,
    validate_grid_scenarios,
)

_PRIVATE_PAYLOAD_KEYS = frozenset(
    {
        "grid_scenarios",
        "qualifying_practice_evidence",
        "race_practice_evidence",
    }
)
_RACE_COMPONENTS = frozenset({"r0", "r1", "r2_no_anchor", "r2_source_anchor"})
_OBSERVED_GRID_SOURCES = frozenset({"actual_qualifying", "actual_starting_grid"})


class ChallengerPredictor(Protocol):
    """Minimal predictor surface required by the research orchestrator."""

    config: Any

    def predict_qualifying(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def predict_race(self, **kwargs: Any) -> Mapping[str, Any]: ...


def fetch_official_race_grid_for_research(
    year: int,
    race_name: str,
    *,
    qualifying_classification: Sequence[Mapping[str, Any]] | None = None,
    session_name: str = "R",
) -> tuple[list[dict[str, Any]], str] | None:
    """Fetch an official grid ready for the research orchestrator.

    This explicit helper keeps network-backed result ingestion outside production
    prediction flows.  A caller can pass the returned tuple directly as
    ``race_grid`` and ``race_grid_source_detail``.  ``None`` means that the official
    grid is not available yet, so the caller must keep the predicted-grid view.
    """

    classification = (
        [dict(row) for row in qualifying_classification]
        if qualifying_classification is not None
        else None
    )
    official_grid = fetch_official_starting_grid(
        int(year),
        str(race_name),
        session_name=session_name,
        qualifying_classification=classification,  # type: ignore[arg-type]
    )
    if official_grid is None:
        return None
    return [dict(row) for row in official_grid], "actual_starting_grid"


def _resolve_race_grid_handoff(
    *,
    raw_qualifying_grid: Sequence[Mapping[str, Any]],
    race_grid: Sequence[Mapping[str, Any]] | None,
    requested_source_detail: str | None,
    joint_scenarios: list[list[str]] | None,
) -> tuple[list[Mapping[str, Any]], str, list[list[str]] | None]:
    """Resolve grid precedence and prevent observed grids carrying prediction noise."""

    normalized_source = (
        str(requested_source_detail).strip().lower()
        if requested_source_detail is not None and str(requested_source_detail).strip()
        else None
    )
    if normalized_source is not None and normalized_source not in GRID_SOURCE_DETAILS:
        raise ValueError(
            "race_grid_source_detail must be one of "
            f"{sorted(GRID_SOURCE_DETAILS)}, got {requested_source_detail!r}"
        )
    if race_grid is None and normalized_source in _OBSERVED_GRID_SOURCES:
        raise ValueError(
            f"race_grid_source_detail={normalized_source!r} requires an explicit race_grid"
        )

    selected_grid = list(race_grid) if race_grid is not None else list(raw_qualifying_grid)
    start_metadata = ["start_type" in row for row in selected_grid]
    if any(start_metadata) and not all(start_metadata):
        raise ValueError("Official starting-grid metadata must be present on every grid row")
    has_official_start_metadata = bool(start_metadata) and all(start_metadata)
    if has_official_start_metadata:
        if normalized_source not in (None, "actual_starting_grid"):
            raise ValueError(
                "Grid rows containing start_type require "
                "race_grid_source_detail='actual_starting_grid'"
            )
        normalized_source = "actual_starting_grid"
    elif normalized_source == "actual_starting_grid":
        raise ValueError(
            "race_grid_source_detail='actual_starting_grid' requires start_type on every row"
        )

    source_detail = normalized_source or (
        "predicted_joint" if joint_scenarios is not None else "predicted_marginal_fallback"
    )
    if source_detail in _OBSERVED_GRID_SOURCES:
        return selected_grid, source_detail, None
    if source_detail == "predicted_joint" and joint_scenarios is None:
        raise ValueError("race_grid_source_detail='predicted_joint' requires joint scenarios")
    if source_detail != "predicted_joint" and joint_scenarios is not None:
        raise ValueError("Joint scenarios require race_grid_source_detail='predicted_joint'")
    return selected_grid, source_detail, joint_scenarios


def _normalise_variant(
    predictor: ChallengerPredictor, variant_id: str
) -> tuple[str, frozenset[str]]:
    """Validate an explicit non-champion variant and its predictor configuration."""

    variant = str(variant_id).strip().lower()
    if variant not in VARIANT_COMPONENTS:
        raise ValueError(f"Unknown challenger variant_id: {variant!r}")
    if variant == CHAMPION_VARIANT:
        raise ValueError("Research challenger orchestration requires a non-champion variant")

    configured_variant = resolve_model_variant(predictor.config)
    if configured_variant != variant:
        raise ValueError(
            "Predictor configuration does not match the requested research variant "
            f"({configured_variant!r} != {variant!r})"
        )
    return variant, VARIANT_COMPONENTS[variant]


def _validate_q1_launch_diagnostics(
    raw_qualifying: Mapping[str, Any],
    *,
    candidate_id: str,
    variant_id: str,
    manifest_sha256: str,
) -> None:
    """Prevent a forecast from being frozen under an unrelated Q1 manifest."""

    raw_diagnostics = raw_qualifying.get("qualifying_practice_challenger")
    if not isinstance(raw_diagnostics, Mapping) or raw_diagnostics.get("used") is not True:
        raise ValueError("Q1 manifest-bound run did not use its launch envelope")
    if str(raw_diagnostics.get("variant", "")).strip().lower() != variant_id:
        raise ValueError("Q1 runtime diagnostics returned the wrong variant")
    launch = raw_diagnostics.get("artifact_launch")
    if not isinstance(launch, Mapping):
        raise ValueError("Q1 runtime diagnostics are missing artifact_launch")
    if str(launch.get("candidate_id", "")).strip() != candidate_id:
        raise ValueError("Q1 launch candidate does not match the supplied manifest")
    if str(launch.get("variant_id", "")).strip().lower() != variant_id:
        raise ValueError("Q1 launch variant does not match the supplied manifest")
    expected_manifest_digest = f"sha256:{manifest_sha256}"
    if str(launch.get("manifest_digest", "")).strip().lower() != expected_manifest_digest:
        raise ValueError("Q1 launch manifest digest does not match the supplied manifest")
    for field_name in ("launch_digest", "bundle_digest"):
        digest = str(launch.get(field_name, "")).strip().lower()
        raw_digest = digest.removeprefix("sha256:")
        if (
            not digest.startswith("sha256:")
            or len(raw_digest) != 64
            or any(character not in "0123456789abcdef" for character in raw_digest)
        ):
            raise ValueError(f"Q1 runtime diagnostics contain an invalid {field_name}")


def _without_reserved_kwargs(
    raw_kwargs: Mapping[str, Any] | None,
    *,
    reserved: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    """Copy caller arguments while preventing private handoff overrides."""

    kwargs = dict(raw_kwargs or {})
    conflicts = sorted(reserved.intersection(kwargs))
    if conflicts:
        raise ValueError(f"{field_name} contains reserved arguments: {', '.join(conflicts)}")
    return kwargs


def _scrub_private_payload(value: Any) -> Any:
    """Recursively remove raw challenger artifacts from a public result."""

    if isinstance(value, Mapping):
        return {
            str(key): _scrub_private_payload(item)
            for key, item in value.items()
            if str(key) not in _PRIVATE_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_scrub_private_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_private_payload(item) for item in value]
    return value


def _mapping_digest(payload: Mapping[str, Any]) -> str:
    """Return the same prefixed digest style used for grid scenarios."""

    return f"sha256:{stable_json_sha256(dict(payload))}"


def _artifact_reference(path: Path) -> dict[str, str]:
    """Read only the immutable sidecar envelope metadata needed by callers."""

    envelope = json.loads(path.read_text(encoding="utf-8"))
    digest = envelope.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Research sidecar is missing its artifact digest: {path}")
    return {"path": str(path), "digest": f"sha256:{digest}"}


def _persist_sidecars(
    *,
    store: ResearchSidecarStore,
    manifest: Mapping[str, Any],
    variant_id: str,
    qualifying_evidence: Mapping[str, Any] | None,
    grid_scenarios: Sequence[Sequence[str]] | None,
    central_grid_drivers: Sequence[str],
    race_evidence: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Persist raw research payloads only after the complete run succeeds."""

    validated_manifest = validate_challenger_manifest(
        manifest,
        expected_variant_id=variant_id,
    )

    manifest_path = store.write_manifest(manifest)
    references: dict[str, dict[str, str]] = {
        "manifest": {
            "path": str(manifest_path),
            "digest": f"sha256:{validated_manifest.manifest_sha256}",
        }
    }
    artifact_payloads: list[tuple[str, Mapping[str, Any]]] = []
    if qualifying_evidence is not None:
        artifact_payloads.append(("qualifying_practice_evidence", qualifying_evidence))
    if grid_scenarios:
        artifact_payloads.append(
            (
                "grid_scenarios",
                {
                    "schema_version": 1,
                    "central_grid_drivers": list(central_grid_drivers),
                    "scenario_count": len(grid_scenarios),
                    "scenario_digest": grid_scenario_digest(grid_scenarios),
                    "scenarios": [list(scenario) for scenario in grid_scenarios],
                },
            )
        )
    if race_evidence is not None:
        artifact_payloads.append(("race_practice_evidence", race_evidence))

    for artifact_kind, payload in artifact_payloads:
        path = store.write_artifact(
            manifest=manifest,
            artifact_kind=artifact_kind,
            payload=payload,
        )
        references[artifact_kind] = _artifact_reference(path)
    return references


def run_challenger_pipeline(
    predictor: ChallengerPredictor,
    *,
    variant_id: str,
    qualifying_kwargs: Mapping[str, Any],
    race_kwargs: Mapping[str, Any] | None = None,
    race_grid: Sequence[Mapping[str, Any]] | None = None,
    race_grid_source_detail: str | None = None,
    race_practice_evidence: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    sidecar_store: ResearchSidecarStore | None = None,
    champion_prediction: Mapping[str, Any] | None = None,
    forecast_year: int | None = None,
    forecast_event_name: str | None = None,
    forecast_frozen_at: str | None = None,
) -> dict[str, Any]:
    """Run one explicit challenger without activating or mutating production state.

    ``R1`` grid scenarios and ``Q1`` evidence are requested only when their registered
    components need them.  Raw payloads are never returned.  If both ``manifest`` and
    ``sidecar_store`` are supplied, the payloads are persisted in immutable research
    sidecars and only their paths/digests are exposed.
    """

    variant, components = _normalise_variant(predictor, variant_id)
    if (manifest is None) != (sidecar_store is None):
        raise ValueError("manifest and sidecar_store must be supplied together")
    forecast_freeze_values = (
        champion_prediction,
        forecast_year,
        forecast_event_name,
        forecast_frozen_at,
    )
    if any(value is not None for value in forecast_freeze_values) and not all(
        value is not None for value in forecast_freeze_values
    ):
        raise ValueError("forecast freeze arguments must be supplied together")
    if champion_prediction is not None and (manifest is None or sidecar_store is None):
        raise ValueError("forecast freezing requires manifest and sidecar_store")
    validated_manifest = None
    if manifest is not None:
        validated_manifest = validate_challenger_manifest(
            manifest,
            expected_variant_id=variant,
        )
    if components.intersection(_RACE_COMPONENTS) and race_kwargs is None:
        raise ValueError(f"Variant {variant!r} contains race components and requires race_kwargs")
    if race_grid is not None and race_kwargs is None:
        raise ValueError("race_grid requires race_kwargs")
    if race_practice_evidence is not None and "r0" not in components:
        raise ValueError("race_practice_evidence can only be supplied to a variant containing R0")

    q_kwargs = _without_reserved_kwargs(
        qualifying_kwargs,
        reserved=frozenset({"include_grid_scenarios", "include_challenger_evidence"}),
        field_name="qualifying_kwargs",
    )
    needs_joint_grid = "r1" in components
    needs_qualifying_evidence = "q1" in components
    raw_qualifying = dict(
        predictor.predict_qualifying(
            **q_kwargs,
            include_grid_scenarios=needs_joint_grid,
            include_challenger_evidence=needs_qualifying_evidence,
        )
    )
    if needs_qualifying_evidence and validated_manifest is not None:
        _validate_q1_launch_diagnostics(
            raw_qualifying,
            candidate_id=validated_manifest.candidate_id,
            variant_id=validated_manifest.variant_id,
            manifest_sha256=validated_manifest.manifest_sha256,
        )

    raw_grid = raw_qualifying.get("grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("Challenger qualifying result must contain a non-empty grid")
    central_grid_drivers = [
        str(row.get("driver", "")).strip() for row in raw_grid if isinstance(row, Mapping)
    ]
    if len(central_grid_drivers) != len(raw_grid) or any(
        not driver for driver in central_grid_drivers
    ):
        raise ValueError("Challenger qualifying grid contains an invalid driver entry")
    if len(set(central_grid_drivers)) != len(central_grid_drivers):
        raise ValueError("Challenger qualifying grid drivers must be unique")

    qualifying_evidence = (
        raw_qualifying.get("qualifying_practice_evidence") if needs_qualifying_evidence else None
    )
    if qualifying_evidence is not None and not isinstance(qualifying_evidence, Mapping):
        raise ValueError("qualifying_practice_evidence must be a mapping when returned")

    validated_scenarios: list[list[str]] | None = None
    if needs_joint_grid:
        raw_scenarios = raw_qualifying.get("grid_scenarios")
        if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, str | bytes):
            raise ValueError("R1 requires complete joint grid_scenarios from qualifying")
        try:
            validated_scenarios = validate_grid_scenarios(
                raw_scenarios,
                expected_drivers=central_grid_drivers,
            )
        except ValueError as exc:
            raise ValueError(f"R1 qualifying grid_scenarios failed closed: {exc}") from exc

    public_qualifying = _scrub_private_payload(raw_qualifying)
    public_qualifying["model_variant"] = variant
    if validated_scenarios is not None:
        public_qualifying.update(
            {
                "grid_source_detail": "predicted_joint",
                "grid_scenario_count": len(validated_scenarios),
                "grid_scenario_digest": grid_scenario_digest(validated_scenarios),
                "grid_uncertainty_mode": "joint_scenarios",
            }
        )
    if isinstance(qualifying_evidence, Mapping):
        public_qualifying.update(
            {
                "qualifying_practice_evidence_session_count": len(qualifying_evidence),
                "qualifying_practice_evidence_digest": _mapping_digest(qualifying_evidence),
            }
        )

    public_race: dict[str, Any] | None = None
    if race_kwargs is not None:
        r_kwargs = _without_reserved_kwargs(
            race_kwargs,
            reserved=frozenset(
                {
                    "qualifying_grid",
                    "grid_scenarios",
                    "grid_source_detail",
                    "race_practice_evidence",
                }
            ),
            field_name="race_kwargs",
        )
        selected_race_grid, source_detail, race_grid_scenarios = _resolve_race_grid_handoff(
            raw_qualifying_grid=raw_grid,
            race_grid=race_grid,
            requested_source_detail=race_grid_source_detail,
            joint_scenarios=validated_scenarios,
        )
        raw_race = dict(
            predictor.predict_race(
                **r_kwargs,
                qualifying_grid=selected_race_grid,
                grid_scenarios=race_grid_scenarios,
                grid_source_detail=source_detail,
                race_practice_evidence=(race_practice_evidence if "r0" in components else None),
            )
        )
        public_race = _scrub_private_payload(raw_race)
        public_race["model_variant"] = variant
        if race_practice_evidence is not None:
            public_race.update(
                {
                    "race_practice_evidence_count": len(race_practice_evidence),
                    "race_practice_evidence_digest": _mapping_digest(race_practice_evidence),
                }
            )

    sidecars: dict[str, dict[str, str]] = {}
    if sidecar_store is not None and manifest is not None:
        sidecars = _persist_sidecars(
            store=sidecar_store,
            manifest=manifest,
            variant_id=variant,
            qualifying_evidence=(
                dict(qualifying_evidence) if isinstance(qualifying_evidence, Mapping) else None
            ),
            grid_scenarios=validated_scenarios,
            central_grid_drivers=central_grid_drivers,
            race_evidence=(
                dict(race_practice_evidence) if race_practice_evidence is not None else None
            ),
        )
        if champion_prediction is not None:
            assert forecast_year is not None
            assert forecast_event_name is not None
            assert forecast_frozen_at is not None
            sidecars["frozen_forecasts"] = freeze_forecast_pair(
                store=sidecar_store,
                manifest=manifest,
                year=forecast_year,
                event_name=forecast_event_name,
                champion_prediction=champion_prediction,
                challenger_prediction={
                    "model_variant": variant,
                    "qualifying": public_qualifying,
                    "race": public_race,
                },
                frozen_at=forecast_frozen_at,
            )

    return {
        "artifact_type": "challenger_research_run",
        "schema_version": 1,
        "variant_id": variant,
        "components": sorted(components),
        "production_activation": False,
        "qualifying": public_qualifying,
        "race": public_race,
        "research_sidecars": sidecars,
    }
