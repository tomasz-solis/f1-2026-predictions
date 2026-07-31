"""Non-mutating registration, actuals attachment, and release-readiness policy."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analysis.challenger_governance import (
    DEFAULT_REPLAY_CHECKPOINTS,
    DEFAULT_REPLAY_SEEDS,
    ValidatedChallengerManifest,
    race_candidate_kind_for_components,
    stable_json_sha256,
    validate_challenger_manifest,
)
from src.models.challenger_variants import CHAMPION_VARIANT
from src.persistence.research_sidecar import ResearchSidecarStore
from src.utils.grid_validation import validate_qualifying_grid

REQUIRED_POST_RACE_AUDITS = (
    "evaluation",
    "candidate",
    "shadow",
    "movement",
    "promotion",
    "leakage",
)
MIN_CHAMPION_SHADOW_WEEKENDS = 3
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_FORECAST_KEYS = frozenset(
    {"grid_scenarios", "qualifying_practice_evidence", "race_practice_evidence"}
)
_QUALIFYING_GATE_CHECKS = frozenset(
    {
        "finite_metrics",
        "minimum_scored_events",
        "event_identity_populated",
        "checkpoint_accounting_complete",
        "checkpoint_accounting_covers_events",
        "grid_mae_improvement",
        "grid_mae_ci90_above_zero",
        "h2h_brier_relative_improvement",
        "h2h_log_loss_not_regressed",
        "ece_within_tolerance",
        "interval_coverage_in_band",
        "interval_width_within_tolerance",
        "checkpoint_slices_within_tolerance",
        "conditional_actual_grid_unchanged",
        "end_to_end_race_not_regressed",
        "movement_review_complete",
        "race_view_event_counts_match",
        "race_view_event_identity_matches",
        "qualifying_race_event_identity_matches",
    }
)
_RACE_GATE_BASE_CHECKS = frozenset(
    {
        "finite_metrics",
        "has_paired_events",
        "race_view_event_counts_match",
        "race_view_event_identity_matches",
        "checkpoint_accounting_covers_events",
        "winner_accuracy_not_regressed",
        "top3_accuracy_within_tolerance",
        "dnf_brier_within_tolerance",
    }
)
_RACE_GATE_KIND_CHECKS: Mapping[str, frozenset[str]] = {
    "qualifying_only": frozenset(
        {"conditional_actual_grid_unchanged", "end_to_end_predicted_grid_not_regressed"}
    ),
    "race_input_or_grid_propagation": frozenset(
        {"end_to_end_predicted_grid_improved", "conditional_actual_grid_within_tolerance"}
    ),
    "anchor_or_physics": frozenset(
        {"conditional_actual_grid_improved", "end_to_end_predicted_grid_improved"}
    ),
}


def _utc_timestamp(value: str | datetime, *, field_name: str) -> datetime:
    candidate: datetime
    if isinstance(value, datetime):
        candidate = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            candidate = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if candidate.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return candidate.astimezone(UTC)


def _utc_text(value: str | datetime, *, field_name: str) -> str:
    return _utc_timestamp(value, field_name=field_name).isoformat().replace("+00:00", "Z")


def _reject_retrospective_diagnostic_marker(manifest: Mapping[str, Any], *, action: str) -> None:
    """Refuse an action for any manifest carrying the Q1 runtime's retrospective_
    diagnostic marker (a bundle built today resolved against an already-past
    historical cutoff -- see qualifying_practice_bundle.py). Distinct from the
    pre-existing ``classify_shadow_registration`` classification string of the same
    name: this checks the metadata *marker* a research backend sets, not a lateness
    classification, and it is never acceptable on this path regardless of timing."""
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("retrospective_diagnostic"):
        raise ValueError(f"{action} refused: manifest carries a retrospective_diagnostic marker")


def _sha256(value: Any, *, field_name: str) -> str:
    digest = str(value).strip()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a raw lowercase SHA-256 digest")
    return digest


def _event_identity(year: int, event_name: str) -> dict[str, Any]:
    event = str(event_name).strip()
    if int(year) <= 0 or not event:
        raise ValueError("event identity requires a positive year and non-blank event_name")
    identity: dict[str, Any] = {"year": int(year), "event_name": event}
    identity["event_sha256"] = stable_json_sha256(identity)
    return identity


def _private_forecast_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _PRIVATE_FORECAST_KEYS:
                return str(key)
            nested = _private_forecast_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            nested = _private_forecast_key(item)
            if nested is not None:
                return nested
    return None


def build_frozen_forecast_pair(
    *,
    manifest: Mapping[str, Any],
    year: int,
    event_name: str,
    champion_prediction: Mapping[str, Any],
    challenger_prediction: Mapping[str, Any],
    frozen_at: str | datetime,
) -> dict[str, Any]:
    """Build a digest-bound pair of scrubbed pre-event forecast artifacts."""

    _reject_retrospective_diagnostic_marker(manifest, action="freezing a forecast pair")
    validated = validate_challenger_manifest(manifest)
    frozen = _utc_timestamp(frozen_at, field_name="frozen_at")
    if frozen < validated.created_at:
        raise ValueError("frozen_at must be at or after manifest created_at")
    forecasts = {
        "champion": dict(champion_prediction),
        "challenger": dict(challenger_prediction),
    }
    for role, prediction in forecasts.items():
        if not prediction:
            raise ValueError(f"{role} prediction must be a non-empty mapping")
        private_key = _private_forecast_key(prediction)
        if private_key is not None:
            raise ValueError(f"frozen forecasts cannot contain private payload key {private_key!r}")
    if str(forecasts["champion"].get("model_variant", "")).strip().lower() != CHAMPION_VARIANT:
        raise ValueError("champion forecast must declare model_variant='champion'")
    if (
        str(forecasts["challenger"].get("model_variant", "")).strip().lower()
        != validated.variant_id
    ):
        raise ValueError("challenger forecast model_variant does not match the manifest")

    payload: dict[str, Any] = {
        "artifact_type": "challenger_frozen_forecast_pair",
        "schema_version": 1,
        "manifest_sha256": validated.manifest_sha256,
        "candidate_id": validated.candidate_id,
        "variant_id": validated.variant_id,
        "event": _event_identity(year, event_name),
        "frozen_at": frozen.isoformat().replace("+00:00", "Z"),
        "forecast_sha256": {
            role: stable_json_sha256(prediction) for role, prediction in forecasts.items()
        },
        "forecasts": forecasts,
    }
    payload["forecast_pair_sha256"] = stable_json_sha256(payload)
    validate_frozen_forecast_pair(payload, manifest=manifest)
    return payload


def validate_frozen_forecast_pair(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen pair and recompute every payload and event digest."""

    validated = validate_challenger_manifest(manifest)
    pair = dict(payload)
    if (
        pair.get("artifact_type") != "challenger_frozen_forecast_pair"
        or pair.get("schema_version") != 1
    ):
        raise ValueError("frozen forecast pair type or schema is invalid")
    if pair.get("manifest_sha256") != validated.manifest_sha256:
        raise ValueError("frozen forecast pair is not linked to the manifest")
    if (
        pair.get("candidate_id") != validated.candidate_id
        or pair.get("variant_id") != validated.variant_id
    ):
        raise ValueError("frozen forecast pair candidate or variant is invalid")
    event = pair.get("event")
    if not isinstance(event, Mapping):
        raise ValueError("frozen forecast pair event identity is missing")
    expected_event = _event_identity(int(event.get("year", 0)), str(event.get("event_name", "")))
    if dict(event) != expected_event:
        raise ValueError("frozen forecast pair event digest does not match its identity")
    frozen = _utc_timestamp(str(pair.get("frozen_at", "")), field_name="frozen_at")
    if frozen < validated.created_at:
        raise ValueError("frozen forecast timestamp predates manifest creation")

    forecasts = pair.get("forecasts")
    forecast_digests = pair.get("forecast_sha256")
    if not isinstance(forecasts, Mapping) or set(forecasts) != {"champion", "challenger"}:
        raise ValueError("frozen forecast pair must contain champion and challenger")
    if not isinstance(forecast_digests, Mapping) or set(forecast_digests) != {
        "champion",
        "challenger",
    }:
        raise ValueError("frozen forecast pair digests are incomplete")
    for role, expected_variant in (
        ("champion", CHAMPION_VARIANT),
        ("challenger", validated.variant_id),
    ):
        prediction = forecasts.get(role)
        if not isinstance(prediction, Mapping) or not prediction:
            raise ValueError(f"frozen {role} prediction must be a non-empty mapping")
        private_key = _private_forecast_key(prediction)
        if private_key is not None:
            raise ValueError(f"frozen forecasts cannot contain private payload key {private_key!r}")
        if str(prediction.get("model_variant", "")).strip().lower() != expected_variant:
            raise ValueError(f"frozen {role} prediction model_variant is invalid")
        digest = _sha256(forecast_digests.get(role), field_name=f"forecast_sha256.{role}")
        if digest != stable_json_sha256(dict(prediction)):
            raise ValueError(f"frozen {role} forecast digest does not match its payload")
    pair_digest = _sha256(
        pair.get("forecast_pair_sha256"),
        field_name="forecast_pair_sha256",
    )
    if pair_digest != stable_json_sha256(
        {key: value for key, value in pair.items() if key != "forecast_pair_sha256"}
    ):
        raise ValueError("forecast_pair_sha256 does not match the frozen pair")
    return pair


def freeze_forecast_pair(
    *,
    store: ResearchSidecarStore,
    manifest: Mapping[str, Any],
    year: int,
    event_name: str,
    champion_prediction: Mapping[str, Any],
    challenger_prediction: Mapping[str, Any],
    frozen_at: str | datetime,
) -> dict[str, str]:
    """Persist both scrubbed forecasts immutably and return a verifiable reference."""

    pair = build_frozen_forecast_pair(
        manifest=manifest,
        year=year,
        event_name=event_name,
        champion_prediction=champion_prediction,
        challenger_prediction=challenger_prediction,
        frozen_at=frozen_at,
    )
    path = store.write_artifact(
        manifest=manifest,
        artifact_kind="frozen_forecasts",
        payload=pair,
    )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "artifact_sha256": str(envelope["artifact_sha256"]),
        "manifest_sha256": str(pair["manifest_sha256"]),
        "forecast_pair_sha256": str(pair["forecast_pair_sha256"]),
        "event_sha256": str(pair["event"]["event_sha256"]),
    }


def validate_frozen_forecast_reference(
    reference: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and verify one immutable sidecar reference to a forecast pair."""

    validated = validate_challenger_manifest(manifest)
    path_text = str(reference.get("path", "")).strip()
    if not path_text:
        raise ValueError("frozen forecast reference path is missing")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise ValueError("frozen forecast reference path does not exist")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen forecast reference is not a readable JSON sidecar") from exc
    if not isinstance(envelope, Mapping):
        raise ValueError("frozen forecast sidecar envelope must be a mapping")
    artifact_digest = _sha256(
        envelope.get("artifact_sha256"),
        field_name="forecast sidecar artifact_sha256",
    )
    if artifact_digest != stable_json_sha256(
        {key: value for key, value in envelope.items() if key != "artifact_sha256"}
    ):
        raise ValueError("frozen forecast sidecar artifact digest does not match")
    if (
        envelope.get("artifact_type") != "challenger_research_sidecar"
        or envelope.get("artifact_kind") != "frozen_forecasts"
    ):
        raise ValueError("reference does not target a frozen forecast sidecar")
    if envelope.get("manifest_sha256") != validated.manifest_sha256:
        raise ValueError("frozen forecast sidecar is linked to a different manifest")
    pair_payload = envelope.get("payload")
    if not isinstance(pair_payload, Mapping):
        raise ValueError("frozen forecast sidecar payload is missing")
    pair = validate_frozen_forecast_pair(pair_payload, manifest=manifest)
    expected_reference = {
        "path": str(path),
        "artifact_sha256": artifact_digest,
        "manifest_sha256": validated.manifest_sha256,
        "forecast_pair_sha256": str(pair["forecast_pair_sha256"]),
        "event_sha256": str(pair["event"]["event_sha256"]),
    }
    if dict(reference) != expected_reference:
        raise ValueError("frozen forecast reference metadata does not match its sidecar")
    return pair


def classify_shadow_registration(
    manifest: Mapping[str, Any],
    *,
    qualifying_start_at: str | datetime,
    frozen_forecast_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a frozen manifest and forecast pair as preregistered or retrospective."""

    _reject_retrospective_diagnostic_marker(manifest, action="shadow registration")
    validated = validate_challenger_manifest(manifest)
    qualifying_start = _utc_timestamp(
        qualifying_start_at,
        field_name="qualifying_start_at",
    )
    forecasts_frozen = False
    forecast_pair_sha256: str | None = None
    if frozen_forecast_reference is not None:
        pair = validate_frozen_forecast_reference(
            frozen_forecast_reference,
            manifest=manifest,
        )
        forecasts_frozen = True
        forecast_pair_sha256 = str(pair["forecast_pair_sha256"])
    frozen_before_qualifying = (
        forecasts_frozen
        and validated.cutoff_at <= validated.created_at < qualifying_start
        and _utc_timestamp(str(pair["frozen_at"]), field_name="frozen_at") < qualifying_start
    )
    return {
        "classification": (
            "preregistered_shadow" if frozen_before_qualifying else "retrospective_diagnostic"
        ),
        "frozen_before_qualifying": frozen_before_qualifying,
        "forecasts_frozen": forecasts_frozen,
        "manifest_created_at": validated.created_at.isoformat().replace("+00:00", "Z"),
        "input_cutoff_at": validated.cutoff_at.isoformat().replace("+00:00", "Z"),
        "qualifying_start_at": qualifying_start.isoformat().replace("+00:00", "Z"),
        "manifest_sha256": validated.manifest_sha256,
        "forecast_pair_sha256": forecast_pair_sha256,
    }


def build_weekend_actuals_attachment(
    *,
    manifest: Mapping[str, Any],
    frozen_forecast_reference: Mapping[str, Any],
    year: int,
    event_name: str,
    qualifying_actual: Sequence[Mapping[str, Any]],
    race_actual: Sequence[Mapping[str, Any]],
    attached_at: str | datetime,
    champion_prediction_sha256: str | None = None,
    challenger_prediction_sha256: str | None = None,
) -> dict[str, Any]:
    """Attach actuals only to a validated immutable pre-event forecast pair."""

    validated = validate_challenger_manifest(manifest)
    pair = validate_frozen_forecast_reference(
        frozen_forecast_reference,
        manifest=manifest,
    )
    event = _event_identity(year, event_name)
    if pair["event"] != event:
        raise ValueError("actuals event identity does not match the frozen forecasts")
    forecast_digests = dict(pair["forecast_sha256"])
    supplied = {
        "champion": champion_prediction_sha256,
        "challenger": challenger_prediction_sha256,
    }
    for role, digest in supplied.items():
        if digest is not None and _sha256(digest, field_name=f"{role}_prediction_sha256") != str(
            forecast_digests[role]
        ):
            raise ValueError(f"{role} prediction digest does not match the frozen forecast")

    qualifying = validate_qualifying_grid(
        qualifying_actual,
        require_sequential_positions=True,
    )
    race = validate_qualifying_grid(
        race_actual,
        require_sequential_positions=True,
    )
    if {row["driver"] for row in qualifying} != {row["driver"] for row in race}:
        raise ValueError("qualifying and race actuals must contain the same driver set")
    attached = _utc_timestamp(attached_at, field_name="attached_at")
    if attached < validated.cutoff_at:
        raise ValueError("actuals cannot be attached before the frozen manifest cutoff")
    payload: dict[str, Any] = {
        "artifact_type": "challenger_weekend_actuals_attachment",
        "schema_version": 2,
        "manifest_sha256": validated.manifest_sha256,
        "variant_id": validated.variant_id,
        "event": event,
        "attached_at": attached.isoformat().replace("+00:00", "Z"),
        "frozen_forecast_reference": dict(frozen_forecast_reference),
        "forecast_pair_sha256": str(pair["forecast_pair_sha256"]),
        "forecast_sha256": forecast_digests,
        "actuals": {"qualifying": qualifying, "race": race},
    }
    payload["attachment_sha256"] = stable_json_sha256(payload)
    validate_weekend_actuals_attachment(payload, manifest=manifest)
    return payload


def validate_weekend_actuals_attachment(
    attachment: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and recompute a complete actuals attachment and forecast linkage."""

    validated = validate_challenger_manifest(manifest)
    payload = dict(attachment)
    if (
        payload.get("artifact_type") != "challenger_weekend_actuals_attachment"
        or payload.get("schema_version") != 2
    ):
        raise ValueError("actuals attachment type or schema is invalid")
    if (
        payload.get("manifest_sha256") != validated.manifest_sha256
        or payload.get("variant_id") != validated.variant_id
    ):
        raise ValueError("actuals attachment is linked to a different manifest")
    event = payload.get("event")
    if not isinstance(event, Mapping) or dict(event) != _event_identity(
        int(event.get("year", 0)),
        str(event.get("event_name", "")),
    ):
        raise ValueError("actuals attachment event identity or digest is invalid")
    reference = payload.get("frozen_forecast_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("actuals attachment frozen forecast reference is missing")
    pair = validate_frozen_forecast_reference(reference, manifest=manifest)
    if pair["event"] != event:
        raise ValueError("actuals attachment event differs from the frozen forecasts")
    if (
        payload.get("forecast_pair_sha256") != pair["forecast_pair_sha256"]
        or payload.get("forecast_sha256") != pair["forecast_sha256"]
    ):
        raise ValueError("actuals attachment forecast digests do not match the frozen forecasts")
    attached = _utc_timestamp(str(payload.get("attached_at", "")), field_name="attached_at")
    if attached < validated.cutoff_at:
        raise ValueError("actuals attachment predates the frozen manifest cutoff")
    actuals = payload.get("actuals")
    if not isinstance(actuals, Mapping):
        raise ValueError("actuals attachment payload is missing")
    qualifying = validate_qualifying_grid(
        actuals.get("qualifying", []),
        require_sequential_positions=True,
    )
    race = validate_qualifying_grid(
        actuals.get("race", []),
        require_sequential_positions=True,
    )
    if {row["driver"] for row in qualifying} != {row["driver"] for row in race}:
        raise ValueError("qualifying and race actuals must contain the same driver set")
    digest = _sha256(payload.get("attachment_sha256"), field_name="attachment_sha256")
    if digest != stable_json_sha256(
        {key: value for key, value in payload.items() if key != "attachment_sha256"}
    ):
        raise ValueError("attachment_sha256 does not match the actuals payload")
    return payload


def build_gate_result_envelope(
    *,
    manifest: Mapping[str, Any],
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one promotion-gate result to the exact manifest that produced it."""

    validated = validate_challenger_manifest(manifest)
    gate = dict(gate_result)
    if (
        gate.get("variant_id") != validated.variant_id
        or gate.get("manifest_sha256") != validated.manifest_sha256
    ):
        raise ValueError("gate result is not bound to the supplied manifest")
    expected_kind = race_candidate_kind_for_components(validated.components)
    if gate.get("candidate_kind") != expected_kind:
        raise ValueError("gate candidate kind does not match the manifest components")
    target = str(gate.get("target", "")).strip()
    qualifying_components = validated.components.issubset({"q0", "q1"})
    if qualifying_components:
        if target not in {"main_qualifying", "sprint_qualifying"}:
            raise ValueError("qualifying-only manifest gates require a qualifying target")
        required_checks = _QUALIFYING_GATE_CHECKS
    else:
        if target not in {"grand_prix_race", "sprint_race"}:
            raise ValueError("race-component manifest gates require a race target")
        required_checks = _RACE_GATE_BASE_CHECKS | _RACE_GATE_KIND_CHECKS[expected_kind]
    checks = gate.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise ValueError("gate result requires non-empty boolean checks")
    if set(checks) != required_checks:
        raise ValueError("gate result checks do not match the required target gate schema")
    if not isinstance(gate.get("passed"), bool) or gate["passed"] != all(checks.values()):
        raise ValueError("gate passed flag does not match its checks")
    replay = gate.get("replay_provenance")
    if not isinstance(replay, Mapping) or set(replay) != {
        "seeds",
        "simulation_counts",
        "dry_only",
        "checkpoint_event_counts",
        "replay_sha256",
    }:
        raise ValueError("gate result replay provenance is incomplete")
    if replay.get("seeds") != list(DEFAULT_REPLAY_SEEDS):
        raise ValueError("gate result replay seeds do not match the fixed seed contract")
    if replay.get("simulation_counts") != dict(validated.simulation_counts):
        raise ValueError("gate result simulation counts do not match the manifest")
    if replay.get("dry_only") is not True:
        raise ValueError("gate result must record dry_only=true")
    _sha256(replay.get("replay_sha256"), field_name="gate replay_sha256")
    checkpoint_counts = replay.get("checkpoint_event_counts")
    if not isinstance(checkpoint_counts, Mapping) or tuple(checkpoint_counts) != (
        DEFAULT_REPLAY_CHECKPOINTS
    ):
        raise ValueError("gate result checkpoint accounting schema is invalid")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in checkpoint_counts.values()
    ) or not any(count > 0 for count in checkpoint_counts.values()):
        raise ValueError("gate result checkpoint accounting is not populated")
    _sha256(gate.get("event_set_sha256"), field_name="gate event_set_sha256")
    envelope: dict[str, Any] = {
        "artifact_type": "challenger_gate_result",
        "schema_version": 1,
        "manifest": dict(manifest),
        "gate": gate,
    }
    envelope["gate_envelope_sha256"] = stable_json_sha256(envelope)
    return envelope


def _validate_gate_result_envelope(
    envelope: Mapping[str, Any],
) -> tuple[
    ValidatedChallengerManifest,
    dict[str, Any],
]:
    payload = dict(envelope)
    if (
        payload.get("artifact_type") != "challenger_gate_result"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("gate result envelope type or schema is invalid")
    digest = _sha256(payload.get("gate_envelope_sha256"), field_name="gate_envelope_sha256")
    if digest != stable_json_sha256(
        {key: value for key, value in payload.items() if key != "gate_envelope_sha256"}
    ):
        raise ValueError("gate result envelope digest does not match")
    manifest = payload.get("manifest")
    gate = payload.get("gate")
    if not isinstance(manifest, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("gate result envelope is incomplete")
    validated = validate_challenger_manifest(manifest)
    rebuilt = build_gate_result_envelope(manifest=manifest, gate_result=gate)
    if rebuilt != payload:
        raise ValueError("gate result envelope payload is not canonical")
    return validated, dict(gate)


def _expected_component_kind(component: str) -> str:
    return race_candidate_kind_for_components(frozenset({component}))


def evaluate_release_readiness(
    *,
    release_at: str | datetime,
    component_gate_results: Mapping[str, Mapping[str, Any]],
    combination_gate_result: Mapping[str, Any] | None,
    post_race_audits: Mapping[str, str],
    actuals_attachment: Mapping[str, Any] | None,
    rollback_variant: str,
    champion_shadow_weekends: int,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a manifest-bound release decision without changing any state."""

    release_time = _utc_timestamp(release_at, field_name="release_at")
    validated: ValidatedChallengerManifest | None = None
    manifest_valid = False
    if manifest is not None:
        try:
            validated = validate_challenger_manifest(manifest)
            manifest_valid = True
        except (TypeError, ValueError):
            validated = None
    # A manifest built under a research-only gate relaxation (e.g. Q1/R2 walk-forward
    # minimum-training-event floors lowered for the 2026 short-season study) proves
    # nothing about the production thresholds it bypassed and must never reach the
    # promotion path, no matter how many other checks it passes.
    research_gate_relaxation_present = bool(
        manifest is not None
        and isinstance(manifest.get("metadata"), Mapping)
        and manifest["metadata"].get("research_gate_relaxation")
    )
    # Same rule for the Q1 runtime's retrospective_diagnostic escape hatch (a bundle
    # built today resolved against an already-past historical cutoff): permanently
    # labeled, never promotable, regardless of whether research_gate_relaxation is
    # also present on the same manifest.
    retrospective_diagnostic_present = bool(
        manifest is not None
        and isinstance(manifest.get("metadata"), Mapping)
        and manifest["metadata"].get("retrospective_diagnostic")
    )

    required_components = sorted(validated.components) if validated is not None else []
    component_checks: dict[str, bool] = {component: False for component in required_components}
    component_event_digests: dict[str, str] = {}
    component_contract_valid = validated is not None and set(component_gate_results) == set(
        required_components
    )
    if component_contract_valid:
        for component in required_components:
            try:
                gate_manifest, gate = _validate_gate_result_envelope(
                    component_gate_results[component]
                )
                component_checks[component] = (
                    gate_manifest.components == frozenset({component})
                    and gate.get("candidate_kind") == _expected_component_kind(component)
                    and gate.get("passed") is True
                )
                component_event_digests[component] = str(gate.get("event_set_sha256", ""))
            except (TypeError, ValueError):
                component_checks[component] = False

    combination_valid = False
    combination_event_digest: str | None = None
    if validated is not None and isinstance(combination_gate_result, Mapping):
        try:
            combination_manifest, gate = _validate_gate_result_envelope(combination_gate_result)
            combination_valid = (
                combination_manifest.manifest_sha256 == validated.manifest_sha256
                and gate.get("candidate_kind")
                == race_candidate_kind_for_components(validated.components)
                and gate.get("passed") is True
            )
            combination_event_digest = str(gate.get("event_set_sha256", ""))
        except (TypeError, ValueError):
            combination_valid = False
    event_sets_match = (
        combination_event_digest is not None
        and len(component_event_digests) == len(required_components)
        and all(digest == combination_event_digest for digest in component_event_digests.values())
    )

    audit_checks = {
        audit: str(post_race_audits.get(audit, "missing")).strip().lower() == "passed"
        for audit in REQUIRED_POST_RACE_AUDITS
    }
    actuals_valid = False
    if validated is not None and isinstance(actuals_attachment, Mapping):
        try:
            validate_weekend_actuals_attachment(actuals_attachment, manifest=manifest or {})
            actuals_valid = True
        except (TypeError, ValueError):
            actuals_valid = False
    checks = {
        "manifest_valid": manifest_valid,
        "no_research_gate_relaxation": not research_gate_relaxation_present,
        "no_retrospective_diagnostic": not retrospective_diagnostic_present,
        "weekday_release": release_time.weekday() < 5,
        "independent_component_gates_passed": component_contract_valid
        and bool(component_checks)
        and all(component_checks.values()),
        "passing_combination_tested": combination_valid,
        "gate_event_sets_match": event_sets_match,
        "post_race_audits_passed": all(audit_checks.values()),
        "actuals_attached": actuals_valid,
        "rollback_is_champion": str(rollback_variant).strip().lower() == CHAMPION_VARIANT,
        "champion_shadow_window_planned": int(champion_shadow_weekends)
        >= MIN_CHAMPION_SHADOW_WEEKENDS,
    }
    reasons = {
        "manifest_valid": "release requires a complete valid challenger manifest",
        "no_research_gate_relaxation": (
            "manifest carries a research_gate_relaxation marker and can never be promoted"
        ),
        "no_retrospective_diagnostic": (
            "manifest carries a retrospective_diagnostic marker and can never be promoted"
        ),
        "weekday_release": "release must occur Monday-Friday",
        "independent_component_gates_passed": (
            "every registered component requires its own valid manifest-bound gate"
        ),
        "passing_combination_tested": "the exact manifest combination was not gated",
        "gate_event_sets_match": (
            "component and combination gates do not use the same explicit event set"
        ),
        "post_race_audits_passed": "one or more mandatory post-race audits are missing or failed",
        "actuals_attached": "validated actuals are not linked to the frozen forecast pair",
        "rollback_is_champion": "rollback variant must remain champion",
        "champion_shadow_window_planned": "old champion must remain shadowed for three weekends",
    }
    return {
        "artifact_type": "challenger_release_readiness",
        "schema_version": 2,
        "release_allowed": all(checks.values()),
        "release_at": release_time.isoformat().replace("+00:00", "Z"),
        "manifest_sha256": validated.manifest_sha256 if validated is not None else None,
        "variant_id": validated.variant_id if validated is not None else None,
        "required_components": required_components,
        "checks": checks,
        "component_checks": component_checks,
        "audit_checks": audit_checks,
        "reasons": [reasons[name] for name, passed in checks.items() if not passed],
        "rollback_variant": CHAMPION_VARIANT,
        "champion_shadow_weekends": int(champion_shadow_weekends),
    }
