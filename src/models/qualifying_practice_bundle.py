"""Immutable manifest, bundle, and launch bindings for the Q1 challenger.

The live champion never loads this module. An explicit Q1 research overlay points
at one immutable launch envelope. That envelope binds the semantic candidate
definition to a frozen manifest and content-addressed artifact bundle without
creating a circular manifest/config digest dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analysis.challenger_governance import validate_challenger_manifest
from src.models.challenger_variants import VARIANT_COMPONENTS
from src.models.qualifying_practice_challenger import (
    MODEL_SCHEMA_VERSION,
    FittedQualifyingPracticeModel,
)
from src.models.qualifying_practice_evidence import FittedPracticeNormalization

BUNDLE_SCHEMA_VERSION = 1
LAUNCH_ENVELOPE_SCHEMA_VERSION = 1
NORMALIZATION_ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINTS = ("PRE", "FP1", "FP2", "FP3")
SESSION_KINDS = ("main", "sprint")
_BUNDLE_FIELDS = {
    "artifact_type",
    "schema_version",
    "candidate_id",
    "variant_id",
    "manifest_sha256",
    "manifest_cutoff_at",
    "manifest_created_at",
    "created_at",
    "dry_only",
    "models",
    "normalizations",
    "bundle_sha256",
}


@dataclass(frozen=True)
class ResolvedQualifyingPracticeBundle:
    """Validated Q1 artifacts and non-sensitive provenance diagnostics."""

    model: FittedQualifyingPracticeModel
    normalization: FittedPracticeNormalization | None
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedQualifyingPracticeLaunch:
    """Validated Q1 launch, artifacts, and semantic runtime settings."""

    model: FittedQualifyingPracticeModel
    normalization: FittedPracticeNormalization | None
    uncertainty_scale: float
    diagnostics: Mapping[str, Any]


class MissingQualifyingPracticeTrackClassError(ValueError):
    """Raised when a practice checkpoint has no preregistered track-class binding."""


def stable_json_sha256(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible data."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 for one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware_timestamp(value: Any, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _actual_creation_timestamp(*, not_before: datetime, artifact_kind: str) -> str:
    created = datetime.now(UTC)
    if created < not_before:
        raise ValueError(f"{artifact_kind} cannot be created before its manifest")
    return created.isoformat()


def _hex_digest(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return digest


def _safe_token(value: Any, *, field_name: str, uppercase: bool = False) -> str:
    token = str(value or "").strip()
    token = token.upper() if uppercase else token.lower()
    if not token or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in token.lower()
    ):
        raise ValueError(f"{field_name} must be a non-blank filesystem-safe token")
    return token


def validate_challenger_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    candidate_id: str,
    variant_id: str,
) -> tuple[str, datetime]:
    """Validate the manifest fields needed to bind a Q1 artifact bundle."""

    validated = validate_challenger_manifest(
        manifest,
        expected_variant_id=variant_id,
    )
    if validated.candidate_id != candidate_id:
        raise ValueError("bundle candidate_id does not match the challenger manifest")
    if variant_id not in VARIANT_COMPONENTS or "q1" not in VARIANT_COMPONENTS[variant_id]:
        raise ValueError("bundle variant must contain the Q1 component")
    return validated.manifest_sha256, validated.cutoff_at


def _artifact_metadata(
    payload: Mapping[str, Any],
    *,
    artifact_kind: str,
    candidate_id: str,
    checkpoint: str,
    session_kind: str,
    track_class: str | None,
    latest_allowed_cutoff: datetime,
    latest_allowed_generated_at: datetime,
    retrospective_diagnostic: bool = False,
) -> tuple[Mapping[str, Any], datetime]:
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{artifact_kind} artifact is missing training_metadata")
    if str(metadata.get("candidate_id", "")).strip() != candidate_id:
        raise ValueError(f"{artifact_kind} candidate_id mismatch")
    if str(metadata.get("checkpoint", "")).strip().upper() != checkpoint:
        raise ValueError(f"{artifact_kind} checkpoint mismatch")
    if str(metadata.get("session_kind", "")).strip().lower() != session_kind:
        raise ValueError(f"{artifact_kind} session_kind mismatch")
    if metadata.get("dry_only") is not True:
        raise ValueError(f"{artifact_kind} must be dry-only")
    if (
        track_class is not None
        and str(metadata.get("track_class", "")).strip().lower() != track_class
    ):
        raise ValueError(f"{artifact_kind} track_class mismatch")

    cutoff = _aware_timestamp(
        metadata.get("cutoff_timestamp"),
        field_name=f"{artifact_kind}.training_metadata.cutoff_timestamp",
    )
    if cutoff > latest_allowed_cutoff:
        raise ValueError(f"{artifact_kind} artifact cutoff is later than the prediction cutoff")
    maximum_input = _aware_timestamp(
        metadata.get("max_input_timestamp"),
        field_name=f"{artifact_kind}.training_metadata.max_input_timestamp",
    )
    if maximum_input >= cutoff:
        raise ValueError(f"{artifact_kind} input timestamp is not strictly before its cutoff")
    calibration_maximum = metadata.get("calibration_max_input_timestamp")
    if calibration_maximum is not None:
        calibration_timestamp = _aware_timestamp(
            calibration_maximum,
            field_name=f"{artifact_kind}.training_metadata.calibration_max_input_timestamp",
        )
        if calibration_timestamp >= cutoff:
            raise ValueError(
                f"{artifact_kind} calibration timestamp is not strictly before its cutoff"
            )
    generated = _aware_timestamp(
        payload.get("generated_at"),
        field_name=f"{artifact_kind}.generated_at",
    )
    if generated < cutoff:
        raise ValueError(f"{artifact_kind} generated_at is earlier than its information cutoff")
    # retrospective_diagnostic relaxes ONLY this artifact-creation-time boundary (a
    # bundle fit today for an already-completed historical fold is necessarily
    # "generated after" that fold's own cutoff). Every leakage-relevant check above
    # (cutoff vs the prediction's allowed cutoff, input-before-cutoff, calibration
    # disjointness) is unaffected and stays fully enforced.
    if not retrospective_diagnostic and generated > latest_allowed_generated_at:
        raise ValueError(f"{artifact_kind} was generated after its allowed creation boundary")
    return metadata, cutoff


def validate_normalization_artifact(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    checkpoint: str,
    session_kind: str,
    track_class: str,
    latest_allowed_cutoff: datetime,
    latest_allowed_generated_at: datetime,
    retrospective_diagnostic: bool = False,
) -> tuple[FittedPracticeNormalization, datetime]:
    """Validate and materialize one exact normalization artifact."""

    if payload.get("artifact_type") != "qualifying_practice_normalization":
        raise ValueError("not a qualifying_practice_normalization artifact")
    if int(payload.get("schema_version", -1)) != NORMALIZATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported qualifying practice normalization schema")
    _metadata, cutoff = _artifact_metadata(
        payload,
        artifact_kind="normalization",
        candidate_id=candidate_id,
        checkpoint=checkpoint,
        session_kind=session_kind,
        track_class=track_class,
        latest_allowed_cutoff=latest_allowed_cutoff,
        latest_allowed_generated_at=latest_allowed_generated_at,
        retrospective_diagnostic=retrospective_diagnostic,
    )
    raw = payload.get("normalization")
    if not isinstance(raw, Mapping):
        raise ValueError("normalization artifact payload must contain an object")
    coverage = raw.get("comparison_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    effects = raw.get("compound_effect_s")
    effects = effects if isinstance(effects, Mapping) else {}
    coefficient_provenance = raw.get("coefficient_provenance")
    coefficient_provenance = (
        coefficient_provenance if isinstance(coefficient_provenance, Mapping) else {}
    )
    normalization = FittedPracticeNormalization(
        reference_compound=str(raw.get("reference_compound", "SOFT")),
        compound_effect_s={str(key): float(value) for key, value in effects.items()},
        tyre_age_effect_s_per_lap=float(raw.get("tyre_age_effect_s_per_lap", 0.0)),
        evolution_effect_s_per_unit=float(raw.get("evolution_effect_s_per_unit", 0.0)),
        uncertainty_s=float(raw.get("measurement_uncertainty_s", 0.5)),
        provenance=str(raw.get("provenance", "unknown")),
        prior_source=str(raw.get("prior_source", "unknown")),
        comparison_count=int(coverage.get("comparisons", 0)),
        driver_count=int(coverage.get("drivers", 0)),
        team_count=int(coverage.get("teams", 0)),
        empirical_weight=float(raw.get("empirical_weight", 0.0)),
        coefficient_provenance={
            str(key): str(value) for key, value in coefficient_provenance.items()
        },
        fallback_reasons=tuple(str(value) for value in raw.get("fallback_reasons", [])),
    )
    finite_values = (
        *normalization.compound_effect_s.values(),
        normalization.tyre_age_effect_s_per_lap,
        normalization.evolution_effect_s_per_unit,
        normalization.uncertainty_s,
        normalization.empirical_weight,
    )
    if any(not np.isfinite(float(value)) for value in finite_values):
        raise ValueError("normalization artifact contains non-finite values")
    if normalization.uncertainty_s < 0.0:
        raise ValueError("normalization uncertainty must be non-negative")
    return normalization, cutoff


def validate_model_artifact(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    checkpoint: str,
    session_kind: str,
    latest_allowed_cutoff: datetime,
    latest_allowed_generated_at: datetime,
    retrospective_diagnostic: bool = False,
) -> tuple[FittedQualifyingPracticeModel, datetime]:
    """Validate and materialize one exact checkpoint model artifact."""

    if int(payload.get("schema_version", -1)) != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported qualifying practice model schema")
    _metadata, cutoff = _artifact_metadata(
        payload,
        artifact_kind="model",
        candidate_id=candidate_id,
        checkpoint=checkpoint,
        session_kind=session_kind,
        track_class=None,
        latest_allowed_cutoff=latest_allowed_cutoff,
        latest_allowed_generated_at=latest_allowed_generated_at,
        retrospective_diagnostic=retrospective_diagnostic,
    )
    model = FittedQualifyingPracticeModel.from_dict(dict(payload))
    if model.checkpoint != checkpoint:
        raise ValueError("model checkpoint does not match its bundle key")
    return model, cutoff


def _load_json_object(path: Path, *, field_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return payload


def _artifact_reference(
    *,
    artifact_path: Path,
    bundle_directory: Path,
    cutoff: datetime,
) -> dict[str, Any]:
    absolute = artifact_path.resolve()
    root = bundle_directory.resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError("Q1 artifacts must be inside the candidate bundle directory") from exc
    return {
        "path": relative.as_posix(),
        "sha256": file_sha256(absolute),
        "cutoff_timestamp": cutoff.isoformat().replace("+00:00", "Z"),
    }


def build_qualifying_practice_bundle(
    *,
    candidate_id: str,
    variant_id: str,
    manifest: Mapping[str, Any],
    bundle_directory: str | Path,
    model_paths: Sequence[str | Path],
    normalization_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Build one deterministic bundle envelope from fitted immutable artifacts."""

    candidate = _safe_token(candidate_id, field_name="candidate_id")
    variant = _safe_token(variant_id, field_name="variant_id")
    manifest_digest, manifest_cutoff = validate_challenger_manifest_identity(
        manifest,
        candidate_id=candidate,
        variant_id=variant,
    )
    manifest_created = _aware_timestamp(
        manifest.get("created_at"),
        field_name="manifest.created_at",
    )
    root = Path(bundle_directory).resolve()
    models: dict[str, dict[str, Any]] = {}
    normalizations: dict[str, dict[str, dict[str, Any]]] = {}

    for raw_path in sorted((Path(path).resolve() for path in model_paths), key=str):
        payload = _load_json_object(raw_path, field_name="model artifact")
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("model artifact is missing training_metadata")
        checkpoint = str(metadata.get("checkpoint", "")).strip().upper()
        session_kind = str(metadata.get("session_kind", "")).strip().lower()
        if checkpoint not in CHECKPOINTS or session_kind not in SESSION_KINDS:
            raise ValueError("model artifact has unsupported checkpoint/session_kind")
        expected_path = root / "models" / session_kind / f"{checkpoint.lower()}.json"
        if raw_path != expected_path.resolve():
            raise ValueError(f"model artifact must use candidate layout: {expected_path}")
        _model, cutoff = validate_model_artifact(
            payload,
            candidate_id=candidate,
            checkpoint=checkpoint,
            session_kind=session_kind,
            latest_allowed_cutoff=manifest_cutoff,
            latest_allowed_generated_at=manifest_created,
        )
        session_models = models.setdefault(session_kind, {})
        if checkpoint in session_models:
            raise ValueError(f"duplicate model artifact for {session_kind}/{checkpoint}")
        session_models[checkpoint] = _artifact_reference(
            artifact_path=raw_path,
            bundle_directory=root,
            cutoff=cutoff,
        )

    for raw_path in sorted((Path(path).resolve() for path in normalization_paths), key=str):
        payload = _load_json_object(raw_path, field_name="normalization artifact")
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("normalization artifact is missing training_metadata")
        checkpoint = str(metadata.get("checkpoint", "")).strip().upper()
        session_kind = str(metadata.get("session_kind", "")).strip().lower()
        track_class = _safe_token(metadata.get("track_class"), field_name="track_class")
        if checkpoint not in CHECKPOINTS or session_kind not in SESSION_KINDS:
            raise ValueError("normalization artifact has unsupported checkpoint/session_kind")
        expected_path = (
            root / "normalizations" / session_kind / checkpoint.lower() / f"{track_class}.json"
        )
        if raw_path != expected_path.resolve():
            raise ValueError(f"normalization artifact must use candidate layout: {expected_path}")
        _normalization, cutoff = validate_normalization_artifact(
            payload,
            candidate_id=candidate,
            checkpoint=checkpoint,
            session_kind=session_kind,
            track_class=track_class,
            latest_allowed_cutoff=manifest_cutoff,
            latest_allowed_generated_at=manifest_created,
        )
        checkpoint_normalizations = normalizations.setdefault(session_kind, {}).setdefault(
            checkpoint,
            {},
        )
        if track_class in checkpoint_normalizations:
            raise ValueError(
                f"duplicate normalization artifact for {session_kind}/{checkpoint}/{track_class}"
            )
        checkpoint_normalizations[track_class] = _artifact_reference(
            artifact_path=raw_path,
            bundle_directory=root,
            cutoff=cutoff,
        )

    if not models:
        raise ValueError("Q1 bundle requires at least one checkpoint model")
    bundle: dict[str, Any] = {
        "artifact_type": "qualifying_practice_bundle",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "candidate_id": candidate,
        "variant_id": variant,
        "manifest_sha256": manifest_digest,
        "manifest_cutoff_at": manifest_cutoff.isoformat().replace("+00:00", "Z"),
        "manifest_created_at": manifest_created.isoformat().replace("+00:00", "Z"),
        "created_at": _actual_creation_timestamp(
            not_before=manifest_created,
            artifact_kind="Q1 bundle",
        ),
        "dry_only": True,
        "models": models,
        "normalizations": normalizations,
    }
    bundle["bundle_sha256"] = stable_json_sha256(bundle)
    return bundle


def _repository_display_path(path: Path, *, repo_root: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _configured_path(value: Any, *, repo_root: Path, field_name: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field_name} must not be blank")
    path = Path(raw)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _normalise_launch_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "model_variant",
        "candidate_id",
        "launch_envelope_path",
        "bundle_path",
        "track_class_by_event",
        "uncertainty_scale",
    }
    if set(settings) != expected_fields:
        raise ValueError("Q1 semantic settings do not match the launch schema")
    variant = _safe_token(settings.get("model_variant"), field_name="model_variant")
    candidate = _safe_token(settings.get("candidate_id"), field_name="candidate_id")
    raw_track_classes = settings.get("track_class_by_event")
    if not isinstance(raw_track_classes, Mapping):
        raise ValueError("track_class_by_event must be an object")
    track_classes: dict[str, str] = {}
    for raw_event, raw_track_class in sorted(
        raw_track_classes.items(),
        key=lambda item: str(item[0]),
    ):
        event = str(raw_event).strip()
        if not event:
            raise ValueError("track_class_by_event contains a blank event")
        track_classes[event] = _safe_token(raw_track_class, field_name="track_class")
    raw_uncertainty_scale = settings.get("uncertainty_scale")
    if raw_uncertainty_scale is None:
        raise ValueError("uncertainty_scale must be numeric")
    try:
        uncertainty_scale = float(raw_uncertainty_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("uncertainty_scale must be numeric") from exc
    if not np.isfinite(uncertainty_scale) or uncertainty_scale < 0.0:
        raise ValueError("uncertainty_scale must be finite and non-negative")
    launch_path = str(settings.get("launch_envelope_path") or "").strip()
    bundle_path = str(settings.get("bundle_path") or "").strip()
    if not launch_path or not bundle_path:
        raise ValueError("launch_envelope_path and bundle_path must not be blank")
    return {
        "model_variant": variant,
        "candidate_id": candidate,
        "launch_envelope_path": launch_path.replace("\\", "/"),
        "bundle_path": bundle_path.replace("\\", "/"),
        "track_class_by_event": track_classes,
        "uncertainty_scale": uncertainty_scale,
    }


def _candidate_definition_settings(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Q1 candidate definition must contain a YAML/JSON object")
    expected_fields = {
        "artifact_type",
        "schema_version",
        "model_variant",
        "candidate_id",
        "launch_envelope_path",
        "bundle_path",
        "track_class_by_event",
        "uncertainty_scale",
    }
    if set(payload) != expected_fields:
        raise ValueError("Q1 candidate definition fields do not match schema v1")
    if payload.get("artifact_type") != "qualifying_practice_candidate_definition":
        raise ValueError("Q1 candidate definition has the wrong artifact_type")
    if payload.get("schema_version") != 1:
        raise ValueError("Q1 candidate definition has an unsupported schema_version")
    return _normalise_launch_settings(
        {
            key: value
            for key, value in payload.items()
            if key not in {"artifact_type", "schema_version"}
        }
    )


def load_qualifying_practice_candidate_definition(path: str | Path) -> dict[str, Any]:
    """Load and normalize the standalone semantic Q1 candidate definition."""

    source = Path(path)
    return _candidate_definition_settings(yaml.safe_load(source.read_text(encoding="utf-8")))


def _validate_bundle_identity(
    bundle: Mapping[str, Any],
    *,
    candidate_id: str,
    variant_id: str,
    manifest_sha256: str,
) -> str:
    if set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("Q1 bundle fields do not match schema v1")
    if bundle.get("artifact_type") != "qualifying_practice_bundle":
        raise ValueError("not a qualifying_practice_bundle artifact")
    if int(bundle.get("schema_version", -1)) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported qualifying practice bundle schema")
    bundle_digest = _hex_digest(bundle.get("bundle_sha256"), field_name="bundle_sha256")
    if bundle_digest != stable_json_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    ):
        raise ValueError("bundle_sha256 does not match the Q1 bundle payload")
    if str(bundle.get("candidate_id", "")).strip() != candidate_id:
        raise ValueError("Q1 bundle candidate_id does not match the launch envelope")
    if str(bundle.get("variant_id", "")).strip().lower() != variant_id:
        raise ValueError("Q1 bundle variant_id does not match the launch envelope")
    if str(bundle.get("manifest_sha256", "")).strip().lower() != manifest_sha256:
        raise ValueError("Q1 bundle manifest digest does not match the launch envelope")
    if bundle.get("dry_only") is not True:
        raise ValueError("Q1 bundle must be dry-only")
    return bundle_digest


def build_qualifying_practice_launch_envelope(
    *,
    candidate_id: str,
    variant_id: str,
    manifest: Mapping[str, Any],
    bundle_path: str | Path,
    launch_directory: str | Path,
    semantic_config_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the final acyclic launch binding for one Q1 candidate."""

    root = Path(launch_directory).resolve()
    repository = Path(repo_root).resolve()
    candidate = _safe_token(candidate_id, field_name="candidate_id")
    variant = _safe_token(variant_id, field_name="variant_id")
    manifest_digest, manifest_cutoff = validate_challenger_manifest_identity(
        manifest,
        candidate_id=candidate,
        variant_id=variant,
    )
    manifest_created = _aware_timestamp(
        manifest.get("created_at"),
        field_name="manifest.created_at",
    )
    settings = load_qualifying_practice_candidate_definition(semantic_config_path)
    if settings["candidate_id"] != candidate or settings["model_variant"] != variant:
        raise ValueError("semantic Q1 candidate identity does not match the manifest")

    source_config = Path(semantic_config_path).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(f"Q1 semantic config does not exist: {source_config}")
    config_display_path = _repository_display_path(source_config, repo_root=repository)
    config_digest = file_sha256(source_config)
    try:
        semantic_source_text = source_config.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Q1 semantic config must be UTF-8") from exc
    provenance = manifest.get("provenance")
    configuration = provenance.get("configuration") if isinstance(provenance, Mapping) else None
    config_files = configuration.get("files") if isinstance(configuration, Mapping) else None
    if not isinstance(config_files, Sequence) or isinstance(config_files, str | bytes):
        raise ValueError("manifest has no configuration provenance")
    matching_config = [
        item
        for item in config_files
        if isinstance(item, Mapping) and str(item.get("path", "")) == config_display_path
    ]
    if len(matching_config) != 1 or str(matching_config[0].get("sha256", "")) != config_digest:
        raise ValueError("semantic Q1 config is not frozen by the manifest")

    source_bundle = Path(bundle_path).resolve()
    if source_bundle != (root / "bundle.json").resolve():
        raise ValueError("Q1 launch requires candidate_root/bundle.json")
    if (
        _configured_path(
            settings["bundle_path"],
            repo_root=repository,
            field_name="bundle_path",
        )
        != source_bundle
    ):
        raise ValueError("semantic Q1 bundle_path does not match the candidate bundle")
    expected_launch_path = (root / "launch.json").resolve()
    if (
        _configured_path(
            settings["launch_envelope_path"],
            repo_root=repository,
            field_name="launch_envelope_path",
        )
        != expected_launch_path
    ):
        raise ValueError(
            "semantic Q1 launch_envelope_path does not match candidate_root/launch.json"
        )

    bundle = _load_json_object(source_bundle, field_name="Q1 bundle")
    bundle_digest = _validate_bundle_identity(
        bundle,
        candidate_id=candidate,
        variant_id=variant,
        manifest_sha256=manifest_digest,
    )
    bundle_manifest_created = _aware_timestamp(
        bundle.get("manifest_created_at"),
        field_name="bundle.manifest_created_at",
    )
    if bundle_manifest_created != manifest_created:
        raise ValueError("Q1 bundle manifest creation time does not match the manifest")
    bundle_created = _aware_timestamp(bundle.get("created_at"), field_name="bundle.created_at")
    if bundle_created < manifest_created:
        raise ValueError("Q1 bundle was created before its manifest")
    settings_digest = stable_json_sha256(settings)
    envelope: dict[str, Any] = {
        "artifact_type": "qualifying_practice_launch_envelope",
        "schema_version": LAUNCH_ENVELOPE_SCHEMA_VERSION,
        "candidate_id": candidate,
        "variant_id": variant,
        "manifest_sha256": manifest_digest,
        "manifest": dict(manifest),
        "manifest_cutoff_at": manifest_cutoff.isoformat().replace("+00:00", "Z"),
        "manifest_created_at": manifest_created.isoformat().replace("+00:00", "Z"),
        "created_at": _actual_creation_timestamp(
            not_before=bundle_created,
            artifact_kind="Q1 launch envelope",
        ),
        "dry_only": True,
        "bundle": {
            "path": "bundle.json",
            "sha256": file_sha256(source_bundle),
            "bundle_sha256": bundle_digest,
        },
        "semantic_config": {
            "path": config_display_path,
            "sha256": config_digest,
            "source_text": semantic_source_text,
            "settings_sha256": settings_digest,
            "settings": settings,
        },
        "layout": {
            "model": "models/{session_kind}/{checkpoint}.json",
            "normalization": ("normalizations/{session_kind}/{checkpoint}/{track_class}.json"),
        },
    }
    envelope["launch_sha256"] = stable_json_sha256(envelope)
    return envelope


def _resolved_reference(
    reference: Any,
    *,
    bundle_directory: Path,
    field_name: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(reference, Mapping):
        raise ValueError(f"bundle {field_name} reference must be an object")
    raw_path = str(reference.get("path", "")).strip()
    if not raw_path or Path(raw_path).is_absolute():
        raise ValueError(f"bundle {field_name} path must be relative")
    root = bundle_directory.resolve()
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bundle {field_name} path escapes the candidate directory") from exc
    expected_digest = _hex_digest(reference.get("sha256"), field_name=f"{field_name}.sha256")
    if not path.is_file() or file_sha256(path) != expected_digest:
        raise ValueError(f"bundle {field_name} artifact is missing or its digest changed")
    return path, reference


def resolve_qualifying_practice_bundle(
    bundle_path: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_candidate_id: str,
    expected_variant_id: str,
    expected_manifest_sha256: str,
    checkpoint: str,
    session_kind: str,
    track_class: str,
    inference_cutoff: datetime,
    require_normalization: bool,
    retrospective_diagnostic: bool = False,
) -> ResolvedQualifyingPracticeBundle:
    """Resolve and validate the exact Q1 artifacts for one prediction context."""

    source = Path(bundle_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Q1 bundle does not exist: {source}")
    bundle = _load_json_object(source, field_name="Q1 bundle")
    if set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("Q1 bundle fields do not match schema v1")
    if bundle.get("artifact_type") != "qualifying_practice_bundle":
        raise ValueError("not a qualifying_practice_bundle artifact")
    if int(bundle.get("schema_version", -1)) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported qualifying practice bundle schema")
    digest = _hex_digest(bundle.get("bundle_sha256"), field_name="bundle_sha256")
    configured_bundle_digest = _hex_digest(
        expected_bundle_sha256,
        field_name="expected_bundle_sha256",
    )
    if digest != configured_bundle_digest:
        raise ValueError("Q1 bundle digest does not match configuration")
    if digest != stable_json_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    ):
        raise ValueError("bundle_sha256 does not match the Q1 bundle payload")

    candidate = _safe_token(expected_candidate_id, field_name="expected_candidate_id")
    variant = _safe_token(expected_variant_id, field_name="expected_variant_id")
    manifest_digest = _hex_digest(
        expected_manifest_sha256,
        field_name="expected_manifest_sha256",
    )
    if str(bundle.get("candidate_id", "")).strip() != candidate:
        raise ValueError("Q1 bundle candidate_id does not match configuration")
    if str(bundle.get("variant_id", "")).strip().lower() != variant:
        raise ValueError("Q1 bundle variant_id does not match the selected model variant")
    if str(bundle.get("manifest_sha256", "")).strip().lower() != manifest_digest:
        raise ValueError("Q1 bundle manifest digest does not match configuration")
    if bundle.get("dry_only") is not True:
        raise ValueError("Q1 bundle must be dry-only")

    resolved_checkpoint = str(checkpoint).strip().upper()
    resolved_session_kind = str(session_kind).strip().lower()
    resolved_track_class = _safe_token(track_class, field_name="track_class")
    if resolved_checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported Q1 checkpoint: {resolved_checkpoint}")
    if resolved_session_kind not in SESSION_KINDS:
        raise ValueError(f"unsupported Q1 session_kind: {resolved_session_kind}")
    cutoff = inference_cutoff
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("Q1 inference cutoff must include a timezone")
    cutoff = cutoff.astimezone(UTC)
    manifest_cutoff = _aware_timestamp(
        bundle.get("manifest_cutoff_at"),
        field_name="bundle.manifest_cutoff_at",
    )
    manifest_created = _aware_timestamp(
        bundle.get("manifest_created_at"),
        field_name="bundle.manifest_created_at",
    )
    created_at = _aware_timestamp(bundle.get("created_at"), field_name="bundle.created_at")
    if not manifest_cutoff <= manifest_created <= created_at:
        raise ValueError("Q1 bundle chronology is invalid")
    if not retrospective_diagnostic and created_at > cutoff:
        raise ValueError("Q1 bundle was frozen after the prediction cutoff")
    artifact_cutoff = min(cutoff, manifest_cutoff)

    models = bundle.get("models")
    session_models = models.get(resolved_session_kind) if isinstance(models, Mapping) else None
    model_reference = (
        session_models.get(resolved_checkpoint) if isinstance(session_models, Mapping) else None
    )
    model_path, _model_reference = _resolved_reference(
        model_reference,
        bundle_directory=source.parent,
        field_name="model",
    )
    model_payload = _load_json_object(model_path, field_name="model artifact")
    model, model_cutoff = validate_model_artifact(
        model_payload,
        candidate_id=candidate,
        checkpoint=resolved_checkpoint,
        session_kind=resolved_session_kind,
        latest_allowed_cutoff=artifact_cutoff,
        latest_allowed_generated_at=cutoff,
        retrospective_diagnostic=retrospective_diagnostic,
    )

    normalization: FittedPracticeNormalization | None = None
    normalization_cutoff: datetime | None = None
    if require_normalization:
        normalizations = bundle.get("normalizations")
        session_normalizations = (
            normalizations.get(resolved_session_kind)
            if isinstance(normalizations, Mapping)
            else None
        )
        checkpoint_normalizations = (
            session_normalizations.get(resolved_checkpoint)
            if isinstance(session_normalizations, Mapping)
            else None
        )
        normalization_reference = (
            checkpoint_normalizations.get(resolved_track_class)
            if isinstance(checkpoint_normalizations, Mapping)
            else None
        )
        normalization_path, _normalization_reference = _resolved_reference(
            normalization_reference,
            bundle_directory=source.parent,
            field_name="normalization",
        )
        normalization_payload = _load_json_object(
            normalization_path,
            field_name="normalization artifact",
        )
        normalization, normalization_cutoff = validate_normalization_artifact(
            normalization_payload,
            candidate_id=candidate,
            checkpoint=resolved_checkpoint,
            session_kind=resolved_session_kind,
            track_class=resolved_track_class,
            latest_allowed_cutoff=artifact_cutoff,
            latest_allowed_generated_at=cutoff,
            retrospective_diagnostic=retrospective_diagnostic,
        )

    return ResolvedQualifyingPracticeBundle(
        model=model,
        normalization=normalization,
        diagnostics={
            "bundle_digest": f"sha256:{digest}",
            "manifest_digest": f"sha256:{manifest_digest}",
            "candidate_id": candidate,
            "variant_id": variant,
            "checkpoint": resolved_checkpoint,
            "session_kind": resolved_session_kind,
            "track_class": resolved_track_class,
            "model_cutoff_at": model_cutoff.isoformat().replace("+00:00", "Z"),
            "normalization_cutoff_at": (
                normalization_cutoff.isoformat().replace("+00:00", "Z")
                if normalization_cutoff is not None
                else None
            ),
            "retrospective_diagnostic": bool(retrospective_diagnostic),
        },
    )


def resolve_qualifying_practice_launch_envelope(
    launch_path: str | Path,
    *,
    expected_variant_id: str,
    event_year: int,
    race_name: str,
    checkpoint: str,
    session_kind: str,
    inference_cutoff: datetime,
    require_normalization: bool,
    retrospective_diagnostic: bool = False,
) -> ResolvedQualifyingPracticeLaunch:
    """Validate one launch envelope and resolve its exact runtime artifacts.

    ``retrospective_diagnostic`` defaults to False (identical behavior to before this
    parameter existed). When explicitly set True by an offline research caller, it
    relaxes only the envelope/bundle/artifact *creation-time* boundary against
    ``inference_cutoff`` -- never the input-before-cutoff leakage guards, hash
    checks, or train/calibration disjointness, which stay fully enforced. Every
    resolution performed with it set carries ``retrospective_diagnostic: true`` in
    ``diagnostics`` so it can never be mistaken for a live/preregistered result.
    """

    source = Path(launch_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Q1 launch envelope does not exist: {source}")
    envelope = _load_json_object(source, field_name="Q1 launch envelope")
    expected_fields = {
        "artifact_type",
        "schema_version",
        "candidate_id",
        "variant_id",
        "manifest_sha256",
        "manifest",
        "manifest_cutoff_at",
        "manifest_created_at",
        "created_at",
        "dry_only",
        "bundle",
        "semantic_config",
        "layout",
        "launch_sha256",
    }
    if set(envelope) != expected_fields:
        raise ValueError("Q1 launch envelope fields do not match schema v1")
    if envelope.get("artifact_type") != "qualifying_practice_launch_envelope":
        raise ValueError("not a qualifying_practice_launch_envelope artifact")
    if int(envelope.get("schema_version", -1)) != LAUNCH_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("unsupported qualifying practice launch schema")
    launch_digest = _hex_digest(envelope.get("launch_sha256"), field_name="launch_sha256")
    if launch_digest != stable_json_sha256(
        {key: value for key, value in envelope.items() if key != "launch_sha256"}
    ):
        raise ValueError("launch_sha256 does not match the Q1 launch payload")
    if envelope.get("dry_only") is not True:
        raise ValueError("Q1 launch envelope must be dry-only")

    variant = _safe_token(expected_variant_id, field_name="expected_variant_id")
    candidate = _safe_token(envelope.get("candidate_id"), field_name="candidate_id")
    if str(envelope.get("variant_id", "")).strip().lower() != variant:
        raise ValueError("Q1 launch variant does not match the selected model variant")
    manifest_digest = _hex_digest(
        envelope.get("manifest_sha256"),
        field_name="manifest_sha256",
    )
    raw_manifest = envelope.get("manifest")
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("Q1 launch envelope is missing its frozen manifest")
    validated_manifest = validate_challenger_manifest(
        raw_manifest,
        expected_variant_id=variant,
    )
    if (
        validated_manifest.candidate_id != candidate
        or validated_manifest.manifest_sha256 != manifest_digest
    ):
        raise ValueError("Q1 launch manifest identity is inconsistent")
    cutoff = inference_cutoff
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("Q1 inference cutoff must include a timezone")
    cutoff = cutoff.astimezone(UTC)
    manifest_cutoff = _aware_timestamp(
        envelope.get("manifest_cutoff_at"),
        field_name="launch.manifest_cutoff_at",
    )
    manifest_created = _aware_timestamp(
        envelope.get("manifest_created_at"),
        field_name="launch.manifest_created_at",
    )
    if (
        manifest_cutoff != validated_manifest.cutoff_at
        or manifest_created != validated_manifest.created_at
    ):
        raise ValueError("Q1 launch manifest chronology is inconsistent")
    created_at = _aware_timestamp(envelope.get("created_at"), field_name="launch.created_at")
    if not manifest_cutoff <= manifest_created <= created_at:
        raise ValueError("Q1 launch envelope chronology is invalid")
    if not retrospective_diagnostic and created_at > cutoff:
        raise ValueError("Q1 launch envelope was frozen after the prediction cutoff")

    layout = envelope.get("layout")
    expected_layout = {
        "model": "models/{session_kind}/{checkpoint}.json",
        "normalization": "normalizations/{session_kind}/{checkpoint}/{track_class}.json",
    }
    if not isinstance(layout, Mapping) or dict(layout) != expected_layout:
        raise ValueError("Q1 launch artifact layout is invalid")

    semantic = envelope.get("semantic_config")
    if not isinstance(semantic, Mapping) or set(semantic) != {
        "path",
        "sha256",
        "source_text",
        "settings_sha256",
        "settings",
    }:
        raise ValueError("Q1 launch semantic config binding is invalid")
    semantic_config_digest = _hex_digest(
        semantic.get("sha256"),
        field_name="semantic_config.sha256",
    )
    semantic_source_text = semantic.get("source_text")
    if not isinstance(semantic_source_text, str):
        raise ValueError("Q1 launch semantic source text is invalid")
    if hashlib.sha256(semantic_source_text.encode("utf-8")).hexdigest() != semantic_config_digest:
        raise ValueError("Q1 semantic source text does not match its manifest digest")
    settings_digest = _hex_digest(
        semantic.get("settings_sha256"),
        field_name="semantic_config.settings_sha256",
    )
    raw_settings = semantic.get("settings")
    if not isinstance(raw_settings, Mapping):
        raise ValueError("Q1 launch semantic settings must be an object")
    settings = _normalise_launch_settings(raw_settings)
    settings_from_source = _candidate_definition_settings(yaml.safe_load(semantic_source_text))
    if settings_from_source != settings:
        raise ValueError("Q1 semantic settings do not match the frozen candidate definition")
    if settings_digest != stable_json_sha256(settings):
        raise ValueError("Q1 semantic settings digest does not match its payload")
    if settings["candidate_id"] != candidate or settings["model_variant"] != variant:
        raise ValueError("Q1 launch semantic identity is inconsistent")
    semantic_path = str(semantic.get("path", "")).strip()
    manifest_configuration = raw_manifest.get("provenance")
    manifest_configuration = (
        manifest_configuration.get("configuration")
        if isinstance(manifest_configuration, Mapping)
        else None
    )
    manifest_config_files = (
        manifest_configuration.get("files") if isinstance(manifest_configuration, Mapping) else None
    )
    matching_config = [
        item
        for item in manifest_config_files or []
        if isinstance(item, Mapping) and str(item.get("path", "")) == semantic_path
    ]
    if (
        not semantic_path
        or len(matching_config) != 1
        or str(matching_config[0].get("sha256", "")) != semantic_config_digest
    ):
        raise ValueError("Q1 semantic config is not bound by the frozen manifest")

    track_class = "pre"
    if require_normalization:
        track_classes = settings["track_class_by_event"]
        raw_track_class = track_classes.get(f"{int(event_year)}:{race_name}")
        if raw_track_class is None:
            raw_track_class = track_classes.get(str(race_name))
        track_class = str(raw_track_class or "").strip().lower()
        if not track_class:
            raise MissingQualifyingPracticeTrackClassError(
                "Q1 launch has no track class for the prediction event"
            )

    bundle_reference = envelope.get("bundle")
    if not isinstance(bundle_reference, Mapping) or set(bundle_reference) != {
        "path",
        "sha256",
        "bundle_sha256",
    }:
        raise ValueError("Q1 launch bundle binding is invalid")
    bundle_path, _reference = _resolved_reference(
        bundle_reference,
        bundle_directory=source.parent,
        field_name="bundle",
    )
    if bundle_path != (source.parent / "bundle.json").resolve():
        raise ValueError("Q1 launch bundle must be candidate_root/bundle.json")
    bundle_digest = _hex_digest(
        bundle_reference.get("bundle_sha256"),
        field_name="launch.bundle.bundle_sha256",
    )
    resolved_bundle = resolve_qualifying_practice_bundle(
        bundle_path,
        expected_bundle_sha256=bundle_digest,
        expected_candidate_id=candidate,
        expected_variant_id=variant,
        expected_manifest_sha256=manifest_digest,
        checkpoint=checkpoint,
        session_kind=session_kind,
        track_class=track_class,
        inference_cutoff=cutoff,
        require_normalization=require_normalization,
        retrospective_diagnostic=retrospective_diagnostic,
    )
    return ResolvedQualifyingPracticeLaunch(
        model=resolved_bundle.model,
        normalization=resolved_bundle.normalization,
        uncertainty_scale=float(settings["uncertainty_scale"]),
        diagnostics={
            **resolved_bundle.diagnostics,
            "launch_digest": f"sha256:{launch_digest}",
            "semantic_config_digest": f"sha256:{semantic_config_digest}",
            "semantic_settings_digest": f"sha256:{settings_digest}",
        },
    )
