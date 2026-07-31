"""Leakage-safe checkpoint walk-forward orchestration for model challengers.

The production predictor is intentionally not imported here.  A research backend
owns fold fitting and prediction construction, while this module owns the parts that
must not be left to convention: chronological event ordering, disjoint fold inputs,
dry-only filtering, common random seeds, complete-grid validation, and event-equal
metrics.  This makes the same runner usable for Q-only and qualifying-to-race
challengers without allowing it to activate a runtime model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Protocol, cast

from src.analysis.challenger_governance import (
    DEFAULT_REPLAY_CHECKPOINTS,
    DEFAULT_REPLAY_SEEDS,
    QualifyingGateMetrics,
    RaceGateMetrics,
    ReplayProvenance,
    ValidatedChallengerManifest,
    build_race_metric_views,
    paired_weekend_bootstrap,
    stable_json_sha256,
    validate_challenger_manifest,
)
from src.analysis.challenger_movements import build_full_field_movement_audit
from src.analysis.model_evaluation import (
    compute_dnf_calibration,
    compute_prediction_accuracy,
)
from src.models.challenger_variants import CHAMPION_VARIANT
from src.persistence.research_sidecar import ResearchSidecarStore
from src.utils.grid_validation import validate_qualifying_grid

WALK_FORWARD_SCHEMA_VERSION = 1
SESSION_KINDS = frozenset({"main", "sprint"})
MINIMUM_Q1_TRAINING_EVENTS = {"main": 30, "sprint": 8}
MINIMUM_R2_TRAINING_EVENTS = 8
_RACE_VIEWS = ("conditional_actual_grid", "end_to_end_predicted_grid")
_FORECAST_BUNDLE_SCHEMA_VERSION = 1
_FORECAST_BUNDLE_ARTIFACT_TYPE = "challenger_walk_forward_checkpoint_forecasts"
_FORECAST_ARTIFACT_KIND_PREFIX = "walk_forward_checkpoint_"
_FORECAST_REGISTRATIONS = frozenset({"preregistered_shadow", "retrospective_diagnostic"})


@dataclass(frozen=True)
class ReplayEvent:
    """One immutable weekend input to the chronological replay."""

    event_id: str
    event_start_at: datetime
    qualifying_start_at: datetime
    session_kind: Literal["main", "sprint"]
    is_dry: bool
    checkpoint_payloads: Mapping[str, Mapping[str, Any]]
    actual_qualifying_grid: tuple[Mapping[str, Any], ...]
    actual_race_finish_order: tuple[Mapping[str, Any], ...] | None
    input_snapshot_ids: tuple[str, ...]
    payload: Mapping[str, Any]


class CheckpointInputUnavailable(RuntimeError):
    """A backend raises this to fail closed on one event-checkpoint's own inputs.

    Distinct from a bare exception so the runner can skip exactly this
    event-checkpoint and keep scoring every other eligible one, while any other
    exception (a real bug, a structural validation failure) still aborts the whole
    replay loudly instead of being silently swallowed.
    """


class WalkForwardBackend(Protocol):
    """Research adapter used by :func:`run_challenger_walk_forward`.

    Implementations should create fresh predictor instances inside each prediction
    call.  The runner passes the same seed to champion and challenger calls but never
    shares mutable predictors between them.
    """

    def fit_fold(
        self,
        *,
        training_events: Sequence[ReplayEvent],
        calibration_events: Sequence[ReplayEvent],
        target_event: ReplayEvent,
        checkpoint: str,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def predict_qualifying(
        self,
        *,
        role: Literal["champion", "challenger"],
        seed: int,
        event: ReplayEvent,
        checkpoint: str,
        checkpoint_payload: Mapping[str, Any],
        fold_artifacts: Mapping[str, Any] | None,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def predict_race_views(
        self,
        *,
        role: Literal["champion", "challenger"],
        seed: int,
        event: ReplayEvent,
        checkpoint: str,
        checkpoint_payload: Mapping[str, Any],
        qualifying_prediction: Mapping[str, Any],
        fold_artifacts: Mapping[str, Any] | None,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]: ...


def _sha256_text(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a raw lowercase SHA-256 digest")
    return digest


def _normalise_seed_forecasts(
    payload: Mapping[str, Any],
    *,
    section: str,
    variant_id: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(payload) != {"champion", "challenger"}:
        raise ValueError(f"{section} must contain exactly champion and challenger")
    expected_seed_keys = {str(seed) for seed in DEFAULT_REPLAY_SEEDS}
    normalised: dict[str, dict[str, dict[str, Any]]] = {}
    for role, expected_variant in (
        ("champion", CHAMPION_VARIANT),
        ("challenger", variant_id),
    ):
        raw_role = payload.get(role)
        if not isinstance(raw_role, Mapping):
            raise ValueError(f"{section}.{role} must be a seed mapping")
        seed_payloads = {str(key): value for key, value in raw_role.items()}
        if len(seed_payloads) != len(raw_role) or set(seed_payloads) != expected_seed_keys:
            raise ValueError(f"{section}.{role} seeds must be exactly {sorted(expected_seed_keys)}")
        resolved_role: dict[str, dict[str, Any]] = {}
        for seed_key in sorted(seed_payloads, key=int):
            prediction = seed_payloads[seed_key]
            if not isinstance(prediction, Mapping):
                raise ValueError(f"{section}.{role}.{seed_key} must be an object")
            prediction_copy = dict(prediction)
            if str(prediction_copy.get("model_variant", "")).strip().lower() != expected_variant:
                raise ValueError(f"{section}.{role}.{seed_key} has the wrong model_variant")
            resolved_role[seed_key] = prediction_copy
        normalised[role] = resolved_role
    return normalised


def _normalise_race_seed_forecasts(
    payload: Mapping[str, Any],
    *,
    variant_id: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(payload) != {"champion", "challenger"}:
        raise ValueError("race_view_predictions must contain exactly champion and challenger")
    expected_seed_keys = {str(seed) for seed in DEFAULT_REPLAY_SEEDS}
    normalised: dict[str, dict[str, dict[str, Any]]] = {}
    for role, expected_variant in (
        ("champion", CHAMPION_VARIANT),
        ("challenger", variant_id),
    ):
        raw_role = payload.get(role)
        if not isinstance(raw_role, Mapping):
            raise ValueError(f"race_view_predictions.{role} must be a seed mapping")
        seed_payloads = {str(key): value for key, value in raw_role.items()}
        if len(seed_payloads) != len(raw_role) or set(seed_payloads) != expected_seed_keys:
            raise ValueError(
                f"race_view_predictions.{role} seeds must be exactly {sorted(expected_seed_keys)}"
            )
        resolved_role: dict[str, dict[str, Any]] = {}
        for seed_key in sorted(seed_payloads, key=int):
            raw_views = seed_payloads[seed_key]
            if not isinstance(raw_views, Mapping) or set(raw_views) != set(_RACE_VIEWS):
                raise ValueError(
                    f"race_view_predictions.{role}.{seed_key} must contain both race views"
                )
            resolved_views: dict[str, Any] = {}
            for view in _RACE_VIEWS:
                prediction = raw_views.get(view)
                if not isinstance(prediction, Mapping):
                    raise ValueError(
                        f"race_view_predictions.{role}.{seed_key}.{view} must be an object"
                    )
                prediction_copy = dict(prediction)
                if (
                    str(prediction_copy.get("model_variant", "")).strip().lower()
                    != expected_variant
                ):
                    raise ValueError(
                        f"race_view_predictions.{role}.{seed_key}.{view} "
                        "has the wrong model_variant"
                    )
                resolved_views[view] = prediction_copy
            resolved_role[seed_key] = resolved_views
        normalised[role] = resolved_role
    return normalised


def build_frozen_checkpoint_forecast_bundle(
    *,
    manifest: Mapping[str, Any],
    event_id: str,
    event_start_at: str | datetime,
    session_kind: Literal["main", "sprint"],
    checkpoint: str,
    information_cutoff_at: str | datetime,
    qualifying_start_at: str | datetime,
    frozen_at: str | datetime,
    qualifying_predictions: Mapping[str, Any],
    race_view_predictions: Mapping[str, Any] | None = None,
    fold_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one digest-bound checkpoint bundle without embedding any actual result."""

    validated_manifest = validate_challenger_manifest(manifest)
    resolved_event_id = str(event_id).strip()
    if not resolved_event_id:
        raise ValueError("checkpoint forecast event_id must not be blank")
    resolved_event_start = _aware_utc(
        event_start_at,
        field_name="checkpoint_forecast.event_start_at",
    )
    resolved_session_kind = str(session_kind).strip().lower()
    if resolved_session_kind not in SESSION_KINDS:
        raise ValueError("checkpoint forecast session_kind must be 'main' or 'sprint'")
    resolved_checkpoint = str(checkpoint).strip().upper()
    if resolved_checkpoint not in DEFAULT_REPLAY_CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint forecast checkpoint: {resolved_checkpoint!r}")
    information_cutoff = _aware_utc(
        information_cutoff_at,
        field_name="checkpoint_forecast.information_cutoff_at",
    )
    qualifying_start = _aware_utc(
        qualifying_start_at,
        field_name="checkpoint_forecast.qualifying_start_at",
    )
    frozen = _aware_utc(frozen_at, field_name="checkpoint_forecast.frozen_at")
    if not resolved_event_start <= information_cutoff < qualifying_start:
        raise ValueError("checkpoint information cutoff must be inside the pre-Q window")
    if frozen < validated_manifest.created_at:
        raise ValueError("checkpoint forecast cannot be frozen before its manifest exists")

    normalized_qualifying = _normalise_seed_forecasts(
        qualifying_predictions,
        section="qualifying_predictions",
        variant_id=validated_manifest.variant_id,
    )
    normalized_race: dict[str, dict[str, dict[str, Any]]] | None = None
    if race_view_predictions is not None:
        normalized_race = _normalise_race_seed_forecasts(
            race_view_predictions,
            variant_id=validated_manifest.variant_id,
        )

    registration = (
        "preregistered_shadow"
        if validated_manifest.created_at < qualifying_start and frozen < qualifying_start
        else "retrospective_diagnostic"
    )
    payload: dict[str, Any] = {
        "artifact_type": _FORECAST_BUNDLE_ARTIFACT_TYPE,
        "schema_version": _FORECAST_BUNDLE_SCHEMA_VERSION,
        "manifest_sha256": validated_manifest.manifest_sha256,
        "candidate_id": validated_manifest.candidate_id,
        "variant_id": validated_manifest.variant_id,
        "event_id": resolved_event_id,
        "event_start_at": resolved_event_start.isoformat().replace("+00:00", "Z"),
        "session_kind": resolved_session_kind,
        "checkpoint": resolved_checkpoint,
        "information_cutoff_at": information_cutoff.isoformat().replace("+00:00", "Z"),
        "qualifying_start_at": qualifying_start.isoformat().replace("+00:00", "Z"),
        "frozen_at": frozen.isoformat().replace("+00:00", "Z"),
        "registration": registration,
        "fold_artifacts": dict(fold_artifacts) if fold_artifacts is not None else None,
        "qualifying_predictions": normalized_qualifying,
        "race_view_predictions": normalized_race,
    }
    payload["bundle_sha256"] = stable_json_sha256(payload)
    return payload


def freeze_checkpoint_forecast_bundle(
    *,
    store: ResearchSidecarStore,
    manifest: Mapping[str, Any],
    event_id: str,
    event_start_at: str | datetime,
    session_kind: Literal["main", "sprint"],
    checkpoint: str,
    information_cutoff_at: str | datetime,
    qualifying_start_at: str | datetime,
    frozen_at: str | datetime,
    qualifying_predictions: Mapping[str, Any],
    race_view_predictions: Mapping[str, Any] | None = None,
    fold_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one immutable checkpoint bundle and return its exact reference."""

    bundle = build_frozen_checkpoint_forecast_bundle(
        manifest=manifest,
        event_id=event_id,
        event_start_at=event_start_at,
        session_kind=session_kind,
        checkpoint=checkpoint,
        information_cutoff_at=information_cutoff_at,
        qualifying_start_at=qualifying_start_at,
        frozen_at=frozen_at,
        qualifying_predictions=qualifying_predictions,
        race_view_predictions=race_view_predictions,
        fold_artifacts=fold_artifacts,
    )
    bundle_digest = str(bundle["bundle_sha256"])
    artifact_kind = f"{_FORECAST_ARTIFACT_KIND_PREFIX}{bundle_digest[:16]}"
    path = store.write_artifact(
        manifest=manifest,
        artifact_kind=artifact_kind,
        payload=bundle,
    )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        "path": str(path.resolve()),
        "artifact_sha256": str(envelope["artifact_sha256"]),
        "manifest_sha256": str(bundle["manifest_sha256"]),
        "bundle_sha256": bundle_digest,
        "event_id": str(bundle["event_id"]),
        "event_start_at": str(bundle["event_start_at"]),
        "session_kind": str(bundle["session_kind"]),
        "checkpoint": str(bundle["checkpoint"]),
        "information_cutoff_at": str(bundle["information_cutoff_at"]),
        "qualifying_start_at": str(bundle["qualifying_start_at"]),
        "frozen_at": str(bundle["frozen_at"]),
        "registration": str(bundle["registration"]),
    }
    validate_frozen_checkpoint_forecast_reference(
        reference,
        manifest=manifest,
        event_id=event_id,
        event_start_at=event_start_at,
        session_kind=session_kind,
        checkpoint=checkpoint,
        information_cutoff_at=information_cutoff_at,
        qualifying_start_at=qualifying_start_at,
    )
    return reference


def validate_frozen_checkpoint_forecast_reference(
    reference: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    event_id: str,
    event_start_at: str | datetime,
    session_kind: str,
    checkpoint: str,
    information_cutoff_at: str | datetime,
    qualifying_start_at: str | datetime,
) -> dict[str, Any]:
    """Resolve and fully verify one manifest-bound immutable checkpoint sidecar."""

    validated_manifest = validate_challenger_manifest(manifest)
    expected_reference_keys = {
        "path",
        "artifact_sha256",
        "manifest_sha256",
        "bundle_sha256",
        "event_id",
        "event_start_at",
        "session_kind",
        "checkpoint",
        "information_cutoff_at",
        "qualifying_start_at",
        "frozen_at",
        "registration",
    }
    if set(reference) != expected_reference_keys:
        raise ValueError("checkpoint forecast reference fields do not match schema v1")
    path = Path(str(reference.get("path", "")).strip()).resolve()
    if not path.is_file():
        raise ValueError("checkpoint forecast sidecar does not exist")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint forecast sidecar is not readable JSON") from exc
    if not isinstance(envelope, Mapping):
        raise ValueError("checkpoint forecast sidecar envelope must be an object")
    if set(envelope) != {
        "artifact_type",
        "artifact_kind",
        "candidate_id",
        "manifest_sha256",
        "payload",
        "artifact_sha256",
    }:
        raise ValueError("checkpoint forecast sidecar envelope fields do not match schema")
    artifact_digest = _sha256_text(
        envelope.get("artifact_sha256"),
        field_name="checkpoint forecast artifact_sha256",
    )
    if artifact_digest != stable_json_sha256(
        {key: value for key, value in envelope.items() if key != "artifact_sha256"}
    ):
        raise ValueError("checkpoint forecast sidecar artifact digest does not match")
    if reference.get("artifact_sha256") != artifact_digest:
        raise ValueError("checkpoint forecast reference artifact digest does not match")
    if envelope.get("artifact_type") != "challenger_research_sidecar":
        raise ValueError("checkpoint forecast reference is not a research sidecar")
    if (
        envelope.get("candidate_id") != validated_manifest.candidate_id
        or envelope.get("manifest_sha256") != validated_manifest.manifest_sha256
    ):
        raise ValueError("checkpoint forecast sidecar is bound to another manifest")

    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint forecast sidecar has no bundle payload")
    bundle = dict(payload)
    if (
        bundle.get("artifact_type") != _FORECAST_BUNDLE_ARTIFACT_TYPE
        or bundle.get("schema_version") != _FORECAST_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint forecast bundle type or schema is invalid")
    if set(bundle) != {
        "artifact_type",
        "schema_version",
        "manifest_sha256",
        "candidate_id",
        "variant_id",
        "event_id",
        "event_start_at",
        "session_kind",
        "checkpoint",
        "information_cutoff_at",
        "qualifying_start_at",
        "frozen_at",
        "registration",
        "fold_artifacts",
        "qualifying_predictions",
        "race_view_predictions",
        "bundle_sha256",
    }:
        raise ValueError("checkpoint forecast bundle fields do not match schema v1")
    bundle_digest = _sha256_text(
        bundle.get("bundle_sha256"),
        field_name="checkpoint forecast bundle_sha256",
    )
    if bundle_digest != stable_json_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    ):
        raise ValueError("checkpoint forecast bundle digest does not match")
    if reference.get("bundle_sha256") != bundle_digest:
        raise ValueError("checkpoint forecast reference bundle digest does not match")
    if envelope.get("artifact_kind") != f"{_FORECAST_ARTIFACT_KIND_PREFIX}{bundle_digest[:16]}":
        raise ValueError("checkpoint forecast sidecar artifact kind is invalid")
    if (
        bundle.get("manifest_sha256") != validated_manifest.manifest_sha256
        or bundle.get("candidate_id") != validated_manifest.candidate_id
    ):
        raise ValueError("checkpoint forecast bundle is bound to another manifest")
    if str(bundle.get("variant_id", "")).strip().lower() != validated_manifest.variant_id:
        raise ValueError("checkpoint forecast bundle variant does not match the manifest")

    resolved_event_id = str(event_id).strip()
    resolved_event_start = _aware_utc(event_start_at, field_name="event.event_start_at")
    resolved_session_kind = str(session_kind).strip().lower()
    if resolved_session_kind not in SESSION_KINDS:
        raise ValueError("event session_kind must be 'main' or 'sprint'")
    resolved_checkpoint = str(checkpoint).strip().upper()
    expected_cutoff = _aware_utc(
        information_cutoff_at,
        field_name="checkpoint_payload.information_cutoff_at",
    )
    expected_qualifying_start = _aware_utc(
        qualifying_start_at,
        field_name="event.qualifying_start_at",
    )
    bundle_cutoff = _aware_utc(
        bundle.get("information_cutoff_at"),
        field_name="checkpoint_forecast.information_cutoff_at",
    )
    bundle_qualifying_start = _aware_utc(
        bundle.get("qualifying_start_at"),
        field_name="checkpoint_forecast.qualifying_start_at",
    )
    frozen = _aware_utc(bundle.get("frozen_at"), field_name="checkpoint_forecast.frozen_at")
    bundle_event_start = _aware_utc(
        bundle.get("event_start_at"),
        field_name="checkpoint_forecast.event_start_at",
    )
    if (
        bundle.get("event_id") != resolved_event_id
        or bundle_event_start != resolved_event_start
        or bundle.get("session_kind") != resolved_session_kind
        or bundle.get("checkpoint") != resolved_checkpoint
    ):
        raise ValueError("checkpoint forecast bundle event/checkpoint identity does not match")
    if bundle_cutoff != expected_cutoff or bundle_qualifying_start != expected_qualifying_start:
        raise ValueError("checkpoint forecast bundle information boundary does not match")
    if not bundle_event_start <= bundle_cutoff < bundle_qualifying_start:
        raise ValueError("checkpoint forecast information cutoff is not inside the pre-Q window")
    if frozen < validated_manifest.created_at:
        raise ValueError("checkpoint forecast predates its challenger manifest")
    expected_registration = (
        "preregistered_shadow"
        if validated_manifest.created_at < bundle_qualifying_start
        and frozen < bundle_qualifying_start
        else "retrospective_diagnostic"
    )
    if bundle.get("registration") != expected_registration:
        raise ValueError("checkpoint forecast registration classification is invalid")
    expected_reference = {
        "path": str(path),
        "artifact_sha256": artifact_digest,
        "manifest_sha256": validated_manifest.manifest_sha256,
        "bundle_sha256": bundle_digest,
        "event_id": resolved_event_id,
        "event_start_at": bundle_event_start.isoformat().replace("+00:00", "Z"),
        "session_kind": resolved_session_kind,
        "checkpoint": resolved_checkpoint,
        "information_cutoff_at": bundle_cutoff.isoformat().replace("+00:00", "Z"),
        "qualifying_start_at": bundle_qualifying_start.isoformat().replace("+00:00", "Z"),
        "frozen_at": frozen.isoformat().replace("+00:00", "Z"),
        "registration": expected_registration,
    }
    if dict(reference) != expected_reference:
        raise ValueError("checkpoint forecast reference metadata does not match its bundle")
    raw_qualifying = bundle.get("qualifying_predictions")
    if not isinstance(raw_qualifying, Mapping):
        raise ValueError("checkpoint forecast qualifying_predictions are missing")
    _normalise_seed_forecasts(
        raw_qualifying,
        section="qualifying_predictions",
        variant_id=validated_manifest.variant_id,
    )
    raw_race = bundle.get("race_view_predictions")
    if raw_race is not None:
        _normalise_race_seed_forecasts(
            cast(Mapping[str, Any], raw_race),
            variant_id=validated_manifest.variant_id,
        )
    return bundle


class FrozenPredictionBundleBackend:
    """Replay immutable fold outputs embedded in each checkpoint payload.

    This adapter is deliberately boring: it performs no fitting and no prediction.
    It lets the CLI evaluate forecasts that were frozen before their event, while the
    runner validates the supplied fold chronology and exact seed pairing.  A live
    research adapter can implement :class:`WalkForwardBackend` directly.
    """

    def __init__(self) -> None:
        self._bundle_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def _checkpoint_bundle(self, **kwargs: Any) -> dict[str, Any]:
        event = cast(ReplayEvent, kwargs["event"])
        checkpoint = str(kwargs["checkpoint"]).strip().upper()
        checkpoint_payload = cast(Mapping[str, Any], kwargs["checkpoint_payload"])
        manifest = cast(Mapping[str, Any], kwargs["manifest"])
        reference = checkpoint_payload.get("forecast_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("frozen checkpoint requires a forecast_reference")
        validated_manifest = validate_challenger_manifest(manifest)
        cache_key = (
            str(reference.get("artifact_sha256", "")),
            validated_manifest.manifest_sha256,
            event.event_id,
            checkpoint,
        )
        cached = self._bundle_cache.get(cache_key)
        if cached is not None:
            return cached
        bundle = validate_frozen_checkpoint_forecast_reference(
            reference,
            manifest=manifest,
            event_id=event.event_id,
            event_start_at=event.event_start_at,
            session_kind=event.session_kind,
            checkpoint=checkpoint,
            information_cutoff_at=checkpoint_payload["information_cutoff_at"],
            qualifying_start_at=event.qualifying_start_at,
        )
        self._bundle_cache[cache_key] = bundle
        return bundle

    @staticmethod
    def _seed_payload(
        bundle: Mapping[str, Any],
        *,
        section: str,
        role: str,
        seed: int,
    ) -> Mapping[str, Any]:
        raw_section = bundle.get(section)
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"frozen checkpoint has no {section}")
        raw_role = raw_section.get(role)
        if not isinstance(raw_role, Mapping):
            raise ValueError(f"frozen {section} has no {role} payload")
        payload = raw_role.get(str(seed), raw_role.get(seed))
        if not isinstance(payload, Mapping):
            raise ValueError(f"frozen {section}.{role} has no seed {seed}")
        return payload

    def fit_fold(self, **kwargs: Any) -> Mapping[str, Any]:
        target_event = cast(ReplayEvent, kwargs["target_event"])
        checkpoint = str(kwargs["checkpoint"])
        checkpoint_payload = target_event.checkpoint_payloads[checkpoint]
        payload = self._checkpoint_bundle(
            event=target_event,
            checkpoint=checkpoint,
            checkpoint_payload=checkpoint_payload,
            manifest=kwargs["manifest"],
        ).get("fold_artifacts")
        if not isinstance(payload, Mapping):
            raise ValueError("frozen checkpoint has no fold_artifacts")
        return payload

    def predict_qualifying(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._seed_payload(
            self._checkpoint_bundle(**kwargs),
            section="qualifying_predictions",
            role=str(kwargs["role"]),
            seed=int(kwargs["seed"]),
        )

    def predict_race_views(self, **kwargs: Any) -> Mapping[str, Mapping[str, Any]]:
        payload = self._seed_payload(
            self._checkpoint_bundle(**kwargs),
            section="race_view_predictions",
            role=str(kwargs["role"]),
            seed=int(kwargs["seed"]),
        )
        if not all(isinstance(payload.get(view), Mapping) for view in _RACE_VIEWS):
            raise ValueError("frozen race payload does not contain both required views")
        return {view: cast(Mapping[str, Any], payload[view]) for view in _RACE_VIEWS}

    def checkpoint_registration(self, **kwargs: Any) -> dict[str, Any]:
        bundle = self._checkpoint_bundle(**kwargs)
        checkpoint_payload = cast(Mapping[str, Any], kwargs["checkpoint_payload"])
        reference = cast(Mapping[str, Any], checkpoint_payload["forecast_reference"])
        return {
            "classification": str(bundle["registration"]),
            "frozen_at": str(bundle["frozen_at"]),
            "information_cutoff_at": str(bundle["information_cutoff_at"]),
            "forecast_reference": dict(reference),
        }


def _aware_utc(value: Any, *, field_name: str) -> datetime:
    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            candidate = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if not isinstance(candidate, datetime) or candidate.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return candidate.astimezone(UTC)


def _normalise_event(raw: Mapping[str, Any]) -> ReplayEvent:
    event_id = str(raw.get("event_id", "")).strip()
    if not event_id:
        raise ValueError("walk-forward event_id must not be blank")
    event_start = _aware_utc(raw.get("event_start_at"), field_name=f"{event_id}.event_start_at")
    qualifying_start = _aware_utc(
        raw.get("qualifying_start_at"),
        field_name=f"{event_id}.qualifying_start_at",
    )
    if event_start >= qualifying_start:
        raise ValueError(f"{event_id} requires event_start_at < qualifying_start_at")

    session_kind = str(raw.get("session_kind", "")).strip().lower()
    if session_kind not in SESSION_KINDS:
        raise ValueError(f"{event_id}.session_kind must be 'main' or 'sprint'")

    raw_checkpoints = raw.get("checkpoint_payloads")
    if not isinstance(raw_checkpoints, Mapping) or not raw_checkpoints:
        raise ValueError(f"{event_id}.checkpoint_payloads must be a non-empty mapping")
    checkpoints: dict[str, Mapping[str, Any]] = {}
    previous_information_cutoff: datetime | None = None
    for raw_checkpoint, payload in raw_checkpoints.items():
        checkpoint = str(raw_checkpoint).strip().upper()
        if checkpoint not in DEFAULT_REPLAY_CHECKPOINTS:
            raise ValueError(f"{event_id} has unsupported checkpoint {checkpoint!r}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{event_id}.{checkpoint} payload must be a mapping")
        embedded_sections = {
            "fold_artifacts",
            "qualifying_predictions",
            "race_view_predictions",
        }.intersection(payload)
        if embedded_sections:
            raise ValueError(
                f"{event_id}.{checkpoint} embeds frozen payloads; use forecast_reference"
            )
        information_cutoff = _aware_utc(
            payload.get("information_cutoff_at", payload.get("available_at")),
            field_name=f"{event_id}.{checkpoint}.information_cutoff_at",
        )
        if information_cutoff < event_start or information_cutoff >= qualifying_start:
            raise ValueError(
                f"{event_id}.{checkpoint} information cutoff must be inside the pre-Q window"
            )
        payload_copy = dict(payload)
        payload_copy["information_cutoff_at"] = information_cutoff.isoformat()
        checkpoints[checkpoint] = payload_copy
    checkpoints = {
        checkpoint: checkpoints[checkpoint]
        for checkpoint in DEFAULT_REPLAY_CHECKPOINTS
        if checkpoint in checkpoints
    }
    for checkpoint, payload in checkpoints.items():
        information_cutoff = _aware_utc(
            payload["information_cutoff_at"],
            field_name=f"{event_id}.{checkpoint}.information_cutoff_at",
        )
        if (
            previous_information_cutoff is not None
            and information_cutoff < previous_information_cutoff
        ):
            raise ValueError(f"{event_id} checkpoint information cutoffs are not chronological")
        previous_information_cutoff = information_cutoff

    raw_actual = raw.get("actual_qualifying_grid")
    if not isinstance(raw_actual, Sequence) or isinstance(raw_actual, str | bytes):
        raise ValueError(f"{event_id}.actual_qualifying_grid must be a sequence")
    actual_grid = tuple(
        cast(Mapping[str, Any], row) for row in raw_actual if isinstance(row, Mapping)
    )
    if len(actual_grid) != len(raw_actual) or not actual_grid:
        raise ValueError(f"{event_id}.actual_qualifying_grid contains invalid rows")
    validate_qualifying_grid(actual_grid, require_sequential_positions=True)

    raw_race = raw.get("actual_race_finish_order")
    actual_race: tuple[Mapping[str, Any], ...] | None = None
    if raw_race is not None:
        if not isinstance(raw_race, Sequence) or isinstance(raw_race, str | bytes):
            raise ValueError(f"{event_id}.actual_race_finish_order must be a sequence")
        actual_race = tuple(
            cast(Mapping[str, Any], row) for row in raw_race if isinstance(row, Mapping)
        )
        if len(actual_race) != len(raw_race) or not actual_race:
            raise ValueError(f"{event_id}.actual_race_finish_order contains invalid rows")
        validate_qualifying_grid(actual_race, require_sequential_positions=True)
        actual_q_drivers = {str(row["driver"]) for row in actual_grid}
        actual_race_drivers = {str(row["driver"]) for row in actual_race}
        if actual_q_drivers != actual_race_drivers:
            raise ValueError(
                f"{event_id} qualifying and race actuals must contain the same drivers"
            )

    raw_snapshot_ids = raw.get("input_snapshot_ids")
    if not isinstance(raw_snapshot_ids, Sequence) or isinstance(raw_snapshot_ids, str | bytes):
        raise ValueError(f"{event_id}.input_snapshot_ids must be a sequence")
    snapshot_ids = tuple(str(value).strip() for value in raw_snapshot_ids)
    if not snapshot_ids or any(not value for value in snapshot_ids):
        raise ValueError(f"{event_id}.input_snapshot_ids cannot be empty")
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError(f"{event_id}.input_snapshot_ids must be unique")

    is_dry = raw.get("is_dry")
    if not isinstance(is_dry, bool):
        raise ValueError(f"{event_id}.is_dry must be a boolean")
    return ReplayEvent(
        event_id=event_id,
        event_start_at=event_start,
        qualifying_start_at=qualifying_start,
        session_kind=cast(Literal["main", "sprint"], session_kind),
        is_dry=is_dry,
        checkpoint_payloads=checkpoints,
        actual_qualifying_grid=actual_grid,
        actual_race_finish_order=actual_race,
        input_snapshot_ids=snapshot_ids,
        payload=dict(raw),
    )


def _normalise_catalog(events: Sequence[Mapping[str, Any]]) -> list[ReplayEvent]:
    normalised = [_normalise_event(event) for event in events]
    if not normalised:
        raise ValueError("walk-forward replay requires at least one event")
    event_ids = [event.event_id for event in normalised]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("walk-forward event IDs must be unique")
    return sorted(normalised, key=lambda event: (event.event_start_at, event.event_id))


def _split_prior_events(
    prior_events: Sequence[ReplayEvent],
    *,
    minimum_training_events: int,
) -> tuple[list[ReplayEvent], list[ReplayEvent]] | None:
    """Use only earlier same-target events and reserve a chronological holdout."""

    if len(prior_events) < minimum_training_events + 1:
        return None
    maximum_holdout = len(prior_events) - minimum_training_events
    holdout_count = min(maximum_holdout, max(1, len(prior_events) // 5))
    return list(prior_events[:-holdout_count]), list(prior_events[-holdout_count:])


def _validate_fold_artifacts(
    artifacts: Mapping[str, Any],
    *,
    variant_id: str,
    checkpoint: str,
    target_event: ReplayEvent,
    checkpoint_payload: Mapping[str, Any],
    training_events: Sequence[ReplayEvent],
    calibration_events: Sequence[ReplayEvent],
) -> dict[str, Any]:
    payload = dict(artifacts)
    if str(payload.get("variant_id", "")).strip().lower() != variant_id:
        raise ValueError("fold artifacts do not match the manifest variant")
    if str(payload.get("checkpoint", "")).strip().upper() != checkpoint:
        raise ValueError("fold artifacts do not match the replay checkpoint")
    if str(payload.get("session_kind", "")).strip().lower() != target_event.session_kind:
        raise ValueError("fold artifacts do not match the target session kind")
    if str(payload.get("target_event_id", "")).strip() != target_event.event_id:
        raise ValueError("fold artifacts do not match the target event")

    expected_training = [event.event_id for event in training_events]
    expected_calibration = [event.event_id for event in calibration_events]
    if list(payload.get("training_event_ids", [])) != expected_training:
        raise ValueError("fold artifacts changed the runner-selected training events")
    if list(payload.get("calibration_event_ids", [])) != expected_calibration:
        raise ValueError("fold artifacts changed the runner-selected calibration events")
    if set(expected_training).intersection(expected_calibration):
        raise ValueError("training and calibration folds must be disjoint")

    cutoff = _aware_utc(payload.get("cutoff_at"), field_name="fold_artifacts.cutoff_at")
    max_input = _aware_utc(
        payload.get("max_input_timestamp"),
        field_name="fold_artifacts.max_input_timestamp",
    )
    checkpoint_cutoff = _aware_utc(
        checkpoint_payload.get("information_cutoff_at"),
        field_name="checkpoint_payload.information_cutoff_at",
    )
    if max_input >= cutoff or cutoff > checkpoint_cutoff:
        raise ValueError("fold artifacts cross the target event's information boundary")
    if any(event.event_start_at >= target_event.event_start_at for event in training_events):
        raise ValueError("training fold contains the target or a future event")
    if any(event.event_start_at >= target_event.event_start_at for event in calibration_events):
        raise ValueError("calibration fold contains the target or a future event")
    return payload


def _grid_positions(rows: Sequence[Mapping[str, Any]], *, field_name: str) -> dict[str, int]:
    try:
        validated = validate_qualifying_grid(rows, require_sequential_positions=True)
    except ValueError as exc:
        raise ValueError(f"{field_name} is invalid: {exc}") from exc
    return {str(row["driver"]): int(row["position"]) for row in validated}


def _expected_teammate_pairs(
    actual_grid: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    drivers_by_team: dict[str, list[str]] = {}
    for row in actual_grid:
        team = str(row.get("team", "")).strip()
        driver = str(row.get("driver", "")).strip()
        if not team or not driver:
            raise ValueError("actual qualifying grid requires driver and team identities")
        drivers_by_team.setdefault(team, []).append(driver)
    expected: dict[tuple[str, str], str] = {}
    for team, drivers in drivers_by_team.items():
        ordered = sorted(drivers)
        for index, driver_a in enumerate(ordered):
            for driver_b in ordered[index + 1 :]:
                expected[(driver_a, driver_b)] = team
    return expected


def _validate_complete_qualifying_uncertainty(
    prediction: Mapping[str, Any],
    *,
    role: str,
    actual_grid: Sequence[Mapping[str, Any]],
) -> None:
    grid = cast(Sequence[Mapping[str, Any]], prediction["grid"])
    actual_team = {str(row["driver"]): str(row["team"]) for row in actual_grid}
    field_size = len(grid)
    for row in grid:
        driver = str(row["driver"])
        if str(row.get("team", "")) != actual_team[driver]:
            raise ValueError(f"{role} qualifying team identity does not match actuals")
        try:
            lower = float(row["p5"])
            upper = float(row["p95"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{role} qualifying grid requires p5/p95 for every driver") from exc
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and 1.0 <= lower <= upper <= float(field_size)
        ):
            raise ValueError(f"{role} qualifying p5/p95 intervals are invalid")

    expected_pairs = _expected_teammate_pairs(actual_grid)
    rows = _normalised_h2h_rows(prediction)
    observed_pairs: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (str(row["driver_a"]), str(row["driver_b"]))
        team = str(row.get("team", ""))
        if key in observed_pairs:
            raise ValueError(f"{role} teammate H2H contains duplicate pair {key}")
        observed_pairs[key] = team
    if observed_pairs != expected_pairs:
        raise ValueError(f"{role} teammate H2H must contain exactly every actual teammate pair")


def _validate_qualifying_prediction(
    prediction: Mapping[str, Any],
    *,
    role: str,
    variant_id: str,
    actual_grid: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_variant = CHAMPION_VARIANT if role == "champion" else variant_id
    if str(prediction.get("model_variant", "")).strip().lower() != expected_variant:
        raise ValueError(f"{role} prediction returned the wrong model_variant")
    raw_grid = prediction.get("grid")
    if not isinstance(raw_grid, Sequence) or isinstance(raw_grid, str | bytes):
        raise ValueError(f"{role} qualifying prediction has no grid")
    grid = [cast(Mapping[str, Any], row) for row in raw_grid if isinstance(row, Mapping)]
    if len(grid) != len(raw_grid):
        raise ValueError(f"{role} qualifying prediction contains invalid grid rows")
    predicted_positions = _grid_positions(grid, field_name=f"{role}.grid")
    actual_positions = _grid_positions(actual_grid, field_name="actual_qualifying_grid")
    if set(predicted_positions) != set(actual_positions):
        raise ValueError(f"{role} qualifying driver set does not match actuals")
    _validate_complete_qualifying_uncertainty(
        prediction,
        role=role,
        actual_grid=actual_grid,
    )
    return dict(prediction)


def _qualifying_grid_metrics(
    prediction: Mapping[str, Any],
    actual_grid: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    grid = cast(Sequence[Mapping[str, Any]], prediction["grid"])
    predicted = _grid_positions(grid, field_name="prediction.grid")
    actual = _grid_positions(actual_grid, field_name="actual.grid")
    errors = [abs(predicted[driver] - actual[driver]) for driver in sorted(actual)]

    intervals: list[tuple[float, float, float]] = []
    for row in grid:
        driver = str(row["driver"])
        lower = float(row["p5"])
        upper = float(row["p95"])
        intervals.append((lower, upper, float(actual[driver])))
    coverage = fmean(float(lower <= position <= upper) for lower, upper, position in intervals)
    width = fmean(upper - lower for lower, upper, _ in intervals)
    return {
        "grid_mae": float(fmean(errors)),
        "interval_coverage": coverage,
        "interval_width": width,
    }


def _normalised_h2h_rows(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = prediction.get("teammate_head_to_head")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, str | bytes):
        raise ValueError("qualifying prediction must include teammate_head_to_head")
    raw_mappings = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    if len(raw_mappings) != len(raw_rows):
        raise ValueError("teammate_head_to_head contains invalid rows")
    rows: list[dict[str, Any]] = []
    for row in raw_mappings:
        team = str(row.get("team", "")).strip()
        driver_a = str(row.get("driver_a", "")).strip()
        driver_b = str(row.get("driver_b", "")).strip()
        if not team or not driver_a or not driver_b or driver_a == driver_b:
            raise ValueError("teammate_head_to_head contains invalid identities")
        try:
            probability_a = float(row["p_driver_a_ahead"])
            probability_b = float(row["p_driver_b_ahead"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("teammate_head_to_head requires both probabilities") from exc
        if not (
            math.isfinite(probability_a)
            and math.isfinite(probability_b)
            and 0.0 <= probability_a <= 1.0
            and 0.0 <= probability_b <= 1.0
            and math.isclose(probability_a + probability_b, 1.0, abs_tol=1e-9)
        ):
            raise ValueError("teammate_head_to_head probabilities are invalid")
        if driver_b < driver_a:
            driver_a, driver_b = driver_b, driver_a
            probability_a, probability_b = probability_b, probability_a
        rows.append(
            {
                "team": team,
                "driver_a": driver_a,
                "driver_b": driver_b,
                "p_driver_a_ahead": probability_a,
                "p_driver_b_ahead": probability_b,
            }
        )
    return rows


def _h2h_observations(
    prediction: Mapping[str, Any],
    actual_grid: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    actual = _grid_positions(actual_grid, field_name="actual.grid")
    observations: list[tuple[float, float]] = []
    for row in _normalised_h2h_rows(prediction):
        driver_a = str(row.get("driver_a", "")).strip()
        driver_b = str(row.get("driver_b", "")).strip()
        if not driver_a or not driver_b or driver_a not in actual or driver_b not in actual:
            raise ValueError("teammate H2H identities do not match the actual grid")
        raw_probability = row.get("p_driver_a_ahead")
        if raw_probability is None:
            raise ValueError("teammate H2H row has no p_driver_a_ahead")
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("teammate H2H probability must be finite and in [0, 1]")
        outcome = float(actual[driver_a] < actual[driver_b])
        observations.append((probability, outcome))
    if not observations:
        raise ValueError("qualifying prediction has no teammate H2H observations")
    return observations


def _score_h2h(observations: Sequence[tuple[float, float]]) -> dict[str, float]:
    epsilon = 1e-12
    brier = fmean((probability - outcome) ** 2 for probability, outcome in observations)
    log_loss = -fmean(
        outcome * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1.0 - outcome) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for probability, outcome in observations
    )
    bins: dict[int, list[tuple[float, float]]] = {}
    for probability, outcome in observations:
        index = min(9, int(probability * 10.0))
        bins.setdefault(index, []).append((probability, outcome))
    total = len(observations)
    ece = sum(
        (len(values) / total)
        * abs(fmean(value[0] for value in values) - fmean(value[1] for value in values))
        for values in bins.values()
    )
    return {"brier": float(brier), "log_loss": float(log_loss), "ece": float(ece)}


def _race_metrics(
    prediction: Mapping[str, Any],
    actual_finish_order: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    raw_finish_order = prediction.get("finish_order")
    if not isinstance(raw_finish_order, Sequence) or isinstance(raw_finish_order, str | bytes):
        raise ValueError("race prediction has no finish_order")
    predicted_rows = [
        cast(Mapping[str, Any], row) for row in raw_finish_order if isinstance(row, Mapping)
    ]
    if len(predicted_rows) != len(raw_finish_order):
        raise ValueError("race prediction contains invalid finish_order rows")
    predicted = _grid_positions(predicted_rows, field_name="prediction.finish_order")
    actual = _grid_positions(actual_finish_order, field_name="actual_race_finish_order")
    if set(predicted) != set(actual):
        raise ValueError("race prediction driver set does not match actuals")

    predicted_metric_rows = [dict(row) for row in predicted_rows]
    actual_metric_rows = [dict(row) for row in actual_finish_order]
    accuracy = compute_prediction_accuracy(predicted_metric_rows, actual_metric_rows)
    dnf_calibration = compute_dnf_calibration(predicted_metric_rows, actual_metric_rows)
    predicted_winner = min(predicted, key=predicted.__getitem__)
    actual_winner = min(actual, key=actual.__getitem__)
    top_count = min(3, len(actual))
    predicted_top = set(sorted(predicted, key=predicted.__getitem__)[:top_count])
    actual_top = set(sorted(actual, key=actual.__getitem__)[:top_count])

    return {
        # Full phase-1 position-accuracy set (src.analysis.model_evaluation), kept
        # alongside the original finisher_mae/winner_accuracy_percent/
        # top3_accuracy_percent/dnf_brier names for backward compatibility with
        # existing gate consumers.
        "mae": float(accuracy["mae"]),
        "finisher_mae": float(accuracy["finisher_mae"]),
        "weighted_mae": float(accuracy["weighted_mae"]),
        "top_heavy_weighted_mae": float(accuracy["top_heavy_weighted_mae"]),
        "top_3_pct": float(accuracy["top_3_pct"]),
        "top_10_pct": float(accuracy["top_10_pct"]),
        "spearman_rank": float(accuracy["spearman_rank"]),
        "kendall_tau": float(accuracy["kendall_tau"]),
        "winner_accuracy_percent": float(predicted_winner == actual_winner) * 100.0,
        "top3_accuracy_percent": 100.0 * len(predicted_top.intersection(actual_top)) / top_count,
        "dnf_brier": float(dnf_calibration["brier_score"]),
    }


def _validate_and_score_race_views(
    views: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
    variant_id: str,
    components: frozenset[str],
    actual_finish_order: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if set(views) != set(_RACE_VIEWS):
        raise ValueError(f"{role} race replay must return both required grid views")
    expected_variant = CHAMPION_VARIANT if role == "champion" else variant_id
    # Champion is always the empty component set by definition, regardless of which
    # challenger manifest it is being scored alongside: its own predictor can never
    # produce joint grid scenarios (component_enabled(champion_cfg, "r1") is always
    # False), so an end-to-end champion prediction must stay on the marginal-grid
    # fallback even while comparing against an R1 challenger.
    expected_components = frozenset() if role == "champion" else components
    scored: dict[str, dict[str, float]] = {}
    for view_name in _RACE_VIEWS:
        prediction = views[view_name]
        if str(prediction.get("model_variant", "")).strip().lower() != expected_variant:
            raise ValueError(f"{role} {view_name} race prediction has the wrong model_variant")
        expected_detail = (
            "actual_starting_grid"
            if view_name == "conditional_actual_grid"
            else (
                "predicted_joint" if "r1" in expected_components else "predicted_marginal_fallback"
            )
        )
        if str(prediction.get("grid_source_detail", "")).strip() != expected_detail:
            raise ValueError(f"{role} {view_name} race prediction has the wrong grid source")
        scored[view_name] = _race_metrics(prediction, actual_finish_order)
    return scored


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    keys = set.intersection(*(set(row) for row in rows))
    return {key: float(fmean(float(row[key]) for row in rows)) for key in sorted(keys)}


def _consensus_grid(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    position_samples: dict[str, list[int]] = {}
    team_by_driver: dict[str, str] = {}
    for prediction in predictions:
        for row in cast(Sequence[Mapping[str, Any]], prediction["grid"]):
            driver = str(row["driver"])
            team_by_driver[driver] = str(row.get("team", ""))
            position_samples.setdefault(driver, []).append(int(row["position"]))
    ordered = sorted(position_samples, key=lambda driver: (fmean(position_samples[driver]), driver))
    return [
        {"driver": driver, "team": team_by_driver[driver], "position": index + 1}
        for index, driver in enumerate(ordered)
    ]


def _consensus_h2h(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for prediction in predictions:
        for row in _normalised_h2h_rows(prediction):
            team = str(row.get("team", ""))
            driver_a = str(row["driver_a"])
            driver_b = str(row["driver_b"])
            key = (team, driver_a, driver_b)
            values.setdefault(key, []).append(float(row["p_driver_a_ahead"]))
    return [
        {
            "team": team,
            "driver_a": driver_a,
            "driver_b": driver_b,
            "p_driver_a_ahead": float(fmean(probabilities)),
            "p_driver_b_ahead": float(1.0 - fmean(probabilities)),
        }
        for (team, driver_a, driver_b), probabilities in sorted(values.items())
    ]


# Research-only floors for the walk-forward minimum-training-event gates. These are
# the *lowest* value a research_gate_relaxation override may request -- they exist
# so a caller cannot relax a gate to near-zero evidence, not to loosen it further
# than this. Production defaults (MINIMUM_Q1_TRAINING_EVENTS / MINIMUM_R2_TRAINING_
# EVENTS above) are untouched; a relaxation is only ever applied when a caller
# explicitly passes research_gate_relaxation to run_challenger_walk_forward.
RESEARCH_GATE_RELAXATION_FLOORS = {"q1": 4, "r2_source_anchor": 3}


def _required_training_events(
    components: frozenset[str],
    session_kind: str,
    research_gate_relaxation: Mapping[str, int] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Return the effective minimum training-event count and, if relaxed, its detail.

    The detail dict (when not None) is the exact ``research_gate_relaxation`` block
    every checkpoint scored under the relaxation must carry in its own output.
    """
    minimum = 0
    detail: dict[str, Any] | None = None
    if "q1" in components:
        base = MINIMUM_Q1_TRAINING_EVENTS[session_kind]
        if research_gate_relaxation and "q1" in research_gate_relaxation:
            floor = RESEARCH_GATE_RELAXATION_FLOORS["q1"]
            relaxed = max(floor, int(research_gate_relaxation["q1"]))
            minimum = max(minimum, relaxed)
            detail = {
                "component": "q1",
                "original_threshold": base,
                "relaxed_threshold": relaxed,
            }
        else:
            minimum = max(minimum, base)
    if "r2_source_anchor" in components:
        base = MINIMUM_R2_TRAINING_EVENTS
        if research_gate_relaxation and "r2_source_anchor" in research_gate_relaxation:
            floor = RESEARCH_GATE_RELAXATION_FLOORS["r2_source_anchor"]
            relaxed = max(floor, int(research_gate_relaxation["r2_source_anchor"]))
            minimum = max(minimum, relaxed)
            detail = {
                "component": "r2_source_anchor",
                "original_threshold": base,
                "relaxed_threshold": relaxed,
            }
        else:
            minimum = max(minimum, base)
    return minimum, detail


def _normalise_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if values != DEFAULT_REPLAY_SEEDS:
        raise ValueError(f"walk-forward seeds must be exactly {list(DEFAULT_REPLAY_SEEDS)}")
    return values


def run_challenger_walk_forward(
    *,
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    backend: WalkForwardBackend,
    seeds: Sequence[int] = DEFAULT_REPLAY_SEEDS,
    movement_reviews: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    research_gate_relaxation: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Run one frozen challenger chronologically over all supplied dry events.

    The runner never writes artifacts and never changes predictor configuration.  A
    backend may persist fold artifacts in a challenger-only store, but must return
    their complete chronology metadata for validation before predictions are scored.

    ``research_gate_relaxation`` (e.g. ``{"q1": 4}``) lowers the Q1/R2-source-anchor
    minimum-training-event gate for this run only -- production defaults
    (``MINIMUM_Q1_TRAINING_EVENTS`` / ``MINIMUM_R2_TRAINING_EVENTS``) are untouched,
    and every checkpoint scored under it carries the exact relaxation detail in its
    own output. The manifest must independently disclose the same relaxation in
    ``manifest["metadata"]["research_gate_relaxation"]`` -- this ties the runtime
    override to a manifest that ``evaluate_release_readiness`` will always reject,
    so a relaxed research run can never be mistaken for (or slip into) a promotion.
    """

    validated_manifest = validate_challenger_manifest(manifest)
    if research_gate_relaxation:
        manifest_relaxation = cast(Mapping[str, Any], manifest.get("metadata", {})).get(
            "research_gate_relaxation"
        )
        if not isinstance(manifest_relaxation, Mapping):
            raise ValueError(
                "research_gate_relaxation requires a matching "
                "manifest['metadata']['research_gate_relaxation'] disclosure"
            )
    replay_seeds = _normalise_seeds(seeds)
    catalog = _normalise_catalog(events)
    manifest_snapshots = set(
        str(value)
        for value in cast(Mapping[str, Any], manifest["provenance"])["input_snapshot_ids"]
    )
    checkpoint_counts = {checkpoint: 0 for checkpoint in DEFAULT_REPLAY_CHECKPOINTS}
    prior_dry_by_kind: dict[str, list[ReplayEvent]] = {"main": [], "sprint": []}
    scored_events: list[dict[str, Any]] = []
    skipped_events: list[dict[str, Any]] = []
    checkpoint_refusals: list[dict[str, Any]] = []
    leakage_checks: list[dict[str, Any]] = []
    review_payloads = movement_reviews or {}
    registration_counts = {
        "preregistered_shadow": 0,
        "retrospective_diagnostic": 0,
        "research_backend_generated": 0,
    }

    for event in catalog:
        if not set(event.input_snapshot_ids).issubset(manifest_snapshots):
            raise ValueError(f"{event.event_id} uses a snapshot absent from the frozen manifest")
        if not event.is_dry:
            skipped_events.append({"event_id": event.event_id, "reason": "dry_only_candidate"})
            continue

        # ponytail: q1's real eligibility (``_q1_track_class_eligibility``) pools
        # prior dry events ACROSS session_kind by track class, not by session_kind --
        # the outer training/calibration split must use the same pool, or a
        # same-session-kind-only shortage (e.g. only 3 prior sprint events) refuses
        # every checkpoint before fit_fold's own (correct) track-class gate ever runs.
        candidate_priors = (
            [prior for priors in prior_dry_by_kind.values() for prior in priors]
            if "q1" in validated_manifest.components
            else prior_dry_by_kind[event.session_kind]
        )
        prior_events = sorted(
            (prior for prior in candidate_priors if prior.event_start_at < event.event_start_at),
            key=lambda item: (item.event_start_at, item.event_id),
        )
        minimum_training, relaxation_detail = _required_training_events(
            validated_manifest.components,
            event.session_kind,
            research_gate_relaxation,
        )
        split = (
            _split_prior_events(prior_events, minimum_training_events=minimum_training)
            if minimum_training > 0
            else (prior_events, [])
        )
        if split is None:
            skipped_events.append(
                {
                    "event_id": event.event_id,
                    "reason": "insufficient_prior_training_events",
                    "session_kind": event.session_kind,
                    "prior_event_count": len(prior_events),
                    "minimum_training_events": minimum_training,
                }
            )
            prior_dry_by_kind[event.session_kind].append(event)
            continue
        training_events, calibration_events = split

        event_checkpoints: dict[str, Any] = {}
        for checkpoint, checkpoint_payload in event.checkpoint_payloads.items():
            try:
                registration_getter = getattr(backend, "checkpoint_registration", None)
                if callable(registration_getter):
                    raw_registration = registration_getter(
                        event=event,
                        checkpoint=checkpoint,
                        checkpoint_payload=checkpoint_payload,
                        manifest=manifest,
                    )
                    if not isinstance(raw_registration, Mapping):
                        raise ValueError("checkpoint registration metadata must be a mapping")
                    forecast_registration = dict(raw_registration)
                    registration_classification = str(
                        forecast_registration.get("classification", "")
                    ).strip()
                    if registration_classification not in _FORECAST_REGISTRATIONS:
                        raise ValueError("frozen checkpoint registration classification is invalid")
                else:
                    registration_classification = "research_backend_generated"
                    forecast_registration = {"classification": registration_classification}
                registration_counts[registration_classification] += 1

                fold_artifacts: Mapping[str, Any] | None = None
                if minimum_training > 0:
                    raw_artifacts = backend.fit_fold(
                        training_events=training_events,
                        calibration_events=calibration_events,
                        target_event=event,
                        checkpoint=checkpoint,
                        manifest=manifest,
                    )
                    fold_artifacts = _validate_fold_artifacts(
                        raw_artifacts,
                        variant_id=validated_manifest.variant_id,
                        checkpoint=checkpoint,
                        target_event=event,
                        checkpoint_payload=checkpoint_payload,
                        training_events=training_events,
                        calibration_events=calibration_events,
                    )

                champion_predictions: list[dict[str, Any]] = []
                challenger_predictions: list[dict[str, Any]] = []
                champion_grid_metrics: list[dict[str, float]] = []
                challenger_grid_metrics: list[dict[str, float]] = []
                champion_h2h: list[tuple[float, float]] = []
                challenger_h2h: list[tuple[float, float]] = []
                champion_race_metrics: dict[str, list[dict[str, float]]] = {
                    view: [] for view in _RACE_VIEWS
                }
                challenger_race_metrics: dict[str, list[dict[str, float]]] = {
                    view: [] for view in _RACE_VIEWS
                }
                for seed in replay_seeds:
                    champion = _validate_qualifying_prediction(
                        backend.predict_qualifying(
                            role="champion",
                            seed=seed,
                            event=event,
                            checkpoint=checkpoint,
                            checkpoint_payload=checkpoint_payload,
                            fold_artifacts=None,
                            manifest=manifest,
                        ),
                        role="champion",
                        variant_id=validated_manifest.variant_id,
                        actual_grid=event.actual_qualifying_grid,
                    )
                    challenger = _validate_qualifying_prediction(
                        backend.predict_qualifying(
                            role="challenger",
                            seed=seed,
                            event=event,
                            checkpoint=checkpoint,
                            checkpoint_payload=checkpoint_payload,
                            fold_artifacts=fold_artifacts,
                            manifest=manifest,
                        ),
                        role="challenger",
                        variant_id=validated_manifest.variant_id,
                        actual_grid=event.actual_qualifying_grid,
                    )
                    champion_predictions.append(champion)
                    challenger_predictions.append(challenger)
                    champion_grid_metrics.append(
                        _qualifying_grid_metrics(champion, event.actual_qualifying_grid)
                    )
                    challenger_grid_metrics.append(
                        _qualifying_grid_metrics(challenger, event.actual_qualifying_grid)
                    )
                    champion_h2h.extend(_h2h_observations(champion, event.actual_qualifying_grid))
                    challenger_h2h.extend(
                        _h2h_observations(challenger, event.actual_qualifying_grid)
                    )
                    if event.actual_race_finish_order is not None:
                        champion_views = backend.predict_race_views(
                            role="champion",
                            seed=seed,
                            event=event,
                            checkpoint=checkpoint,
                            checkpoint_payload=checkpoint_payload,
                            qualifying_prediction=champion,
                            fold_artifacts=None,
                            manifest=manifest,
                        )
                        challenger_views = backend.predict_race_views(
                            role="challenger",
                            seed=seed,
                            event=event,
                            checkpoint=checkpoint,
                            checkpoint_payload=checkpoint_payload,
                            qualifying_prediction=challenger,
                            fold_artifacts=fold_artifacts,
                            manifest=manifest,
                        )
                        scored_champion_views = _validate_and_score_race_views(
                            champion_views,
                            role="champion",
                            variant_id=validated_manifest.variant_id,
                            components=validated_manifest.components,
                            actual_finish_order=event.actual_race_finish_order,
                        )
                        scored_challenger_views = _validate_and_score_race_views(
                            challenger_views,
                            role="challenger",
                            variant_id=validated_manifest.variant_id,
                            components=validated_manifest.components,
                            actual_finish_order=event.actual_race_finish_order,
                        )
                        for view_name in _RACE_VIEWS:
                            champion_race_metrics[view_name].append(
                                scored_champion_views[view_name]
                            )
                            challenger_race_metrics[view_name].append(
                                scored_challenger_views[view_name]
                            )

                movement_key = f"{event.event_id}:{checkpoint}"
                movement_audit = build_full_field_movement_audit(
                    champion_grid=_consensus_grid(champion_predictions),
                    challenger_grid=_consensus_grid(challenger_predictions),
                    champion_teammate_h2h=_consensus_h2h(champion_predictions),
                    challenger_teammate_h2h=_consensus_h2h(challenger_predictions),
                    reviews=review_payloads.get(movement_key),
                )
                event_checkpoints[checkpoint] = {
                    "champion": {
                        **_mean_metrics(champion_grid_metrics),
                        **_score_h2h(champion_h2h),
                    },
                    "challenger": {
                        **_mean_metrics(challenger_grid_metrics),
                        **_score_h2h(challenger_h2h),
                    },
                    "movement_audit": movement_audit,
                    "race_views": (
                        {
                            view_name: {
                                "champion": _mean_metrics(champion_race_metrics[view_name]),
                                "challenger": _mean_metrics(challenger_race_metrics[view_name]),
                            }
                            for view_name in _RACE_VIEWS
                        }
                        if event.actual_race_finish_order is not None
                        else None
                    ),
                    "fold_artifact_sha256": (
                        stable_json_sha256(fold_artifacts) if fold_artifacts is not None else None
                    ),
                    "forecast_registration": forecast_registration,
                    "prediction_sha256": {
                        "champion": stable_json_sha256(champion_predictions),
                        "challenger": stable_json_sha256(challenger_predictions),
                    },
                    "research_gate_relaxation": (
                        {
                            **relaxation_detail,
                            "training_events_used": len(training_events),
                            "shrinkage_applied": min(
                                1.0,
                                len(training_events)
                                / float(relaxation_detail["original_threshold"]),
                            ),
                        }
                        if relaxation_detail is not None
                        else None
                    ),
                }
                checkpoint_counts[checkpoint] += 1
            except CheckpointInputUnavailable as exc:
                checkpoint_refusals.append(
                    {
                        "event_id": event.event_id,
                        "checkpoint": checkpoint,
                        "reason": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
                continue

        # An event whose every checkpoint failed closed (see checkpoint_refusals)
        # contributes nothing scoreable; keep it out of scored_events so the scored
        # count reflects the exclusion, while it still counts as a real prior event
        # for later folds (its race happened; only this run's own extraction failed).
        if event_checkpoints:
            leakage_checks.append(
                {
                    "event_id": event.event_id,
                    "training_event_ids": [item.event_id for item in training_events],
                    "calibration_event_ids": [item.event_id for item in calibration_events],
                    "target_event_start_at": event.event_start_at.isoformat(),
                    "passed": all(
                        item.event_start_at < event.event_start_at
                        for item in (*training_events, *calibration_events)
                    )
                    and not set(item.event_id for item in training_events).intersection(
                        item.event_id for item in calibration_events
                    ),
                }
            )
            scored_events.append(
                {
                    "event_id": event.event_id,
                    "event_start_at": event.event_start_at.isoformat(),
                    "qualifying_start_at": event.qualifying_start_at.isoformat(),
                    "session_kind": event.session_kind,
                    "checkpoints": event_checkpoints,
                }
            )
        prior_dry_by_kind[event.session_kind].append(event)

    if not scored_events:
        raise ValueError("walk-forward replay produced no eligible scored events")
    replay_provenance = ReplayProvenance(
        seeds=replay_seeds,
        simulation_counts=dict(validated_manifest.simulation_counts),
        dry_only=True,
        checkpoint_event_counts=checkpoint_counts,
    )
    payload: dict[str, Any] = {
        "artifact_type": "challenger_walk_forward_replay",
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "variant_id": validated_manifest.variant_id,
        "manifest_sha256": validated_manifest.manifest_sha256,
        "runtime_activation_allowed": False,
        "seeds": list(replay_seeds),
        "simulation_counts": dict(validated_manifest.simulation_counts),
        "dry_only": True,
        "event_catalog_sha256": stable_json_sha256([event.payload for event in catalog]),
        "scored_events": scored_events,
        "skipped_events": skipped_events,
        "checkpoint_refusals": checkpoint_refusals,
        "checkpoint_event_counts": checkpoint_counts,
        "forecast_registration_counts": registration_counts,
        "replay_provenance": {
            "seeds": list(replay_provenance.seeds),
            "simulation_counts": dict(replay_provenance.simulation_counts),
            "dry_only": replay_provenance.dry_only,
            "checkpoint_event_counts": dict(replay_provenance.checkpoint_event_counts),
        },
        "leakage_audit": {
            "passed": bool(leakage_checks) and all(row["passed"] for row in leakage_checks),
            "events": leakage_checks,
        },
    }
    payload["replay_sha256"] = stable_json_sha256(payload)
    return payload


def _validate_walk_forward_replay(
    replay: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    require_frozen_references: bool = True,
) -> tuple[dict[str, Any], ValidatedChallengerManifest]:
    """Recompute every replay binding before any promotion metric is constructed."""

    validated_manifest = validate_challenger_manifest(manifest)
    payload = dict(replay)
    if (
        payload.get("artifact_type") != "challenger_walk_forward_replay"
        or payload.get("schema_version") != WALK_FORWARD_SCHEMA_VERSION
    ):
        raise ValueError("replay artifact_type or schema_version is invalid")
    replay_digest = _sha256_text(payload.get("replay_sha256"), field_name="replay_sha256")
    if replay_digest != stable_json_sha256(
        {key: value for key, value in payload.items() if key != "replay_sha256"}
    ):
        raise ValueError("replay_sha256 does not match the replay payload")
    if payload.get("manifest_sha256") != validated_manifest.manifest_sha256:
        raise ValueError("replay is linked to a different challenger manifest")
    if str(payload.get("variant_id", "")).strip().lower() != validated_manifest.variant_id:
        raise ValueError("replay variant does not match the challenger manifest")
    if payload.get("runtime_activation_allowed") is not False:
        raise ValueError("walk-forward replay cannot allow runtime activation")
    if payload.get("seeds") != list(DEFAULT_REPLAY_SEEDS):
        raise ValueError("replay seeds do not match the fixed seed contract")
    if payload.get("simulation_counts") != dict(validated_manifest.simulation_counts):
        raise ValueError("replay simulation counts do not match the frozen manifest")
    if payload.get("dry_only") is not True:
        raise ValueError("walk-forward replay must be dry-only")
    _sha256_text(payload.get("event_catalog_sha256"), field_name="event_catalog_sha256")

    raw_events = payload.get("scored_events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, str | bytes):
        raise ValueError("replay scored_events must be a sequence")
    scored_events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
    if not scored_events or len(scored_events) != len(raw_events):
        raise ValueError("replay scored_events contains invalid rows")
    event_ids = [str(event.get("event_id", "")).strip() for event in scored_events]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("replay scored event identities must be non-empty and unique")

    recomputed_checkpoint_counts = {checkpoint: 0 for checkpoint in DEFAULT_REPLAY_CHECKPOINTS}
    recomputed_registration_counts = {
        "preregistered_shadow": 0,
        "retrospective_diagnostic": 0,
        "research_backend_generated": 0,
    }
    for event in scored_events:
        event_id = str(event["event_id"])
        event_start = _aware_utc(
            event.get("event_start_at"),
            field_name=f"{event_id}.event_start_at",
        )
        qualifying_start = _aware_utc(
            event.get("qualifying_start_at"),
            field_name=f"{event_id}.qualifying_start_at",
        )
        if event_start >= qualifying_start:
            raise ValueError(f"{event_id} replay event window is invalid")
        session_kind = str(event.get("session_kind", "")).strip().lower()
        if session_kind not in SESSION_KINDS:
            raise ValueError(f"{event_id} replay session_kind is invalid")
        checkpoints = event.get("checkpoints")
        if not isinstance(checkpoints, Mapping) or not checkpoints:
            raise ValueError(f"{event_id} replay checkpoints are missing")
        if any(str(checkpoint) not in DEFAULT_REPLAY_CHECKPOINTS for checkpoint in checkpoints):
            raise ValueError(f"{event_id} replay contains an unsupported checkpoint")
        for checkpoint, raw_checkpoint in checkpoints.items():
            if not isinstance(raw_checkpoint, Mapping):
                raise ValueError(f"{event_id}.{checkpoint} replay checkpoint is invalid")
            recomputed_checkpoint_counts[str(checkpoint)] += 1
            registration = raw_checkpoint.get("forecast_registration")
            if not isinstance(registration, Mapping):
                raise ValueError(f"{event_id}.{checkpoint} forecast registration is missing")
            classification = str(registration.get("classification", "")).strip()
            if classification not in recomputed_registration_counts:
                raise ValueError(f"{event_id}.{checkpoint} forecast registration is invalid")
            recomputed_registration_counts[classification] += 1
            if classification in _FORECAST_REGISTRATIONS:
                reference = registration.get("forecast_reference")
                if not isinstance(reference, Mapping):
                    raise ValueError(
                        f"{event_id}.{checkpoint} immutable forecast reference is missing"
                    )
                registration_cutoff = registration.get("information_cutoff_at")
                if not isinstance(registration_cutoff, str):
                    raise ValueError(
                        f"{event_id}.{checkpoint} forecast information cutoff is missing"
                    )
                bundle = validate_frozen_checkpoint_forecast_reference(
                    reference,
                    manifest=manifest,
                    event_id=event_id,
                    event_start_at=event_start,
                    session_kind=session_kind,
                    checkpoint=str(checkpoint),
                    information_cutoff_at=registration_cutoff,
                    qualifying_start_at=qualifying_start,
                )
                expected_registration = {
                    "classification": str(bundle["registration"]),
                    "frozen_at": str(bundle["frozen_at"]),
                    "information_cutoff_at": str(bundle["information_cutoff_at"]),
                    "forecast_reference": dict(reference),
                }
                if dict(registration) != expected_registration:
                    raise ValueError(
                        f"{event_id}.{checkpoint} forecast registration metadata changed"
                    )
            elif require_frozen_references:
                raise ValueError(
                    "promotion gate conversion requires immutable checkpoint forecast references"
                )

    raw_checkpoint_counts = payload.get("checkpoint_event_counts")
    if (
        not isinstance(raw_checkpoint_counts, Mapping)
        or dict(raw_checkpoint_counts) != recomputed_checkpoint_counts
    ):
        raise ValueError("replay checkpoint counts do not match scored events")
    raw_registration_counts = payload.get("forecast_registration_counts")
    if (
        not isinstance(raw_registration_counts, Mapping)
        or dict(raw_registration_counts) != recomputed_registration_counts
    ):
        raise ValueError("replay forecast registration counts do not match scored events")

    expected_provenance = {
        "seeds": list(DEFAULT_REPLAY_SEEDS),
        "simulation_counts": dict(validated_manifest.simulation_counts),
        "dry_only": True,
        "checkpoint_event_counts": recomputed_checkpoint_counts,
    }
    if payload.get("replay_provenance") != expected_provenance:
        raise ValueError("replay provenance does not match its manifest and scored events")
    leakage = payload.get("leakage_audit")
    if not isinstance(leakage, Mapping) or leakage.get("passed") is not True:
        raise ValueError("replay leakage audit is missing or failed")
    leakage_events = leakage.get("events")
    if not isinstance(leakage_events, Sequence) or isinstance(leakage_events, str | bytes):
        raise ValueError("replay leakage audit events are invalid")
    leakage_rows = [row for row in leakage_events if isinstance(row, Mapping)]
    if (
        len(leakage_rows) != len(leakage_events)
        or [str(row.get("event_id", "")) for row in leakage_rows] != event_ids
        or any(row.get("passed") is not True for row in leakage_rows)
    ):
        raise ValueError("replay leakage audit does not match every scored event")
    return payload, validated_manifest


def _selected_replay_events(
    replay: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    session_kind: Literal["main", "sprint"],
) -> list[Mapping[str, Any]]:
    validated_replay, _validated_manifest = _validate_walk_forward_replay(
        replay,
        manifest=manifest,
    )
    raw_events = cast(Sequence[Any], validated_replay["scored_events"])
    selected = [
        cast(Mapping[str, Any], event)
        for event in raw_events
        if isinstance(event, Mapping)
        and str(event.get("session_kind", "")).strip().lower() == session_kind
    ]
    if not selected:
        raise ValueError(f"replay has no scored {session_kind} events")
    return selected


def _primary_checkpoint(event: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    raw_checkpoints = event.get("checkpoints")
    if not isinstance(raw_checkpoints, Mapping) or not raw_checkpoints:
        raise ValueError("scored replay event has no checkpoint metrics")
    for checkpoint in reversed(DEFAULT_REPLAY_CHECKPOINTS):
        payload = raw_checkpoints.get(checkpoint)
        if isinstance(payload, Mapping):
            return checkpoint, payload
    raise ValueError("scored replay event has no recognized checkpoint metrics")


def _role_metrics(checkpoint: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    payload = checkpoint.get(role)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint has no {role} metrics")
    return payload


def _finite_mean(values: Sequence[float], *, field_name: str) -> float:
    resolved = [float(value) for value in values]
    if not resolved or not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{field_name} requires finite replay metrics")
    return float(fmean(resolved))


def _replay_provenance_for_events(
    replay: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> ReplayProvenance:
    validated_replay, _validated_manifest = _validate_walk_forward_replay(
        replay,
        manifest=manifest,
    )
    checkpoint_counts = {checkpoint: 0 for checkpoint in DEFAULT_REPLAY_CHECKPOINTS}
    for event in events:
        raw_checkpoints = event.get("checkpoints")
        if not isinstance(raw_checkpoints, Mapping):
            raise ValueError("scored replay event has invalid checkpoints")
        for checkpoint in DEFAULT_REPLAY_CHECKPOINTS:
            if checkpoint in raw_checkpoints:
                checkpoint_counts[checkpoint] += 1
    return ReplayProvenance(
        seeds=tuple(int(seed) for seed in validated_replay["seeds"]),
        simulation_counts=dict(validated_replay["simulation_counts"]),
        dry_only=bool(validated_replay["dry_only"]),
        checkpoint_event_counts=checkpoint_counts,
        replay_sha256=str(validated_replay["replay_sha256"]),
    )


def build_qualifying_gate_metrics_from_walk_forward(
    replay: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    target: Literal["main_qualifying", "sprint_qualifying"],
) -> QualifyingGateMetrics:
    """Convert a complete replay into the exact qualifying promotion envelope."""

    session_kind: Literal["main", "sprint"] = "main" if target == "main_qualifying" else "sprint"
    events = _selected_replay_events(
        replay,
        manifest=manifest,
        session_kind=session_kind,
    )
    champion_mae: dict[str, float] = {}
    challenger_mae: dict[str, float] = {}
    champion_brier: list[float] = []
    challenger_brier: list[float] = []
    champion_log_loss: list[float] = []
    challenger_log_loss: list[float] = []
    champion_ece: list[float] = []
    challenger_ece: list[float] = []
    champion_width: list[float] = []
    challenger_width: list[float] = []
    challenger_coverage: list[float] = []
    conditional_champion: dict[str, float] = {}
    conditional_challenger: dict[str, float] = {}
    end_to_end_champion: dict[str, float] = {}
    end_to_end_challenger: dict[str, float] = {}
    movements_required = 0
    movements_reviewed = 0

    checkpoint_role_mae: dict[str, dict[str, list[float]]] = {
        checkpoint: {"champion": [], "challenger": []} for checkpoint in DEFAULT_REPLAY_CHECKPOINTS
    }
    for event in events:
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("scored replay event has no event_id")
        raw_checkpoints = cast(Mapping[str, Any], event["checkpoints"])
        for checkpoint in DEFAULT_REPLAY_CHECKPOINTS:
            raw_checkpoint = raw_checkpoints.get(checkpoint)
            if not isinstance(raw_checkpoint, Mapping):
                continue
            for role in ("champion", "challenger"):
                checkpoint_role_mae[checkpoint][role].append(
                    float(_role_metrics(raw_checkpoint, role)["grid_mae"])
                )
            checkpoint_movement = raw_checkpoint.get("movement_audit")
            if not isinstance(checkpoint_movement, Mapping):
                raise ValueError(f"{checkpoint} has no movement audit")
            movements_required += int(checkpoint_movement.get("review_required_count", 0))
            movements_reviewed += int(checkpoint_movement.get("reviewed_count", 0))

        _, primary = _primary_checkpoint(event)
        champion = _role_metrics(primary, "champion")
        challenger = _role_metrics(primary, "challenger")
        champion_mae[event_id] = float(champion["grid_mae"])
        challenger_mae[event_id] = float(challenger["grid_mae"])
        champion_brier.append(float(champion["brier"]))
        challenger_brier.append(float(challenger["brier"]))
        champion_log_loss.append(float(champion["log_loss"]))
        challenger_log_loss.append(float(challenger["log_loss"]))
        champion_ece.append(float(champion["ece"]))
        challenger_ece.append(float(challenger["ece"]))
        champion_width.append(float(champion["interval_width"]))
        challenger_width.append(float(challenger["interval_width"]))
        challenger_coverage.append(float(challenger["interval_coverage"]))

        race_views = primary.get("race_views")
        if not isinstance(race_views, Mapping):
            raise ValueError("qualifying gate replay requires both race evaluation views")
        for view_name, champion_target, challenger_target in (
            (
                "conditional_actual_grid",
                conditional_champion,
                conditional_challenger,
            ),
            (
                "end_to_end_predicted_grid",
                end_to_end_champion,
                end_to_end_challenger,
            ),
        ):
            view = race_views.get(view_name)
            if not isinstance(view, Mapping):
                raise ValueError(f"primary checkpoint has no {view_name} race metrics")
            champion_target[event_id] = float(
                cast(Mapping[str, Any], view["champion"])["finisher_mae"]
            )
            challenger_target[event_id] = float(
                cast(Mapping[str, Any], view["challenger"])["finisher_mae"]
            )

    checkpoint_regressions = {
        checkpoint: _finite_mean(values["challenger"], field_name=f"{checkpoint} MAE")
        - _finite_mean(values["champion"], field_name=f"{checkpoint} MAE")
        for checkpoint, values in checkpoint_role_mae.items()
        if values["champion"] and values["challenger"]
    }
    champion_brier_mean = _finite_mean(champion_brier, field_name="champion H2H Brier")
    challenger_brier_mean = _finite_mean(challenger_brier, field_name="challenger H2H Brier")
    if champion_brier_mean <= 0.0:
        raise ValueError("relative H2H Brier improvement requires a positive champion score")
    champion_width_mean = _finite_mean(champion_width, field_name="champion interval width")
    if champion_width_mean <= 0.0:
        raise ValueError("interval width comparison requires a positive champion width")

    return QualifyingGateMetrics(
        target=target,
        grid_mae=paired_weekend_bootstrap(champion_mae, challenger_mae),
        h2h_brier_relative_improvement=(champion_brier_mean - challenger_brier_mean)
        / champion_brier_mean,
        h2h_log_loss_delta=(
            _finite_mean(challenger_log_loss, field_name="challenger H2H log loss")
            - _finite_mean(champion_log_loss, field_name="champion H2H log loss")
        ),
        ece_delta=(
            _finite_mean(challenger_ece, field_name="challenger ECE")
            - _finite_mean(champion_ece, field_name="champion ECE")
        ),
        interval_coverage=_finite_mean(
            challenger_coverage,
            field_name="challenger interval coverage",
        ),
        interval_width_relative_change=(
            _finite_mean(challenger_width, field_name="challenger interval width")
            - champion_width_mean
        )
        / champion_width_mean,
        checkpoint_mae_regressions=checkpoint_regressions,
        race_views=build_race_metric_views(
            conditional_champion_by_event=conditional_champion,
            conditional_challenger_by_event=conditional_challenger,
            end_to_end_champion_by_event=end_to_end_champion,
            end_to_end_challenger_by_event=end_to_end_challenger,
        ),
        movements_requiring_review=movements_required,
        movements_reviewed=movements_reviewed,
        manifest=manifest,
        replay_provenance=_replay_provenance_for_events(
            replay,
            events,
            manifest=manifest,
        ),
    )


def build_race_gate_metrics_from_walk_forward(
    replay: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    target: Literal["grand_prix_race", "sprint_race"],
) -> RaceGateMetrics:
    """Convert a complete replay into the exact race promotion envelope."""

    session_kind: Literal["main", "sprint"] = "main" if target == "grand_prix_race" else "sprint"
    events = _selected_replay_events(
        replay,
        manifest=manifest,
        session_kind=session_kind,
    )
    conditional_champion: dict[str, float] = {}
    conditional_challenger: dict[str, float] = {}
    end_to_end_champion: dict[str, float] = {}
    end_to_end_challenger: dict[str, float] = {}
    winner_champion: list[float] = []
    winner_challenger: list[float] = []
    top3_champion: list[float] = []
    top3_challenger: list[float] = []
    dnf_champion: list[float] = []
    dnf_challenger: list[float] = []

    for event in events:
        event_id = str(event.get("event_id", "")).strip()
        _, primary = _primary_checkpoint(event)
        raw_views = primary.get("race_views")
        if not isinstance(raw_views, Mapping):
            raise ValueError("race gate replay requires both race evaluation views")
        for view_name, champion_target, challenger_target in (
            (
                "conditional_actual_grid",
                conditional_champion,
                conditional_challenger,
            ),
            (
                "end_to_end_predicted_grid",
                end_to_end_champion,
                end_to_end_challenger,
            ),
        ):
            raw_view = raw_views.get(view_name)
            if not isinstance(raw_view, Mapping):
                raise ValueError(f"race gate replay has no {view_name} metrics")
            champion = cast(Mapping[str, Any], raw_view["champion"])
            challenger = cast(Mapping[str, Any], raw_view["challenger"])
            champion_target[event_id] = float(champion["finisher_mae"])
            challenger_target[event_id] = float(challenger["finisher_mae"])
            if view_name == "end_to_end_predicted_grid":
                winner_champion.append(float(champion["winner_accuracy_percent"]))
                winner_challenger.append(float(challenger["winner_accuracy_percent"]))
                top3_champion.append(float(champion["top3_accuracy_percent"]))
                top3_challenger.append(float(challenger["top3_accuracy_percent"]))
                dnf_champion.append(float(champion["dnf_brier"]))
                dnf_challenger.append(float(challenger["dnf_brier"]))

    return RaceGateMetrics(
        target=target,
        race_views=build_race_metric_views(
            conditional_champion_by_event=conditional_champion,
            conditional_challenger_by_event=conditional_challenger,
            end_to_end_champion_by_event=end_to_end_champion,
            end_to_end_challenger_by_event=end_to_end_challenger,
        ),
        winner_accuracy_delta_pp=(
            _finite_mean(winner_challenger, field_name="challenger winner accuracy")
            - _finite_mean(winner_champion, field_name="champion winner accuracy")
        ),
        top3_accuracy_delta_pp=(
            _finite_mean(top3_challenger, field_name="challenger top-three accuracy")
            - _finite_mean(top3_champion, field_name="champion top-three accuracy")
        ),
        dnf_brier_delta=(
            _finite_mean(dnf_challenger, field_name="challenger DNF Brier")
            - _finite_mean(dnf_champion, field_name="champion DNF Brier")
        ),
        manifest=manifest,
        replay_provenance=_replay_provenance_for_events(
            replay,
            events,
            manifest=manifest,
        ),
    )
