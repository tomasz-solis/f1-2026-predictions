"""Research-only governance helpers for prediction model challengers.

The helpers in this module do not select a runtime model or write production
artifacts.  They make challenger provenance and promotion decisions explicit so
that a replay can be reproduced and audited before any separate release step.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from src.models.challenger_variants import (
    CHAMPION_VARIANT,
    VARIANT_COMPONENTS,
)

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_REPLAY_SEEDS = (17, 42, 91)
DEFAULT_REPLAY_CHECKPOINTS = ("PRE", "FP1", "FP2", "FP3")
DEFAULT_CONFIG_PATHS = (
    Path("config/default.yaml"),
    Path("config/production_config.json"),
)

QualifyingTarget = Literal["main_qualifying", "sprint_qualifying"]
RaceTarget = Literal["grand_prix_race", "sprint_race"]
RaceCandidateKind = Literal[
    "qualifying_only",
    "race_input_or_grid_propagation",
    "anchor_or_physics",
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")
_REQUIRED_CONFIGURATION_PATHS = frozenset(path.as_posix() for path in DEFAULT_CONFIG_PATHS)


def stable_json_sha256(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible payload."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    """Run one read-only git query and return its raw output."""
    completed = subprocess.run(
        ["git", "-c", "core.excludesFile=", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def collect_git_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Collect the source commit and a content-aware digest of all dirty changes.

    The dirty digest includes staged and unstaged tracked changes plus the paths
    and bytes of untracked files.  This is deliberately stronger than hashing
    ``git status`` alone, which cannot distinguish edits to an untracked file.
    """
    root = Path(repo_root).resolve()
    source_sha = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _run_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = _run_git(root, "diff", "--binary", "HEAD", "--")
    untracked_output = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = sorted(
        path
        for path in untracked_output.decode("utf-8", errors="surrogateescape").split("\0")
        if path
    )

    dirty_digest = hashlib.sha256()
    dirty_digest.update(b"git-status-v1\0")
    dirty_digest.update(status)
    dirty_digest.update(b"\0git-diff-head-binary\0")
    dirty_digest.update(tracked_diff)
    for relative_path in untracked_paths:
        path_bytes = relative_path.encode("utf-8", errors="surrogateescape")
        dirty_digest.update(b"\0untracked-path\0")
        dirty_digest.update(path_bytes)
        candidate = root / relative_path
        if candidate.is_symlink():
            dirty_digest.update(b"\0symlink\0")
            dirty_digest.update(str(candidate.readlink()).encode("utf-8"))
        elif candidate.is_file():
            dirty_digest.update(b"\0file\0")
            with candidate.open("rb") as file_handle:
                for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                    dirty_digest.update(block)
        else:
            dirty_digest.update(b"\0missing-or-non-file\0")

    return {
        "source_sha": source_sha,
        "is_dirty": bool(status),
        "dirty_diff_sha256": dirty_digest.hexdigest(),
        "dirty_status_sha256": hashlib.sha256(status).hexdigest(),
        "untracked_file_count": len(untracked_paths),
    }


def collect_configuration_provenance(
    repo_root: str | Path,
    config_paths: Sequence[str | Path] = DEFAULT_CONFIG_PATHS,
) -> dict[str, Any]:
    """Hash each effective configuration input and their ordered bundle."""
    root = Path(repo_root).resolve()
    files: list[dict[str, str]] = []
    for configured_path in config_paths:
        candidate = Path(configured_path)
        absolute_path = candidate if candidate.is_absolute() else root / candidate
        absolute_path = absolute_path.resolve()
        if not absolute_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {absolute_path}")
        try:
            display_path = absolute_path.relative_to(root).as_posix()
        except ValueError:
            display_path = absolute_path.as_posix()
        files.append({"path": display_path, "sha256": file_sha256(absolute_path)})

    return {
        "files": files,
        "effective_bundle_sha256": stable_json_sha256(files),
    }


def _normalise_timestamp(value: datetime | str, *, field_name: str) -> str:
    """Return a timezone-aware timestamp in canonical UTC form."""
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
    return candidate.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: Any, *, field_name: str) -> datetime:
    """Parse one required timezone-aware timestamp as UTC."""

    if not isinstance(value, datetime | str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    return datetime.fromisoformat(
        _normalise_timestamp(value, field_name=field_name).replace("Z", "+00:00")
    )


def _normalise_nonempty_strings(values: Sequence[str], *, field_name: str) -> list[str]:
    """Validate and de-duplicate a sequence while preserving its order."""
    normalised: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field_name} cannot contain blank values")
        if text not in normalised:
            normalised.append(text)
    return normalised


def _require_sha256(value: Any, *, field_name: str) -> str:
    """Validate and return one raw, lowercase SHA-256 digest."""

    digest = str(value).strip()
    if len(digest) != 64 or _HEX_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a raw lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class ValidatedChallengerManifest:
    """Security-relevant fields from one fully validated research manifest."""

    candidate_id: str
    variant_id: str
    components: frozenset[str]
    manifest_sha256: str
    created_at: datetime
    cutoff_at: datetime
    simulation_counts: Mapping[str, int]


def validate_challenger_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_variant_id: str | None = None,
) -> ValidatedChallengerManifest:
    """Fail closed unless ``manifest`` satisfies the complete challenger contract.

    This is the single trust boundary used by manifest creation, research
    persistence, orchestration, preregistration, and release.  Callers must not
    reimplement a weaker subset of these checks.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("challenger manifest must be a mapping")
    payload = dict(manifest)
    if payload.get("artifact_type") != "prediction_challenger_manifest":
        raise ValueError("manifest artifact_type is invalid")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    expected_top_level = {
        "artifact_type",
        "schema_version",
        "candidate_id",
        "variant_id",
        "created_at",
        "cutoff_at",
        "default_variant",
        "runtime_activation_allowed",
        "variants",
        "provenance",
        "metadata",
        "manifest_sha256",
    }
    if set(payload) != expected_top_level:
        raise ValueError("manifest top-level fields do not match schema v1")
    manifest_digest = _require_sha256(
        payload.get("manifest_sha256"),
        field_name="manifest_sha256",
    )
    unsigned_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    try:
        expected_digest = stable_json_sha256(unsigned_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must contain only finite JSON-compatible values") from exc
    if manifest_digest != expected_digest:
        raise ValueError("manifest_sha256 does not match the manifest payload")

    candidate_id = str(payload.get("candidate_id", "")).strip()
    if _IDENTIFIER_PATTERN.fullmatch(candidate_id) is None:
        raise ValueError("candidate_id must be a non-empty safe identifier")
    variant_id = str(payload.get("variant_id", "")).strip().lower()
    if variant_id == CHAMPION_VARIANT or variant_id not in VARIANT_COMPONENTS:
        raise ValueError("manifest variant_id must be a registered non-champion variant")
    if expected_variant_id is not None and variant_id != str(expected_variant_id).strip().lower():
        raise ValueError("manifest variant_id does not match the expected challenger variant")
    registered_components = VARIANT_COMPONENTS[variant_id]
    if not registered_components:
        raise ValueError("challenger variant must enable at least one registered component")

    if payload.get("default_variant") != CHAMPION_VARIANT:
        raise ValueError("research manifests must keep champion as the default variant")
    if payload.get("runtime_activation_allowed") is not False:
        raise ValueError("research manifests cannot allow runtime activation")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {CHAMPION_VARIANT, variant_id}:
        raise ValueError("manifest variants must contain exactly champion and the challenger")
    champion = variants.get(CHAMPION_VARIANT)
    challenger = variants.get(variant_id)
    if not isinstance(champion, Mapping) or dict(champion) != {
        "role": "champion",
        "default": True,
        "components": [],
    }:
        raise ValueError("manifest champion declaration is invalid")
    if not isinstance(challenger, Mapping):
        raise ValueError("manifest challenger declaration is missing")
    challenger_components = challenger.get("components")
    if not isinstance(challenger_components, Sequence) or isinstance(
        challenger_components, str | bytes
    ):
        raise ValueError("manifest challenger components must be a sequence")
    if dict(challenger) != {
        "role": "challenger",
        "default": False,
        "components": sorted(registered_components),
    }:
        raise ValueError("challenger components do not match the model variant registry")

    created_at = _utc_datetime(payload.get("created_at"), field_name="created_at")
    cutoff_at = _utc_datetime(payload.get("cutoff_at"), field_name="cutoff_at")
    if cutoff_at > created_at:
        raise ValueError("manifest requires cutoff_at <= created_at")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("manifest provenance must be a mapping")
    if set(provenance) != {
        "git",
        "configuration",
        "feature_schema",
        "feature_schema_sha256",
        "input_snapshot_ids",
        "seeds",
        "checkpoints",
        "dry_only",
        "simulation_counts",
    }:
        raise ValueError("manifest provenance fields do not match schema v1")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("manifest git provenance must be a mapping")
    if set(git) != {
        "source_sha",
        "is_dirty",
        "dirty_diff_sha256",
        "dirty_status_sha256",
        "untracked_file_count",
    }:
        raise ValueError("manifest git provenance fields do not match schema v1")
    source_sha = str(git.get("source_sha", "")).strip().lower()
    if len(source_sha) not in (40, 64) or _HEX_PATTERN.fullmatch(source_sha) is None:
        raise ValueError("provenance.git.source_sha must be a full hexadecimal commit hash")
    _require_sha256(
        git.get("dirty_diff_sha256"),
        field_name="provenance.git.dirty_diff_sha256",
    )
    _require_sha256(
        git.get("dirty_status_sha256"),
        field_name="provenance.git.dirty_status_sha256",
    )
    if not isinstance(git.get("is_dirty"), bool):
        raise ValueError("provenance.git.is_dirty must be boolean")
    untracked_count = git.get("untracked_file_count")
    if (
        isinstance(untracked_count, bool)
        or not isinstance(untracked_count, int)
        or untracked_count < 0
    ):
        raise ValueError("provenance.git.untracked_file_count must be a non-negative integer")

    configuration = provenance.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("manifest configuration provenance must be a mapping")
    if set(configuration) != {"files", "effective_bundle_sha256"}:
        raise ValueError("manifest configuration provenance fields do not match schema v1")
    config_files = configuration.get("files")
    if not isinstance(config_files, Sequence) or isinstance(config_files, str | bytes):
        raise ValueError("provenance.configuration.files must be a sequence")
    normalised_files: list[dict[str, str]] = []
    seen_config_paths: set[str] = set()
    for index, row in enumerate(config_files):
        if not isinstance(row, Mapping):
            raise ValueError("each configuration provenance row must be a mapping")
        if set(row) != {"path", "sha256"}:
            raise ValueError("configuration provenance rows require exactly path and sha256")
        path = str(row.get("path", "")).strip().replace("\\", "/")
        if not path or path in seen_config_paths:
            raise ValueError("configuration provenance paths must be non-empty and unique")
        seen_config_paths.add(path)
        normalised_files.append(
            {
                "path": path,
                "sha256": _require_sha256(
                    row.get("sha256"),
                    field_name=f"provenance.configuration.files[{index}].sha256",
                ),
            }
        )
    if not _REQUIRED_CONFIGURATION_PATHS.issubset(seen_config_paths):
        raise ValueError("manifest must record both default and production configuration hashes")
    bundle_digest = _require_sha256(
        configuration.get("effective_bundle_sha256"),
        field_name="provenance.configuration.effective_bundle_sha256",
    )
    if bundle_digest != stable_json_sha256(normalised_files):
        raise ValueError("effective configuration bundle digest does not match its files")

    feature_schema = provenance.get("feature_schema")
    if not (
        isinstance(feature_schema, str)
        and bool(feature_schema.strip())
        or isinstance(feature_schema, Mapping)
        and bool(feature_schema)
    ):
        raise ValueError("provenance.feature_schema must be non-empty")
    feature_schema_digest = _require_sha256(
        provenance.get("feature_schema_sha256"),
        field_name="provenance.feature_schema_sha256",
    )
    if feature_schema_digest != stable_json_sha256(feature_schema):
        raise ValueError("feature schema digest does not match the feature schema")

    snapshot_ids = provenance.get("input_snapshot_ids")
    if not isinstance(snapshot_ids, Sequence) or isinstance(snapshot_ids, str | bytes):
        raise ValueError("provenance.input_snapshot_ids must be a sequence")
    normalised_snapshot_ids = _normalise_nonempty_strings(
        snapshot_ids,
        field_name="input_snapshot_ids",
    )
    if not normalised_snapshot_ids or list(snapshot_ids) != normalised_snapshot_ids:
        raise ValueError("input_snapshot_ids must be non-empty, unique, normalized strings")

    seeds = provenance.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, str | bytes):
        raise ValueError("provenance.seeds must be a sequence")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("replay seeds must be integers")
    if tuple(seeds) != DEFAULT_REPLAY_SEEDS:
        raise ValueError(f"replay seeds must be exactly {list(DEFAULT_REPLAY_SEEDS)}")
    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, str | bytes):
        raise ValueError("provenance.checkpoints must be a sequence")
    if tuple(checkpoints) != DEFAULT_REPLAY_CHECKPOINTS:
        raise ValueError(f"replay checkpoints must be exactly {list(DEFAULT_REPLAY_CHECKPOINTS)}")
    if provenance.get("dry_only") is not True:
        raise ValueError("challenger v1 manifests must declare dry_only=true")

    simulation_counts = provenance.get("simulation_counts")
    if not isinstance(simulation_counts, Mapping) or not simulation_counts:
        raise ValueError("provenance.simulation_counts must be a non-empty mapping")
    normalised_simulation_counts: dict[str, int] = {}
    for raw_target, raw_count in sorted(simulation_counts.items()):
        target = str(raw_target).strip()
        if (
            not target
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise ValueError("simulation counts require normalized targets and positive integers")
        normalised_simulation_counts[target] = raw_count
    if dict(simulation_counts) != normalised_simulation_counts:
        raise ValueError("simulation count targets must be unique normalized strings")
    if not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("manifest metadata must be a mapping")

    return ValidatedChallengerManifest(
        candidate_id=candidate_id,
        variant_id=variant_id,
        components=frozenset(registered_components),
        manifest_sha256=manifest_digest,
        created_at=created_at,
        cutoff_at=cutoff_at,
        simulation_counts=normalised_simulation_counts,
    )


def build_challenger_manifest(
    *,
    repo_root: str | Path,
    candidate_id: str,
    variant_id: str,
    components: Sequence[str] | None = None,
    feature_schema: str | Mapping[str, Any],
    input_snapshot_ids: Sequence[str],
    cutoff_at: datetime | str,
    simulation_counts: Mapping[str, int],
    seeds: Sequence[int] = DEFAULT_REPLAY_SEEDS,
    config_paths: Sequence[str | Path] = DEFAULT_CONFIG_PATHS,
    created_at: datetime | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable-provenance manifest with champion as the only default.

    Creating this manifest never activates the challenger.  Runtime promotion is
    intentionally outside this research module and must be a separate reviewed
    configuration change.
    """
    normalised_candidate_id = str(candidate_id).strip()
    if not normalised_candidate_id or normalised_candidate_id == CHAMPION_VARIANT:
        raise ValueError(f"candidate_id must be non-empty and cannot be {CHAMPION_VARIANT!r}")
    normalised_variant_id = str(variant_id).strip().lower()
    if not normalised_variant_id or normalised_variant_id == CHAMPION_VARIANT:
        raise ValueError("variant_id must identify a non-champion model variant")
    if normalised_variant_id not in VARIANT_COMPONENTS:
        raise ValueError(f"unknown challenger variant_id: {normalised_variant_id}")
    registered_components = sorted(VARIANT_COMPONENTS[normalised_variant_id])
    if not registered_components:
        raise ValueError("challenger variant must enable at least one registered component")
    if components is None:
        normalised_components = registered_components
    else:
        normalised_components = sorted(
            _normalise_nonempty_strings(components, field_name="components")
        )
        if normalised_components != registered_components:
            raise ValueError(
                f"components for {normalised_variant_id!r} must match the model variant registry"
            )

    normalised_seeds = [int(seed) for seed in seeds]
    if normalised_seeds != list(DEFAULT_REPLAY_SEEDS):
        raise ValueError(f"replay seeds must be exactly {list(DEFAULT_REPLAY_SEEDS)}")

    normalised_simulation_counts: dict[str, int] = {}
    for target, raw_count in sorted(simulation_counts.items()):
        target_name = str(target).strip()
        if isinstance(raw_count, bool):
            raise ValueError("simulation counts must be positive integers")
        count = int(raw_count)
        if not target_name or count <= 0:
            raise ValueError("simulation_counts requires non-empty targets and positive counts")
        normalised_simulation_counts[target_name] = count
    if not normalised_simulation_counts:
        raise ValueError("simulation_counts must not be empty")

    schema_payload: str | dict[str, Any]
    if isinstance(feature_schema, str):
        schema_payload = feature_schema.strip()
        if not schema_payload:
            raise ValueError("feature_schema must not be blank")
    elif isinstance(feature_schema, Mapping):
        schema_payload = dict(feature_schema)
        if not schema_payload:
            raise ValueError("feature_schema must not be empty")
    else:
        raise TypeError("feature_schema must be a string or mapping")

    timestamp = created_at or datetime.now(UTC)
    manifest: dict[str, Any] = {
        "artifact_type": "prediction_challenger_manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "candidate_id": normalised_candidate_id,
        "variant_id": normalised_variant_id,
        "created_at": _normalise_timestamp(timestamp, field_name="created_at"),
        "cutoff_at": _normalise_timestamp(cutoff_at, field_name="cutoff_at"),
        "default_variant": CHAMPION_VARIANT,
        "runtime_activation_allowed": False,
        "variants": {
            CHAMPION_VARIANT: {
                "role": "champion",
                "default": True,
                "components": [],
            },
            normalised_variant_id: {
                "role": "challenger",
                "default": False,
                "components": normalised_components,
            },
        },
        "provenance": {
            "git": collect_git_provenance(repo_root),
            "configuration": collect_configuration_provenance(repo_root, config_paths),
            "feature_schema": schema_payload,
            "feature_schema_sha256": stable_json_sha256(schema_payload),
            "input_snapshot_ids": _normalise_nonempty_strings(
                input_snapshot_ids,
                field_name="input_snapshot_ids",
            ),
            "seeds": normalised_seeds,
            "checkpoints": list(DEFAULT_REPLAY_CHECKPOINTS),
            "dry_only": True,
            "simulation_counts": normalised_simulation_counts,
        },
        "metadata": dict(metadata or {}),
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    validate_challenger_manifest(manifest)
    return manifest


@dataclass(frozen=True)
class PairedMetricSummary:
    """Champion/challenger means and a paired event-level improvement interval."""

    events: int
    champion_mean: float
    challenger_mean: float
    improvement: float
    ci90_low: float
    ci90_high: float
    event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "events": self.events,
            "champion_mean": self.champion_mean,
            "challenger_mean": self.challenger_mean,
            "improvement": self.improvement,
            "ci90_low": self.ci90_low,
            "ci90_high": self.ci90_high,
            "event_ids": list(self.event_ids),
            "event_set_sha256": stable_json_sha256(list(self.event_ids)),
        }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile from sorted finite values."""
    if not sorted_values:
        raise ValueError("cannot calculate a quantile from no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


def paired_weekend_bootstrap(
    champion_by_event: Mapping[str, float],
    challenger_by_event: Mapping[str, float],
    *,
    confidence: float = 0.90,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_REPLAY_SEEDS[0],
) -> PairedMetricSummary:
    """Summarise paired weekend metrics with a deterministic bootstrap interval.

    Improvements are always ``champion - challenger`` so positive values are
    better for lower-is-better metrics such as MAE, Brier score, or log loss.
    Only finite observations present on both sides are included.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    paired: list[tuple[str, float, float]] = []
    for event_id in sorted(set(champion_by_event).intersection(challenger_by_event)):
        champion = float(champion_by_event[event_id])
        challenger = float(challenger_by_event[event_id])
        if math.isfinite(champion) and math.isfinite(challenger):
            paired.append((str(event_id), champion, challenger))
    if not paired:
        raise ValueError("no finite paired event metrics were available")

    improvements = [champion - challenger for _, champion, challenger in paired]
    rng = random.Random(int(seed))
    bootstrap_means = sorted(
        fmean(improvements[rng.randrange(len(improvements))] for _ in improvements)
        for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return PairedMetricSummary(
        events=len(paired),
        champion_mean=float(fmean(champion for _, champion, _ in paired)),
        challenger_mean=float(fmean(challenger for _, _, challenger in paired)),
        improvement=float(fmean(improvements)),
        ci90_low=_linear_quantile(bootstrap_means, tail),
        ci90_high=_linear_quantile(bootstrap_means, 1.0 - tail),
        event_ids=tuple(event_id for event_id, _, _ in paired),
    )


@dataclass(frozen=True)
class RaceMetricViews:
    """Race replay metrics under actual-grid and predicted-grid conditions."""

    conditional_actual_grid: PairedMetricSummary
    end_to_end_predicted_grid: PairedMetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "conditional_actual_grid": self.conditional_actual_grid.to_dict(),
            "end_to_end_predicted_grid": self.end_to_end_predicted_grid.to_dict(),
        }


def build_race_metric_views(
    *,
    conditional_champion_by_event: Mapping[str, float],
    conditional_challenger_by_event: Mapping[str, float],
    end_to_end_champion_by_event: Mapping[str, float],
    end_to_end_challenger_by_event: Mapping[str, float],
    confidence: float = 0.90,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_REPLAY_SEEDS[0],
) -> RaceMetricViews:
    """Build both mandatory race replay views with identical bootstrap policy."""
    return RaceMetricViews(
        conditional_actual_grid=paired_weekend_bootstrap(
            conditional_champion_by_event,
            conditional_challenger_by_event,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        ),
        end_to_end_predicted_grid=paired_weekend_bootstrap(
            end_to_end_champion_by_event,
            end_to_end_challenger_by_event,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        ),
    )


@dataclass(frozen=True)
class ReplayProvenance:
    """Replay controls that must exactly match the frozen manifest."""

    seeds: Sequence[int]
    simulation_counts: Mapping[str, int]
    dry_only: bool
    checkpoint_event_counts: Mapping[str, int]
    replay_sha256: str | None = None


@dataclass(frozen=True)
class QualifyingGateMetrics:
    """Metrics required to decide one qualifying target's promotion."""

    target: QualifyingTarget
    grid_mae: PairedMetricSummary
    h2h_brier_relative_improvement: float
    h2h_log_loss_delta: float
    ece_delta: float
    interval_coverage: float
    interval_width_relative_change: float
    checkpoint_mae_regressions: Mapping[str, float]
    race_views: RaceMetricViews
    movements_requiring_review: int = 0
    movements_reviewed: int = 0
    manifest: Mapping[str, Any] | None = None
    replay_provenance: ReplayProvenance | None = None


@dataclass(frozen=True)
class RaceGateMetrics:
    """Metrics required to decide one race target's promotion."""

    target: RaceTarget
    race_views: RaceMetricViews
    winner_accuracy_delta_pp: float
    top3_accuracy_delta_pp: float
    dnf_brier_delta: float
    manifest: Mapping[str, Any] | None = None
    replay_provenance: ReplayProvenance | None = None


@dataclass(frozen=True)
class GateResult:
    """Auditable target-specific promotion decision."""

    target: str
    candidate_kind: str
    passed: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    thresholds: Mapping[str, Any]
    variant_id: str
    manifest_sha256: str
    replay_provenance: Mapping[str, Any]
    event_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "target": self.target,
            "candidate_kind": self.candidate_kind,
            "passed": self.passed,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "thresholds": dict(self.thresholds),
            "variant_id": self.variant_id,
            "manifest_sha256": self.manifest_sha256,
            "replay_provenance": dict(self.replay_provenance),
            "event_set_sha256": self.event_set_sha256,
        }


def race_candidate_kind_for_components(
    components: Sequence[str] | frozenset[str],
) -> RaceCandidateKind:
    """Derive the race gate family from registered model components."""

    selected = frozenset(str(component).strip().lower() for component in components)
    if selected.intersection({"r2_no_anchor", "r2_source_anchor"}):
        return "anchor_or_physics"
    if selected.intersection({"r0", "r1"}):
        return "race_input_or_grid_propagation"
    return "qualifying_only"


def _validated_gate_context(
    manifest: Mapping[str, Any] | None,
    replay: ReplayProvenance | None,
) -> ValidatedChallengerManifest:
    """Validate a gate's manifest binding and exact replay controls."""

    if manifest is None:
        raise ValueError("promotion gate metrics require a challenger manifest")
    validated = validate_challenger_manifest(manifest)
    if replay is None:
        raise ValueError("promotion gate metrics require replay provenance")
    if tuple(replay.seeds) != DEFAULT_REPLAY_SEEDS:
        raise ValueError(f"gate replay seeds must be exactly {list(DEFAULT_REPLAY_SEEDS)}")
    if replay.dry_only is not True:
        raise ValueError("challenger v1 gate replay must be dry-only")
    if dict(replay.simulation_counts) != dict(validated.simulation_counts):
        raise ValueError("gate simulation counts do not match the frozen manifest")
    _require_sha256(replay.replay_sha256, field_name="gate replay_sha256")
    counts = replay.checkpoint_event_counts
    if not isinstance(counts, Mapping) or tuple(counts) != DEFAULT_REPLAY_CHECKPOINTS:
        raise ValueError("gate checkpoint accounting must use PRE, FP1, FP2, and FP3 in order")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ) or not any(count > 0 for count in counts.values()):
        raise ValueError("gate checkpoint accounting requires non-negative populated counts")
    return validated


def _event_identity_is_valid(summary: PairedMetricSummary) -> bool:
    """Return whether summary counts are backed by explicit unique event identities."""

    return (
        summary.events > 0
        and len(summary.event_ids) == summary.events
        and len(set(summary.event_ids)) == summary.events
        and all(
            isinstance(event_id, str) and bool(event_id.strip()) for event_id in summary.event_ids
        )
    )


def _replay_provenance_payload(replay: ReplayProvenance) -> dict[str, Any]:
    return {
        "seeds": list(replay.seeds),
        "simulation_counts": dict(replay.simulation_counts),
        "dry_only": replay.dry_only,
        "checkpoint_event_counts": dict(replay.checkpoint_event_counts),
        "replay_sha256": replay.replay_sha256,
    }


def evaluate_qualifying_gate(metrics: QualifyingGateMetrics) -> GateResult:
    """Evaluate the qualifying thresholds from the challenger release plan."""
    if metrics.target not in ("main_qualifying", "sprint_qualifying"):
        raise ValueError(f"unsupported qualifying target: {metrics.target}")
    validated_manifest = _validated_gate_context(
        metrics.manifest,
        metrics.replay_provenance,
    )
    if not validated_manifest.components.intersection({"q0", "q1"}):
        raise ValueError("qualifying gates require a manifest containing Q0 or Q1")
    if validated_manifest.components.intersection({"r0", "r1", "r2_no_anchor", "r2_source_anchor"}):
        raise ValueError("independent qualifying gates cannot contain race components")
    assert metrics.replay_provenance is not None
    checkpoint_counts = metrics.replay_provenance.checkpoint_event_counts
    populated_checkpoints = {
        checkpoint for checkpoint, count in checkpoint_counts.items() if count > 0
    }

    min_events = 30 if metrics.target == "main_qualifying" else 8
    max_checkpoint_regression = max(
        (float(value) for value in metrics.checkpoint_mae_regressions.values()),
        default=-math.inf,
    )
    conditional_improvement = metrics.race_views.conditional_actual_grid.improvement
    end_to_end_improvement = metrics.race_views.end_to_end_predicted_grid.improvement
    finite_values = [
        metrics.grid_mae.champion_mean,
        metrics.grid_mae.challenger_mean,
        metrics.grid_mae.improvement,
        metrics.grid_mae.ci90_low,
        metrics.grid_mae.ci90_high,
        metrics.h2h_brier_relative_improvement,
        metrics.h2h_log_loss_delta,
        metrics.ece_delta,
        metrics.interval_coverage,
        metrics.interval_width_relative_change,
        conditional_improvement,
        end_to_end_improvement,
        *metrics.checkpoint_mae_regressions.values(),
    ]
    checks = {
        "finite_metrics": all(math.isfinite(float(value)) for value in finite_values),
        "minimum_scored_events": metrics.grid_mae.events >= min_events,
        "event_identity_populated": _event_identity_is_valid(metrics.grid_mae),
        "checkpoint_accounting_complete": set(metrics.checkpoint_mae_regressions)
        == populated_checkpoints,
        "checkpoint_accounting_covers_events": sum(checkpoint_counts.values())
        >= metrics.grid_mae.events,
        "grid_mae_improvement": metrics.grid_mae.improvement >= 0.15,
        "grid_mae_ci90_above_zero": metrics.grid_mae.ci90_low > 0.0,
        "h2h_brier_relative_improvement": metrics.h2h_brier_relative_improvement >= 0.05,
        "h2h_log_loss_not_regressed": metrics.h2h_log_loss_delta <= 0.0,
        "ece_within_tolerance": metrics.ece_delta <= 0.02,
        "interval_coverage_in_band": 0.85 <= metrics.interval_coverage <= 0.95,
        "interval_width_within_tolerance": metrics.interval_width_relative_change <= 0.10,
        "checkpoint_slices_within_tolerance": max_checkpoint_regression <= 0.10,
        "conditional_actual_grid_unchanged": abs(conditional_improvement) <= 1e-9,
        "end_to_end_race_not_regressed": end_to_end_improvement >= -0.02,
        "movement_review_complete": (
            metrics.movements_requiring_review >= 0
            and metrics.movements_reviewed >= 0
            and metrics.movements_reviewed >= metrics.movements_requiring_review
        ),
        "race_view_event_counts_match": (
            metrics.race_views.conditional_actual_grid.events
            == metrics.race_views.end_to_end_predicted_grid.events
        ),
        "race_view_event_identity_matches": (
            _event_identity_is_valid(metrics.race_views.conditional_actual_grid)
            and _event_identity_is_valid(metrics.race_views.end_to_end_predicted_grid)
            and metrics.race_views.conditional_actual_grid.event_ids
            == metrics.race_views.end_to_end_predicted_grid.event_ids
        ),
        "qualifying_race_event_identity_matches": (
            metrics.grid_mae.event_ids == metrics.race_views.conditional_actual_grid.event_ids
        ),
    }
    reason_text = {
        "finite_metrics": "one or more qualifying gate metrics are not finite",
        "minimum_scored_events": f"requires at least {min_events} scored events",
        "event_identity_populated": "qualifying replay events are not explicitly identified",
        "checkpoint_accounting_complete": (
            "checkpoint MAE slices do not match the populated checkpoint accounting"
        ),
        "checkpoint_accounting_covers_events": (
            "checkpoint accounting does not cover every scored qualifying event"
        ),
        "grid_mae_improvement": "pre-qualifying grid MAE improvement is below 0.15 positions",
        "grid_mae_ci90_above_zero": "paired weekend bootstrap 90% CI is not above zero",
        "h2h_brier_relative_improvement": "H2H Brier improvement is below 5% relative",
        "h2h_log_loss_not_regressed": "H2H log loss regressed",
        "ece_within_tolerance": "calibration ECE regressed by more than 0.02",
        "interval_coverage_in_band": "nominal 90% position coverage is outside 85-95%",
        "interval_width_within_tolerance": "mean interval width increased by more than 10%",
        "checkpoint_slices_within_tolerance": "a populated checkpoint regressed by over 0.10 MAE",
        "conditional_actual_grid_unchanged": "actual-grid race replay changed under matched seeds",
        "end_to_end_race_not_regressed": "end-to-end predicted-grid race MAE regressed over 0.02",
        "movement_review_complete": "not all material full-grid movements were reviewed",
        "race_view_event_counts_match": "race evaluation views use different event counts",
        "race_view_event_identity_matches": (
            "race evaluation views do not contain the same explicit event identities"
        ),
        "qualifying_race_event_identity_matches": (
            "qualifying and race guardrails do not score the same event identities"
        ),
    }
    thresholds = {
        "minimum_scored_events": min_events,
        "grid_mae_improvement": 0.15,
        "h2h_brier_relative_improvement": 0.05,
        "ece_delta_max": 0.02,
        "interval_coverage": [0.85, 0.95],
        "interval_width_relative_change_max": 0.10,
        "checkpoint_mae_regression_max": 0.10,
        "conditional_actual_grid_abs_change_max": 1e-9,
        "end_to_end_race_regression_max": 0.02,
    }
    reasons = tuple(reason_text[name] for name, passed in checks.items() if not passed)
    return GateResult(
        target=metrics.target,
        candidate_kind="qualifying_only",
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
        thresholds=thresholds,
        variant_id=validated_manifest.variant_id,
        manifest_sha256=validated_manifest.manifest_sha256,
        replay_provenance=_replay_provenance_payload(metrics.replay_provenance),
        event_set_sha256=stable_json_sha256(list(metrics.grid_mae.event_ids)),
    )


def evaluate_race_gate(metrics: RaceGateMetrics) -> GateResult:
    """Evaluate target-specific qualifying/race propagation and physics gates."""
    if metrics.target not in ("grand_prix_race", "sprint_race"):
        raise ValueError(f"unsupported race target: {metrics.target}")
    validated_manifest = _validated_gate_context(
        metrics.manifest,
        metrics.replay_provenance,
    )
    candidate_kind = race_candidate_kind_for_components(validated_manifest.components)
    assert metrics.replay_provenance is not None
    checkpoint_counts = metrics.replay_provenance.checkpoint_event_counts

    conditional_improvement = metrics.race_views.conditional_actual_grid.improvement
    end_to_end_improvement = metrics.race_views.end_to_end_predicted_grid.improvement
    finite_values = [
        metrics.race_views.conditional_actual_grid.champion_mean,
        metrics.race_views.conditional_actual_grid.challenger_mean,
        conditional_improvement,
        metrics.race_views.conditional_actual_grid.ci90_low,
        metrics.race_views.conditional_actual_grid.ci90_high,
        metrics.race_views.end_to_end_predicted_grid.champion_mean,
        metrics.race_views.end_to_end_predicted_grid.challenger_mean,
        end_to_end_improvement,
        metrics.race_views.end_to_end_predicted_grid.ci90_low,
        metrics.race_views.end_to_end_predicted_grid.ci90_high,
        metrics.winner_accuracy_delta_pp,
        metrics.top3_accuracy_delta_pp,
        metrics.dnf_brier_delta,
    ]
    checks: dict[str, bool] = {
        "finite_metrics": all(math.isfinite(float(value)) for value in finite_values),
        "has_paired_events": (
            metrics.race_views.conditional_actual_grid.events > 0
            and metrics.race_views.end_to_end_predicted_grid.events > 0
        ),
        "race_view_event_counts_match": (
            metrics.race_views.conditional_actual_grid.events
            == metrics.race_views.end_to_end_predicted_grid.events
        ),
        "race_view_event_identity_matches": (
            _event_identity_is_valid(metrics.race_views.conditional_actual_grid)
            and _event_identity_is_valid(metrics.race_views.end_to_end_predicted_grid)
            and metrics.race_views.conditional_actual_grid.event_ids
            == metrics.race_views.end_to_end_predicted_grid.event_ids
        ),
        "checkpoint_accounting_covers_events": sum(checkpoint_counts.values())
        >= metrics.race_views.conditional_actual_grid.events,
        "winner_accuracy_not_regressed": metrics.winner_accuracy_delta_pp >= 0.0,
        "top3_accuracy_within_tolerance": metrics.top3_accuracy_delta_pp >= -2.0,
        "dnf_brier_within_tolerance": metrics.dnf_brier_delta <= 0.005,
    }
    reason_text = {
        "finite_metrics": "one or more race gate metrics are not finite",
        "has_paired_events": "both race metric views require paired scored events",
        "race_view_event_counts_match": "race evaluation views use different event counts",
        "race_view_event_identity_matches": (
            "race evaluation views do not contain the same explicit event identities"
        ),
        "checkpoint_accounting_covers_events": (
            "checkpoint accounting does not cover every scored race event"
        ),
        "winner_accuracy_not_regressed": "winner accuracy dropped",
        "top3_accuracy_within_tolerance": "top-three accuracy dropped by over 2 pp",
        "dnf_brier_within_tolerance": "DNF Brier score regressed by more than 0.005",
    }

    if candidate_kind == "qualifying_only":
        checks["conditional_actual_grid_unchanged"] = abs(conditional_improvement) <= 1e-9
        checks["end_to_end_predicted_grid_not_regressed"] = end_to_end_improvement >= -0.02
        reason_text["conditional_actual_grid_unchanged"] = (
            "actual-grid race replay changed under matched seeds"
        )
        reason_text["end_to_end_predicted_grid_not_regressed"] = (
            "end-to-end predicted-grid race MAE regressed over 0.02"
        )
    elif candidate_kind == "race_input_or_grid_propagation":
        checks["end_to_end_predicted_grid_improved"] = end_to_end_improvement >= 0.10
        checks["conditional_actual_grid_within_tolerance"] = conditional_improvement >= -0.05
        reason_text["end_to_end_predicted_grid_improved"] = (
            "end-to-end predicted-grid race MAE improvement is below 0.10"
        )
        reason_text["conditional_actual_grid_within_tolerance"] = (
            "conditional actual-grid race MAE regressed over 0.05"
        )
    else:
        checks["conditional_actual_grid_improved"] = conditional_improvement >= 0.10
        checks["end_to_end_predicted_grid_improved"] = end_to_end_improvement >= 0.10
        reason_text["conditional_actual_grid_improved"] = (
            "conditional actual-grid race MAE improvement is below 0.10"
        )
        reason_text["end_to_end_predicted_grid_improved"] = (
            "end-to-end predicted-grid race MAE improvement is below 0.10"
        )

    thresholds = {
        "qualifying_only": {
            "conditional_actual_grid_abs_change_max": 1e-9,
            "end_to_end_regression_max": 0.02,
        },
        "race_input_or_grid_propagation": {
            "end_to_end_improvement_min": 0.10,
            "conditional_regression_max": 0.05,
        },
        "anchor_or_physics": {
            "conditional_improvement_min": 0.10,
            "end_to_end_improvement_min": 0.10,
        },
        "winner_accuracy_delta_pp_min": 0.0,
        "top3_accuracy_delta_pp_min": -2.0,
        "dnf_brier_delta_max": 0.005,
    }
    reasons = tuple(reason_text[name] for name, passed in checks.items() if not passed)
    return GateResult(
        target=metrics.target,
        candidate_kind=candidate_kind,
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
        thresholds=thresholds,
        variant_id=validated_manifest.variant_id,
        manifest_sha256=validated_manifest.manifest_sha256,
        replay_provenance=_replay_provenance_payload(metrics.replay_provenance),
        event_set_sha256=stable_json_sha256(
            list(metrics.race_views.conditional_actual_grid.event_ids)
        ),
    )
