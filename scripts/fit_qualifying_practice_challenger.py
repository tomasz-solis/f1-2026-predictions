#!/usr/bin/env python3
"""Fit dry-only Q1 normalization and checkpoint models into challenger storage.

This CLI intentionally cannot write active Q1 normalization/model paths. A canonical
candidate root or an explicit output below a challenger-only tree is required, and an
existing artifact is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.challenger_governance import DEFAULT_REPLAY_SEEDS  # noqa: E402
from src.models.qualifying_practice_bundle import (  # noqa: E402
    build_qualifying_practice_bundle,
    build_qualifying_practice_launch_envelope,
)
from src.models.qualifying_practice_challenger import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    calibrate_temperature,
    fit_bradley_terry_model,
)
from src.models.qualifying_practice_evidence import (  # noqa: E402
    NORMALIZATION_COMPARISON_COLUMNS,
    PracticeNormalizationPrior,
    fit_practice_normalization,
)
from src.persistence.research_sidecar import DEFAULT_RESEARCH_SIDECAR_ROOT  # noqa: E402
from src.utils.file_operations import atomic_json_write  # noqa: E402

CHECKPOINTS = ("PRE", "FP1", "FP2", "FP3")
SESSION_KINDS = ("main", "sprint")
MINIMUM_EVENTS = {"main": 30, "sprint": 8}
CHALLENGER_MODEL_ROOT = Path("data/processed/model_artifacts/qualifying_practice/challengers")
_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "dry"})


def _checkpoint(value: str) -> str:
    resolved = str(value).strip().upper()
    if resolved not in CHECKPOINTS:
        raise argparse.ArgumentTypeError(f"checkpoint must be one of {', '.join(CHECKPOINTS)}")
    return resolved


def _timezone_aware_timestamp(value: str) -> str:
    candidate = str(value).strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoff must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("cutoff must include a timezone")
    return candidate


def _parse_aware_timestamp(value: Any, *, field_name: str) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise ValueError(f"{field_name} must not contain missing timestamps")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains an invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} timestamps must include a timezone: {value!r}")
    return parsed.tz_convert("UTC")


def _generation_timestamp(*, cutoff: str) -> str:
    generated = datetime.now(UTC)
    cutoff_at = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00")).astimezone(UTC)
    if generated < cutoff_at:
        raise ValueError("artifact cutoff cannot be later than its actual generation time")
    return generated.isoformat()


def _validate_cutoff(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    cutoff: str,
    input_name: str,
) -> str:
    """Fail closed unless every selected observation predates the frozen cutoff."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"{input_name} is missing timestamp column {timestamp_column!r}")
    cutoff_timestamp = _parse_aware_timestamp(cutoff, field_name="cutoff")
    timestamps = [
        _parse_aware_timestamp(value, field_name=f"{input_name}.{timestamp_column}")
        for value in frame[timestamp_column]
    ]
    maximum = max(timestamps)
    violations = [value for value in timestamps if value >= cutoff_timestamp]
    if violations:
        raise ValueError(
            f"{input_name}.{timestamp_column} must be strictly before cutoff {cutoff}; "
            f"found {len(violations)} future/equal selected row(s), max={maximum.isoformat()}"
        )
    return maximum.isoformat()


def _positive_float(value: str) -> float:
    try:
        resolved = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not np.isfinite(resolved) or resolved <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return resolved


def _nonnegative_float(value: str) -> float:
    try:
        resolved = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not np.isfinite(resolved) or resolved < 0:
        raise argparse.ArgumentTypeError("value must be non-negative and finite")
    return resolved


def _resolved_path(path: Path, *, repo_root: Path) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    return candidate.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_path(path: Path, *, repo_root: Path) -> Path:
    """Resolve a new JSON destination and reject every active artifact location."""

    root = repo_root.resolve()
    destination = _resolved_path(path, repo_root=root)
    allowed_roots = (
        (root / DEFAULT_RESEARCH_SIDECAR_ROOT).resolve(),
        (root / CHALLENGER_MODEL_ROOT).resolve(),
    )
    if destination.suffix.lower() != ".json":
        raise ValueError("challenger artifact output must be a .json file")
    if not any(_is_within(destination, allowed) for allowed in allowed_roots):
        allowed = ", ".join(str(value) for value in allowed_roots)
        raise ValueError(f"output must be below a challenger-only root: {allowed}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen challenger artifact: {destination}")
    return destination


def _candidate_root(args: argparse.Namespace, *, repo_root: Path) -> Path | None:
    raw_root = getattr(args, "candidate_root", None)
    if raw_root is None:
        return None
    root = _resolved_path(raw_root, repo_root=repo_root)
    if root.name != str(args.candidate_id).strip():
        raise ValueError("candidate root directory must be named exactly candidate_id")
    return root


def _resolved_artifact_output(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    artifact_kind: str,
) -> Path:
    """Resolve either an explicit legacy output or the canonical candidate layout."""

    root = _candidate_root(args, repo_root=repo_root)
    raw_output = getattr(args, "output", None)
    expected: Path | None = None
    if root is not None:
        checkpoint = str(args.checkpoint).strip().lower()
        session_kind = str(args.session_kind).strip().lower()
        if artifact_kind == "normalization":
            track_class = str(args.track_class).strip().lower()
            expected = root / "normalizations" / session_kind / checkpoint / f"{track_class}.json"
        elif artifact_kind == "model":
            expected = root / "models" / session_kind / f"{checkpoint}.json"
        else:
            raise ValueError(f"unsupported Q1 artifact kind: {artifact_kind}")
    if raw_output is None and expected is None:
        raise ValueError("provide --candidate-root for canonical layout or an explicit --output")
    if raw_output is not None and expected is not None:
        resolved_output = _resolved_path(raw_output, repo_root=repo_root)
        if resolved_output != expected.resolve():
            raise ValueError(f"explicit output does not match candidate layout: {expected}")
    destination = expected if expected is not None else raw_output
    if destination is None:  # Defensive narrowing for static type checkers.
        raise ValueError("Q1 artifact output could not be resolved")
    return validate_output_path(destination, repo_root=repo_root)


def _read_frame(path: Path) -> pd.DataFrame:
    source = path.resolve()
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    if suffix == ".json":
        return pd.read_json(source)
    raise ValueError("input must be CSV, JSON, JSONL, or NDJSON")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dry_mask(values: pd.Series) -> pd.Series:
    def is_dry(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or pd.isna(value):
            return False
        return str(value).strip().lower() in _TRUE_VALUES

    return values.map(is_dry).astype(bool)


def _checkpoint_dry_rows(
    frame: pd.DataFrame,
    *,
    checkpoint: str,
    session_kind: str,
    required_columns: set[str],
    track_class: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"checkpoint", "session_kind", "is_dry", *required_columns}
    if track_class is not None:
        required.add("track_class")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing prepared-data columns: {missing}")
    checkpoint_mask = frame["checkpoint"].astype(str).str.strip().str.upper().eq(checkpoint)
    session_kind_mask = frame["session_kind"].astype(str).str.strip().str.lower().eq(session_kind)
    dry_mask = _dry_mask(frame["is_dry"])
    track_class_mask = pd.Series(True, index=frame.index, dtype=bool)
    if track_class is not None:
        track_class_mask = frame["track_class"].astype(str).str.strip().str.lower().eq(track_class)
    selected_mask = checkpoint_mask & session_kind_mask & dry_mask & track_class_mask
    selected = frame.loc[selected_mask].copy().reset_index(drop=True)
    if selected.empty:
        suffix = f"/{track_class}" if track_class is not None else ""
        raise ValueError(
            f"no dry {session_kind}/{checkpoint}{suffix} rows remain after artifact filtering"
        )
    return selected, {
        "input_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "excluded_other_checkpoint": int((~checkpoint_mask).sum()),
        "excluded_other_session_kind": int((checkpoint_mask & ~session_kind_mask).sum()),
        "excluded_non_dry": int((checkpoint_mask & ~dry_mask).sum()),
        "excluded_other_track_class": int((checkpoint_mask & ~track_class_mask).sum()),
    }


def _normalization_prior(path: Path | None) -> tuple[PracticeNormalizationPrior, str | None]:
    if path is None:
        return PracticeNormalizationPrior(), None
    source = path.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("normalization prior must be a JSON object")
    nested = payload.get("normalization")
    values = nested if isinstance(nested, dict) else payload
    effects = values.get("compound_effect_s", {})
    if not isinstance(effects, dict):
        raise ValueError("normalization prior compound_effect_s must be an object")
    return (
        PracticeNormalizationPrior(
            reference_compound=str(values.get("reference_compound", "SOFT")),
            compound_effect_s={str(key): float(value) for key, value in effects.items()},
            tyre_age_effect_s_per_lap=float(values.get("tyre_age_effect_s_per_lap", 0.0)),
            evolution_effect_s_per_unit=float(values.get("evolution_effect_s_per_unit", 0.0)),
            uncertainty_s=float(values.get("measurement_uncertainty_s", 0.5)),
            source=str(values.get("prior_source", values.get("source", source.stem))),
        ),
        _sha256(source),
    )


def _metadata(
    *,
    args: argparse.Namespace,
    source: Path,
    row_counts: dict[str, int],
    max_input_timestamp: str,
    event_count: int | None = None,
    calibration_source: Path | None = None,
    calibration_max_input_timestamp: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": str(args.candidate_id),
        "checkpoint": str(args.checkpoint),
        "session_kind": str(args.session_kind),
        "dry_only": True,
        "cutoff_timestamp": str(args.cutoff),
        "input_snapshot_id": str(args.input_snapshot_id),
        "input_sha256": _sha256(source),
        "replay_seeds": list(DEFAULT_REPLAY_SEEDS),
        "row_counts": dict(row_counts),
        "event_timestamp_column": str(args.event_timestamp_column),
        "max_input_timestamp": max_input_timestamp,
    }
    if event_count is not None:
        payload["training_event_count"] = int(event_count)
        payload["minimum_training_events"] = int(MINIMUM_EVENTS[args.session_kind])
    if calibration_source is not None:
        payload["calibration_input_sha256"] = _sha256(calibration_source)
        payload["temperature_calibration"] = "disjoint_holdout_log_loss"
        payload["calibration_max_input_timestamp"] = calibration_max_input_timestamp
    return payload


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"challenger artifact already exists and is immutable: {path}")
    atomic_json_write(path, payload, create_backup=False)


def fit_normalization(args: argparse.Namespace) -> Path:
    """Fit one checkpoint/track-class normalization artifact."""

    repo_root = args.repo_root.resolve()
    source = _resolved_path(args.input, repo_root=repo_root)
    output = _resolved_artifact_output(
        args,
        repo_root=repo_root,
        artifact_kind="normalization",
    )
    frame = _read_frame(source)
    selected, counts = _checkpoint_dry_rows(
        frame,
        checkpoint=args.checkpoint,
        session_kind=args.session_kind,
        required_columns=set(NORMALIZATION_COMPARISON_COLUMNS),
        track_class=str(args.track_class).strip().lower(),
    )
    max_input_timestamp = _validate_cutoff(
        selected,
        timestamp_column=args.event_timestamp_column,
        cutoff=args.cutoff,
        input_name="normalization input",
    )
    prior_path = _resolved_path(args.prior, repo_root=repo_root) if args.prior is not None else None
    prior, prior_digest = _normalization_prior(prior_path)
    fitted = fit_practice_normalization(
        selected,
        prior=prior,
        prior_strength=args.prior_strength,
    )
    metadata = _metadata(
        args=args,
        source=source,
        row_counts=counts,
        max_input_timestamp=max_input_timestamp,
    )
    metadata["track_class"] = str(args.track_class).strip().lower()
    metadata["prior_sha256"] = prior_digest
    artifact = {
        "artifact_type": "qualifying_practice_normalization",
        "schema_version": 1,
        "generated_at": _generation_timestamp(cutoff=args.cutoff),
        "normalization": fitted.to_dict(),
        "training_metadata": metadata,
    }
    _write_artifact(output, artifact)
    return output


def _validate_training_rows(frame: pd.DataFrame, *, feature_columns: tuple[str, ...]) -> None:
    required = {"event_id", "driver", "actual_position", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing qualifying model columns: {missing}")
    if frame[["event_id", "driver", "actual_position"]].isna().any(axis=None):
        raise ValueError("event_id, driver, and actual_position must not be missing")
    if (
        frame["event_id"].astype(str).str.strip().eq("").any()
        or frame["driver"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("event_id and driver must not be blank")
    if frame.duplicated(["event_id", "driver"]).any():
        raise ValueError("prepared rows must be unique by event_id and driver")
    positions = pd.to_numeric(frame["actual_position"], errors="coerce")
    if positions.isna().any() or positions.le(0).any() or positions.mod(1).ne(0).any():
        raise ValueError("actual_position must contain positive integers")
    for event_id, event_positions in positions.groupby(frame["event_id"], sort=False):
        ordered = sorted(int(value) for value in event_positions)
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError(
                f"event {event_id!s} is not a complete, uniquely ranked qualifying grid"
            )
    if any(len(group) < 2 for _, group in frame.groupby("event_id")):
        raise ValueError("every training event must contain at least two drivers")


def _calibration_pairs(
    model: Any,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    differences: list[float] = []
    outcomes: list[float] = []
    event_count = 0
    for _, event in frame.groupby("event_id", sort=True):
        event_count += 1
        utilities = model.utilities(event)
        ordered = event.sort_values("driver", kind="mergesort")
        drivers = [str(value) for value in ordered["driver"]]
        positions = {
            str(driver): float(position)
            for driver, position in zip(
                ordered["driver"],
                ordered["actual_position"],
                strict=True,
            )
        }
        for left_index, left in enumerate(drivers):
            for right in drivers[left_index + 1 :]:
                differences.append(float(utilities[left] - utilities[right]))
                outcomes.append(float(positions[left] < positions[right]))
    if not differences:
        raise ValueError("calibration data has no usable driver pairs")
    return np.asarray(differences), np.asarray(outcomes), event_count


def fit_checkpoint_model(args: argparse.Namespace) -> Path:
    """Fit and holdout-calibrate one checkpoint Bradley--Terry artifact."""

    repo_root = args.repo_root.resolve()
    source = _resolved_path(args.input, repo_root=repo_root)
    calibration_source = _resolved_path(args.calibration_input, repo_root=repo_root)
    output = _resolved_artifact_output(
        args,
        repo_root=repo_root,
        artifact_kind="model",
    )
    feature_columns = tuple(args.feature_column or DEFAULT_FEATURE_COLUMNS)

    training, counts = _checkpoint_dry_rows(
        _read_frame(source),
        checkpoint=args.checkpoint,
        session_kind=args.session_kind,
        required_columns={"event_id", "driver", "actual_position", *feature_columns},
    )
    calibration, calibration_counts = _checkpoint_dry_rows(
        _read_frame(calibration_source),
        checkpoint=args.checkpoint,
        session_kind=args.session_kind,
        required_columns={"event_id", "driver", "actual_position", *feature_columns},
    )
    max_input_timestamp = _validate_cutoff(
        training,
        timestamp_column=args.event_timestamp_column,
        cutoff=args.cutoff,
        input_name="training input",
    )
    calibration_max_input_timestamp = _validate_cutoff(
        calibration,
        timestamp_column=args.event_timestamp_column,
        cutoff=args.cutoff,
        input_name="calibration input",
    )
    _validate_training_rows(training, feature_columns=feature_columns)
    _validate_training_rows(calibration, feature_columns=feature_columns)
    training_events = {str(value) for value in training["event_id"]}
    calibration_events = {str(value) for value in calibration["event_id"]}
    overlap = sorted(training_events.intersection(calibration_events))
    if overlap:
        raise ValueError(
            f"temperature calibration events must be disjoint from training events: {overlap[:5]}"
        )
    minimum = MINIMUM_EVENTS[args.session_kind]
    if len(training_events) < minimum:
        raise ValueError(
            f"{args.session_kind} Q1 fitting requires at least {minimum} training events; "
            f"found {len(training_events)}"
        )

    model = fit_bradley_terry_model(
        training,
        checkpoint=args.checkpoint,
        feature_columns=feature_columns,
        regularization_c=args.regularization_c,
        temperature=1.0,
    )
    differences, outcomes, calibration_event_count = _calibration_pairs(model, calibration)
    temperature = calibrate_temperature(
        utility_differences=differences,
        outcomes=outcomes,
    )
    model = replace(
        model,
        temperature=temperature,
        generated_at=_generation_timestamp(cutoff=args.cutoff),
    )
    counts.update(
        {
            "calibration_input_rows": calibration_counts["input_rows"],
            "calibration_selected_rows": calibration_counts["selected_rows"],
            "calibration_event_count": calibration_event_count,
        }
    )
    artifact = model.to_dict()
    artifact["training_metadata"] = _metadata(
        args=args,
        source=source,
        row_counts=counts,
        max_input_timestamp=max_input_timestamp,
        event_count=len(training_events),
        calibration_source=calibration_source,
        calibration_max_input_timestamp=calibration_max_input_timestamp,
    )
    _write_artifact(output, artifact)
    return output


def build_candidate_bundle(args: argparse.Namespace) -> Path:
    """Bind canonical Q1 artifacts to one immutable challenger manifest."""

    repo_root = args.repo_root.resolve()
    root = _candidate_root(args, repo_root=repo_root)
    if root is None:
        raise ValueError("bundle creation requires --candidate-root")
    output = validate_output_path(
        Path(args.output) if args.output is not None else root / "bundle.json",
        repo_root=repo_root,
    )
    if output.parent != root:
        raise ValueError("Q1 bundle output must be candidate_root/bundle.json")
    manifest_path = _resolved_path(args.manifest, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("challenger manifest must contain a JSON object")
    model_paths = sorted((root / "models").glob("*/*.json"))
    normalization_paths = sorted((root / "normalizations").glob("*/*/*.json"))
    bundle = build_qualifying_practice_bundle(
        candidate_id=str(args.candidate_id),
        variant_id=str(args.variant_id),
        manifest=manifest,
        bundle_directory=root,
        model_paths=model_paths,
        normalization_paths=normalization_paths,
    )
    _write_artifact(output, bundle)
    return output


def build_candidate_launch(args: argparse.Namespace) -> Path:
    """Bind semantic config, manifest, and bundle into the final launch envelope."""

    repo_root = args.repo_root.resolve()
    root = _candidate_root(args, repo_root=repo_root)
    if root is None:
        raise ValueError("launch creation requires --candidate-root")
    output = validate_output_path(
        Path(args.output) if args.output is not None else root / "launch.json",
        repo_root=repo_root,
    )
    if output.parent != root or output.name != "launch.json":
        raise ValueError("Q1 launch output must be candidate_root/launch.json")
    manifest_path = _resolved_path(args.manifest, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("challenger manifest must contain a JSON object")
    semantic_config_path = _resolved_path(args.semantic_config, repo_root=repo_root)
    launch = build_qualifying_practice_launch_envelope(
        candidate_id=str(args.candidate_id),
        variant_id=str(args.variant_id),
        manifest=manifest,
        bundle_path=root / "bundle.json",
        launch_directory=root,
        semantic_config_path=semantic_config_path,
        repo_root=repo_root,
    )
    _write_artifact(output, launch)
    return output


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        help="Canonical immutable root named after candidate-id; derives artifact output paths.",
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--input-snapshot-id", required=True)
    parser.add_argument("--checkpoint", type=_checkpoint, required=True)
    parser.add_argument("--session-kind", choices=SESSION_KINDS, required=True)
    parser.add_argument("--cutoff", type=_timezone_aware_timestamp, required=True)
    parser.add_argument("--event-timestamp-column", default="event_start_at")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalization = subparsers.add_parser(
        "normalization",
        help="Fit same-driver compound/age/evolution normalization.",
    )
    _shared_arguments(normalization)
    normalization.add_argument("--track-class", required=True)
    normalization.add_argument("--prior", type=Path)
    normalization.add_argument("--prior-strength", type=_nonnegative_float, default=8.0)
    normalization.set_defaults(handler=fit_normalization)

    model = subparsers.add_parser(
        "model",
        help="Fit and out-of-fold calibrate one checkpoint Bradley--Terry model.",
    )
    _shared_arguments(model)
    model.add_argument("--calibration-input", type=Path, required=True)
    model.add_argument("--feature-column", action="append")
    model.add_argument("--regularization-c", type=_positive_float, default=1.0)
    model.set_defaults(handler=fit_checkpoint_model)

    bundle = subparsers.add_parser(
        "bundle",
        help="Bind canonical checkpoint/track-class artifacts to a frozen manifest.",
    )
    bundle.add_argument("--candidate-id", required=True)
    bundle.add_argument("--variant-id", required=True)
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--candidate-root", type=Path, required=True)
    bundle.add_argument("--output", type=Path)
    bundle.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    bundle.set_defaults(handler=build_candidate_bundle)

    launch = subparsers.add_parser(
        "launch",
        help="Bind semantic config, manifest, and bundle into an immutable launch envelope.",
    )
    launch.add_argument("--candidate-id", required=True)
    launch.add_argument("--variant-id", required=True)
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--semantic-config", type=Path, required=True)
    launch.add_argument("--candidate-root", type=Path, required=True)
    launch.add_argument("--output", type=Path)
    launch.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    launch.set_defaults(handler=build_candidate_launch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.handler(args)
    bundle_digest = None
    launch_digest = None
    if args.command == "bundle":
        bundle_payload = json.loads(Path(output).read_text(encoding="utf-8"))
        bundle_digest = (
            bundle_payload.get("bundle_sha256") if isinstance(bundle_payload, dict) else None
        )
    if args.command == "launch":
        launch_payload = json.loads(Path(output).read_text(encoding="utf-8"))
        launch_digest = (
            launch_payload.get("launch_sha256") if isinstance(launch_payload, dict) else None
        )
    print(
        json.dumps(
            {
                "artifact": args.command,
                "checkpoint": getattr(args, "checkpoint", None),
                "dry_only": True,
                "output": str(output),
                "bundle_sha256": bundle_digest,
                "launch_sha256": launch_digest,
                "replay_seeds": list(DEFAULT_REPLAY_SEEDS),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
