#!/usr/bin/env python3
"""Compare local dashboard datapoints against Supabase and optionally sync them.

This script is intentionally narrower than ``scripts/backfill_to_db.py``. It focuses on
the prediction artifacts and accuracy-snapshot rows that back dashboard charts for one
race across one or more checkpoints.

Typical usage:

    uv run python scripts/sync_dashboard_datapoints_to_db.py \
      --env-file .env.local \
      --year 2026 \
      --race-name "Chinese Grand Prix" \
      --checkpoint FP1 \
      --checkpoint SQ \
      --checkpoint SPRINT

    uv run python scripts/sync_dashboard_datapoints_to_db.py \
      --env-file .env.local \
      --year 2026 \
      --race-name "Chinese Grand Prix" \
      --checkpoint FP1 \
      --checkpoint SQ \
      --checkpoint SPRINT \
      --include-auxiliary-targets \
      --sync
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.persistence.artifact_store import ArtifactStore

_PRIMARY_TARGET_KEYS: tuple[str, ...] = ("main_qualifying", "grand_prix_race")


def _artifact_store_class() -> type[ArtifactStore]:
    """Return the artifact store class after the project root is on ``sys.path``."""
    from src.persistence.artifact_store import ArtifactStore

    return ArtifactStore


def _accuracy_snapshot_key(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    target_key: str,
) -> str:
    """Build one accuracy snapshot artifact key lazily.

    The import lives here so this script stays runnable as
    ``uv run python scripts/...`` without module-level path hacks that lint badly.
    """
    from src.utils.accuracy_snapshots import accuracy_snapshot_artifact_key

    return accuracy_snapshot_artifact_key(
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session,
        target_key=target_key,
    )


@dataclass(frozen=True)
class ArtifactSpec:
    """Describe one artifact row that should match between local files and Supabase."""

    artifact_type: str
    artifact_key: str
    local_path: Path


@dataclass(frozen=True)
class ArtifactComparison:
    """Store one local-vs-remote comparison result."""

    spec: ArtifactSpec
    local_exists: bool
    remote_exists: bool
    local_checksum: str | None
    remote_checksum: str | None
    match: bool


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare local dashboard datapoints with Supabase and optionally sync them.",
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, for example 2026.")
    parser.add_argument(
        "--race-name",
        required=True,
        help="Exact race name, for example 'Chinese Grand Prix'.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint code to inspect. Repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Local data root containing predictions and accuracy snapshots.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional env file to load before connecting to Supabase, for example .env.local.",
    )
    parser.add_argument(
        "--include-auxiliary-targets",
        action="store_true",
        help="Include every local accuracy snapshot file under each checkpoint directory.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Write mismatched local artifacts to Supabase and verify the saved payloads.",
    )
    return parser.parse_args()


def _load_env_file(env_file: Path) -> None:
    """Load a simple KEY=VALUE env file into the process environment."""
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip())


def _configure_db_environment(env_file: Path | None) -> None:
    """Load optional env vars and force DB-backed storage for the script run."""
    if env_file is not None:
        _load_env_file(env_file)
    os.environ["USE_DB_STORAGE"] = "db_only"


def _payload_checksum(payload: dict[str, Any] | None) -> str | None:
    """Return a stable short checksum for a JSON payload."""
    if payload is None:
        return None
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _load_local_payload(path: Path) -> dict[str, Any] | None:
    """Load a local JSON object payload when the file exists."""
    if not path.exists():
        return None
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _build_accuracy_snapshot_specs(
    *,
    store: ArtifactStore,
    data_root: Path,
    year: int,
    race_name: str,
    checkpoint: str,
    include_auxiliary_targets: bool,
) -> list[ArtifactSpec]:
    """Build local/remote specs for accuracy snapshots at one checkpoint."""
    checkpoint_upper = str(checkpoint).strip().upper()
    if include_auxiliary_targets:
        snapshot_dir = data_root / "accuracy_snapshot" / str(year) / race_name / checkpoint_upper
        if not snapshot_dir.exists():
            return []
        target_keys = sorted(file_path.stem for file_path in snapshot_dir.glob("*.json"))
    else:
        target_keys = list(_PRIMARY_TARGET_KEYS)

    specs: list[ArtifactSpec] = []
    for target_key in target_keys:
        artifact_key = _accuracy_snapshot_key(
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint_upper,
            target_key=target_key,
        )
        specs.append(
            ArtifactSpec(
                artifact_type="accuracy_snapshot",
                artifact_key=artifact_key,
                local_path=store._get_file_path("accuracy_snapshot", artifact_key),
            )
        )
    return specs


def build_artifact_specs(
    *,
    data_root: Path,
    year: int,
    race_name: str,
    checkpoints: list[str],
    include_auxiliary_targets: bool,
) -> list[ArtifactSpec]:
    """Build the artifact list that should be compared or synced."""
    store = _artifact_store_class()(data_root=data_root)
    deduped: dict[tuple[str, str], ArtifactSpec] = {}

    for checkpoint in checkpoints:
        checkpoint_upper = str(checkpoint).strip().upper()
        prediction_key = f"{int(year)}::{race_name}::{checkpoint_upper}"
        prediction_spec = ArtifactSpec(
            artifact_type="prediction",
            artifact_key=prediction_key,
            local_path=store._get_file_path("prediction", prediction_key),
        )
        deduped[(prediction_spec.artifact_type, prediction_spec.artifact_key)] = prediction_spec

        for snapshot_spec in _build_accuracy_snapshot_specs(
            store=store,
            data_root=data_root,
            year=year,
            race_name=race_name,
            checkpoint=checkpoint_upper,
            include_auxiliary_targets=include_auxiliary_targets,
        ):
            deduped[(snapshot_spec.artifact_type, snapshot_spec.artifact_key)] = snapshot_spec

    return [deduped[key] for key in sorted(deduped, key=lambda item: (item[0], item[1]))]


def compare_artifacts(
    *,
    store: ArtifactStore,
    specs: list[ArtifactSpec],
) -> list[ArtifactComparison]:
    """Compare local payloads with the latest Supabase payloads for each artifact."""
    comparisons: list[ArtifactComparison] = []
    for spec in specs:
        local_payload = _load_local_payload(spec.local_path)
        remote_payload = store._read_db(spec.artifact_type, spec.artifact_key, "latest", None)
        comparisons.append(
            ArtifactComparison(
                spec=spec,
                local_exists=local_payload is not None,
                remote_exists=remote_payload is not None,
                local_checksum=_payload_checksum(local_payload),
                remote_checksum=_payload_checksum(remote_payload),
                match=local_payload == remote_payload,
            )
        )
    return comparisons


def _comparison_status(comparison: ArtifactComparison) -> str:
    """Return a small status label for one comparison result."""
    if comparison.match:
        return "MATCH"
    if comparison.local_exists and not comparison.remote_exists:
        return "REMOTE_MISSING"
    if comparison.remote_exists and not comparison.local_exists:
        return "LOCAL_MISSING"
    if not comparison.local_exists and not comparison.remote_exists:
        return "MISSING_BOTH"
    return "MISMATCH"


def _print_comparisons(comparisons: list[ArtifactComparison]) -> None:
    """Print a compact comparison report."""
    print("\nArtifact comparison:")
    for comparison in comparisons:
        print(
            f"  - {_comparison_status(comparison):<14} "
            f"{comparison.spec.artifact_type}::{comparison.spec.artifact_key}"
        )
        print(f"    local : {comparison.local_checksum or '-'}")
        print(f"    remote: {comparison.remote_checksum or '-'}")


def sync_mismatched_artifacts(
    *,
    store: ArtifactStore,
    comparisons: list[ArtifactComparison],
) -> list[ArtifactComparison]:
    """Write local payloads for mismatched artifacts to Supabase, then re-compare them."""
    synced_specs: list[ArtifactSpec] = []

    for comparison in comparisons:
        if comparison.match or not comparison.local_exists:
            continue

        payload = _load_local_payload(comparison.spec.local_path)
        if payload is None:
            continue

        store.save_artifact(
            artifact_type=comparison.spec.artifact_type,
            artifact_key=comparison.spec.artifact_key,
            data=payload,
        )
        synced_specs.append(comparison.spec)

    if not synced_specs:
        return []
    return compare_artifacts(store=store, specs=synced_specs)


def main() -> int:
    """Run the dashboard datapoint comparison or sync workflow."""
    args = _parse_args()
    _configure_db_environment(args.env_file.resolve() if args.env_file else None)

    data_root = args.data_root.resolve()
    race_name = str(args.race_name).strip()
    checkpoints = [str(checkpoint).strip().upper() for checkpoint in args.checkpoints]
    store = _artifact_store_class()(data_root=data_root)
    specs = build_artifact_specs(
        data_root=data_root,
        year=int(args.year),
        race_name=race_name,
        checkpoints=checkpoints,
        include_auxiliary_targets=bool(args.include_auxiliary_targets),
    )

    print("=" * 72)
    print("Dashboard Datapoint Sync")
    print("=" * 72)
    print(f"Race: {race_name} ({int(args.year)})")
    print(f"Checkpoints: {', '.join(checkpoints)}")
    print(f"Data root: {data_root}")
    print(f"Include auxiliary targets: {bool(args.include_auxiliary_targets)}")
    print(f"Sync enabled: {bool(args.sync)}")
    print(f"Artifacts selected: {len(specs)}")

    comparisons = compare_artifacts(store=store, specs=specs)
    _print_comparisons(comparisons)

    mismatches = [comparison for comparison in comparisons if not comparison.match]
    print(f"\nInitial mismatches: {len(mismatches)}")
    if not args.sync:
        return 0 if not mismatches else 1

    print("\nSyncing local payloads to Supabase...")
    synced = sync_mismatched_artifacts(store=store, comparisons=comparisons)
    if synced:
        print("\nPost-sync verification:")
        _print_comparisons(synced)
    else:
        print("\nNothing needed syncing.")

    remaining = [comparison for comparison in synced if not comparison.match]
    print(f"\nRemaining mismatches after sync: {len(remaining)}")
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
