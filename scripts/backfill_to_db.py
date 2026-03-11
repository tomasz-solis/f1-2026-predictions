#!/usr/bin/env python3
"""
Backfill script: Migrate local persistence data to Supabase.

This script discovers file-backed artifacts plus runtime-state files under the
`data/` directory and writes them to the matching Supabase storage layer:

- artifact payloads -> `artifacts`
- runtime-state payloads -> `runtime_state`

It preserves explicit artifact versions where available and upserts runtime
state by namespace and state key.

Usage:
    # Dry run (no writes)
    python scripts/backfill_to_db.py --dry-run

    # Actual migration
    export SUPABASE_URL=https://xxxxx.supabase.co
    export SUPABASE_KEY=eyJhbGc...
    python scripts/backfill_to_db.py

    # Batch size control
    python scripts/backfill_to_db.py --batch-size 50
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.driver_experience import load_driver_debuts_from_csv
from persistence.artifact_store import ArtifactStore
from persistence.config import USE_DB_STORAGE
from persistence.runtime_state_store import RuntimeStateStore
from utils.car_snapshot_history import SNAPSHOT_ARTIFACT_TYPE


def compute_checksum(data: dict) -> str:
    """Compute SHA256 checksum of JSON data."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def _load_json_file(file_path: Path) -> dict[str, Any]:
    """Load a JSON file and return an object payload."""
    with open(file_path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {file_path}")
    return loaded


def _season_year_from_payload(payload: dict[str, Any], default_year: int = 2026) -> int:
    """Resolve a season year from common payload fields with a safe fallback."""
    for key in ("year", "season", "season_year", "characteristics_year"):
        raw_value = payload.get(key)
        try:
            if raw_value is not None:
                return int(raw_value)
        except (TypeError, ValueError):
            continue
    return int(default_year)


def discover_artifacts(data_root: Path) -> list[dict[str, Any]]:
    """
    Discover all JSON artifacts in data/ directory.

    Returns list of artifact metadata:
    - file_path: Path to JSON file
    - artifact_type: Type classification
    - artifact_key: Unique key
    - data: Loaded JSON data
    - version: Extracted version (if available)
    - checksum: Data checksum for validation
    """
    artifacts: list[dict[str, Any]] = []

    # 1. Car characteristics (multiple years)
    car_chars_dir = data_root / "processed" / "car_characteristics"
    if car_chars_dir.exists():
        for file in sorted(car_chars_dir.glob("*_car_characteristics.json")):
            try:
                data = _load_json_file(file)
                year = file.stem.split("_")[0]
                artifacts.append(
                    {
                        "file_path": file,
                        "artifact_type": "car_characteristics",
                        "artifact_key": f"{year}::car_characteristics",
                        "data": data,
                        "version": data.get("version", 1),
                        "checksum": compute_checksum(data),
                    }
                )
                print(f"  Found: {file.relative_to(data_root)}")
            except Exception as e:
                print(f"  [WARN]  Failed to load {file}: {e}")

    # 2. Session snapshots
    snapshot_dir = data_root / SNAPSHOT_ARTIFACT_TYPE
    if snapshot_dir.exists():
        for file in sorted(snapshot_dir.rglob("*.json")):
            parts = file.relative_to(snapshot_dir).parts
            if len(parts) < 3:
                print(f"  [WARN]  Skipping malformed snapshot path: {file.relative_to(data_root)}")
                continue
            year_token = str(parts[0]).strip()
            event_name = str(parts[1]).strip()
            session_name = file.stem.strip()
            if not year_token or not event_name or not session_name:
                print(f"  [WARN]  Skipping malformed snapshot path: {file.relative_to(data_root)}")
                continue
            try:
                data = _load_json_file(file)
                artifacts.append(
                    {
                        "file_path": file,
                        "artifact_type": SNAPSHOT_ARTIFACT_TYPE,
                        "artifact_key": f"{year_token}::{event_name}::{session_name}",
                        "data": data,
                        "version": data.get("version", 1),
                        "checksum": compute_checksum(data),
                    }
                )
                print(f"  Found: {file.relative_to(data_root)}")
            except Exception as e:
                print(f"  [WARN]  Failed to load {file}: {e}")

    # 3. Driver characteristics
    season_driver_dir = data_root / "processed" / "driver_characteristics"
    season_driver_files = sorted(season_driver_dir.glob("*_driver_characteristics.json"))
    if season_driver_files:
        for file in season_driver_files:
            try:
                data = _load_json_file(file)
                year = file.stem.split("_")[0]
                artifacts.append(
                    {
                        "file_path": file,
                        "artifact_type": "driver_characteristics",
                        "artifact_key": f"{year}::driver_characteristics",
                        "data": data,
                        "version": data.get("version", 1),
                        "checksum": compute_checksum(data),
                    }
                )
                print(f"  Found: {file.relative_to(data_root)}")
            except Exception as e:
                print(f"  [WARN]  Failed to load {file}: {e}")
    else:
        driver_file = data_root / "processed" / "driver_characteristics.json"
        if driver_file.exists():
            try:
                data = _load_json_file(driver_file)
                year = _season_year_from_payload(data)
                artifacts.append(
                    {
                        "file_path": driver_file,
                        "artifact_type": "driver_characteristics",
                        "artifact_key": f"{year}::driver_characteristics",
                        "data": data,
                        "version": data.get("version", 1),
                        "checksum": compute_checksum(data),
                    }
                )
                print(f"  Found: {driver_file.relative_to(data_root)}")
            except Exception as e:
                print(f"  [WARN]  Failed to load {driver_file}: {e}")

    # 4. Driver debut years (CSV -> JSON artifact)
    debut_file = data_root / "driver_debuts.csv"
    if debut_file.exists():
        try:
            debuts = load_driver_debuts_from_csv(debut_file)
            data = {
                "driver_debuts": debuts,
                "source_file": "driver_debuts.csv",
                "total_drivers": len(debuts),
            }
            artifacts.append(
                {
                    "file_path": debut_file,
                    "artifact_type": "driver_debuts",
                    "artifact_key": "driver_debuts",
                    "data": data,
                    "version": 1,
                    "checksum": compute_checksum(data),
                }
            )
            print(f"  Found: {debut_file.relative_to(data_root)}")
        except Exception as e:
            print(f"  [WARN]  Failed to load {debut_file}: {e}")

    # 5. Track characteristics (multiple years)
    track_chars_dir = data_root / "processed" / "track_characteristics"
    if track_chars_dir.exists():
        for file in sorted(track_chars_dir.glob("*_track_characteristics.json")):
            try:
                data = _load_json_file(file)
                year = file.stem.split("_")[0]
                artifacts.append(
                    {
                        "file_path": file,
                        "artifact_type": "track_characteristics",
                        "artifact_key": f"{year}::track_characteristics",
                        "data": data,
                        "version": data.get("version", 1),
                        "checksum": compute_checksum(data),
                    }
                )
                print(f"  Found: {file.relative_to(data_root)}")
            except Exception as e:
                print(f"  [WARN]  Failed to load {file}: {e}")

    # 6. Learning state artifact (legacy compatibility)
    learning_file = data_root / "learning_state.json"
    if learning_file.exists():
        try:
            data = _load_json_file(learning_file)
            year = _season_year_from_payload(data)
            artifacts.append(
                {
                    "file_path": learning_file,
                    "artifact_type": "learning_state",
                    "artifact_key": f"{year}::learning_state",
                    "data": data,
                    "version": 1,
                    "checksum": compute_checksum(data),
                }
            )
            print(f"  Found: {learning_file.relative_to(data_root)}")
        except Exception as e:
            print(f"  [WARN]  Failed to load {learning_file}: {e}")

    # 7. Predictions (scan all years/races)
    predictions_dir = data_root / "predictions"
    if predictions_dir.exists():
        for pred_file in sorted(predictions_dir.rglob("*.json")):
            # Parse path: predictions/2026/bahrain_grand_prix/bahrain_grand_prix_qualifying.json
            parts = pred_file.relative_to(predictions_dir).parts
            if len(parts) >= 3:
                year = parts[0]
                race_dir = parts[1]
                session_file = parts[2]
                session_name = session_file.replace(f"{race_dir}_", "").replace(".json", "")

                # Reconstruct race name (approximate)
                race_name = race_dir.replace("_", " ").title()

                try:
                    with open(pred_file) as f:
                        data = json.load(f)

                    # Generate run_id from predicted_at timestamp (for uniqueness)
                    predicted_at = data.get("metadata", {}).get("predicted_at", "")
                    run_id = None
                    if predicted_at:
                        run_id = hashlib.sha256(
                            f"{year}::{race_name}::{session_name}::{predicted_at}".encode()
                        ).hexdigest()[:32]

                    artifacts.append(
                        {
                            "file_path": pred_file,
                            "artifact_type": "prediction",
                            "artifact_key": f"{year}::{race_name}::{session_name}",
                            "data": data,
                            "version": 1,
                            "run_id": run_id,
                            "checksum": compute_checksum(data),
                        }
                    )
                    print(f"  Found: {pred_file.relative_to(data_root)}")
                except Exception as e:
                    print(f"  [WARN]  Failed to load {pred_file}: {e}")

    return artifacts


def _extract_runtime_races_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract `{state_key: payload}` records from a `{races: {...}}` file."""
    races = payload.get("races")
    if not isinstance(races, dict):
        return {}
    return {
        str(state_key): state
        for state_key, state in races.items()
        if str(state_key).strip() and isinstance(state, dict)
    }


def _extract_runtime_entries_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract `{state_key: payload}` records from a file-backed `entries` container."""
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        str(state_key): state
        for state_key, state in entries.items()
        if str(state_key).strip() and isinstance(state, dict)
    }


def discover_runtime_state_records(data_root: Path) -> list[dict[str, Any]]:
    """Discover local runtime-state files that should be backfilled to Supabase."""
    discovered: list[dict[str, Any]] = []

    runtime_sources = [
        (
            data_root / "systems" / "practice_characteristics_state.json",
            "practice_characteristics",
            _extract_runtime_races_records,
        ),
        (
            data_root / "systems" / "event_boundary_refresh_state.json",
            "event_boundary_refresh",
            _extract_runtime_races_records,
        ),
        (
            data_root / "systems" / "precomputed_predictions.json",
            "precomputed_predictions",
            _extract_runtime_entries_records,
        ),
        (
            data_root / "systems" / "precomputed_base_features.json",
            "precomputed_prediction_base_features",
            _extract_runtime_entries_records,
        ),
        (
            data_root / "systems" / "precompute_horizon_index.json",
            "prediction_precompute_horizon_index",
            _extract_runtime_entries_records,
        ),
        (
            data_root / "systems" / "session_automation_schedule.json",
            "session_automation_schedule",
            lambda payload: {
                str(state_key): state
                for state_key, state in payload.items()
                if str(state_key).strip() and isinstance(state, dict)
            },
        ),
        (
            data_root / "learning_state.json",
            "race_learning",
            lambda payload: {str(_season_year_from_payload(payload)): payload},
        ),
    ]

    for file_path, namespace, extractor in runtime_sources:
        if not file_path.exists():
            continue
        try:
            payload = _load_json_file(file_path)
            records = extractor(payload)
            if not records:
                print(
                    f"  [WARN]  No runtime-state records found in {file_path.relative_to(data_root)}"
                )
                continue
            discovered.append(
                {
                    "file_path": file_path,
                    "namespace": namespace,
                    "records": records,
                    "record_count": len(records),
                    "checksum": compute_checksum(records),
                }
            )
            print(
                f"  Found runtime state: {file_path.relative_to(data_root)} "
                f"({namespace}, {len(records)} record(s))"
            )
        except Exception as e:
            print(f"  [WARN]  Failed to load runtime state {file_path}: {e}")

    return discovered


def backfill_artifacts(
    artifacts: list[dict[str, Any]],
    dry_run: bool = False,
    batch_size: int = 100,
) -> tuple[int, int]:
    """
    Backfill artifacts to Supabase.

    Args:
        artifacts: List of artifact metadata
        dry_run: If True, skip actual writes
        batch_size: Reserved for parity with runtime-state backfill

    Returns:
        Tuple of (success_count, failure_count)
    """
    store = ArtifactStore()
    success = 0
    failure = 0

    for i, artifact in enumerate(artifacts, 1):
        artifact_id = f"{artifact['artifact_type']}::{artifact['artifact_key']}"
        print(f"\n[{i}/{len(artifacts)}] Processing: {artifact_id} (v{artifact['version']})")

        if dry_run:
            print(f"  [DRY RUN] Would save: {artifact['file_path'].name}")
            print(f"  Checksum: {artifact['checksum']}")
            success += 1
            continue

        try:
            result = store.save_artifact(
                artifact_type=artifact["artifact_type"],
                artifact_key=artifact["artifact_key"],
                data=artifact["data"],
                version=artifact["version"],
                run_id=artifact.get("run_id"),
            )

            # Validate checksum
            saved_checksum = compute_checksum(result.get("data", artifact["data"]))
            if saved_checksum != artifact["checksum"]:
                print(
                    f"  [WARN]  Checksum mismatch! Expected {artifact['checksum']}, got {saved_checksum}"
                )
            else:
                print("  [OK] Saved successfully (checksum verified)")

            success += 1

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            failure += 1

    return success, failure


def backfill_runtime_state(
    runtime_state_payloads: list[dict[str, Any]],
    dry_run: bool = False,
    batch_size: int = 100,
) -> tuple[int, int]:
    """Backfill runtime-state records to Supabase namespaces."""
    store = RuntimeStateStore()
    success = 0
    failure = 0
    normalized_batch_size = max(1, int(batch_size))

    for i, payload in enumerate(runtime_state_payloads, 1):
        namespace = str(payload["namespace"])
        records = payload.get("records", {})
        if not isinstance(records, dict):
            continue

        print(
            f"\n[{i}/{len(runtime_state_payloads)}] Processing runtime state: "
            f"{namespace} ({len(records)} record(s))"
        )

        if dry_run:
            print(f"  [DRY RUN] Would save: {payload['file_path'].name}")
            print(f"  Checksum: {payload['checksum']}")
            success += len(records)
            continue

        try:
            items = list(records.items())
            for start in range(0, len(items), normalized_batch_size):
                batch = {
                    str(state_key): state
                    for state_key, state in items[start : start + normalized_batch_size]
                    if isinstance(state, dict)
                }
                if batch:
                    store.upsert_many(namespace, batch)
            print("  [OK] Saved successfully")
            success += len(records)
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            failure += len(records)

    return success, failure


def main():
    parser = argparse.ArgumentParser(description="Backfill JSON artifacts to Supabase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually write to DB, just simulate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of runtime-state rows per batch (default: 100)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Backfill Script: Migrate JSON Files to Supabase")
    print("=" * 70)

    # Check storage mode
    if not args.dry_run and USE_DB_STORAGE == "file_only":
        print("\n[ERROR] ERROR: USE_DB_STORAGE is set to 'file_only'")
        print("   Set USE_DB_STORAGE=db_only or dual_write to enable DB writes")
        print("\n   Example:")
        print("   export USE_DB_STORAGE=db_only")
        print("   python scripts/backfill_to_db.py")
        return 1

    if args.dry_run:
        print("\n DRY RUN MODE - No writes will be performed")
    else:
        print(f"\n Storage mode: {USE_DB_STORAGE}")

    print(f" Data root: {args.data_root.absolute()}")
    print(f" Batch size: {args.batch_size}")

    # Discover artifacts
    print(f"\n1. Discovering artifacts in {args.data_root}/...")
    artifacts = discover_artifacts(args.data_root)
    runtime_state_payloads = discover_runtime_state_records(args.data_root)

    if not artifacts and not runtime_state_payloads:
        print("\n[WARN]  No artifacts or runtime-state payloads found!")
        return 0

    print(f"\n[OK] Found {len(artifacts)} artifact(s)")
    print("\n   Artifact breakdown:")
    type_counts = {}
    for a in artifacts:
        type_counts[a["artifact_type"]] = type_counts.get(a["artifact_type"], 0) + 1
    for artifact_type, count in sorted(type_counts.items()):
        print(f"   - {artifact_type}: {count}")

    runtime_record_count = sum(
        int(payload.get("record_count", 0)) for payload in runtime_state_payloads
    )
    print(
        f"\n[OK] Found {runtime_record_count} runtime-state record(s) "
        f"across {len(runtime_state_payloads)} file(s)"
    )
    print("\n   Runtime-state breakdown:")
    for payload in runtime_state_payloads:
        print(f"   - {payload['namespace']}: {payload['record_count']}")

    # Confirm before proceeding
    if not args.dry_run:
        print("\n" + "=" * 70)
        response = input("Proceed with backfill? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0

    # Backfill
    print("\n2. Backfilling artifacts...")
    artifact_success, artifact_failure = backfill_artifacts(
        artifacts, args.dry_run, args.batch_size
    )

    print("\n3. Backfilling runtime state...")
    runtime_success, runtime_failure = backfill_runtime_state(
        runtime_state_payloads,
        args.dry_run,
        args.batch_size,
    )
    success = artifact_success + runtime_success
    failure = artifact_failure + runtime_failure

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    total_units = len(artifacts) + runtime_record_count
    print(f"[OK] Success: {success}/{total_units}")
    print(f"[ERROR] Failure: {failure}/{total_units}")
    print(f"   Artifacts: {artifact_success} ok / {artifact_failure} failed")
    print(f"   Runtime state rows: {runtime_success} ok / {runtime_failure} failed")

    if args.dry_run:
        print("\n DRY RUN completed. No changes were made.")
        print("   Run without --dry-run to perform actual migration.")
    else:
        print("\n[OK] Backfill completed!")
        print("\nNext steps:")
        print("1. Verify data in Supabase Dashboard → Table Editor → artifacts/runtime_state")
        print("2. Test app with USE_DB_STORAGE=fallback")
        print("3. Monitor for 1 week, then switch to db_only")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
