#!/usr/bin/env python3
"""
Backfill script: Migrate existing JSON files to Supabase.

This script reads all JSON files from the data/ directory and inserts them
into the Supabase artifacts table, preserving versions and timestamps.

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


def compute_checksum(data: dict) -> str:
    """Compute SHA256 checksum of JSON data."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


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
    artifacts = []

    # 1. Car characteristics (multiple years)
    car_chars_dir = data_root / "processed" / "car_characteristics"
    if car_chars_dir.exists():
        for file in car_chars_dir.glob("*_car_characteristics.json"):
            year = file.stem.split("_")[0]
            try:
                with open(file) as f:
                    data = json.load(f)
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

    # 2. Driver characteristics
    driver_file = data_root / "processed" / "driver_characteristics.json"
    if driver_file.exists():
        try:
            with open(driver_file) as f:
                data = json.load(f)
            artifacts.append(
                {
                    "file_path": driver_file,
                    "artifact_type": "driver_characteristics",
                    "artifact_key": "2026::driver_characteristics",
                    "data": data,
                    "version": data.get("version", 1),
                    "checksum": compute_checksum(data),
                }
            )
            print(f"  Found: {driver_file.relative_to(data_root)}")
        except Exception as e:
            print(f"  [WARN]  Failed to load {driver_file}: {e}")

    # 3. Driver debut years (CSV -> JSON artifact)
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

    # 4. Track characteristics (multiple years)
    track_chars_dir = data_root / "processed" / "track_characteristics"
    if track_chars_dir.exists():
        for file in track_chars_dir.glob("*_track_characteristics.json"):
            year = file.stem.split("_")[0]
            try:
                with open(file) as f:
                    data = json.load(f)
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

    # 5. Learning state
    learning_file = data_root / "learning_state.json"
    if learning_file.exists():
        try:
            with open(learning_file) as f:
                data = json.load(f)
            artifacts.append(
                {
                    "file_path": learning_file,
                    "artifact_type": "learning_state",
                    "artifact_key": "2026::learning_state",
                    "data": data,
                    "version": 1,
                    "checksum": compute_checksum(data),
                }
            )
            print(f"  Found: {learning_file.relative_to(data_root)}")
        except Exception as e:
            print(f"  [WARN]  Failed to load {learning_file}: {e}")

    # 6. Practice state
    practice_file = data_root / "systems" / "practice_characteristics_state.json"
    if practice_file.exists():
        try:
            with open(practice_file) as f:
                data = json.load(f)
            artifacts.append(
                {
                    "file_path": practice_file,
                    "artifact_type": "practice_state",
                    "artifact_key": "2026::practice_state",
                    "data": data,
                    "version": 1,
                    "checksum": compute_checksum(data),
                }
            )
            print(f"  Found: {practice_file.relative_to(data_root)}")
        except Exception as e:
            print(f"  [WARN]  Failed to load {practice_file}: {e}")

    # 7. Predictions (scan all years/races)
    predictions_dir = data_root / "predictions"
    if predictions_dir.exists():
        for pred_file in predictions_dir.rglob("*.json"):
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


def backfill_artifacts(
    artifacts: list[dict],
    dry_run: bool = False,
    batch_size: int = 100,
) -> tuple[int, int]:
    """
    Backfill artifacts to Supabase.

    Args:
        artifacts: List of artifact metadata
        dry_run: If True, skip actual writes
        batch_size: Number of artifacts per batch

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
        help="Number of artifacts per batch (default: 100)",
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

    if not artifacts:
        print("\n[WARN]  No artifacts found!")
        return 0

    print(f"\n[OK] Found {len(artifacts)} artifact(s)")
    print("\n   Breakdown:")
    type_counts = {}
    for a in artifacts:
        type_counts[a["artifact_type"]] = type_counts.get(a["artifact_type"], 0) + 1
    for artifact_type, count in sorted(type_counts.items()):
        print(f"   - {artifact_type}: {count}")

    # Confirm before proceeding
    if not args.dry_run:
        print("\n" + "=" * 70)
        response = input("Proceed with backfill? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0

    # Backfill
    print("\n2. Backfilling artifacts...")
    success, failure = backfill_artifacts(artifacts, args.dry_run, args.batch_size)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[OK] Success: {success}/{len(artifacts)}")
    print(f"[ERROR] Failure: {failure}/{len(artifacts)}")

    if args.dry_run:
        print("\n DRY RUN completed. No changes were made.")
        print("   Run without --dry-run to perform actual migration.")
    else:
        print("\n[OK] Backfill completed!")
        print("\nNext steps:")
        print("1. Verify data in Supabase Dashboard → Table Editor → artifacts")
        print("2. Test app with USE_DB_STORAGE=fallback")
        print("3. Monitor for 1 week, then switch to db_only")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
