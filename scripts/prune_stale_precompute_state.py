#!/usr/bin/env python3
"""Delete stale dashboard precompute runtime-state rows from Supabase.

This script is intentionally narrow. It only touches the runtime-state namespaces
used by the prediction-page warmup cache:

- ``precomputed_predictions``
- ``precomputed_prediction_base_features``
- ``prediction_precompute_horizon_index``

It does not delete saved checkpoint prediction artifacts used by the accuracy
tracker. The goal is to keep the deployed dashboard aligned to the current
artifact hash without leaving older warmed hashes behind.

Typical usage:

    uv run python scripts/prune_stale_precompute_state.py \
      --year 2026 \
      --env-file .env.local \
      --require-db \
      --apply

To force a clean re-warm after a code-only change that does not move the artifact
hash (for example a bug fix in circuit/track-data routing), the stale-hash filter is
not enough - the old rows still carry the current hash. Use ``--all-hashes`` to purge
every precompute row for the year so the next warmup regenerates from scratch:

    uv run python scripts/prune_stale_precompute_state.py \
      --year 2026 \
      --env-file .env.local \
      --require-db \
      --all-hashes \
      --apply
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.env_file import load_env_file as _load_env_file  # noqa: E402

if TYPE_CHECKING:
    from src.persistence.runtime_state_store import RuntimeStateStore

_PRECOMPUTE_NAMESPACES = (
    "precomputed_predictions",
    "precomputed_prediction_base_features",
    "prediction_precompute_horizon_index",
)


@dataclass(frozen=True)
class StaleStateRecord:
    """Describe one stale runtime-state record slated for removal."""

    namespace: str
    state_key: str
    artifact_hash: str
    payload_year: int


def _runtime_state_store() -> RuntimeStateStore:
    """Return the runtime-state store class after project bootstrap is available."""
    from src.persistence.runtime_state_store import RuntimeStateStore

    return RuntimeStateStore()


def _get_storage_mode() -> str:
    """Return current persistence mode lazily."""
    from src.persistence.config import get_storage_mode

    return str(get_storage_mode())


def _should_read_db_first() -> bool:
    """Return whether the current mode reads runtime state from Supabase."""
    from src.persistence.config import should_read_db_first

    return bool(should_read_db_first())


def _should_write_to_db() -> bool:
    """Return whether the current mode writes runtime state to Supabase."""
    from src.persistence.config import should_write_to_db

    return bool(should_write_to_db())


def _artifact_hash_for_year(year: int) -> str:
    """Resolve the current artifact hash for one season year."""
    from src.dashboard.cache import get_artifact_versions
    from src.dashboard.precomputed_predictions import compute_artifact_hash

    return compute_artifact_hash(get_artifact_versions(year=int(year)))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prune stale dashboard precompute runtime-state rows from Supabase.",
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, for example 2026.")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional env file to load before connecting to Supabase, for example .env.local.",
    )
    parser.add_argument(
        "--require-db",
        action="store_true",
        help="Fail if the current storage mode is not DB-backed.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete stale rows. Without this flag, the script is a dry-run report.",
    )
    parser.add_argument(
        "--all-hashes",
        action="store_true",
        help=(
            "Purge ALL precompute rows for the year regardless of artifact hash. "
            "Use this to force a clean re-warm after a code-only change (one that "
            "does not move the artifact hash), where stale rows still carry the "
            "current hash and would otherwise be reused."
        ),
    )
    return parser.parse_args()


def _load_namespace_rows(
    *,
    store: RuntimeStateStore,
    namespace: str,
) -> dict[str, dict[str, Any]]:
    """Load one runtime-state namespace as a ``state_key -> payload`` mapping."""
    loaded = store.load_namespace(namespace)
    return loaded if isinstance(loaded, dict) else {}


def _stale_records_for_year(
    *,
    store: RuntimeStateStore,
    target_year: int,
    current_artifact_hash: str,
    all_hashes: bool = False,
) -> list[StaleStateRecord]:
    """Return precompute rows to delete for a season year.

    By default only rows whose artifact hash differs from ``current_artifact_hash``
    are returned. When ``all_hashes`` is set, every precompute row for the year is
    returned regardless of hash - the purge mode used to force a clean re-warm after
    a code-only change that does not move the artifact hash.
    """
    stale_records: list[StaleStateRecord] = []
    for namespace in _PRECOMPUTE_NAMESPACES:
        rows = _load_namespace_rows(store=store, namespace=namespace)
        for state_key, payload in rows.items():
            if not isinstance(payload, dict):
                continue
            try:
                payload_year = int(payload.get("year"))
            except (TypeError, ValueError):
                continue
            if payload_year != int(target_year):
                continue
            artifact_hash = str(payload.get("artifact_hash", "")).strip()
            if not all_hashes and (not artifact_hash or artifact_hash == current_artifact_hash):
                continue
            stale_records.append(
                StaleStateRecord(
                    namespace=namespace,
                    state_key=str(state_key).strip(),
                    artifact_hash=artifact_hash or "(unset)",
                    payload_year=payload_year,
                )
            )
    return stale_records


def _print_report(
    *,
    target_year: int,
    current_artifact_hash: str,
    stale_records: list[StaleStateRecord],
    apply: bool,
    all_hashes: bool = False,
) -> None:
    """Print a compact stale-row summary."""
    print()
    print("Purge ALL Precompute Runtime State" if all_hashes else "Stale Precompute Runtime State")
    print()
    print(f"Season year: {int(target_year)}")
    print(f"Current artifact hash: {current_artifact_hash}")
    print(f"Mode: {'purge all hashes' if all_hashes else 'stale hashes only'}")
    print(f"Apply deletes: {bool(apply)}")

    if not stale_records:
        print("\nNo matching precompute runtime-state rows found.")
        return

    by_namespace: dict[str, list[StaleStateRecord]] = defaultdict(list)
    for record in stale_records:
        by_namespace[record.namespace].append(record)

    print("\nDelete candidates:")
    for namespace in sorted(by_namespace):
        records = by_namespace[namespace]
        hash_counts: dict[str, int] = defaultdict(int)
        for record in records:
            hash_counts[record.artifact_hash] += 1
        print(f"  - {namespace}: {len(records)} row(s)")
        for artifact_hash, count in sorted(hash_counts.items()):
            print(f"    {artifact_hash}: {count}")


def main() -> int:
    """Run the stale precompute runtime-state pruning workflow."""
    args = _parse_args()
    if args.env_file is not None:
        _load_env_file(args.env_file.resolve())

    if args.require_db and not (_should_read_db_first() and _should_write_to_db()):
        print(
            f"DB-backed mode required, but current USE_DB_STORAGE resolves to {_get_storage_mode()}."
        )
        return 2

    store = _runtime_state_store()
    current_artifact_hash = _artifact_hash_for_year(int(args.year))
    stale_records = _stale_records_for_year(
        store=store,
        target_year=int(args.year),
        current_artifact_hash=current_artifact_hash,
        all_hashes=bool(args.all_hashes),
    )
    _print_report(
        target_year=int(args.year),
        current_artifact_hash=current_artifact_hash,
        stale_records=stale_records,
        apply=bool(args.apply),
        all_hashes=bool(args.all_hashes),
    )

    if not args.apply or not stale_records:
        return 0

    by_namespace: dict[str, list[str]] = defaultdict(list)
    for record in stale_records:
        by_namespace[record.namespace].append(record.state_key)

    for namespace, state_keys in by_namespace.items():
        store.delete_records(namespace, state_keys)

    print("\nDeleted stale rows successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
