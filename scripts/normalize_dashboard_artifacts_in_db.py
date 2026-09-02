#!/usr/bin/env python3
"""Normalize and deduplicate dashboard artifact rows stored in Supabase.

This script repairs historical `prediction` and `accuracy_snapshot` rows when
their artifact keys drifted because of inconsistent whitespace or casing. It
can also delete duplicate rows that collapse to the same canonical identity.

Typical usage:

    uv run python scripts/normalize_dashboard_artifacts_in_db.py \
      --env-file .env.local

    uv run python scripts/normalize_dashboard_artifacts_in_db.py \
      --env-file .env.local \
      --apply
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.env_file import load_env_file as _load_env_file  # noqa: E402

_SUPPORTED_ARTIFACT_TYPES: tuple[str, ...] = ("prediction", "accuracy_snapshot")
_CANONICAL_SINGLETON_VERSION = 1


@dataclass(frozen=True)
class ArtifactRow:
    """Store the subset of artifact-row fields needed for cleanup."""

    id: str
    artifact_type: str
    artifact_key: str
    version: int
    run_id: str | None
    data: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class NormalizedArtifactRow:
    """Attach the canonical key/data representation to one raw artifact row."""

    row: ArtifactRow
    normalized_key: str
    normalized_data: dict[str, Any]
    completeness_score: int


@dataclass(frozen=True)
class CleanupAction:
    """Describe the winner row and any cleanup it needs."""

    artifact_type: str
    normalized_key: str
    canonical_version: int
    winner_id: str
    update_winner: bool
    delete_ids: tuple[str, ...]
    duplicate_count: int


@dataclass(frozen=True)
class CleanupPlan:
    """Summarize all remote cleanup actions for the selected artifact types."""

    scanned_rows: int
    actions: tuple[CleanupAction, ...]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and deduplicate Supabase dashboard artifacts "
            "(prediction + accuracy_snapshot)."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional env file to load before connecting to Supabase, for example .env.local.",
    )
    parser.add_argument(
        "--artifact-type",
        dest="artifact_types",
        action="append",
        choices=list(_SUPPORTED_ARTIFACT_TYPES),
        help="Optional artifact type filter. Repeat to process multiple types.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Supabase page size for listing artifact rows (default: 500).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply remote updates and deletes. Default mode is dry-run.",
    )
    return parser.parse_args()


def _configure_db_environment(env_file: Path | None) -> None:
    """Load optional credentials and force DB-backed storage for this run."""
    if env_file is not None:
        _load_env_file(env_file)
    os.environ["USE_DB_STORAGE"] = "db_only"


def _get_supabase_client():
    """Import and return the shared Supabase client lazily."""
    from src.persistence.db import get_supabase_client

    return get_supabase_client()


def _prediction_logger_class():
    """Import and return the prediction logger class lazily."""
    from src.utils.prediction_logger import PredictionLogger

    return PredictionLogger


def _accuracy_snapshot_key(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    target_key: str,
) -> str:
    """Import and build the canonical accuracy-snapshot artifact key lazily."""
    from src.utils.accuracy_snapshots import accuracy_snapshot_artifact_key

    return accuracy_snapshot_artifact_key(
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session,
        target_key=target_key,
    )


def _artifact_row_from_remote(raw_row: dict[str, Any]) -> ArtifactRow:
    """Convert one Supabase row payload into the typed cleanup representation."""
    payload = raw_row.get("data")
    if not isinstance(payload, dict):
        payload = {}
    return ArtifactRow(
        id=str(raw_row.get("id", "")),
        artifact_type=str(raw_row.get("artifact_type", "")).strip(),
        artifact_key=str(raw_row.get("artifact_key", "")),
        version=int(raw_row.get("version", 1) or 1),
        run_id=(
            str(raw_row.get("run_id")).strip() if raw_row.get("run_id") not in (None, "") else None
        ),
        data=payload,
        created_at=str(raw_row.get("created_at", "")),
        updated_at=str(raw_row.get("updated_at", "")),
    )


def _list_remote_artifacts(artifact_type: str, page_size: int) -> list[ArtifactRow]:
    """Load every remote artifact row for one artifact type."""
    client = _get_supabase_client()
    offset = 0
    rows: list[ArtifactRow] = []

    while True:
        result = (
            client.table("artifacts")
            .select(
                "id, artifact_type, artifact_key, version, run_id, data, created_at, updated_at"
            )
            .eq("artifact_type", artifact_type)
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        rows.extend(_artifact_row_from_remote(raw_row) for raw_row in batch)
        if len(batch) < page_size:
            break
        offset += page_size

    return rows


def _object_copy(value: Any) -> dict[str, Any]:
    """Return a shallow dict copy or an empty object for non-dict inputs."""
    return dict(value) if isinstance(value, dict) else {}


def _prediction_artifact_completeness(payload: dict[str, Any]) -> int:
    """Score how much real outcome truth a prediction payload contains."""
    actuals = payload.get("actuals")
    if not isinstance(actuals, dict):
        return 0

    score = 0
    if isinstance(actuals.get("qualifying"), list):
        score += 1
    if isinstance(actuals.get("race"), list):
        score += 1

    target_actuals = actuals.get("targets")
    if isinstance(target_actuals, dict):
        score += sum(
            1
            for target_value in target_actuals.values()
            if target_value is not None and target_value != "null"
        )

    return score


def _normalize_prediction_row(row: ArtifactRow) -> NormalizedArtifactRow:
    """Return the canonical key and metadata payload for one prediction row."""
    parts = row.artifact_key.split("::")
    if len(parts) != 3:
        return NormalizedArtifactRow(
            row=row,
            normalized_key=row.artifact_key,
            normalized_data=_object_copy(row.data),
            completeness_score=_prediction_artifact_completeness(_object_copy(row.data)),
        )

    normalized_key = _prediction_logger_class()._artifact_key_for_prediction(*parts)
    normalized_year, normalized_race_name, normalized_session_name = normalized_key.split("::", 2)

    payload = _object_copy(row.data)
    metadata = _object_copy(payload.get("metadata"))
    metadata["year"] = int(normalized_year)
    metadata["race_name"] = normalized_race_name
    metadata["session_name"] = normalized_session_name
    payload["metadata"] = metadata

    return NormalizedArtifactRow(
        row=row,
        normalized_key=normalized_key,
        normalized_data=payload,
        completeness_score=_prediction_artifact_completeness(payload),
    )


def _normalize_accuracy_snapshot_row(row: ArtifactRow) -> NormalizedArtifactRow:
    """Return the canonical key and metadata payload for one accuracy-snapshot row."""
    parts = row.artifact_key.split("::")
    if len(parts) != 4:
        return NormalizedArtifactRow(
            row=row,
            normalized_key=row.artifact_key,
            normalized_data=_object_copy(row.data),
            completeness_score=0,
        )

    normalized_key = _accuracy_snapshot_key(
        year=int(parts[0]),
        race_name=parts[1],
        checkpoint_session=parts[2],
        target_key=parts[3],
    )
    normalized_year, normalized_race_name, normalized_checkpoint, normalized_target = (
        normalized_key.split("::", 3)
    )

    payload = _object_copy(row.data)
    metadata = _object_copy(payload.get("metadata"))
    metadata["year"] = int(normalized_year)
    metadata["race_name"] = normalized_race_name
    metadata["checkpoint_session"] = normalized_checkpoint
    metadata["target_key"] = normalized_target
    payload["metadata"] = metadata

    return NormalizedArtifactRow(
        row=row,
        normalized_key=normalized_key,
        normalized_data=payload,
        completeness_score=0,
    )


def normalize_artifact_row(row: ArtifactRow) -> NormalizedArtifactRow:
    """Normalize one dashboard artifact row based on its artifact type."""
    if row.artifact_type == "prediction":
        return _normalize_prediction_row(row)
    if row.artifact_type == "accuracy_snapshot":
        return _normalize_accuracy_snapshot_row(row)
    raise ValueError(f"Unsupported artifact type: {row.artifact_type}")


def _winner_sort_key(row: NormalizedArtifactRow) -> tuple[Any, ...]:
    """Return the ranking tuple used to keep the best duplicate row."""
    base_key = (
        1 if row.row.run_id else 0,
        row.row.updated_at,
        row.row.created_at,
        row.row.id,
    )
    if row.row.artifact_type == "prediction":
        return (row.completeness_score, *base_key)
    return base_key


def plan_cleanup(rows: list[ArtifactRow]) -> CleanupPlan:
    """Build the update/delete plan needed to canonicalize the given rows."""
    grouped_rows: dict[tuple[str, str, int], list[NormalizedArtifactRow]] = {}
    for row in rows:
        normalized = normalize_artifact_row(row)
        grouping_key = (row.artifact_type, normalized.normalized_key)
        grouped_rows.setdefault(grouping_key, []).append(normalized)

    actions: list[CleanupAction] = []
    for (artifact_type, normalized_key), group_rows in sorted(grouped_rows.items()):
        ranked_rows = sorted(group_rows, key=_winner_sort_key, reverse=True)
        winner = ranked_rows[0]
        delete_ids = tuple(candidate.row.id for candidate in ranked_rows[1:])
        update_winner = (
            winner.row.artifact_key != winner.normalized_key
            or winner.row.data != winner.normalized_data
            or winner.row.version != _CANONICAL_SINGLETON_VERSION
        )
        if update_winner or delete_ids:
            actions.append(
                CleanupAction(
                    artifact_type=artifact_type,
                    normalized_key=normalized_key,
                    canonical_version=_CANONICAL_SINGLETON_VERSION,
                    winner_id=winner.row.id,
                    update_winner=update_winner,
                    delete_ids=delete_ids,
                    duplicate_count=max(len(group_rows) - 1, 0),
                )
            )

    return CleanupPlan(scanned_rows=len(rows), actions=tuple(actions))


def _apply_cleanup(rows: list[ArtifactRow], plan: CleanupPlan) -> tuple[int, int]:
    """Apply the cleanup plan to Supabase and return update/delete counts."""
    client = _get_supabase_client()
    rows_by_id = {row.id: row for row in rows}
    normalized_by_id = {row.id: normalize_artifact_row(row) for row in rows}
    updated_rows = 0
    deleted_rows = 0

    for action in plan.actions:
        if action.delete_ids:
            client.table("artifacts").delete().in_("id", list(action.delete_ids)).execute()
            deleted_rows += len(action.delete_ids)

        if action.update_winner:
            normalized_winner = normalized_by_id[action.winner_id]
            client.table("artifacts").update(
                {
                    "artifact_key": normalized_winner.normalized_key,
                    "version": _CANONICAL_SINGLETON_VERSION,
                    "data": normalized_winner.normalized_data,
                }
            ).eq("id", action.winner_id).execute()
            updated_rows += 1

        if action.winner_id not in rows_by_id:
            raise RuntimeError(f"Winner row disappeared from plan: {action.winner_id}")

    return updated_rows, deleted_rows


def _print_plan(plan: CleanupPlan) -> None:
    """Print a compact human-readable summary of the cleanup plan."""
    update_count = sum(1 for action in plan.actions if action.update_winner)
    delete_count = sum(len(action.delete_ids) for action in plan.actions)
    duplicate_groups = sum(1 for action in plan.actions if action.delete_ids)

    print(f"Rows scanned: {plan.scanned_rows}")
    print(f"Artifact groups needing cleanup: {len(plan.actions)}")
    print(f"Winner rows to update: {update_count}")
    print(f"Duplicate groups to collapse: {duplicate_groups}")
    print(f"Duplicate rows to delete: {delete_count}")

    preview_limit = 10
    if not plan.actions:
        return

    print("\nSample cleanup actions:")
    for action in plan.actions[:preview_limit]:
        print(
            "  "
            f"{action.artifact_type} v{action.canonical_version} -> {action.normalized_key} "
            f"(update={action.update_winner}, delete={len(action.delete_ids)})"
        )
    if len(plan.actions) > preview_limit:
        print(f"  ... {len(plan.actions) - preview_limit} more action(s)")


def main() -> int:
    """Run the remote dashboard-artifact normalization flow."""
    args = _parse_args()
    _configure_db_environment(args.env_file)

    artifact_types = tuple(args.artifact_types or _SUPPORTED_ARTIFACT_TYPES)
    print("Supabase Dashboard Artifact Cleanup")
    print(f"Artifact types: {', '.join(artifact_types)}")
    print(f"Mode: {'apply' if args.apply else 'dry-run'}")

    all_rows: list[ArtifactRow] = []
    for artifact_type in artifact_types:
        rows = _list_remote_artifacts(artifact_type=artifact_type, page_size=args.page_size)
        print(f"Loaded {len(rows)} remote {artifact_type} row(s)")
        all_rows.extend(rows)

    plan = plan_cleanup(all_rows)
    print()
    _print_plan(plan)

    if not args.apply:
        return 0

    updated_rows, deleted_rows = _apply_cleanup(all_rows, plan)
    print("\nCleanup applied.")
    print(f"Updated rows: {updated_rows}")
    print(f"Deleted rows: {deleted_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
