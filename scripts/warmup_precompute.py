"""Run checkpoint-aware prediction warmup outside the Streamlit request path."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.warmup import run_warmup_precompute_cycle  # noqa: E402
from src.persistence.config import (  # noqa: E402
    get_storage_mode,
    should_read_db_first,
    should_write_to_db,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _is_db_backed_mode() -> bool:
    """Return True when current storage mode reads and writes Supabase state."""
    return bool(should_write_to_db() and should_read_db_first())


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for scheduled warmup runs."""
    parser = argparse.ArgumentParser(
        description=(
            "Precompute checkpoint-aware horizon predictions and persist them for instant dashboard load."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(UTC).year,
        help="Season year to warm (default: current UTC year).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan warmup actions without writing base features, predictions, or horizon index.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print target-level checkpoint and boundary details.",
    )
    parser.add_argument(
        "--no-verify-writes",
        action="store_true",
        help="Skip DB read-after-write verification checks.",
    )
    parser.add_argument(
        "--require-db",
        action="store_true",
        help=(
            "Fail fast when storage mode is not DB-backed or when DB read-after-write verification "
            "reports warnings."
        ),
    )
    return parser


def _log_verbose_summary(summary: dict) -> None:
    """Print verbose warmup details for operational debugging."""
    anchor_race_name = str(summary.get("anchor_race_name", "")).strip()
    anchor_checkpoint = str(summary.get("checkpoint", "")).strip()
    anchor_boundary = ""
    target_contexts = summary.get("target_contexts", [])
    if isinstance(target_contexts, list):
        for context in target_contexts:
            if not isinstance(context, dict):
                continue
            if str(context.get("race_name", "")).strip() == anchor_race_name:
                anchor_boundary = str(context.get("boundary_signature", "")).strip()
            logger.info(
                "Target context: race=%s checkpoint=%s boundary=%s sprint=%s",
                context.get("race_name", ""),
                context.get("checkpoint", ""),
                context.get("boundary_signature", ""),
                bool(context.get("is_sprint", False)),
            )

    logger.info(
        "Warmup anchor: race=%s checkpoint=%s boundary=%s reason=%s",
        anchor_race_name,
        anchor_checkpoint,
        anchor_boundary,
        summary.get("reason", ""),
    )
    logger.info(
        "Warmup counts: base_reused=%s base_computed=%s predictions_reused=%s predictions_computed=%s",
        summary.get("base_reused", 0),
        summary.get("base_generated", 0),
        summary.get("predictions_reused", 0),
        summary.get("predictions_generated", 0),
    )

    verification_warnings = summary.get("db_verification_warnings", [])
    if isinstance(verification_warnings, list):
        for warning in verification_warnings:
            logger.warning("DB verification: %s", warning)


def main() -> int:
    """Program entrypoint for warmup precompute scheduler runs."""
    args = _build_parser().parse_args()
    storage_mode = get_storage_mode()
    db_backed_mode = _is_db_backed_mode()
    if args.require_db and args.no_verify_writes:
        logger.error("--require-db cannot be combined with --no-verify-writes.")
        return 2
    if args.require_db and not db_backed_mode:
        logger.error(
            "Warmup requires DB-backed storage, but USE_DB_STORAGE=%s is not DB-backed.",
            storage_mode,
        )
        return 2
    if not db_backed_mode:
        logger.warning(
            "Warmup is running with USE_DB_STORAGE=%s. For Supabase multi-instance deployments, "
            "use a DB-backed mode (fallback, dual_write, or db_only).",
            storage_mode,
        )
    try:
        summary = run_warmup_precompute_cycle(
            year=int(args.year),
            dry_run=bool(args.dry_run),
            verify_db_writes=not bool(args.no_verify_writes),
        )
        summary_payload = summary.to_dict()
        logger.info("Warmup summary: %s", summary_payload)
        if args.verbose:
            _log_verbose_summary(summary_payload)
        if args.require_db:
            verification_warnings = summary_payload.get("db_verification_warnings", [])
            if isinstance(verification_warnings, list) and verification_warnings:
                logger.error(
                    "Warmup DB verification warnings detected in --require-db mode: %s",
                    verification_warnings,
                )
                return 2
        return 0
    except Exception as exc:
        logger.exception("Warmup precompute failed unexpectedly: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
