#!/usr/bin/env python3
"""Freeze a champion-default challenger manifest into the research sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.challenger_governance import (  # noqa: E402
    DEFAULT_CONFIG_PATHS,
    DEFAULT_REPLAY_SEEDS,
    build_challenger_manifest,
)
from src.persistence.research_sidecar import (  # noqa: E402
    DEFAULT_RESEARCH_SIDECAR_ROOT,
    ResearchSidecarStore,
)


def _simulation_count(value: str) -> tuple[str, int]:
    """Parse one TARGET=COUNT command-line value."""
    target, separator, raw_count = value.partition("=")
    if not separator or not target.strip():
        raise argparse.ArgumentTypeError("simulation counts must use TARGET=COUNT")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("simulation count must be an integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("simulation count must be positive")
    return target.strip(), count


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--variant-id", required=True, help="Registered non-champion variant.")
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--input-snapshot-id", action="append", default=[])
    parser.add_argument("--cutoff", required=True, help="Timezone-aware ISO-8601 cutoff.")
    parser.add_argument(
        "--simulation-count",
        action="append",
        type=_simulation_count,
        required=True,
        metavar="TARGET=COUNT",
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional effective candidate-overlay config to hash. "
            "default.yaml and production_config.json are always included."
        ),
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=DEFAULT_RESEARCH_SIDECAR_ROOT,
    )
    return parser


def main() -> int:
    """Build and immutably persist one manifest."""
    args = build_parser().parse_args()
    simulation_counts = dict(args.simulation_count)
    if len(simulation_counts) != len(args.simulation_count):
        raise ValueError("each simulation target may be provided only once")
    manifest = build_challenger_manifest(
        repo_root=args.repo_root,
        candidate_id=args.candidate_id,
        variant_id=args.variant_id,
        feature_schema=args.feature_schema,
        input_snapshot_ids=args.input_snapshot_id,
        cutoff_at=args.cutoff,
        simulation_counts=simulation_counts,
        seeds=DEFAULT_REPLAY_SEEDS,
        config_paths=[*DEFAULT_CONFIG_PATHS, *args.config_path],
    )
    store = ResearchSidecarStore(args.sidecar_root, repo_root=args.repo_root)
    output_path = store.write_manifest(manifest)
    print(
        json.dumps(
            {
                "candidate_id": manifest["candidate_id"],
                "default_variant": manifest["default_variant"],
                "manifest_sha256": manifest["manifest_sha256"],
                "path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
