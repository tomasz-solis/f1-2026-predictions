#!/usr/bin/env python3
"""Build and persist replay/leakage diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.replay_leakage_diagnostics import (  # noqa: E402
    REPLAY_LEAKAGE_ARTIFACT_TYPE,
    build_replay_leakage_diagnostics,
    format_replay_leakage_diagnostics_markdown,
    replay_leakage_artifact_key,
)
from src.persistence.artifact_store import ArtifactStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the diagnostics builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Season year to inspect.")
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("data/historical_replay"),
        help="Historical replay root containing reports and checkpoint predictions.",
    )
    parser.add_argument(
        "--mapping-artifact",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/latest.json"),
        help="Frozen Phase 7 team-strength seconds mapping artifact.",
    )
    parser.add_argument(
        "--candidate-diagnostics",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/candidate_diagnostics.json"),
        help="Phase 7 candidate diagnostics with historical residual means.",
    )
    parser.add_argument(
        "--prior-artifact",
        type=Path,
        default=Path("data/processed/teammate_network_prior/latest.json"),
        help="Phase 6 teammate-network prior artifact.",
    )
    parser.add_argument(
        "--regulation-reset-observations",
        type=Path,
        default=None,
        help="Optional 2026 construct-aligned observation CSV.",
    )
    parser.add_argument(
        "--regulation-reset-raw-matched-laps",
        type=Path,
        default=None,
        help="Optional 2026 raw matched-lap CSV used to build transfer observations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="ArtifactStore data root for JSON persistence.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path. Defaults beside the JSON artifact.",
    )
    return parser


def main() -> int:
    """Build, save, and print the replay/leakage diagnostics summary."""
    args = build_parser().parse_args()
    artifact = build_replay_leakage_diagnostics(
        year=int(args.year),
        replay_root=args.replay_root,
        mapping_artifact_path=args.mapping_artifact,
        candidate_diagnostics_path=args.candidate_diagnostics,
        prior_artifact_path=args.prior_artifact,
        regulation_reset_observations_path=args.regulation_reset_observations,
        regulation_reset_raw_matched_laps_path=args.regulation_reset_raw_matched_laps,
    )

    store = ArtifactStore(data_root=args.output_root)
    store.save_artifact(
        artifact_type=REPLAY_LEAKAGE_ARTIFACT_TYPE,
        artifact_key=replay_leakage_artifact_key(int(args.year)),
        data=artifact,
    )

    markdown = format_replay_leakage_diagnostics_markdown(artifact)
    output_md = args.output_md or (
        args.output_root
        / REPLAY_LEAKAGE_ARTIFACT_TYPE
        / str(int(args.year))
        / "replay_leakage_diagnostics.md"
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(json.dumps({"artifact_key": replay_leakage_artifact_key(int(args.year))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
