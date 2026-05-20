#!/usr/bin/env python3
"""Build and persist the team-strength construct-row audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.team_strength_construct_audit import (  # noqa: E402
    TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE,
    build_team_strength_construct_audit,
    format_team_strength_construct_audit_markdown,
    team_strength_construct_audit_artifact_key,
)
from src.persistence.artifact_store import ArtifactStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the construct-row audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Season year to audit.")
    parser.add_argument(
        "--mapping-artifact",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/latest.json"),
        help="Frozen team-strength seconds mapping artifact.",
    )
    parser.add_argument(
        "--candidate-diagnostics",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/candidate_diagnostics.json"),
        help="Candidate diagnostics artifact with historical reference folds.",
    )
    parser.add_argument(
        "--prior-artifact",
        type=Path,
        default=Path("data/processed/teammate_network_prior/latest.json"),
        help="Teammate-network prior artifact used for driver seconds.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="Optional construct-aligned observation CSV.",
    )
    parser.add_argument(
        "--raw-matched-laps",
        type=Path,
        default=Path("data/diagnostics/2026_team_strength_matched_laps/raw_matched_laps.csv"),
        help="Raw matched-lap rows used to rebuild construct-aligned observations.",
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
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=20,
        help="Maximum residual detail rows to keep in top-row tables.",
    )
    return parser


def main() -> int:
    """Build, save, and print the construct-row audit summary."""
    args = build_parser().parse_args()
    artifact = build_team_strength_construct_audit(
        year=int(args.year),
        mapping_artifact_path=args.mapping_artifact,
        candidate_diagnostics_path=args.candidate_diagnostics,
        prior_artifact_path=args.prior_artifact,
        observations_path=args.observations,
        raw_matched_laps_path=args.raw_matched_laps,
        max_detail_rows=int(args.max_detail_rows),
    )

    store = ArtifactStore(data_root=args.output_root)
    store.save_artifact(
        artifact_type=TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE,
        artifact_key=team_strength_construct_audit_artifact_key(int(args.year)),
        data=artifact,
    )

    markdown = format_team_strength_construct_audit_markdown(artifact)
    output_md = args.output_md or (
        args.output_root
        / TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE
        / str(int(args.year))
        / "team_strength_construct_row_audit.md"
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(json.dumps({"artifact_key": team_strength_construct_audit_artifact_key(int(args.year))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
