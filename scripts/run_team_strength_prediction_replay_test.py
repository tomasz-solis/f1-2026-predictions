#!/usr/bin/env python3
"""Run full prediction replay for a race-only team-strength scale candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.team_strength_prediction_replay_test import (  # noqa: E402
    SCALE_ONLY_CANDIDATE,
    TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE,
    build_race_scale_only_mapping_payload,
    compare_prediction_replay_summaries,
    format_team_strength_prediction_replay_test_markdown,
    load_construct_observations,
    team_strength_prediction_replay_artifact_key,
)
from src.models.team_strength_mapping import (  # noqa: E402
    TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV,
    load_live_team_strength_mappings,
)
from src.persistence.artifact_store import ArtifactStore  # noqa: E402
from src.utils.accuracy_targets import TARGET_SPRINT_QUALIFYING  # noqa: E402
from src.utils.historical_replay import run_historical_checkpoint_replay  # noqa: E402
from src.utils.json_io import read_json_object as _read_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the replay-candidate runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Season year to test.")
    parser.add_argument(
        "--source-processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed source directory used to seed each sidecar replay.",
    )
    parser.add_argument(
        "--current-summary",
        type=Path,
        default=Path("data/historical_replay/reports/summary.json"),
        help="Current frozen-mapping replay summary to compare against.",
    )
    parser.add_argument(
        "--mapping-artifact",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/latest.json"),
        help="Frozen team-strength mapping artifact.",
    )
    parser.add_argument(
        "--prior-artifact",
        type=Path,
        default=Path("data/processed/teammate_network_prior/latest.json"),
        help="Teammate-network prior artifact used for construct observations.",
    )
    parser.add_argument(
        "--raw-matched-laps",
        type=Path,
        default=Path("data/diagnostics/2026_team_strength_matched_laps/raw_matched_laps.csv"),
        help="Raw matched-lap rows used to fit held-out scale candidates.",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("data/diagnostics/2026_team_strength_prediction_replay"),
        help="Directory for sidecar candidate replay outputs.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data"),
        help="ArtifactStore data root for the comparison JSON.",
    )
    parser.add_argument(
        "--overwrite-folds",
        action="store_true",
        help="Rebuild completed fold replay roots instead of reusing their summaries.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path. Defaults beside the JSON artifact.",
    )
    return parser


def main() -> int:
    """Run held-out candidate replays, persist comparison metrics, and print Markdown."""
    args = build_parser().parse_args()
    year = int(args.year)
    frozen_payload = _read_json(args.mapping_artifact)
    mapping_policy = str(frozen_payload.get("policy", "same_session_construct"))
    frozen_race_mapping = load_live_team_strength_mappings(args.mapping_artifact)["race"]
    observations = load_construct_observations(
        raw_matched_laps_path=args.raw_matched_laps,
        prior_artifact_path=args.prior_artifact,
    )
    holdout_races = sorted(
        observations[observations["year"].eq(year)]["race_name"].dropna().unique()
    )
    candidate_summaries: dict[str, Path] = {}
    previous_mapping_override = os.environ.get(TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV)
    try:
        for holdout_race in holdout_races:
            race_label = _slugify(str(holdout_race))
            fold_root = args.replay_root / SCALE_ONLY_CANDIDATE / race_label
            summary_path = fold_root / "reports" / "summary.json"
            if summary_path.exists() and not args.overwrite_folds:
                candidate_summaries[str(holdout_race)] = summary_path
                continue
            mapping_path = fold_root / "candidate_mapping.json"
            mapping_payload = build_race_scale_only_mapping_payload(
                frozen_mapping_payload=frozen_payload,
                frozen_race_mapping=frozen_race_mapping,
                observations=observations,
                mapping_policy=mapping_policy,
                year=year,
                holdout_race=str(holdout_race),
            )
            _write_json(mapping_path, mapping_payload)
            os.environ[TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV] = str(mapping_path)
            summary = run_historical_checkpoint_replay(
                year=year,
                source_processed_dir=args.source_processed_dir,
                output_root=fold_root,
                overwrite=True,
                excluded_scoring_targets={TARGET_SPRINT_QUALIFYING},
                stop_after_race=str(holdout_race),
            )
            candidate_summaries[str(holdout_race)] = (
                Path(summary.output_root) / "reports" / "summary.json"
            )
    finally:
        if previous_mapping_override is None:
            os.environ.pop(TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV, None)
        else:
            os.environ[TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV] = previous_mapping_override

    artifact = compare_prediction_replay_summaries(
        year=year,
        current_summary_path=args.current_summary,
        candidate_summaries=candidate_summaries,
        candidate_name=SCALE_ONLY_CANDIDATE,
    )
    store = ArtifactStore(data_root=args.artifact_root)
    store.save_artifact(
        artifact_type=TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE,
        artifact_key=team_strength_prediction_replay_artifact_key(year),
        data=artifact,
    )

    markdown = format_team_strength_prediction_replay_test_markdown(artifact)
    output_md = args.output_md or (
        args.artifact_root
        / TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE
        / str(year)
        / "team_strength_prediction_replay_test.md"
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(json.dumps({"artifact_key": team_strength_prediction_replay_artifact_key(year)}))
    return 0


def _slugify(value: str) -> str:
    """Return a stable filesystem-safe race label."""
    return str(value).strip().lower().replace(" ", "_").replace("'", "")


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
