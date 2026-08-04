"""Freeze the live Phase 7 team-strength seconds mapping artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.models.team_strength_mapping import (  # noqa: E402
    evaluate_within_season_folds,
    fit_linear_team_strength_mapping,
    resolve_era_training_years,
)
from src.utils.config_schema import ModelConfig  # noqa: E402
from src.utils.model_version import get_model_version  # noqa: E402

DEFAULT_TRAINING_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_POLICY = "same_session_construct"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for freezing the live mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/calibration_observations.csv"),
        help="Construct-aligned calibration observations.",
    )
    parser.add_argument(
        "--candidate-diagnostics",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/candidate_diagnostics.json"),
        help="Candidate diagnostics used to document the selected policy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/latest.json"),
        help="Frozen live mapping JSON artifact.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping/latest.md"),
        help="Human-readable frozen mapping summary.",
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help="Team-strength proxy policy to freeze.",
    )
    parser.add_argument(
        "--training-years",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the seasons the mapping is fitted on. By default they are "
            "resolved from the regulation era covering --target-year, because the "
            "seconds spread of the field is regulation-dependent and a mapping fitted "
            "across a boundary mixes two different fields."
        ),
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="Season the mapping is being frozen for. Defaults to the latest in the data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Config file supplying model.regulation_eras.",
    )
    return parser


def _regulation_eras(config_path: Path) -> list[dict[str, Any]]:
    """Read the regulation-era table, falling back to the schema default."""
    if not config_path.exists():
        return [era.model_dump() for era in ModelConfig().regulation_eras]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    eras = (raw.get("model") or {}).get("regulation_eras")
    if not eras:
        return [era.model_dump() for era in ModelConfig().regulation_eras]
    return list(eras)


def main() -> None:
    """Fit the selected mappings and write the live artifact."""
    args = build_parser().parse_args()
    observations = pd.read_csv(args.observations)
    diagnostics = _load_candidate_diagnostics(args.candidate_diagnostics)
    target_year = int(args.target_year or observations["year"].max())
    if args.training_years:
        training_years = tuple(int(year) for year in args.training_years)
    else:
        training_years = resolve_era_training_years(
            observations,
            target_year=target_year,
            regulation_eras=_regulation_eras(args.config),
        )
    artifact = build_mapping_artifact(
        observations=observations,
        diagnostics=diagnostics,
        policy=str(args.policy),
        training_years=training_years,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    args.output_md.write_text(format_mapping_summary(artifact), encoding="utf-8")
    print(format_mapping_summary(artifact))


def build_mapping_artifact(
    *,
    observations: pd.DataFrame,
    diagnostics: dict[str, Any],
    policy: str,
    training_years: tuple[int, ...],
) -> dict[str, Any]:
    """Build the frozen mapping artifact from calibration observations."""
    mappings = {
        session_kind: fit_linear_team_strength_mapping(
            observations,
            session_kind=session_kind,
            policy=policy,
            training_years=training_years,
        )
        for session_kind in ("race", "qualifying")
    }
    selected_eval = diagnostics.get("policy_evaluations", {}).get(policy, {})
    # Leave-one-season-out folds compare across seasons, so they only validate a
    # mapping fitted across seasons. Fitting on one year - which a regulation change
    # forces - needs leave-one-round-out inside that year instead, or the artifact
    # ships validation drawn from a field it was deliberately not fitted on.
    within_season = (
        evaluate_within_season_folds(observations, policy=policy, year=training_years[0])
        if len(training_years) == 1
        else None
    )
    return {
        "artifact_type": "team_strength_seconds_mapping",
        "schema_version": 1,
        "model_version": get_model_version(),
        "built_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "training_years": list(training_years),
        "stored_state_policy": "single_team_strength_scalar",
        "sign_convention": "positive_seconds_means_faster_than_field",
        "decision": (
            "Freeze separate race and qualifying seconds mappings over one stored "
            "team_strength scalar; do not split short-run and long-run state."
        ),
        "mappings": {
            session_kind: {
                "session_kind": mapping.session_kind,
                "policy": mapping.policy,
                "intercept_s": mapping.intercept_s,
                "slope_s_per_unit": mapping.slope_s_per_unit,
                "training_years": list(mapping.training_years),
            }
            for session_kind, mapping in mappings.items()
        },
        "validation": {
            "selected_policy": policy,
            "primary_folds": (
                "within_season_leave_one_round_out" if within_season else "leave_one_season_out"
            ),
            "within_season_folds": (within_season or {}).get("folds", []),
            "cross_season_folds_note": (
                "Cross-season folds below are retained for provenance only. They are "
                "measured on seasons this mapping was not fitted on, across a "
                "regulation boundary, so they do not validate it."
                if within_season
                else ""
            ),
            "folds": selected_eval.get("folds", []),
            "coverage": [
                row for row in diagnostics.get("coverage", []) if row.get("policy") == policy
            ],
        },
    }


def format_mapping_summary(artifact: dict[str, Any]) -> str:
    """Format the frozen mapping artifact as compact Markdown."""
    lines = [
        "# Team-Strength Seconds Mapping",
        "",
        f"Built at: `{artifact['built_at']}`",
        f"Model version: `{artifact['model_version']}`",
        f"Policy: `{artifact['policy']}`",
        f"Stored state: `{artifact['stored_state_policy']}`",
        "",
        artifact["decision"],
        "",
        "Positive seconds mean a faster-than-field team contribution.",
        "",
        "## Frozen mappings",
        "",
        "| Session | Intercept (s) | Slope (s/unit) | Training years |",
        "| --- | ---: | ---: | --- |",
    ]
    for session_kind, mapping in artifact["mappings"].items():
        years = ", ".join(str(year) for year in mapping["training_years"])
        lines.append(
            f"| `{session_kind}` | {mapping['intercept_s']:.6f} | "
            f"{mapping['slope_s_per_unit']:.6f} | {years} |"
        )

    lines.extend(
        [
            "",
            "## Validation fold summary",
            "",
            "| Session | Holdout season | Rows | RMSE (s) | Prediction slope |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in artifact.get("validation", {}).get("folds", []):
        lines.append(
            f"| `{fold['session_kind']}` | {fold['holdout_year']} | {fold['n_rows']} | "
            f"{_fmt(fold.get('rmse_s'))} | {_fmt(fold.get('prediction_slope'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_candidate_diagnostics(path: Path) -> dict[str, Any]:
    """Load candidate diagnostics when present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    """Format optional numeric values for Markdown."""
    if value is None:
        return " - "
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
