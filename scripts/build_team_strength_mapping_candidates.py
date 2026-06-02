"""Build Phase 7 team-strength mapping candidate diagnostics.

This script intentionally stops short of freezing a final mapping policy. It
materializes the construct-aligned calibration rows and compares the explicit
team-strength proxy choices that are still open at the start of Phase 7.
"""

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

from src.models.team_strength_mapping import (  # noqa: E402
    TEAM_STRENGTH_POLICY_COLUMNS,
    build_construct_aligned_driver_observations,
    evaluate_policy_folds,
    summarize_policy_coverage,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Phase 7 candidate diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-matched-laps",
        type=Path,
        default=Path("data/processed/teammate_network_observations/latest/raw_matched_laps.csv"),
        help="Phase 5 raw matched-lap artifact.",
    )
    parser.add_argument(
        "--prior-artifact",
        type=Path,
        default=Path("data/processed/teammate_network_prior/latest.json"),
        help="Phase 6 teammate-network prior artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/team_strength_seconds_mapping"),
        help="Directory for Phase 7 candidate diagnostics.",
    )
    return parser


def main() -> None:
    """Build observations, evaluate every candidate policy, and write outputs."""
    args = build_parser().parse_args()
    raw_matched_laps = pd.read_csv(args.raw_matched_laps)
    prior_artifact = json.loads(args.prior_artifact.read_text())
    observations = build_construct_aligned_driver_observations(
        raw_matched_laps,
        driver_mu_by_kind=_driver_mu_by_kind(prior_artifact),
    )
    usable = observations.dropna(subset=["driver_rating_mu_s", "team_target_s"]).reset_index(
        drop=True
    )
    diagnostics = build_candidate_diagnostics(usable)
    write_candidate_outputs(
        output_dir=args.output_dir,
        observations=usable,
        diagnostics=diagnostics,
    )
    print(format_candidate_summary(diagnostics))


def build_candidate_diagnostics(observations: pd.DataFrame) -> dict[str, Any]:
    """Evaluate all current team-strength proxy policies on 2022-2025 folds."""
    evaluations = {
        policy: _json_safe(evaluate_policy_folds(observations, policy=policy))
        for policy in TEAM_STRENGTH_POLICY_COLUMNS
    }
    for evaluation in evaluations.values():
        evaluation.pop("held_out_predictions", None)
    return {
        "built_at": datetime.now(UTC).isoformat(),
        "coverage": summarize_policy_coverage(observations),
        "policy_evaluations": evaluations,
    }


def write_candidate_outputs(
    *,
    output_dir: Path,
    observations: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    """Write the current Phase 7 candidate artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "calibration_observations.csv", index=False)
    (output_dir / "candidate_diagnostics.json").write_text(
        json.dumps(_json_safe(diagnostics), indent=2),
        encoding="utf-8",
    )
    (output_dir / "candidate_diagnostics.md").write_text(
        format_candidate_summary(diagnostics),
        encoding="utf-8",
    )


def format_candidate_summary(diagnostics: dict[str, Any]) -> str:
    """Format candidate diagnostics as compact Markdown."""
    lines = [
        "# Team-Strength Seconds Mapping Candidate Diagnostics",
        "",
        f"Built at: `{diagnostics['built_at']}`",
        "",
        "## Coverage",
        "",
        "| Policy | Session kind | Usable rows | Total rows |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in diagnostics["coverage"]:
        lines.append(
            f"| `{row['policy']}` | `{row['session_kind']}` | "
            f"{row['usable_rows']} | {row['total_rows']} |"
        )

    for policy, evaluation in diagnostics["policy_evaluations"].items():
        lines.extend(
            [
                "",
                f"## `{policy}`",
                "",
                "| Session kind | Holdout season | Rows | Intercept | Slope | R² | "
                "Prediction slope | RMSE |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for fold in evaluation["folds"]:
            lines.append(
                f"| `{fold['session_kind']}` | {fold['holdout_year']} | {fold['n_rows']} | "
                f"{_fmt(fold['intercept_s'])} | {_fmt(fold['slope_s_per_unit'])} | "
                f"{_fmt(fold['r_squared'])} | {_fmt(fold['prediction_slope'])} | "
                f"{_fmt(fold['rmse_s'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _driver_mu_by_kind(prior_artifact: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract race and qualifying prior means from the Phase 6 artifact."""
    return {
        "race": {
            driver_code: float(payload["mu_s"])
            for driver_code, payload in prior_artifact["race_network"]["drivers"].items()
        },
        "qualifying": {
            driver_code: float(payload["mu_s"])
            for driver_code, payload in prior_artifact["quali_network"]["drivers"].items()
        },
    }


def _fmt(value: Any) -> str:
    """Format compact Markdown table values."""
    if value is None:
        return " - "
    return f"{float(value):.3f}"


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible plain Python types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


if __name__ == "__main__":
    main()
