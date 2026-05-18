"""Test whether split FP practice signals beat a shared team signal.

The diagnostic is intentionally limited to conventional weekends with cached
FP1, FP2, FP3, Qualifying, and Race sessions. It compares three single-signal
policies with one split policy on common held-out rows so coverage differences
cannot masquerade as accuracy gains.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.practice_signal_split import (  # noqa: E402
    NORMAL_WEEKEND_SESSIONS,
    aggregate_weekend_profile_scores,
    attach_practice_signals,
    discover_cached_normal_weekends,
    evaluate_practice_signal_policies,
    extract_practice_profile_scores,
    pivot_weekend_profile_scores,
    restrict_to_common_signal_rows,
    summarize_signal_coverage,
    summarize_weighted_policy_metrics,
)
from src.models.team_strength_mapping import (  # noqa: E402
    build_construct_aligned_driver_observations,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the practice-signal split diagnostic."""
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
        "--cache-dir",
        type=Path,
        default=Path("data/raw/.fastf1_cache"),
        help="Local FastF1 cache used for offline practice-session loading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/practice_signal_split"),
        help="Directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Seasons used in held-out-fold evaluation.",
    )
    return parser


def main() -> None:
    """Run the normal-weekend practice-signal split diagnostic."""
    args = build_parser().parse_args()
    prior_artifact = json.loads(args.prior_artifact.read_text(encoding="utf-8"))
    raw_matched_laps = pd.read_csv(args.raw_matched_laps)
    raw_matched_laps = raw_matched_laps[raw_matched_laps["year"].isin(args.years)].copy()

    observations = build_construct_aligned_driver_observations(
        raw_matched_laps,
        driver_mu_by_kind=_driver_mu_by_kind(prior_artifact),
    )
    observations = observations.dropna(subset=["driver_rating_mu_s", "team_target_s"]).reset_index(
        drop=True
    )
    normal_weekends = discover_cached_normal_weekends(
        events=observations[["year", "race_name"]],
        cache_dir=args.cache_dir,
    )
    practice_profile_scores = build_practice_profile_scores(
        normal_weekends=normal_weekends,
        cache_dir=args.cache_dir,
    )
    weekend_signals = pivot_weekend_profile_scores(practice_profile_scores)
    enriched = attach_practice_signals(
        observations=observations.merge(
            normal_weekends[["year", "race_name", "is_normal_weekend"]],
            on=["year", "race_name"],
            how="left",
            validate="many_to_one",
        ),
        weekend_signals=weekend_signals,
    )
    enriched = enriched[enriched["is_normal_weekend"].eq(True)].reset_index(drop=True)

    coverage = summarize_signal_coverage(enriched)
    evaluation = evaluate_practice_signal_policies(enriched)
    common_rows = restrict_to_common_signal_rows(enriched)
    diagnostics = {
        "built_at": datetime.now(UTC).isoformat(),
        "years": [int(year) for year in args.years],
        "normal_weekend_count": int(normal_weekends["is_normal_weekend"].sum()),
        "excluded_weekend_count": int((~normal_weekends["is_normal_weekend"]).sum()),
        "normal_weekend_counts_by_year": (
            normal_weekends[normal_weekends["is_normal_weekend"].eq(True)]
            .groupby("year", as_index=False)
            .agg(normal_weekends=("race_name", "nunique"))
        ),
        "coverage": coverage,
        "fold_metrics": evaluation["fold_metrics"],
        "weighted_policy_metrics": summarize_weighted_policy_metrics(evaluation["fold_metrics"]),
        "comparison_vs_best_shared": evaluation["comparison_vs_best_shared"],
    }
    write_outputs(
        output_dir=args.output_dir,
        normal_weekends=normal_weekends,
        practice_profile_scores=practice_profile_scores,
        observations=enriched,
        common_rows=common_rows,
        diagnostics=diagnostics,
    )
    print(format_report(diagnostics))


def build_practice_profile_scores(
    *,
    normal_weekends: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    """Load cached FP sessions and build long-form weekend profile scores."""
    fastf1 = _configure_fastf1(cache_dir=cache_dir)
    frames: list[pd.DataFrame] = []
    selected_weekends = normal_weekends[normal_weekends["is_normal_weekend"].eq(True)]

    for weekend in selected_weekends.itertuples(index=False):
        session_scores: dict[str, dict[str, dict[str, float]]] = {}
        for session_code in NORMAL_WEEKEND_SESSIONS:
            session = fastf1.get_session(int(weekend.year), str(weekend.race_name), session_code)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                session.load(laps=True, weather=True, telemetry=False, messages=False)
            session_scores[session_code] = extract_practice_profile_scores(session)
        frames.append(
            aggregate_weekend_profile_scores(
                year=int(weekend.year),
                race_name=str(weekend.race_name),
                session_scores=session_scores,
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_outputs(
    *,
    output_dir: Path,
    normal_weekends: pd.DataFrame,
    practice_profile_scores: pd.DataFrame,
    observations: pd.DataFrame,
    common_rows: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    """Write tabular and human-readable diagnostic artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_weekends.to_csv(output_dir / "normal_weekends.csv", index=False)
    practice_profile_scores.to_csv(output_dir / "practice_profile_scores.csv", index=False)
    observations.to_csv(output_dir / "observations_with_practice_signals.csv", index=False)
    common_rows.to_csv(output_dir / "common_evaluation_rows.csv", index=False)
    diagnostics["fold_metrics"].to_csv(output_dir / "fold_metrics.csv", index=False)
    diagnostics["weighted_policy_metrics"].to_csv(
        output_dir / "weighted_policy_metrics.csv",
        index=False,
    )
    diagnostics["comparison_vs_best_shared"].to_csv(
        output_dir / "comparison_vs_best_shared.csv",
        index=False,
    )
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(_json_safe(diagnostics), indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostic_report.md").write_text(
        format_report(diagnostics),
        encoding="utf-8",
    )


def format_report(diagnostics: dict[str, Any]) -> str:
    """Format the split diagnostic as compact Markdown."""
    coverage = diagnostics["coverage"]
    fold_metrics = diagnostics["fold_metrics"]
    weighted_policy_metrics = diagnostics["weighted_policy_metrics"]
    comparison = diagnostics["comparison_vs_best_shared"]
    lines = [
        "# Practice Signal Split Diagnostic",
        "",
        f"- Built at: `{diagnostics['built_at']}`",
        f"- Years: `{', '.join(str(year) for year in diagnostics['years'])}`",
        f"- Normal weekends included: `{diagnostics['normal_weekend_count']}`",
        f"- Weekends excluded: `{diagnostics['excluded_weekend_count']}`",
        "",
        "## Interpretation",
        "",
        "- This is a pre-2026 support diagnostic, not the regulation-reset acceptance test.",
        "- The normal-weekend counts below reflect currently cached practice payloads, not the "
        "historical season calendar. Low counts in a season can therefore mean missing local FP "
        "cache coverage, not only sprint-weekend format.",
        "- On the current cached rows, the split policy does not show a consistent accuracy gain: "
        "the row-weighted combined MSE is worse than `shared_long_run`, it wins only two of four "
        "combined folds against the best shared policy, and it wins no qualifying fold against the "
        "best shared policy.",
        "- Current decision: keep one stored team-strength state in v1 and do not split it into "
        "separate short-run and long-run states yet. Separate race and qualifying seconds mappings "
        "remain a different, still-valid design choice.",
        "- The decisive transfer-era test is 2026 conventional-weekend evidence under the new "
        "regulations. Reopen the state split only if 2026 shows a consistent MSE gain, not a one-off "
        "improvement.",
        "",
        "## Normal Weekend Coverage By Year",
        "",
        "| Year | Normal weekends |",
        "| ---: | ---: |",
    ]
    for row in diagnostics["normal_weekend_counts_by_year"].to_dict(orient="records"):
        lines.append(f"| {row['year']} | {row['normal_weekends']} |")

    lines.extend(
        [
            "",
            "## Coverage Before Common-Row Restriction",
            "",
            "| Session kind | Total rows | Balanced | Short run | Long run | Common rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in coverage.to_dict(orient="records"):
        lines.append(
            f"| `{row['session_kind']}` | {row['total_rows']} | {row['balanced_rows']} | "
            f"{row['short_run_rows']} | {row['long_run_rows']} | {row['common_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Row-Weighted Policy Summary",
            "",
            "| Session kind | Policy | Rows | Weighted MSE | Weighted RMSE |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in weighted_policy_metrics.to_dict(orient="records"):
        lines.append(
            f"| `{row['session_kind']}` | `{row['policy']}` | {row['n_rows']} | "
            f"{_fmt(row['weighted_mse_s2'])} | {_fmt(row['weighted_rmse_s'])} |"
        )

    lines.extend(
        [
            "",
            "## Held-Out Fold Metrics",
            "",
            "| Session kind | Holdout | Policy | Profile | Rows | MSE | RMSE | R² |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_metrics.to_dict(orient="records"):
        lines.append(
            f"| `{row['session_kind']}` | {row['holdout_year']} | `{row['policy']}` | "
            f"`{row['profile']}` | {row['n_rows']} | {_fmt(row['mse_s'])} | "
            f"{_fmt(row['rmse_s'])} | {_fmt(row['r_squared'])} |"
        )

    lines.extend(
        [
            "",
            "## Split Versus Best Shared Fold",
            "",
            "| Session kind | Holdout | Split MSE | Best shared | Best shared MSE | Delta | Split wins |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in comparison.to_dict(orient="records"):
        lines.append(
            f"| `{row['session_kind']}` | {row['holdout_year']} | "
            f"{_fmt(row['split_mse_s2'])} | `{row['best_shared_policy']}` | "
            f"{_fmt(row['best_shared_mse_s2'])} | {_fmt(row['delta_mse_s2'])} | "
            f"`{row['split_wins']}` |"
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


def _configure_fastf1(*, cache_dir: Path) -> Any:
    """Configure FastF1 for offline reads from the local cache."""
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    if hasattr(fastf1.Cache, "offline_mode"):
        fastf1.Cache.offline_mode(True)
    try:
        fastf1.set_log_level("CRITICAL")
    except (AttributeError, TypeError):
        pass
    return fastf1


def _fmt(value: Any) -> str:
    """Format numeric report values compactly."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def _json_safe(value: Any) -> Any:
    """Convert pandas and numpy objects into JSON-safe Python values."""
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
