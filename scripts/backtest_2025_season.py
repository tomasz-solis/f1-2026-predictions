"""
Season backtest runner with ablation support and overfitting checks.

Usage example:
  python scripts/backtest_2025_season.py --year 2025 --max-races 8 \
    --experiment "higher_grid_anchor:baseline_predictor.race.grid_anchor.base=0.45" \
    --experiment "lower_sc_noise:baseline_predictor.race.safety_car_luck_range=0.15"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import fastf1

# Add project root to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.persistence.artifact_store import ArtifactStore
from src.predictors import Baseline2026Predictor
from src.utils.backtesting import (
    NestedDictConfig,
    aggregate_race_metrics,
    apply_config_overrides,
    build_checked_backtest_summary,
    build_error_analysis,
    build_overlap_comparison,
    build_segment_breakdown,
    get_races_for_year,
    load_config_dict,
    parse_experiment_spec,
    rank_experiments_for_generalization,
    run_previous_race_naive_backtest,
    run_single_race_backtest,
    summarize_generalization,
    warm_fastf1_results_cache,
    write_csv,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("requests_cache").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("_").lower() or "experiment"


def _parse_races_arg(raw_races: str | None) -> list[str]:
    if not raw_races:
        return []
    return [race.strip() for race in raw_races.split(",") if race.strip()]


def _expand_learning_modes(raw_mode: str) -> list[str]:
    """Expand CLI learning-mode selection into concrete run labels."""
    normalized = str(raw_mode).strip().lower()
    if normalized == "both":
        return ["static", "adaptive"]
    return [normalized]


def _build_backtest_predictor(
    *,
    season_year: int,
    seed: int,
    merged_config: dict[str, Any],
    artifact_store: ArtifactStore,
    data_dir: str | Path = "data/processed",
) -> Baseline2026Predictor:
    """Build one predictor instance with season-matched priors for replay runs."""
    return Baseline2026Predictor(
        data_dir=str(data_dir),
        seed=seed,
        season_year=season_year,
        config=NestedDictConfig(merged_config),
        artifact_store=artifact_store,
    )


def _resolve_predictor_data_dir() -> Path:
    """Resolve the same processed-data root the predictor will read from."""
    env_data_dir = os.getenv("F1_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)
    return Path("data/processed")


def _normalize_season_prior_mode(raw_mode: str) -> str:
    """Return one normalized season-prior mode string."""
    normalized = str(raw_mode).strip().lower().replace("_", "-")
    if normalized not in {"auto", "allow", "proxy-only"}:
        raise ValueError("Unsupported season_prior_mode. Use one of: auto, allow, proxy-only.")
    return normalized


def _resolve_effective_season_prior_mode(
    *,
    requested_mode: str,
    evaluation_mode: str,
) -> str:
    """Resolve the effective season-prior mode for one backtest run.

    Historical replay should default to the retained cross-season proxy path
    unless we explicitly opt into season-scoped priors. That keeps the
    canonical review packet reproducible even when local untracked prior files
    exist on disk.
    """
    normalized_requested = _normalize_season_prior_mode(requested_mode)
    if normalized_requested != "auto":
        return normalized_requested
    normalized_evaluation = str(evaluation_mode).strip().lower()
    return "proxy-only" if normalized_evaluation == "historical" else "allow"


def _prepare_backtest_data_dir(
    *,
    source_data_dir: Path,
    output_dir: Path,
    season_year: int,
    season_prior_mode: str,
) -> Path:
    """Prepare the processed-data directory used by the backtest.

    In ``proxy-only`` mode we copy the processed tree into the run output and
    strip the target season's team/driver/track priors. That gives us a clean,
    explicit evaluation path without mutating the user's local data files.
    """
    normalized_mode = _normalize_season_prior_mode(season_prior_mode)
    if normalized_mode == "allow":
        return source_data_dir

    prepared_root = (
        output_dir / "_backtest_inputs" / f"{season_year}_{normalized_mode.replace('-', '_')}"
    )
    prepared_data_dir = prepared_root / "processed"
    prepared_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_data_dir, prepared_data_dir, dirs_exist_ok=True)

    removed_paths: list[str] = []
    for relative_path in (
        Path("car_characteristics") / f"{season_year}_car_characteristics.json",
        Path("driver_characteristics") / f"{season_year}_driver_characteristics.json",
        Path("track_characteristics") / f"{season_year}_track_characteristics.json",
    ):
        target_path = prepared_data_dir / relative_path
        if target_path.exists():
            target_path.unlink()
            removed_paths.append(str(target_path))

    logger.info(
        "Prepared %s replay data root at %s from %s.",
        normalized_mode,
        prepared_data_dir,
        source_data_dir,
    )
    if removed_paths:
        logger.info(
            "Removed season-scoped priors for %s replay: %s",
            season_year,
            ", ".join(removed_paths),
        )
    return prepared_data_dir


def _inspect_season_prior_status(*, data_dir: Path, season_year: int) -> dict[str, dict[str, Any]]:
    """Report whether season-scoped prior artifacts exist for one replay season."""
    driver_path = data_dir / "driver_characteristics" / f"{season_year}_driver_characteristics.json"
    legacy_driver_path = data_dir / "driver_characteristics.json"
    return {
        "team": {
            "path": str(
                data_dir / "car_characteristics" / f"{season_year}_car_characteristics.json"
            ),
            "season_scoped": (
                data_dir / "car_characteristics" / f"{season_year}_car_characteristics.json"
            ).exists(),
            "fallback_path": None,
        },
        "driver": {
            "path": str(driver_path),
            "season_scoped": driver_path.exists(),
            "fallback_path": str(legacy_driver_path) if legacy_driver_path.exists() else None,
        },
        "track": {
            "path": str(
                data_dir / "track_characteristics" / f"{season_year}_track_characteristics.json"
            ),
            "season_scoped": (
                data_dir / "track_characteristics" / f"{season_year}_track_characteristics.json"
            ).exists(),
            "fallback_path": None,
        },
    }


def _summarize_missing_season_priors(prior_status: dict[str, dict[str, Any]]) -> list[str]:
    """Return the prior categories that still lack season-scoped artifacts."""
    return [
        category
        for category, status in prior_status.items()
        if isinstance(status, dict) and not bool(status.get("season_scoped"))
    ]


def _emit_markdown_recommendations(
    output_path: Path,
    *,
    ranked: list[dict[str, Any]],
    baseline_test_mae: float | None,
    min_improvement: float,
    max_gap: float,
) -> None:
    lines = [
        "# Backtest Recommendations",
        "",
        "Selection rule:",
        (
            f"- Improve test race MAE by at least `{min_improvement:.2f}` versus baseline "
            f"and keep generalization gap <= `{max_gap:.2f}`"
        ),
        "",
    ]

    if baseline_test_mae is not None:
        lines.append(f"Baseline test race MAE: `{baseline_test_mae:.3f}`")
        lines.append("")

    recommended = [item for item in ranked if item.get("recommended")]
    if not recommended:
        lines.append("No ablation passed the generalization threshold.")
    else:
        lines.append("Recommended experiments:")
        for item in recommended:
            lines.append(
                "- "
                f"{item['name']} | test_mae={item['test_race_mae']:.3f} | "
                f"improvement={item['test_race_mae_improvement_vs_baseline']:.3f} | "
                f"gap={item['generalization_gap_race_mae']:.3f} | "
                f"overrides={item['overrides']}"
            )

    lines.append("")
    lines.append("All experiments:")
    for item in ranked:
        lines.append(
            "- "
            f"{item['name']} | recommended={item['recommended']} | "
            f"train_mae={item['train_race_mae']} | test_mae={item['test_race_mae']} | "
            f"gap={item['generalization_gap_race_mae']} | "
            f"improvement={item['test_race_mae_improvement_vs_baseline']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _fmt_metric(value: Any, decimals: int = 3) -> str:
    """Format numeric review metrics consistently."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _describe_signed_delta(
    value: Any,
    *,
    positive_label: str,
    negative_label: str,
    decimals: int = 3,
) -> str | None:
    """Render a signed metric delta in plain language."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    magnitude = _fmt_metric(abs(numeric), decimals)
    if numeric > 0:
        return f"{positive_label} `{magnitude}`"
    if numeric < 0:
        return f"{negative_label} `{magnitude}`"
    return f"matched exactly (`{magnitude}` delta)"


def _select_segment_extreme(
    segment_breakdown: dict[str, dict[str, dict[str, Any]]],
    *,
    metric_key: str,
    prefer_lowest: bool,
) -> dict[str, Any] | None:
    """Pick the strongest or weakest segment bucket from aggregated backtest slices."""
    candidates: list[dict[str, Any]] = []
    for dimension, buckets in segment_breakdown.items():
        for bucket_name, summary in buckets.items():
            metric_value = summary.get(metric_key)
            events = int(summary.get("events", 0) or 0)
            if metric_value is None or events == 0:
                continue
            candidates.append(
                {
                    "dimension": dimension,
                    "bucket": bucket_name,
                    "metric_value": float(metric_value),
                    "events": events,
                }
            )

    if not candidates:
        return None

    def _sort_key(item: dict[str, Any]) -> tuple[float, int, str, str]:
        return (
            item["metric_value"],
            -item["events"],
            item["dimension"],
            item["bucket"],
        )

    return min(candidates, key=_sort_key) if prefer_lowest else max(candidates, key=_sort_key)


def _build_learning_mode_comparison(
    adaptive_report: dict[str, Any],
    static_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compare adaptive and static baselines in reviewer-friendly units."""
    if not isinstance(static_report, dict):
        return None

    adaptive_summary = adaptive_report.get("summary", {})
    static_summary = static_report.get("summary", {})

    def _delta(lower_is_better_key: str) -> float | None:
        adaptive_value = adaptive_summary.get(lower_is_better_key)
        static_value = static_summary.get(lower_is_better_key)
        if adaptive_value is None or static_value is None:
            return None
        return float(static_value) - float(adaptive_value)

    def _lift(higher_is_better_key: str) -> float | None:
        adaptive_value = adaptive_summary.get(higher_is_better_key)
        static_value = static_summary.get(higher_is_better_key)
        if adaptive_value is None or static_value is None:
            return None
        return float(adaptive_value) - float(static_value)

    return {
        "adaptive_name": adaptive_report.get("name"),
        "static_name": static_report.get("name"),
        "race_mae_improvement": _delta("race_mae_mean"),
        "qualifying_mae_improvement": _delta("qualifying_mae_mean"),
        "top3_accuracy_delta": _lift("top3_accuracy_mean"),
        "winner_accuracy_delta": _lift("winner_accuracy_percent"),
    }


def _build_reviewer_takeaways(packet: dict[str, Any]) -> list[str]:
    """Convert the main bundle metrics into short reviewer-facing takeaways."""
    takeaways: list[str] = []
    canonical_summary = (packet.get("canonical_model") or {}).get("summary", {})
    overlap = packet.get("overlap_comparison", {})
    learning_delta = packet.get("adaptive_vs_static_comparison")
    segment_breakdown = packet.get("canonical_segment_breakdown", {})
    recommended = packet.get("recommended_experiments", [])
    prior_status = packet.get("season_prior_status", {})
    season_prior_mode = str(packet.get("season_prior_mode", "allow")).strip().lower()

    missing_priors = _summarize_missing_season_priors(prior_status)
    if missing_priors:
        if season_prior_mode == "proxy-only":
            takeaways.append(
                "Replay ran in `proxy-only` prior mode, so season-scoped priors were "
                f"intentionally disabled for {', '.join(missing_priors)}."
            )
        else:
            takeaways.append(
                "Season-scoped priors are still missing for "
                f"{', '.join(missing_priors)}, so this replay still leans on fallback artifacts."
            )

    race_mae = canonical_summary.get("race_mae_mean")
    top3_accuracy = canonical_summary.get("top3_accuracy_mean")
    if race_mae is not None and top3_accuracy is not None:
        takeaways.append(
            "Canonical adaptive model averaged "
            f"`{_fmt_metric(race_mae)}` race MAE with `{_fmt_metric(top3_accuracy, 1)}%` top-3 accuracy."
        )

    race_mae_improvement = overlap.get("race_mae_improvement")
    if race_mae_improvement is not None:
        overlap_text = _describe_signed_delta(
            race_mae_improvement,
            positive_label="beat the naive previous-race baseline on overlap race MAE by",
            negative_label="trailed the naive previous-race baseline by",
        )
        if overlap_text is not None:
            takeaways.append(
                f"Against the naive previous-race baseline, the canonical model {overlap_text} on shared weekends."
            )

    if isinstance(learning_delta, dict) and learning_delta.get("race_mae_improvement") is not None:
        learning_text = _describe_signed_delta(
            learning_delta.get("race_mae_improvement"),
            positive_label="improved race MAE versus the static replay by",
            negative_label="hurt race MAE versus the static replay by",
        )
        if learning_text is not None:
            takeaways.append(f"Adaptive learning {learning_text}.")

    strongest_segment = _select_segment_extreme(
        segment_breakdown,
        metric_key="race_mae_mean",
        prefer_lowest=True,
    )
    weakest_segment = _select_segment_extreme(
        segment_breakdown,
        metric_key="race_mae_mean",
        prefer_lowest=False,
    )
    if strongest_segment is not None and weakest_segment is not None:
        takeaways.append(
            "Best segment: "
            f"`{strongest_segment['dimension']}={strongest_segment['bucket']}` "
            f"at race MAE `{_fmt_metric(strongest_segment['metric_value'])}`; "
            "weakest segment: "
            f"`{weakest_segment['dimension']}={weakest_segment['bucket']}` "
            f"at `{_fmt_metric(weakest_segment['metric_value'])}`."
        )

    if recommended:
        takeaways.append(
            f"`{len(recommended)}` ablation experiment(s) cleared the generalization guardrails."
        )
    else:
        takeaways.append("No ablation beat the adaptive baseline cleanly enough to recommend.")

    return takeaways


def _emit_review_packet_markdown(output_path: Path, packet: dict[str, Any]) -> None:
    """Write a compact reviewer-facing summary of the latest backtest bundle."""
    canonical = packet.get("canonical_model", {})
    canonical_summary = canonical.get("summary", {})
    generalization = canonical.get("generalization", {})
    overlap = packet.get("overlap_comparison", {})
    static_baseline = packet.get("static_baseline")
    learning_delta = packet.get("adaptive_vs_static_comparison")
    segment_breakdown = packet.get("canonical_segment_breakdown", {})
    error_analysis = packet.get("canonical_error_analysis", {})
    reviewer_takeaways = packet.get("reviewer_takeaways", [])
    recommended_experiments = packet.get("recommended_experiments", [])
    prior_status = packet.get("season_prior_status", {})
    season_prior_mode = packet.get("season_prior_mode")
    season_prior_source_data_dir = packet.get("season_prior_source_data_dir")
    season_prior_data_dir = packet.get("season_prior_data_dir")

    lines = [
        "# Review Packet",
        "",
        f"- Season: `{packet.get('season')}`",
        f"- Evaluation mode: `{packet.get('evaluation_mode')}`",
        f"- Canonical model: `{canonical.get('name', 'unknown')}`",
        f"- Race MAE: `{canonical_summary.get('race_mae_mean')}`",
        f"- Qualifying MAE: `{canonical_summary.get('qualifying_mae_mean')}`",
        f"- Top-3 accuracy: `{canonical_summary.get('top3_accuracy_mean')}`",
        f"- Winner accuracy: `{canonical_summary.get('winner_accuracy_percent')}`",
        f"- Generalization gap (race MAE): `{generalization.get('generalization_gap_race_mae')}`",
        f"- Weather assumption: `{packet.get('weather')}`",
        "",
    ]

    if isinstance(prior_status, dict) and prior_status:
        lines.extend(["## Prior Provenance", ""])
        if season_prior_mode:
            lines.append(f"- Mode: `{season_prior_mode}`")
        if season_prior_source_data_dir:
            lines.append(f"- Source data dir: `{season_prior_source_data_dir}`")
        if season_prior_data_dir:
            lines.append(f"- Effective data dir: `{season_prior_data_dir}`")
        for category, status in sorted(prior_status.items()):
            if not isinstance(status, dict):
                continue
            path = status.get("path", "n/a")
            fallback_path = status.get("fallback_path")
            season_scoped = bool(status.get("season_scoped"))
            suffix = "season-scoped" if season_scoped else "missing season-scoped file"
            if fallback_path:
                suffix = f"{suffix}; fallback `{fallback_path}`" if not season_scoped else suffix
            lines.append(f"- {category.title()}: `{path}` ({suffix})")
        lines.append("")

    if reviewer_takeaways:
        lines.extend(["## Plain-Language Takeaways", ""])
        for takeaway in reviewer_takeaways:
            lines.append(f"- {takeaway}")
        lines.append("")

    lines.extend(
        [
            "## Baselines",
            "",
        ]
    )
    overlap_text = _describe_signed_delta(
        overlap.get("race_mae_improvement"),
        positive_label="Model beat naive overlap race MAE by",
        negative_label="Model trailed naive overlap race MAE by",
    )
    if overlap_text is not None:
        lines.append(f"- {overlap_text}")

    if isinstance(static_baseline, dict):
        static_summary = static_baseline.get("summary", {})
        lines.append(
            f"- Static baseline race MAE: `{static_summary.get('race_mae_mean')}` "
            f"({static_baseline.get('name')})"
        )

    if isinstance(learning_delta, dict):
        learning_text = _describe_signed_delta(
            learning_delta.get("race_mae_improvement"),
            positive_label="Adaptive vs static race MAE improvement",
            negative_label="Adaptive vs static race MAE regression",
        )
        lines.extend(
            [
                f"- {learning_text}"
                if learning_text is not None
                else "- Adaptive vs static race MAE delta: `n/a`",
                (
                    "- Adaptive vs static top-3 delta: "
                    f"`{_fmt_metric(learning_delta.get('top3_accuracy_delta'), 1)}` percentage points"
                ),
            ]
        )

    interval_lines: list[str] = []
    for session_label, prefix in (("Qualifying", "qualifying"), ("Race", "race")):
        interval_count = canonical_summary.get(f"{prefix}_interval_count")
        if not interval_count:
            continue
        races_with_data = canonical_summary.get(f"{prefix}_interval_races")
        empirical_coverage = canonical_summary.get(f"{prefix}_interval_empirical_coverage")
        nominal_coverage = canonical_summary.get(f"{prefix}_interval_nominal_coverage")
        interval_width = canonical_summary.get(f"{prefix}_interval_width_mean")
        calibration_error = canonical_summary.get(f"{prefix}_interval_calibration_error")
        interval_lines.append(
            "- "
            f"{session_label}: coverage `{_fmt_metric(float(empirical_coverage) * 100.0, 1)}%` "
            f"vs target `{_fmt_metric(float(nominal_coverage) * 100.0, 1)}%` across "
            f"`{int(interval_count)}` driver-session intervals from "
            f"`{int(races_with_data or 0)}` race(s); mean width "
            f"`{_fmt_metric(interval_width)}`; calibration error "
            f"`{_fmt_metric(float(calibration_error) * 100.0, 1)}`pp."
        )
    if interval_lines:
        lines.extend(["", "## Interval Calibration", "", *interval_lines, ""])

    if segment_breakdown:
        lines.extend(["", "## Segment Breakdown", ""])
        for dimension, buckets in sorted(segment_breakdown.items()):
            lines.extend(
                [
                    f"### {dimension.replace('_', ' ').title()}",
                    "",
                    "| Bucket | Events | Race MAE | Top-3 accuracy | Winner accuracy |",
                    "|---|---|---|---|---|",
                ]
            )
            for bucket_name, summary in sorted(buckets.items()):
                lines.append(
                    f"| {bucket_name} | {summary.get('events', 0)} | "
                    f"{_fmt_metric(summary.get('race_mae_mean'))} | "
                    f"{_fmt_metric(summary.get('top3_accuracy_mean'), 1)}% | "
                    f"{_fmt_metric(summary.get('winner_accuracy_percent'), 1)}% |"
                )
            lines.append("")

    if error_analysis:
        lines.extend(["## Error Analysis", ""])
        worst_races = error_analysis.get("worst_race_events", [])
        if worst_races:
            lines.append("Worst race weekends by MAE:")
            for row in worst_races[:3]:
                lines.append(
                    "- "
                    f"{row.get('race_name')} "
                    f"(`{row.get('track_type')}`, `{row.get('weekend_format')}`, `{row.get('weather')}`) "
                    f"race_mae={_fmt_metric(row.get('race_mae'))}"
                )
            lines.append("")

        winner_misses = error_analysis.get("winner_miss_events", [])
        if winner_misses:
            lines.append("Winner misses:")
            for row in winner_misses[:3]:
                lines.append(
                    "- "
                    f"{row.get('race_name')} "
                    f"(race_mae={_fmt_metric(row.get('race_mae'))}, "
                    f"top3={_fmt_metric(row.get('top3_accuracy'), 1)}%)"
                )
            lines.append("")

    if recommended_experiments:
        lines.extend(["## Recommended Ablations", ""])
        for item in recommended_experiments:
            lines.append(
                "- "
                f"{item.get('name')} | "
                f"test_mae={_fmt_metric(item.get('test_race_mae'))} | "
                f"improvement={_fmt_metric(item.get('test_race_mae_improvement_vs_baseline'))} | "
                f"gap={_fmt_metric(item.get('generalization_gap_race_mae'))}"
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `evaluation_packet.json` - machine-readable summary for review and CI",
            "- `recommendations.md` - experiment ranking and selection notes",
            "- `experiment_comparison.csv` - all experiment summaries in one table",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _plot_error_distribution(output_dir: Path, reports: list[dict[str, Any]]) -> None:
    mpl_cache_dir = output_dir / ".mpl_cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.info(f"Skipping plot output (matplotlib unavailable): {exc}")
        return

    plt.figure(figsize=(10, 6))
    has_data = False
    for report in reports:
        race_mae = [
            float(row["race_mae"])
            for row in report.get("race_results", [])
            if row.get("status") == "ok" and row.get("race_mae") is not None
        ]
        if not race_mae:
            continue
        has_data = True
        plt.hist(race_mae, bins=8, alpha=0.45, label=report["name"])

    if not has_data:
        logger.info("Skipping plot output (no successful races)")
        plt.close()
        return

    plt.title("Race MAE Distribution by Experiment")
    plt.xlabel("Race MAE")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "race_mae_distribution.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    logger.info(f"Wrote plot: {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run season backtesting and ablation analysis with overfitting safeguards."
    )
    parser.add_argument("--year", type=int, default=2025, help="Season year to backtest")
    parser.add_argument(
        "--races",
        type=str,
        default=None,
        help="Optional comma-separated race names to run instead of full schedule",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=None,
        help="Limit number of races from schedule (useful for faster iteration)",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="dry",
        choices=["dry", "rain", "mixed", "wet"],
        help="Weather assumption used for race predictions during backtest",
    )
    parser.add_argument(
        "--quali-sims",
        type=int,
        default=120,
        help="Qualifying Monte Carlo iterations per race",
    )
    parser.add_argument(
        "--race-sims",
        type=int,
        default=120,
        help="Race Monte Carlo iterations per race",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Predictor seed for reproducibility",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Train split fraction for generalization checks",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Base config file for experiments",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/.fastf1_cache_backtest",
        help="FastF1 cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/backtest_2025",
        help="Directory for summary outputs",
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="historical",
        choices=["historical", "live"],
        help="Use historical cutoff timing or current live timing during backtest replay",
    )
    parser.add_argument(
        "--season-prior-mode",
        type=str,
        default="auto",
        choices=["auto", "allow", "proxy-only"],
        help=(
            "Control whether replay uses exact target-season priors. "
            "`auto` defaults to proxy-only for historical replay and allow for live mode."
        ),
    )
    parser.add_argument(
        "--learning-mode",
        type=str,
        default="both",
        choices=["adaptive", "static", "both"],
        help="Replay adaptive learning updates, keep weights static, or write both reports",
    )
    parser.add_argument(
        "--checked-summary-path",
        type=str,
        default=None,
        help=(
            "Optional path for the checked-in summary JSON. "
            "Defaults to data/backtesting/<year>_backtest_results.json."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Ablation experiment spec. Format: "
            "'name:key=value,key2=value2'. Repeat this flag for multiple experiments."
        ),
    )
    parser.add_argument(
        "--min-test-improvement",
        type=float,
        default=0.10,
        help="Minimum test race MAE improvement vs baseline to recommend an experiment",
    )
    parser.add_argument(
        "--max-generalization-gap",
        type=float,
        default=0.35,
        help="Maximum acceptable (test - train) race MAE gap",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if zero races were evaluated",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Attempt to prefetch missing Q/R FastF1 sessions into the local cache first",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(args.cache_dir)
    except Exception as exc:
        logger.warning(f"Could not enable FastF1 cache at {args.cache_dir}: {exc}")

    base_config = load_config_dict(args.config)
    source_predictor_data_dir = _resolve_predictor_data_dir()
    effective_season_prior_mode = _resolve_effective_season_prior_mode(
        requested_mode=args.season_prior_mode,
        evaluation_mode=args.evaluation_mode,
    )
    predictor_data_dir = _prepare_backtest_data_dir(
        source_data_dir=source_predictor_data_dir,
        output_dir=output_dir,
        season_year=args.year,
        season_prior_mode=effective_season_prior_mode,
    )
    season_prior_status = _inspect_season_prior_status(
        data_dir=predictor_data_dir,
        season_year=args.year,
    )
    missing_priors = _summarize_missing_season_priors(season_prior_status)
    logger.info(
        "Season prior mode requested=%s effective=%s",
        args.season_prior_mode,
        effective_season_prior_mode,
    )
    if effective_season_prior_mode == "proxy-only":
        logger.info(
            "Historical replay will use cross-season proxies instead of local %s season priors.",
            args.year,
        )
    elif missing_priors:
        logger.warning(
            "Season-scoped priors missing for %s under %s: %s. "
            "Historical replay will fall back to cross-season proxies where available.",
            args.year,
            predictor_data_dir,
            ", ".join(missing_priors),
        )

    races = _parse_races_arg(args.races)
    if not races:
        races = get_races_for_year(year=args.year, max_races=args.max_races)

    if not races:
        logger.error("No races were resolved for backtesting.")
        return 1

    logger.info(f"Running backtest for {len(races)} races: {races}")

    if args.fetch_missing:
        preload_report = warm_fastf1_results_cache(
            year=args.year,
            race_names=races,
            session_names=("Q", "R"),
        )
        cache_hits = sum(1 for row in preload_report if row["status"] == "ok")
        cache_errors = [row for row in preload_report if row["status"] != "ok"]
        logger.info(
            "Prefetched %s FastF1 result sessions before backtest (%s issues)",
            cache_hits,
            len(cache_errors),
        )
        for row in cache_errors[:10]:
            logger.warning(
                "Prefetch failed for %s %s: %s",
                row["race_name"],
                row["session_name"],
                row.get("reason", "unknown"),
            )

    experiment_specs: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    seen_names = {"baseline"}
    for raw_spec in args.experiment:
        name, overrides = parse_experiment_spec(raw_spec)
        sanitized = _sanitize_name(name)
        if sanitized in seen_names:
            suffix = 2
            while f"{sanitized}_{suffix}" in seen_names:
                suffix += 1
            sanitized = f"{sanitized}_{suffix}"
        seen_names.add(sanitized)
        experiment_specs.append((sanitized, overrides))

    learning_modes = _expand_learning_modes(args.learning_mode)
    reports: list[dict[str, Any]] = []
    for base_name, overrides in experiment_specs:
        for learning_mode in learning_modes:
            name = base_name if len(learning_modes) == 1 else f"{base_name}_{learning_mode}"
            logger.info(
                "Experiment '%s' mode=%s evaluation=%s overrides=%s",
                name,
                learning_mode,
                args.evaluation_mode,
                overrides,
            )
            merged_config = apply_config_overrides(base_config, overrides)
            artifact_store = ArtifactStore(data_root=output_dir / "_backtest_runtime" / name)
            predictor = _build_backtest_predictor(
                season_year=args.year,
                seed=args.seed,
                merged_config=merged_config,
                artifact_store=artifact_store,
                data_dir=predictor_data_dir,
            )
            reset_learning_state = getattr(
                getattr(predictor, "calibration_system", None),
                "reset_state",
                None,
            )
            if callable(reset_learning_state):
                reset_learning_state(season=args.year)

            race_results: list[dict[str, Any]] = []
            for index, race_name in enumerate(races, start=1):
                row = run_single_race_backtest(
                    predictor=predictor,
                    year=args.year,
                    race_name=race_name,
                    weather=args.weather,
                    qualifying_simulations=args.quali_sims,
                    race_simulations=args.race_sims,
                    evaluation_mode=args.evaluation_mode,
                    learning_mode=learning_mode,
                )
                race_results.append(row)
                if row["status"] == "ok":
                    logger.info(
                        f"[{name}] {index}/{len(races)} {race_name}: "
                        f"race_mae={row['race_mae']:.3f}, top3={row['top3_accuracy']:.1f}%"
                    )
                else:
                    logger.info(
                        f"[{name}] {index}/{len(races)} {race_name}: skipped ({row.get('reason')})"
                    )

            summary = aggregate_race_metrics(race_results)
            generalization = summarize_generalization(
                race_results,
                train_fraction=args.train_fraction,
                seed=args.seed,
            )

            report = {
                "name": name,
                "base_name": base_name,
                "learning_mode": learning_mode,
                "evaluation_mode": args.evaluation_mode,
                "overrides": overrides,
                "summary": summary,
                "generalization": generalization,
                "race_results": race_results,
            }
            reports.append(report)

            experiment_dir = output_dir / name
            write_json(experiment_dir / "summary.json", report)
            write_csv(
                experiment_dir / "race_results.csv",
                race_results,
                columns=[
                    "race_name",
                    "status",
                    "evaluation_mode",
                    "learning_mode",
                    "reason",
                    "qualifying_mae",
                    "qualifying_exact_accuracy",
                    "race_mae",
                    "race_exact_accuracy",
                    "race_within_3",
                    "top3_accuracy",
                    "winner_correct",
                ],
            )
            write_json(experiment_dir / "race_results_detailed.json", {"races": race_results})

    ranking_candidates = [report for report in reports if report.get("learning_mode") == "adaptive"]
    if not ranking_candidates:
        ranking_candidates = reports
    ranked = rank_experiments_for_generalization(
        ranking_candidates,
        min_test_race_mae_improvement=args.min_test_improvement,
        max_generalization_gap=args.max_generalization_gap,
    )

    comparison_rows = []
    for report in reports:
        summary = report["summary"]
        generalization = report["generalization"]
        comparison_rows.append(
            {
                "name": report["name"],
                "base_name": report.get("base_name"),
                "learning_mode": report.get("learning_mode"),
                "evaluation_mode": report.get("evaluation_mode"),
                "overrides": report["overrides"],
                "races_evaluated": summary.get("races_evaluated"),
                "race_mae_mean": summary.get("race_mae_mean"),
                "top3_accuracy_mean": summary.get("top3_accuracy_mean"),
                "winner_accuracy_percent": summary.get("winner_accuracy_percent"),
                "train_race_mae": generalization.get("train", {}).get("race_mae_mean"),
                "test_race_mae": generalization.get("test", {}).get("race_mae_mean"),
                "generalization_gap_race_mae": generalization.get("generalization_gap_race_mae"),
            }
        )

    write_csv(
        output_dir / "experiment_comparison.csv",
        comparison_rows,
        columns=[
            "name",
            "base_name",
            "learning_mode",
            "evaluation_mode",
            "overrides",
            "races_evaluated",
            "race_mae_mean",
            "top3_accuracy_mean",
            "winner_accuracy_percent",
            "train_race_mae",
            "test_race_mae",
            "generalization_gap_race_mae",
        ],
    )
    write_json(output_dir / "experiment_rankings.json", {"rankings": ranked})

    baseline = next(
        (
            item
            for item in reports
            if item.get("base_name") == "baseline" and item.get("learning_mode") == "adaptive"
        ),
        next(
            (item for item in reports if item.get("base_name") == "baseline"),
            reports[0],
        ),
    )
    naive_report = run_previous_race_naive_backtest(year=args.year, race_names=races)
    overlap_comparison = build_overlap_comparison(
        model_race_results=baseline.get("race_results", []),
        naive_race_results=naive_report.get("race_results", []),
    )
    checked_summary_path = Path(args.checked_summary_path or "")
    if not args.checked_summary_path:
        checked_summary_path = Path("data/backtesting") / f"{args.year}_backtest_results.json"
    checked_summary = build_checked_backtest_summary(
        year=args.year,
        baseline_report=baseline,
        naive_report=naive_report,
        overlap_comparison=overlap_comparison,
        reports_dir=str(output_dir),
    )
    write_json(checked_summary_path, checked_summary)

    static_baseline = next(
        (
            item
            for item in reports
            if item.get("base_name") == "baseline" and item.get("learning_mode") == "static"
        ),
        None,
    )
    canonical_segment_breakdown = build_segment_breakdown(baseline.get("race_results", []))
    canonical_error_analysis = build_error_analysis(baseline.get("race_results", []))
    recommended_experiments = [item for item in ranked if item.get("recommended")]
    adaptive_vs_static_comparison = _build_learning_mode_comparison(baseline, static_baseline)
    evaluation_packet = {
        "season": int(args.year),
        "seed": int(args.seed),
        "weather": args.weather,
        "evaluation_mode": args.evaluation_mode,
        "canonical_model": baseline,
        "static_baseline": static_baseline,
        "adaptive_vs_static_comparison": adaptive_vs_static_comparison,
        "naive_previous_race_baseline": naive_report,
        "overlap_comparison": overlap_comparison,
        "canonical_segment_breakdown": canonical_segment_breakdown,
        "canonical_error_analysis": canonical_error_analysis,
        "season_prior_status": season_prior_status,
        "season_prior_mode": effective_season_prior_mode,
        "season_prior_mode_requested": args.season_prior_mode,
        "season_prior_data_dir": str(predictor_data_dir),
        "season_prior_source_data_dir": str(source_predictor_data_dir),
        "recommended_experiments": recommended_experiments,
        "rankings": ranked,
        "experiments": comparison_rows,
        "checked_summary_path": str(checked_summary_path),
    }
    evaluation_packet["reviewer_takeaways"] = _build_reviewer_takeaways(evaluation_packet)
    write_json(output_dir / "evaluation_packet.json", evaluation_packet)
    _emit_review_packet_markdown(output_dir / "REVIEW_PACKET.md", evaluation_packet)

    baseline_test_mae = baseline.get("generalization", {}).get("test", {}).get("race_mae_mean")
    _emit_markdown_recommendations(
        output_path=output_dir / "recommendations.md",
        ranked=ranked,
        baseline_test_mae=baseline_test_mae,
        min_improvement=args.min_test_improvement,
        max_gap=args.max_generalization_gap,
    )
    _plot_error_distribution(output_dir, reports)

    total_evaluated = sum(report["summary"].get("races_evaluated", 0) for report in reports)
    logger.info(f"Backtest complete. Outputs written to {output_dir}")
    if args.strict and total_evaluated == 0:
        logger.error("Strict mode enabled and zero races were evaluated.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
