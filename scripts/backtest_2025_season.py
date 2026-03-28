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
import sys
from pathlib import Path
from typing import Any

import fastf1

# Add project root to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictors import Baseline2026Predictor
from src.utils.backtesting import (
    NestedDictConfig,
    aggregate_race_metrics,
    apply_config_overrides,
    get_races_for_year,
    load_config_dict,
    parse_experiment_spec,
    rank_experiments_for_generalization,
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

    reports: list[dict[str, Any]] = []
    for name, overrides in experiment_specs:
        logger.info(f"Experiment '{name}' overrides={overrides}")
        merged_config = apply_config_overrides(base_config, overrides)
        predictor = Baseline2026Predictor(seed=args.seed, config=NestedDictConfig(merged_config))

        race_results: list[dict[str, Any]] = []
        for index, race_name in enumerate(races, start=1):
            row = run_single_race_backtest(
                predictor=predictor,
                year=args.year,
                race_name=race_name,
                weather=args.weather,
                qualifying_simulations=args.quali_sims,
                race_simulations=args.race_sims,
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

    ranked = rank_experiments_for_generalization(
        reports,
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

    baseline = next((item for item in reports if item["name"] == "baseline"), reports[0])
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
