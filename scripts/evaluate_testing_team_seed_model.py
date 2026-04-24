# ruff: noqa: E402
"""Run the challenger-program research evaluation for model recovery."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.conformal_calibration import (
    build_conformal_calibration_artifact,
    save_conformal_calibration_artifact,
)
from src.models.qualifying_residual_model import (
    build_qualifying_residual_dataset,
    fit_qualifying_residual_model,
    save_qualifying_residual_model,
    summarize_qualifying_residual_dataset,
)
from src.models.race_residual_model import (
    build_race_residual_dataset,
    fit_race_residual_model,
    save_race_residual_model,
    summarize_race_residual_dataset,
)
from src.models.testing_team_seed import (
    build_neutral_team_seed_payload,
    build_prior_year_ranking_seed_payload,
    build_testing_model_team_payload,
)
from src.persistence.artifact_store import ArtifactStore
from src.predictors import Baseline2026Predictor
from src.utils.backtesting import (
    NestedDictConfig,
    aggregate_race_metrics,
    apply_config_overrides,
    build_overlap_comparison,
    get_races_for_year,
    load_config_dict,
    run_previous_race_naive_backtest,
    run_single_race_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_years(raw_years: str) -> tuple[int, ...]:
    """Parse comma-separated season input into a deterministic tuple."""
    years = [int(year_text.strip()) for year_text in raw_years.split(",") if year_text.strip()]
    if not years:
        raise ValueError("At least one year is required.")
    return tuple(years)


def _copy_processed_tree(*, source_data_root: Path, target_data_root: Path) -> None:
    """Copy the processed artifact tree into an isolated evaluation data root."""
    source_processed = source_data_root / "processed"
    target_processed = target_data_root / "processed"
    target_processed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_processed, target_processed, dirs_exist_ok=True)

    learning_state = source_data_root / "learning_state.json"
    if learning_state.exists():
        shutil.copy2(learning_state, target_data_root / "learning_state.json")


def _write_team_payload(*, data_root: Path, year: int, payload: dict[str, Any]) -> Path:
    """Persist one season-scoped team payload under the isolated data root."""
    output_path = (
        data_root / "processed" / "car_characteristics" / f"{year}_car_characteristics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _remove_season_specific_artifacts(*, data_root: Path, year: int) -> None:
    """Remove isolated season-specific priors so the canonical proxy path is tested honestly."""
    for relative_path in (
        Path("processed") / "car_characteristics" / f"{year}_car_characteristics.json",
        Path("processed") / "driver_characteristics" / f"{year}_driver_characteristics.json",
        Path("processed") / "track_characteristics" / f"{year}_track_characteristics.json",
    ):
        target_path = data_root / relative_path
        if target_path.exists():
            target_path.unlink()


def _build_evaluation_predictor(
    *,
    data_root: Path,
    season_year: int,
    seed: int,
    config_path: str,
    config_overrides: dict[str, Any] | None = None,
) -> Baseline2026Predictor:
    """Build one predictor pinned to an isolated data root."""
    config_dict = load_config_dict(config_path)
    if config_overrides:
        config_dict = apply_config_overrides(config_dict, config_overrides)
    predictor = Baseline2026Predictor(
        data_dir=str(data_root / "processed"),
        season_year=season_year,
        seed=seed,
        config=NestedDictConfig(config_dict),
        artifact_store=ArtifactStore(data_root=data_root),
    )
    reset_learning_state = getattr(
        getattr(predictor, "calibration_system", None), "reset_state", None
    )
    if callable(reset_learning_state):
        reset_learning_state(season=season_year)
    return predictor


def _run_compact_backtest(
    *,
    year: int,
    data_root: Path,
    max_races: int | None,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    seed: int,
    config_path: str,
    config_overrides: dict[str, Any] | None = None,
    include_prediction_payloads: bool = False,
    max_consecutive_missing_actuals: int = 3,
) -> dict[str, Any]:
    """Run one backtest packet against an isolated data root."""
    predictor = _build_evaluation_predictor(
        data_root=data_root,
        season_year=year,
        seed=seed,
        config_path=config_path,
        config_overrides=config_overrides,
    )
    races = get_races_for_year(year=year, max_races=max_races)
    race_results: list[dict[str, Any]] = []
    attempted_races: list[str] = []
    consecutive_missing_actuals = 0

    for race_name in races:
        attempted_races.append(race_name)
        result = run_single_race_backtest(
            predictor=predictor,
            year=year,
            race_name=race_name,
            weather=weather,
            qualifying_simulations=qualifying_simulations,
            race_simulations=race_simulations,
            evaluation_mode="historical",
            learning_mode="adaptive",
            include_prediction_payloads=include_prediction_payloads,
        )
        race_results.append(result)
        if result.get("status") == "skipped" and result.get("reason") == "missing_actual_results":
            consecutive_missing_actuals += 1
        else:
            consecutive_missing_actuals = 0

        if (
            max_consecutive_missing_actuals > 0
            and consecutive_missing_actuals >= max_consecutive_missing_actuals
        ):
            logger.warning(
                "Stopping %s backtest after %s consecutive races without actual results. "
                "This usually means FastF1 is rate-limited or its circuit breaker is open.",
                year,
                consecutive_missing_actuals,
            )
            break

    summary = aggregate_race_metrics(race_results)
    naive_report = run_previous_race_naive_backtest(year=year, race_names=attempted_races)
    overlap = build_overlap_comparison(
        model_race_results=race_results,
        naive_race_results=naive_report.get("race_results", []),
    )
    return {
        "year": year,
        "races": attempted_races,
        "summary": summary,
        "naive_summary": naive_report.get("summary", {}),
        "overlap": overlap,
        "race_results": race_results,
    }


def _compare_reports(
    *,
    experimental_report: dict[str, Any],
    baseline_report: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Compare two backtest packets in one compact summary."""
    experimental_summary = experimental_report.get("summary", {})
    baseline_summary = baseline_report.get("summary", {})
    experimental_overlap = experimental_report.get("overlap", {})
    baseline_overlap = baseline_report.get("overlap", {})

    def _delta(summary_key: str, *, lower_is_better: bool) -> float | None:
        experimental_value = experimental_summary.get(summary_key)
        baseline_value = baseline_summary.get(summary_key)
        if experimental_value is None or baseline_value is None:
            return None
        if lower_is_better:
            return float(baseline_value) - float(experimental_value)
        return float(experimental_value) - float(baseline_value)

    race_delta = _delta("race_mae_mean", lower_is_better=True)
    qualifying_delta = _delta("qualifying_mae_mean", lower_is_better=True)
    return {
        "label": label,
        "wins_promotion_gate": bool(
            race_delta is not None
            and qualifying_delta is not None
            and race_delta > 0.0
            and qualifying_delta > 0.0
            and float(experimental_summary.get("race_mae_mean") or 0.0)
            <= float(baseline_summary.get("race_mae_mean") or 0.0) + 0.30
            and float(experimental_summary.get("qualifying_mae_mean") or 0.0)
            <= float(baseline_summary.get("qualifying_mae_mean") or 0.0) + 0.30
        ),
        "experimental_summary": experimental_summary,
        "baseline_summary": baseline_summary,
        "experimental_overlap": experimental_overlap,
        "baseline_overlap": baseline_overlap,
        "deltas": {
            "race_mae_improvement": race_delta,
            "qualifying_mae_improvement": qualifying_delta,
            "top3_accuracy_delta": _delta("top3_accuracy_mean", lower_is_better=False),
            "winner_accuracy_delta": _delta("winner_accuracy_percent", lower_is_better=False),
            "overlap_race_mae_delta": (
                None
                if experimental_overlap.get("race_mae_improvement") is None
                or baseline_overlap.get("race_mae_improvement") is None
                else float(experimental_overlap["race_mae_improvement"])
                - float(baseline_overlap["race_mae_improvement"])
            ),
            "overlap_qualifying_mae_delta": (
                None
                if experimental_overlap.get("qualifying_mae_improvement") is None
                or baseline_overlap.get("qualifying_mae_improvement") is None
                else float(experimental_overlap["qualifying_mae_improvement"])
                - float(baseline_overlap["qualifying_mae_improvement"])
            ),
        },
    }


def _select_worst_weekends(report: dict[str, Any], *, top_n: int = 2) -> list[dict[str, Any]]:
    """Return the worst weekends in a stable, reviewer-friendly shape."""
    scored_rows = [
        row
        for row in report.get("race_results", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    scored_rows.sort(
        key=lambda row: (
            float(row.get("race_mae", 0.0) or 0.0) + float(row.get("qualifying_mae", 0.0) or 0.0)
        ),
        reverse=True,
    )
    return [
        {
            "race_name": row.get("race_name"),
            "qualifying_mae": row.get("qualifying_mae"),
            "race_mae": row.get("race_mae"),
            "track_type": row.get("track_type"),
            "weather": row.get("weather"),
        }
        for row in scored_rows[:top_n]
    ]


def _calibration_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact interval-calibration diagnostics."""
    summary = report.get("summary", {})
    return {
        "qualifying": {
            "coverage": summary.get("qualifying_interval_empirical_coverage"),
            "count": summary.get("qualifying_interval_count"),
            "width_mean": summary.get("qualifying_interval_width_mean"),
        },
        "race": {
            "coverage": summary.get("race_interval_empirical_coverage"),
            "count": summary.get("race_interval_count"),
            "width_mean": summary.get("race_interval_width_mean"),
        },
    }


def _build_conformal_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract conformal calibration rows from one backtest report."""

    def _actual_positions_by_driver(rows: list[dict[str, Any]]) -> dict[str, int]:
        """Return actual finishing positions keyed by driver code."""
        positions: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("driver"):
                continue
            try:
                positions[str(row["driver"])] = int(row["position"])
            except (TypeError, ValueError):
                continue
        return positions

    calibration_rows: list[dict[str, Any]] = []
    for race_row in report.get("race_results", []):
        if not isinstance(race_row, dict) or race_row.get("status") != "ok":
            continue

        qualifying_regime = str(race_row.get("qualifying_regime", "")).strip().lower()
        qualifying_actual_by_driver = _actual_positions_by_driver(
            race_row.get("qualifying_actual_rows", [])
        )
        for predicted_row in race_row.get("qualifying_prediction_rows", []):
            if not isinstance(predicted_row, dict) or not predicted_row.get("driver"):
                continue
            try:
                center = int(predicted_row.get("median_position", predicted_row.get("position")))
                actual_position = int(qualifying_actual_by_driver[str(predicted_row["driver"])])
                p5 = int(predicted_row.get("p5", center))
                p95 = int(predicted_row.get("p95", center))
            except (KeyError, TypeError, ValueError):
                continue
            calibration_rows.append(
                {
                    "session": "qualifying",
                    "regime": qualifying_regime or "model_only",
                    "residual": abs(center - actual_position),
                    "covered": min(p5, p95) <= actual_position <= max(p5, p95),
                }
            )

        race_regime = str(race_row.get("race_regime", "")).strip().lower()
        race_actual_by_driver = _actual_positions_by_driver(race_row.get("race_actual_rows", []))
        for predicted_row in race_row.get("race_prediction_rows", []):
            if not isinstance(predicted_row, dict) or not predicted_row.get("driver"):
                continue
            try:
                center = int(predicted_row.get("median_position", predicted_row.get("position")))
                actual_position = int(race_actual_by_driver[str(predicted_row["driver"])])
                p5 = int(predicted_row.get("p5", center))
                p95 = int(predicted_row.get("p95", center))
            except (KeyError, TypeError, ValueError):
                continue
            calibration_rows.append(
                {
                    "session": "race",
                    "regime": race_regime or "model_only",
                    "residual": abs(center - actual_position),
                    "covered": min(p5, p95) <= actual_position <= max(p5, p95),
                }
            )

    return calibration_rows


def _build_challenger_overrides(
    *,
    qualifying_artifact_path: Path | None,
    race_artifact_path: Path | None,
    conformal_artifact_path: Path | None,
) -> dict[str, Any]:
    """Build config overrides for the challenger stack."""
    overrides: dict[str, Any] = {
        "baseline_predictor.qualifying.qualifying_residual_model.enabled": bool(
            qualifying_artifact_path and qualifying_artifact_path.exists()
        ),
        "baseline_predictor.race.race_residual_model.enabled": bool(
            race_artifact_path and race_artifact_path.exists()
        ),
        "baseline_predictor.conformal_calibration.enabled": bool(
            conformal_artifact_path and conformal_artifact_path.exists()
        ),
    }
    if qualifying_artifact_path is not None:
        overrides["baseline_predictor.qualifying.qualifying_residual_model.artifact_path"] = str(
            qualifying_artifact_path
        )
    if race_artifact_path is not None:
        overrides["baseline_predictor.race.race_residual_model.artifact_path"] = str(
            race_artifact_path
        )
    if conformal_artifact_path is not None:
        overrides["baseline_predictor.conformal_calibration.artifact_path"] = str(
            conformal_artifact_path
        )
    return overrides


def _build_component_overrides(
    *,
    qualifying_artifact_path: Path | None,
    race_artifact_path: Path | None,
    conformal_artifact_path: Path | None,
    use_qualifying_residual: bool,
    use_race_residual: bool,
    use_conformal: bool,
    allow_residuals_with_testing_seed: bool = False,
) -> dict[str, Any]:
    """Build config overrides for one ablation variant."""
    overrides = _build_challenger_overrides(
        qualifying_artifact_path=qualifying_artifact_path if use_qualifying_residual else None,
        race_artifact_path=race_artifact_path if use_race_residual else None,
        conformal_artifact_path=conformal_artifact_path if use_conformal else None,
    )
    overrides["baseline_predictor.qualifying.qualifying_residual_model.enabled"] = bool(
        use_qualifying_residual and qualifying_artifact_path and qualifying_artifact_path.exists()
    )
    overrides["baseline_predictor.race.race_residual_model.enabled"] = bool(
        use_race_residual and race_artifact_path and race_artifact_path.exists()
    )
    overrides["baseline_predictor.conformal_calibration.enabled"] = bool(
        use_conformal and conformal_artifact_path and conformal_artifact_path.exists()
    )
    overrides["baseline_predictor.qualifying.qualifying_residual_model.allow_with_testing_seed"] = (
        bool(allow_residuals_with_testing_seed)
    )
    overrides["baseline_predictor.race.race_residual_model.allow_with_testing_seed"] = bool(
        allow_residuals_with_testing_seed
    )
    return overrides


def _ablation_variants(
    *,
    qualifying_artifact_path: Path | None,
    race_artifact_path: Path | None,
    conformal_artifact_path: Path | None,
) -> list[dict[str, Any]]:
    """Return the standard component-isolation variants."""
    return [
        {
            "label": "testing_seed_only",
            "uses_testing_seed": True,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=False,
                use_race_residual=False,
                use_conformal=False,
            ),
        },
        {
            "label": "qualifying_residual_only",
            "uses_testing_seed": False,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=True,
                use_race_residual=False,
                use_conformal=False,
            ),
        },
        {
            "label": "race_residual_only",
            "uses_testing_seed": False,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=False,
                use_race_residual=True,
                use_conformal=False,
            ),
        },
        {
            "label": "conformal_only",
            "uses_testing_seed": False,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=False,
                use_race_residual=False,
                use_conformal=True,
            ),
        },
        {
            "label": "testing_seed_plus_residuals",
            "uses_testing_seed": True,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=True,
                use_race_residual=True,
                use_conformal=False,
                allow_residuals_with_testing_seed=True,
            ),
        },
        {
            "label": "full_challenger",
            "uses_testing_seed": True,
            "overrides": _build_component_overrides(
                qualifying_artifact_path=qualifying_artifact_path,
                race_artifact_path=race_artifact_path,
                conformal_artifact_path=conformal_artifact_path,
                use_qualifying_residual=True,
                use_race_residual=True,
                use_conformal=True,
            ),
        },
    ]


def _race_delta_summary(
    *,
    experimental_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    """Count race-level wins and losses against a baseline report."""
    baseline_by_race = {
        row.get("race_name"): row
        for row in baseline_report.get("race_results", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for experimental_row in experimental_report.get("race_results", []):
        if not isinstance(experimental_row, dict) or experimental_row.get("status") != "ok":
            continue
        race_name = experimental_row.get("race_name")
        baseline_row = baseline_by_race.get(race_name)
        if not isinstance(baseline_row, dict):
            continue
        qualifying_mae = experimental_row.get("qualifying_mae")
        baseline_qualifying_mae = baseline_row.get("qualifying_mae")
        race_mae = experimental_row.get("race_mae")
        baseline_race_mae = baseline_row.get("race_mae")
        if (
            qualifying_mae is None
            or baseline_qualifying_mae is None
            or race_mae is None
            or baseline_race_mae is None
        ):
            continue
        rows.append(
            {
                "race_name": race_name,
                "qualifying_delta": float(qualifying_mae) - float(baseline_qualifying_mae),
                "race_delta": float(race_mae) - float(baseline_race_mae),
            }
        )

    if not rows:
        return {
            "races_compared": 0,
            "qualifying_worse_count": 0,
            "qualifying_better_count": 0,
            "race_worse_count": 0,
            "race_better_count": 0,
            "mean_qualifying_delta": None,
            "mean_race_delta": None,
            "worst_race_deltas": [],
        }

    return {
        "races_compared": len(rows),
        "qualifying_worse_count": sum(1 for row in rows if row["qualifying_delta"] > 0.0),
        "qualifying_better_count": sum(1 for row in rows if row["qualifying_delta"] < 0.0),
        "race_worse_count": sum(1 for row in rows if row["race_delta"] > 0.0),
        "race_better_count": sum(1 for row in rows if row["race_delta"] < 0.0),
        "mean_qualifying_delta": sum(row["qualifying_delta"] for row in rows) / len(rows),
        "mean_race_delta": sum(row["race_delta"] for row in rows) / len(rows),
        "worst_race_deltas": sorted(rows, key=lambda row: row["race_delta"], reverse=True)[:5],
    }


def _build_model_artifacts(
    *,
    years: tuple[int, ...],
    output_dir: Path,
    config_path: str,
    seed: int,
    max_races: int,
) -> dict[str, Any]:
    """Build residual-model artifacts for the challenger stack."""

    def _build_optional_residual_artifact(
        *,
        label: str,
        dataset_builder: Callable[[], Any],
        dataset_summarizer: Callable[[Any], dict[str, Any]],
        model_fitter: Callable[..., Any],
        model_saver: Callable[..., Any],
        artifact_path: Path,
        summary_path: Path,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one residual artifact when training data is available."""
        dataset = dataset_builder()
        dataset_summary = dataset_summarizer(dataset)
        packet = {
            "artifact_path": str(artifact_path),
            "summary_path": str(summary_path),
            "enabled": False,
            "model_summary": None,
            "dataset_summary": dataset_summary,
            "error": None,
        }
        if bool(getattr(dataset, "empty", False)):
            packet["error"] = (
                f"No training rows were available for the {label} residual model. "
                "The challenger stack will skip this artifact."
            )
            logger.warning("%s", packet["error"])
            return packet

        try:
            model = model_fitter(dataset, **(fit_kwargs or {}))
        except ValueError as exc:
            packet["error"] = str(exc)
            logger.warning("Could not fit %s residual model: %s", label, exc)
            return packet

        model_saver(
            model=model,
            artifact_path=artifact_path,
            summary_path=summary_path,
        )
        packet["enabled"] = True
        packet["model_summary"] = model.summary()
        return packet

    artifact_root = output_dir / "model_artifacts"
    qualifying_artifact_path = (
        artifact_root / "qualifying_residual" / "qualifying_residual_model.pkl"
    )
    qualifying_summary_path = (
        artifact_root / "qualifying_residual" / "qualifying_residual_model.summary.json"
    )
    race_artifact_path = artifact_root / "race_residual" / "race_residual_model.pkl"
    race_summary_path = artifact_root / "race_residual" / "race_residual_model.summary.json"

    qualifying_packet = _build_optional_residual_artifact(
        label="qualifying",
        dataset_builder=lambda: build_qualifying_residual_dataset(
            years=list(years),
            max_races=max_races,
            config_path=config_path,
            seed=seed,
        ),
        dataset_summarizer=summarize_qualifying_residual_dataset,
        model_fitter=fit_qualifying_residual_model,
        model_saver=save_qualifying_residual_model,
        artifact_path=qualifying_artifact_path,
        summary_path=qualifying_summary_path,
    )
    race_packet = _build_optional_residual_artifact(
        label="race",
        dataset_builder=lambda: build_race_residual_dataset(
            years=list(years),
            max_races=max_races,
            config_path=config_path,
            seed=seed,
        ),
        dataset_summarizer=summarize_race_residual_dataset,
        model_fitter=fit_race_residual_model,
        model_saver=save_race_residual_model,
        artifact_path=race_artifact_path,
        summary_path=race_summary_path,
    )
    return {
        "qualifying": qualifying_packet,
        "race": race_packet,
    }


def _evaluate_team_seed_holdouts(
    *,
    years: tuple[int, ...],
    repo_data_root: Path,
    output_dir: Path,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    seed: int,
    config_path: str,
    max_races: int,
    max_consecutive_missing_actuals: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate ranking, neutral, and testing-model team seeds on historical holdouts."""
    holdout_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []

    for holdout_year in years:
        training_years = tuple(year for year in years if year != holdout_year)
        year_output_dir = output_dir / "holdouts" / str(holdout_year)
        reports_by_seed: dict[str, dict[str, Any]] = {}
        payload_builders = {
            "ranking": build_prior_year_ranking_seed_payload(
                target_year=holdout_year,
                source_year=holdout_year - 1,
            ),
            "neutral": build_neutral_team_seed_payload(target_year=holdout_year),
            "testing_model": build_testing_model_team_payload(
                target_year=holdout_year,
                training_years=training_years,
            ),
        }

        for seed_mode, payload in payload_builders.items():
            data_root = year_output_dir / f"{seed_mode}_data"
            _copy_processed_tree(source_data_root=repo_data_root, target_data_root=data_root)
            payload_path = _write_team_payload(
                data_root=data_root,
                year=holdout_year,
                payload=payload,
            )
            logger.info("Prepared %s %s payload at %s", holdout_year, seed_mode, payload_path)
            report = _run_compact_backtest(
                year=holdout_year,
                data_root=data_root,
                max_races=max_races,
                weather=weather,
                qualifying_simulations=qualifying_simulations,
                race_simulations=race_simulations,
                seed=seed,
                config_path=config_path,
                include_prediction_payloads=(seed_mode == "testing_model"),
                max_consecutive_missing_actuals=max_consecutive_missing_actuals,
            )
            reports_by_seed[seed_mode] = report
            (year_output_dir / f"{seed_mode}_report.json").write_text(json.dumps(report, indent=2))

        testing_vs_ranking = _compare_reports(
            experimental_report=reports_by_seed["testing_model"],
            baseline_report=reports_by_seed["ranking"],
            label="testing_model_vs_ranking",
        )
        testing_vs_neutral = _compare_reports(
            experimental_report=reports_by_seed["testing_model"],
            baseline_report=reports_by_seed["neutral"],
            label="testing_model_vs_neutral",
        )
        comparison = {
            "year": int(holdout_year),
            "reports": {
                seed_mode: {
                    "summary": report.get("summary", {}),
                    "overlap": report.get("overlap", {}),
                    "worst_weekends": _select_worst_weekends(report),
                    "calibration": _calibration_summary(report),
                }
                for seed_mode, report in reports_by_seed.items()
            },
            "comparisons": {
                "testing_vs_ranking": testing_vs_ranking,
                "testing_vs_neutral": testing_vs_neutral,
            },
        }
        holdout_rows.append(comparison)
        (year_output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
        conformal_rows.extend(_build_conformal_rows_from_report(reports_by_seed["testing_model"]))

    summary = {
        "years": list(years),
        "comparisons": holdout_rows,
        "promotion_gate_passes": [
            row["year"]
            for row in holdout_rows
            if row["comparisons"]["testing_vs_ranking"]["wins_promotion_gate"]
        ],
    }
    return summary, conformal_rows


def _run_season_champion_vs_challenger(
    *,
    season_year: int,
    training_years: tuple[int, ...],
    repo_data_root: Path,
    output_dir: Path,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    seed: int,
    config_path: str,
    challenger_overrides: dict[str, Any],
    max_races: int | None = None,
    max_consecutive_missing_actuals: int = 3,
) -> dict[str, Any]:
    """Run a full-season champion-vs-challenger comparison on one isolated year."""
    champion_data_root = output_dir / "champion_data"
    challenger_data_root = output_dir / "challenger_data"
    _copy_processed_tree(source_data_root=repo_data_root, target_data_root=champion_data_root)
    _copy_processed_tree(source_data_root=repo_data_root, target_data_root=challenger_data_root)
    _remove_season_specific_artifacts(data_root=champion_data_root, year=season_year)
    _remove_season_specific_artifacts(data_root=challenger_data_root, year=season_year)

    challenger_payload = build_testing_model_team_payload(
        target_year=season_year,
        training_years=training_years,
    )
    _write_team_payload(
        data_root=challenger_data_root, year=season_year, payload=challenger_payload
    )

    champion_report = _run_compact_backtest(
        year=season_year,
        data_root=champion_data_root,
        max_races=max_races,
        weather=weather,
        qualifying_simulations=qualifying_simulations,
        race_simulations=race_simulations,
        seed=seed,
        config_path=config_path,
        include_prediction_payloads=True,
        max_consecutive_missing_actuals=max_consecutive_missing_actuals,
        config_overrides={
            "baseline_predictor.qualifying.qualifying_residual_model.enabled": False,
            "baseline_predictor.race.race_residual_model.enabled": False,
            "baseline_predictor.conformal_calibration.enabled": False,
        },
    )
    challenger_report = _run_compact_backtest(
        year=season_year,
        data_root=challenger_data_root,
        max_races=max_races,
        weather=weather,
        qualifying_simulations=qualifying_simulations,
        race_simulations=race_simulations,
        seed=seed,
        config_path=config_path,
        include_prediction_payloads=True,
        config_overrides=challenger_overrides,
        max_consecutive_missing_actuals=max_consecutive_missing_actuals,
    )

    comparison = _compare_reports(
        experimental_report=challenger_report,
        baseline_report=champion_report,
        label=f"{season_year}_challenger_vs_champion",
    )
    packet = {
        "season_year": int(season_year),
        "champion": {
            "summary": champion_report.get("summary", {}),
            "overlap": champion_report.get("overlap", {}),
            "worst_weekends": _select_worst_weekends(champion_report, top_n=3),
            "calibration": _calibration_summary(champion_report),
        },
        "challenger": {
            "summary": challenger_report.get("summary", {}),
            "overlap": challenger_report.get("overlap", {}),
            "worst_weekends": _select_worst_weekends(challenger_report, top_n=3),
            "calibration": _calibration_summary(challenger_report),
        },
        "comparison": comparison,
    }
    (output_dir / "champion_report.json").write_text(json.dumps(champion_report, indent=2))
    (output_dir / "challenger_report.json").write_text(json.dumps(challenger_report, indent=2))
    (output_dir / "comparison.json").write_text(json.dumps(packet, indent=2))
    return packet


def _run_component_ablation(
    *,
    season_year: int,
    training_years: tuple[int, ...],
    repo_data_root: Path,
    output_dir: Path,
    weather: str,
    qualifying_simulations: int,
    race_simulations: int,
    seed: int,
    config_path: str,
    qualifying_artifact_path: Path | None,
    race_artifact_path: Path | None,
    conformal_artifact_path: Path | None,
    max_races: int | None = None,
    max_consecutive_missing_actuals: int = 3,
) -> dict[str, Any]:
    """Run a one-season ablation matrix against the champion baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    champion_data_root = output_dir / "champion_data"
    _copy_processed_tree(source_data_root=repo_data_root, target_data_root=champion_data_root)
    _remove_season_specific_artifacts(data_root=champion_data_root, year=season_year)

    champion_report = _run_compact_backtest(
        year=season_year,
        data_root=champion_data_root,
        max_races=max_races,
        weather=weather,
        qualifying_simulations=qualifying_simulations,
        race_simulations=race_simulations,
        seed=seed,
        config_path=config_path,
        include_prediction_payloads=True,
        max_consecutive_missing_actuals=max_consecutive_missing_actuals,
        config_overrides={
            "baseline_predictor.qualifying.qualifying_residual_model.enabled": False,
            "baseline_predictor.race.race_residual_model.enabled": False,
            "baseline_predictor.conformal_calibration.enabled": False,
        },
    )
    (output_dir / "champion_report.json").write_text(json.dumps(champion_report, indent=2))

    variant_packets: list[dict[str, Any]] = []
    testing_payload = build_testing_model_team_payload(
        target_year=season_year,
        training_years=training_years,
    )
    for variant in _ablation_variants(
        qualifying_artifact_path=qualifying_artifact_path,
        race_artifact_path=race_artifact_path,
        conformal_artifact_path=conformal_artifact_path,
    ):
        label = str(variant["label"])
        data_root = output_dir / f"{label}_data"
        _copy_processed_tree(source_data_root=repo_data_root, target_data_root=data_root)
        _remove_season_specific_artifacts(data_root=data_root, year=season_year)
        if variant["uses_testing_seed"]:
            _write_team_payload(data_root=data_root, year=season_year, payload=testing_payload)

        report = _run_compact_backtest(
            year=season_year,
            data_root=data_root,
            max_races=max_races,
            weather=weather,
            qualifying_simulations=qualifying_simulations,
            race_simulations=race_simulations,
            seed=seed,
            config_path=config_path,
            include_prediction_payloads=True,
            config_overrides=variant["overrides"],
            max_consecutive_missing_actuals=max_consecutive_missing_actuals,
        )
        comparison = _compare_reports(
            experimental_report=report,
            baseline_report=champion_report,
            label=f"{season_year}_{label}_vs_champion",
        )
        packet = {
            "label": label,
            "uses_testing_seed": bool(variant["uses_testing_seed"]),
            "overrides": variant["overrides"],
            "summary": report.get("summary", {}),
            "overlap": report.get("overlap", {}),
            "worst_weekends": _select_worst_weekends(report, top_n=3),
            "calibration": _calibration_summary(report),
            "comparison": comparison,
            "race_delta_summary": _race_delta_summary(
                experimental_report=report,
                baseline_report=champion_report,
            ),
        }
        variant_packets.append(packet)
        (output_dir / f"{label}_report.json").write_text(json.dumps(report, indent=2))

    ablation = {
        "season_year": int(season_year),
        "training_years": list(training_years),
        "champion": {
            "summary": champion_report.get("summary", {}),
            "overlap": champion_report.get("overlap", {}),
            "worst_weekends": _select_worst_weekends(champion_report, top_n=3),
            "calibration": _calibration_summary(champion_report),
        },
        "variants": variant_packets,
    }
    (output_dir / "ablation.json").write_text(json.dumps(ablation, indent=2))
    _write_ablation_markdown(ablation=ablation, output_path=output_dir / "ablation.md")
    return ablation


def _write_ablation_markdown(*, ablation: dict[str, Any], output_path: Path) -> None:
    """Write a compact markdown table for one ablation packet."""
    lines = [
        f"# Component Ablation {ablation.get('season_year')}",
        "",
        "| Variant | Race MAE delta | Qualifying MAE delta | Top-3 delta | Winner delta | Race worse/better | Qualifying worse/better |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ablation.get("variants", []):
        comparison = variant.get("comparison", {})
        deltas = comparison.get("deltas", {})
        race_delta_summary = variant.get("race_delta_summary", {})
        lines.append(
            "| {label} | {race_delta} | {qualifying_delta} | {top3_delta} | "
            "{winner_delta} | {race_worse}/{race_better} | {qualifying_worse}/{qualifying_better} |".format(
                label=variant.get("label"),
                race_delta=deltas.get("race_mae_improvement"),
                qualifying_delta=deltas.get("qualifying_mae_improvement"),
                top3_delta=deltas.get("top3_accuracy_delta"),
                winner_delta=deltas.get("winner_accuracy_delta"),
                race_worse=race_delta_summary.get("race_worse_count"),
                race_better=race_delta_summary.get("race_better_count"),
                qualifying_worse=race_delta_summary.get("qualifying_worse_count"),
                qualifying_better=race_delta_summary.get("qualifying_better_count"),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _write_markdown_summary(*, comparisons: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact reviewer-facing markdown summary."""
    lines = [
        "# Model Recovery Research",
        "",
        "Champion-vs-challenger summary across holdouts and season evaluation.",
        "",
    ]

    for comparison in comparisons:
        label = comparison.get("label") or comparison.get("year")
        deltas = comparison.get("deltas", {})
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Promotion gate: `{comparison.get('wins_promotion_gate')}`",
                f"- Race MAE improvement: `{deltas.get('race_mae_improvement')}`",
                f"- Qualifying MAE improvement: `{deltas.get('qualifying_mae_improvement')}`",
                f"- Top-3 accuracy delta: `{deltas.get('top3_accuracy_delta')}`",
                f"- Winner accuracy delta: `{deltas.get('winner_accuracy_delta')}`",
                f"- Overlap race delta: `{deltas.get('overlap_race_mae_delta')}`",
                f"- Overlap qualifying delta: `{deltas.get('overlap_qualifying_mae_delta')}`",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk with a clear error when it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot stitch existing evaluation outputs; missing {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return payload


def _existing_artifact_packet(*, output_dir: Path, label: str) -> dict[str, Any]:
    """Build a JSON-safe artifact packet from files that already exist."""
    artifact_path = (
        output_dir / "model_artifacts" / f"{label}_residual" / f"{label}_residual_model.pkl"
    )
    summary_path = (
        output_dir
        / "model_artifacts"
        / f"{label}_residual"
        / f"{label}_residual_model.summary.json"
    )
    summary_payload = _read_json_file(summary_path) if summary_path.exists() else None
    return {
        "artifact_path": str(artifact_path),
        "summary_path": str(summary_path),
        "enabled": artifact_path.exists(),
        "model_summary": summary_payload,
        "dataset_summary": None,
        "error": None
        if artifact_path.exists()
        else f"No existing {label} residual artifact found.",
    }


def _stitch_existing_outputs(
    *,
    output_dir: Path,
    season_year: int,
    live_year: int,
) -> dict[str, Any]:
    """Write top-level summaries from already completed child reports."""
    holdout_summary = _read_json_file(output_dir / "holdout_summary.json")
    season_packet = _read_json_file(output_dir / f"season_{season_year}" / "comparison.json")
    live_packet = _read_json_file(output_dir / f"live_{live_year}" / "comparison.json")
    conformal_path = (
        output_dir / "model_artifacts" / "conformal_calibration" / "conformal_calibration.json"
    )
    conformal_payload = _read_json_file(conformal_path) if conformal_path.exists() else {}

    artifacts = {
        "qualifying": _existing_artifact_packet(output_dir=output_dir, label="qualifying"),
        "race": _existing_artifact_packet(output_dir=output_dir, label="race"),
    }
    summary = {
        "holdouts": holdout_summary,
        "artifacts": artifacts,
        "conformal": conformal_payload,
        "season_evaluation": season_packet,
        "live_observational": live_packet,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    markdown_rows = [
        *[
            {
                **row["comparisons"]["testing_vs_ranking"],
                "label": f"holdout_{row['year']}",
            }
            for row in holdout_summary.get("comparisons", [])
            if isinstance(row, dict) and "comparisons" in row
        ],
        season_packet["comparison"],
        live_packet["comparison"],
    ]
    _write_markdown_summary(comparisons=markdown_rows, output_path=output_dir / "summary.md")
    return summary


def _load_existing_holdouts(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load completed holdout summary when resuming an interrupted evaluation."""
    return _read_json_file(output_dir / "holdout_summary.json"), []


def _load_existing_artifacts(output_dir: Path) -> dict[str, Any]:
    """Load residual artifact metadata without rebuilding training datasets."""
    return {
        "qualifying": _existing_artifact_packet(output_dir=output_dir, label="qualifying"),
        "race": _existing_artifact_packet(output_dir=output_dir, label="race"),
    }


def _load_existing_conformal_artifact(output_dir: Path) -> tuple[Any, Path | None]:
    """Load an existing conformal artifact or report that none is available."""
    conformal_path = (
        output_dir / "model_artifacts" / "conformal_calibration" / "conformal_calibration.json"
    )
    if not conformal_path.exists():
        return {}, None
    return _read_json_file(conformal_path), conformal_path


def main() -> int:
    """Run the principal-level challenger evaluation program."""
    parser = argparse.ArgumentParser(
        description="Run champion-vs-challenger model-recovery evaluation."
    )
    parser.add_argument("--years", type=str, default="2022,2023,2024")
    parser.add_argument("--artifact-training-years", type=str, default="2023,2024")
    parser.add_argument("--live-training-years", type=str, default="2022,2023,2024")
    parser.add_argument("--season-year", type=int, default=2022)
    parser.add_argument("--live-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="reports/testing_team_seed_model")
    parser.add_argument("--max-races", type=int, default=6)
    parser.add_argument("--season-max-races", type=int, default=None)
    parser.add_argument("--live-max-races", type=int, default=None)
    parser.add_argument(
        "--weather", type=str, default="dry", choices=["dry", "rain", "mixed", "wet"]
    )
    parser.add_argument("--quali-sims", type=int, default=120)
    parser.add_argument("--race-sims", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--max-consecutive-missing-actuals",
        type=int,
        default=3,
        help=(
            "Stop a compact backtest after this many consecutive races with missing actuals. "
            "Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--stitch-existing",
        action="store_true",
        help="Rebuild summary.json and summary.md from existing child reports without rerunning.",
    )
    parser.add_argument(
        "--reuse-existing-holdouts",
        action="store_true",
        help="Read holdout_summary.json instead of rerunning holdout backtests.",
    )
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help="Reuse residual and conformal model artifacts already under output-dir.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live-year observational evaluation and reuse its existing comparison if present.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run component ablations for the season and live evaluations.",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Reuse existing season/live comparisons and only run requested ablation matrices.",
    )
    parser.add_argument(
        "--ablation-target",
        type=str,
        default="both",
        choices=["season", "live", "both"],
        help="Choose which evaluation target receives the component ablation matrix.",
    )
    args = parser.parse_args()

    holdout_years = _parse_years(args.years)
    artifact_training_years = _parse_years(args.artifact_training_years)
    live_training_years = _parse_years(args.live_training_years)
    repo_data_root = Path("data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stitch_existing:
        _stitch_existing_outputs(
            output_dir=output_dir,
            season_year=args.season_year,
            live_year=args.live_year,
        )
        logger.info("Stitched existing research outputs under %s", output_dir)
        return 0

    if args.reuse_existing_holdouts:
        holdout_summary, conformal_rows = _load_existing_holdouts(output_dir)
    else:
        holdout_summary, conformal_rows = _evaluate_team_seed_holdouts(
            years=holdout_years,
            repo_data_root=repo_data_root,
            output_dir=output_dir,
            weather=args.weather,
            qualifying_simulations=args.quali_sims,
            race_simulations=args.race_sims,
            seed=args.seed,
            config_path=args.config,
            max_races=args.max_races,
            max_consecutive_missing_actuals=args.max_consecutive_missing_actuals,
        )
        (output_dir / "holdout_summary.json").write_text(json.dumps(holdout_summary, indent=2))

    artifacts = (
        _load_existing_artifacts(output_dir)
        if args.reuse_existing_artifacts
        else _build_model_artifacts(
            years=artifact_training_years,
            output_dir=output_dir,
            config_path=args.config,
            seed=args.seed,
            max_races=args.max_races,
        )
    )
    qualifying_artifact_path = Path(artifacts["qualifying"]["artifact_path"])
    race_artifact_path = Path(artifacts["race"]["artifact_path"])

    conformal_summary: dict[str, Any]
    if args.reuse_existing_artifacts:
        conformal_summary, conformal_artifact_path = _load_existing_conformal_artifact(output_dir)
    else:
        conformal_artifact = build_conformal_calibration_artifact(
            rows=conformal_rows, min_samples=10
        )
        conformal_artifact_path = (
            output_dir / "model_artifacts" / "conformal_calibration" / "conformal_calibration.json"
        )
        save_conformal_calibration_artifact(
            artifact=conformal_artifact,
            path=conformal_artifact_path,
        )
        conformal_summary = conformal_artifact.to_dict()

    challenger_overrides = _build_challenger_overrides(
        qualifying_artifact_path=qualifying_artifact_path,
        race_artifact_path=race_artifact_path,
        conformal_artifact_path=conformal_artifact_path,
    )

    season_packet = (
        _read_json_file(output_dir / f"season_{args.season_year}" / "comparison.json")
        if args.ablation_only
        else _run_season_champion_vs_challenger(
            season_year=args.season_year,
            training_years=artifact_training_years,
            repo_data_root=repo_data_root,
            output_dir=output_dir / f"season_{args.season_year}",
            weather=args.weather,
            qualifying_simulations=args.quali_sims,
            race_simulations=args.race_sims,
            seed=args.seed,
            config_path=args.config,
            challenger_overrides=challenger_overrides,
            max_races=args.season_max_races,
            max_consecutive_missing_actuals=args.max_consecutive_missing_actuals,
        )
    )

    live_packet = (
        _read_json_file(output_dir / f"live_{args.live_year}" / "comparison.json")
        if args.skip_live or args.ablation_only
        else _run_season_champion_vs_challenger(
            season_year=args.live_year,
            training_years=live_training_years,
            repo_data_root=repo_data_root,
            output_dir=output_dir / f"live_{args.live_year}",
            weather=args.weather,
            qualifying_simulations=args.quali_sims,
            race_simulations=args.race_sims,
            seed=args.seed,
            config_path=args.config,
            challenger_overrides=challenger_overrides,
            max_races=args.live_max_races,
            max_consecutive_missing_actuals=args.max_consecutive_missing_actuals,
        )
    )

    ablations: dict[str, Any] = {}
    if args.run_ablation and args.ablation_target in {"season", "both"}:
        ablations[f"season_{args.season_year}"] = _run_component_ablation(
            season_year=args.season_year,
            training_years=artifact_training_years,
            repo_data_root=repo_data_root,
            output_dir=output_dir / f"season_{args.season_year}" / "ablation",
            weather=args.weather,
            qualifying_simulations=args.quali_sims,
            race_simulations=args.race_sims,
            seed=args.seed,
            config_path=args.config,
            qualifying_artifact_path=qualifying_artifact_path,
            race_artifact_path=race_artifact_path,
            conformal_artifact_path=conformal_artifact_path,
            max_races=args.season_max_races,
            max_consecutive_missing_actuals=args.max_consecutive_missing_actuals,
        )
    if args.run_ablation and args.ablation_target in {"live", "both"}:
        ablations[f"live_{args.live_year}"] = _run_component_ablation(
            season_year=args.live_year,
            training_years=live_training_years,
            repo_data_root=repo_data_root,
            output_dir=output_dir / f"live_{args.live_year}" / "ablation",
            weather=args.weather,
            qualifying_simulations=args.quali_sims,
            race_simulations=args.race_sims,
            seed=args.seed,
            config_path=args.config,
            qualifying_artifact_path=qualifying_artifact_path,
            race_artifact_path=race_artifact_path,
            conformal_artifact_path=conformal_artifact_path,
            max_races=args.live_max_races,
            max_consecutive_missing_actuals=args.max_consecutive_missing_actuals,
        )

    summary = {
        "holdouts": holdout_summary,
        "artifacts": artifacts,
        "conformal": conformal_summary,
        "season_evaluation": season_packet,
        "live_observational": live_packet,
    }
    if ablations:
        summary["ablations"] = ablations
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    markdown_rows = [
        *[
            {
                **row["comparisons"]["testing_vs_ranking"],
                "label": f"holdout_{row['year']}",
            }
            for row in holdout_summary["comparisons"]
        ],
        season_packet["comparison"],
        live_packet["comparison"],
    ]
    _write_markdown_summary(comparisons=markdown_rows, output_path=output_dir / "summary.md")
    logger.info("Wrote research outputs under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
