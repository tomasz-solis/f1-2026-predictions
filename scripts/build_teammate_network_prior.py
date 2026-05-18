"""Fit teammate-network race and qualifying driver priors.

This is the Phase 6 builder. It consumes the Phase 5 aggregate matched-lap
observations, fits separate dry race and dry qualifying teammate networks,
evaluates the locked source-backed checks, and writes the versioned prior
artifact used by later migration phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.matched_laps import MatchedLapConfig  # noqa: E402

SessionKind = Literal["race", "qualifying"]
NetworkKey = Literal["race_network", "quali_network"]


@dataclass(frozen=True)
class PriorFitConfig:
    """Configuration for the teammate-network prior fit."""

    historical_start: int = 2022
    historical_end: int = 2025
    bootstrap_replicates: int = 1000
    bootstrap_random_seed: int = 2026
    race_sigma_floor_s: float = 0.05
    quali_sigma_floor_s: float = 0.10
    min_driver_observations: int = 24
    effective_n_cap: int = 32
    main_min_observation_share: float = 0.90
    main_min_driver_share: float = 0.80


@dataclass(frozen=True)
class ValidationCheck:
    """One source-backed validation or supplemental check."""

    check_id: str
    network_key: NetworkKey
    faster_driver: str
    slower_driver: str
    threshold_s: float
    source: str
    source_type: str
    scope: str
    tier: str = "HARD"


HARD_VALIDATION_CHECKS: tuple[ValidationCheck, ...] = ()


CONTEXT_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck(
        check_id="verstappen_perez_race_2022",
        network_key="race_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.234,
        source="Motorsport.com / PACETEQ Perez trend",
        source_type="teammate race-pace delta",
        scope="Red Bull race pace, 2022 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="verstappen_perez_race_2023",
        network_key="race_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.451,
        source="Motorsport.com / PACETEQ 2023 review",
        source_type="teammate race-pace delta",
        scope="Red Bull race pace, 2023 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="verstappen_perez_race_2024",
        network_key="race_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.56,
        source="Motorsport-Total / PACETEQ Red Bull duel",
        source_type="teammate race-pace delta",
        scope="Red Bull race pace, 2024 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="alonso_stroll_race_2023",
        network_key="race_network",
        faster_driver="ALO",
        slower_driver="STR",
        threshold_s=0.486,
        source="Motorsport.com / PACETEQ 2023 review",
        source_type="teammate race-pace delta",
        scope="Aston Martin race pace, 2023 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="alonso_stroll_race_2024",
        network_key="race_network",
        faster_driver="ALO",
        slower_driver="STR",
        threshold_s=0.25,
        source="Motorsport-Total / PACETEQ Aston Martin duel",
        source_type="teammate race-pace delta",
        scope="Aston Martin race pace, 2024 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="albon_sargeant_race_2023",
        network_key="race_network",
        faster_driver="ALB",
        slower_driver="SAR",
        threshold_s=0.293,
        source="Motorsport.com / PACETEQ 2023 review",
        source_type="teammate race-pace delta",
        scope="Williams race pace, 2023 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="albon_sargeant_race_2024",
        network_key="race_network",
        faster_driver="ALB",
        slower_driver="SAR",
        threshold_s=0.38,
        source="Motorsport-Total / PACETEQ Williams duel",
        source_type="teammate race-pace delta",
        scope="Williams race pace, 2024 Sargeant sample",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="verstappen_perez_quali_2022",
        network_key="quali_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.290,
        source="Motorsport.com / PACETEQ Perez trend",
        source_type="teammate qualifying delta",
        scope="Red Bull qualifying, 2022",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="verstappen_perez_quali_2023",
        network_key="quali_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.621,
        source="Motorsport.com / PACETEQ 2023 review",
        source_type="teammate qualifying delta",
        scope="Red Bull qualifying, 2023",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="verstappen_perez_quali_2024",
        network_key="quali_network",
        faster_driver="VER",
        slower_driver="PER",
        threshold_s=0.66,
        source="Motorsport-Total / PACETEQ Red Bull duel",
        source_type="teammate qualifying delta",
        scope="Red Bull qualifying, 2024",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="russell_hamilton_quali_2024",
        network_key="quali_network",
        faster_driver="RUS",
        slower_driver="HAM",
        threshold_s=0.23,
        source="Motorsport-Total / PACETEQ Mercedes duel",
        source_type="teammate qualifying delta",
        scope="Mercedes qualifying, 2024 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="albon_sargeant_quali_2023",
        network_key="quali_network",
        faster_driver="ALB",
        slower_driver="SAR",
        threshold_s=0.522,
        source="Motorsport.com / PACETEQ 2023 review",
        source_type="teammate qualifying delta",
        scope="Williams qualifying, 2023 only",
        tier="EXTERNAL_CONTEXT",
    ),
    ValidationCheck(
        check_id="albon_sargeant_quali_2024",
        network_key="quali_network",
        faster_driver="ALB",
        slower_driver="SAR",
        threshold_s=0.66,
        source="Motorsport-Total / PACETEQ Williams duel",
        source_type="teammate qualifying delta",
        scope="Williams qualifying, 2024 Sargeant sample",
        tier="EXTERNAL_CONTEXT",
    ),
)

SUPPLEMENTAL_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck(
        check_id="bottas_zhou_race_2024",
        network_key="race_network",
        faster_driver="BOT",
        slower_driver="ZHO",
        threshold_s=0.01,
        source="Motorsport-Total / PACETEQ Sauber duel",
        source_type="teammate race-pace delta",
        scope="Stake/Sauber race pace, 2024 only",
        tier="SUPPLEMENTAL",
    ),
)

CUT_CHECKS: tuple[dict[str, str], ...] = (
    {
        "check_id": "russell_latifi_race_2022",
        "status": "CUT_IMPOSSIBLE_PAIRING_YEAR",
        "reason": "Russell drove for Mercedes in 2022; Latifi's Williams teammate was Albon.",
    },
    {
        "check_id": "bottas_zhou_race_2022",
        "status": "CUT_NO_NUMERIC_RACE_SOURCE",
        "reason": "No defensible numeric race-pace source was found.",
    },
    {
        "check_id": "bottas_zhou_race_2023",
        "status": "CUT_DIRECTION_CONFLICT",
        "reason": "Accepted source reports Zhou slightly faster, conflicting with the candidate.",
    },
    {
        "check_id": "tsunoda_devries_race_2023",
        "status": "SMOKE_ONLY",
        "reason": "Sample too thin for source-backed validation.",
    },
    {
        "check_id": "leclerc_sainz_quali_2022_2024",
        "status": "SMOKE_ONLY",
        "reason": "Contested and hedged source base; excluded from HARD validation.",
    },
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Phase 6 prior builder."""
    parser = argparse.ArgumentParser(description="Fit teammate-network driver priors.")
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
        ),
        help="Phase 5 aggregate observation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/teammate_network_prior"),
        help="Directory for timestamped and latest prior artifacts.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Cluster bootstrap replicate count.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Random seed for cluster bootstrap sampling.",
    )
    parser.add_argument(
        "--effective-n-cap",
        type=int,
        default=32,
        help="Maximum matched-pair count contribution per aggregate row.",
    )
    return parser


def main() -> None:
    """Fit and write the teammate-network prior artifact."""
    args = build_parser().parse_args()
    config = PriorFitConfig(
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_random_seed=args.bootstrap_seed,
        effective_n_cap=args.effective_n_cap,
    )
    observations = pd.read_csv(args.observations)
    artifact = build_teammate_network_prior(observations, config=config)
    written = write_prior_artifacts(artifact, output_dir=args.output_dir)
    print(format_prior_summary(artifact, written_paths=written))


def build_teammate_network_prior(
    observations: pd.DataFrame,
    *,
    config: PriorFitConfig,
    built_at: str | None = None,
) -> dict[str, Any]:
    """Build the full teammate-network prior artifact from aggregate rows."""
    built_at = built_at or datetime.now(UTC).isoformat()
    fit_rows = _valid_fit_rows(observations)
    race_network = build_network_prior(fit_rows, session_kind="race", config=config)
    quali_network = build_network_prior(fit_rows, session_kind="qualifying", config=config)
    artifact: dict[str, Any] = {
        "built_at": built_at,
        "config": {
            "historical_scope": {
                "start": config.historical_start,
                "end": config.historical_end,
            },
            "matched_lap_config_race": asdict(MatchedLapConfig()),
            "matched_lap_config_quali": asdict(MatchedLapConfig()),
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_random_seed": config.bootstrap_random_seed,
            "race_sigma_floor_s": config.race_sigma_floor_s,
            "quali_sigma_floor_s": config.quali_sigma_floor_s,
            "min_driver_observations": config.min_driver_observations,
            "effective_n_cap": config.effective_n_cap,
            "main_min_observation_share": config.main_min_observation_share,
            "main_min_driver_share": config.main_min_driver_share,
        },
        "race_network": race_network,
        "quali_network": quali_network,
    }
    artifact["validation"] = evaluate_validation(artifact, observations=fit_rows, config=config)
    return artifact


def build_network_prior(
    observations: pd.DataFrame,
    *,
    session_kind: SessionKind,
    config: PriorFitConfig,
) -> dict[str, Any]:
    """Fit one dry teammate network for race or qualifying observations."""
    observations = _valid_fit_rows(observations)
    network_rows = observations[
        observations["session_kind"].eq(session_kind) & observations["weather_bucket"].eq("dry")
    ].copy()
    sigma_floor_s = _sigma_floor(session_kind, config)
    if network_rows.empty:
        return _empty_network(session_kind)

    components = _component_summaries(network_rows)
    main_component = components[0]
    if (
        main_component["observation_share"] < config.main_min_observation_share
        or main_component["driver_share"] < config.main_min_driver_share
    ):
        raise ValueError(
            f"{session_kind} network has no dominant component: "
            f"obs_share={main_component['observation_share']:.3f}, "
            f"driver_share={main_component['driver_share']:.3f}"
        )

    rng = np.random.default_rng(config.bootstrap_random_seed + (0 if session_kind == "race" else 1))
    fitted_components: list[dict[str, Any]] = []
    raw_component_fits: dict[int, dict[str, float]] = {}

    for component in components:
        component_id = int(component["component_id"])
        component_rows = _component_rows(network_rows, set(component["drivers"]))
        theta = _fit_component_theta(component_rows, component["drivers"], config, sigma_floor_s)
        raw_component_fits[component_id] = theta

    main_population_sd = _population_sd(raw_component_fits[int(main_component["component_id"])])
    if main_population_sd <= 0:
        main_population_sd = sigma_floor_s

    driver_entries: dict[str, dict[str, Any]] = {}
    for component in components:
        component_id = int(component["component_id"])
        anchored = component_id == int(main_component["component_id"])
        component_drivers = list(component["drivers"])
        component_rows = _component_rows(network_rows, set(component_drivers))
        theta = raw_component_fits[component_id]
        bootstrap_sigmas = _bootstrap_component_sigmas(
            component_rows,
            component_drivers,
            config,
            sigma_floor_s,
            rng,
        )
        fitted_components.append({**component, "anchored": anchored})

        for driver in component_drivers:
            driver_rows = _driver_rows(component_rows, driver)
            n_observations = int(len(driver_rows))
            sigma_s = _driver_sigma(
                driver=driver,
                anchored=anchored,
                n_observations=n_observations,
                bootstrap_sigmas=bootstrap_sigmas,
                population_sd_s=main_population_sd,
                sigma_floor_s=sigma_floor_s,
                config=config,
            )
            driver_entries[driver] = {
                "mu_s": float(theta[driver]),
                "sigma_s": sigma_s,
                "n_observations": n_observations,
                "n_teammate_partners": _partner_count(driver_rows, driver),
                "component_id": component_id,
                "component_anchored": anchored,
                "first_session": _session_label(driver_rows.sort_values("_input_order").iloc[0]),
                "last_session": _session_label(driver_rows.sort_values("_input_order").iloc[-1]),
            }

    return {
        "drivers": dict(sorted(driver_entries.items())),
        "components": fitted_components,
        "fit_diagnostics": {
            "session_kind": session_kind,
            "weather_bucket": "dry",
            "n_observations": int(len(network_rows)),
            "n_drivers": int(len(driver_entries)),
            "n_components": int(len(fitted_components)),
            "main_component_id": int(main_component["component_id"]),
            "main_component_observation_share": float(main_component["observation_share"]),
            "main_component_driver_share": float(main_component["driver_share"]),
            "main_component_population_sd_s": float(main_population_sd),
            "weight_distribution": _weight_distribution(network_rows, config, sigma_floor_s),
        },
    }


def evaluate_validation(
    artifact: dict[str, Any],
    *,
    observations: pd.DataFrame | None = None,
    config: PriorFitConfig | None = None,
) -> dict[str, Any]:
    """Evaluate locked validation checks and attach direct-pair diagnostics."""
    fit_rows = _valid_fit_rows(observations) if observations is not None else None
    fit_config = config or PriorFitConfig()
    hard_results = [
        _evaluate_check(check, artifact, observations=fit_rows, config=fit_config)
        for check in HARD_VALIDATION_CHECKS
    ]
    context_results = [
        _evaluate_check(check, artifact, observations=fit_rows, config=fit_config)
        for check in CONTEXT_CHECKS
    ]
    supplemental_results = [
        _evaluate_check(check, artifact, observations=fit_rows, config=fit_config)
        for check in SUPPLEMENTAL_CHECKS
    ]
    hard_race = [row for row in hard_results if row["network_key"] == "race_network"]
    hard_quali = [row for row in hard_results if row["network_key"] == "quali_network"]
    failed_hard = [row["check_id"] for row in hard_results if not bool(row["passed"])]
    hard_validation_state = "ready" if hard_results else "provisional_no_same_construct_hard_checks"
    return {
        "source_backed_checks": hard_results,
        "context_checks": context_results,
        "supplemental_checks": supplemental_results,
        "cut_checks": list(CUT_CHECKS),
        "hard_race_passed": _passed_count(hard_race),
        "hard_race_total": len(hard_race),
        "hard_quali_passed": _passed_count(hard_quali),
        "hard_quali_total": len(hard_quali),
        "failed_hard_check_ids": failed_hard,
        "hard_validation_state": hard_validation_state,
        "all_hard_checks_passed": bool(hard_results)
        and all(bool(row["passed"]) for row in hard_results),
        "validation_contract_note": (
            "No PACETEQ rows currently count as HARD because the construct audit did "
            "not prove a same-construct match for either the current paired race "
            "residual or the current multi-run qualifying residual. These rows remain "
            "EXTERNAL_CONTEXT until genuinely aligned evidence is sourced."
        ),
        "quali_validation_note": (
            "Qualifying validation is provisional: the available external qualifying rows "
            "are reported as EXTERNAL_CONTEXT because they do not yet match the current "
            "multi-run qualifying construct closely enough to gate the fit."
        ),
        "smoke_only_note": (
            "Direction-only smoke checks are excluded from the HARD pass count and belong in tests."
        ),
    }


def write_prior_artifacts(artifact: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    """Write timestamped, latest, and validation-report prior artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    built_at = str(artifact["built_at"])
    timestamp = _timestamp_slug(built_at)
    timestamped_path = output_dir / f"{timestamp}.json"
    latest_path = output_dir / "latest.json"
    report_path = output_dir / "validation_report.md"
    payload = json.dumps(_json_safe(artifact), indent=2, sort_keys=True)
    timestamped_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    report_path.write_text(format_validation_report(artifact), encoding="utf-8")
    return {
        "timestamped": str(timestamped_path),
        "latest": str(latest_path),
        "validation_report": str(report_path),
    }


def format_prior_summary(artifact: dict[str, Any], *, written_paths: dict[str, str]) -> str:
    """Format a compact CLI summary for a fitted prior artifact."""
    validation = artifact["validation"]
    race_diag = artifact["race_network"]["fit_diagnostics"]
    quali_diag = artifact["quali_network"]["fit_diagnostics"]
    return "\n".join(
        [
            "# Teammate-Network Prior",
            f"- Built at: {artifact['built_at']}",
            f"- Race rows/drivers: {race_diag['n_observations']} / {race_diag['n_drivers']}",
            f"- Quali rows/drivers: {quali_diag['n_observations']} / {quali_diag['n_drivers']}",
            (
                "- HARD validation: "
                f"race {validation['hard_race_passed']}/{validation['hard_race_total']}, "
                f"quali {validation['hard_quali_passed']}/{validation['hard_quali_total']}"
            ),
            f"- HARD validation state: {validation['hard_validation_state']}",
            f"- All HARD checks passed: {validation['all_hard_checks_passed']}",
            f"- Wrote: {written_paths['latest']}",
            f"- Timestamped: {written_paths['timestamped']}",
            f"- Validation report: {written_paths['validation_report']}",
        ]
    )


def format_validation_report(artifact: dict[str, Any]) -> str:
    """Format the prior validation result as a compact Markdown report."""
    validation = artifact["validation"]
    hard_checks = list(validation["source_backed_checks"])
    race_checks = [row for row in hard_checks if row["network_key"] == "race_network"]
    quali_checks = [row for row in hard_checks if row["network_key"] == "quali_network"]
    context_checks = list(validation.get("context_checks", []))
    lines = [
        "# Teammate-Network Prior Validation Report",
        "",
        f"Built at: `{artifact['built_at']}`",
        "",
        "## Summary",
        "",
        f"- HARD race checks: {validation['hard_race_passed']}/{validation['hard_race_total']}",
        f"- HARD qualifying checks: {validation['hard_quali_passed']}/{validation['hard_quali_total']}",
        f"- External context checks: {len(context_checks)}",
        f"- HARD validation state: `{validation['hard_validation_state']}`",
        f"- All HARD checks passed: `{str(validation['all_hard_checks_passed']).lower()}`",
        (
            "- Failed HARD checks: "
            + _inline_code_list(validation["failed_hard_check_ids"], fallback="none")
        ),
        "",
        "## HARD Race Checks",
        "",
        _format_validation_table(race_checks),
        "",
        "## HARD Qualifying Checks",
        "",
        _format_validation_table(quali_checks),
        "",
        "## External Context Checks",
        "",
        _format_validation_table(context_checks),
        "",
        "## Supplemental Checks",
        "",
        _format_validation_table(list(validation["supplemental_checks"])),
        "",
        "## Cut Checks",
        "",
        _format_cut_checks(validation["cut_checks"]),
        "",
        "## Notes",
        "",
        f"- {validation['validation_contract_note']}",
        f"- {validation['quali_validation_note']}",
        f"- {validation['smoke_only_note']}",
        (
            "- Direct-pair diagnostics use the same aggregate matched-lap rows and WLS weights "
            "as the prior fit. They are diagnostic only; the locked pass rule is still the "
            "fitted driver-prior delta."
        ),
    ]
    return "\n".join(lines) + "\n"


def _valid_fit_rows(observations: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate rows that are valid prior-fit observations."""
    required = {
        "reference_driver_code",
        "comparison_driver_code",
        "year",
        "race_name",
        "session_name",
        "session_kind",
        "team",
        "matched_gap_median_s",
        "matched_gap_se_s",
        "n_matched_pairs",
        "weather_bucket",
        "skip_reason",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Aggregate observations are missing columns: {missing}")

    rows = observations.copy()
    rows = _ensure_input_order(rows)
    counts = pd.to_numeric(rows["n_matched_pairs"], errors="coerce").fillna(0)
    rows["matched_gap_median_s"] = pd.to_numeric(rows["matched_gap_median_s"], errors="coerce")
    rows["matched_gap_se_s"] = pd.to_numeric(rows["matched_gap_se_s"], errors="coerce")
    mask = (
        counts.gt(0)
        & rows["skip_reason"].isna()
        & rows["matched_gap_median_s"].notna()
        & rows["matched_gap_se_s"].notna()
        & rows["session_kind"].isin(["race", "qualifying"])
    )
    return rows[mask].copy()


def _component_summaries(network_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return connected-component summaries sorted by observation count."""
    components = _connected_components(
        network_rows[["reference_driver_code", "comparison_driver_code"]].itertuples(
            index=False,
            name=None,
        )
    )
    total_observations = int(len(network_rows))
    total_drivers = len({driver for component in components for driver in component})
    summaries: list[dict[str, Any]] = []
    for component in components:
        rows = _component_rows(network_rows, component)
        summaries.append(
            {
                "drivers": sorted(component),
                "n_drivers": len(component),
                "n_observations": int(len(rows)),
                "observation_share": len(rows) / total_observations if total_observations else 0.0,
                "driver_share": len(component) / total_drivers if total_drivers else 0.0,
                "edge_weights": _edge_weights(rows),
            }
        )
    summaries = sorted(
        summaries,
        key=lambda row: (-int(row["n_observations"]), -int(row["n_drivers"]), row["drivers"]),
    )
    return [{**summary, "component_id": idx} for idx, summary in enumerate(summaries)]


def _ensure_input_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a stable input-order column."""
    if "_input_order" in rows.columns:
        return rows.copy()
    ordered = rows.copy()
    ordered["_input_order"] = np.arange(len(ordered))
    return ordered


def _connected_components(edges: Any) -> list[set[str]]:
    """Return graph connected components from teammate edges."""
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        left_code = str(left)
        right_code = str(right)
        adjacency.setdefault(left_code, set()).add(right_code)
        adjacency.setdefault(right_code, set()).add(left_code)

    seen: set[str] = set()
    components: list[set[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current].difference(seen)))
        components.append(component)
    return components


def _component_rows(network_rows: pd.DataFrame, component: set[str] | list[str]) -> pd.DataFrame:
    """Return rows whose teammate edge is inside a component."""
    component_set = set(component)
    return network_rows[
        network_rows["reference_driver_code"].isin(component_set)
        & network_rows["comparison_driver_code"].isin(component_set)
    ].copy()


def _driver_rows(rows: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Return rows involving one driver."""
    return rows[
        rows["reference_driver_code"].eq(driver) | rows["comparison_driver_code"].eq(driver)
    ].copy()


def _edge_weights(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return valid observation counts by teammate edge."""
    if rows.empty:
        return []
    grouped = (
        rows.groupby(["reference_driver_code", "comparison_driver_code"], dropna=False)
        .size()
        .reset_index(name="n_observations")
        .sort_values(["reference_driver_code", "comparison_driver_code"])
    )
    return grouped.to_dict(orient="records")


def _fit_component_theta(
    rows: pd.DataFrame,
    drivers: list[str],
    config: PriorFitConfig,
    sigma_floor_s: float,
) -> dict[str, float]:
    """Fit one sum-to-zero WLS component and return driver theta values."""
    ordered_drivers = sorted(drivers)
    if len(ordered_drivers) == 1:
        return {ordered_drivers[0]: 0.0}
    design = _reduced_design_matrix(rows, ordered_drivers)
    y = rows["matched_gap_median_s"].astype(float).to_numpy()
    weights = _observation_weights(rows, config, sigma_floor_s)
    sqrt_weights = np.sqrt(weights)
    weighted_x = design * sqrt_weights[:, None]
    weighted_y = y * sqrt_weights
    beta = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    theta_values = list(beta)
    theta_values.append(-float(np.sum(beta)))
    return {
        driver: float(theta) for driver, theta in zip(ordered_drivers, theta_values, strict=True)
    }


def _reduced_design_matrix(rows: pd.DataFrame, drivers: list[str]) -> np.ndarray:
    """Build the reduced design matrix under a sum-to-zero constraint."""
    free_drivers = drivers[:-1]
    dropped_driver = drivers[-1]
    driver_index = {driver: idx for idx, driver in enumerate(free_drivers)}
    design = np.zeros((len(rows), len(free_drivers)), dtype=float)
    for row_idx, row in enumerate(rows.itertuples(index=False)):
        reference = str(row.reference_driver_code)
        comparison = str(row.comparison_driver_code)
        if reference == dropped_driver:
            design[row_idx, :] -= 1.0
        else:
            design[row_idx, driver_index[reference]] += 1.0
        if comparison == dropped_driver:
            design[row_idx, :] += 1.0
        else:
            design[row_idx, driver_index[comparison]] -= 1.0
    return design


def _observation_weights(
    rows: pd.DataFrame,
    config: PriorFitConfig,
    sigma_floor_s: float,
) -> np.ndarray:
    """Return WLS weights using capped effective sample size and SE floor."""
    effective_n = np.minimum(
        pd.to_numeric(rows["n_matched_pairs"], errors="coerce").fillna(0).astype(float),
        float(config.effective_n_cap),
    )
    se = (
        pd.to_numeric(rows["matched_gap_se_s"], errors="coerce").fillna(sigma_floor_s).astype(float)
    )
    variance = np.maximum(np.square(se), sigma_floor_s**2)
    weights = effective_n / variance
    return np.maximum(weights.to_numpy(dtype=float), 1e-9)


def _bootstrap_component_sigmas(
    rows: pd.DataFrame,
    drivers: list[str],
    config: PriorFitConfig,
    sigma_floor_s: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate driver uncertainty with cluster bootstrap by session/team pair."""
    if config.bootstrap_replicates <= 0 or rows.empty:
        return {driver: sigma_floor_s for driver in drivers}

    cluster_indices = _cluster_row_indices(rows)
    if not cluster_indices:
        return {driver: sigma_floor_s for driver in drivers}

    samples: dict[str, list[float]] = {driver: [] for driver in drivers}
    for _ in range(config.bootstrap_replicates):
        sampled_positions = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        sampled_indices: list[int] = []
        for position in sampled_positions:
            sampled_indices.extend(cluster_indices[int(position)])
        sample = rows.iloc[sampled_indices].copy()
        theta = _fit_component_theta(sample, drivers, config, sigma_floor_s)
        for driver in drivers:
            samples[driver].append(theta[driver])

    sigmas: dict[str, float] = {}
    for driver, values in samples.items():
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else sigma_floor_s
        sigmas[driver] = sigma if np.isfinite(sigma) else sigma_floor_s
    return sigmas


def _cluster_row_indices(rows: pd.DataFrame) -> list[list[int]]:
    """Return row-position clusters by session and teammate pair."""
    cluster_columns = [
        "year",
        "race_name",
        "session_name",
        "team",
        "reference_driver_code",
        "comparison_driver_code",
    ]
    clusters: list[list[int]] = []
    positions = pd.Series(np.arange(len(rows)), index=rows.index)
    grouped = rows.assign(_position=positions).groupby(cluster_columns, dropna=False)
    for _, group in grouped:
        clusters.append([int(position) for position in group["_position"].tolist()])
    return clusters


def _driver_sigma(
    *,
    driver: str,
    anchored: bool,
    n_observations: int,
    bootstrap_sigmas: dict[str, float],
    population_sd_s: float,
    sigma_floor_s: float,
    config: PriorFitConfig,
) -> float:
    """Apply the configured uncertainty floor and fallback rules for one driver."""
    fallback_sigma = max(1.75 * population_sd_s, sigma_floor_s)
    if not anchored:
        return float(fallback_sigma)
    if n_observations < config.min_driver_observations:
        return float(fallback_sigma)
    trusted_sigma = max(
        float(bootstrap_sigmas.get(driver, sigma_floor_s)),
        0.5 * population_sd_s,
        sigma_floor_s,
    )
    return float(trusted_sigma)


def _partner_count(rows: pd.DataFrame, driver: str) -> int:
    """Count distinct teammate partners observed for one driver."""
    partners: set[str] = set()
    for row in rows.itertuples(index=False):
        reference = str(row.reference_driver_code)
        comparison = str(row.comparison_driver_code)
        if reference == driver:
            partners.add(comparison)
        elif comparison == driver:
            partners.add(reference)
    return len(partners)


def _session_label(row: pd.Series) -> str:
    """Return a compact session label for artifact driver metadata."""
    return f"{int(row['year'])} {row['race_name']} {row['session_name']}"


def _sigma_floor(session_kind: SessionKind, config: PriorFitConfig) -> float:
    """Return the configured sigma floor for a session kind."""
    return config.race_sigma_floor_s if session_kind == "race" else config.quali_sigma_floor_s


def _population_sd(theta: dict[str, float]) -> float:
    """Return population standard deviation for fitted component skills."""
    if not theta:
        return 0.0
    values = np.array(list(theta.values()), dtype=float)
    sigma = float(np.std(values, ddof=0))
    return sigma if np.isfinite(sigma) else 0.0


def _weight_distribution(
    rows: pd.DataFrame,
    config: PriorFitConfig,
    sigma_floor_s: float,
) -> dict[str, float | int | None]:
    """Summarise observation weights for diagnostics."""
    weights = _observation_weights(rows, config, sigma_floor_s)
    if len(weights) == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(len(weights)),
        "min": float(np.min(weights)),
        "median": float(np.median(weights)),
        "max": float(np.max(weights)),
    }


def _evaluate_check(
    check: ValidationCheck,
    artifact: dict[str, Any],
    *,
    observations: pd.DataFrame | None,
    config: PriorFitConfig,
) -> dict[str, Any]:
    """Evaluate one validation check from fitted driver ratings."""
    network = artifact[check.network_key]
    drivers = network.get("drivers", {})
    faster = drivers.get(check.faster_driver)
    slower = drivers.get(check.slower_driver)
    observed_delta = None
    passed = False
    reason = None
    if faster is None or slower is None:
        reason = "driver_missing_from_network"
    else:
        observed_delta = float(faster["mu_s"]) - float(slower["mu_s"])
        passed = observed_delta >= check.threshold_s
        if not passed:
            reason = "below_threshold"

    direct_diagnostics = (
        _direct_pair_diagnostics(check, observations, config) if observations is not None else None
    )
    failure_analysis = _validation_failure_analysis(
        check=check,
        passed=passed,
        failure_reason=reason,
        direct_diagnostics=direct_diagnostics,
    )
    return {
        "check_id": check.check_id,
        "tier": check.tier,
        "network_key": check.network_key,
        "scope": check.scope,
        "source": check.source,
        "source_type": check.source_type,
        "pass_rule": (
            f"{check.faster_driver}_mu_s - {check.slower_driver}_mu_s >= {check.threshold_s}"
        ),
        "faster_driver": check.faster_driver,
        "slower_driver": check.slower_driver,
        "threshold_s": check.threshold_s,
        "observed_delta_s": observed_delta,
        "margin_s": None if observed_delta is None else observed_delta - check.threshold_s,
        "passed": passed,
        "failure_reason": reason,
        "failure_analysis": failure_analysis,
        "direct_pair_diagnostics": direct_diagnostics,
    }


def _direct_pair_diagnostics(
    check: ValidationCheck,
    observations: pd.DataFrame,
    config: PriorFitConfig,
) -> dict[str, Any]:
    """Return direct same-pair aggregate diagnostics for one validation check."""
    scope_year = _check_scope_year(check)
    return {
        "scope_year": scope_year,
        "source_scope": _direct_pair_stats(check, observations, config, year=scope_year),
        "all_years": _direct_pair_stats(check, observations, config, year=None),
    }


def _direct_pair_stats(
    check: ValidationCheck,
    observations: pd.DataFrame,
    config: PriorFitConfig,
    *,
    year: int | None,
) -> dict[str, Any]:
    """Summarise direct aggregate rows for the checked teammate pair."""
    session_kind = _network_session_kind(check.network_key)
    rows = observations[
        observations["session_kind"].eq(session_kind)
        & observations["weather_bucket"].eq("dry")
        & (
            (
                observations["reference_driver_code"].eq(check.faster_driver)
                & observations["comparison_driver_code"].eq(check.slower_driver)
            )
            | (
                observations["reference_driver_code"].eq(check.slower_driver)
                & observations["comparison_driver_code"].eq(check.faster_driver)
            )
        )
    ].copy()
    if year is not None:
        rows = rows[rows["year"].eq(year)].copy()

    if rows.empty:
        return {
            "n_observations": 0,
            "n_matched_pairs": 0,
            "weighted_mean_delta_s": None,
            "median_delta_s": None,
            "min_delta_s": None,
            "max_delta_s": None,
        }

    deltas = _direct_pair_deltas(rows, faster_driver=check.faster_driver)
    sigma_floor_s = _sigma_floor(session_kind, config)
    weights = _observation_weights(rows, config, sigma_floor_s)
    return {
        "n_observations": int(len(rows)),
        "n_matched_pairs": int(
            pd.to_numeric(rows["n_matched_pairs"], errors="coerce").fillna(0).sum()
        ),
        "weighted_mean_delta_s": float(np.average(deltas, weights=weights)),
        "median_delta_s": float(np.median(deltas)),
        "min_delta_s": float(np.min(deltas)),
        "max_delta_s": float(np.max(deltas)),
    }


def _direct_pair_deltas(rows: pd.DataFrame, *, faster_driver: str) -> np.ndarray:
    """Return direct row deltas signed as faster-driver minus slower-driver."""
    gaps = rows["matched_gap_median_s"].astype(float).to_numpy()
    signs = np.where(rows["reference_driver_code"].eq(faster_driver), 1.0, -1.0)
    return gaps * signs


def _network_session_kind(network_key: NetworkKey) -> SessionKind:
    """Map an artifact network key to its observation session kind."""
    return "race" if network_key == "race_network" else "qualifying"


def _check_scope_year(check: ValidationCheck) -> int | None:
    """Extract the season year from a validation check id when present."""
    for token in check.check_id.split("_"):
        if len(token) == 4 and token.isdigit():
            return int(token)
    return None


def _validation_failure_analysis(
    *,
    check: ValidationCheck,
    passed: bool,
    failure_reason: str | None,
    direct_diagnostics: dict[str, Any] | None,
) -> str:
    """Classify a failed validation row without changing the pass rule."""
    if passed:
        return "passed"
    if failure_reason == "driver_missing_from_network":
        return "driver_missing_from_network"
    if direct_diagnostics is None:
        return "below_threshold_no_direct_diagnostics"

    scope_delta = _weighted_direct_delta(direct_diagnostics.get("source_scope"))
    all_year_delta = _weighted_direct_delta(direct_diagnostics.get("all_years"))
    if scope_delta is None and all_year_delta is None:
        return "no_direct_pair_observations"
    if scope_delta is not None and scope_delta >= check.threshold_s:
        return "pooled_prior_below_source_scope_direct_delta"
    if all_year_delta is not None and all_year_delta >= check.threshold_s:
        return "pooled_prior_below_all_year_direct_delta"
    return "matched_lap_direct_delta_below_source_threshold"


def _weighted_direct_delta(stats: Any) -> float | None:
    """Return a direct weighted delta from a stats payload if available."""
    if not isinstance(stats, dict):
        return None
    value = stats.get("weighted_mean_delta_s")
    return None if value is None else float(value)


def _inline_code_list(values: list[str], *, fallback: str) -> str:
    """Format a short list as inline code values."""
    if not values:
        return fallback
    return ", ".join(f"`{value}`" for value in values)


def _format_validation_table(rows: list[dict[str, Any]]) -> str:
    """Format validation rows as a Markdown table."""
    if not rows:
        return "_None._"

    lines = [
        (
            "| Check | Source | Threshold | Fitted delta | Scope direct | "
            "All-year direct | Status | Diagnosis |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        status = "PASS" if bool(row["passed"]) else "FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['check_id']}`",
                    str(row["source"]),
                    _format_seconds(row["threshold_s"]),
                    _format_seconds(row["observed_delta_s"]),
                    _format_seconds(_direct_report_delta(row, "source_scope")),
                    _format_seconds(_direct_report_delta(row, "all_years")),
                    status,
                    _human_failure_analysis(str(row["failure_analysis"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _direct_report_delta(row: dict[str, Any], key: str) -> float | None:
    """Return a direct-pair weighted mean delta for report rendering."""
    diagnostics = row.get("direct_pair_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _weighted_direct_delta(diagnostics.get(key))


def _format_seconds(value: Any) -> str:
    """Format a numeric seconds value for Markdown tables."""
    if value is None:
        return "-"
    return f"{float(value):.3f}s"


def _human_failure_analysis(value: str) -> str:
    """Return a human-readable validation diagnosis."""
    labels = {
        "passed": "passed",
        "driver_missing_from_network": "driver missing from fitted network",
        "below_threshold_no_direct_diagnostics": "below threshold; no direct diagnostics",
        "no_direct_pair_observations": "no direct pair observations",
        "pooled_prior_below_source_scope_direct_delta": (
            "pooled prior below source-scope direct delta"
        ),
        "pooled_prior_below_all_year_direct_delta": "pooled prior below all-year direct delta",
        "matched_lap_direct_delta_below_source_threshold": (
            "matched-lap direct delta below source threshold"
        ),
    }
    return labels.get(value, value)


def _format_cut_checks(rows: list[dict[str, Any]]) -> str:
    """Format cut validation candidates as a Markdown list."""
    if not rows:
        return "_None._"
    return "\n".join(f"- `{row['check_id']}`: {row['status']} - {row['reason']}" for row in rows)


def _passed_count(rows: list[dict[str, Any]]) -> int:
    """Count passed validation rows."""
    return sum(1 for row in rows if bool(row["passed"]))


def _empty_network(session_kind: SessionKind) -> dict[str, Any]:
    """Return an empty network payload with diagnostics."""
    return {
        "drivers": {},
        "components": [],
        "fit_diagnostics": {
            "session_kind": session_kind,
            "weather_bucket": "dry",
            "n_observations": 0,
            "n_drivers": 0,
            "n_components": 0,
        },
    }


def _timestamp_slug(built_at: str) -> str:
    """Return a filesystem-safe timestamp slug."""
    parsed = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
