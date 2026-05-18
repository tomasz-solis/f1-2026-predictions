"""Probe race construct variants for locked validation rows.

The runner compares the stored Phase 5 race observations with fresh offline
cache recomputations. It keeps the current paired race residual separate from
broader valid-lap summaries so the validation audit can test whether the HARD
race rows are grading the same construct as the extractor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_matched_lap_observations import (  # noqa: E402
    BulkSessionSpec,
    discover_bulk_sessions,
    load_fastf1_session,
)
from scripts.build_teammate_network_prior import (  # noqa: E402
    CONTEXT_CHECKS,
    HARD_VALIDATION_CHECKS,
    PriorFitConfig,
)

from src.extractors.matched_laps import (  # noqa: E402
    MatchedLapConfig,
    probe_race_pair_constructs,
)


@dataclass(frozen=True)
class RaceProbeTarget:
    """One race validation row to inspect."""

    check_id: str
    year: int
    team: str
    faster_driver: str
    slower_driver: str
    threshold_s: float
    source: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the race construct probe."""
    parser = argparse.ArgumentParser(description="Probe race construct variants.")
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
        ),
        help="Versioned Phase 5 aggregate observation CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/.fastf1_cache"),
        help="FastF1 cache directory used for fresh construct probes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/teammate_network_construct_probe"),
        help="Directory for race construct-probe CSV, JSON, and Markdown output.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow FastF1 network reads instead of forcing local-cache offline mode.",
    )
    return parser


def main() -> None:
    """Run the race construct probe and write diagnostic outputs."""
    args = build_parser().parse_args()
    observations = pd.read_csv(args.observations)
    report, session_rows = run_race_construct_probe(
        observations=observations,
        cache_dir=args.cache_dir,
        offline=not args.online,
    )
    written = write_race_probe_outputs(report, session_rows, output_dir=args.output_dir)
    print(format_race_probe_summary(report, written_paths=written))


def race_probe_targets(observations: pd.DataFrame) -> list[RaceProbeTarget]:
    """Return tracked race validation rows with inferred teammate teams."""
    targets: list[RaceProbeTarget] = []
    for check in (*HARD_VALIDATION_CHECKS, *CONTEXT_CHECKS):
        if check.network_key != "race_network":
            continue
        year = _check_year(check.check_id)
        if year is None:
            continue
        team = _target_team(
            observations,
            year=year,
            faster_driver=check.faster_driver,
            slower_driver=check.slower_driver,
        )
        targets.append(
            RaceProbeTarget(
                check_id=check.check_id,
                year=year,
                team=team,
                faster_driver=check.faster_driver,
                slower_driver=check.slower_driver,
                threshold_s=check.threshold_s,
                source=check.source,
            )
        )
    return targets


def run_race_construct_probe(
    *,
    observations: pd.DataFrame,
    cache_dir: Path,
    offline: bool,
    targets: list[RaceProbeTarget] | None = None,
    config: MatchedLapConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load cached sessions and build a race construct-probe report."""
    targets = targets or race_probe_targets(observations)
    config = config or MatchedLapConfig()
    years = sorted({target.year for target in targets})
    specs = [
        spec
        for spec in discover_bulk_sessions(years=years, cache_dir=cache_dir, offline=offline)
        if spec.session_kind == "race"
    ]
    target_map = _targets_by_year(targets)
    session_rows: list[dict[str, Any]] = []

    for spec in specs:
        session = load_fastf1_session(spec, cache_dir=cache_dir, offline=offline)
        for target in target_map.get(spec.year, []):
            session_rows.append(
                build_race_session_probe_row(
                    spec,
                    target,
                    session=session,
                    observations=observations,
                    config=config,
                )
            )

    session_frame = pd.DataFrame(session_rows)
    report = build_race_probe_report(
        observations=observations,
        session_rows=session_frame,
        targets=targets,
        config=config,
    )
    return report, session_frame


def build_race_session_probe_row(
    spec: BulkSessionSpec,
    target: RaceProbeTarget,
    *,
    session: Any,
    observations: pd.DataFrame,
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Build one fresh-cache race probe row for one target and session."""
    cache_probe = probe_race_pair_constructs(
        session,
        reference_driver=target.faster_driver,
        comparison_driver=target.slower_driver,
        team=target.team,
        weather_mode="mixed",
        target_weather_bucket="dry",
        config=config,
    )
    artifact_row = _artifact_session_row(observations, target, race_name=spec.race_name)
    return {
        "check_id": target.check_id,
        "year": target.year,
        "team": target.team,
        "race_name": spec.race_name,
        "faster_driver": target.faster_driver,
        "slower_driver": target.slower_driver,
        **cache_probe,
        **artifact_row,
    }


def build_race_probe_report(
    *,
    observations: pd.DataFrame,
    session_rows: pd.DataFrame,
    targets: list[RaceProbeTarget],
    config: MatchedLapConfig,
    built_at: str | None = None,
) -> dict[str, Any]:
    """Build the aggregate race construct-probe report."""
    built_at = built_at or datetime.now(UTC).isoformat()
    summaries = [
        _target_summary(
            observations=observations,
            session_rows=session_rows,
            target=target,
        )
        for target in targets
    ]
    return {
        "built_at": built_at,
        "config": {
            "matched_lap_config": asdict(config),
            "target_weather_bucket": "dry",
            "source_observations": (
                "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
            ),
        },
        "targets": [asdict(target) for target in targets],
        "summaries": summaries,
    }


def write_race_probe_outputs(
    report: dict[str, Any],
    session_rows: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write race construct-probe CSV, JSON, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "race_session_rows.csv"
    report_path = output_dir / "race_construct_probe.json"
    text_path = output_dir / "race_construct_probe.md"
    session_rows.to_csv(rows_path, index=False)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    text_path.write_text(format_race_probe_report(report), encoding="utf-8")
    return {
        "rows": str(rows_path),
        "json": str(report_path),
        "markdown": str(text_path),
    }


def format_race_probe_summary(report: dict[str, Any], *, written_paths: dict[str, str]) -> str:
    """Format a compact terminal summary for a race construct probe."""
    mismatched = [
        row["check_id"]
        for row in report["summaries"]
        if int(row["artifact_cache_delta_mismatch_count"]) > 0
    ]
    return "\n".join(
        [
            "# Race Construct Probe",
            f"- Built at: {report['built_at']}",
            f"- Targets: {len(report['summaries'])}",
            f"- Targets with artifact/cache mismatches: {len(mismatched)}",
            f"- Wrote: {written_paths['markdown']}",
            f"- Session rows: {written_paths['rows']}",
        ]
    )


def format_race_probe_report(report: dict[str, Any]) -> str:
    """Format the race construct-probe report as compact Markdown."""
    lines = [
        "# Race Construct Probe",
        "",
        f"Built at: `{report['built_at']}`",
        "",
        "| Check | HARD threshold | Phase 5 WLS | Phase 5 equal mean | "
        "Cache current mean | Broad valid-lap median | Broad valid-lap mean | "
        "Phase 5 rows | Cache current rows | Broad rows | Cache mismatch rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['check_id']}`",
                    _seconds(row["threshold_s"]),
                    _seconds(row["phase5_wls_mean_delta_s"]),
                    _seconds(row["phase5_equal_mean_delta_s"]),
                    _seconds(row["cache_current_equal_mean_delta_s"]),
                    _seconds(row["cache_broad_valid_median_equal_mean_delta_s"]),
                    _seconds(row["cache_broad_valid_mean_equal_mean_delta_s"]),
                    str(row["phase5_valid_rows"]),
                    str(row["cache_current_rows"]),
                    str(row["cache_broad_rows"]),
                    str(row["artifact_cache_delta_mismatch_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Phase 5` columns come from the stored aggregate artifact.",
            "- `Cache current mean` is the current paired race residual recomputed from cache.",
            (
                "- `Broad valid-lap` columns keep the same lap-quality filters but remove "
                "same-compound and same-stint-lap pairing."
            ),
            (
                "- `Cache mismatch rows` counts sessions where the fresh current construct "
                "does not reproduce the stored Phase 5 delta to within 1ms."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _target_summary(
    *,
    observations: pd.DataFrame,
    session_rows: pd.DataFrame,
    target: RaceProbeTarget,
) -> dict[str, Any]:
    """Summarize one race validation target across its season."""
    phase5 = _phase5_target_rows(observations, target)
    cache = session_rows[session_rows["check_id"].eq(target.check_id)].copy()
    valid_phase5 = phase5[phase5["phase5_delta_s"].notna()].copy()
    current_rows = cache[cache["current_construct_delta_s"].notna()].copy()
    broad_rows = cache[cache["broad_valid_median_delta_s"].notna()].copy()
    return {
        **asdict(target),
        "phase5_valid_rows": int(len(valid_phase5)),
        "phase5_wls_mean_delta_s": _weighted_phase5_mean(valid_phase5),
        "phase5_equal_mean_delta_s": _mean_or_none(valid_phase5["phase5_delta_s"]),
        "cache_current_rows": int(len(current_rows)),
        "cache_current_equal_mean_delta_s": _mean_or_none(
            current_rows["current_construct_delta_s"]
        ),
        "cache_broad_rows": int(len(broad_rows)),
        "cache_broad_valid_median_equal_mean_delta_s": _mean_or_none(
            broad_rows["broad_valid_median_delta_s"]
        ),
        "cache_broad_valid_mean_equal_mean_delta_s": _mean_or_none(
            broad_rows["broad_valid_mean_delta_s"]
        ),
        "artifact_cache_delta_mismatch_count": _artifact_cache_mismatch_count(cache),
    }


def _target_team(
    observations: pd.DataFrame,
    *,
    year: int,
    faster_driver: str,
    slower_driver: str,
) -> str:
    """Infer the stored teammate team for one validation pair-year."""
    rows = observations[
        observations["year"].eq(year)
        & observations["session_kind"].eq("race")
        & (
            (
                observations["reference_driver_code"].eq(faster_driver)
                & observations["comparison_driver_code"].eq(slower_driver)
            )
            | (
                observations["reference_driver_code"].eq(slower_driver)
                & observations["comparison_driver_code"].eq(faster_driver)
            )
        )
    ].copy()
    teams = rows["team"].dropna().astype(str)
    if teams.empty:
        raise ValueError(
            f"Could not infer race-probe team for {year}:{faster_driver}-{slower_driver}"
        )
    return str(teams.mode().iloc[0])


def _targets_by_year(targets: list[RaceProbeTarget]) -> dict[int, list[RaceProbeTarget]]:
    """Group race targets by season."""
    grouped: dict[int, list[RaceProbeTarget]] = {}
    for target in targets:
        grouped.setdefault(target.year, []).append(target)
    return grouped


def _artifact_session_row(
    observations: pd.DataFrame,
    target: RaceProbeTarget,
    *,
    race_name: str,
) -> dict[str, Any]:
    """Return stored Phase 5 artifact fields for one race target session."""
    rows = _phase5_target_rows(observations, target)
    rows = rows[rows["race_name"].eq(race_name)].copy()
    if rows.empty:
        return {
            "phase5_row_present": False,
            "phase5_delta_s": None,
            "phase5_n_matched_pairs": 0,
            "phase5_skip_reason": None,
        }
    row = rows.iloc[0]
    return {
        "phase5_row_present": True,
        "phase5_delta_s": _none_if_na(row["phase5_delta_s"]),
        "phase5_n_matched_pairs": int(row["n_matched_pairs"]),
        "phase5_skip_reason": _none_if_na(row["skip_reason"]),
    }


def _phase5_target_rows(
    observations: pd.DataFrame,
    target: RaceProbeTarget,
) -> pd.DataFrame:
    """Return stored dry race rows for one validation target."""
    rows = observations[
        observations["year"].eq(target.year)
        & observations["team"].eq(target.team)
        & observations["session_kind"].eq("race")
        & observations["weather_bucket"].eq("dry")
        & (
            (
                observations["reference_driver_code"].eq(target.faster_driver)
                & observations["comparison_driver_code"].eq(target.slower_driver)
            )
            | (
                observations["reference_driver_code"].eq(target.slower_driver)
                & observations["comparison_driver_code"].eq(target.faster_driver)
            )
        )
    ].copy()
    signs = np.where(
        rows["reference_driver_code"].eq(target.faster_driver),
        1.0,
        -1.0,
    )
    rows["phase5_delta_s"] = pd.to_numeric(rows["matched_gap_median_s"], errors="coerce") * signs
    return rows


def _weighted_phase5_mean(rows: pd.DataFrame) -> float | None:
    """Return the current Phase 5 WLS mean for one race target."""
    if rows.empty:
        return None
    effective_n = pd.to_numeric(rows["n_matched_pairs"], errors="coerce").clip(
        upper=PriorFitConfig().effective_n_cap
    )
    se = pd.to_numeric(rows["matched_gap_se_s"], errors="coerce").clip(
        lower=PriorFitConfig().race_sigma_floor_s
    )
    weights = effective_n / se.pow(2)
    return float(np.average(rows["phase5_delta_s"].astype(float), weights=weights))


def _artifact_cache_mismatch_count(rows: pd.DataFrame) -> int:
    """Count stored/fresh current-construct deltas that differ by over 1ms."""
    comparable = rows[
        rows["phase5_delta_s"].notna() & rows["current_construct_delta_s"].notna()
    ].copy()
    if comparable.empty:
        return 0
    delta = (
        pd.to_numeric(comparable["phase5_delta_s"], errors="coerce")
        - pd.to_numeric(comparable["current_construct_delta_s"], errors="coerce")
    ).abs()
    return int(delta.gt(0.001).sum())


def _check_year(check_id: str) -> int | None:
    """Return the trailing year embedded in a validation check id."""
    year = check_id.rsplit("_", maxsplit=1)[-1]
    return int(year) if year.isdigit() else None


def _mean_or_none(values: pd.Series) -> float | None:
    """Return a numeric mean or None for an empty series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.mean())


def _none_if_na(value: Any) -> Any:
    """Return None for pandas missing values."""
    return None if pd.isna(value) else value


def _seconds(value: Any) -> str:
    """Format a seconds value for Markdown output."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.3f}s"


def _json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
