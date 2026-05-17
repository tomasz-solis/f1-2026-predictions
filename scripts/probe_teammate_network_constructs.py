"""Probe qualifying construct variants for locked validation rows.

The runner compares versioned Phase 5 aggregate observations with fresh
offline-cache qualifying probes. It exists to make construct choices explicit:
current multi-run matched-lap medians, highest-common-segment best laps, and
best valid laps are reported side by side instead of being inferred from a
single failed threshold.
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
    probe_qualifying_pair_constructs,
)


@dataclass(frozen=True)
class QualifyingProbeTarget:
    """One qualifying validation row to inspect."""

    check_id: str
    year: int
    faster_driver: str
    slower_driver: str
    threshold_s: float
    source: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the construct probe."""
    parser = argparse.ArgumentParser(description="Probe qualifying construct variants.")
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
        help="Directory for construct-probe CSV, JSON, and Markdown output.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow FastF1 network reads instead of forcing local-cache offline mode.",
    )
    return parser


def main() -> None:
    """Run the qualifying construct probe and write diagnostic outputs."""
    args = build_parser().parse_args()
    observations = pd.read_csv(args.observations)
    report, session_rows = run_qualifying_construct_probe(
        observations=observations,
        cache_dir=args.cache_dir,
        offline=not args.online,
    )
    written = write_probe_outputs(report, session_rows, output_dir=args.output_dir)
    print(format_probe_summary(report, written_paths=written))


def qualifying_probe_targets() -> list[QualifyingProbeTarget]:
    """Return qualifying validation rows with explicit season scope."""
    targets: list[QualifyingProbeTarget] = []
    for check in (*HARD_VALIDATION_CHECKS, *CONTEXT_CHECKS):
        if check.network_key != "quali_network":
            continue
        year = _check_year(check.check_id)
        if year is None:
            continue
        targets.append(
            QualifyingProbeTarget(
                check_id=check.check_id,
                year=year,
                faster_driver=check.faster_driver,
                slower_driver=check.slower_driver,
                threshold_s=check.threshold_s,
                source=check.source,
            )
        )
    return targets


def run_qualifying_construct_probe(
    *,
    observations: pd.DataFrame,
    cache_dir: Path,
    offline: bool,
    targets: list[QualifyingProbeTarget] | None = None,
    config: MatchedLapConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load cached sessions and build a construct-probe report."""
    targets = targets or qualifying_probe_targets()
    config = config or MatchedLapConfig()
    years = sorted({target.year for target in targets})
    specs = [
        spec
        for spec in discover_bulk_sessions(years=years, cache_dir=cache_dir, offline=offline)
        if spec.session_kind == "qualifying"
    ]
    target_map = _targets_by_year(targets)
    session_rows: list[dict[str, Any]] = []

    for spec in specs:
        session = load_fastf1_session(spec, cache_dir=cache_dir, offline=offline)
        for target in target_map.get(spec.year, []):
            session_rows.append(
                build_session_probe_row(
                    spec,
                    target,
                    session=session,
                    observations=observations,
                    config=config,
                )
            )

    session_frame = pd.DataFrame(session_rows)
    report = build_probe_report(
        observations=observations,
        session_rows=session_frame,
        targets=targets,
        config=config,
    )
    return report, session_frame


def build_session_probe_row(
    spec: BulkSessionSpec,
    target: QualifyingProbeTarget,
    *,
    session: Any,
    observations: pd.DataFrame,
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Build one fresh-cache probe row for one validation target and session."""
    cache_probe = probe_qualifying_pair_constructs(
        session,
        reference_driver=target.faster_driver,
        comparison_driver=target.slower_driver,
        weather_mode="mixed",
        target_weather_bucket="dry",
        config=config,
    )
    artifact_row = _artifact_session_row(observations, target, race_name=spec.race_name)
    return {
        "check_id": target.check_id,
        "year": target.year,
        "race_name": spec.race_name,
        "faster_driver": target.faster_driver,
        "slower_driver": target.slower_driver,
        **cache_probe,
        **artifact_row,
    }


def build_probe_report(
    *,
    observations: pd.DataFrame,
    session_rows: pd.DataFrame,
    targets: list[QualifyingProbeTarget],
    config: MatchedLapConfig,
    built_at: str | None = None,
) -> dict[str, Any]:
    """Build the aggregate construct-probe report."""
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


def write_probe_outputs(
    report: dict[str, Any],
    session_rows: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write construct-probe CSV, JSON, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "qualifying_session_rows.csv"
    report_path = output_dir / "qualifying_construct_probe.json"
    text_path = output_dir / "qualifying_construct_probe.md"
    session_rows.to_csv(rows_path, index=False)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    text_path.write_text(format_probe_report(report), encoding="utf-8")
    return {
        "rows": str(rows_path),
        "json": str(report_path),
        "markdown": str(text_path),
    }


def format_probe_summary(report: dict[str, Any], *, written_paths: dict[str, str]) -> str:
    """Format a compact terminal summary for a probe run."""
    mismatched = [
        row["check_id"]
        for row in report["summaries"]
        if int(row["artifact_cache_delta_mismatch_count"]) > 0
    ]
    return "\n".join(
        [
            "# Qualifying Construct Probe",
            f"- Built at: {report['built_at']}",
            f"- Targets: {len(report['summaries'])}",
            f"- Targets with artifact/cache mismatches: {len(mismatched)}",
            f"- Wrote: {written_paths['markdown']}",
            f"- Session rows: {written_paths['rows']}",
        ]
    )


def format_probe_report(report: dict[str, Any]) -> str:
    """Format the construct-probe report as compact Markdown."""
    lines = [
        "# Qualifying Construct Probe",
        "",
        f"Built at: `{report['built_at']}`",
        "",
        "| Check | HARD threshold | Phase 5 WLS | Phase 5 equal mean | "
        "Cache current mean | Highest-common best | Any-valid best | "
        "Phase 5 rows | Cache rows | Cache mismatch rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    _seconds(row["cache_highest_common_best_equal_mean_delta_s"]),
                    _seconds(row["cache_any_valid_best_equal_mean_delta_s"]),
                    str(row["phase5_valid_rows"]),
                    str(row["cache_current_rows"]),
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
            "- `Cache` columns are fresh offline FastF1 recomputations using the current extractor.",
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
    target: QualifyingProbeTarget,
) -> dict[str, Any]:
    """Summarize one qualifying validation target across a season."""
    phase5 = _phase5_target_rows(observations, target)
    cache = session_rows[session_rows["check_id"].eq(target.check_id)].copy()
    valid_phase5 = phase5[phase5["phase5_delta_s"].notna()].copy()
    return {
        "check_id": target.check_id,
        "year": target.year,
        "faster_driver": target.faster_driver,
        "slower_driver": target.slower_driver,
        "threshold_s": target.threshold_s,
        "source": target.source,
        "phase5_valid_rows": int(len(valid_phase5)),
        "phase5_wls_mean_delta_s": _weighted_phase5_mean(valid_phase5),
        "phase5_equal_mean_delta_s": _mean_or_none(valid_phase5["phase5_delta_s"]),
        "phase5_median_delta_s": _median_or_none(valid_phase5["phase5_delta_s"]),
        "cache_current_rows": int(cache["current_construct_delta_s"].notna().sum()),
        "cache_current_equal_mean_delta_s": _mean_or_none(cache["current_construct_delta_s"]),
        "cache_highest_common_best_rows": int(cache["highest_common_best_delta_s"].notna().sum()),
        "cache_highest_common_best_equal_mean_delta_s": _mean_or_none(
            cache["highest_common_best_delta_s"]
        ),
        "cache_any_valid_best_rows": int(cache["any_valid_best_delta_s"].notna().sum()),
        "cache_any_valid_best_equal_mean_delta_s": _mean_or_none(cache["any_valid_best_delta_s"]),
        "artifact_cache_delta_mismatch_count": _artifact_cache_mismatch_count(cache),
        "phase5_skipped_rows": int(phase5["phase5_delta_s"].isna().sum()),
    }


def _targets_by_year(
    targets: list[QualifyingProbeTarget],
) -> dict[int, list[QualifyingProbeTarget]]:
    """Group construct-probe targets by season."""
    grouped: dict[int, list[QualifyingProbeTarget]] = {}
    for target in targets:
        grouped.setdefault(target.year, []).append(target)
    return grouped


def _artifact_session_row(
    observations: pd.DataFrame,
    target: QualifyingProbeTarget,
    *,
    race_name: str,
) -> dict[str, Any]:
    """Return stored Phase 5 artifact fields for one target/session."""
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
    target: QualifyingProbeTarget,
) -> pd.DataFrame:
    """Return stored dry qualifying rows for one validation target."""
    rows = observations[
        observations["year"].eq(target.year)
        & observations["session_kind"].eq("qualifying")
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
    signs = np.where(rows["reference_driver_code"].eq(target.faster_driver), 1.0, -1.0)
    rows["phase5_delta_s"] = (
        pd.to_numeric(
            rows["matched_gap_median_s"],
            errors="coerce",
        )
        * signs
    )
    return rows


def _weighted_phase5_mean(rows: pd.DataFrame) -> float | None:
    """Return the Phase 5 WLS mean used by the prior builder diagnostics."""
    if rows.empty:
        return None
    effective_n = pd.to_numeric(rows["n_matched_pairs"], errors="coerce").clip(upper=32)
    se = pd.to_numeric(rows["matched_gap_se_s"], errors="coerce").clip(
        lower=PriorFitConfig().quali_sigma_floor_s
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
    """Extract a four-digit season from a validation check id."""
    for token in check_id.split("_"):
        if len(token) == 4 and token.isdigit():
            return int(token)
    return None


def _mean_or_none(values: pd.Series) -> float | None:
    """Return a numeric mean or None for an empty series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.mean())


def _median_or_none(values: pd.Series) -> float | None:
    """Return a numeric median or None for an empty series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.median())


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
