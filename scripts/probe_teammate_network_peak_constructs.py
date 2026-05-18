"""Probe peak qualifying constructs across the full historical teammate network.

This runner broadens the locked-row construct probe into a network-wide
diagnostic. It measures how much coverage a peak comparable qualifying
construct would recover, which qualifying segments supply those rows, and how
session-level dispersion changes relative to the current multi-run median.
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
from scripts.build_teammate_network_prior import PriorFitConfig  # noqa: E402

from src.extractors.matched_laps import (  # noqa: E402
    MatchedLapConfig,
    probe_qualifying_pair_constructs,
)


@dataclass(frozen=True)
class QualifyingNetworkTarget:
    """One stored teammate pair-season to inspect."""

    year: int
    team: str
    reference_driver: str
    comparison_driver: str

    @property
    def target_id(self) -> str:
        """Return a stable identifier for one pair-season target."""
        return f"{self.year}:{self.team}:{self.reference_driver}-{self.comparison_driver}"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the network-wide peak probe."""
    parser = argparse.ArgumentParser(description="Probe peak qualifying constructs network-wide.")
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
        help="Directory for network-wide peak-probe outputs.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow FastF1 network reads instead of forcing local-cache offline mode.",
    )
    return parser


def main() -> None:
    """Run the network-wide qualifying peak probe and write outputs."""
    args = build_parser().parse_args()
    observations = pd.read_csv(args.observations)
    report, session_rows, pair_rows = run_qualifying_network_peak_probe(
        observations=observations,
        cache_dir=args.cache_dir,
        offline=not args.online,
    )
    written = write_network_peak_outputs(
        report,
        session_rows,
        pair_rows,
        output_dir=args.output_dir,
    )
    print(format_network_peak_summary(report, written_paths=written))


def qualifying_network_targets(observations: pd.DataFrame) -> list[QualifyingNetworkTarget]:
    """Return stored qualifying teammate pair-seasons with complete identities."""
    qualifying = observations[observations["session_kind"].eq("qualifying")].copy()
    required = ["year", "team", "reference_driver_code", "comparison_driver_code"]
    qualifying = qualifying.dropna(subset=required)
    rows = qualifying[required].drop_duplicates().sort_values(required)
    return [
        QualifyingNetworkTarget(
            year=int(row["year"]),
            team=str(row["team"]),
            reference_driver=str(row["reference_driver_code"]),
            comparison_driver=str(row["comparison_driver_code"]),
        )
        for _, row in rows.iterrows()
    ]


def run_qualifying_network_peak_probe(
    *,
    observations: pd.DataFrame,
    cache_dir: Path,
    offline: bool,
    targets: list[QualifyingNetworkTarget] | None = None,
    config: MatchedLapConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load cached sessions and build the network-wide peak-probe report."""
    targets = targets or qualifying_network_targets(observations)
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
                build_network_session_probe_row(
                    spec,
                    target,
                    session=session,
                    observations=observations,
                    config=config,
                )
            )

    session_frame = pd.DataFrame(session_rows)
    pair_frame = build_pair_season_summaries(
        observations=observations,
        session_rows=session_frame,
        targets=targets,
    )
    report = build_network_peak_report(
        session_rows=session_frame,
        pair_rows=pair_frame,
        targets=targets,
        config=config,
    )
    return report, session_frame, pair_frame


def build_network_session_probe_row(
    spec: BulkSessionSpec,
    target: QualifyingNetworkTarget,
    *,
    session: Any,
    observations: pd.DataFrame,
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Build one session-level probe row for one stored pair-season target."""
    cache_probe = probe_qualifying_pair_constructs(
        session,
        reference_driver=target.reference_driver,
        comparison_driver=target.comparison_driver,
        team=target.team,
        weather_mode="mixed",
        target_weather_bucket="dry",
        config=config,
    )
    artifact_row = _artifact_session_row(observations, target, race_name=spec.race_name)
    return {
        "target_id": target.target_id,
        "year": target.year,
        "team": target.team,
        "race_name": spec.race_name,
        "reference_driver": target.reference_driver,
        "comparison_driver": target.comparison_driver,
        **cache_probe,
        **artifact_row,
    }


def build_pair_season_summaries(
    *,
    observations: pd.DataFrame,
    session_rows: pd.DataFrame,
    targets: list[QualifyingNetworkTarget],
) -> pd.DataFrame:
    """Summarize construct behavior for every stored qualifying pair-season."""
    rows = [
        _pair_season_summary(
            observations=observations,
            session_rows=session_rows,
            target=target,
        )
        for target in targets
    ]
    return pd.DataFrame(rows)


def build_network_peak_report(
    *,
    session_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    targets: list[QualifyingNetworkTarget],
    config: MatchedLapConfig,
    built_at: str | None = None,
) -> dict[str, Any]:
    """Build the aggregate network-wide peak-probe report."""
    built_at = built_at or datetime.now(UTC).isoformat()
    peak_rows = session_rows[session_rows["highest_common_best_delta_s"].notna()].copy()
    segment_counts = (
        peak_rows["highest_common_segment"].value_counts(dropna=False).sort_index().to_dict()
        if not peak_rows.empty
        else {}
    )
    return {
        "built_at": built_at,
        "config": {
            "matched_lap_config": asdict(config),
            "target_weather_bucket": "dry",
            "source_observations": (
                "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
            ),
        },
        "summary": {
            "pair_seasons": len(targets),
            "pair_seasons_with_current_rows": int(pair_rows["current_rows"].gt(0).sum()),
            "pair_seasons_with_peak_rows": int(pair_rows["highest_common_best_rows"].gt(0).sum()),
            "artifact_rows": int(pair_rows["phase5_valid_rows"].sum()),
            "current_rows": int(pair_rows["current_rows"].sum()),
            "highest_common_best_rows": int(pair_rows["highest_common_best_rows"].sum()),
            "any_valid_best_rows": int(pair_rows["any_valid_best_rows"].sum()),
            "peak_row_gain_vs_current": int(
                pair_rows["highest_common_best_rows"].sum() - pair_rows["current_rows"].sum()
            ),
            "highest_common_best_abs_gt_1s_rows": _absolute_delta_count(
                session_rows["highest_common_best_delta_s"],
                threshold_s=1.0,
            ),
            "highest_common_best_abs_gt_2s_rows": _absolute_delta_count(
                session_rows["highest_common_best_delta_s"],
                threshold_s=2.0,
            ),
            "artifact_cache_delta_mismatch_count": int(
                pair_rows["artifact_cache_delta_mismatch_count"].sum()
            ),
            "current_pair_season_sd_s": _distribution(pair_rows["current_session_sd_s"]),
            "highest_common_best_pair_season_sd_s": _distribution(
                pair_rows["highest_common_best_session_sd_s"]
            ),
            "any_valid_best_pair_season_sd_s": _distribution(
                pair_rows["any_valid_best_session_sd_s"]
            ),
            "current_vs_peak_mean_shift_s": _distribution(
                pair_rows["current_vs_peak_mean_shift_s"]
            ),
            "phase5_wls_vs_equal_abs_shift_s": _distribution(
                pair_rows["phase5_wls_vs_equal_abs_shift_s"]
            ),
            "highest_common_segment_counts": {
                str(key): int(value) for key, value in segment_counts.items()
            },
        },
        "largest_peak_row_gains": _largest_peak_row_gains(pair_rows),
        "largest_abs_peak_session_deltas": _largest_abs_peak_session_deltas(session_rows),
    }


def write_network_peak_outputs(
    report: dict[str, Any],
    session_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write network-wide peak-probe CSV, JSON, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = output_dir / "qualifying_network_session_rows.csv"
    pair_path = output_dir / "qualifying_network_pair_seasons.csv"
    report_path = output_dir / "qualifying_network_peak_probe.json"
    text_path = output_dir / "qualifying_network_peak_probe.md"
    session_rows.to_csv(session_path, index=False)
    pair_rows.to_csv(pair_path, index=False)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    text_path.write_text(format_network_peak_report(report), encoding="utf-8")
    return {
        "session_rows": str(session_path),
        "pair_rows": str(pair_path),
        "json": str(report_path),
        "markdown": str(text_path),
    }


def format_network_peak_summary(report: dict[str, Any], *, written_paths: dict[str, str]) -> str:
    """Format a compact terminal summary for one network-wide peak probe."""
    summary = report["summary"]
    return "\n".join(
        [
            "# Qualifying Network Peak Probe",
            f"- Built at: {report['built_at']}",
            f"- Pair-seasons: {summary['pair_seasons']}",
            (
                "- Current / highest-common-best rows: "
                f"{summary['current_rows']} / {summary['highest_common_best_rows']}"
            ),
            f"- Peak row gain vs current: {summary['peak_row_gain_vs_current']}",
            (f"- Artifact/cache mismatches: {summary['artifact_cache_delta_mismatch_count']}"),
            f"- Wrote: {written_paths['markdown']}",
        ]
    )


def format_network_peak_report(report: dict[str, Any]) -> str:
    """Format the network-wide peak-probe report as compact Markdown."""
    summary = report["summary"]
    lines = [
        "# Qualifying Network Peak Probe",
        "",
        f"Built at: `{report['built_at']}`",
        "",
        "## Coverage",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Pair-seasons | {summary['pair_seasons']} |",
        f"| Pair-seasons with current rows | {summary['pair_seasons_with_current_rows']} |",
        (
            "| Pair-seasons with highest-common-best rows | "
            f"{summary['pair_seasons_with_peak_rows']} |"
        ),
        f"| Phase 5 artifact rows | {summary['artifact_rows']} |",
        f"| Current construct rows | {summary['current_rows']} |",
        f"| Highest-common-best rows | {summary['highest_common_best_rows']} |",
        f"| Any-valid-best rows | {summary['any_valid_best_rows']} |",
        f"| Peak row gain vs current | {summary['peak_row_gain_vs_current']} |",
        (
            "| Highest-common-best rows with absolute delta above `1s` | "
            f"{summary['highest_common_best_abs_gt_1s_rows']} |"
        ),
        (
            "| Highest-common-best rows with absolute delta above `2s` | "
            f"{summary['highest_common_best_abs_gt_2s_rows']} |"
        ),
        (f"| Artifact/cache mismatch rows | {summary['artifact_cache_delta_mismatch_count']} |"),
        "",
        "## Pair-Season Dispersion",
        "",
        "| Construct | Count | Median SD | P75 SD | Max SD |",
        "| --- | ---: | ---: | ---: | ---: |",
        _distribution_row("Current construct", summary["current_pair_season_sd_s"]),
        _distribution_row(
            "Highest-common best",
            summary["highest_common_best_pair_season_sd_s"],
        ),
        _distribution_row("Any-valid best", summary["any_valid_best_pair_season_sd_s"]),
        "",
        "## Pair-Season Mean Shifts",
        "",
        "| Measure | Count | Median | P75 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        _distribution_row(
            "Peak minus current equal mean",
            summary["current_vs_peak_mean_shift_s"],
        ),
        _distribution_row(
            "Phase 5 WLS vs equal absolute shift",
            summary["phase5_wls_vs_equal_abs_shift_s"],
        ),
        "",
        "## Highest Common Segment Mix",
        "",
        "| Segment | Rows |",
        "| --- | ---: |",
    ]
    for segment, count in summary["highest_common_segment_counts"].items():
        lines.append(f"| `{segment}` | {count} |")
    lines.extend(
        [
            "",
            "## Largest Peak Row Gains",
            "",
            "| Pair-season | Current rows | Peak rows | Gain |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["largest_peak_row_gains"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target_id']}`",
                    str(row["current_rows"]),
                    str(row["highest_common_best_rows"]),
                    str(row["peak_row_gain_vs_current"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Largest Absolute Peak Session Deltas",
            "",
            "| Pair-season | Race | Segment | Current delta | Peak delta |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in report["largest_abs_peak_session_deltas"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target_id']}`",
                    str(row["race_name"]),
                    f"`{row['highest_common_segment']}`",
                    _seconds(row["current_construct_delta_s"]),
                    _seconds(row["highest_common_best_delta_s"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Current construct` is the existing multi-run median selected by the extractor.",
            "- `Highest-common best` is one best-lap delta from the highest valid common segment.",
            "- Dispersion rows summarize within-pair-season session SDs, not pooled raw deltas.",
        ]
    )
    return "\n".join(lines) + "\n"


def _pair_season_summary(
    *,
    observations: pd.DataFrame,
    session_rows: pd.DataFrame,
    target: QualifyingNetworkTarget,
) -> dict[str, Any]:
    """Summarize one stored qualifying pair-season across all sessions."""
    phase5 = _phase5_target_rows(observations, target)
    cache = session_rows[session_rows["target_id"].eq(target.target_id)].copy()
    valid_phase5 = phase5[phase5["phase5_delta_s"].notna()].copy()
    current_rows = cache[cache["current_construct_delta_s"].notna()].copy()
    peak_rows = cache[cache["highest_common_best_delta_s"].notna()].copy()
    any_rows = cache[cache["any_valid_best_delta_s"].notna()].copy()
    phase5_wls_mean = _weighted_phase5_mean(valid_phase5)
    phase5_equal_mean = _mean_or_none(valid_phase5["phase5_delta_s"])
    current_mean = _mean_or_none(current_rows["current_construct_delta_s"])
    peak_mean = _mean_or_none(peak_rows["highest_common_best_delta_s"])
    return {
        "target_id": target.target_id,
        "year": target.year,
        "team": target.team,
        "reference_driver": target.reference_driver,
        "comparison_driver": target.comparison_driver,
        "phase5_valid_rows": int(len(valid_phase5)),
        "phase5_wls_mean_delta_s": phase5_wls_mean,
        "phase5_equal_mean_delta_s": phase5_equal_mean,
        "phase5_wls_vs_equal_abs_shift_s": _absolute_difference(
            phase5_wls_mean,
            phase5_equal_mean,
        ),
        "current_rows": int(len(current_rows)),
        "current_equal_mean_delta_s": current_mean,
        "current_session_sd_s": _std_or_none(current_rows["current_construct_delta_s"]),
        "highest_common_best_rows": int(len(peak_rows)),
        "highest_common_best_equal_mean_delta_s": peak_mean,
        "highest_common_best_session_sd_s": _std_or_none(peak_rows["highest_common_best_delta_s"]),
        "any_valid_best_rows": int(len(any_rows)),
        "any_valid_best_equal_mean_delta_s": _mean_or_none(any_rows["any_valid_best_delta_s"]),
        "any_valid_best_session_sd_s": _std_or_none(any_rows["any_valid_best_delta_s"]),
        "peak_row_gain_vs_current": int(len(peak_rows) - len(current_rows)),
        "current_vs_peak_mean_shift_s": _difference(peak_mean, current_mean),
        "artifact_cache_delta_mismatch_count": _artifact_cache_mismatch_count(cache),
    }


def _targets_by_year(
    targets: list[QualifyingNetworkTarget],
) -> dict[int, list[QualifyingNetworkTarget]]:
    """Group pair-season targets by season."""
    grouped: dict[int, list[QualifyingNetworkTarget]] = {}
    for target in targets:
        grouped.setdefault(target.year, []).append(target)
    return grouped


def _artifact_session_row(
    observations: pd.DataFrame,
    target: QualifyingNetworkTarget,
    *,
    race_name: str,
) -> dict[str, Any]:
    """Return stored Phase 5 artifact fields for one pair-season session."""
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
    target: QualifyingNetworkTarget,
) -> pd.DataFrame:
    """Return stored dry qualifying rows for one stored pair-season."""
    rows = observations[
        observations["year"].eq(target.year)
        & observations["team"].eq(target.team)
        & observations["session_kind"].eq("qualifying")
        & observations["weather_bucket"].eq("dry")
        & (
            (
                observations["reference_driver_code"].eq(target.reference_driver)
                & observations["comparison_driver_code"].eq(target.comparison_driver)
            )
            | (
                observations["reference_driver_code"].eq(target.comparison_driver)
                & observations["comparison_driver_code"].eq(target.reference_driver)
            )
        )
    ].copy()
    signs = np.where(
        rows["reference_driver_code"].eq(target.reference_driver),
        1.0,
        -1.0,
    )
    rows["phase5_delta_s"] = pd.to_numeric(rows["matched_gap_median_s"], errors="coerce") * signs
    return rows


def _weighted_phase5_mean(rows: pd.DataFrame) -> float | None:
    """Return the current Phase 5 WLS mean for one stored pair-season."""
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


def _largest_peak_row_gains(pair_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return pair-seasons with the largest peak-construct coverage gains."""
    if pair_rows.empty:
        return []
    rows = pair_rows.sort_values(
        ["peak_row_gain_vs_current", "highest_common_best_rows", "target_id"],
        ascending=[False, False, True],
    ).head(10)
    return [
        {
            "target_id": str(row["target_id"]),
            "current_rows": int(row["current_rows"]),
            "highest_common_best_rows": int(row["highest_common_best_rows"]),
            "peak_row_gain_vs_current": int(row["peak_row_gain_vs_current"]),
        }
        for _, row in rows.iterrows()
    ]


def _largest_abs_peak_session_deltas(session_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return sessions with the largest absolute highest-common-best deltas."""
    if session_rows.empty:
        return []
    rows = session_rows[session_rows["highest_common_best_delta_s"].notna()].copy()
    if rows.empty:
        return []
    rows["abs_peak_delta_s"] = rows["highest_common_best_delta_s"].abs()
    rows = rows.sort_values(
        ["abs_peak_delta_s", "target_id", "race_name"],
        ascending=[False, True, True],
    ).head(10)
    return [
        {
            "target_id": str(row["target_id"]),
            "race_name": str(row["race_name"]),
            "highest_common_segment": str(row["highest_common_segment"]),
            "current_construct_delta_s": _none_if_na(row["current_construct_delta_s"]),
            "highest_common_best_delta_s": _none_if_na(row["highest_common_best_delta_s"]),
        }
        for _, row in rows.iterrows()
    ]


def _absolute_delta_count(values: pd.Series, *, threshold_s: float) -> int:
    """Count non-null deltas whose absolute value exceeds one threshold."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return int(numeric.abs().gt(threshold_s).sum())


def _distribution(values: pd.Series) -> dict[str, Any]:
    """Return compact numeric distribution statistics."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "median": None, "p75": None, "max": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _distribution_row(label: str, distribution: dict[str, Any]) -> str:
    """Format one distribution row for Markdown."""
    return (
        f"| {label} | {distribution['count']} | {_seconds(distribution['median'])} | "
        f"{_seconds(distribution['p75'])} | {_seconds(distribution['max'])} |"
    )


def _mean_or_none(values: pd.Series) -> float | None:
    """Return a numeric mean or None for an empty series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.mean())


def _std_or_none(values: pd.Series) -> float | None:
    """Return a sample standard deviation or None without two numeric rows."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if len(numeric) < 2 else float(numeric.std(ddof=1))


def _difference(left: float | None, right: float | None) -> float | None:
    """Return one signed difference when both values are present."""
    if left is None or right is None:
        return None
    return float(left - right)


def _absolute_difference(left: float | None, right: float | None) -> float | None:
    """Return one absolute difference when both values are present."""
    value = _difference(left, right)
    return None if value is None else abs(value)


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
