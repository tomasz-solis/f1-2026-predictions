"""Build bulk matched-lap observations for the teammate-network prior.

This is the Phase 5 runner. It loads historical Race and Qualifying sessions,
runs the canonical matched-lap extractor, writes observation caches, and
emits the diagnostic dump required before prior fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.matched_laps import (  # noqa: E402
    MatchedLapConfig,
    WeatherMode,
    aggregate_matched_teammate_laps,
    diagnose_matched_lap_filters,
    extract_matched_teammate_laps,
)
from src.utils.weekend import should_skip_schedule_event  # noqa: E402

SessionKind = Literal["race", "qualifying"]


@dataclass(frozen=True)
class BulkSessionSpec:
    """One historical session targeted for bulk matched-lap extraction."""

    year: int
    race_name: str
    event_format: str
    session_code: str
    session_kind: SessionKind
    weather_mode: WeatherMode = "mixed"

    @property
    def session_label(self) -> str:
        """Return a stable human-readable session label."""
        return "Race" if self.session_kind == "race" else "Qualifying"


@dataclass(frozen=True)
class ExtractionError:
    """One failed session load or extraction attempt."""

    year: int
    race_name: str
    session_kind: str
    error_type: str
    message: str


def build_historical_observations(
    *,
    years: list[int],
    cache_dir: Path,
    output_dir: Path,
    offline: bool,
    config: MatchedLapConfig | None = None,
    limit_sessions: int | None = None,
) -> dict[str, Any]:
    """Build historical matched-lap caches and return the diagnostic report."""
    config = config or MatchedLapConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_specs = discover_bulk_sessions(years=years, cache_dir=cache_dir, offline=offline)
    if limit_sessions is not None:
        session_specs = session_specs[:limit_sessions]

    raw_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    errors: list[ExtractionError] = []

    for index, spec in enumerate(session_specs, start=1):
        print(f"[{index}/{len(session_specs)}] {spec.year} {spec.race_name} {spec.session_label}")
        try:
            session = load_fastf1_session(spec, cache_dir=cache_dir, offline=offline)
            raw = extract_matched_teammate_laps(
                session,
                session_kind=spec.session_kind,
                weather_mode=spec.weather_mode,
                config=config,
            )
            aggregate = aggregate_matched_teammate_laps(raw, config=config)
            diagnostics = diagnose_matched_lap_filters(
                session,
                session_kind=spec.session_kind,
                weather_mode=spec.weather_mode,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - bulk extraction must record and continue
            errors.append(
                ExtractionError(
                    year=spec.year,
                    race_name=spec.race_name,
                    session_kind=spec.session_kind,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            print(f"  error: {type(exc).__name__}: {exc}")
            continue

        raw_frames.append(raw)
        aggregate_frames.append(aggregate)
        diagnostic_frames.append(diagnostics)

    raw_all = _concat_or_empty(raw_frames)
    aggregate_all = _concat_or_empty(aggregate_frames)
    diagnostics_all = _concat_or_empty(diagnostic_frames)

    report = build_diagnostic_report(
        session_specs=session_specs,
        raw_observations=raw_all,
        aggregate_observations=aggregate_all,
        filter_diagnostics=diagnostics_all,
        errors=errors,
        config=config,
    )
    write_observation_outputs(
        output_dir=output_dir,
        raw_observations=raw_all,
        aggregate_observations=aggregate_all,
        filter_diagnostics=diagnostics_all,
        report=report,
    )
    return report


def discover_bulk_sessions(
    *,
    years: list[int],
    cache_dir: Path,
    offline: bool,
) -> list[BulkSessionSpec]:
    """Discover historical Race and Qualifying sessions from FastF1 schedules."""
    fastf1 = _configure_fastf1(cache_dir=cache_dir, offline=offline)
    specs: list[BulkSessionSpec] = []

    for year in years:
        schedule = fastf1.get_event_schedule(year)
        for _, event in schedule.iterrows():
            race_name = str(event.get("EventName", "")).strip()
            event_format = str(event.get("EventFormat", "")).strip().lower()
            if should_skip_schedule_event(year, race_name):
                continue
            if "testing" in event_format:
                continue
            specs.append(
                BulkSessionSpec(
                    year=year,
                    race_name=race_name,
                    event_format=event_format,
                    session_code="R",
                    session_kind="race",
                )
            )
            specs.append(
                BulkSessionSpec(
                    year=year,
                    race_name=race_name,
                    event_format=event_format,
                    session_code="Q",
                    session_kind="qualifying",
                )
            )

    return specs


def load_fastf1_session(spec: BulkSessionSpec, *, cache_dir: Path, offline: bool) -> Any:
    """Load one FastF1 session for bulk extraction."""
    fastf1 = _configure_fastf1(cache_dir=cache_dir, offline=offline)
    session = fastf1.get_session(spec.year, spec.race_name, spec.session_code)
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def build_diagnostic_report(
    *,
    session_specs: list[BulkSessionSpec],
    raw_observations: pd.DataFrame,
    aggregate_observations: pd.DataFrame,
    filter_diagnostics: pd.DataFrame,
    errors: list[ExtractionError],
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Build the Phase 5 diagnostic report from extraction outputs."""
    matched = _matched_rows(raw_observations)
    skipped = _skipped_rows(raw_observations)
    valid_aggregates = _valid_aggregate_rows(aggregate_observations)

    return {
        "built_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "n_target_sessions": len(session_specs),
        "n_loaded_sessions": _loaded_session_count(raw_observations),
        "n_error_sessions": len(errors),
        "n_raw_rows": int(len(raw_observations)),
        "n_matched_pair_rows": int(len(matched)),
        "n_skipped_pair_rows": int(len(skipped)),
        "n_aggregate_rows": int(len(aggregate_observations)),
        "n_valid_aggregate_rows": int(len(valid_aggregates)),
        "errors": [asdict(error) for error in errors],
        "n_matched_pairs_distribution": _matched_pair_distribution(valid_aggregates),
        "matched_gap_se_distribution": _numeric_distribution(
            valid_aggregates,
            "matched_gap_se_s",
        ),
        "matched_gap_median_distribution": _numeric_distribution(
            valid_aggregates,
            "matched_gap_median_s",
        ),
        "skip_reason_counts": _count_records(skipped, ["session_kind", "skip_reason"]),
        "zero_observation_sessions": _zero_observation_sessions(session_specs, raw_observations),
        "teammate_pair_coverage": _teammate_pair_coverage(valid_aggregates),
        "weather_bucket_counts": _count_records(matched, ["session_kind", "weather_bucket"]),
        "compound_counts": _count_records(matched, ["session_kind", "compound"]),
        "compound_overlap_counts": _compound_overlap_counts(filter_diagnostics),
        "connected_components": _connected_component_summary(valid_aggregates),
        "filter_totals": _filter_totals(filter_diagnostics),
        "session_team_pair_counts": _session_team_pair_counts(valid_aggregates),
    }


def write_observation_outputs(
    *,
    output_dir: Path,
    raw_observations: pd.DataFrame,
    aggregate_observations: pd.DataFrame,
    filter_diagnostics: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    """Write bulk observation caches and diagnostic files."""
    raw_path = output_dir / "raw_matched_laps.csv"
    aggregate_path = output_dir / "aggregated_observations.csv"
    filters_path = output_dir / "filter_diagnostics.csv"
    report_path = output_dir / "diagnostic_report.json"
    text_path = output_dir / "diagnostic_report.md"

    raw_observations.to_csv(raw_path, index=False)
    aggregate_observations.to_csv(aggregate_path, index=False)
    filter_diagnostics.to_csv(filters_path, index=False)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    text_path.write_text(format_diagnostic_report(report), encoding="utf-8")


def format_diagnostic_report(report: dict[str, Any]) -> str:
    """Format the bulk diagnostic report as compact Markdown."""
    lines = [
        "# Matched-Lap Bulk Extraction Diagnostics",
        "",
        f"- Built at: {report['built_at']}",
        f"- Target sessions: {report['n_target_sessions']}",
        f"- Loaded sessions: {report['n_loaded_sessions']}",
        f"- Error sessions: {report['n_error_sessions']}",
        f"- Raw rows: {report['n_raw_rows']}",
        f"- Matched-pair rows: {report['n_matched_pair_rows']}",
        f"- Skipped-pair rows: {report['n_skipped_pair_rows']}",
        f"- Valid aggregate rows: {report['n_valid_aggregate_rows']}",
        "",
        "## Matched-Pair Distribution",
        "",
        _json_block(report["n_matched_pairs_distribution"]),
        "",
        "## Matched-Gap SE Distribution",
        "",
        _json_block(report["matched_gap_se_distribution"]),
        "",
        "## Skip Reasons",
        "",
        _json_block(report["skip_reason_counts"]),
        "",
        "## Zero-Observation Sessions",
        "",
        _json_block(report["zero_observation_sessions"]),
        "",
        "## Weather Buckets",
        "",
        _json_block(report["weather_bucket_counts"]),
        "",
        "## Connected Components",
        "",
        _json_block(report["connected_components"]),
        "",
        "## Extraction Errors",
        "",
        _json_block(report["errors"]),
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Phase 5 runner."""
    parser = argparse.ArgumentParser(description="Build historical matched-lap observations.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Season years to extract.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/.fastf1_cache"),
        help="FastF1 cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/teammate_network_observations/latest"),
        help="Output directory for observation caches and diagnostics.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow FastF1 network reads instead of forcing local-cache offline mode.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Optional development limit for the number of discovered sessions.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples used for aggregate median SE estimates.",
    )
    return parser


def main() -> None:
    """Run Phase 5 bulk extraction from command-line arguments."""
    args = build_parser().parse_args()
    report = build_historical_observations(
        years=args.years,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        offline=not args.online,
        config=MatchedLapConfig(bootstrap_samples=args.bootstrap_samples),
        limit_sessions=args.limit_sessions,
    )
    print(format_diagnostic_report(report))


def _configure_fastf1(*, cache_dir: Path, offline: bool) -> Any:
    """Import FastF1, configure its cache, and optionally force offline mode."""
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError("fastf1 is required for matched-lap bulk extraction") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    if offline and hasattr(fastf1.Cache, "offline_mode"):
        fastf1.Cache.offline_mode(True)
    return fastf1


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames while preserving an empty-frame fallback."""
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries",
            category=FutureWarning,
        )
        return pd.concat(non_empty, ignore_index=True)


def _matched_rows(raw_observations: pd.DataFrame) -> pd.DataFrame:
    """Return matched-pair rows from the raw observation table."""
    if raw_observations.empty or "row_type" not in raw_observations.columns:
        return pd.DataFrame()
    return raw_observations[raw_observations["row_type"].eq("matched_pair")].copy()


def _skipped_rows(raw_observations: pd.DataFrame) -> pd.DataFrame:
    """Return skipped-pair rows from the raw observation table."""
    if raw_observations.empty or "row_type" not in raw_observations.columns:
        return pd.DataFrame()
    return raw_observations[raw_observations["row_type"].eq("skipped_pair")].copy()


def _valid_aggregate_rows(aggregate_observations: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate rows with usable matched evidence."""
    if aggregate_observations.empty or "n_matched_pairs" not in aggregate_observations.columns:
        return pd.DataFrame()
    counts = pd.to_numeric(aggregate_observations["n_matched_pairs"], errors="coerce").fillna(0)
    usable_mask = counts > 0
    if "skip_reason" in aggregate_observations.columns:
        usable_mask &= aggregate_observations["skip_reason"].isna()
    return aggregate_observations[usable_mask].copy()


def _loaded_session_count(raw_observations: pd.DataFrame) -> int:
    """Count sessions that produced any extractor output."""
    if raw_observations.empty:
        return 0
    return int(raw_observations[["year", "race_name", "session_kind"]].drop_duplicates().shape[0])


def _matched_pair_distribution(valid_aggregates: pd.DataFrame) -> dict[str, Any]:
    """Summarise n_matched_pairs overall and by session kind/weather bucket."""
    if valid_aggregates.empty:
        return {"overall": _empty_numeric_distribution(), "by_session_kind_weather": []}

    by_group: list[dict[str, Any]] = []
    for group_key, group in valid_aggregates.groupby(
        ["session_kind", "weather_bucket"],
        dropna=False,
    ):
        session_kind, weather_bucket = group_key
        by_group.append(
            {
                "session_kind": session_kind,
                "weather_bucket": weather_bucket,
                **_numeric_distribution(group, "n_matched_pairs"),
            }
        )

    return {
        "overall": _numeric_distribution(valid_aggregates, "n_matched_pairs"),
        "by_session_kind_weather": by_group,
    }


def _numeric_distribution(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return count/min/quartile/median/max stats for one numeric column."""
    if frame.empty or column not in frame.columns:
        return _empty_numeric_distribution()
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return _empty_numeric_distribution()
    quantiles = values.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(values.max()),
    }


def _empty_numeric_distribution() -> dict[str, Any]:
    """Return a consistent empty numeric distribution shape."""
    return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}


def _count_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Return sorted count records for grouped categorical diagnostics."""
    if frame.empty or any(column not in frame.columns for column in columns):
        return []
    counts = frame.groupby(columns, dropna=False).size().reset_index(name="count")
    counts = counts.sort_values([*columns, "count"]).reset_index(drop=True)
    return counts.to_dict(orient="records")


def _zero_observation_sessions(
    session_specs: list[BulkSessionSpec],
    raw_observations: pd.DataFrame,
) -> list[dict[str, Any]]:
    """List sessions with zero matched-pair rows."""
    if raw_observations.empty:
        observed: set[tuple[int, str, str]] = set()
    else:
        matched = _matched_rows(raw_observations)
        observed = set(
            matched[["year", "race_name", "session_kind"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

    zeros: list[dict[str, Any]] = []
    for spec in session_specs:
        key = (spec.year, spec.race_name, spec.session_kind)
        if key not in observed:
            zeros.append(
                {
                    "year": spec.year,
                    "race_name": spec.race_name,
                    "session_kind": spec.session_kind,
                }
            )
    return zeros


def _teammate_pair_coverage(valid_aggregates: pd.DataFrame) -> list[dict[str, Any]]:
    """Count valid aggregate observations per teammate edge."""
    if valid_aggregates.empty:
        return []
    grouped = (
        valid_aggregates.groupby(
            [
                "session_kind",
                "weather_bucket",
                "reference_driver_code",
                "comparison_driver_code",
            ],
            dropna=False,
        )
        .agg(
            n_observations=("n_matched_pairs", "count"),
            total_matched_pairs=("n_matched_pairs", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["session_kind", "weather_bucket", "reference_driver_code", "comparison_driver_code"]
    ).to_dict(orient="records")


def _compound_overlap_counts(filter_diagnostics: pd.DataFrame) -> dict[str, int]:
    """Summarise compound-overlap failures from filter diagnostics."""
    if filter_diagnostics.empty or "skip_reason" not in filter_diagnostics.columns:
        return {"no_compound_overlap": 0}
    return {
        "no_compound_overlap": int(
            filter_diagnostics["skip_reason"].eq("no_compound_overlap").sum()
        )
    }


def _connected_component_summary(valid_aggregates: pd.DataFrame) -> list[dict[str, Any]]:
    """Build connected-component summaries by session kind and weather bucket."""
    if valid_aggregates.empty:
        return []

    summaries: list[dict[str, Any]] = []
    for group_key, group in valid_aggregates.groupby(
        ["session_kind", "weather_bucket"],
        dropna=False,
    ):
        session_kind, weather_bucket = group_key
        components = _connected_components(
            group[["reference_driver_code", "comparison_driver_code"]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        observation_counts = [
            _component_observation_count(group, component) for component in components
        ]
        total_observations = int(len(group))
        summaries.append(
            {
                "session_kind": session_kind,
                "weather_bucket": weather_bucket,
                "n_components": len(components),
                "component_sizes": [len(component) for component in components],
                "component_observation_counts": observation_counts,
                "component_observation_shares": [
                    count / total_observations if total_observations else 0.0
                    for count in observation_counts
                ],
                "components": [sorted(component) for component in components],
            }
        )
    return summaries


def _component_observation_count(group: pd.DataFrame, component: set[str]) -> int:
    """Count valid aggregate rows whose driver edge is inside one component."""
    return int(
        (
            group["reference_driver_code"].isin(component)
            & group["comparison_driver_code"].isin(component)
        ).sum()
    )


def _connected_components(edges: Any) -> list[set[str]]:
    """Return graph connected components for driver-pair edges."""
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
            stack.extend(sorted(adjacency.get(current, set()).difference(seen)))
        components.append(component)
    return sorted(components, key=lambda component: (-len(component), sorted(component)))


def _filter_totals(filter_diagnostics: pd.DataFrame) -> dict[str, int]:
    """Return session-wide sums for routine filter diagnostics."""
    columns = [
        "missing_lap_time_laps",
        "pit_laps",
        "deleted_laps",
        "inaccurate_laps",
        "lap1_laps",
        "final_driver_laps",
        "non_green_laps",
        "sc_vsc_laps",
        "large_position_change_laps",
        "stint_outlier_laps",
        "lap_level_weather_unreliable_laps",
        "weather_mode_excluded_laps",
        "non_quick_qualifying_laps",
        "valid_laps",
        "candidate_matched_pairs",
        "matched_pair_rows",
    ]
    return {column: _sum_column(filter_diagnostics, column) for column in columns}


def _session_team_pair_counts(valid_aggregates: pd.DataFrame) -> list[dict[str, Any]]:
    """Return n_matched_pairs by season, session, kind, weather, and team."""
    if valid_aggregates.empty:
        return []
    columns = [
        "year",
        "race_name",
        "session_name",
        "session_kind",
        "weather_bucket",
        "team",
        "reference_driver_code",
        "comparison_driver_code",
        "n_matched_pairs",
        "matched_gap_median_s",
        "matched_gap_se_s",
    ]
    present = [column for column in columns if column in valid_aggregates.columns]
    return valid_aggregates[present].sort_values(present[:7]).to_dict(orient="records")


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    """Return an integer sum for a possibly missing numeric column."""
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _json_block(value: Any) -> str:
    """Return a fenced JSON block for Markdown reports."""
    return "```json\n" + json.dumps(_json_safe(value), indent=2) + "\n```"


def _json_safe(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
