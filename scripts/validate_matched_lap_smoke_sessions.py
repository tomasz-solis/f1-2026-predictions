"""Run Phase 4 smoke validation for the matched-lap extractor.

This script loads the locked smoke sessions, runs the canonical extractor,
and writes compact JSON/text evidence with routine filter diagnostics.
It belongs to Phase 4, not Phase 2: unlike the read-only inspector, it
uses matching, weather routing, and extractor filters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.matched_laps import (  # noqa: E402
    MatchedLapConfig,
    diagnose_matched_lap_filters,
    extract_matched_teammate_laps,
)


@dataclass(frozen=True)
class SmokeSessionSpec:
    """One locked smoke session and its high-level expectations."""

    category: str
    year: int
    event_name: str
    session_code: str
    session_kind: str
    weather_mode: str
    min_matched_pairs: int
    max_matched_pairs: int | None


SMOKE_SESSIONS: tuple[SmokeSessionSpec, ...] = (
    SmokeSessionSpec(
        category="clean_dry_race",
        year=2024,
        event_name="Bahrain Grand Prix",
        session_code="R",
        session_kind="race",
        weather_mode="dry",
        min_matched_pairs=80,
        max_matched_pairs=600,
    ),
    SmokeSessionSpec(
        category="wet_mixed_race",
        year=2024,
        event_name="British Grand Prix",
        session_code="R",
        session_kind="race",
        weather_mode="mixed",
        min_matched_pairs=20,
        max_matched_pairs=None,
    ),
    SmokeSessionSpec(
        category="early_teammate_dnf",
        year=2024,
        event_name="Australian Grand Prix",
        session_code="R",
        session_kind="race",
        weather_mode="dry",
        min_matched_pairs=60,
        max_matched_pairs=None,
    ),
    SmokeSessionSpec(
        category="strategy_asymmetric_race",
        year=2024,
        event_name="Miami Grand Prix",
        session_code="R",
        session_kind="race",
        weather_mode="dry",
        min_matched_pairs=50,
        max_matched_pairs=None,
    ),
    SmokeSessionSpec(
        category="representative_qualifying",
        year=2024,
        event_name="Bahrain Grand Prix",
        session_code="Q",
        session_kind="qualifying",
        weather_mode="dry",
        min_matched_pairs=20,
        max_matched_pairs=None,
    ),
)


def run_smoke_validation(
    *,
    cache_dir: Path,
    output_dir: Path,
    offline: bool,
    sessions: tuple[SmokeSessionSpec, ...] = SMOKE_SESSIONS,
    config: MatchedLapConfig | None = None,
) -> list[dict[str, Any]]:
    """Run extractor smoke validation and write evidence files."""
    config = config or MatchedLapConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for spec in sessions:
        session = _load_fastf1_session(spec, cache_dir=cache_dir, offline=offline)
        matched = extract_matched_teammate_laps(
            session,
            session_kind=spec.session_kind,  # type: ignore[arg-type]
            weather_mode=spec.weather_mode,  # type: ignore[arg-type]
            config=config,
        )
        diagnostics = diagnose_matched_lap_filters(
            session,
            session_kind=spec.session_kind,  # type: ignore[arg-type]
            weather_mode=spec.weather_mode,  # type: ignore[arg-type]
            config=config,
        )
        summary = build_smoke_summary(spec, matched, diagnostics)
        results.append(summary)

        stem = f"{spec.year}_{spec.category}"
        (output_dir / f"{stem}.json").write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{stem}.txt").write_text(
            format_smoke_summary(summary) + "\n",
            encoding="utf-8",
        )

        print(format_smoke_summary(summary))
        print(f"  wrote {stem}.json and {stem}.txt\n")

    return results


def build_smoke_summary(
    spec: SmokeSessionSpec,
    matched: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise matched rows, skip reasons, and filter diagnostics."""
    matched_pairs = int(matched["row_type"].eq("matched_pair").sum())
    skipped_pairs = int(matched["row_type"].eq("skipped_pair").sum())
    weather_counts = _value_counts(matched, "weather_bucket")
    skip_counts = _value_counts(
        matched[matched["row_type"].eq("skipped_pair")],
        "skip_reason",
    )
    filter_totals = {
        "non_green_laps": _sum_column(diagnostics, "non_green_laps"),
        "sc_vsc_laps": _sum_column(diagnostics, "sc_vsc_laps"),
        "pit_laps": _sum_column(diagnostics, "pit_laps"),
        "stint_outlier_laps": _sum_column(diagnostics, "stint_outlier_laps"),
        "lap_level_weather_unreliable_laps": _sum_column(
            diagnostics,
            "lap_level_weather_unreliable_laps",
        ),
        "weather_mode_excluded_laps": _sum_column(diagnostics, "weather_mode_excluded_laps"),
    }
    status, notes = _smoke_status(spec, matched_pairs, weather_counts, skip_counts, filter_totals)
    return {
        "spec": asdict(spec),
        "status": status,
        "notes": notes,
        "matched_pairs": matched_pairs,
        "skipped_pairs": skipped_pairs,
        "weather_counts": weather_counts,
        "skip_counts": skip_counts,
        "filter_totals": filter_totals,
        "team_diagnostics": diagnostics.to_dict(orient="records"),
    }


def format_smoke_summary(summary: dict[str, Any]) -> str:
    """Format one smoke-validation summary for terminal and text files."""
    spec = summary["spec"]
    filters = summary["filter_totals"]
    lines = [
        f"=== {spec['year']} {spec['event_name']} ({spec['category']}) ===",
        f"status: {summary['status']}",
        f"matched_pairs={summary['matched_pairs']}, skipped_pairs={summary['skipped_pairs']}",
        f"weather_counts={summary['weather_counts']}",
        f"skip_counts={summary['skip_counts']}",
        "filter_totals="
        f"non_green={filters['non_green_laps']}, "
        f"sc_vsc={filters['sc_vsc_laps']}, "
        f"pit={filters['pit_laps']}, "
        f"stint_outlier={filters['stint_outlier_laps']}, "
        f"weather_unreliable={filters['lap_level_weather_unreliable_laps']}, "
        f"weather_mode_excluded={filters['weather_mode_excluded_laps']}",
    ]
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without loading FastF1 sessions."""
    parser = argparse.ArgumentParser(description="Validate matched-lap extractor smoke sessions.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/.fastf1_cache"),
        help="FastF1 cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/matched_lap_extractor_smoke"),
        help="Where to write smoke validation JSON and text evidence.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow FastF1 network reads instead of forcing local-cache offline mode.",
    )
    return parser


def main() -> None:
    """Argparse entry point for smoke validation."""
    args = build_parser().parse_args()
    run_smoke_validation(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        offline=not args.online,
    )


def _load_fastf1_session(spec: SmokeSessionSpec, *, cache_dir: Path, offline: bool) -> Any:
    """Load one FastF1 session for smoke validation."""
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError("fastf1 is required for smoke validation") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    if offline and hasattr(fastf1.Cache, "offline_mode"):
        fastf1.Cache.offline_mode(True)

    session = fastf1.get_session(spec.year, spec.event_name, spec.session_code)
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def _smoke_status(
    spec: SmokeSessionSpec,
    matched_pairs: int,
    weather_counts: dict[str, int],
    skip_counts: dict[str, int],
    filter_totals: dict[str, int],
) -> tuple[str, list[str]]:
    """Return pass/fail/partial status and concrete notes."""
    status = "pass"
    notes: list[str] = []

    if matched_pairs < spec.min_matched_pairs:
        status = "fail"
        notes.append(f"matched_pairs below floor {spec.min_matched_pairs}")
    if spec.max_matched_pairs is not None and matched_pairs > spec.max_matched_pairs:
        status = "fail"
        notes.append(f"matched_pairs above ceiling {spec.max_matched_pairs}")

    if spec.category == "clean_dry_race":
        if set(weather_counts) != {"dry"}:
            status = "fail"
            notes.append("clean dry race produced non-dry weather buckets")
        if skip_counts:
            status = "fail"
            notes.append("clean dry race emitted skipped pairs")
    elif spec.category == "wet_mixed_race":
        if weather_counts.get("dry", 0) < 20 or weather_counts.get("wet", 0) == 0:
            status = "fail"
            notes.append("wet/mixed race did not produce both dry and wet matched evidence")
        if filter_totals["lap_level_weather_unreliable_laps"] == 0:
            status = _weaker_status(status)
            notes.append("no unreliable weather laps were counted")
    elif spec.category == "early_teammate_dnf":
        if (
            "insufficient_matched_pairs" not in skip_counts
            and "teammate_dnf_no_matched_laps" not in skip_counts
        ):
            status = "fail"
            notes.append("early-DNF session did not emit a no-update skip reason")
    elif spec.category == "strategy_asymmetric_race":
        if filter_totals["sc_vsc_laps"] == 0 or filter_totals["non_green_laps"] == 0:
            status = "fail"
            notes.append("strategy-asymmetric race did not expose SC/VSC filtering")
        if filter_totals["pit_laps"] == 0:
            status = "fail"
            notes.append("strategy-asymmetric race did not expose pit-window filtering")
        if filter_totals["stint_outlier_laps"] == 0:
            status = _weaker_status(status)
            notes.append("no stint-outlier laps were counted")
    elif spec.category == "representative_qualifying":
        if "no_common_quali_segment" in skip_counts:
            status = "fail"
            notes.append("representative qualifying unexpectedly emitted no_common_quali_segment")
        if weather_counts.get("dry", 0) < spec.min_matched_pairs:
            status = "fail"
            notes.append("representative qualifying did not clear dry matched-pair floor")

    if not notes:
        notes.append("checks satisfied")
    return status, notes


def _weaker_status(status: str) -> str:
    """Downgrade pass to partial without hiding failures."""
    return "partial" if status == "pass" else status


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return JSON-friendly value counts for one DataFrame column."""
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].dropna().astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    """Return an integer sum for a possibly missing diagnostics column."""
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _json_safe(value: Any) -> Any:
    """Convert pandas and pathlib values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
