"""CLI entry point for Phase 2 smoke-session inspection.

For each smoke-session candidate from
``docs/design/matched_lap_extractor_smoke_sessions.md``, this script
loads the session via FastF1, runs the read-only inspector, and writes
both a structured JSON summary and a short text summary to the
configured output directory.

The outputs are evidence the analyst uses to fill the smoke-session
doc's expected behavior table. Per master execution plan Phase 2's
read-only exception, this script must not include matching,
skip-reason, or weather-routing logic.

Usage
-----
::

    python scripts/inspect_smoke_sessions.py \\
        --cache-dir data/raw/.fastf1_cache \\
        --output-dir data/diagnostics/smoke_session_inspections

The session list is hard-coded below. To inspect a different session,
either edit ``SMOKE_SESSIONS`` or import ``run_inspections`` from this
module and pass a custom session list.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.smoke_inspector.inspector import (  # noqa: E402
    SessionInspection,
    inspect_session,
)
from src.diagnostics.smoke_inspector.loader import load_session  # noqa: E402

SMOKE_SESSIONS: list[dict[str, Any]] = [
    {
        "category": "clean_dry_race",
        "year": 2024,
        "event_name": "Bahrain Grand Prix",
        "session_kind": "race",
    },
    {
        "category": "wet_mixed_race",
        "year": 2024,
        "event_name": "British Grand Prix",
        "session_kind": "race",
    },
    {
        "category": "early_teammate_dnf",
        "year": 2024,
        "event_name": "Australian Grand Prix",
        "session_kind": "race",
    },
    {
        "category": "strategy_asymmetric_race",
        "year": 2024,
        "event_name": "Miami Grand Prix",
        "session_kind": "race",
    },
    {
        "category": "representative_qualifying",
        "year": 2024,
        "event_name": "Bahrain Grand Prix",
        "session_kind": "qualifying",
    },
]


def _to_serialisable(obj: Any) -> Any:
    """Recursively convert dataclasses, sets, and timedeltas to JSON-safe types."""
    if is_dataclass(obj):
        return {k: _to_serialisable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [_to_serialisable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def format_text_summary(inspection: SessionInspection) -> str:
    """Produce a short human-readable summary of one inspection.

    Used by the CLI for stdout output and for the ``.txt`` evidence
    file. Kept deliberately compact so the analyst can scan five
    sessions on one screen. Track status values are raw FastF1 status
    row counts, not incident counts; one VSC incident may emit multiple
    status rows.
    """
    lines = [
        f"=== {inspection.year} {inspection.event_name} ({inspection.session_kind}) ===",
        f"weather: samples={inspection.weather.n_samples}, "
        f"rain_true={inspection.weather.n_rainfall_true}, "
        f"rain_false={inspection.weather.n_rainfall_false}, "
        f"mixed={inspection.weather.is_mixed}",
        f"lap counts: drivers={len(inspection.lap_counts.by_driver)}, "
        f"max_lap={inspection.lap_counts.max_observed_lap}, "
        f"partial_distance={len(inspection.lap_counts.partial_distance_drivers)}",
        f"retirements: {len(inspection.retirements.retired_drivers)} driver(s)",
    ]
    for code, rec in inspection.retirements.retired_drivers.items():
        lines.append(
            f"  {code} ({rec.team}): last_lap={rec.retirement_lap}, "
            f"early={rec.is_early}, status={rec.classified_status}"
        )
    lines.append(
        f"track status rows: SC_rows={inspection.track_status.n_safety_car_rows}, "
        f"VSC_rows={inspection.track_status.n_virtual_safety_car_rows}, "
        f"red_rows={inspection.track_status.n_red_flag_rows}, "
        f"yellow_rows={inspection.track_status.n_yellow_rows}"
    )
    if inspection.qualifying is not None:
        lines.append(
            f"qualifying: teams_q3={len(inspection.qualifying.teams_with_q3)}, "
            f"teams_q1_only="
            f"{len(inspection.qualifying.teams_with_q1_eliminated)}, "
            f"teams_split={len(inspection.qualifying.teams_with_split_segments)}"
        )
    return "\n".join(lines)


def run_inspections(
    cache_dir: Path,
    output_dir: Path,
    sessions: list[dict[str, Any]] | None = None,
) -> list[SessionInspection]:
    """Inspect each smoke-candidate session and write evidence files.

    Parameters
    ----------
    cache_dir:
        FastF1 cache directory passed through to the loader.
    output_dir:
        Destination for ``.json`` and ``.txt`` evidence files. Created
        if missing.
    sessions:
        Optional override for the session list. Defaults to
        ``SMOKE_SESSIONS``.

    Returns
    -------
    list[SessionInspection]
        Inspections in the same order as the input session list.
    """
    sessions = sessions or SMOKE_SESSIONS
    output_dir.mkdir(parents=True, exist_ok=True)

    inspections: list[SessionInspection] = []
    for spec in sessions:
        category = spec["category"]
        session = load_session(
            spec["year"],
            spec["event_name"],
            spec["session_kind"],
            cache_dir=cache_dir,
        )
        inspection = inspect_session(
            year=spec["year"],
            event_name=spec["event_name"],
            session_kind=spec["session_kind"],
            laps_df=session.laps,
            results_df=session.results,
            weather_df=session.weather_data,
            track_status_df=session.track_status,
        )
        inspections.append(inspection)

        json_path = output_dir / f"{spec['year']}_{category}.json"
        json_path.write_text(
            json.dumps(_to_serialisable(inspection), indent=2, default=str),
            encoding="utf-8",
        )

        text_summary = format_text_summary(inspection)
        txt_path = output_dir / f"{spec['year']}_{category}.txt"
        txt_path.write_text(text_summary + "\n", encoding="utf-8")

        print(text_summary)
        print(f"  wrote {json_path.name} and {txt_path.name}\n")

    return inspections


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without executing inspections.

    Keeping parser construction separate lets tests verify defaults
    without loading FastF1 sessions.
    """
    parser = argparse.ArgumentParser(description="Phase 2 read-only smoke-session inspection.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/.fastf1_cache"),
        help="FastF1 cache directory (default: data/raw/.fastf1_cache).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/smoke_session_inspections"),
        help="Where to write inspection JSON and text outputs.",
    )
    return parser


def main() -> None:
    """Argparse-driven command-line entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_inspections(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
