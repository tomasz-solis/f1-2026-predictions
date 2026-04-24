# ruff: noqa: E402
"""Build an experimental testing-derived team seed artifact.

This helper learns a preseason team-strength mapping from historical testing
telemetry and writes a normal team-characteristics payload for inspection.
It does not touch driver or track priors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.testing_team_seed import (
    build_testing_model_team_payload,
    write_validation_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_years(raw_years: str) -> tuple[int, ...]:
    """Parse comma-separated year input into an ordered tuple."""
    years = [int(year_text.strip()) for year_text in raw_years.split(",") if year_text.strip()]
    if not years:
        raise ValueError("At least one training year is required.")
    return tuple(years)


def main() -> int:
    """Build one testing-derived team seed payload from cached preseason telemetry."""
    parser = argparse.ArgumentParser(
        description="Build an experimental testing-derived preseason team seed artifact."
    )
    parser.add_argument("--year", type=int, default=2026, help="Target season year.")
    parser.add_argument(
        "--training-years",
        type=str,
        default="2022,2023,2024",
        help=(
            "Comma-separated seasons used to train the RidgeCV testing model. "
            "2025 is intentionally opt-in for 2026 reset work."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/car_characteristics/2026_car_characteristics.testing_model.json",
        help="Output JSON path for the generated team payload.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Optional JSON path for a compact validation report.",
    )
    args = parser.parse_args()

    training_years = _parse_years(args.training_years)
    payload = build_testing_model_team_payload(
        target_year=args.year,
        training_years=training_years,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote testing-derived team seed payload to %s", output_path)

    if args.report_output:
        report_path = write_validation_report(
            payload=payload,
            output_path=args.report_output,
        )
        logger.info("Wrote validation report to %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
