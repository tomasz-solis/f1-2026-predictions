"""Explain candidate rank outputs for selected scored races."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_model_candidates import (  # noqa: E402
    _blend_with_previous,
    _rolling_actual_rank,
)
from scripts.generate_evaluation_report import (  # noqa: E402
    _resolve_session_pair,
    _select_latest_predictions,
    _sort_selected_predictions,
)

from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402
from src.utils.prediction_logger import PredictionLogger  # noqa: E402


def _load_env_file(env_file: Path) -> None:
    """Load KEY=VALUE pairs without overriding exported values."""
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip())


def _rank_summary(rows: list[dict[str, Any]], limit: int) -> str:
    """Render compact position:driver rows."""
    return " ".join(f"{row.get('position')}:{row.get('driver')}" for row in rows[: int(limit)])


def _driver_position(rows: list[dict[str, Any]], driver: str) -> Any:
    """Return one driver's position in rows."""
    return next((row.get("position") for row in rows if row.get("driver") == driver), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--session-kind", choices=["qualifying", "race"], default="qualifying")
    parser.add_argument("--race", action="append", required=True)
    parser.add_argument("--driver", action="append", default=[])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)

    predictions = PredictionLogger().get_all_predictions(args.year)
    selected = _sort_selected_predictions(
        _select_latest_predictions(predictions, session_kind=args.session_kind),
        year=args.year,
    )
    wanted = {race.strip().casefold() for race in args.race}
    actual_history: list[list[dict[str, Any]]] = []

    for prediction in selected:
        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        predicted_rows, actual_rows = _resolve_session_pair(
            prediction,
            session_kind=args.session_kind,
        )
        if race_name.casefold() in wanted:
            candidates = {
                "raw": predicted_rows,
                "previous_race_naive": actual_history[-1] if actual_history else predicted_rows,
                "fixed_blend_model_0.4": _blend_with_previous(
                    predicted_rows,
                    actual_history[-1],
                    model_weight=0.4,
                )
                if actual_history
                else predicted_rows,
                "rolling_actual_2": _rolling_actual_rank(
                    predicted_rows,
                    actual_history,
                    window=2,
                )
                if actual_history
                else predicted_rows,
                "actual": actual_rows,
            }
            print()
            print(
                f"{race_name} | {args.session_kind} | checkpoint={metadata.get('session_name')} "
                f"| target={metadata.get(f'top_level_{args.session_kind}_target')} "
                f"| predicted_at={metadata.get('predicted_at')}"
            )
            for name, rows in candidates.items():
                mae = (
                    compute_prediction_accuracy(rows, actual_rows)["mae"]
                    if name != "actual"
                    else 0.0
                )
                print(f"{name:22s} MAE={mae:.3f} {_rank_summary(rows, args.limit)}")
            for driver in args.driver:
                positions = {
                    name: _driver_position(rows, driver) for name, rows in candidates.items()
                }
                print(f"{driver}: {positions}")
        actual_history.append(actual_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
