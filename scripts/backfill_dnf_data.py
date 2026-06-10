"""Backfill DNF data into stored prediction artifacts so the historical report is complete.

For every prediction artifact under ``data/predictions/<year>/`` this script:

  1. Re-fetches the actual race/sprint classification (which now carries a per-driver
     ``dnf`` flag) and writes it back into the stored ``actuals`` blocks, so
     finisher-only MAE and DNF calibration (Brier) can be scored historically.
  2. Fills any *missing* predicted ``dnf_probability`` on race-like predicted rows by
     re-running the race forecast. Existing recorded probabilities and all finishing
     positions are left untouched.

Re-fetching actuals is exact (the classification is ground truth). Re-derived
probabilities use current model state and only ever fill gaps left by the old
save bug; rows that already carry a probability are preserved.

Usage:
  python scripts/backfill_dnf_data.py --year 2026
  python scripts/backfill_dnf_data.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("USE_DB_STORAGE", "file_only")

import fastf1

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.utils.accuracy_targets import TARGET_SESSION_BY_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill_dnf")

# Race-like targets get a DNF flag; qualifying targets do not have a DNF concept.
_RACE_LIKE_TARGETS = {"grand_prix_race": "R", "sprint_race": "Sprint"}
_FASTF1_CACHE_DIRS = ("data/raw/.fastf1_cache", "data/raw/.fastf1_cache_testing")


def _enable_cache() -> None:
    """Enable the first available local FastF1 cache directory."""
    for cache_dir in _FASTF1_CACHE_DIRS:
        path = Path(cache_dir)
        if path.exists():
            fastf1.Cache.enable_cache(str(path))
            return
    Path(_FASTF1_CACHE_DIRS[0]).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(_FASTF1_CACHE_DIRS[0])


def _dnf_by_driver(fetched_rows: list[dict[str, Any]] | None) -> dict[str, bool]:
    """Map driver code -> DNF flag from a freshly fetched classification."""
    if not fetched_rows:
        return {}
    return {str(row["driver"]): bool(row.get("dnf", False)) for row in fetched_rows}


def _merge_dnf_into_block(
    block_rows: list[dict[str, Any]] | None,
    dnf_by_driver: dict[str, bool],
) -> int:
    """Add a per-driver ``dnf`` flag to existing actual rows in place.

    Positions, drivers, and teams are preserved exactly; only the ``dnf`` flag is
    attached by matching on driver. This never reorders or rewrites the recorded
    classification, so a re-fetch that happens to differ in ordering cannot change
    the ground truth the predictions were scored against.
    """
    if not isinstance(block_rows, list) or not dnf_by_driver:
        return 0
    labelled = 0
    for row in block_rows:
        if not isinstance(row, dict):
            continue
        driver = str(row.get("driver", ""))
        if driver in dnf_by_driver:
            row["dnf"] = dnf_by_driver[driver]
            if row["dnf"]:
                labelled += 1
    return labelled


def _match_session_for_block(
    block_rows: list[dict[str, Any]],
    fetched: dict[str, list[dict[str, Any]] | None],
) -> str | None:
    """Pick the fetched session whose ordering best matches an existing block.

    Used for the legacy top-level race actuals, which on a sprint weekend hold the
    sprint classification rather than the Grand Prix.
    """
    block_order = [str(row.get("driver", "")) for row in block_rows]
    best_session: str | None = None
    best_score = -1
    for session_name, rows in fetched.items():
        if not rows:
            continue
        order = [str(row["driver"]) for row in rows]
        score = sum(1 for left, right in zip(block_order, order, strict=False) if left == right)
        if score > best_score:
            best_score = score
            best_session = session_name
    return best_session


def _backfill_actuals(
    prediction: dict[str, Any],
    *,
    fetched: dict[str, list[dict[str, Any]] | None],
) -> int:
    """Attach DNF flags to the prediction's existing actuals blocks. Returns DNFs labelled."""
    actuals = prediction.get("actuals")
    if not isinstance(actuals, dict):
        return 0

    labelled = 0
    target_actuals = actuals.get("targets")
    if isinstance(target_actuals, dict):
        for target_key, session_name in _RACE_LIKE_TARGETS.items():
            rows = target_actuals.get(target_key)
            labelled += _merge_dnf_into_block(rows, _dnf_by_driver(fetched.get(session_name)))

    # Legacy top-level race actuals: pick the matching session (sprint vs Grand Prix).
    legacy_rows = actuals.get("race")
    if isinstance(legacy_rows, list) and legacy_rows:
        matched_session = _match_session_for_block(legacy_rows, fetched)
        if matched_session is not None:
            _merge_dnf_into_block(legacy_rows, _dnf_by_driver(fetched.get(matched_session)))

    return labelled


def _build_grid(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a starting grid for re-deriving DNF probabilities."""
    quali = (prediction.get("qualifying") or {}).get("predicted_grid") or []
    rows = quali or (prediction.get("race") or {}).get("predicted_results") or []
    grid = [
        {
            "driver": str(row["driver"]),
            "team": str(row.get("team", "")),
            "position": int(row.get("position", index + 1)),
        }
        for index, row in enumerate(rows)
        if row.get("driver")
    ]
    return grid


def _fill_missing_dnf_probabilities(
    prediction: dict[str, Any],
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    context: Any,
) -> int:
    """Fill missing predicted dnf_probability on race-like rows. Returns rows filled."""
    grid = _build_grid(prediction)
    if len(grid) < 2:
        return 0

    def _needs_fill(rows: Any) -> bool:
        return isinstance(rows, list) and any(
            isinstance(row, dict) and row.get("dnf_probability") is None for row in rows
        )

    # Only the target predicted_order rows are scored by the report. Each is filled with
    # the probability for its own race type. The legacy race.predicted_results block is left
    # alone: it is not read by the evaluation and its race mapping is ambiguous on sprint
    # weekends. Rows that already carry a probability are preserved.
    targets = prediction.get("targets") or {}
    target_rows = {
        ("grand_prix_race", False): (targets.get("grand_prix_race") or {}).get("predicted_order"),
        ("sprint_race", True): (targets.get("sprint_race") or {}).get("predicted_order"),
    }
    if not any(_needs_fill(rows) for rows in target_rows.values()):
        return 0

    filled = 0
    for (_target_key, is_sprint), rows in target_rows.items():
        if not isinstance(rows, list) or not _needs_fill(rows):
            continue
        result = predictor.predict_race(
            qualifying_grid=grid,
            weather=weather,
            race_name=race_name,
            year=year,
            is_sprint=is_sprint,
            n_simulations=200,
            prediction_context=context,
        )
        probabilities = {
            str(row["driver"]): float(row.get("dnf_probability", 0.0))
            for row in result["finish_order"]
        }
        for row in rows:
            if isinstance(row, dict) and row.get("dnf_probability") is None:
                value = probabilities.get(str(row.get("driver")))
                if value is not None:
                    row["dnf_probability"] = round(float(value), 3)
                    filled += 1

    return filled


def backfill_file(
    path: Path,
    *,
    predictor: Any,
    dry_run: bool,
) -> dict[str, int]:
    """Backfill one prediction artifact. Returns counts of what changed."""
    from src.utils.prediction_context import PredictionContext

    with open(path, encoding="utf-8") as handle:
        prediction = json.load(handle)

    metadata = prediction.get("metadata", {})
    year = int(metadata.get("year", 0) or 0)
    race_name = str(metadata.get("race_name", "")).strip()
    weather = str(metadata.get("weather", "dry")).strip().lower() or "dry"
    if year <= 0 or not race_name:
        logger.warning("Skipping %s: missing year/race_name", path.name)
        return {"actual_dnfs": 0, "probs_filled": 0}

    # Fetch race/sprint classifications once (cached); reuse for actuals.
    fetched: dict[str, list[dict[str, Any]] | None] = {}
    for session_name in ("R", "Sprint"):
        try:
            result = fetch_actual_session_results(year, race_name, session_name)
            fetched[session_name] = cast("list[dict[str, Any]] | None", result)
        except Exception as exc:  # noqa: BLE001 - best effort per session
            logger.warning("Could not fetch %s %s: %s", race_name, session_name, exc)
            fetched[session_name] = None

    actual_dnfs = _backfill_actuals(prediction, fetched=fetched)

    # A far-future reference time keeps all sessions "available"; DNF probability does
    # not depend on practice data, so this only affects session-availability gating.
    context = PredictionContext(
        mode="historical",
        as_of_datetime=datetime(year, 12, 31, tzinfo=UTC),
        season_year=year,
    )
    probs_filled = _fill_missing_dnf_probabilities(
        prediction,
        predictor=predictor,
        year=year,
        race_name=race_name,
        weather=weather,
        context=context,
    )

    if (actual_dnfs or probs_filled) and not dry_run:
        prediction.setdefault("metadata", {})["dnf_backfilled_at"] = datetime.now(UTC).isoformat()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(prediction, handle, indent=2)

    return {"actual_dnfs": actual_dnfs, "probs_filled": probs_filled}


def main() -> None:
    """CLI entrypoint for the DNF backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=datetime.now(UTC).year)
    parser.add_argument(
        "--predictions-dir",
        default="data/predictions",
        help="Root directory of stored prediction artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files.",
    )
    args = parser.parse_args()

    _ = TARGET_SESSION_BY_KEY  # documents the target->session mapping source
    _enable_cache()

    year_dir = Path(args.predictions_dir) / str(args.year)
    if not year_dir.exists():
        logger.error("No prediction directory for %s at %s", args.year, year_dir)
        return

    files = sorted(year_dir.rglob("*.json"))
    if not files:
        logger.warning("No prediction artifacts found under %s", year_dir)
        return

    from src.predictors import Baseline2026Predictor

    predictor = Baseline2026Predictor(season_year=args.year)

    total_actual_dnfs = 0
    total_probs_filled = 0
    changed_files = 0
    for path in files:
        counts = backfill_file(path, predictor=predictor, dry_run=args.dry_run)
        if counts["actual_dnfs"] or counts["probs_filled"]:
            changed_files += 1
            logger.info(
                "%s: actual DNFs labelled=%s, predicted probabilities filled=%s%s",
                path.name,
                counts["actual_dnfs"],
                counts["probs_filled"],
                " (dry-run)" if args.dry_run else "",
            )
        total_actual_dnfs += counts["actual_dnfs"]
        total_probs_filled += counts["probs_filled"]

    logger.info(
        "Backfill complete: %s file(s) changed, %s actual DNFs labelled, "
        "%s predicted probabilities filled%s.",
        changed_files,
        total_actual_dnfs,
        total_probs_filled,
        " (dry-run, no files written)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
