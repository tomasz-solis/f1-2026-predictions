"""Persistent calibration state derived from completed predictions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.driver_name_mapper import DriverNameMapper

logger = logging.getLogger(__name__)


class SystematicLearningSystem:
    """Persist and serve calibration learned from prediction outcomes."""

    def __init__(self, state_file: str | Path = "data/learning_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "season": 2026,
            "races_completed": 0,
            "history": [],
            "method_performance": {},
            "adaptive_calibration": {
                "version": 1,
                "last_updated": None,
                "driver_position_error": {"qualifying": {}, "race": {}},
                "teammate_gap_error": {"qualifying": {}, "race": {}},
                "interval_residual_history": {"qualifying": [], "race": []},
                "event_history": [],
            },
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return self._default_state()

        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load learning state from %s (%s); resetting state.", self.state_file, exc
            )
            return self._default_state()

        if not isinstance(data, dict):
            logger.warning("Learning state at %s is not a JSON object; resetting.", self.state_file)
            return self._default_state()

        merged = self._default_state()
        merged.update(data)

        adaptive = merged.get("adaptive_calibration")
        if not isinstance(adaptive, dict):
            adaptive = self._default_state()["adaptive_calibration"]
            merged["adaptive_calibration"] = adaptive

        adaptive.setdefault("version", 1)
        adaptive.setdefault("last_updated", None)
        adaptive.setdefault("driver_position_error", {"qualifying": {}, "race": {}})
        adaptive.setdefault("teammate_gap_error", {"qualifying": {}, "race": {}})
        adaptive.setdefault("event_history", [])

        driver_error = adaptive["driver_position_error"]
        if not isinstance(driver_error, dict):
            driver_error = {"qualifying": {}, "race": {}}
            adaptive["driver_position_error"] = driver_error
        driver_error.setdefault("qualifying", {})
        driver_error.setdefault("race", {})

        teammate_error = adaptive["teammate_gap_error"]
        if not isinstance(teammate_error, dict):
            teammate_error = {"qualifying": {}, "race": {}}
            adaptive["teammate_gap_error"] = teammate_error
        teammate_error.setdefault("qualifying", {})
        teammate_error.setdefault("race", {})

        interval_history = adaptive.setdefault(
            "interval_residual_history",
            {"qualifying": [], "race": []},
        )
        if not isinstance(interval_history, dict):
            interval_history = {"qualifying": [], "race": []}
            adaptive["interval_residual_history"] = interval_history
        for session_name in ("qualifying", "race"):
            if not isinstance(interval_history.get(session_name), list):
                interval_history[session_name] = []

        if not isinstance(adaptive.get("event_history"), list):
            adaptive["event_history"] = []

        return merged

    def save_state(self) -> None:
        """Write the current learning state to disk."""
        adaptive = self.state.setdefault("adaptive_calibration", {})
        adaptive["last_updated"] = datetime.now(UTC).isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def reset_state(self, season: int | None = None) -> None:
        """Reset learning state to a clean default payload for one season."""
        next_state = self._default_state()
        if season is not None:
            next_state["season"] = int(season)
        self.state = next_state
        self.save_state()

    @staticmethod
    def _update_ema_slot(slot: dict[str, Any], value: float, alpha: float) -> None:
        count = int(slot.get("count", 0)) + 1
        previous_ema = float(slot.get("ema_error", 0.0))
        previous_abs_ema = float(slot.get("ema_abs_error", 0.0))

        if count == 1:
            ema_error = value
            ema_abs_error = abs(value)
        else:
            ema_error = previous_ema + (alpha * (value - previous_ema))
            abs_value = abs(value)
            ema_abs_error = previous_abs_ema + (alpha * (abs_value - previous_abs_ema))

        slot["count"] = count
        slot["ema_error"] = float(ema_error)
        slot["ema_abs_error"] = float(ema_abs_error)
        slot["last_error"] = float(value)

    @staticmethod
    def _normalize_driver_code(value: Any) -> str | None:
        """Normalize driver identifier while avoiding warning spam for valid new abbreviations."""
        if not isinstance(value, str):
            return None

        candidate = value.strip()
        if not candidate:
            return None

        if candidate.isalpha() and candidate.isupper() and 2 <= len(candidate) <= 4:
            return candidate

        normalized = DriverNameMapper.normalize_driver_name(candidate)
        if isinstance(normalized, str) and normalized.strip():
            return normalized.strip()
        return None

    @staticmethod
    def _coerce_position(value: Any) -> int | None:
        """Convert stored position-like values into integers when possible."""
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        """Parse an ISO-like timestamp into an aware UTC datetime when possible."""
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None

        normalized = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _has_processed_run_id(self, run_id: str | None) -> bool:
        """Return whether one prediction run id already updated learning state."""
        if not isinstance(run_id, str) or not run_id.strip():
            return False

        adaptive = self.state.get("adaptive_calibration", {})
        if not isinstance(adaptive, dict):
            return False

        event_history = adaptive.get("event_history", [])
        if not isinstance(event_history, list):
            return False

        normalized_run_id = run_id.strip()
        for entry in event_history:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("run_id", "")).strip() == normalized_run_id:
                return True
        return False

    def _learning_skip_reason(self, prediction_data: dict[str, Any]) -> str | None:
        """Return a skip reason when a prediction should not train adaptive learning."""
        metadata = prediction_data.get("metadata", {})
        if not isinstance(metadata, dict):
            return None

        run_id = str(metadata.get("run_id", "")).strip()
        if run_id and self._has_processed_run_id(run_id):
            return "duplicate_run_id"

        source = str(metadata.get("source", "")).strip().lower()
        generated_by = str(metadata.get("generated_by", "")).strip().lower()
        checkpoint_source = str(metadata.get("checkpoint_source", "")).strip().lower()
        if source == "checkpoint_reconstruction":
            return "retrospective_prediction"
        if generated_by == "checkpoint_reconstruction":
            return "retrospective_prediction"
        if checkpoint_source == "retrospective":
            return "retrospective_prediction"

        predicted_at = self._parse_timestamp(metadata.get("predicted_at"))
        information_cutoff_at = self._parse_timestamp(metadata.get("information_cutoff_at"))
        if (
            predicted_at is not None
            and information_cutoff_at is not None
            and predicted_at > information_cutoff_at
        ):
            return "retrospective_prediction"

        return None

    def _update_session_errors(
        self,
        session: str,
        predicted: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        alpha: float,
    ) -> dict[str, Any]:
        if not predicted or not actual:
            return {
                "session": session,
                "drivers_updated": 0,
                "pairs_updated": 0,
                "mae": None,
            }

        pred_positions: dict[str, int] = {}
        act_positions: dict[str, int] = {}
        driver_to_team: dict[str, str] = {}

        for row in predicted:
            driver_code = self._normalize_driver_code(row.get("driver"))
            position = row.get("position")
            if driver_code is None or position is None:
                continue
            pred_positions[driver_code] = int(position)
            team_name = row.get("team")
            if isinstance(team_name, str) and team_name.strip():
                driver_to_team[driver_code] = team_name

        for row in actual:
            driver_code = self._normalize_driver_code(row.get("driver"))
            position = row.get("position")
            if driver_code is None or position is None:
                continue
            act_positions[driver_code] = int(position)

        driver_to_team = {driver: team_name for driver, team_name in driver_to_team.items()}

        common_drivers = sorted(set(pred_positions).intersection(act_positions))
        if not common_drivers:
            return {
                "session": session,
                "drivers_updated": 0,
                "pairs_updated": 0,
                "mae": None,
            }

        calibration = self.state["adaptive_calibration"]
        driver_bucket = calibration["driver_position_error"].setdefault(session, {})
        teammate_bucket = calibration["teammate_gap_error"].setdefault(session, {})

        for driver in common_drivers:
            slot = driver_bucket.setdefault(driver, {})
            position_error = float(pred_positions[driver] - act_positions[driver])
            self._update_ema_slot(slot, value=position_error, alpha=alpha)

        team_to_drivers: dict[str, list[str]] = {}
        for driver in common_drivers:
            team_name = driver_to_team.get(driver)
            if not team_name:
                continue
            team_to_drivers.setdefault(team_name, []).append(driver)

        pair_updates = 0
        for team_name, drivers in team_to_drivers.items():
            if len(drivers) < 2:
                continue
            for left, right in combinations(sorted(drivers), 2):
                pair_key = f"{team_name}::{left}::{right}"
                slot = teammate_bucket.setdefault(pair_key, {})
                predicted_gap = float(pred_positions[left] - pred_positions[right])
                actual_gap = float(act_positions[left] - act_positions[right])
                gap_error = predicted_gap - actual_gap
                self._update_ema_slot(slot, value=gap_error, alpha=alpha)
                pair_updates += 1

        mae = float(
            np.mean(
                [abs(pred_positions[driver] - act_positions[driver]) for driver in common_drivers]
            )
        )

        return {
            "session": session,
            "drivers_updated": len(common_drivers),
            "pairs_updated": pair_updates,
            "mae": round(mae, 4),
        }

    def _update_interval_residual_history(
        self,
        *,
        session: str,
        predicted: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        history_limit: int,
    ) -> int:
        """Persist interval residuals so later predictions can calibrate their bands.

        Each stored row represents one driver-session outcome with:
        - residual: absolute miss from the predicted center
        - half_width: half-width of the published p5-p95 interval
        - covered: whether the actual finish landed inside that interval
        """
        if not predicted or not actual:
            return 0

        predicted_intervals: dict[str, dict[str, int]] = {}
        for row in predicted:
            driver_code = self._normalize_driver_code(row.get("driver"))
            center = self._coerce_position(row.get("median_position", row.get("position")))
            p5 = self._coerce_position(row.get("p5"))
            p95 = self._coerce_position(row.get("p95"))
            if driver_code is None or center is None or p5 is None or p95 is None:
                continue
            lower = min(p5, p95)
            upper = max(p5, p95)
            predicted_intervals[driver_code] = {
                "center": center,
                "lower": lower,
                "upper": upper,
            }

        if not predicted_intervals:
            return 0

        actual_positions: dict[str, int] = {}
        for row in actual:
            driver_code = self._normalize_driver_code(row.get("driver"))
            position = self._coerce_position(row.get("position"))
            if driver_code is None or position is None:
                continue
            actual_positions[driver_code] = position

        common_drivers = sorted(set(predicted_intervals).intersection(actual_positions))
        if not common_drivers:
            return 0

        calibration = self.state["adaptive_calibration"]
        interval_history = calibration["interval_residual_history"].setdefault(session, [])

        for driver_code in common_drivers:
            interval_row = predicted_intervals[driver_code]
            actual_position = actual_positions[driver_code]
            center = interval_row["center"]
            lower = interval_row["lower"]
            upper = interval_row["upper"]
            interval_history.append(
                {
                    "residual": float(abs(center - actual_position)),
                    "half_width": float(max(center - lower, upper - center)),
                    "covered": bool(lower <= actual_position <= upper),
                }
            )

        if len(interval_history) > history_limit:
            calibration["interval_residual_history"][session] = interval_history[-history_limit:]

        return len(common_drivers)

    def update_from_prediction_record(
        self,
        prediction_data: dict[str, Any],
        alpha: float = 0.35,
        history_limit: int = 250,
    ) -> dict[str, Any]:
        """Update calibration from a saved prediction record with actual results."""
        skip_reason = self._learning_skip_reason(prediction_data)
        if skip_reason is not None:
            return {
                "sessions_updated": 0,
                "driver_updates": 0,
                "pair_updates": 0,
                "details": [],
                "skipped": True,
                "skip_reason": skip_reason,
            }

        actuals = prediction_data.get("actuals") or {}
        predicted_quali = (prediction_data.get("qualifying") or {}).get("predicted_grid") or []
        predicted_race = (prediction_data.get("race") or {}).get("predicted_results") or []
        actual_quali = actuals.get("qualifying") or []
        actual_race = actuals.get("race") or []

        updates: list[dict[str, Any]] = []
        interval_history_limit = max(500, history_limit * 20)
        if actual_quali:
            qualifying_update = self._update_session_errors(
                session="qualifying",
                predicted=predicted_quali,
                actual=actual_quali,
                alpha=alpha,
            )
            qualifying_update["interval_samples"] = self._update_interval_residual_history(
                session="qualifying",
                predicted=predicted_quali,
                actual=actual_quali,
                history_limit=interval_history_limit,
            )
            updates.append(qualifying_update)
        if actual_race:
            race_update = self._update_session_errors(
                session="race",
                predicted=predicted_race,
                actual=actual_race,
                alpha=alpha,
            )
            race_update["interval_samples"] = self._update_interval_residual_history(
                session="race",
                predicted=predicted_race,
                actual=actual_race,
                history_limit=interval_history_limit,
            )
            updates.append(race_update)

        metadata = prediction_data.get("metadata", {})
        event_entry = {
            "race_name": metadata.get("race_name"),
            "session_name": metadata.get("session_name"),
            "run_id": metadata.get("run_id"),
            "updated_at": datetime.now(UTC).isoformat(),
            "updates": updates,
        }

        calibration = self.state["adaptive_calibration"]
        history = calibration.setdefault("event_history", [])
        history.append(event_entry)
        if len(history) > history_limit:
            calibration["event_history"] = history[-history_limit:]

        self.save_state()

        total_driver_updates = sum(item.get("drivers_updated", 0) for item in updates)
        total_pair_updates = sum(item.get("pairs_updated", 0) for item in updates)
        total_interval_samples = sum(item.get("interval_samples", 0) for item in updates)

        return {
            "sessions_updated": len(updates),
            "driver_updates": total_driver_updates,
            "pair_updates": total_pair_updates,
            "interval_samples": total_interval_samples,
            "details": updates,
            "skipped": False,
            "skip_reason": None,
        }

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return float(np.clip(value, lower, upper))

    def get_driver_position_adjustment(
        self,
        driver: str,
        session: str,
        *,
        min_samples: int = 1,
        error_scale: float = 0.18,
        max_adjustment: float = 2.0,
    ) -> float:
        """Return learned position adjustment for a driver.

        Positive value means the driver should be nudged forward (towards P1).
        """
        driver_code = self._normalize_driver_code(driver)
        if driver_code is None:
            return 0.0
        calibration = self.state.get("adaptive_calibration", {})
        session_bucket = (calibration.get("driver_position_error") or {}).get(session) or {}
        stats = session_bucket.get(driver_code)
        if not isinstance(stats, dict):
            return 0.0

        if int(stats.get("count", 0)) < min_samples:
            return 0.0

        ema_error = float(stats.get("ema_error", 0.0))
        adjustment = ema_error * error_scale
        return self._clip(adjustment, -max_adjustment, max_adjustment)

    def get_teammate_position_adjustment(
        self,
        team: str,
        driver: str,
        teammates: list[str],
        session: str,
        *,
        min_samples: int = 1,
        gap_scale: float = 0.10,
        max_adjustment: float = 0.8,
    ) -> float:
        """Return teammate-gap correction for one driver within a team."""
        driver_code = self._normalize_driver_code(driver)
        if driver_code is None:
            return 0.0
        normalized_teammates = [
            normalized
            for code in teammates
            for normalized in [self._normalize_driver_code(code)]
            if normalized is not None
        ]

        calibration = self.state.get("adaptive_calibration", {})
        pair_bucket = (calibration.get("teammate_gap_error") or {}).get(session) or {}

        total = 0.0
        for teammate in sorted(set(normalized_teammates)):
            if teammate == driver_code:
                continue
            left, right = sorted([driver_code, teammate])
            pair_key = f"{team}::{left}::{right}"
            stats = pair_bucket.get(pair_key)
            if not isinstance(stats, dict):
                continue
            if int(stats.get("count", 0)) < min_samples:
                continue

            ema_gap_error = float(stats.get("ema_error", 0.0))
            direction = 1.0 if driver_code == left else -1.0
            total += direction * ema_gap_error * gap_scale

        return self._clip(total, -max_adjustment, max_adjustment)

    def get_combined_position_adjustment(
        self,
        *,
        team: str,
        driver: str,
        teammates: list[str],
        session: str,
        min_samples: int = 1,
        driver_error_scale: float = 0.18,
        teammate_gap_scale: float = 0.10,
        max_adjustment: float = 2.5,
    ) -> float:
        """Return combined driver + teammate learned position adjustment."""
        driver_adj = self.get_driver_position_adjustment(
            driver=driver,
            session=session,
            min_samples=min_samples,
            error_scale=driver_error_scale,
            max_adjustment=max_adjustment,
        )
        teammate_adj = self.get_teammate_position_adjustment(
            team=team,
            driver=driver,
            teammates=teammates,
            session=session,
            min_samples=min_samples,
            gap_scale=teammate_gap_scale,
            max_adjustment=max_adjustment,
        )
        combined = driver_adj + teammate_adj
        return self._clip(combined, -max_adjustment, max_adjustment)

    def get_interval_calibration_summary(
        self,
        session: str,
        *,
        min_samples: int = 20,
        target_coverage: float = 0.90,
        max_adjustment: float = 6.0,
    ) -> dict[str, float]:
        """Summarize learned interval calibration for one session type.

        The published p5-p95 ranges remain model-derived, but this summary exposes
        an empirical residual radius that can be used as a conformal-style floor
        when the historical intervals have been too tight.
        """
        calibration = self.state.get("adaptive_calibration", {})
        interval_history = (calibration.get("interval_residual_history") or {}).get(session) or []
        rows = [row for row in interval_history if isinstance(row, dict)]

        residuals = []
        half_widths = []
        hits = 0.0
        for row in rows:
            try:
                residuals.append(float(row.get("residual", 0.0)))
                half_widths.append(float(row.get("half_width", 0.0)))
                hits += 1.0 if bool(row.get("covered")) else 0.0
            except (TypeError, ValueError):
                continue

        sample_count = len(residuals)
        learned_radius = 0.0
        clipped_target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
        if sample_count >= max(1, min_samples):
            learned_radius = self._clip(
                float(np.quantile(residuals, clipped_target_coverage)),
                0.0,
                max_adjustment,
            )

        empirical_coverage = (hits / sample_count) if sample_count else 0.0
        mean_half_width = float(np.mean(half_widths)) if half_widths else 0.0
        mean_residual = float(np.mean(residuals)) if residuals else 0.0
        return {
            "sample_count": float(sample_count),
            "empirical_coverage": float(empirical_coverage),
            "target_coverage": clipped_target_coverage,
            "mean_half_width": mean_half_width,
            "mean_residual": mean_residual,
            "learned_radius": float(learned_radius),
        }

    def get_interval_radius(
        self,
        session: str,
        *,
        min_samples: int = 20,
        target_coverage: float = 0.90,
        max_adjustment: float = 6.0,
    ) -> float:
        """Return the learned conformal-style residual radius for one session."""
        summary = self.get_interval_calibration_summary(
            session,
            min_samples=min_samples,
            target_coverage=target_coverage,
            max_adjustment=max_adjustment,
        )
        return float(summary["learned_radius"])
