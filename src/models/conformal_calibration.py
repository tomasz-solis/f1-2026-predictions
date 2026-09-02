"""Bucketed conformal calibration helpers for ranking intervals.

The current runtime already stores interval residual history, but it learns one
global floor per session. This module adds an explicit artifact-based layer so
we can calibrate by data regime and version those overlays separately from the
online learning state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_SESSION_NAMES = ("qualifying", "race")
_DATA_REGIMES = (
    "model_only",
    "testing_fallback",
    "practice_backed",
    "checkpoint_backed",
)


def resolve_qualifying_data_regime(
    *,
    practice_like_stored_profiles: bool,
    session_name: str | None,
    testing_fallback_used: bool,
) -> str:
    """Resolve the qualifying regime bucket for one prediction."""
    if practice_like_stored_profiles:
        return "checkpoint_backed"
    if session_name:
        return "practice_backed"
    if testing_fallback_used:
        return "testing_fallback"
    return "model_only"


def resolve_race_data_regime(
    *,
    input_confidence: float | None,
    mean_grid_confidence: float | None,
) -> str:
    """Resolve the race regime bucket from available grid-confidence signals.

    Race predictions do not carry the same explicit practice/testing source tags
    as qualifying, so we use the strongest stable proxy available at runtime:
    input confidence when present, otherwise the mean qualifying-grid confidence.
    """
    confidence_candidates = [
        value for value in (input_confidence, mean_grid_confidence) if value is not None
    ]
    if not confidence_candidates:
        return "model_only"

    confidence = float(confidence_candidates[0])
    if confidence > 1.0:
        confidence /= 100.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence >= 0.80:
        return "checkpoint_backed"
    if confidence >= 0.60:
        return "practice_backed"
    if confidence >= 0.40:
        return "testing_fallback"
    return "model_only"


@dataclass(frozen=True)
class ConformalCalibrationArtifact:
    """Serializable conformal summary keyed by session and data regime."""

    generated_at: str
    target_coverage: float
    max_radius: float
    buckets: dict[str, dict[str, dict[str, float | int]]]

    def get_radius(self, *, session: str, regime: str) -> float:
        """Return the learned interval radius for one bucket or zero when missing."""
        session_bucket = self.buckets.get(str(session), {})
        regime_bucket = session_bucket.get(str(regime), {})
        try:
            return float(regime_bucket.get("radius", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this artifact to a JSON-compatible dictionary."""
        return {
            "generated_at": self.generated_at,
            "target_coverage": self.target_coverage,
            "max_radius": self.max_radius,
            "buckets": self.buckets,
        }


def _coerce_float(value: Any) -> float | None:
    """Return one finite float or ``None`` when coercion fails."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_conformal_calibration_artifact(
    *,
    rows: list[dict[str, Any]],
    target_coverage: float = 0.90,
    min_samples: int = 20,
    max_radius: float = 6.0,
    generated_at: str | None = None,
) -> ConformalCalibrationArtifact:
    """Build a conformal artifact from residual rows grouped by regime.

    Expected input rows:
    - ``session``: ``qualifying`` or ``race``
    - ``regime``: one of the supported regime buckets
    - ``residual``: absolute miss from the predicted center
    """
    clipped_target = float(np.clip(target_coverage, 0.0, 1.0))
    buckets: dict[str, dict[str, dict[str, float | int]]] = {
        session_name: {
            regime_name: {
                "sample_count": 0,
                "empirical_coverage": 0.0,
                "mean_residual": 0.0,
                "radius": 0.0,
            }
            for regime_name in _DATA_REGIMES
        }
        for session_name in _SESSION_NAMES
    }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        session_name = str(row.get("session", "")).strip().lower()
        regime_name = str(row.get("regime", "")).strip().lower()
        if session_name not in _SESSION_NAMES or regime_name not in _DATA_REGIMES:
            continue
        grouped.setdefault((session_name, regime_name), []).append(row)

    for (session_name, regime_name), bucket_rows in grouped.items():
        residuals: list[float] = []
        covered_hits = 0.0
        for row in bucket_rows:
            residual = _coerce_float(row.get("residual"))
            if residual is None:
                continue
            residuals.append(residual)
            covered_hits += 1.0 if bool(row.get("covered")) else 0.0

        if not residuals:
            continue

        sample_count = len(residuals)
        radius = 0.0
        if sample_count >= max(1, int(min_samples)):
            radius = float(
                np.clip(
                    np.quantile(residuals, clipped_target),
                    0.0,
                    float(max_radius),
                )
            )

        buckets[session_name][regime_name] = {
            "sample_count": int(sample_count),
            "empirical_coverage": float(covered_hits / sample_count),
            "mean_residual": float(np.mean(residuals)),
            "radius": radius,
        }

    return ConformalCalibrationArtifact(
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        target_coverage=clipped_target,
        max_radius=float(max_radius),
        buckets=buckets,
    )


def load_conformal_calibration_artifact(
    path: str | Path,
) -> ConformalCalibrationArtifact | None:
    """Load a conformal artifact from disk when present."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None

    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    buckets = payload.get("buckets", {})
    if not isinstance(buckets, dict):
        return None

    return ConformalCalibrationArtifact(
        generated_at=str(payload.get("generated_at", "")),
        target_coverage=float(payload.get("target_coverage", 0.90)),
        max_radius=float(payload.get("max_radius", 6.0)),
        buckets=buckets,
    )


def save_conformal_calibration_artifact(
    *,
    artifact: ConformalCalibrationArtifact,
    path: str | Path,
) -> Path:
    """Persist one conformal artifact to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact.to_dict(), indent=2))
    return output_path
