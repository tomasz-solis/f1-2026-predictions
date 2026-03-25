"""Small metric tracker kept around for older scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class PerformanceTracker:
    """Store simple metric histories for legacy call sites."""

    _metrics: dict[str, list[float]] = field(default_factory=dict)

    def record(self, metric: str, value: float) -> None:
        """Append a numeric value for a named metric."""
        self._metrics.setdefault(metric, []).append(float(value))

    def get_average(self, metric: str) -> float | None:
        """Return mean value for metric, or None when empty."""
        values = self._metrics.get(metric, [])
        if not values:
            return None
        return float(mean(values))

    # Compatibility alias used by some archived scripts.
    add_result = record

    def to_dict(self) -> dict[str, dict[str, float | list[float] | None]]:
        """Export metric history with averages."""
        return {
            metric: {
                "values": values,
                "avg": (float(mean(values)) if values else None),
            }
            for metric, values in self._metrics.items()
        }
