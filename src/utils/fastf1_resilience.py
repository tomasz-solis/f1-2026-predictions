"""Retry/circuit-breaker helpers for FastF1 network-bound operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from src.utils.operational_observability import record_alert, record_counter

logger = logging.getLogger(__name__)


class FastF1CircuitOpenError(RuntimeError):
    """Raised when circuit breaker is open for an operation."""


class FastF1RetryExhaustedError(RuntimeError):
    """Raised when retries/budget are exhausted for an operation."""


@dataclass(frozen=True)
class FastF1ResiliencePolicy:
    """Policy controls for resilient FastF1 calls."""

    max_attempts: int = 3
    timeout_budget_seconds: float = 10.0
    initial_backoff_seconds: float = 0.35
    max_backoff_seconds: float = 3.0
    backoff_multiplier: float = 2.0
    circuit_breaker_failure_threshold: int = 4
    circuit_breaker_cooldown_seconds: float = 45.0


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_until_monotonic: float = 0.0


_circuit_lock = RLock()
_circuit_state: dict[str, _CircuitState] = {}


def _state_for(operation_name: str) -> _CircuitState:
    with _circuit_lock:
        return _circuit_state.setdefault(str(operation_name), _CircuitState())


def call_with_resilience(
    operation_name: str,
    fn: Callable[[], Any],
    *,
    labels: dict[str, Any] | None = None,
    policy: FastF1ResiliencePolicy | None = None,
) -> Any:
    """
    Execute FastF1 operation with retry/backoff timeout budget and circuit breaker.

    Timeout budget is a total elapsed-time budget across all attempts.
    """
    cfg = policy or FastF1ResiliencePolicy()
    op_name = str(operation_name)
    start = time.monotonic()
    backoff = max(0.0, float(cfg.initial_backoff_seconds))
    attempt = 0

    while True:
        attempt += 1
        now = time.monotonic()
        elapsed = now - start
        budget_remaining = float(cfg.timeout_budget_seconds) - elapsed
        if budget_remaining <= 0:
            record_counter(
                "fastf1_timeout_budget_exhausted_total",
                labels={**(labels or {}), "operation": op_name},
            )
            raise FastF1RetryExhaustedError(
                f"FastF1 timeout budget exhausted for operation={op_name}"
            )

        state = _state_for(op_name)
        if state.opened_until_monotonic > now:
            record_counter(
                "fastf1_circuit_open_total",
                labels={**(labels or {}), "operation": op_name},
            )
            raise FastF1CircuitOpenError(
                f"FastF1 circuit is open for operation={op_name} "
                f"(retry after {state.opened_until_monotonic - now:.1f}s)"
            )

        try:
            result = fn()
            if state.consecutive_failures > 0:
                with _circuit_lock:
                    state.consecutive_failures = 0
                    state.opened_until_monotonic = 0.0
            if attempt > 1:
                record_counter(
                    "fastf1_retry_recovered_total",
                    labels={**(labels or {}), "operation": op_name, "attempts": attempt},
                )
            return result
        except Exception as exc:
            record_counter(
                "fastf1_call_failure_total",
                labels={**(labels or {}), "operation": op_name, "attempt": attempt},
            )

            with _circuit_lock:
                state.consecutive_failures += 1
                if state.consecutive_failures >= int(cfg.circuit_breaker_failure_threshold):
                    state.opened_until_monotonic = now + float(cfg.circuit_breaker_cooldown_seconds)
                    record_counter(
                        "fastf1_circuit_trip_total",
                        labels={**(labels or {}), "operation": op_name},
                    )
                    record_alert(
                        "fastf1_circuit_trip",
                        (
                            f"FastF1 circuit opened for operation={op_name} after "
                            f"{state.consecutive_failures} consecutive failures."
                        ),
                        labels={**(labels or {}), "operation": op_name},
                    )

            if attempt >= int(cfg.max_attempts):
                raise FastF1RetryExhaustedError(
                    f"FastF1 retries exhausted for operation={op_name}; attempts={attempt}"
                ) from exc

            budget_remaining = float(cfg.timeout_budget_seconds) - (time.monotonic() - start)
            if budget_remaining <= 0:
                raise FastF1RetryExhaustedError(
                    f"FastF1 timeout budget exhausted for operation={op_name}"
                ) from exc

            sleep_seconds = min(
                max(0.0, backoff),
                max(0.0, budget_remaining / 2.0),
                float(cfg.max_backoff_seconds),
            )
            if sleep_seconds <= 0:
                raise FastF1RetryExhaustedError(
                    f"FastF1 budget did not allow retry for operation={op_name}"
                ) from exc

            record_counter(
                "fastf1_retry_attempt_total",
                labels={**(labels or {}), "operation": op_name, "attempt": attempt},
            )
            logger.warning(
                "FastF1 call failed: operation=%s attempt=%s/%s error=%s; retrying in %.2fs",
                op_name,
                attempt,
                cfg.max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            backoff = min(float(cfg.max_backoff_seconds), backoff * float(cfg.backoff_multiplier))
