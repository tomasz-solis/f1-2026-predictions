from __future__ import annotations

import pytest

import src.utils.fastf1_resilience as resilience


@pytest.fixture(autouse=True)
def _reset_circuit_state():
    resilience._circuit_state.clear()


def test_call_with_resilience_succeeds_without_retry():
    value = resilience.call_with_resilience("unit_op", lambda: 42)
    assert value == 42


def test_call_with_resilience_retries_and_recovers(patcher):
    calls = {"count": 0}

    def _flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary FastF1 error")
        return "ok"

    patcher.setattr(resilience.time, "sleep", lambda _seconds: None)
    value = resilience.call_with_resilience(
        "retry_op",
        _flaky,
        policy=resilience.FastF1ResiliencePolicy(max_attempts=3, timeout_budget_seconds=5.0),
    )

    assert value == "ok"
    assert calls["count"] == 2


def test_call_with_resilience_opens_circuit_after_threshold(patcher):
    patcher.setattr(resilience.time, "sleep", lambda _seconds: None)
    policy = resilience.FastF1ResiliencePolicy(
        max_attempts=1,
        timeout_budget_seconds=5.0,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_cooldown_seconds=60.0,
    )

    with pytest.raises(resilience.FastF1RetryExhaustedError):
        resilience.call_with_resilience(
            "circuit_op", lambda: (_ for _ in ()).throw(RuntimeError("boom")), policy=policy
        )

    with pytest.raises(resilience.FastF1CircuitOpenError):
        resilience.call_with_resilience("circuit_op", lambda: "should not run", policy=policy)
