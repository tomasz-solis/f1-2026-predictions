"""Focused test for the single-concurrent-runner lockfile guard."""

from __future__ import annotations

import os

import pytest
import scripts.run_challenger_research_walk_forward as runner


def test_lock_refuses_a_second_runner_with_the_same_live_pid(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_pid_is_alive", lambda pid: pid == os.getpid())

    lock_path = tmp_path / ".lock-mytag"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(SystemExit, match="already holds"):
        with runner._single_runner_lock("mytag"):
            pass


def test_lock_is_acquired_and_released_when_no_live_holder_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_pid_is_alive", lambda pid: False)

    with runner._single_runner_lock("mytag"):
        assert (tmp_path / ".lock-mytag").is_file()
    # Released on clean exit -- a later run must not be blocked by a stale lock.
    assert not (tmp_path / ".lock-mytag").exists()
