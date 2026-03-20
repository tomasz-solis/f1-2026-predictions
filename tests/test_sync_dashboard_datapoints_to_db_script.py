"""Tests for the targeted dashboard artifact sync script."""

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "sync_dashboard_datapoints_to_db.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sync_dashboard_datapoints_to_db_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeStore:
    """Capture save calls without touching disk or Supabase."""

    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def save_artifact(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "row-1"}


def test_sync_mismatched_artifacts_uses_singleton_version(monkeypatch, tmp_path):
    """Dashboard sync should overwrite the singleton row instead of creating v2 duplicates."""
    module = _load_script_module()
    store = _FakeStore()
    local_path = tmp_path / "prediction.json"
    local_path.write_text('{"metadata": {"year": 2026}}')
    comparison = module.ArtifactComparison(
        spec=module.ArtifactSpec(
            artifact_type="prediction",
            artifact_key="2026::Chinese Grand Prix::FP1",
            local_path=local_path,
        ),
        local_exists=True,
        remote_exists=True,
        local_checksum="abc",
        remote_checksum="def",
        match=False,
    )
    monkeypatch.setattr(module, "compare_artifacts", lambda *, store, specs: [])

    module.sync_mismatched_artifacts(store=store, comparisons=[comparison])

    assert store.calls == [
        {
            "artifact_type": "prediction",
            "artifact_key": "2026::Chinese Grand Prix::FP1",
            "data": {"metadata": {"year": 2026}},
            "version": 1,
        }
    ]
