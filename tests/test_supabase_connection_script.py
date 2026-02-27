import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "test_supabase_connection.py"
    spec = importlib.util.spec_from_file_location("test_supabase_connection_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeStore:
    storage_mode = "db_only"

    def save_artifact(self, **_kwargs):
        return {"id": "test-id"}

    def load_artifact(self, **_kwargs):
        return {"test": True, "message": "Hello from test"}

    def list_artifacts(self, **_kwargs):
        return [{"id": "test-id"}]


def test_main_accepts_string_health_message(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module, "check_connection", lambda: "Supabase connection healthy")
    monkeypatch.setattr(module, "ArtifactStore", _FakeStore)

    assert module.main() == 0


def test_main_fails_when_health_check_raises(monkeypatch):
    module = _load_script_module()

    def _raise():
        raise RuntimeError("auth failed")

    monkeypatch.setattr(module, "check_connection", _raise)

    assert module.main() == 1
