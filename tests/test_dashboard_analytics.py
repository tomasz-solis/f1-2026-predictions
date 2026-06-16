from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.dashboard import analytics


class _FakeQuery:
    def __init__(self, sink: list[dict], *, fail: bool = False):
        self._sink = sink
        self._fail = fail

    def insert(self, record: dict):
        self._sink.append(record)
        return self

    def execute(self):
        if self._fail:
            raise RuntimeError("db down")
        return SimpleNamespace(data=[{"id": 1}])


class _FakeClient:
    def __init__(self, sink: list[dict], *, fail: bool = False):
        self._sink = sink
        self._fail = fail

    def table(self, table_name: str):
        assert table_name == "app_events"
        return _FakeQuery(self._sink, fail=self._fail)


def _patch_streamlit(monkeypatch, *, user_agent: str | None, query_params: dict | None = None):
    fake_st = SimpleNamespace(
        session_state={},
        query_params=query_params or {},
        context=SimpleNamespace(headers={"User-Agent": user_agent} if user_agent else {}),
    )
    monkeypatch.setattr(analytics, "st", fake_st)
    monkeypatch.setattr(analytics, "is_db_enabled", lambda: True)
    monkeypatch.setattr(analytics, "format_model_version_label", lambda: "test-version")
    return fake_st


def test_track_event_coarsens_user_agent_and_bounds_payload(monkeypatch):
    records: list[dict] = []
    _patch_streamlit(
        monkeypatch,
        user_agent="Mozilla/5.0 Instagram 123.0 device-specific raw string",
        query_params={"utm_source": "x" * 200},
    )
    monkeypatch.setattr(analytics, "get_supabase_client", lambda: _FakeClient(records))

    analytics.track_event("predict_clicked", page="Predict", detail="y" * 900)

    assert len(records) == 1
    record = records[0]
    assert record["user_agent"] == "instagram_in_app"
    assert "Mozilla" not in record["user_agent"]
    assert record["payload"]["utm_source"] == "x" * 128
    assert record["payload"]["detail"] == "y" * 512


def test_track_event_failure_is_swallowed(monkeypatch):
    records: list[dict] = []
    _patch_streamlit(monkeypatch, user_agent="Mozilla/5.0")
    monkeypatch.setattr(
        analytics,
        "get_supabase_client",
        lambda: _FakeClient(records, fail=True),
    )

    analytics.track_event("page_view", page="Home")

    assert len(records) == 1
    assert records[0]["user_agent"] == "browser"


def test_app_events_hardening_migration_forces_rls_and_revokes() -> None:
    migration = Path("migrations/006_harden_app_events.sql").read_text()

    assert "FORCE ROW LEVEL SECURITY" in migration
    assert "REVOKE ALL ON TABLE public.app_events FROM PUBLIC" in migration
    assert "REVOKE ALL ON TABLE public.app_events FROM anon" in migration
    assert "REVOKE ALL ON TABLE public.app_events FROM authenticated" in migration
    assert "GRANT ALL ON TABLE public.app_events TO service_role" in migration
    assert "service_role_all_app_events" in migration
