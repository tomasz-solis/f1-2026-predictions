"""The operator panel is token-gated and does no forecasting work of its own."""

from typing import Any

import pytest

from src.dashboard import admin_page, layout

_SECRET = {"TL_ADMIN_TOKEN": "secret"}


class _St:
    """Streamlit stub recording what the panel drew."""

    def __init__(self, params: dict[str, Any] | None = None, pressed: str | None = None):
        self.query_params = params or {}
        self.session_state: dict[str, Any] = {}
        self.pressed = pressed
        self.heroes: list[dict[str, Any]] = []
        self.buttons: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.cleared = 0

    def subheader(self, text: str, *_a: Any, **_k: Any) -> None:
        self.markdowns.append(str(text))

    def markdown(self, text: str, *_a: Any, **_k: Any) -> None:
        self.markdowns.append(str(text))

    def caption(self, text: str, *_a: Any, **_k: Any) -> None:
        self.captions.append(str(text))

    def error(self, text: str, *_a: Any, **_k: Any) -> None:
        self.errors.append(str(text))

    def success(self, text: str, *_a: Any, **_k: Any) -> None:
        self.successes.append(str(text))

    def warning(self, text: str, *_a: Any, **_k: Any) -> None:
        self.warnings.append(str(text))

    def info(self, text: str, *_a: Any, **_k: Any) -> None:
        self.infos.append(str(text))

    def button(self, label: str, *_a: Any, **_k: Any) -> bool:
        self.buttons.append(str(label))
        return str(label) == self.pressed

    class _Cache:
        def __init__(self, owner: "_St"):
            self._owner = owner

        def clear(self) -> None:
            self._owner.cleared += 1

    @property
    def cache_resource(self) -> "_St._Cache":
        return _St._Cache(self)

    @property
    def cache_data(self) -> "_St._Cache":
        return _St._Cache(self)


def _stub_body(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Replace the heavy sections so the test never touches FastF1 or the artifact store."""
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        admin_page, "_render_selectors", lambda _st, _year: (2026, "Italian Grand Prix")
    )
    monkeypatch.setattr(
        admin_page,
        "_read_precompute_status",
        lambda _year: {"artifact_hash": "76d26fcf00", "horizon": None, "error": None},
    )
    monkeypatch.setattr(admin_page, "_render_hero", lambda st_module, **kwargs: None)
    monkeypatch.setattr(
        admin_page,
        "render_grid_penalty_editor",
        lambda **kwargs: seen.append(("penalties", kwargs["year"])),
    )
    monkeypatch.setattr(
        admin_page,
        "render_driver_substitution_editor",
        lambda **kwargs: seen.append(("substitutions", kwargs["year"])),
    )
    return seen


def test_a_visitor_without_the_token_gets_no_panel(monkeypatch: pytest.MonkeyPatch):
    seen = _stub_body(monkeypatch)
    st_module = _St({"admin": "guess"})

    admin_page.render_admin_page(st_module=st_module, env=_SECRET)

    assert seen == []
    assert st_module.buttons == []
    assert st_module.errors


def test_the_operator_gets_both_editors_and_the_ops_buttons(monkeypatch: pytest.MonkeyPatch):
    seen = _stub_body(monkeypatch)
    st_module = _St({"admin": "secret"})

    admin_page.render_admin_page(st_module=st_module, env=_SECRET)

    assert seen == [("penalties", 2026), ("substitutions", 2026)]
    assert st_module.buttons == [
        "Trigger precompute run",
        "Restart web service",
        "Clear dashboard caches",
    ]


def test_missing_render_settings_are_named_instead_of_hidden():
    st_module = _St()

    admin_page._render_ops_controls(st_module, env={})

    assert any("RENDER_API_KEY" in message for message in st_module.infos)


def test_pressing_trigger_reports_what_the_api_said(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        admin_page.render_ops, "trigger_precompute_run", lambda _env: (True, "started")
    )
    st_module = _St(pressed="Trigger precompute run")

    admin_page._render_ops_controls(st_module, env={"RENDER_API_KEY": "k"})

    assert st_module.successes == ["started"]


def test_clearing_caches_needs_no_render_credentials():
    st_module = _St(pressed="Clear dashboard caches")

    admin_page._render_ops_controls(st_module, env={})

    assert st_module.cleared == 2
    assert st_module.successes == ["Caches cleared."]


def test_an_unwarmed_season_is_called_out_as_a_post_deploy_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.dashboard.cache.get_artifact_versions", lambda _year: {})
    monkeypatch.setattr(
        "src.dashboard.precomputed_predictions.compute_artifact_hash", lambda _v: "76d26fcf00"
    )
    monkeypatch.setattr(
        "src.dashboard.precomputed_predictions.load_precompute_horizon_index",
        lambda **_kwargs: None,
    )
    st_module = _St()

    status = admin_page._read_precompute_status(2026)
    admin_page._render_precompute_detail(status, st_module)

    assert status["artifact_hash"] == "76d26fcf00"
    assert any("Nothing is warmed" in message for message in st_module.errors)


def test_a_partly_warmed_season_names_the_races_still_missing():
    status = {
        "artifact_hash": "abc",
        "horizon": {
            "ready_races": ["Italian Grand Prix"],
            "expected_targets": ["Italian Grand Prix", "Azerbaijan Grand Prix"],
        },
        "error": None,
    }
    st_module = _St()

    admin_page._render_precompute_detail(status, st_module)

    assert any("Azerbaijan Grand Prix" in message for message in st_module.warnings)


def test_the_hero_deck_summarises_the_precompute_state(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.dashboard.rendering.render_page_hero_deck",
        lambda **kwargs: captured.update(kwargs),
    )
    status = {
        "artifact_hash": "76d26fcf0012",
        "horizon": {
            "ready_races": ["Italian Grand Prix"],
            "expected_targets": ["Italian Grand Prix"],
        },
        "error": None,
    }

    admin_page._render_hero(_St(), year=2026, status=status, env={})

    values = {card["label"]: card["value"] for card in captured["cards"]}
    assert values == {
        "Season": "2026",
        "Warmed": "1/1",
        "Artifact": "76d26fcf",
        "Render ops": "Not set",
    }


def test_the_admin_tab_exists_only_for_the_operator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(layout, "NAVIGATION_PAGES", ["Prediction", "Contact"])

    monkeypatch.setattr(
        "src.dashboard.grid_penalty_admin.admin_access_granted", lambda *_a, **_k: False
    )
    assert layout._navigation_pages() == ["Prediction", "Contact"]

    monkeypatch.setattr(
        "src.dashboard.grid_penalty_admin.admin_access_granted", lambda *_a, **_k: True
    )
    assert layout._navigation_pages() == ["Prediction", "Contact", "Admin"]


def test_the_operator_lands_on_admin_not_on_a_forecast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(layout, "NAVIGATION_PAGES", ["Prediction", "Contact"])
    monkeypatch.setattr(
        "src.dashboard.grid_penalty_admin.admin_access_granted", lambda *_a, **_k: True
    )

    # Admin is listed last, but it is still where an operator lands.
    assert layout._navigation_pages()[-1] == "Admin"
    assert layout._active_navigation_page() == "Admin"


def test_a_broken_admin_check_leaves_navigation_intact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(layout, "NAVIGATION_PAGES", ["Prediction", "Contact"])

    def _explode(*_a: Any, **_k: Any) -> bool:
        raise RuntimeError("query params unavailable")

    monkeypatch.setattr("src.dashboard.grid_penalty_admin.admin_access_granted", _explode)

    assert layout._navigation_pages() == ["Prediction", "Contact"]


def test_the_activity_feed_stays_quiet_without_render_credentials():
    st_module = _St()

    admin_page._render_render_activity(st_module, env={})

    assert st_module.buttons == []
    assert st_module.markdowns == []


def test_a_failed_run_is_shown_with_its_cause(monkeypatch: pytest.MonkeyPatch):
    configured = {
        "RENDER_API_KEY": "k",
        "RENDER_PRECOMPUTE_CRON_ID": "crn-a",
        "RENDER_WEB_SERVICE_ID": "srv-b",
    }
    monkeypatch.setattr(
        admin_page.render_ops,
        "precompute_run_events",
        lambda _env: (
            [
                {
                    "timestamp": "2026-09-02 19:42",
                    "type": "cron job run ended",
                    "outcome": "unsuccessful, oomKilled",
                }
            ],
            "",
        ),
    )
    monkeypatch.setattr(admin_page.render_ops, "web_service_events", lambda _env: ([], ""))
    st_module = _St()

    admin_page._render_render_activity(st_module, env=configured)

    assert st_module.buttons == ["Refresh"]
    assert any("oomKilled" in caption for caption in st_module.captions)
