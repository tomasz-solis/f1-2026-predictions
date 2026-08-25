"""The penalty editor is operator-only on a public dashboard."""

from typing import Any

from src.dashboard.grid_penalty_admin import admin_access_granted, render_grid_penalty_editor


class _St:
    """Streamlit stub exposing only what the gate reads."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.query_params = params or {}
        self.rendered = False

    def expander(self, *_args, **_kwargs):
        self.rendered = True
        raise AssertionError("the editor must not render without the token")


def test_no_token_configured_means_no_editor():
    assert admin_access_granted(env={}, st_module=_St({"admin": "anything"})) is False


def test_a_wrong_token_is_refused():
    assert (
        admin_access_granted(env={"TL_ADMIN_TOKEN": "secret"}, st_module=_St({"admin": "guess"}))
        is False
    )


def test_a_missing_query_parameter_is_refused():
    assert admin_access_granted(env={"TL_ADMIN_TOKEN": "secret"}, st_module=_St()) is False


def test_the_matching_token_is_accepted():
    assert (
        admin_access_granted(env={"TL_ADMIN_TOKEN": "secret"}, st_module=_St({"admin": "secret"}))
        is True
    )


def test_a_list_valued_query_parameter_still_matches():
    assert (
        admin_access_granted(env={"TL_ADMIN_TOKEN": "secret"}, st_module=_St({"admin": ["secret"]}))
        is True
    )


def test_an_ordinary_visitor_renders_nothing():
    st_module = _St({"admin": "guess"})

    render_grid_penalty_editor(
        race_name="Italian Grand Prix",
        year=2026,
        st_module=st_module,
        env={"TL_ADMIN_TOKEN": "secret"},
    )

    assert st_module.rendered is False
