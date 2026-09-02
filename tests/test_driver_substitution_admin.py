"""The substitution editor is operator-only, and it reads a whole chain in one submit."""

from typing import Any

import pytest

from src.dashboard.driver_substitution_admin import (
    _format,
    _parse,
    render_driver_substitution_editor,
)


class _St:
    """Streamlit stub exposing only what the gate reads."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.query_params = params or {}
        self.rendered = False

    def expander(self, *_args, **_kwargs):
        self.rendered = True
        raise AssertionError("the editor must not render without the token")


def test_an_ordinary_visitor_renders_nothing():
    st_module = _St({"admin": "guess"})

    render_driver_substitution_editor(
        race_name="Dutch Grand Prix",
        year=2026,
        st_module=st_module,
        env={"TL_ADMIN_TOKEN": "secret"},
    )

    assert st_module.rendered is False


def test_a_chain_is_read_one_swap_per_line():
    assert _parse("HAD > LAW\nlaw -> tsu\n\n") == {"HAD": "LAW", "LAW": "TSU"}


@pytest.mark.parametrize("text", ["HAD LAW", "HAD >", "HAD > LAW > TSU"])
def test_a_line_that_is_not_a_swap_is_refused(text):
    with pytest.raises(ValueError):
        _parse(text)


def test_the_stored_map_round_trips_through_the_text_box():
    substitutions = {"HAD": "LAW", "LAW": "TSU"}

    assert _parse(_format(substitutions)) == substitutions
