"""Direct tests for extracted dashboard rendering modules."""

from __future__ import annotations

import pandas as pd

from src.dashboard import rendering_html, rendering_qualifying, rendering_race


def test_rendering_html_build_surface_header_html_escapes_inputs() -> None:
    html = rendering_html._build_surface_header_html(
        title="Race <Sim>",
        summary="Summary & detail",
        eyebrow="Weekend",
        tone="success",
    )

    assert "ts-surface-header--success" in html
    assert "Race &lt;Sim&gt;" in html
    assert "Summary &amp; detail" in html
    assert "Weekend" in html


def test_rendering_race_build_position_change_frame_merges_and_sorts_rows() -> None:
    finish_df = pd.DataFrame(
        [
            {"position": 1, "driver": "NOR", "team": "McLaren"},
            {"position": 2, "driver": "LEC", "team": "Ferrari"},
            {"position": 3, "driver": "VER", "team": "Red Bull Racing"},
        ]
    )
    starting_grid = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
        {"position": 3, "driver": "LEC", "team": "Ferrari"},
    ]

    frame = rendering_race._build_position_change_frame(finish_df, starting_grid)

    assert list(frame["driver"]) == ["NOR", "LEC", "VER"]
    assert list(frame["positions_gained"]) == [1, 1, -2]
    assert list(frame["team"]) == ["McLaren", "Ferrari", "Red Bull Racing"]


def test_rendering_qualifying_renders_teammate_matchups_directly(monkeypatch) -> None:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    messages: list[str] = []
    monkeypatch.setattr(
        rendering_qualifying.st,
        "expander",
        lambda *_args, **_kwargs: _Ctx(),
    )
    monkeypatch.setattr(
        rendering_qualifying.st,
        "markdown",
        lambda message, **_kwargs: messages.append(str(message)),
    )

    rendering_qualifying._render_teammate_head_to_head_probabilities(
        [
            {
                "team": "McLaren",
                "driver_a": "NOR",
                "driver_b": "PIA",
                "p_driver_a_ahead": 0.72,
                "n_samples": 1000,
            },
            {
                "team": "Ferrari",
                "driver_a": "LEC",
                "driver_b": "HAM",
                "p_driver_a_ahead": "bad",
                "n_samples": 1000,
            },
        ]
    )

    assert any("How to read" in message for message in messages)
    assert any("McLaren" in message and "moderate edge" in message for message in messages)
    assert not any("Ferrari" in message for message in messages)
