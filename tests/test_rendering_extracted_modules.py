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


def test_rendering_html_short_data_source_label_prefers_checkpoint_blend() -> None:
    label = rendering_html._short_data_source_label(
        "FP2 checkpoint profile blend (latest stored snapshot: Australian Grand Prix / FP2)",
        blend_used=True,
    )

    assert label == "Checkpoint blend"


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


def test_rendering_race_movement_ladder_rows_only_returns_movers() -> None:
    """Movement ladder should not show drivers projected to hold position."""
    comparison = pd.DataFrame(
        [
            {
                "driver": "NOR",
                "team": "McLaren",
                "start_position": 1,
                "finish_position": 1,
                "positions_gained": 0,
            },
            {
                "driver": "LEC",
                "team": "Ferrari",
                "start_position": 5,
                "finish_position": 2,
                "positions_gained": 3,
            },
            {
                "driver": "VER",
                "team": "Red Bull Racing",
                "start_position": 2,
                "finish_position": 4,
                "positions_gained": -2,
            },
        ]
    )

    ladder_rows = rendering_race._movement_ladder_rows(comparison)

    assert list(ladder_rows["driver"]) == ["VER", "LEC"]
    assert all(ladder_rows["positions_gained"] != 0)


def test_rendering_race_movement_ladder_reserves_hover_space() -> None:
    rows = pd.DataFrame(
        [
            {
                "driver": "LEC",
                "team": "Ferrari",
                "start_position": 5,
                "finish_position": 2,
                "positions_gained": 3,
            },
            {
                "driver": "VER",
                "team": "Red Bull Racing",
                "start_position": 2,
                "finish_position": 4,
                "positions_gained": -2,
            },
        ]
    )

    figure = rendering_race._position_change_ladder_figure(rows)

    assert tuple(figure.layout.xaxis.range) == (-0.55, 1.90)
    assert tuple(figure.layout.yaxis.range) == (5.7, 1.3)
    assert figure.layout.margin.to_plotly_json() == {"l": 16, "r": 16, "t": 20, "b": 40}
    assert figure.layout.title.text is None
    assert figure.layout.height == 340
    assert "P%{customdata[2]} → P%{customdata[3]}" in figure.data[0].hovertemplate
    assert figure.layout.annotations[1].text == "LEC P2 +3"


def test_rendering_race_movement_ladder_adds_height_for_field_span() -> None:
    rows = pd.DataFrame(
        [
            {
                "driver": "LEC",
                "team": "Ferrari",
                "start_position": 15,
                "finish_position": 2,
                "positions_gained": 13,
            }
        ]
    )

    figure = rendering_race._position_change_ladder_figure(rows)

    assert figure.layout.height == 560


def test_rendering_qualifying_orders_teammate_matchups_by_edge() -> None:
    """Teammate cards should read strongest simulated edge first."""
    rows = rendering_qualifying._normalize_teammate_matchups(
        [
            {
                "team": "McLaren",
                "driver_a": "NOR",
                "driver_b": "PIA",
                "p_driver_a_ahead": "0.62",
                "n_samples": 1000,
            },
            {
                "team": "Mercedes",
                "driver_a": "RUS",
                "driver_b": "ANT",
                "p_driver_a_ahead": 0.84,
                "n_samples": 1000,
            },
            {
                "team": "Ferrari",
                "driver_a": "LEC",
                "driver_b": "HAM",
                "p_driver_a_ahead": 0.44,
                "n_samples": 1000,
            },
        ]
    )

    assert [row["team"] for row in rows] == ["Mercedes", "McLaren", "Ferrari"]
    assert [row["favorite"] for row in rows] == ["RUS", "NOR", "HAM"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


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
    assert any("#1" in message and "+22.0 pp toward NOR" in message for message in messages)
    assert any("50/50" in message and "PIA" in message for message in messages)
    assert not any("Ferrari" in message for message in messages)
