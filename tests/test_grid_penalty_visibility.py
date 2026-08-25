"""Penalties must be visible wherever a penalised grid is shown."""

from src.dashboard.accuracy_view import _starting_grid_from_saved_penalties
from src.dashboard.rendering_race import _render_grid_penalty_notice, format_grid_penalty


class _St:
    """Capture banner calls from the notice renderer."""

    def __init__(self):
        self.messages: list[str] = []


def test_a_saved_checkpoint_rebuilds_the_grid_it_actually_raced_from():
    """A checkpoint stores the qualifying order plus the penalties, not the moved grid."""
    qualifying_rows = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "ANT", "team": "Mercedes", "position": 2},
        {"driver": "NOR", "team": "McLaren", "position": 3},
    ]
    saved = [{"driver": "ANT", "qualified": 2, "starts": 3, "penalty": "20"}]

    rebuilt = _starting_grid_from_saved_penalties(qualifying_rows, saved)

    assert [row["driver"] for row in rebuilt] == ["VER", "NOR", "ANT"]
    assert [row["position"] for row in rebuilt] == [1, 2, 3]


def test_a_checkpoint_without_penalties_keeps_the_qualifying_order():
    qualifying_rows = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "ANT", "team": "Mercedes", "position": 2},
    ]

    assert _starting_grid_from_saved_penalties(qualifying_rows, []) is qualifying_rows


def test_penalty_descriptions_read_like_a_race_engineer_said_them():
    assert (
        format_grid_penalty({"driver": "ANT", "qualified": 3, "starts": 22, "penalty": "20"})
        == "ANT P3 -> P22 (20-place penalty)"
    )
    assert (
        format_grid_penalty({"driver": "ANT", "qualified": 3, "starts": 22, "penalty": "pit"})
        == "ANT P3 -> P22 (pit-lane start)"
    )
    assert (
        format_grid_penalty({"driver": "ANT", "qualified": 3, "starts": 22, "penalty": "back"})
        == "ANT P3 -> P22 (back of the grid)"
    )


def test_the_notice_stays_silent_when_no_penalty_was_applied(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(
        "src.dashboard.rendering_race.render_notice_banner",
        lambda message, **_kwargs: captured.append(message),
    )

    _render_grid_penalty_notice({"finish_order": []})

    assert captured == []


def test_the_notice_names_every_penalised_driver(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(
        "src.dashboard.rendering_race.render_notice_banner",
        lambda message, **_kwargs: captured.append(message),
    )

    _render_grid_penalty_notice(
        {"grid_penalties": [{"driver": "ANT", "qualified": 3, "starts": 22, "penalty": "20"}]}
    )

    assert len(captured) == 1
    assert "ANT P3 -> P22 (20-place penalty)" in captured[0]
