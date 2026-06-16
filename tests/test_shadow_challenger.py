from __future__ import annotations

from src.models.shadow_challenger import build_shadow_challengers_for_prediction
from src.utils.accuracy_targets import TARGET_MAIN_QUALIFYING


def _prediction(
    race_name: str,
    predicted_order: list[dict],
    actual_order: list[dict] | None,
    *,
    predicted_at: str,
) -> dict:
    return {
        "metadata": {
            "year": 2026,
            "race_name": race_name,
            "session_name": "FP1",
            "predicted_at": predicted_at,
            "weekend_format": "normal",
        },
        "targets": {
            TARGET_MAIN_QUALIFYING: {
                "target_session": "Q",
                "predicted_order": predicted_order,
                "eligible_at_save": True,
            }
        },
        "actuals": {
            "targets": {
                TARGET_MAIN_QUALIFYING: actual_order,
            }
        },
    }


def test_shadow_challenger_blends_main_qualifying_with_prior_target_actuals():
    historical = [
        _prediction(
            "Australian Grand Prix",
            [
                {"position": 1, "driver": "BBB", "team": "B"},
                {"position": 2, "driver": "AAA", "team": "A"},
            ],
            [
                {"position": 1, "driver": "AAA", "team": "A"},
                {"position": 2, "driver": "BBB", "team": "B"},
            ],
            predicted_at="2026-03-01T10:00:00+00:00",
        )
    ]
    current = _prediction(
        "Chinese Grand Prix",
        [
            {"position": 1, "driver": "BBB", "team": "B"},
            {"position": 2, "driver": "AAA", "team": "A"},
        ],
        None,
        predicted_at="2026-03-10T10:00:00+00:00",
    )

    challengers = build_shadow_challengers_for_prediction(
        current,
        historical_predictions=historical,
    )

    challenger = challengers[TARGET_MAIN_QUALIFYING]
    assert challenger["status"] == "active"
    assert [row["driver"] for row in challenger["predicted_order"]] == ["AAA", "BBB"]
    assert challenger["uses_current_event_actuals"] is False


def test_shadow_challenger_excludes_same_race_actuals_from_history():
    historical = [
        _prediction(
            "Australian Grand Prix",
            [
                {"position": 1, "driver": "BBB", "team": "B"},
                {"position": 2, "driver": "AAA", "team": "A"},
            ],
            [
                {"position": 1, "driver": "AAA", "team": "A"},
                {"position": 2, "driver": "BBB", "team": "B"},
            ],
            predicted_at="2026-03-01T10:00:00+00:00",
        ),
        _prediction(
            "Chinese Grand Prix",
            [
                {"position": 1, "driver": "BBB", "team": "B"},
                {"position": 2, "driver": "AAA", "team": "A"},
            ],
            [
                {"position": 1, "driver": "BBB", "team": "B"},
                {"position": 2, "driver": "AAA", "team": "A"},
            ],
            predicted_at="2026-03-11T10:00:00+00:00",
        ),
    ]
    current = _prediction(
        "Chinese Grand Prix",
        [
            {"position": 1, "driver": "BBB", "team": "B"},
            {"position": 2, "driver": "AAA", "team": "A"},
        ],
        None,
        predicted_at="2026-03-10T10:00:00+00:00",
    )

    challengers = build_shadow_challengers_for_prediction(
        current,
        historical_predictions=historical,
    )

    assert [row["driver"] for row in challengers[TARGET_MAIN_QUALIFYING]["predicted_order"]] == [
        "AAA",
        "BBB",
    ]
