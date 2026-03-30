"""Shared normalization helpers for performance comparisons."""

from __future__ import annotations

import numpy as np


def rank_normalize(
    values: dict[str, float],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Normalize metric values onto a 0-1 rank scale.

    Rank-based scoring is intentionally resistant to single outliers. Teams with
    tied values receive the same averaged rank.
    """
    if not values:
        return {}

    item_count = len(values)
    if item_count < 2:
        return {name: 0.5 for name in values}

    sorted_items = sorted(values.items(), key=lambda item: item[1], reverse=higher_is_better)

    grouped_items: list[tuple[float, list[str]]] = []
    for team_name, value in sorted_items:
        numeric_value = float(value)
        if grouped_items and np.isclose(numeric_value, grouped_items[-1][0]):
            grouped_items[-1][1].append(team_name)
        else:
            grouped_items.append((numeric_value, [team_name]))

    normalized: dict[str, float] = {}
    rank_cursor = 0
    for _value, tied_teams in grouped_items:
        average_rank = rank_cursor + ((len(tied_teams) - 1) / 2.0)
        score = float(1.0 - (average_rank / (item_count - 1)))
        for team_name in tied_teams:
            normalized[team_name] = float(np.clip(score, 0.0, 1.0))
        rank_cursor += len(tied_teams)

    return normalized
