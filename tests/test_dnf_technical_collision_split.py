"""Unit tests for the technical/collision DNF split (docs/DNF_CALIBRATION_BRIEF.md).

The split is gated behind ``dnf_technical_collision_split_enabled`` (default False).
With the flag off, the DNF probability must be byte-identical to the pre-split
model. With it on, ``p_dnf = 1-(1-p_technical)*(1-p_collision)`` with a team/PU-level
technical component that decays across the 2026 regulation reset, and a
driver x track collision component.
"""

from __future__ import annotations

import pytest

from src.predictors.baseline.race.preparation_flow import (
    _EXPERIENCE_DNF_MODIFIERS,
    _combine_independent_dnf_risks,
    _compute_driver_dnf_probability,
    _resolve_collision_dnf_probability,
    _resolve_driver_collision_crash_rate,
    _resolve_technical_dnf_probability,
)

_CAP_ARGS = dict(
    dnf_rate_historical_cap=0.20,
    dnf_rate_final_cap=0.35,
    dnf_rate_floor=0.02,
    team_uncertainty_dnf_multiplier=0.20,
)


def _old_formula(dnf_rate: float, tier: str, team_uncertainty: float) -> float:
    """Reference re-implementation of the pre-split computation."""
    rate = min(max(0.0, dnf_rate), _CAP_ARGS["dnf_rate_historical_cap"])
    floor = min(max(_CAP_ARGS["dnf_rate_floor"], 0.0), _CAP_ARGS["dnf_rate_final_cap"])
    mod = _EXPERIENCE_DNF_MODIFIERS.get(tier, 0.0)
    if team_uncertainty >= 0.40:
        adjusted = rate + mod + team_uncertainty * _CAP_ARGS["team_uncertainty_dnf_multiplier"]
    else:
        adjusted = rate + mod
    return max(floor, min(adjusted, _CAP_ARGS["dnf_rate_final_cap"]))


def _compute(flag: bool, *, dnf_rate: float, tier: str, team_uncertainty: float, **kw) -> float:
    return _compute_driver_dnf_probability(
        {"dnf_risk": {"dnf_rate": dnf_rate}},
        team_uncertainty=team_uncertainty,
        resolve_effective_experience_tier_for_race_fn=lambda _d: tier,
        technical_collision_split_enabled=flag,
        **_CAP_ARGS,
        **kw,
    )


# -- flag-off byte-identical ---------------------------------------------------


@pytest.mark.parametrize("dnf_rate", [0.02, 0.10, 0.18, 0.30])
@pytest.mark.parametrize("tier", ["rookie", "second_year", "developing", "established"])
@pytest.mark.parametrize("team_uncertainty", [0.0, 0.30, 0.40, 0.80])
def test_flag_off_matches_old_formula_exactly(dnf_rate, tier, team_uncertainty):
    got = _compute(False, dnf_rate=dnf_rate, tier=tier, team_uncertainty=team_uncertainty)
    assert got == _old_formula(dnf_rate, tier, team_uncertainty)


def test_flag_off_ignores_all_new_technical_params():
    """With the flag off, none of the new kwargs may influence the result."""
    base = _compute(False, dnf_rate=0.10, tier="established", team_uncertainty=0.0)
    perturbed = _compute(
        False,
        dnf_rate=0.10,
        tier="established",
        team_uncertainty=0.0,
        races_completed=0,
        technical_floor=0.30,
        technical_amplitude=0.30,
        technical_decay_tau_races=1.0,
        collision_multiplier=5.0,
    )
    assert base == perturbed


# -- technical decay -----------------------------------------------------------


def test_technical_decays_from_high_prior_toward_floor():
    floor, amp, tau = 0.02, 0.08, 8.0
    at_zero = _resolve_technical_dnf_probability(
        races_completed=0, floor=floor, amplitude=amp, decay_tau_races=tau
    )
    late = _resolve_technical_dnf_probability(
        races_completed=40, floor=floor, amplitude=amp, decay_tau_races=tau
    )
    assert at_zero == pytest.approx(floor + amp)  # new-regs high prior
    assert late == pytest.approx(floor, abs=1e-3)  # levelled off
    assert at_zero > late


def test_technical_is_monotonically_non_increasing_in_races():
    vals = [
        _resolve_technical_dnf_probability(
            races_completed=n, floor=0.02, amplitude=0.08, decay_tau_races=8.0
        )
        for n in range(0, 20)
    ]
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))


def test_technical_is_grid_uniform_documents_current_limitation():
    """Technical rate depends only on races_completed, not driver/team identity
    (the honest v1 limitation flagged in the brief: not yet per-team)."""
    a = _resolve_technical_dnf_probability(
        races_completed=3, floor=0.02, amplitude=0.08, decay_tau_races=8.0
    )
    b = _resolve_technical_dnf_probability(
        races_completed=3, floor=0.02, amplitude=0.08, decay_tau_races=8.0
    )
    assert a == b


# -- collision component -------------------------------------------------------


@pytest.mark.parametrize(
    "rate,mult,expected",
    [(0.10, 1.0, 0.10), (0.10, 2.0, 0.20), (0.10, 0.0, 0.0), (0.60, 2.0, 1.0)],
)
def test_collision_scales_driver_rate_by_track_multiplier(rate, mult, expected):
    assert _resolve_collision_dnf_probability(rate, mult) == pytest.approx(expected)


# -- combination ---------------------------------------------------------------


def test_combine_independent_risks_formula():
    assert _combine_independent_dnf_risks(0.10, 0.10) == pytest.approx(0.19)
    assert _combine_independent_dnf_risks(0.0, 0.25) == pytest.approx(0.25)
    assert _combine_independent_dnf_risks(1.0, 0.5) == pytest.approx(1.0)


# -- split-on end-to-end -------------------------------------------------------


def test_split_on_new_team_early_season_elevated_even_with_no_track_record():
    """races_completed=0 (reg-reset era) -> high technical prior. v2: with no
    ``collision_dnf_track_record`` and no ``collision_prior_by_tier`` supplied,
    collision falls back to the resolver's own established-tier default (0.08)
    -- the per-driver EMA ``dnf_rate`` is NO LONGER consulted for collision at
    all under the split (see test_v2_collision_ignores_raw_dnf_rate below)."""
    p = _compute(
        True,
        dnf_rate=0.02,
        tier="established",
        team_uncertainty=0.0,
        races_completed=0,
        technical_floor=0.02,
        technical_amplitude=0.08,
        technical_decay_tau_races=8.0,
        collision_multiplier=1.0,
    )
    # technical=0.10, collision=0.08 (established-tier fallback) -> 1-(0.9)(0.92)=0.172
    assert p == pytest.approx(0.172, abs=1e-3)


def test_v2_collision_ignores_raw_dnf_rate_without_a_prior_or_track_record():
    """The v1 per-driver EMA dnf_rate must NOT leak into v2's collision
    component -- only collision_prior_by_tier / collision_dnf_track_record do."""
    low_rate = _compute(
        True, dnf_rate=0.02, tier="established", team_uncertainty=0.0, races_completed=0
    )
    high_rate = _compute(
        True, dnf_rate=0.30, tier="established", team_uncertainty=0.0, races_completed=0
    )
    assert low_rate == pytest.approx(high_rate)


def test_split_on_respects_final_cap_and_floor():
    high = _compute(
        True,
        dnf_rate=0.20,
        tier="established",
        team_uncertainty=0.0,
        races_completed=0,
        technical_floor=0.30,
        technical_amplitude=0.30,
        technical_decay_tau_races=8.0,
        collision_multiplier=3.0,
    )
    assert high == pytest.approx(_CAP_ARGS["dnf_rate_final_cap"])  # clamped to final cap

    low = _compute(
        True,
        dnf_rate=0.0,
        tier="established",
        team_uncertainty=0.0,
        races_completed=100,
        technical_floor=0.0,
        technical_amplitude=0.0,
        technical_decay_tau_races=8.0,
        collision_multiplier=0.0,
    )
    assert low == pytest.approx(_CAP_ARGS["dnf_rate_floor"])  # clamped up to floor


def test_split_on_decays_toward_collision_only_late_season():
    """Late season, technical -> floor, so combined ≈ 1-(1-floor)*(1-collision),
    where v2's collision fallback (no prior dict / track record) is the
    resolver's established-tier default (0.08), not the raw dnf_rate."""
    p = _compute(
        True,
        dnf_rate=0.10,
        tier="established",
        team_uncertainty=0.0,
        races_completed=60,
        technical_floor=0.02,
        technical_amplitude=0.08,
        technical_decay_tau_races=8.0,
        collision_multiplier=1.0,
    )
    assert p == pytest.approx(1.0 - (1.0 - 0.02) * (1.0 - 0.08), abs=1e-3)


# -- v2: per-team technical evidence blend --------------------------------------


def test_technical_v2_new_team_no_evidence_returns_pure_prior():
    """A brand-new team (n=0, e.g. Cadillac before its first race) must return
    EXACTLY the v1 grid-uniform prior, regardless of prior_strength."""
    prior = _resolve_technical_dnf_probability(
        races_completed=0, floor=0.02, amplitude=0.08, decay_tau_races=8.0
    )
    with_record = _resolve_technical_dnf_probability(
        races_completed=0,
        floor=0.02,
        amplitude=0.08,
        decay_tau_races=8.0,
        team_technical_track_record={"races_observed": 0, "dnf_rate": 0.30},
        technical_prior_strength=3.0,
    )
    assert with_record == pytest.approx(prior)


def test_technical_v2_reliable_team_pulls_below_a_new_teams_prior():
    """A team with real evidence of a LOW mechanical rate must sit below both
    the pure prior and a new team with no evidence."""
    prior = _resolve_technical_dnf_probability(
        races_completed=0, floor=0.02, amplitude=0.08, decay_tau_races=8.0
    )
    reliable = _resolve_technical_dnf_probability(
        races_completed=0,
        floor=0.02,
        amplitude=0.08,
        decay_tau_races=8.0,
        team_technical_track_record={"races_observed": 10, "dnf_rate": 0.02},
        technical_prior_strength=3.0,
    )
    assert reliable < prior


def test_technical_v2_unreliable_team_sits_above_a_reliable_team():
    """Cadillac (persistently high observed mechanical rate) must end up above
    an established team with a low observed rate -- the whole point of the
    per-team split (v1's grid-uniform curve could never express this)."""
    shared_kwargs = dict(
        races_completed=5,
        floor=0.02,
        amplitude=0.08,
        decay_tau_races=8.0,
        technical_prior_strength=3.0,
    )
    cadillac = _resolve_technical_dnf_probability(
        team_technical_track_record={"races_observed": 5, "dnf_rate": 0.35}, **shared_kwargs
    )
    reliable_team = _resolve_technical_dnf_probability(
        team_technical_track_record={"races_observed": 5, "dnf_rate": 0.02}, **shared_kwargs
    )
    assert cadillac > reliable_team


def test_technical_v2_blend_shifts_toward_observed_as_evidence_grows():
    """Holding the observed rate fixed, more races_observed must pull the
    blended value monotonically closer to that observed rate."""
    observed_rate = 0.30
    prior = _resolve_technical_dnf_probability(
        races_completed=0, floor=0.02, amplitude=0.08, decay_tau_races=8.0
    )
    values = [
        _resolve_technical_dnf_probability(
            races_completed=0,
            floor=0.02,
            amplitude=0.08,
            decay_tau_races=8.0,
            team_technical_track_record={"races_observed": n, "dnf_rate": observed_rate},
            technical_prior_strength=3.0,
        )
        for n in (1, 3, 10, 50)
    ]
    distances = [abs(observed_rate - v) for v in values]
    assert distances == sorted(distances, reverse=True)  # monotonically closer to observed
    assert all(v > prior for v in values)  # observed_rate (0.30) > prior, so blend moves up
    assert values[-1] == pytest.approx(observed_rate, abs=0.02)  # n=50 nearly converged


def test_technical_v2_leakage_uses_only_the_supplied_prior_event_snapshot():
    """The resolver is a pure function of its arguments -- races_observed/
    dnf_rate are season-state values that ``_update_team_technical_dnf_rate_ema``
    only ever writes AFTER a race completes, so passing a fixed snapshot here
    is by construction the leakage-safe "events strictly before target" state.
    Calling twice with the identical snapshot must be perfectly reproducible
    (no hidden global/mutable state leaking in future information)."""
    record = {"races_observed": 4, "dnf_rate": 0.12}
    first = _resolve_technical_dnf_probability(
        races_completed=4,
        floor=0.02,
        amplitude=0.08,
        decay_tau_races=8.0,
        team_technical_track_record=record,
        technical_prior_strength=3.0,
    )
    second = _resolve_technical_dnf_probability(
        races_completed=4,
        floor=0.02,
        amplitude=0.08,
        decay_tau_races=8.0,
        team_technical_track_record=record,
        technical_prior_strength=3.0,
    )
    assert first == second


# -- v2: driver-specific adaptive collision --------------------------------------


def test_collision_v2_rookie_with_no_history_returns_the_tier_prior():
    priors = {"rookie": 0.13, "established": 0.05}
    rate = _resolve_driver_collision_crash_rate(
        experience_tier="rookie",
        collision_prior_by_tier=priors,
        driver_collision_track_record=None,
        collision_prior_strength=5.0,
    )
    assert rate == pytest.approx(0.13)

    rate_zero_races = _resolve_driver_collision_crash_rate(
        experience_tier="rookie",
        collision_prior_by_tier=priors,
        driver_collision_track_record={"races_observed": 0, "collisions_observed": 0},
        collision_prior_strength=5.0,
    )
    assert rate_zero_races == pytest.approx(0.13)


def test_collision_v2_many_clean_races_converge_well_below_the_rookie_prior():
    """The 'young-crashy-Max becomes consistent-Max' arc: a driver who started
    as a rookie (high prior) but has accumulated many collision-free races
    converges toward their own near-zero observed rate."""
    priors = {"rookie": 0.13, "established": 0.05}
    converged = _resolve_driver_collision_crash_rate(
        experience_tier="rookie",
        collision_prior_by_tier=priors,
        driver_collision_track_record={"races_observed": 80, "collisions_observed": 2},
        collision_prior_strength=5.0,
    )
    assert converged < 0.13 * 0.5  # well below the rookie prior
    assert converged == pytest.approx(2 / 80, abs=0.01)  # close to the true observed rate


def test_collision_v2_convergence_is_monotonic_in_races_observed():
    priors = {"established": 0.08}
    observed_rate = 0.02
    values = []
    for n in (1, 5, 20, 100):
        collisions = round(observed_rate * n)
        values.append(
            _resolve_driver_collision_crash_rate(
                experience_tier="established",
                collision_prior_by_tier=priors,
                driver_collision_track_record={
                    "races_observed": n,
                    "collisions_observed": collisions,
                },
                collision_prior_strength=5.0,
            )
        )
    distances = [abs(observed_rate - v) for v in values]
    assert distances == sorted(distances, reverse=True)


def test_collision_v2_empty_prior_dict_falls_back_safely():
    rate = _resolve_driver_collision_crash_rate(
        experience_tier="rookie", collision_prior_by_tier={}, driver_collision_track_record=None
    )
    assert rate == pytest.approx(0.08)  # the resolver's own hardcoded safety fallback


# -- v2: end-to-end through _compute_driver_dnf_probability ----------------------


def test_compute_dnf_probability_v2_wires_team_and_driver_track_records():
    """A driver with a bad team AND a bad personal collision record must score
    higher than an otherwise-identical driver on a good team with a clean
    record, once both v2 fields are populated on driver_data."""
    priors = {"established": 0.05}
    common = dict(
        team_uncertainty=0.0,
        resolve_effective_experience_tier_for_race_fn=lambda _d: "established",
        technical_collision_split_enabled=True,
        races_completed=5,
        technical_floor=0.02,
        technical_amplitude=0.08,
        technical_decay_tau_races=8.0,
        technical_prior_strength=3.0,
        collision_prior_by_tier=priors,
        collision_prior_strength=5.0,
        collision_multiplier=1.0,
        **_CAP_ARGS,
    )
    risky = _compute_driver_dnf_probability(
        {
            "dnf_risk": {"dnf_rate": 0.10},
            "team_technical_dnf_risk": {"races_observed": 5, "dnf_rate": 0.35},
            "collision_dnf_track_record": {"races_observed": 20, "collisions_observed": 6},
        },
        **common,
    )
    safe = _compute_driver_dnf_probability(
        {
            "dnf_risk": {"dnf_rate": 0.10},
            "team_technical_dnf_risk": {"races_observed": 5, "dnf_rate": 0.02},
            "collision_dnf_track_record": {"races_observed": 20, "collisions_observed": 0},
        },
        **common,
    )
    assert risky > safe
