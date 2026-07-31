"""Pure extraction of dry-practice evidence for a qualifying challenger.

The champion prediction path deliberately does not import this module.  It defines a
deterministic sidecar contract that can be exercised in replay/shadow mode before any
runtime integration is considered.

Pace normalization is learned from *paired, same-driver* observations supplied by the
caller.  This module contains no universal compound delta: sparse evidence falls back to
an explicit track-class prior (or a neutral zero prior when none is supplied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

QUALIFYING_PRACTICE_EVIDENCE_SCHEMA = "qualifying_practice_evidence"
QUALIFYING_PRACTICE_EVIDENCE_VERSION = 2

MIN_NORMALIZATION_COMPARISONS = 8
MIN_NORMALIZATION_DRIVERS = 4
MIN_NORMALIZATION_TEAMS = 3

QUALI_SIM_MAX_TIMED_LAPS = 5
QUALI_SIM_MAX_TYRE_AGE = 5.0
RACE_SIM_MIN_CONSECUTIVE_LAPS = 8

RUN_QUALI_SIM = "quali_sim"
RUN_RACE_SIM = "race_sim"
RUN_OTHER = "other"

DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})
WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET"})

NORMALIZATION_COMPARISON_COLUMNS = frozenset(
    {
        "driver",
        "team",
        "lap_time_a_s",
        "lap_time_b_s",
        "compound_a",
        "compound_b",
        "tyre_age_a",
        "tyre_age_b",
        "evolution_a",
        "evolution_b",
    }
)

_FEATURE_QUANTILE_FIELDS = (
    "best_adjusted_lap_s",
    "best_two_mean_adjusted_lap_s",
    "q20_adjusted_lap_s",
    "theoretical_adjusted_lap_s",
    "execution_loss_s",
    "mad_s",
)

_QUALITY_FILTER_COLUMNS = (
    "IsAccurate",
    "Deleted",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "Rainfall",
)


def _normalize_compound(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    aliases = {
        "S": "SOFT",
        "M": "MEDIUM",
        "H": "HARD",
        "I": "INTERMEDIATE",
        "W": "WET",
    }
    return aliases.get(text, text)


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


@dataclass(frozen=True)
class PracticeNormalizationPrior:
    """Track-class prior used when paired empirical evidence is sparse.

    ``compound_effect_s`` is intentionally empty by default.  A caller may supply a
    leakage-safe, historically fitted track-class prior; the extractor never invents a
    SOFT/MEDIUM/HARD delta.
    """

    reference_compound: str = "SOFT"
    compound_effect_s: dict[str, float] = field(default_factory=dict)
    tyre_age_effect_s_per_lap: float = 0.0
    evolution_effect_s_per_unit: float = 0.0
    uncertainty_s: float = 0.5
    source: str = "neutral_no_track_class_prior"

    def normalized(self) -> PracticeNormalizationPrior:
        normalized_reference = _normalize_compound(self.reference_compound)
        reference = normalized_reference if normalized_reference in DRY_COMPOUNDS else "SOFT"
        effects = {
            compound: _finite_float(effect)
            for raw_compound, effect in self.compound_effect_s.items()
            if (compound := _normalize_compound(raw_compound)) in DRY_COMPOUNDS
        }
        effects[reference] = 0.0
        return PracticeNormalizationPrior(
            reference_compound=reference,
            compound_effect_s=effects,
            tyre_age_effect_s_per_lap=_finite_float(self.tyre_age_effect_s_per_lap),
            evolution_effect_s_per_unit=_finite_float(self.evolution_effect_s_per_unit),
            uncertainty_s=max(0.0, _finite_float(self.uncertainty_s, default=0.5)),
            source=str(self.source),
        )


@dataclass(frozen=True)
class FittedPracticeNormalization:
    """Shrunk compound, tyre-age, and session-evolution normalization."""

    reference_compound: str
    compound_effect_s: dict[str, float]
    tyre_age_effect_s_per_lap: float
    evolution_effect_s_per_unit: float
    uncertainty_s: float
    provenance: str
    prior_source: str
    comparison_count: int
    driver_count: int
    team_count: int
    empirical_weight: float
    coefficient_provenance: dict[str, str]
    fallback_reasons: tuple[str, ...] = ()

    def adjustment_seconds(
        self,
        *,
        compound: Any,
        tyre_age: Any,
        evolution: Any,
    ) -> float:
        """Return the fitted non-pace contribution to subtract from a lap."""
        normalized_compound = _normalize_compound(compound)
        compound_effect = self.compound_effect_s.get(normalized_compound or "", 0.0)
        age = _finite_float(tyre_age, default=1.0)
        evolution_value = float(np.clip(_finite_float(evolution), 0.0, 1.0))
        return float(
            compound_effect
            + (self.tyre_age_effect_s_per_lap * (age - 1.0))
            + (self.evolution_effect_s_per_unit * evolution_value)
        )

    def normalize_lap_seconds(
        self,
        lap_seconds: Any,
        *,
        compound: Any,
        tyre_age: Any,
        evolution: Any,
    ) -> float:
        return _finite_float(lap_seconds) - self.adjustment_seconds(
            compound=compound,
            tyre_age=tyre_age,
            evolution=evolution,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_compound": self.reference_compound,
            "compound_effect_s": {
                key: float(value) for key, value in sorted(self.compound_effect_s.items())
            },
            "tyre_age_effect_s_per_lap": float(self.tyre_age_effect_s_per_lap),
            "evolution_effect_s_per_unit": float(self.evolution_effect_s_per_unit),
            "measurement_uncertainty_s": float(self.uncertainty_s),
            "provenance": self.provenance,
            "prior_source": self.prior_source,
            "comparison_coverage": {
                "comparisons": int(self.comparison_count),
                "drivers": int(self.driver_count),
                "teams": int(self.team_count),
                "minimums": {
                    "comparisons": MIN_NORMALIZATION_COMPARISONS,
                    "drivers": MIN_NORMALIZATION_DRIVERS,
                    "teams": MIN_NORMALIZATION_TEAMS,
                },
            },
            "empirical_weight": float(self.empirical_weight),
            "coefficient_provenance": dict(sorted(self.coefficient_provenance.items())),
            "fallback_reasons": list(self.fallback_reasons),
        }


def _prior_normalization(
    prior: PracticeNormalizationPrior,
    *,
    comparison_count: int,
    driver_count: int,
    team_count: int,
    reasons: list[str],
) -> FittedPracticeNormalization:
    normalized_prior = prior.normalized()
    coefficient_provenance = {
        f"compound:{compound}": (
            "reference" if compound == normalized_prior.reference_compound else "track_class_prior"
        )
        for compound in normalized_prior.compound_effect_s
    }
    coefficient_provenance.update(
        {
            "tyre_age": "track_class_prior",
            "track_evolution": "track_class_prior",
        }
    )
    return FittedPracticeNormalization(
        reference_compound=normalized_prior.reference_compound,
        compound_effect_s=dict(normalized_prior.compound_effect_s),
        tyre_age_effect_s_per_lap=normalized_prior.tyre_age_effect_s_per_lap,
        evolution_effect_s_per_unit=normalized_prior.evolution_effect_s_per_unit,
        uncertainty_s=normalized_prior.uncertainty_s,
        provenance="track_class_prior",
        prior_source=normalized_prior.source,
        comparison_count=comparison_count,
        driver_count=driver_count,
        team_count=team_count,
        empirical_weight=0.0,
        coefficient_provenance=coefficient_provenance,
        fallback_reasons=tuple(reasons),
    )


def fit_practice_normalization(
    comparisons: pd.DataFrame | None,
    *,
    prior: PracticeNormalizationPrior | None = None,
    prior_strength: float = 8.0,
) -> FittedPracticeNormalization:
    """Fit a ridge-shrunk normalizer from paired same-driver comparisons.

    Each comparison row must describe two comparable observations for the driver in
    ``driver`` and team in ``team``.  The caller is responsible for leakage-safe
    historical construction (same checkpoint/train fold).  Eight rows, four drivers,
    and three teams are all required before empirical coefficients are allowed.
    """
    normalized_prior = (prior or PracticeNormalizationPrior()).normalized()
    if comparisons is None or comparisons.empty:
        return _prior_normalization(
            normalized_prior,
            comparison_count=0,
            driver_count=0,
            team_count=0,
            reasons=["no_empirical_comparisons"],
        )

    missing = sorted(NORMALIZATION_COMPARISON_COLUMNS.difference(comparisons.columns))
    if missing:
        return _prior_normalization(
            normalized_prior,
            comparison_count=0,
            driver_count=0,
            team_count=0,
            reasons=[f"missing_comparison_columns:{','.join(missing)}"],
        )

    prepared = comparisons.loc[:, sorted(NORMALIZATION_COMPARISON_COLUMNS)].copy()
    for column in (
        "lap_time_a_s",
        "lap_time_b_s",
        "tyre_age_a",
        "tyre_age_b",
        "evolution_a",
        "evolution_b",
    ):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["compound_a"] = prepared["compound_a"].map(_normalize_compound)
    prepared["compound_b"] = prepared["compound_b"].map(_normalize_compound)
    numeric_columns = [
        "lap_time_a_s",
        "lap_time_b_s",
        "tyre_age_a",
        "tyre_age_b",
        "evolution_a",
        "evolution_b",
    ]
    valid_mask = (
        prepared[numeric_columns].notna().all(axis=1)
        & prepared["driver"].notna()
        & prepared["team"].notna()
        & prepared["compound_a"].isin(DRY_COMPOUNDS)
        & prepared["compound_b"].isin(DRY_COMPOUNDS)
        & prepared["lap_time_a_s"].gt(0.0)
        & prepared["lap_time_b_s"].gt(0.0)
    )
    valid = prepared[valid_mask].copy()

    comparison_count = int(len(valid))
    driver_count = int(valid["driver"].astype(str).nunique()) if not valid.empty else 0
    team_count = int(valid["team"].astype(str).nunique()) if not valid.empty else 0
    reasons: list[str] = []
    if comparison_count < MIN_NORMALIZATION_COMPARISONS:
        reasons.append(
            f"insufficient_comparisons:{comparison_count}<{MIN_NORMALIZATION_COMPARISONS}"
        )
    if driver_count < MIN_NORMALIZATION_DRIVERS:
        reasons.append(f"insufficient_drivers:{driver_count}<{MIN_NORMALIZATION_DRIVERS}")
    if team_count < MIN_NORMALIZATION_TEAMS:
        reasons.append(f"insufficient_teams:{team_count}<{MIN_NORMALIZATION_TEAMS}")
    if reasons:
        return _prior_normalization(
            normalized_prior,
            comparison_count=comparison_count,
            driver_count=driver_count,
            team_count=team_count,
            reasons=reasons,
        )

    reference = normalized_prior.reference_compound
    observed_compounds = sorted(
        set(valid["compound_a"]).union(valid["compound_b"]).difference({reference})
    )
    design_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    prior_coefficients: list[float] = []
    for compound in observed_compounds:
        contrast = (
            valid["compound_a"].eq(compound).astype(float)
            - valid["compound_b"].eq(compound).astype(float)
        ).to_numpy(dtype=float)
        if np.any(np.abs(contrast) > 0.0):
            design_columns.append(contrast)
            feature_names.append(f"compound:{compound}")
            prior_coefficients.append(normalized_prior.compound_effect_s.get(compound, 0.0))

    age_contrast = (valid["tyre_age_a"] - valid["tyre_age_b"]).to_numpy(dtype=float)
    if np.any(np.abs(age_contrast) > 0.0):
        design_columns.append(age_contrast)
        feature_names.append("tyre_age")
        prior_coefficients.append(normalized_prior.tyre_age_effect_s_per_lap)

    evolution_contrast = (valid["evolution_a"] - valid["evolution_b"]).to_numpy(dtype=float)
    if np.any(np.abs(evolution_contrast) > 0.0):
        design_columns.append(evolution_contrast)
        feature_names.append("track_evolution")
        prior_coefficients.append(normalized_prior.evolution_effect_s_per_unit)

    if not design_columns:
        return _prior_normalization(
            normalized_prior,
            comparison_count=comparison_count,
            driver_count=driver_count,
            team_count=team_count,
            reasons=["no_varying_normalization_features"],
        )

    design = np.column_stack(design_columns)
    target = (valid["lap_time_a_s"] - valid["lap_time_b_s"]).to_numpy(dtype=float)
    prior_vector = np.asarray(prior_coefficients, dtype=float)
    shrinkage = max(0.0, _finite_float(prior_strength, default=8.0))
    penalty = shrinkage * np.eye(design.shape[1], dtype=float)
    coefficients = np.linalg.solve(
        (design.T @ design) + penalty + (1e-12 * np.eye(design.shape[1])),
        (design.T @ target) + (penalty @ prior_vector),
    )

    compound_effects = dict(normalized_prior.compound_effect_s)
    compound_effects[reference] = 0.0
    age_effect = normalized_prior.tyre_age_effect_s_per_lap
    evolution_effect = normalized_prior.evolution_effect_s_per_unit
    coefficient_provenance = {
        f"compound:{compound}": ("reference" if compound == reference else "track_class_prior")
        for compound in compound_effects
    }
    coefficient_provenance.update(
        {"tyre_age": "track_class_prior", "track_evolution": "track_class_prior"}
    )

    for feature_name, coefficient in zip(feature_names, coefficients, strict=True):
        if feature_name.startswith("compound:"):
            compound = feature_name.split(":", 1)[1]
            compound_effects[compound] = float(coefficient)
        elif feature_name == "tyre_age":
            age_effect = float(coefficient)
        else:
            evolution_effect = float(coefficient)
        coefficient_provenance[feature_name] = "empirical_shrunk"

    residuals = target - (design @ coefficients)
    residual_uncertainty = float(np.sqrt(np.mean(np.square(residuals))))
    empirical_weight = float(comparison_count / (comparison_count + max(shrinkage, 1e-12)))
    uncertainty = float(
        np.sqrt(
            ((1.0 - empirical_weight) * normalized_prior.uncertainty_s**2)
            + (empirical_weight * residual_uncertainty**2)
        )
    )
    return FittedPracticeNormalization(
        reference_compound=reference,
        compound_effect_s=compound_effects,
        tyre_age_effect_s_per_lap=age_effect,
        evolution_effect_s_per_unit=evolution_effect,
        uncertainty_s=uncertainty,
        provenance="empirical_shrunk",
        prior_source=normalized_prior.source,
        comparison_count=comparison_count,
        driver_count=driver_count,
        team_count=team_count,
        empirical_weight=empirical_weight,
        coefficient_provenance=coefficient_provenance,
    )


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_timedelta(values, errors="coerce").dt.total_seconds()
    return numeric.where(numeric.notna(), parsed)


def _explicit_false(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def _truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "y", "rain", "wet"}


def _is_green_status(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return bool(text) and set(text) <= {"1"}


def _stable_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _prepare_raw_laps(laps: pd.DataFrame) -> pd.DataFrame:
    prepared = laps.copy().reset_index(drop=True)
    prepared["_source_order"] = np.arange(len(prepared), dtype=int)
    prepared["_lap_s"] = _seconds(prepared["LapTime"])
    prepared["_compound"] = prepared["Compound"].map(_normalize_compound)
    if "TyreLife" in prepared.columns:
        prepared["_tyre_age"] = pd.to_numeric(prepared["TyreLife"], errors="coerce")
    else:
        prepared["_tyre_age"] = np.nan

    if "LapNumber" in prepared.columns:
        lap_order = pd.to_numeric(prepared["LapNumber"], errors="coerce")
    else:
        lap_order = pd.Series(np.nan, index=prepared.index)
    prepared["_lap_order"] = lap_order.fillna(prepared["_source_order"])
    ordered = prepared.sort_values(["Driver", "_lap_order", "_source_order"])
    prepared.loc[ordered.index, "_driver_sequence"] = ordered.groupby(
        "Driver", dropna=False
    ).cumcount()

    base_run = pd.Series(index=prepared.index, dtype=object)
    for driver, driver_laps in prepared.groupby("Driver", dropna=False):
        del driver
        derived_run = 0
        previous_compound: str | None = None
        previous_pit_in = False
        for index, row in driver_laps.sort_values(["_lap_order", "_source_order"]).iterrows():
            stint = row.get("Stint", pd.NA)
            if pd.notna(stint):
                base_run.at[index] = f"stint:{_stable_label(stint)}"
            else:
                compound = row["_compound"]
                current_pit_out = "PitOutTime" in prepared and pd.notna(row.get("PitOutTime"))
                if (
                    previous_compound is not None
                    and compound != previous_compound
                    or previous_pit_in
                    or current_pit_out
                ):
                    derived_run += 1
                base_run.at[index] = f"derived:{derived_run}"
                previous_compound = compound
                previous_pit_in = "PitInTime" in prepared and pd.notna(row.get("PitInTime"))
    prepared["_base_run"] = base_run
    return prepared


def _filter_clean_dry_laps(
    prepared: pd.DataFrame,
    *,
    implausible_lap_ratio: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    reasons = pd.Series(pd.NA, index=prepared.index, dtype="object")

    def exclude(mask: pd.Series, reason: str) -> None:
        selected = mask.fillna(False) & reasons.isna()
        reasons.loc[selected] = reason

    exclude(prepared["Driver"].isna() | prepared["Team"].isna(), "missing_identity")
    exclude(prepared["_lap_s"].isna() | prepared["_lap_s"].le(0.0), "missing_lap_time")
    exclude(prepared["_compound"].isna(), "missing_compound")
    exclude(prepared["_compound"].isin(WET_COMPOUNDS), "wet_compound")
    exclude(~prepared["_compound"].isin(DRY_COMPOUNDS), "unsupported_compound")

    if "IsAccurate" in prepared.columns:
        exclude(prepared["IsAccurate"].map(_explicit_false), "inaccurate")
    if "Deleted" in prepared.columns:
        exclude(prepared["Deleted"].map(_truthy), "deleted")
    for column in ("Aborted", "IsAborted"):
        if column in prepared.columns:
            exclude(prepared[column].map(_truthy), "aborted")
    pit_mask = pd.Series(False, index=prepared.index)
    for column in ("PitInTime", "PitOutTime"):
        if column in prepared.columns:
            pit_mask |= prepared[column].notna()
    exclude(pit_mask, "pit_lap")
    if "TrackStatus" in prepared.columns:
        exclude(~prepared["TrackStatus"].map(_is_green_status), "non_green")
    if "Rainfall" in prepared.columns:
        exclude(prepared["Rainfall"].map(_truthy), "wet_lap")

    provisional = prepared[reasons.isna()].copy()
    ratio = max(1.0, _finite_float(implausible_lap_ratio, default=1.20))
    if not provisional.empty:
        driver_best = provisional.groupby("Driver", dropna=False)["_lap_s"].transform("min")
        implausible = provisional["_lap_s"] > (driver_best * ratio)
        exclude(implausible.reindex(prepared.index, fill_value=False), "implausible_or_aborted")

    clean = prepared[reasons.isna()].copy()
    counts = {
        str(reason): int(count)
        for reason, count in reasons.dropna().value_counts().sort_index().items()
    }
    return clean, counts


def _elapsed_seconds(values: pd.Series) -> pd.Series:
    return _seconds(values)


def _attach_track_evolution(clean: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    evolved = clean.copy()
    source = "row_order"
    progress = pd.Series(np.nan, index=evolved.index, dtype=float)
    for column in ("SessionTime", "Time", "LapStartTime"):
        if column not in evolved.columns:
            continue
        candidate = _elapsed_seconds(evolved[column])
        if candidate.notna().sum() >= 2 and float(candidate.max() - candidate.min()) > 0.0:
            progress = candidate
            source = column
            break
    if progress.notna().sum() < 2 or float(progress.max() - progress.min()) <= 0.0:
        if "LapNumber" in evolved.columns:
            candidate = pd.to_numeric(evolved["LapNumber"], errors="coerce")
            if candidate.notna().sum() >= 2 and float(candidate.max() - candidate.min()) > 0.0:
                progress = candidate
                source = "LapNumber"
    if progress.notna().sum() < 2 or float(progress.max() - progress.min()) <= 0.0:
        progress = evolved["_source_order"].astype(float)

    minimum = float(progress.min()) if progress.notna().any() else 0.0
    spread = float(progress.max() - minimum) if progress.notna().any() else 0.0
    evolved["_evolution"] = (progress - minimum) / spread if spread > 0.0 else 0.5
    evolved["_evolution"] = evolved["_evolution"].fillna(0.5).clip(0.0, 1.0)
    return evolved, source


def classify_practice_runs(
    clean_laps: pd.DataFrame,
    *,
    quick_lap_ratio: float = 1.05,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Classify consecutive clean run segments as qualifying, race, or other."""
    if clean_laps.empty:
        return clean_laps.copy(), []
    classified = clean_laps.copy()
    ratio = max(1.0, _finite_float(quick_lap_ratio, default=1.05))
    classified["_driver_compound_best_s"] = classified.groupby(
        ["Driver", "_compound"], dropna=False
    )["_lap_s"].transform("min")
    classified["_is_quick_lap"] = (
        classified["_lap_s"] <= classified["_driver_compound_best_s"] * ratio
    )

    segment_ids = pd.Series(index=classified.index, dtype=int)
    group_columns = ["Driver", "_base_run", "_compound"]
    for _, group in classified.groupby(group_columns, dropna=False):
        ordered = group.sort_values(["_driver_sequence", "_source_order"])
        sequence_gap = ordered["_driver_sequence"].diff().fillna(1).ne(1)
        if "LapNumber" in ordered.columns:
            lap_number = pd.to_numeric(ordered["LapNumber"], errors="coerce")
            lap_gap = lap_number.diff().fillna(1).gt(1)
            sequence_gap |= lap_gap
        segment_ids.loc[ordered.index] = sequence_gap.cumsum().astype(int) - 1
    classified["_clean_segment"] = segment_ids.astype(int)
    classified["_run_id"] = classified.apply(
        lambda row: ":".join(
            (
                str(row["Driver"]),
                str(row["_base_run"]),
                str(row["_compound"]),
                str(int(row["_clean_segment"])),
            )
        ),
        axis=1,
    )

    run_summaries: list[dict[str, Any]] = []
    for run_id, run in classified.groupby("_run_id", sort=True):
        lap_count = int(len(run))
        age_ok = run["_tyre_age"].between(1.0, QUALI_SIM_MAX_TYRE_AGE, inclusive="both")
        has_quick_fresh_lap = bool((run["_is_quick_lap"] & age_ok.fillna(False)).any())
        full_run_tyre_age_eligible = bool(age_ok.notna().all() and age_ok.fillna(False).all())
        if (
            lap_count <= QUALI_SIM_MAX_TIMED_LAPS
            and full_run_tyre_age_eligible
            and has_quick_fresh_lap
        ):
            run_class = RUN_QUALI_SIM
            evidence_weight = 1.0
        elif lap_count >= RACE_SIM_MIN_CONSECUTIVE_LAPS:
            run_class = RUN_RACE_SIM
            evidence_weight = 0.0
        else:
            run_class = RUN_OTHER
            evidence_weight = 0.25
        classified.loc[run.index, "_run_class"] = run_class
        classified.loc[run.index, "_evidence_weight"] = evidence_weight
        ages = run["_tyre_age"].dropna()
        run_summaries.append(
            {
                "run_id": str(run_id),
                "driver": str(run["Driver"].iloc[0]),
                "team": str(run["Team"].iloc[0]),
                "compound": str(run["_compound"].iloc[0]),
                "classification": run_class,
                "clean_consecutive_laps": lap_count,
                "has_quick_fresh_lap": has_quick_fresh_lap,
                "full_run_tyre_age_eligible": full_run_tyre_age_eligible,
                "min_tyre_age": float(ages.min()) if not ages.empty else None,
                "max_tyre_age": float(ages.max()) if not ages.empty else None,
                "evolution_start": float(run["_evolution"].min()),
                "evolution_end": float(run["_evolution"].max()),
                "evidence_weight": evidence_weight,
            }
        )
    return classified, run_summaries


def _theoretical_lap_seconds(
    candidate_laps: pd.DataFrame,
    normalization: FittedPracticeNormalization,
) -> float | None:
    sector_columns = ("Sector1Time", "Sector2Time", "Sector3Time")
    if any(column not in candidate_laps.columns for column in sector_columns):
        return None
    sector_laps = candidate_laps.copy()
    for column in sector_columns:
        sector_laps[f"_{column}_s"] = _seconds(sector_laps[column])
    valid_sectors = sector_laps[[f"_{column}_s" for column in sector_columns]].notna().all(axis=1)
    sector_laps = sector_laps[valid_sectors].copy()
    if sector_laps.empty:
        return None

    # These buckets prevent a theoretical lap from combining sectors across stints,
    # compounds, materially different tyre ages, or distant evolution windows.
    sector_laps["_age_bucket"] = (
        (sector_laps["_tyre_age"].fillna(1.0) - 1.0).clip(lower=0.0) // 2.0
    ).astype(int)
    sector_laps["_evolution_bucket"] = np.minimum(
        4,
        np.floor(sector_laps["_evolution"].clip(0.0, 1.0) / 0.20).astype(int),
    )
    theoretical_values: list[float] = []
    bucket_columns = ["_run_id", "_compound", "_age_bucket", "_evolution_bucket"]
    for _, bucket in sector_laps.groupby(bucket_columns, dropna=False):
        raw_theoretical = sum(float(bucket[f"_{column}_s"].min()) for column in sector_columns)
        correction = normalization.adjustment_seconds(
            compound=bucket["_compound"].iloc[0],
            tyre_age=float(bucket["_tyre_age"].median())
            if bucket["_tyre_age"].notna().any()
            else 1.0,
            evolution=float(bucket["_evolution"].median()),
        )
        theoretical_values.append(raw_theoretical - correction)
    return min(theoretical_values) if theoretical_values else None


def _driver_features(
    driver_laps: pd.DataFrame,
    normalization: FittedPracticeNormalization,
) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    qualifying_laps = driver_laps[driver_laps["_run_class"].eq(RUN_QUALI_SIM)]
    source_class = RUN_QUALI_SIM
    if qualifying_laps.empty:
        qualifying_laps = driver_laps[driver_laps["_run_class"].eq(RUN_OTHER)]
        source_class = RUN_OTHER
        flags.append("no_quali_sim_using_other_run_fallback")
    if qualifying_laps.empty:
        flags.append("no_qualifying_candidate_laps")
        return {
            "source_run_class": None,
            "best_adjusted_lap_s": None,
            "best_two_mean_adjusted_lap_s": None,
            "q20_adjusted_lap_s": None,
            "theoretical_adjusted_lap_s": None,
            "execution_loss_s": None,
            "mad_s": None,
            "measurement_uncertainty_s": None,
            "effective_lap_count": 0.0,
        }, flags

    candidates = qualifying_laps.copy()
    candidates["_adjusted_lap_s"] = candidates.apply(
        lambda row: normalization.normalize_lap_seconds(
            row["_lap_s"],
            compound=row["_compound"],
            tyre_age=row["_tyre_age"],
            evolution=row["_evolution"],
        ),
        axis=1,
    )
    adjusted = candidates["_adjusted_lap_s"].dropna().sort_values()
    best = float(adjusted.iloc[0])
    best_two = float(adjusted.iloc[: min(2, len(adjusted))].mean())
    q20 = float(adjusted.quantile(0.20))
    median = float(adjusted.median())
    mad = float((adjusted - median).abs().median())
    theoretical = _theoretical_lap_seconds(candidates, normalization)
    execution_loss = max(0.0, best - theoretical) if theoretical is not None else None
    evidence_weight = 1.0 if source_class == RUN_QUALI_SIM else 0.25
    effective_laps = float(len(adjusted) * evidence_weight)
    robust_sigma = 1.4826 * mad
    uncertainty = float(
        np.sqrt(
            (robust_sigma**2 / max(effective_laps, 1.0))
            + (normalization.uncertainty_s**2 / max(effective_laps, 0.25))
        )
    )

    if len(adjusted) == 1:
        flags.append("single_lap_evidence")
    if theoretical is None:
        flags.append("theoretical_lap_unavailable")
    if candidates["_tyre_age"].isna().any():
        flags.append("missing_tyre_age_assumed_reference")
    known_compounds = set(normalization.compound_effect_s).union({normalization.reference_compound})
    if not set(candidates["_compound"].dropna()).issubset(known_compounds):
        flags.append("compound_adjustment_unseen_using_zero")
    if normalization.provenance != "empirical_shrunk":
        flags.append("normalization_prior_fallback")

    return {
        "source_run_class": source_class,
        "best_adjusted_lap_s": best,
        "best_two_mean_adjusted_lap_s": best_two,
        "q20_adjusted_lap_s": q20,
        "theoretical_adjusted_lap_s": theoretical,
        "execution_loss_s": execution_loss,
        "mad_s": mad,
        "measurement_uncertainty_s": uncertainty,
        "effective_lap_count": effective_laps,
    }, flags


def _driver_run_feature_candidates(
    driver_laps: pd.DataFrame,
    normalization: FittedPracticeNormalization,
) -> list[dict[str, Any]]:
    """Summarize each qualifying candidate run without crossing run boundaries.

    These rows are research-sidecar evidence for the Q1 bootstrap.  Every pace,
    consistency, and theoretical-lap value is calculated from one already-classified
    clean run.  Race simulations are excluded, and ``other`` runs are used only when
    the driver has no qualifying-simulation run, matching the aggregate fallback.
    """

    candidate_laps = driver_laps[driver_laps["_run_class"].eq(RUN_QUALI_SIM)]
    if candidate_laps.empty:
        candidate_laps = driver_laps[driver_laps["_run_class"].eq(RUN_OTHER)]
    if candidate_laps.empty:
        return []

    candidates: list[dict[str, Any]] = []
    for run_id, run in candidate_laps.groupby("_run_id", sort=True):
        run = run.sort_values(["_driver_sequence", "_source_order"]).copy()
        run["_adjusted_lap_s"] = run.apply(
            lambda row: normalization.normalize_lap_seconds(
                row["_lap_s"],
                compound=row["_compound"],
                tyre_age=row["_tyre_age"],
                evolution=row["_evolution"],
            ),
            axis=1,
        )
        adjusted = run["_adjusted_lap_s"].dropna().sort_values()
        if adjusted.empty:
            continue

        best = float(adjusted.iloc[0])
        best_two = float(adjusted.iloc[: min(2, len(adjusted))].mean())
        q20 = float(adjusted.quantile(0.20))
        median = float(adjusted.median())
        mad = float((adjusted - median).abs().median())
        theoretical = _theoretical_lap_seconds(run, normalization)
        execution_loss = max(0.0, best - theoretical) if theoretical is not None else None
        run_class = str(run["_run_class"].iloc[0])
        evidence_weight = 1.0 if run_class == RUN_QUALI_SIM else 0.25
        effective_laps = float(len(adjusted) * evidence_weight)
        robust_sigma = 1.4826 * mad
        uncertainty = float(
            np.sqrt(
                (robust_sigma**2 / max(effective_laps, 1.0))
                + (normalization.uncertainty_s**2 / max(effective_laps, 0.25))
            )
        )
        ages = run["_tyre_age"].dropna()
        candidates.append(
            {
                "run_id": str(run_id),
                "run_class": run_class,
                "compound": str(run["_compound"].iloc[0]),
                "clean_lap_count": int(len(adjusted)),
                "effective_lap_count": effective_laps,
                "evidence_weight": evidence_weight,
                "min_tyre_age": float(ages.min()) if not ages.empty else None,
                "max_tyre_age": float(ages.max()) if not ages.empty else None,
                "evolution_start": float(run["_evolution"].min()),
                "evolution_end": float(run["_evolution"].max()),
                "best_adjusted_lap_s": best,
                "best_two_mean_adjusted_lap_s": best_two,
                "q20_adjusted_lap_s": q20,
                "theoretical_adjusted_lap_s": theoretical,
                "execution_loss_s": execution_loss,
                "mad_s": mad,
                "measurement_uncertainty_s": uncertainty,
            }
        )
    return candidates


def _attach_feature_quantiles(driver_payloads: dict[str, dict[str, Any]]) -> None:
    for feature_name in _FEATURE_QUANTILE_FIELDS:
        values = {
            driver: payload["features"].get(feature_name)
            for driver, payload in driver_payloads.items()
            if payload["features"].get(feature_name) is not None
        }
        if not values:
            continue
        series = pd.Series(values, dtype=float)
        ranks = series.rank(method="average", ascending=True)
        quantiles = (
            pd.Series(0.5, index=series.index)
            if len(series) == 1
            else 1.0 - ((ranks - 1.0) / (len(series) - 1.0))
        )
        for driver, quantile in quantiles.items():
            driver_payloads[str(driver)]["feature_quantiles"][feature_name] = float(quantile)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _ineligible_payload(
    *,
    session_code: str,
    track_name: str | None,
    track_class: str | None,
    input_laps: int,
    reason: str,
    normalization: FittedPracticeNormalization,
) -> dict[str, Any]:
    return {
        "artifact_type": QUALIFYING_PRACTICE_EVIDENCE_SCHEMA,
        "schema_version": QUALIFYING_PRACTICE_EVIDENCE_VERSION,
        "session": {
            "session_code": str(session_code),
            "track_name": track_name,
            "track_class": track_class,
            "weather_scope": "dry_only",
        },
        "eligibility": {"eligible": False, "fallback_reasons": [reason]},
        "normalization": normalization.to_dict(),
        "exclusions": {
            "input_laps": int(input_laps),
            "accepted_clean_laps": 0,
            "by_reason": {reason: int(input_laps)},
        },
        "runs": [],
        "drivers": {},
    }


def build_qualifying_practice_evidence(
    laps: pd.DataFrame,
    *,
    session_code: str,
    session_is_dry: bool | None,
    track_name: str | None = None,
    track_class: str | None = None,
    normalization: FittedPracticeNormalization | None = None,
    normalization_comparisons: pd.DataFrame | None = None,
    normalization_prior: PracticeNormalizationPrior | None = None,
    quick_lap_ratio: float = 1.05,
    implausible_lap_ratio: float = 1.20,
) -> dict[str, Any]:
    """Return a deterministic, versioned qualifying-practice evidence sidecar.

    ``session_is_dry`` must be explicitly true.  Unknown, wet, or mixed sessions fail
    closed instead of silently contaminating a dry qualifying model.
    """
    fitted = normalization or fit_practice_normalization(
        normalization_comparisons,
        prior=normalization_prior,
    )
    input_laps = int(len(laps)) if isinstance(laps, pd.DataFrame) else 0
    dry_is_confirmed = isinstance(session_is_dry, (bool, np.bool_)) and bool(session_is_dry)
    if not dry_is_confirmed:
        return _json_safe(
            _ineligible_payload(
                session_code=session_code,
                track_name=track_name,
                track_class=track_class,
                input_laps=input_laps,
                reason="dry_session_not_confirmed",
                normalization=fitted,
            )
        )
    if not isinstance(laps, pd.DataFrame) or laps.empty:
        return _json_safe(
            _ineligible_payload(
                session_code=session_code,
                track_name=track_name,
                track_class=track_class,
                input_laps=input_laps,
                reason="no_laps",
                normalization=fitted,
            )
        )

    required_columns = {"Driver", "Team", "LapTime", "Compound"}
    missing = sorted(required_columns.difference(laps.columns))
    if missing:
        return _json_safe(
            _ineligible_payload(
                session_code=session_code,
                track_name=track_name,
                track_class=track_class,
                input_laps=input_laps,
                reason=f"missing_lap_columns:{','.join(missing)}",
                normalization=fitted,
            )
        )

    prepared = _prepare_raw_laps(laps)
    clean, exclusion_counts = _filter_clean_dry_laps(
        prepared,
        implausible_lap_ratio=implausible_lap_ratio,
    )
    if clean.empty:
        payload = _ineligible_payload(
            session_code=session_code,
            track_name=track_name,
            track_class=track_class,
            input_laps=input_laps,
            reason="no_clean_dry_laps",
            normalization=fitted,
        )
        payload["exclusions"]["by_reason"] = exclusion_counts
        return _json_safe(payload)

    clean, evolution_source = _attach_track_evolution(clean)
    classified, run_summaries = classify_practice_runs(
        clean,
        quick_lap_ratio=quick_lap_ratio,
    )
    missing_quality_columns = sorted(
        column for column in _QUALITY_FILTER_COLUMNS if column not in laps.columns
    )
    session_quality_flags = [
        f"filter_field_unavailable:{column}" for column in missing_quality_columns
    ]
    if "Stint" not in laps.columns or not bool(laps["Stint"].notna().any()):
        session_quality_flags.append("stint_ids_unavailable_derived_runs")
    if evolution_source == "row_order":
        session_quality_flags.append("track_evolution_uses_row_order")

    driver_payloads: dict[str, dict[str, Any]] = {}
    for driver, driver_laps in classified.groupby("Driver", sort=True):
        driver_code = str(driver)
        features, flags = _driver_features(driver_laps, fitted)
        run_feature_candidates = _driver_run_feature_candidates(driver_laps, fitted)
        run_counts = {
            run_class: int(
                driver_laps["_run_class"].eq(run_class).groupby(driver_laps["_run_id"]).any().sum()
            )
            for run_class in (RUN_QUALI_SIM, RUN_RACE_SIM, RUN_OTHER)
        }
        driver_payloads[driver_code] = {
            "team": str(driver_laps["Team"].iloc[0]),
            "features": features,
            "run_feature_candidates": run_feature_candidates,
            "feature_quantiles": {},
            "counts": {
                "clean_laps": int(len(driver_laps)),
                "qualifying_candidate_laps": int(
                    driver_laps["_run_class"].isin({RUN_QUALI_SIM, RUN_OTHER}).sum()
                ),
                "runs": run_counts,
                "compounds": sorted(str(value) for value in driver_laps["_compound"].unique()),
            },
            "quality_flags": sorted(set(flags).union(session_quality_flags)),
        }
    _attach_feature_quantiles(driver_payloads)

    eligible_drivers = sum(
        payload["features"]["best_adjusted_lap_s"] is not None
        for payload in driver_payloads.values()
    )
    fallback_reasons: list[str] = []
    if eligible_drivers == 0:
        fallback_reasons.append("no_qualifying_candidate_laps")
    if fitted.provenance != "empirical_shrunk":
        fallback_reasons.append("normalization_uses_track_class_prior")

    payload = {
        "artifact_type": QUALIFYING_PRACTICE_EVIDENCE_SCHEMA,
        "schema_version": QUALIFYING_PRACTICE_EVIDENCE_VERSION,
        "session": {
            "session_code": str(session_code),
            "track_name": track_name,
            "track_class": track_class,
            "weather_scope": "dry_only",
            "track_evolution_source": evolution_source,
        },
        "eligibility": {
            "eligible": eligible_drivers > 0,
            "eligible_drivers": int(eligible_drivers),
            "fallback_reasons": fallback_reasons,
        },
        "normalization": fitted.to_dict(),
        "quality": {
            "missing_optional_lap_columns": missing_quality_columns,
            "flags": sorted(session_quality_flags),
        },
        "parameters": {
            "quali_sim_max_timed_laps": QUALI_SIM_MAX_TIMED_LAPS,
            "quali_sim_max_tyre_age": QUALI_SIM_MAX_TYRE_AGE,
            "race_sim_min_consecutive_laps": RACE_SIM_MIN_CONSECUTIVE_LAPS,
            "quick_lap_ratio": max(1.0, _finite_float(quick_lap_ratio, default=1.05)),
            "implausible_lap_ratio": max(1.0, _finite_float(implausible_lap_ratio, default=1.20)),
            "run_feature_candidate_scope": "single_compatible_run",
        },
        "exclusions": {
            "input_laps": input_laps,
            "accepted_clean_laps": int(len(classified)),
            "by_reason": exclusion_counts,
        },
        "runs": run_summaries,
        "drivers": driver_payloads,
    }
    return _json_safe(payload)
