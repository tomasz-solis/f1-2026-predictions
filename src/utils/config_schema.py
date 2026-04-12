"""Pydantic schemas for repository configuration.

The config file has grown into a large part of the model surface area. These
schemas keep the defaults in one place, validate the shipped YAML, and make
drift visible when a new key appears without a matching typed home.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Base model that rejects undeclared configuration keys."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictConfigModel):
    """File paths configuration."""

    data_dir: str = Field(default="data", min_length=1)
    processed: str = Field(default="data/processed", min_length=1)
    raw: str = Field(default="data/raw", min_length=1)
    driver_chars: str = Field(default="data/processed/driver_characteristics.json", min_length=1)
    track_chars: str = Field(default="data/processed/track_characteristics.json", min_length=1)
    lineups: str = Field(default="data/current_lineups.json", min_length=1)
    cache: str = Field(default=".fastf1_cache", min_length=1)


class GridConfig(StrictConfigModel):
    """Race-weekend field size assumptions."""

    size: int = Field(default=22, ge=2)


class BayesianConfig(StrictConfigModel):
    """Bayesian model parameters."""

    base_volatility: float = Field(default=0.1, ge=0.0, le=1.0)
    base_observation_noise: float = Field(default=2.0, ge=0.0)
    shock_threshold: float = Field(default=2.0, ge=0.0)
    shock_multiplier: float = Field(default=0.5, ge=0.0, le=2.0)
    teammate_relative_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    qualifying_update_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    sprint_race_confidence: float = Field(default=0.20, ge=0.0, le=1.0)


class RaceWeightsConfig(StrictConfigModel):
    """Top-level race weighting inputs."""

    pace_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    grid_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    overtaking_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    tire_deg_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class UncertaintyMultipliersConfig(StrictConfigModel):
    """Named multipliers applied to race uncertainty."""

    rain: float = Field(default=1.5, ge=0.0)
    easy_overtaking: float = Field(default=0.8, ge=0.0)


class DNFConfig(StrictConfigModel):
    """DNF risk parameters."""

    base_risk: float = Field(default=0.05, ge=0.0, le=1.0)
    driver_error_factor: float = Field(default=0.15, ge=0.0, le=1.0)
    street_circuit_risk: float = Field(default=0.05, ge=0.0, le=0.5)
    rain_risk: float = Field(default=0.10, ge=0.0, le=0.5)


class Lap1Config(StrictConfigModel):
    """Lap-one variance parameters."""

    midfield_variance: float = Field(default=1.5, ge=0.0)
    front_row_variance: float = Field(default=0.0, ge=0.0)


class TireConfig(StrictConfigModel):
    """Top-level tire degradation parameters."""

    degradation_multiplier: float = Field(default=4.0, ge=0.0)
    skill_reduction_factor: float = Field(default=0.2, ge=0.0, le=1.0)


class WeatherConfig(StrictConfigModel):
    """Weather impact parameters."""

    rain_position_swing: float = Field(default=6.0, ge=0.0)
    mixed_intensity: float = Field(default=0.5, ge=0.0, le=1.0)


class SafetyCarConfig(StrictConfigModel):
    """Safety car parameters."""

    compression_factor: float = Field(default=0.1, ge=0.0, le=1.0)


class PaceConfig(StrictConfigModel):
    """Top-level pace calculation parameters."""

    pace_delta_multiplier: float = Field(default=3.0, ge=0.0)


class RaceConfig(StrictConfigModel):
    """Top-level race simulation parameters."""

    weights: RaceWeightsConfig = Field(default_factory=RaceWeightsConfig)
    base_uncertainty: float = Field(default=2.5, ge=0.0)
    uncertainty_multipliers: UncertaintyMultipliersConfig = Field(
        default_factory=UncertaintyMultipliersConfig
    )
    dnf: DNFConfig = Field(default_factory=DNFConfig)
    lap1: Lap1Config = Field(default_factory=Lap1Config)
    tire: TireConfig = Field(default_factory=TireConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    safety_car: SafetyCarConfig = Field(default_factory=SafetyCarConfig)
    pace: PaceConfig = Field(default_factory=PaceConfig)
    dnf_position_penalty: int = Field(default=22, ge=1)


class BlendConfig(StrictConfigModel):
    """Top-level qualifying blend weights."""

    default: float = Field(default=0.7, ge=0.0, le=1.0)
    fp3_only: float = Field(default=0.8, ge=0.0, le=1.0)
    fp1_only: float = Field(default=0.4, ge=0.0, le=1.0)


class SessionConfidenceConfig(StrictConfigModel):
    """Session confidence weights."""

    fp1: float = Field(default=0.2, ge=0.0, le=1.0)
    fp2: float = Field(default=0.5, ge=0.0, le=1.0)
    fp3: float = Field(default=0.9, ge=0.0, le=1.0)
    sprint_quali: float = Field(default=0.85, ge=0.0, le=1.0)


class QualifyingConfig(StrictConfigModel):
    """Top-level qualifying prediction parameters."""

    blend: BlendConfig = Field(default_factory=BlendConfig)
    session_confidence: SessionConfidenceConfig = Field(default_factory=SessionConfidenceConfig)
    base_uncertainty: float = Field(default=1.5, ge=0.0)


class LearningConfig(StrictConfigModel):
    """Top-level learning parameters."""

    performance_window: int = Field(default=5, ge=1)
    min_samples: int = Field(default=3, ge=1)
    min_races_for_blend: int | None = Field(default=None, ge=1)
    driver_error_scale: float = Field(default=0.18, ge=0.0)
    teammate_gap_scale: float = Field(default=0.10, ge=0.0)
    max_adjustment: float = Field(default=2.5, ge=0.0)


class TrackDefaultsConfig(StrictConfigModel):
    """Fallback track characteristics."""

    pit_stop_loss: float = Field(default=22.0, ge=0.0)
    safety_car_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    overtaking_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)


class LoggingConfig(StrictConfigModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s | %(levelname)s | %(message)s")


class DashboardPredictionPrecomputeConfig(StrictConfigModel):
    """Dashboard precompute settings."""

    enabled: bool = True
    horizon_races: int = Field(default=3, ge=1)
    weather_scenarios: list[str] = Field(default_factory=lambda: ["dry", "mixed", "rain"])
    max_file_entries: int = Field(default=2048, ge=1)
    qualifying_n_simulations: int = Field(default=300, ge=1)
    race_n_simulations: int = Field(default=300, ge=1)


class DashboardConfig(StrictConfigModel):
    """Dashboard runtime settings."""

    prediction_precompute: DashboardPredictionPrecomputeConfig = Field(
        default_factory=DashboardPredictionPrecomputeConfig
    )


class CurrentSeasonFormConfig(StrictConfigModel):
    """How fast stored actuals can reshape team form."""

    infer_from_saved_actuals: bool = True
    recency_exponent: float = Field(default=1.5, ge=0.0)
    stabilization_strength: float = Field(default=1.5, ge=0.0)
    saved_actual_race_weight: float = Field(default=0.70, ge=0.0, le=1.0)


class DriverFormConfig(StrictConfigModel):
    """How in-season driver form feeds the prediction inputs."""

    bayesian_pace_blend_per_race: float = Field(default=0.20, ge=0.0, le=1.0)
    bayesian_pace_blend_cap: float = Field(default=0.60, ge=0.0, le=1.0)
    bayesian_race_skill_blend_per_race: float = Field(default=0.20, ge=0.0, le=1.0)
    bayesian_race_skill_blend_cap: float = Field(default=0.60, ge=0.0, le=1.0)
    quali_pace_update_blend: float = Field(default=0.30, ge=0.0, le=1.0)
    race_pace_update_blend: float = Field(default=0.25, ge=0.0, le=1.0)
    wet_skill_update_blend: float = Field(default=0.15, ge=0.0, le=0.5)
    wet_skill_observation_scale: float = Field(default=0.40, ge=0.1, le=1.0)


class ExperienceFloatMapConfig(StrictConfigModel):
    """Experience-tier float map used for shrinkage and caps."""

    rookie: float = Field(default=0.0, ge=0.0)
    second_year: float = Field(default=0.0, ge=0.0)
    developing: float = Field(default=0.0, ge=0.0)
    unknown: float = Field(default=0.0, ge=0.0)
    sunset: float | None = Field(default=None, ge=0.0)


class ExperienceIntMapConfig(StrictConfigModel):
    """Experience-tier integer map used for race-count cutoffs."""

    rookie: int = Field(default=0, ge=0)
    second_year: int = Field(default=0, ge=0)
    developing: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    sunset: int | None = Field(default=None, ge=0)


class BaselineQualifyingDataConfidenceConfig(StrictConfigModel):
    """Fallback qualifying confidence by data source."""

    model_only: float = Field(default=0.25, ge=0.0, le=1.0)
    testing_fallback: float = Field(default=0.45, ge=0.0, le=1.0)
    sprint_race: float = Field(default=0.70, ge=0.0, le=1.0)


class BaselineQualifyingConfig(StrictConfigModel):
    """Detailed qualifying config used by the baseline predictor."""

    noise_std_sprint: float = Field(default=0.030, ge=0.0)
    noise_std_normal: float = Field(default=0.026, ge=0.0)
    wet_noise_multiplier: float = Field(default=1.60, ge=1.0)
    mixed_noise_multiplier: float = Field(default=1.28, ge=1.0)
    wet_skill_weight: float = Field(default=0.18, ge=0.0, le=1.0)
    sprint_wet_skill_scale: float = Field(default=0.75, ge=0.0, le=1.0)
    wet_skill_neutral: float = Field(default=0.70, ge=0.0, le=1.0)
    team_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    skill_weight: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_weight_sum(self) -> BaselineQualifyingConfig:
        total = self.team_weight + self.skill_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"team_weight ({self.team_weight}) + skill_weight ({self.skill_weight}) "
                f"must sum to 1.0 (got {total})"
            )
        return self

    team_strength_compression: float = Field(default=0.60, ge=0.0)
    driver_quali_pace_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    driver_skill_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    teammate_setup_std: float = Field(default=0.018, ge=0.0)
    driver_offset_cap: float = Field(default=0.22, ge=0.0)
    driver_signal_softness: float = Field(default=0.35, ge=0.0)
    weekend_form_std: float = Field(default=0.0, ge=0.0)
    recent_form_scale: float = Field(default=0.12, ge=0.0)
    recent_form_cap: float = Field(default=0.03, ge=0.0)
    model_only_team_weight_multiplier: float = Field(default=0.90, ge=0.0)
    model_only_skill_weight_multiplier: float = Field(default=1.10, ge=0.0)
    model_only_team_compression_multiplier: float = Field(default=0.87, ge=0.0)
    model_only_driver_offset_cap_multiplier: float = Field(default=1.10, ge=0.0)
    model_only_driver_signal_shrink: float = Field(default=0.35, ge=0.0)
    model_only_experience_shrink: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.45,
            second_year=0.30,
            developing=0.20,
            sunset=0.05,
            unknown=0.30,
        )
    )
    model_only_noise_multiplier: float = Field(default=1.12, ge=0.0)
    model_only_teammate_setup_multiplier: float = Field(default=1.10, ge=0.0)
    model_only_weekend_form_multiplier: float = Field(default=1.0, ge=0.0)
    model_only_teammate_anchor_experience_multiplier: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.30,
            second_year=0.45,
            developing=0.55,
            sunset=1.00,
            unknown=0.45,
        )
    )
    model_only_teammate_gap_cap_by_experience: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.14,
            second_year=0.10,
            developing=0.08,
            unknown=0.10,
        )
    )
    model_only_teammate_gap_cap_max_races_by_experience: ExperienceIntMapConfig = Field(
        default_factory=lambda: ExperienceIntMapConfig(
            rookie=40,
            second_year=55,
            developing=55,
            unknown=45,
        )
    )
    model_only_teammate_gap_cap_min_scale: float = Field(default=0.20, ge=0.0)
    model_only_negative_delta_threshold: float = Field(default=0.08, ge=0.0)
    model_only_negative_delta_shrink_scale: float = Field(default=1.0, ge=0.0)
    model_only_negative_delta_shrink_cap: float = Field(default=0.25, ge=0.0)
    testing_short_run_modifier_scale: float = Field(default=0.04, ge=0.0)
    preferred_short_run_compound: str = Field(default="SOFT")
    testing_fallback_min_teams: int = Field(default=8, ge=2)
    testing_fallback_absolute_blend_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    testing_fallback_modifier_scale: float = Field(default=0.03, ge=0.0)
    testing_fallback_modifier_clip_range: list[float] = Field(
        default_factory=lambda: [-0.015, 0.015]
    )
    testing_fallback_checkpoint_blend_weight_scale: float = Field(default=0.65, ge=0.0)
    testing_fallback_checkpoint_blend_weight_cap: float = Field(default=0.55, ge=0.0)
    testing_fallback_checkpoint_modifier_scale: float = Field(default=0.10, ge=0.0)
    testing_fallback_checkpoint_modifier_cap: float = Field(default=0.08, ge=0.0)
    testing_fallback_checkpoint_clip_scale: float = Field(default=0.055, ge=0.0)
    testing_fallback_checkpoint_clip_cap: float = Field(default=0.045, ge=0.0)
    stored_checkpoint_blend_weight_multiplier: float = Field(default=1.12, ge=0.0)
    stored_checkpoint_blend_weight_cap: float = Field(default=0.90, ge=0.0)
    checkpoint_driver_profile_smoothing_seconds: float = Field(default=0.35, ge=0.0)
    checkpoint_driver_profile_quali_scale: float = Field(default=0.10, ge=0.0)
    checkpoint_driver_profile_skill_scale: float = Field(default=0.02, ge=0.0)
    testing_fallback_short_weight_min: float = Field(default=0.35, ge=0.0, le=1.0)
    testing_fallback_short_weight_max: float = Field(default=0.85, ge=0.0, le=1.0)
    testing_fallback_divergence_scale: float = Field(default=1.4, ge=0.0)
    testing_fallback_after_sprint_main_short_weight: float = Field(default=1.0, ge=0.0)
    testing_fallback_teammate_guard_enabled: bool = True
    testing_fallback_driver_signal_shrink: float = Field(default=0.02, ge=0.0)
    testing_fallback_driver_offset_cap_multiplier: float = Field(default=1.33, ge=0.0)
    testing_fallback_team_weight_multiplier: float = Field(default=0.92, ge=0.0)
    testing_fallback_skill_weight_multiplier: float = Field(default=1.08, ge=0.0)
    testing_fallback_experience_shrink: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.22,
            second_year=0.14,
            developing=0.09,
            sunset=0.03,
            unknown=0.14,
        )
    )
    testing_fallback_teammate_anchor_scale: float = Field(default=0.05, ge=0.0)
    testing_fallback_teammate_anchor_cap: float = Field(default=0.02, ge=0.0)
    testing_fallback_teammate_anchor_experience_multiplier: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.55,
            second_year=0.70,
            developing=0.80,
            sunset=1.00,
            unknown=0.70,
        )
    )
    testing_fallback_teammate_setup_multiplier: float = Field(default=1.20, ge=0.0)
    testing_fallback_teammate_gap_cap_by_experience: ExperienceFloatMapConfig = Field(
        default_factory=lambda: ExperienceFloatMapConfig(
            rookie=0.18,
            second_year=0.14,
            developing=0.12,
            unknown=0.14,
        )
    )
    testing_fallback_teammate_gap_cap_max_races_by_experience: ExperienceIntMapConfig = Field(
        default_factory=lambda: ExperienceIntMapConfig(
            rookie=55,
            second_year=80,
            developing=90,
            unknown=80,
        )
    )
    testing_fallback_teammate_gap_cap_min_scale: float = Field(default=0.30, ge=0.0)
    testing_fallback_negative_delta_threshold: float = Field(default=0.08, ge=0.0)
    testing_fallback_negative_delta_shrink_scale: float = Field(default=1.0, ge=0.0)
    testing_fallback_negative_delta_shrink_cap: float = Field(default=0.18, ge=0.0)
    testing_fallback_noise_multiplier: float = Field(default=1.30, ge=0.0)
    testing_fallback_weekend_form_std_floor: float = Field(default=0.008, ge=0.0)
    fp_blend_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    fp_blend_weight_min: float = Field(default=0.45, ge=0.0, le=1.0)
    fp_blend_weight_max: float = Field(default=0.85, ge=0.0, le=1.0)
    fp_blend_confidence_scale: float = Field(default=0.30, ge=0.0)
    fp_blend_weight_testing: bool = True
    practice_data_team_weight_multiplier: float = Field(default=0.94, ge=0.0)
    practice_data_skill_weight_multiplier: float = Field(default=1.12, ge=0.0)
    practice_data_team_compression_multiplier: float = Field(default=0.88, ge=0.0)
    practice_data_driver_offset_cap_multiplier: float = Field(default=1.33, ge=0.0)
    practice_data_teammate_setup_multiplier: float = Field(default=1.05, ge=0.0)
    data_confidence: BaselineQualifyingDataConfidenceConfig = Field(
        default_factory=BaselineQualifyingDataConfidenceConfig
    )
    session_confidence_scale: float = Field(default=10.0, ge=0.0)
    confidence_cap: int = Field(default=60, ge=1)
    confidence_min: int = Field(default=40, ge=1)
    default_skill: float = Field(default=0.5, ge=0.0, le=1.0)
    default_team_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_driver_fp_adjustment: bool = True
    driver_fp_adjustment_scale: float = Field(default=0.10, ge=0.0)
    driver_fp_adjustment_smoothing: float = Field(default=0.5, ge=0.0)


class CompoundSelectionConfig(StrictConfigModel):
    """Shared compound-selection thresholds."""

    high_stress_threshold: float = Field(default=3.5, ge=0.0)
    low_stress_threshold: float = Field(default=2.5, ge=0.0)
    default_stress_fallback: float = Field(default=3.0, ge=0.0)


class DryWetFloatConfig(StrictConfigModel):
    """Simple dry/wet mapping."""

    dry: float = Field(default=0.0, ge=0.0)
    wet: float = Field(default=0.0, ge=0.0)


class PositionScalingConfig(StrictConfigModel):
    """Position-zone scaling for racecraft gains."""

    front_threshold: int = Field(default=3, ge=1)
    front_scale: float = Field(default=0.10, ge=0.0)
    upper_threshold: int = Field(default=7, ge=1)
    upper_scale: float = Field(default=0.30, ge=0.0)
    mid_threshold: int = Field(default=12, ge=1)
    mid_scale: float = Field(default=0.60, ge=0.0)
    back_scale: float = Field(default=1.00, ge=0.0)


class Lap1ChaosConfig(StrictConfigModel):
    """Lap-one variance by part of the grid."""

    front_row: float = Field(default=0.10, ge=0.0)
    upper_midfield: float = Field(default=0.28, ge=0.0)
    midfield: float = Field(default=0.35, ge=0.0)
    back_field: float = Field(default=0.25, ge=0.0)


class SprintCheckpointCapsConfig(StrictConfigModel):
    """Main-race confidence caps on sprint weekends."""

    SQ: float = Field(default=0.50, ge=0.0, le=1.0)
    SPRINT: float = Field(default=0.65, ge=0.0, le=1.0)


class TestingProfileMetricConfig(StrictConfigModel):
    """Metric weights for one stored testing profile."""

    overall_pace: float | None = Field(default=None, ge=0.0)
    top_speed: float | None = Field(default=None, ge=0.0)
    medium_corner_performance: float | None = Field(default=None, ge=0.0)
    fast_corner_performance: float | None = Field(default=None, ge=0.0)
    tire_deg_performance: float | None = Field(default=None, ge=0.0)
    consistency: float | None = Field(default=None, ge=0.0)


class TestingProfileWeightsConfig(StrictConfigModel):
    """Stored testing-profile weightings by program type."""

    short_run: TestingProfileMetricConfig = Field(
        default_factory=lambda: TestingProfileMetricConfig(
            overall_pace=0.55,
            top_speed=0.20,
            medium_corner_performance=0.15,
            fast_corner_performance=0.10,
        )
    )
    long_run: TestingProfileMetricConfig = Field(
        default_factory=lambda: TestingProfileMetricConfig(
            overall_pace=0.50,
            tire_deg_performance=0.35,
            consistency=0.15,
        )
    )
    balanced: TestingProfileMetricConfig = Field(
        default_factory=lambda: TestingProfileMetricConfig(
            overall_pace=0.65,
            tire_deg_performance=0.20,
            top_speed=0.15,
        )
    )


class DefensiveSkillWeightsConfig(StrictConfigModel):
    """Blend used to infer defensive skill from driver traits."""

    overtaking_component: float = Field(default=0.65, ge=0.0)
    skill_component: float = Field(default=0.35, ge=0.0)


class OvertakeModelConfig(StrictConfigModel):
    """Detailed overtaking model defaults, including hidden fallbacks."""

    dirty_air_window_s: float = Field(default=1.8, ge=0.0)
    dirty_air_penalty_base: float = Field(default=0.05, ge=0.0)
    dirty_air_penalty_track_scale: float = Field(default=0.12, ge=0.0)
    pass_window_s: float = Field(default=1.2, ge=0.0)
    pass_threshold_base: float = Field(default=0.06, ge=0.0)
    pass_threshold_track_scale: float = Field(default=0.16, ge=0.0)
    pass_probability_base: float = Field(default=0.30, ge=0.0)
    pass_probability_scale: float = Field(default=0.45, ge=0.0)
    pass_time_bonus_range: list[float] = Field(default_factory=lambda: [0.08, 0.35])
    pace_diff_scale: float = Field(default=0.55, ge=0.0)
    skill_scale: float = Field(default=0.25, ge=0.0)
    defense_scale: float = Field(default=0.28, ge=0.0)
    race_adv_scale: float = Field(default=0.20, ge=0.0)
    track_ease_scale: float = Field(default=0.18, ge=0.0)
    zone_front_threshold_boost: float = Field(default=0.22)
    zone_upper_threshold_boost: float = Field(default=0.10)
    zone_mid_threshold_boost: float = Field(default=0.02)
    zone_back_threshold_boost: float = Field(default=-0.03)
    zone_front_probability_scale: float = Field(default=0.55, ge=0.0)
    zone_upper_probability_scale: float = Field(default=0.75, ge=0.0)
    zone_mid_probability_scale: float = Field(default=0.92, ge=0.0)
    zone_back_probability_scale: float = Field(default=1.08, ge=0.0)
    zone_front_bonus_scale: float = Field(default=0.55, ge=0.0)
    zone_upper_bonus_scale: float = Field(default=0.78, ge=0.0)
    zone_mid_bonus_scale: float = Field(default=0.93, ge=0.0)
    zone_back_bonus_scale: float = Field(default=1.05, ge=0.0)


class PitStopsConfig(StrictConfigModel):
    """Pit lane loss model."""

    loss_duration: float = Field(default=22.0, ge=0.0)
    overtake_loss_range: list[float] = Field(default_factory=lambda: [0.0, 3.0])


class TireStrategyWindowsConfig(StrictConfigModel):
    """Nominal stop windows for race-strategy generation."""

    one_stop: list[int] = Field(default_factory=lambda: [23, 37])
    two_stop_first: list[int] = Field(default_factory=lambda: [15, 25])
    two_stop_second: list[int] = Field(default_factory=lambda: [35, 45])


class TireStrategyStopProbabilityConfig(StrictConfigModel):
    """Stress-sensitive stop-count probabilities."""

    high_stress_2stop: float = Field(default=0.80, ge=0.0, le=1.0)
    medium_stress_1stop: float = Field(default=0.90, ge=0.0, le=1.0)
    low_stress_1stop: float = Field(default=0.95, ge=0.0, le=1.0)


class CompoundPreferenceConfig(StrictConfigModel):
    """Relative compound preferences for strategy generation."""

    SOFT: float = Field(default=1.0, ge=0.0)
    MEDIUM: float = Field(default=0.8, ge=0.0)
    HARD: float = Field(default=0.6, ge=0.0)


class TireStrategyConfig(StrictConfigModel):
    """Race strategy generator defaults."""

    windows: TireStrategyWindowsConfig = Field(default_factory=TireStrategyWindowsConfig)
    stop_probability: TireStrategyStopProbabilityConfig = Field(
        default_factory=TireStrategyStopProbabilityConfig
    )
    compound_preferences: CompoundPreferenceConfig = Field(default_factory=CompoundPreferenceConfig)


class FuelConfig(StrictConfigModel):
    """Fuel-load modeling."""

    effect_per_lap: float = Field(default=0.035, ge=0.0)
    initial_load_kg: float = Field(default=110.0, ge=0.0)
    burn_rate_kg_per_lap: float = Field(default=1.5, ge=0.0)
    deg_multiplier: float = Field(default=0.10, ge=0.0)


class SessionWeightBySessionConfig(StrictConfigModel):
    """Blend weights for observed track temperatures by session."""

    R: float = Field(default=0.90, ge=0.0, le=1.0)
    Q: float = Field(default=0.80, ge=0.0, le=1.0)
    Sprint: float = Field(default=0.80, ge=0.0, le=1.0)
    SQ: float = Field(default=0.75, ge=0.0, le=1.0)
    FP3: float = Field(default=0.70, ge=0.0, le=1.0)
    FP2: float = Field(default=0.65, ge=0.0, le=1.0)
    FP1: float = Field(default=0.60, ge=0.0, le=1.0)


class TrackTemperatureBlendConfig(StrictConfigModel):
    """How observed session weather should influence race track temperature."""

    enabled: bool = True
    session_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    session_weight_by_session: SessionWeightBySessionConfig = Field(
        default_factory=SessionWeightBySessionConfig
    )


class TrackTemperatureConfig(StrictConfigModel):
    """Track-temperature defaults for the tire model."""

    dry_c: float = Field(default=36.0)
    mixed_c: float = Field(default=29.0)
    rain_c: float = Field(default=23.0)
    min_c: float = Field(default=5.0)
    max_c: float = Field(default=65.0)
    air_to_track_offset_c: float = Field(default=9.0)
    blend: TrackTemperatureBlendConfig = Field(default_factory=TrackTemperatureBlendConfig)


class WeatherMismatchConfig(StrictConfigModel):
    """Penalty applied when forecast and observed weather disagree."""

    chaos_boost: float = Field(default=0.18, ge=0.0)
    variance_boost: float = Field(default=0.10, ge=0.0)
    confidence_penalty: float = Field(default=2.0, ge=0.0)


class WeatherFeaturesConfig(StrictConfigModel):
    """Non-competitive weather modifiers carried into race predictions."""

    mismatch: WeatherMismatchConfig = Field(default_factory=WeatherMismatchConfig)


class CompoundFloatConfig(StrictConfigModel):
    """Compound-indexed float map."""

    SOFT: float = Field(default=0.0, ge=0.0)
    MEDIUM: float = Field(default=0.0, ge=0.0)
    HARD: float = Field(default=0.0, ge=0.0)


class CompoundIntConfig(StrictConfigModel):
    """Compound-indexed integer map."""

    SOFT: int = Field(default=0, ge=0)
    MEDIUM: int = Field(default=0, ge=0)
    HARD: int = Field(default=0, ge=0)


class TirePhysicsDegradationTemperatureConfig(StrictConfigModel):
    """Temperature effect on degradation."""

    reference_c: float = Field(default=35.0)
    sensitivity_per_c: float = Field(default=0.006, ge=0.0)
    min_multiplier: float = Field(default=0.88, ge=0.0)
    max_multiplier: float = Field(default=1.18, ge=0.0)


class TirePhysicsFreshTemperatureConfig(StrictConfigModel):
    """Temperature effect on fresh-tire warm-up."""

    optimal_c: float = Field(default=30.0)
    decay_per_c: float = Field(default=0.01, ge=0.0)
    min_multiplier: float = Field(default=0.70, ge=0.0)


class TirePhysicsTemperatureConfig(StrictConfigModel):
    """Temperature subconfig for tire physics."""

    degradation: TirePhysicsDegradationTemperatureConfig = Field(
        default_factory=TirePhysicsDegradationTemperatureConfig
    )
    fresh: TirePhysicsFreshTemperatureConfig = Field(
        default_factory=TirePhysicsFreshTemperatureConfig
    )


class TirePhysicsConfig(StrictConfigModel):
    """Tire wear and warm-up model defaults."""

    fresh_tire_advantage: CompoundFloatConfig = Field(
        default_factory=lambda: CompoundFloatConfig(SOFT=0.5, MEDIUM=0.3, HARD=0.1)
    )
    fresh_tire_duration: CompoundIntConfig = Field(
        default_factory=lambda: CompoundIntConfig(SOFT=3, MEDIUM=3, HARD=2)
    )
    default_deg_slope: float = Field(default=0.15, ge=0.0)
    traffic_deg_penalty: float = Field(default=0.05, ge=0.0)
    clean_air_bonus: float = Field(default=0.05, ge=0.0)
    compound_max_age: CompoundIntConfig = Field(
        default_factory=lambda: CompoundIntConfig(SOFT=24, MEDIUM=34, HARD=42)
    )
    cliff_multiplier: float = Field(default=2.8, ge=0.0)
    temperature: TirePhysicsTemperatureConfig = Field(default_factory=TirePhysicsTemperatureConfig)


class LapTimeConfig(StrictConfigModel):
    """Lap-time conversion defaults for the simulator."""

    reference_base: float = Field(default=90.0, ge=0.0)
    team_pace_penalty_range: float = Field(default=5.0, ge=0.0)
    skill_improvement_max: float = Field(default=0.75, ge=0.0)
    team_strength_compression: float = Field(default=0.35, ge=0.0)
    elite_skill_threshold: float = Field(default=0.88, ge=0.0)
    elite_skill_lap_bonus_max: float = Field(default=0.22, ge=0.0)
    elite_skill_exponent: float = Field(default=0.85, gt=0.0, le=2.0)
    bounds: list[float] = Field(default_factory=lambda: [70.0, 120.0])
    wet_skill_lap_weight: float = Field(default=0.80, ge=0.0, le=2.0)
    wet_skill_neutral: float = Field(default=0.70, ge=0.0, le=1.0)
    track_wet_severity_base: float = Field(default=0.80, ge=0.0, le=2.0)
    track_wet_severity_scale: float = Field(default=0.40, ge=0.0, le=2.0)


class GridUncertaintyConfig(StrictConfigModel):
    """How grid uncertainty flows into the race model."""

    base_std: float = Field(default=0.25, ge=0.0)
    interval_divisor: float = Field(default=4.5, ge=0.0)
    confidence_scale: float = Field(default=0.75, ge=0.0)
    input_confidence_scale: float = Field(default=0.30, ge=0.0)
    position_delta_scale: float = Field(default=0.20, ge=0.0)


class GridAnchorConfig(StrictConfigModel):
    """Final blend between simulated race order and qualifying grid."""

    base: float = Field(default=0.28, ge=0.0)
    track_scale: float = Field(default=0.36, ge=0.0)
    min: float = Field(default=0.42, ge=0.0)
    main_max: float = Field(default=0.70, ge=0.0)
    sprint_min: float = Field(default=0.78, ge=0.0)
    low_confidence_scale: float = Field(default=0.30, ge=0.0)


class PredictedGridUncertaintyConfig(StrictConfigModel):
    """Extra damping when the starting grid is itself probabilistic."""

    activation_width: float = Field(default=2.0, ge=0.0)
    width_scale: float = Field(default=5.0, ge=0.0)
    anchor_scale: float = Field(default=0.18, ge=0.0)
    racecraft_damp_scale: float = Field(default=0.24, ge=0.0)
    max_gain_damp_scale: float = Field(default=0.30, ge=0.0)


class PositionIntervalFloorConfig(StrictConfigModel):
    """Low-confidence interval floor near the front of the field."""

    apply_below_input_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    top_n: int = Field(default=3, ge=1)
    min_width: int = Field(default=1, ge=0)
    max_extra_width: int = Field(default=1, ge=0)


class OvertakingTransitionConfig(StrictConfigModel):
    """How observed 2026 overtaking data should blend with priors."""

    min_observed_weight: float = Field(default=0.12, ge=0.0, le=1.0)
    max_observed_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    races_to_full_weight: int = Field(default=8, ge=1)
    max_delta_from_prior: float = Field(default=0.25, ge=0.0)


class HypotheticalPointsFloorConfig(StrictConfigModel):
    """Gate settings for the hypothetical team-swap points floor."""

    portable_skill_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    top_grid_limit: int = Field(default=10, ge=1)
    team_strength_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    dnf_probability_cap: float = Field(default=0.12, ge=0.0, le=1.0)


class FinalBlendConfig(StrictConfigModel):
    """Final race ranking blend after simulation sampling."""

    overtaking_skill_scale: float = Field(default=1.0, ge=0.0)
    race_advantage_scale: float = Field(default=0.8, ge=0.0)
    driver_skill_scale: float = Field(default=0.6, ge=0.0)
    low_confidence_racecraft_floor: float = Field(default=0.50, ge=0.0)
    low_confidence_max_gain_floor: float = Field(default=0.55, ge=0.0)
    elite_driver_skill_threshold: float = Field(default=0.88, ge=0.0)
    elite_driver_scale: float = Field(default=1.2, ge=0.0)
    elite_driver_exponent: float = Field(default=1.35, ge=0.0)
    max_driver_adjustment_positions: float = Field(default=1.2, ge=0.0)
    max_gain_base: float = Field(default=4.5, ge=0.0)
    max_gain_track_scale: float = Field(default=3.0, ge=0.0)
    max_gain_overtaking_skill_scale: float = Field(default=1.5, ge=0.0)
    max_gain_race_advantage_scale: float = Field(default=1.5, ge=0.0)
    max_gain_floor: float = Field(default=4.0, ge=0.0)
    max_gain_ceiling: float = Field(default=11.0, ge=0.0)
    hypothetical_points_floor: HypotheticalPointsFloorConfig = Field(
        default_factory=HypotheticalPointsFloorConfig
    )


class PodiumProbabilityConfig(StrictConfigModel):
    """Post-processing settings for podium-probability estimates."""

    min_sample_count: int = Field(default=250, ge=1)
    resample_seed_offset: int = Field(default=99173, ge=0)
    enforce_monotonic: bool = True


class PitLapVarianceConfig(StrictConfigModel):
    """Per-strategy stop-window variance."""

    one_stop: float = Field(default=3.0, ge=0.0)
    two_stop: float = Field(default=2.0, ge=0.0)


class StrategyConstraintsConfig(StrictConfigModel):
    """Hard strategy constraints and randomness controls."""

    min_pit_lap: int = Field(default=5, ge=1)
    max_pit_lap_from_end: int = Field(default=5, ge=1)
    min_laps_between_stops: int = Field(default=8, ge=1)
    pit_lap_variance: PitLapVarianceConfig = Field(default_factory=PitLapVarianceConfig)
    strategy_optimality: float = Field(default=0.60, ge=0.0, le=1.0)


class BaselineRaceConfig(StrictConfigModel):
    """Detailed baseline race configuration."""

    default_experience_tier: str = Field(default="developing")
    missing_driver_teammate_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    missing_driver_default_dnf_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    missing_driver_rookie_dnf_penalty: float = Field(default=0.02, ge=0.0)
    missing_driver_rookie_quali_penalty: float = Field(default=0.08, ge=0.0)
    missing_driver_rookie_race_penalty: float = Field(default=0.07, ge=0.0)
    missing_driver_rookie_skill_penalty: float = Field(default=0.08, ge=0.0)
    missing_driver_rookie_overtaking_penalty: float = Field(default=0.06, ge=0.0)
    missing_driver_second_year_penalty_scale: float = Field(default=0.55, ge=0.0)
    base_chaos: DryWetFloatConfig = Field(
        default_factory=lambda: DryWetFloatConfig(dry=0.28, wet=0.42)
    )
    track_chaos_multiplier: float = Field(default=0.4, ge=0.0)
    sc_base_probability: DryWetFloatConfig = Field(
        default_factory=lambda: DryWetFloatConfig(dry=0.45, wet=0.70)
    )
    sc_track_modifier: float = Field(default=0.25, ge=0.0)
    grid_weight_min: float = Field(default=0.15, ge=0.0, le=1.0)
    grid_weight_multiplier: float = Field(default=0.35, ge=0.0)
    grid_divisor: int = Field(default=21, ge=1)
    position_scaling: PositionScalingConfig = Field(default_factory=PositionScalingConfig)
    race_advantage_multiplier: float = Field(default=0.15, ge=0.0)
    overtaking_skill_multiplier: float = Field(default=0.25, ge=0.0)
    overtaking_grid_threshold: int = Field(default=5, ge=1)
    overtaking_track_threshold: float = Field(default=0.5, ge=0.0)
    lap1_chaos: Lap1ChaosConfig = Field(default_factory=Lap1ChaosConfig)
    strategy_variance_base: float = Field(default=0.30, ge=0.0)
    strategy_track_modifier: float = Field(default=0.5, ge=0.0)
    safety_car_luck_range: float = Field(default=0.25, ge=0.0)
    pace_weight_base: float = Field(default=0.40, ge=0.0)
    pace_weight_track_modifier: float = Field(default=0.10, ge=0.0)
    teammate_variance_std: float = Field(default=0.13, ge=0.0)
    teammate_setup_offset_ratio: float = Field(default=0.30, ge=0.0)
    teammate_variance_lap_ratio: float = Field(default=0.45, ge=0.0)
    dnf_rate_historical_cap: float = Field(default=0.20, ge=0.0, le=1.0)
    dnf_rate_final_cap: float = Field(default=0.35, ge=0.0, le=1.0)
    dnf_rate_floor: float = Field(default=0.02, ge=0.0, le=1.0)
    testing_long_run_modifier_scale: float = Field(default=0.05, ge=0.0)
    main_race_predicted_grid_sprint_confidence_cap: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )
    main_race_predicted_grid_sprint_confidence_caps_by_checkpoint: SprintCheckpointCapsConfig = (
        Field(default_factory=SprintCheckpointCapsConfig)
    )
    weekend_long_run_min_laps: int = Field(default=12, ge=1)
    long_run_outlier_threshold: float = Field(default=1.5, ge=0.0)
    long_run_trim_ends: bool = True
    testing_profile_weights: TestingProfileWeightsConfig = Field(
        default_factory=TestingProfileWeightsConfig
    )
    testing_modifier_clip_range: list[float] = Field(default_factory=lambda: [-0.04, 0.04])
    min_laps_for_compound_data: int = Field(default=10, ge=1)
    team_uncertainty_dnf_multiplier: float = Field(default=0.20, ge=0.0)
    defensive_skill_weights: DefensiveSkillWeightsConfig = Field(
        default_factory=DefensiveSkillWeightsConfig
    )
    safety_car_trigger_lap: int = Field(default=10, ge=1)
    overtake_model: OvertakeModelConfig = Field(default_factory=OvertakeModelConfig)
    pit_stops: PitStopsConfig = Field(default_factory=PitStopsConfig)
    tire_strategy: TireStrategyConfig = Field(default_factory=TireStrategyConfig)
    fuel: FuelConfig = Field(default_factory=FuelConfig)
    track_temperature: TrackTemperatureConfig = Field(default_factory=TrackTemperatureConfig)
    weather_features: WeatherFeaturesConfig = Field(default_factory=WeatherFeaturesConfig)
    tire_physics: TirePhysicsConfig = Field(default_factory=TirePhysicsConfig)
    lap_time: LapTimeConfig = Field(default_factory=LapTimeConfig)
    race_advantage_lap_impact: float = Field(default=0.20, ge=0.0)
    grid_uncertainty: GridUncertaintyConfig = Field(default_factory=GridUncertaintyConfig)
    grid_anchor: GridAnchorConfig = Field(default_factory=GridAnchorConfig)
    predicted_grid_uncertainty: PredictedGridUncertaintyConfig = Field(
        default_factory=PredictedGridUncertaintyConfig
    )
    main_race_movement_floor: float = Field(default=0.70, ge=0.0)
    main_race_movement_floor_track_scale: float = Field(default=0.25, ge=0.0, le=1.0)
    main_race_movement_quantile: float = Field(default=20.0, ge=0.0)
    position_interval_floor: PositionIntervalFloorConfig = Field(
        default_factory=PositionIntervalFloorConfig
    )
    overtaking_transition: OvertakingTransitionConfig = Field(
        default_factory=OvertakingTransitionConfig
    )
    final_blend: FinalBlendConfig = Field(default_factory=FinalBlendConfig)
    podium_probability: PodiumProbabilityConfig = Field(default_factory=PodiumProbabilityConfig)
    strategy_constraints: StrategyConstraintsConfig = Field(
        default_factory=StrategyConstraintsConfig
    )


class PracticeCaptureConfig(StrictConfigModel):
    """Practice-session auto-capture settings."""

    new_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    directionality_scale: float = Field(default=0.08, ge=0.0)
    session_aggregation: str = Field(default="laps_weighted")
    run_profile: str = Field(default="balanced")


class CompoundBlendWeightsConfig(StrictConfigModel):
    """Blend weights for compound characteristics by session type."""

    practice: float = Field(default=0.30, ge=0.0, le=1.0)
    sprint: float = Field(default=0.50, ge=0.0, le=1.0)
    race: float = Field(default=0.70, ge=0.0, le=1.0)


class BaselinePredictorSectionConfig(StrictConfigModel):
    """Baseline predictor configuration block."""

    team_strength_schedule: str = Field(default="rapid_adaptive")
    baseline_learning_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    mixed_wet_blend: float = Field(default=0.50, ge=0.0, le=1.0)
    current_season_form: CurrentSeasonFormConfig = Field(default_factory=CurrentSeasonFormConfig)
    driver_form: DriverFormConfig = Field(default_factory=DriverFormConfig)
    qualifying: BaselineQualifyingConfig = Field(default_factory=BaselineQualifyingConfig)
    compound_selection: CompoundSelectionConfig = Field(default_factory=CompoundSelectionConfig)
    race: BaselineRaceConfig = Field(default_factory=BaselineRaceConfig)
    practice_capture: PracticeCaptureConfig = Field(default_factory=PracticeCaptureConfig)
    compound_blend_weights: CompoundBlendWeightsConfig = Field(
        default_factory=CompoundBlendWeightsConfig
    )


class TestingConfig(StrictConfigModel):
    """Standalone testing helpers configuration."""

    seed: int = Field(default=42, ge=0)
    monte_carlo_runs: int = Field(default=100, ge=1)


class BaselinePredictorConfig(StrictConfigModel):
    """Root application configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    race: RaceConfig = Field(default_factory=RaceConfig)
    qualifying: QualifyingConfig = Field(default_factory=QualifyingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    track_defaults: TrackDefaultsConfig = Field(default_factory=TrackDefaultsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    baseline_predictor: BaselinePredictorSectionConfig = Field(
        default_factory=BaselinePredictorSectionConfig
    )
    testing: TestingConfig = Field(default_factory=TestingConfig)


def validate_config(config_dict: dict[str, Any]) -> BaselinePredictorConfig:
    """Validate the loaded YAML configuration against the strict schema."""
    return BaselinePredictorConfig(**config_dict)
