# Compound Performance Analysis System

This guide covers how compound-specific pace and degradation are extracted and then folded into race simulation.

## What It Covers

Teams behave differently across compounds (`SOFT`, `MEDIUM`, `HARD`). This system captures those differences from session data and feeds them into race simulation.

## What Compound Data Is Collected

### Data Sources

Compound performance is extracted from:
- Race sessions ([src/systems/updater.py](../src/systems/updater.py))
- Practice sessions (FP1/FP2/FP3) ([src/systems/testing_updater.py](../src/systems/testing_updater.py))
- Pre-season testing (via testing_updater)

### Metrics Extracted

For each team and compound combination, we collect:

1. Median lap time (raw seconds)
   - Most representative lap time on that compound
   - Resistant to outliers (uses median, not mean)

2. Tire degradation slope (seconds/lap)
   - How much lap time increases per lap
   - Linear regression across stint
   - Filtered: only slopes between -0.3 and +1.0 accepted

3. Consistency (standard deviation)
   - Lap time variance across stint
   - Lower = more consistent

4. Laps sampled (count)
   - How many laps contributed to these metrics
   - Used for reliability weighting

### MIN_LAPS_PER_COMPOUND Threshold

Threshold: 8 laps minimum ([src/systems/compound_analyzer.py:22](../src/systems/compound_analyzer.py#L22))

Rationale:
- First 2-3 laps = tire warm-up (not representative)
- Laps 4-8+ = stable performance window
- <8 laps = insufficient data, skipped

This prevents noise from short stints or out-laps.

## How Compound Data Is Normalized

### Track-Specific Normalization

Compound performance is normalized within each track, never across tracks.

Why? Melbourne SOFT ≠ Monaco SOFT
- Different track surfaces
- Different temperatures
- Different layout characteristics

Process ([src/systems/compound_analyzer.py:196-274](../src/systems/compound_analyzer.py#L196-L274)):
1. Collect all teams' data for same compound at same track
2. Find best/worst values for each metric
3. Normalize to 0-1 scale (1.0 = best, 0.0 = worst)
4. Store both raw and normalized values

### Normalized Metrics Stored

- `pace_performance` (0-1): Inverted median lap time
- `tire_deg_performance` (0-1): Inverted degradation slope (1.0 = low deg)
- `consistency_performance` (0-1): Inverted std deviation

## How Compound Data Is Applied

### 1. Dynamic Compound Selection

Races now select compounds using tire-stress data.

Implemented in:
- [src/utils/track_data_loader.py](../src/utils/track_data_loader.py) (`get_tire_stress_score`)
- [src/utils/pit_strategy.py](../src/utils/pit_strategy.py) (`_sample_compound_sequence`)

Uses [data/2025_pirelli_info.json](../data/2025_pirelli_info.json) (fallback for 2026):
- Calculate average stress: (traction + braking + lateral + abrasion) / 4
- High stress (>threshold): HARD compound (Bahrain, Singapore, Hungary)
- Low stress (<threshold): SOFT compound (Monaco, Canada)
- Medium stress (between thresholds): MEDIUM compound (most tracks)

Thresholds configured in [config/default.yaml](../config/default.yaml):
```yaml
baseline_predictor:
  compound_selection:
    high_stress_threshold: 3.5    # Above this: HARD
    low_stress_threshold: 2.5     # Below this: SOFT
    default_stress_fallback: 3.0  # Default if metric missing
```

This keeps tuning in config instead of code.

### 2. Team Strength Adjustment

Function: [src/utils/compound_performance.py:17-70](../src/utils/compound_performance.py#L17-L70)

Modifier calculation:
- Weighted combination: 70% pace + 30% tire degradation
- Centered around 0.5 (neutral)
- Scaled to ±0.05 modifier range
- Applied to base team strength

Example:
- Team base strength: 0.75
- Compound modifier: +0.03 (good on SOFT)
- Adjusted strength: 0.78

### 3. Race Prediction Integration

Applied in:
- [src/predictors/baseline/race/prediction_mixin.py](../src/predictors/baseline/race/prediction_mixin.py)
- [src/predictors/baseline/race/preparation_mixin.py](../src/predictors/baseline/race/preparation_mixin.py)

Flow:
1. Determine tire stress score for the race.
2. Generate per-driver pit strategy + compound sequence.
3. Use per-compound team strength and degradation slopes in lap-by-lap Monte Carlo simulation.

## When Compound Adjustments Are Used

Reliability Check: [src/utils/compound_performance.py:106-133](../src/utils/compound_performance.py#L106-L133)

Compound data is only applied if:
- ≥2 compounds have data
- ≥10 total laps sampled across compounds
- Each compound has ≥3 laps

If reliability check fails → use base team strength (no compound adjustment)

## Storage Format

Location: [data/processed/car_characteristics/2026_car_characteristics.json](../data/processed/car_characteristics/2026_car_characteristics.json)

```json
{
  "teams": {
    "McLaren": {
      "overall_performance": 0.85,
      "compound_characteristics": {
        "SOFT": {
          "track_name": "Bahrain Grand Prix",
          "median_lap_time": 91.234,
          "tire_deg_slope": 0.045,
          "consistency": 0.187,
          "pace_performance": 0.92,
          "tire_deg_performance": 0.78,
          "consistency_performance": 0.85,
          "laps_sampled": 24,
          "sessions_used": 2
        },
        "MEDIUM": { ... },
        "HARD": { ... }
      }
    }
  }
}
```

## Track-Aware Blending

When new session data arrives ([src/systems/compound_analyzer.py:277-351](../src/systems/compound_analyzer.py#L277-L351)):
- Same track: Blend old + new (default 50/50 weight)
- Different track: Replace entirely (no cross-track contamination)

This prevents Monaco SOFT data from contaminating Monza SOFT data.

## Multi-Stint Race Strategy System

### Lap-by-Lap Simulation

The race predictor now uses full lap-by-lap simulation with multi-compound pit stop strategies:

Architecture: Three core modules
- [src/utils/tire_degradation.py](../src/utils/tire_degradation.py) - Tire physics and fuel effects
- [src/utils/pit_strategy.py](../src/utils/pit_strategy.py) - Monte Carlo pit strategy generation
- [src/utils/lap_by_lap_simulator.py](../src/utils/lap_by_lap_simulator.py) - Race simulation engine

The simulation enforces the FIA two-compound rule for dry races and models
pit timing with Monte Carlo variance (±3 laps for a 1-stop strategy). Tire
degradation is linear from the compound-specific `tire_deg_slope`, modified by
fuel load (heavier = faster wear), a fresh-tire advantage window (SOFT 0.5 s,
MEDIUM 0.3 s, HARD 0.1 s over the first 2-3 laps), and a traffic-dependent
correction (front-runners get ~5 % better tire life, backmarkers ~5 % worse).
Pit-stop loss is track-specific using real circuit data (Monaco 19 s, Singapore
24 s). Strategy generation is driven by tire stress - high-stress tracks see
roughly 80 % two-stop probability.

Configuration: All parameters in [config/default.yaml](../config/default.yaml) under `baseline_predictor.race`:
- `tire_strategy.windows` - Pit stop lap windows (1-stop, 2-stop)
- `tire_strategy.stop_probability` - Stress-based stop count probabilities
- `tire_physics.fresh_tire_advantage` - Compound-specific fresh tire gains
- `strategy_constraints` - FIA rules, safety margins, optimality ratio

Data Sources:
- Tire degradation slopes: [data/processed/car_characteristics/2026_car_characteristics.json](../data/processed/car_characteristics/2026_car_characteristics.json)
- Tire stress scores: [data/2025_pirelli_info.json](../data/2025_pirelli_info.json)
- Track-specific pit loss: [data/processed/track_characteristics/2026_track_characteristics.json](../data/processed/track_characteristics/2026_track_characteristics.json)

Example Output:
```
Tire Compound Strategies
SOFT→MEDIUM: 62.5%
MEDIUM→HARD: 28.3%
SOFT→HARD: 9.2%

Pit Stop Windows
L25-30: 35 stops
L30-35: 28 stops
L20-25: 12 stops
```

Relevant tests:
- [tests/test_tire_degradation.py](../tests/test_tire_degradation.py) (18 tests)
- [tests/test_pit_strategy.py](../tests/test_pit_strategy.py) (22 tests)

## Current Limitations

1. No wet compound modeling
   - INTERMEDIATE and WET compounds collected but not used in predictions
   - Fallback: Base team strength in rain conditions
   - Future: Apply compound adjustments for wet races

2. Temperature sensitivity not modeled
   - Track temperature affects compound performance
   - Hot tracks favor HARD, cool tracks favor SOFT
   - Future: Integrate temperature forecast data into compound selection

## Testing

Test coverage: [tests/test_compound_analyzer.py](../tests/test_compound_analyzer.py)
- Extraction from lap data
- Normalization across teams
- Aggregation with track awareness
- Minimum lap threshold enforcement

## Observed Impact

With compound adjustments (realistic scenario):
- Teams with good compound data: ±0.02 to ±0.05 strength adjustment
- Average impact: ~0.5-1.0 position change in race predictions
- Benefit: More accurate predictions for compound-sensitive tracks (Monaco, Singapore, Bahrain)

## Planned Extensions

1. Temperature sensitivity
   - Track temperature affects compound performance
   - Hot tracks favor HARD, cool tracks favor SOFT
   - Integration point: `get_fresh_tire_advantage()` already has track_temp parameter

2. Track evolution
   - Rubber buildup improves grip over race
   - Affects compound degradation patterns
   - Could reduce tire_deg_slope dynamically as race progresses

3. Compound-specific driver skill
   - Some drivers excel at managing tire deg
   - Could add driver × compound interaction effects
   - Would require collecting driver-level stint data

4. Undercut/overcut dynamics
   - Model strategic pit stop timing advantages
   - Track position changes during pit stop phases
   - Requires modeling gap intervals between cars
