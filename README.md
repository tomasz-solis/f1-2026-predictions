# F1 2026 Predictions

Predict F1 qualifying results using practice session telemetry and Bayesian ranking.

Built for 2026 when regulations reset and past performance becomes less reliable.

## Current Status

**What's here:**
- Data extraction from FastF1 API (notebook 01)
- Telemetry feature engineering (notebook 02)

**What's coming:**
- Bayesian ranking system (notebook 03 - in progress)
- Validation results
- Production scripts

This is early-stage research code being converted to production.

## What This Will Do

Combine two sources of information:
1. Championship standings (what we know about teams/drivers)
2. Practice telemetry (what they're showing this weekend)

Use Bayesian inference to weight evidence and predict qualifying positions.

## Current Notebooks

### 01: Data Validation
Wraps FastF1 API to fetch session data:
- Practice sessions, qualifying, races
- Lap times, telemetry, weather
- Caches to Parquet for faster reloading

### 02: Feature Engineering  
Extracts performance metrics from raw telemetry:

**Per-lap features:**
- Corner speeds (classified as slow/medium/high by speed range)
- Throttle patterns (full throttle %, average, smoothness)
- Speed metrics (min/max/average)

**Per-session aggregates:**
- Best performance (max across clean laps)
- Average performance (mean)
- Consistency (standard deviation)

**Example output:**
```
Driver 1 - Bahrain Testing Day 3:
  best_medium_corner_speed: 149.5 km/h
  avg_medium_corner_speed: 148.2 km/h
  std_medium_corner_speed: 1.3 km/h
  clean_laps: 12
```

## Why These Features?

Corner speeds show car performance in different speed ranges. Different tracks have different corner profiles (Monaco = slow corners, Monza = high-speed corners). This helps predict track-specific performance.

Throttle metrics reveal driver confidence and car stability. Smooth throttle + high full-throttle percentage = good car balance.

## Structure
```
notebooks/
├── 01_data_validation.ipynb       # ✓ FastF1 data fetching
├── 02_feature_engineering.ipynb   # ✓ Telemetry → features
└── 03_bayesian_validation.ipynb   # [coming soon]

src/                                # [planned]
├── data/          # Data loading and caching
├── features/      # Feature extraction pipeline
├── models/        # Bayesian ranking system
└── prediction/    # Prediction orchestration
```

## Dependencies

- FastF1 (F1 data API)
- pandas, numpy (data processing)
- scipy (corner classification, statistics)

Install: `pip install fastf1 pandas numpy scipy`

## Usage

Currently notebooks only. Run in order:
1. `01_data_validation.ipynb` - Fetch and cache session data
2. `02_feature_engineering.ipynb` - Extract features from telemetry

Output: Parquet files with aggregated features per driver-session.

## Why 2026?

2026 brings major regulation changes:
- New power units (50/50 electric/ICE split)
- Active aerodynamics
- 30kg weight reduction
- New team (Cadillac)

When regulations reset, historical performance matters less. Testing data matters more. This system is designed for that scenario.

## What's Next

Adding the Bayesian ranking system that combines features with prior knowledge to make actual predictions. Once that's validated, extracting everything to production Python modules.

## License

MIT

## Acknowledgments

Built using [FastF1](https://github.com/theOehrly/Fast-F1) for F1 data access.