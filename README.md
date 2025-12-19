# F1 2026 Predictions

Predict qualifying results using practice telemetry and Bayesian rankings.

Built for 2026 when regulations reset and historical data becomes less useful.

## What's Working

Validated through notebook 03:
- Data extraction from FastF1
- Telemetry feature engineering
- Bayesian ranking with temporal validation
- Sequential learning (normal vs sprint weekends)

Key finding: Testing data hurts predictions in stable regulations (2.88 MAE vs 2.60 MAE using just priors). This makes sense - teams sandbag when they know their cars. In 2026 with regulation reset, testing becomes the primary signal because nobody knows anything yet.

## How It Works

Two information sources:
1. Priors: Championship standings, team/driver tiers
2. Testing: Practice session telemetry

Bayesian inference combines them based on how much you should trust each. In 2024 (stable regs), trust priors. In 2026 (reset), trust testing.

## Notebooks

### 01: Data Validation
Wraps FastF1 API:
- Fetch practice, qualifying, race data
- Extract lap times, telemetry, weather
- Cache to Parquet (faster reloading)

### 02: Feature Engineering
Converts raw telemetry into performance metrics:
- Corner speeds (slow/medium/high by speed range)
- Throttle patterns (% at full, average, smoothness)
- Speed metrics (min/max/average across lap)

Aggregates per session:
- Median performance (robust to outliers)
- Consistency (std dev)
- Clean lap count

Example:
```
VER - Bahrain Testing Day 3:
  medium_corner_speed: 149.5 km/h
  pct_full_throttle: 67.2%
  clean_laps: 12
```

### 03: Bayesian Core
Implements ranking system with proper validation:
- DriverPrior: Initialize from championship standings
- BayesianDriverRanking: Update beliefs from evidence
- Temporal validation: Train on past, test on future

Results (2024 season, 24 races):
- Prior-only: 2.60 MAE (baseline)
- With testing: 2.88 MAE (11% worse)

Why testing hurts: Teams sandbag. They run heavy fuel, test programs, hide pace. Testing reveals car characteristics (good in slow corners, etc.) but not final positions.

### 04: Sequential Learning
Tests normal vs sprint weekends (24 races from 2024):

Results:
- Normal weekends: 6.3% improvement (FP1+FP2+FP3)
- Sprint weekends: 14.5% improvement (FP1+Sprint Quali)

Sprint qualifying is competitive (points on line), practice is sandbagged. One competitive session beats three practice sessions.

Stats: p = 0.0009, d = 1.80. Sprint weekends definitively better.

## Why These Features?

Corner speeds show car performance at different speed ranges. Different tracks emphasize different corners (Monaco = slow, Monza = high-speed).

Throttle metrics reveal driver confidence and car stability. Smooth throttle + high % full throttle = good balance.

Speed ranges correlate with aero efficiency (slow), mechanical grip (medium), power/drag (high).

## Structure

```
notebooks/
├── 01_data_validation.ipynb
├── 02_feature_engineering.ipynb
├── 03_bayesian_core.ipynb
└── 04_sequential_learning.ipynb

src/
├── features/
│   ├── extractors.py    # Telemetry → features
│   └── pipeline.py      # Full pipeline
└── models/
    ├── bayesian.py      # Ranking system
    ├── scoring.py       # Performance scores
    ├── car.py           # Car profiles
    └── helpers.py       # Utilities
```

## Install

```bash
pip install -r requirements.txt
```

Core packages: fastf1, pandas, numpy, scipy, pyarrow

## Usage

Run notebooks in order:

```bash
jupyter notebook notebooks/01_data_validation.ipynb   # Fetch data
jupyter notebook notebooks/02_feature_engineering.ipynb  # Extract features
jupyter notebook notebooks/03_bayesian_core.ipynb     # Validate system
jupyter notebook notebooks/04_sequential_learning.ipynb  # Test weekends
```

Outputs: Parquet files with features, validation metrics, predictions.

## Why 2026?

Major regulation changes:
- New power units (50/50 electric/ICE vs current 80/20)
- Active aero (adjustable front/rear wings)
- 30kg lighter (720kg → 690kg)
- New team (Cadillac)
- New tires

When regs reset, historical performance matters less. Testing matters more. Current validation proves the system handles both scenarios - just needs weight adjustment.

## Known Issues

Testing data degrades predictions in stable regulations. This is expected - validates the design. Architecture is correct for 2026, weights need adjustment based on reg state.

Exact race positions are chaotic (strategy, safety cars, failures). Better targets: podium probability, points probability, beat teammate.

Track characteristics manually defined (Bahrain = 25% slow corners). Should extract from telemetry automatically.

No sandbagging detection yet. Needs: fuel load estimation, throttle patterns, lap time variance.

## What's Next

Research validation:
- Track characteristic extraction from data
- Sandbagging detection system
- Car profile extraction (testing → car traits)

Production deployment:
- API endpoints for live predictions
- Automated retraining after race weekends
- Regulation-aware weight adjustment

## License

MIT

Data from [FastF1](https://github.com/theOehrly/Fast-F1).

## Contact

Tomasz Solis
- Email: tomasz.solis@gmail.com
- LinkedIn: linkedin.com/in/tomaszsolis
- GitHub: github.com/tomasz-solis
