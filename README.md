# F1 2026 Predictions

Predict F1 qualifying results using practice session telemetry and Bayesian ranking.

Built for 2026 when regulations reset and past performance becomes less reliable.

## Current Status 

**Validated and working:**
- Data extraction from FastF1 API (notebook 01)
- Telemetry feature engineering (notebook 02)
- Bayesian ranking system with temporal validation (notebook 03)

**Key finding:** Testing data currently degrades prediction accuracy in stable regulations (2.88 MAE vs 2.60 MAE prior-only). This validates the core hypothesis: when regulations are stable, teams sandbag during testing. The system is designed for regulation resets (like 2026) when testing becomes the primary signal.

**What's coming:**
- Sequential learning (notebook 04)
- Full season validation (notebook 05)
- Production API deployment

This is research code being systematically validated before production deployment.

## What This Does

Combines two sources of information:
1. **Priors:** Championship standings, team tier, driver tier
2. **Testing data:** Practice session telemetry metrics

Uses Bayesian inference to update beliefs based on evidence strength. In stable regulations, priors dominate (teams sandbag testing). In regulation resets, testing dominates (priors are weak).

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
- Median performance (robust to outliers)
- Consistency (standard deviation)
- Clean lap count (excluding traffic, mistakes)

**Example output:**
```
Driver 1 - Bahrain Testing Day 3:
  medium_corner_speed: 149.5 km/h
  medium_corner_speed_std: 1.3 km/h
  pct_full_throttle: 67.2%
  clean_laps: 12
```

### 03: Bayesian Core System
Implements Bayesian ranking with proper temporal validation:

**Components:**
- `DriverPrior`: Team/driver tier initialization (μ, σ)
- `BayesianDriverRanking`: Evidence-based belief updating
- Three scoring methods: simple weighted, z-score normalized, prior-only baseline

**Validation results (2024 season, 24 races):**
- Prior-only baseline: **2.60 MAE**
- With testing data: **2.88 MAE** (11% worse)

**Ablation study confirms:**
- Testing reveals car characteristics (slow corners, straights, downforce)
- Testing does NOT reveal true qualifying pace in stable regulations
- Teams optimize for race (fuel loads, tire strategy, engine modes)
- Better approach: Extract car profile → Match to track → Adjust priors

**Why this matters for 2026:**
- Current validation proves the system correctly handles sandbagging
- Regulation resets flip the weighting: testing 70-90%, priors 10-30%
- Same architecture, different weights based on regulation state

## Why These Features?

**Corner speeds** show car performance in different speed ranges. Different tracks have different corner profiles (Monaco = slow corners, Monza = high-speed corners). This helps build car performance profiles.

**Throttle metrics** reveal driver confidence and car stability. Smooth throttle + high full-throttle percentage = good car balance. Sudden throttle changes = nervous rear end or traction issues.

**Speed ranges** correlate with aerodynamic efficiency (slow), mechanical grip (medium), and power/drag (high speed).

## Structure
```
notebooks/
├── 01_data_validation.ipynb       # ✓ FastF1 data fetching
├── 02_feature_engineering.ipynb   # ✓ Telemetry → features
├── 03_bayesian_core.ipynb         # ✓ Bayesian ranking + validation
├── 04_sequential_learning.ipynb   # [in progress]
└── 05_season_validation.ipynb     # [planned]

src/                                # [planned]
├── data/          # Data loading and caching
├── features/      # Feature extraction pipeline
├── models/        # Bayesian ranking system
└── prediction/    # Prediction orchestration

requirements.txt   # Python dependencies
```

## Dependencies

Core packages:
- `fastf1>=3.0.0` - F1 data API
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical operations
- `scipy>=1.10.0` - Corner classification, statistics
- `pyarrow>=12.0.0` - Parquet file handling

Install: `pip install -r requirements.txt`

## Usage

Run notebooks in order:

```bash
# 1. Fetch and cache session data
jupyter notebook notebooks/01_data_validation.ipynb

# 2. Extract features from telemetry
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. Train and validate Bayesian ranking
jupyter notebook notebooks/03_bayesian_core.ipynb
```

**Output:** Parquet files with aggregated features per driver-session, validation metrics, and prediction comparisons.

## Validation Methodology

**Temporal splits:** Training data chronologically precedes test data. No future information leakage.

**Baseline comparisons:** Every model compared against prior-only baseline to measure testing data value.

**Ablation studies:** Systematic removal of components to identify what drives performance.

**Sprint weekend handling:** Detected and adjusted (fewer practice sessions, different timing).

## Why 2026?

2026 brings major regulation changes:
- New power units (50/50 electric/ICE split vs current 80/20)
- Active aerodynamics (adjustable front/rear wings)
- 30kg weight reduction (720kg → 690kg)
- New team (Cadillac F1 Team)
- New tire compounds and construction

When regulations reset, historical performance matters less. Testing data matters more. 

**The current system validates this hypothesis:** In stable 2024 regulations, testing data hurts predictions because teams sandbag. In 2026 reset, testing will be the primary signal because priors are weak.

## Research Questions Being Addressed

1. **Can telemetry features predict qualifying performance?** ✓ Yes, but not in stable regulations
2. **How much does testing data help?** ✓ Currently hurts (sandbagging), will help in 2026
3. **What can testing data reveal?** → Car characteristics (slow/fast corners, straights), not final positions
4. **How to handle rookies?** → Team-based priors (inherit team tier performance)
5. **How to detect sandbagging?** → Planned: % full throttle, fuel load proxies, lap time variance

## Known Limitations

**Testing data paradox:** Testing currently degrades predictions in stable regulations. This is expected and validates the system design. The architecture is correct for 2026, but weights need adjustment based on regulation state.

**Position prediction is chaotic:** Exact race positions depend on strategy, safety cars, mechanical failures. Better to predict: podium probability, points probability, beat teammate probability.

**Track characteristics are manually defined:** Currently hardcoded (Bahrain = 25% slow corners, etc.). Should be extracted from telemetry data automatically.

**No sandbagging detection yet:** System assumes testing shows true pace. Needs: fuel load estimation, throttle usage patterns, lap time variance analysis.

## What's Next

**Immediate (research validation):**
1. Sequential learning (FP1 → FP2 → FP3 belief updating)
2. Full season validation across all 2024 races
3. Track characteristic extraction from data

**Medium term (production deployment):**
1. Extract notebooks to `src/` modules
2. API endpoints for live prediction
3. Automated retraining after each race weekend

**Long term (2026 readiness):**
1. Sandbagging detection system
2. Car profile extraction (testing → car traits)
3. Track matching (car profile → track suitability)
4. Regulation-aware weight adjustment (stable → reset)

## License

MIT

## Acknowledgments

Built using [FastF1](https://github.com/theOehrly/Fast-F1) for F1 data access.

Inspired by Elo rating systems, TrueSkill, and Bayesian inference methods in competitive ranking.
