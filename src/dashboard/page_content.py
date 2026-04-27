"""Static markdown/HTML content for dashboard pages."""

MODEL_INSIGHTS_MARKDOWN = """
### How the model works

The dashboard uses one predictor (`Baseline2026Predictor`) for both qualifying and race forecasts.

**1. Team baseline**
- Builds team strength from preseason baseline, testing directionality, and current-season performance.
- Uses the `rapid_adaptive` reset-year schedule so trust shifts toward current-season evidence.

**2. Qualifying forecast**
- Uses the best available weekend practice pace for blending.
- Falls back to testing short-run profiles when weekend practice data is unavailable.
- Applies driver/team adjustments, then runs Monte Carlo simulations.
- Outputs median grid positions with confidence intervals.

**3. Race forecast**
- Starts from predicted or actual qualifying order, depending on session availability.
- Runs lap-by-lap Monte Carlo simulation with pace, tire degradation, strategy, overtaking, and reliability effects.
- Sprint weekends use adjusted race dynamics (lower chaos variance and higher grid influence).
- Derives podium probability from ranked simulation outcomes.

**4. Learning loop**
- Saved predictions with actuals update a persistent calibration state.
- Tracks per-driver and teammate residual errors by session type.
- Applies learned adjustments in qualifying and race scoring.
- Skips retrospective records, duplicate run IDs, missing actuals, and tiny actual overlaps.

**5. Outputs**
- Expected finish order, uncertainty bands, podium probabilities, and strategy distribution summaries.

**6. Research safeguards**
- Experimental model components must pass promotion gates before stacking.
- Ablation reports compare champion and challenger movement against actual results.
"""

QUALIFYING_HYPERPARAMETERS_MARKDOWN = """
**Qualifying (active path):**
- Default prior: team/driver score starts at 70% team + 30% driver
- Practice blend when available; testing fallback otherwise
- Model-only mode rebalances weights and applies teammate/experience controls
- Learned adjustment offsets are applied when calibration history is available
- Output: Monte Carlo median grid + confidence intervals
"""

RACE_HYPERPARAMETERS_MARKDOWN = """
**Race (active path):**
- Default prior: pace term starts at 40% and is adjusted by track overtaking profile
- Grid influence is dynamic by overtaking difficulty and starting position context
- Driver skill term starts at 20% and is normalized with grid/pace terms per simulation
- Includes DNF probability, lap-1 chaos, strategy variance, and safety car modifiers
- Podium probability from ranked outcomes with monotonic smoothing
"""

CONTACT_PAGE_HTML = """
<div class="contact-grid">
  <section class="contact-card">
    <h3>Project Links</h3>
    <div class="contact-link-stack">
      <a class="contact-link-row" href="https://github.com/tomasz-solis/f1-2026-predictions" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">GitHub</span>
        <span class="contact-link-row__value">f1-2026-predictions</span>
      </a>
      <a class="contact-link-row" href="https://www.linkedin.com/in/tomaszsolis/" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">LinkedIn</span>
        <span class="contact-link-row__value">Tomasz Solis</span>
      </a>
    </div>
  </section>
  <section class="contact-card">
    <h3>Project Scope</h3>
    <p>Race weekend prediction workflow built on a 2026 baseline with persistent learning and accuracy tracking.</p>
    <ul>
      <li>Baseline/testing/current-season team blending</li>
      <li>Practice-aware qualifying and race simulation</li>
      <li>Session-based logging for gated learning and post-race accuracy analysis</li>
    </ul>
  </section>
</div>
<section class="contact-card contact-card--full">
  <h3>Disclaimer</h3>
  <p>Independent analytics project. Not affiliated with any racing series, team, or governing body.</p>
</section>
"""
