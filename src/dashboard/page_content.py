"""Static markdown/HTML content for dashboard pages."""

MODEL_INSIGHTS_MARKDOWN = """
### Runtime path

The dashboard currently runs `Baseline2026Predictor` for both qualifying and race.

**1. Team strength**
- Uses baseline (pre-season), testing directionality, and current-season performance.
- Applies a race-by-race weight schedule that shifts toward current-season data.
- Uses compound-aware modifiers when validated compound samples are available.

**2. Qualifying**
- Uses best available weekend practice data for team pace blending.
- Falls back to testing short-run profiles when no weekend practice data is available.
- Applies model-only stabilization for teammate gaps and experience tiers.
- Runs Monte Carlo simulations and reports median/interval grid outputs.

**3. Race**
- Uses predicted or actual qualifying grid depending on session availability.
- Uses lap-by-lap simulation with team pace, racecraft, strategy, and reliability.
- Applies track-aware overtaking and pit timing bias (undercut/overcut tendency).
- Derives podium probability from ranked simulation outcomes for consistency.

**4. Learning**
- Saved predictions with actuals update a persistent calibration state.
- Driver and teammate residual errors are tracked per session type.
- Learned adjustments are applied in qualifying and race scoring.

**5. Supporting systems**
- Auto-updater ingests completed races into characteristics.
- Testing updater refreshes run-profile and compound characteristics.
- Bayesian ranking tools remain available for offline analysis workflows.
"""

QUALIFYING_HYPERPARAMETERS_MARKDOWN = """
**Qualifying (active path):**
- Team/driver score: 70% team + 30% driver
- Practice blend when available; testing fallback otherwise
- Model-only teammate gap controls + learned adjustment offsets
- Output: Monte Carlo median grid + confidence intervals
"""

RACE_HYPERPARAMETERS_MARKDOWN = """
**Race (active path):**
- Base pace weight: 40% (track-adjusted)
- Grid influence: dynamic by overtaking difficulty
- Driver skill term: 20%
- DNF probability + chaos + strategy + safety car modifiers
- Podium probability from ranked outcomes with monotonic smoothing
"""

CONTACT_PAGE_HTML = """
<div class="contact-grid">
  <section class="contact-card">
    <h3>Links</h3>
    <div class="contact-link-stack">
      <a class="contact-link-row" href="https://github.com/tomasz-solis" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">GitHub</span>
        <span class="contact-link-row__value">@tomasz-solis</span>
      </a>
      <a class="contact-link-row" href="https://linkedin.com/in/tomaszsolis" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">LinkedIn</span>
        <span class="contact-link-row__value">/in/tomaszsolis</span>
      </a>
    </div>
  </section>
  <section class="contact-card">
    <h3>Project Scope</h3>
    <p>Race weekend prediction workflow for the 2026 season with persistent learning and accuracy tracking.</p>
    <ul>
      <li>Baseline/testing/current-season team blending</li>
      <li>Practice-aware qualifying and race simulation</li>
      <li>Session-based logging for post-race accuracy analysis</li>
    </ul>
  </section>
</div>
<section class="contact-card contact-card--full">
  <h3>Disclaimer</h3>
  <p>Independent analytics project. Not affiliated with any racing series, team, or governing body.</p>
</section>
"""
