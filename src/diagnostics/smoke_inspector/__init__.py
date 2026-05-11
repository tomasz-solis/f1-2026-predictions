"""Read-only FastF1 session inspector for the Phase 2 smoke-session lock.

This package is scoped to Phase 2 of the master execution plan. It
loads FastF1 sessions and emits human-readable summaries the analyst
uses to fill the smoke-session doc's expected behavior table. It does
not contain matching logic, skip-reason emission, or weather-routing
classification; that work belongs to the Phase 3 extractor.

The deletion test: if this package were removed after Phase 2 closes,
nothing Phase 3 needs to implement should be lost. If that ever fails
to hold, the extractor has leaked into Phase 2.
"""
