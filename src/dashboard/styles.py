"""Shared Streamlit CSS for dashboard layout."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap');

/* ---- Brand tokens (assets/BRAND_GUIDE.md) ---- */
:root {
  --ts-graphite: #0B0F14;
  --ts-soft-light: #E8EDF2;
  --ts-steel: #8B949E;
  --ts-heat: #FF4D2D;
  --ts-heat-soft: #FF7A62;
  --ts-panel: #111826;
  --ts-panel-alt: #0f1623;
  --ts-card: rgba(15,24,38,0.74);
  --ts-card-strong: rgba(16,26,42,0.88);
  --ts-card-border: rgba(232,237,242,0.14);
  --ts-border: rgba(232,237,242,0.12);
  --ts-nav-bg: rgba(10,16,28,0.75);
  --ts-nav-pill: rgba(18,28,44,0.92);
  --ts-nav-pill-active: linear-gradient(135deg, #ff6a4f 0%, #f0452c 100%);
  --ts-gap: 1rem;
  --ts-page-max: 1280px;
  --ts-readable-max: 980px;
  --ts-movement-chart-max: 760px;
}

/* --- Sticky header (collapsible) --- */
.ts-sticky-header {
  position: sticky;
  top: 0;
  z-index: 200;
  background: rgba(11,15,20,0.45);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(232,237,242,0.10);
  padding: 0.7rem 0;
  transition: padding 180ms ease, background 180ms ease;
}

/* Make header content animate nicely */
.ts-sticky-header .brand-row {
  transition: margin 180ms ease;
}
.ts-sticky-header .brand-logo {
  transition: width 180ms ease, transform 180ms ease, opacity 180ms ease;
  transform-origin: left center;
}
.ts-sticky-header .sub-header,
.ts-sticky-header .micro-disclaimer {
  transition: opacity 160ms ease, max-height 180ms ease, margin 180ms ease;
  overflow: hidden;
}

/* Collapsed state (added by JS) */
.ts-sticky-header.is-collapsed {
  padding: 0.35rem 0;
  background: rgba(11,15,20,0.70);
}

.ts-sticky-header.is-collapsed .brand-row {
  margin-bottom: 0.35rem;
}

.ts-sticky-header.is-collapsed .brand-logo {
  transform: scale(0.86);
}

.ts-sticky-header.is-collapsed .sub-header {
  opacity: 0;
  max-height: 0;
  margin: 0;
}

.ts-sticky-header.is-collapsed .micro-disclaimer {
  opacity: 0;
  max-height: 0;
  margin: 0;
}

/* Mobile: collapse a bit earlier + keep it tighter */
@media (max-width: 760px) {
  .ts-sticky-header .brand-logo { transform-origin: left center; }
  .ts-sticky-header.is-collapsed .brand-logo { transform: scale(0.80); }
}

html, body {
  background: var(--ts-graphite) !important;
  color-scheme: dark;
}

[data-testid="stApp"] {
  background: var(--ts-graphite) !important;
}

[data-testid="stAppViewContainer"] > div:first-child {
  background: transparent !important;
}

@media (max-width: 760px) {

  /* Header spacing + scale */
  .brand-row {
    margin-bottom: 0.55rem;
  }

  .brand-logo {
    width: min(420px, 92vw);
  }

  .sub-header {
    font-size: 0.98rem;
  }

  .micro-disclaimer {
    font-size: 0.80rem;
    margin-bottom: 0.85rem;
  }

  /* Segmented nav fills width nicely */
  [data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
    width: 100%;
  }

  [data-testid="stSegmentedControl"] [data-baseweb="button-group"] button {
    flex: 1 1 auto;
    text-align: center;
  }

  /* Settings expander lighter */
  [data-testid="stExpander"] {
    margin-bottom: 0.8rem;
    border-radius: 12px !important;
  }

  /* Reduce double padding on mobile */
  [data-testid="stAppViewContainer"] .main .block-container,
  [data-testid="stMain"] .block-container,
  [data-testid="stMainBlockContainer"],
  section.main .block-container,
  main .block-container {
    padding-top: 1rem !important;
    padding-right: 1rem !important;
    padding-left: 1rem !important;
  }

  /* Prevent column extra padding feel */
  div[data-testid="column"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
  }
}

[data-testid="stAppViewContainer"] {
  position: relative;
  z-index: 0;
}

[data-testid="stAppViewContainer"] .main {
  position: relative;
  z-index: 1;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(62% 70% at 12% 0%, rgba(255,77,45,0.06), rgba(11,15,20,0) 58%),
      radial-gradient(58% 68% at 88% 6%, rgba(52,99,184,0.10), rgba(11,15,20,0) 60%),
      linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.0) 24%),
      repeating-linear-gradient(90deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 132px);
    opacity: 0.30;
    z-index: 0;
}
[data-testid="stHeader"] {
    background: rgba(11,15,20,0.0);
}
[data-testid="stSidebar"] {
    background: #0E1521;
    border-right: 1px solid rgba(255,255,255,0.09);
}

/* ---- Shared page rail ---- */
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"],
section.main .block-container,
main .block-container {
  width: 100%;
  max-width: var(--ts-page-max) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 1.25rem !important;
  padding-right: 2.6rem !important;
  padding-left: 2.6rem !important;
  padding-bottom: 2.2rem !important;
}

@media (max-width: 1080px) {
  [data-testid="stAppViewContainer"] .main .block-container,
  [data-testid="stMain"] .block-container,
  [data-testid="stMainBlockContainer"],
  section.main .block-container,
  main .block-container {
    padding-right: 1.6rem !important;
    padding-left: 1.6rem !important;
  }
}

@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container,
  [data-testid="stMain"] .block-container,
  [data-testid="stMainBlockContainer"],
  section.main .block-container,
  main .block-container {
    padding-top: 1rem !important;
    padding-right: 1rem !important;
    padding-left: 1rem !important;
  }
}

/* ---- Typography ---- */
html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}
.main-header {
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: 2.1rem;
  font-weight: 650;
  letter-spacing: 0.015em;
  text-align: left;
  margin: 0 0 0.15rem 0;
  color: var(--ts-soft-light);
}
.sub-header {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: 1.03rem;
  line-height: 1.45;
  max-width: 62ch;
  text-align: left;
  color: rgba(232,237,242,0.94);
  margin: 0 0 0.35rem 0;
  letter-spacing: 0.01em;
}
.micro-disclaimer {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: 0.86rem;
  line-height: 1.5;
  max-width: 90ch;
  text-align: left;
  color: rgba(139,148,158,0.95);
  margin: 0 0 1.15rem 0;
}
.brand-shell {
  margin: 0 0 0.52rem 0;
  padding-top: 0.2rem;
}
.brand-row {
  display: flex;
  align-items: flex-end;
  justify-content: flex-start;
  margin: 0 0 0.95rem 0;
}
.brand-logo {
  width: clamp(340px, 44vw, 700px);
  max-width: 100%;
  height: auto;
  display: block;
}

.brand-shell--center .brand-row {
  justify-content: center;
}
.brand-shell--center .sub-header,
.brand-shell--center .micro-disclaimer,
.brand-shell--center .main-header {
  text-align: center;
  margin-left: auto;
  margin-right: auto;
}
.brand-shell--center .brand-logo {
  width: min(560px, 90vw);
}

@media (max-width: 960px) {
  .brand-logo {
    width: min(520px, 92vw);
  }
}

/* Stronger global text defaults (Streamlit DOM-safe) */
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] *,
[data-testid="stHeader"] *,
[data-testid="stToolbar"] * {
  color: rgba(232,237,242,0.9);
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  line-height: 1.6;
}
[data-testid="stMarkdownContainer"] a {
  color: #9dc6ff;
  text-decoration: none;
}
[data-testid="stMarkdownContainer"] a:hover {
  color: #c5ddff;
  text-decoration: underline;
}

/* Headings */
h1, h2, h3, h4 {
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  color: var(--ts-soft-light) !important;
  letter-spacing: 0.008em;
}
h2 {
  margin-top: 0.62rem;
  margin-bottom: 0.92rem;
}

[data-testid="stSegmentedControl"] {
  margin-bottom: 0.45rem;
}

[data-testid="stTabs"] [role="tablist"] {
  gap: 0.35rem;
  border-bottom: 0;
  background: linear-gradient(180deg, rgba(10,16,28,0.88), rgba(8,14,24,0.84));
  border: 1px solid rgba(232,237,242,0.16);
  border-radius: 12px;
  padding: 0.34rem;
  width: 100%;
  max-width: var(--ts-page-max);
  margin: 0 auto 0.9rem;
  overflow-x: auto;
  flex-wrap: nowrap;
  scrollbar-width: none;
}
[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
  display: none;
}
[data-testid="stTabs"] [role="tab"] {
  flex: 0 0 auto;
  min-height: 2.5rem;
  border-radius: 9px;
  border: 1px solid transparent;
  color: rgba(232,237,242,0.78);
  font-weight: 650;
  padding: 0.45rem 0.85rem;
  transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
}
[data-testid="stTabs"] [role="tab"]:hover {
  border-color: rgba(255,122,98,0.38);
  background: rgba(255,122,98,0.08);
  color: rgba(232,237,242,0.96);
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(255,122,98,0.95), rgba(255,77,45,0.92));
  border-color: rgba(255,146,124,0.9);
  color: #fff;
  box-shadow: 0 8px 18px rgba(255,77,45,0.24);
}
[data-testid="stTabs"] [role="tabpanel"] {
  width: 100%;
  max-width: var(--ts-page-max);
  margin: 0 auto;
  padding-top: 0.35rem;
}

/* ---- Navigation (segmented tabs) ---- */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  align-items: center;
  background: linear-gradient(180deg, rgba(10,16,28,0.92), rgba(8,14,24,0.9));
  border: 1px solid rgba(232,237,242,0.24);
  border-radius: 14px;
  padding: 0.32rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.07), 0 10px 20px rgba(0,0,0,0.24);
}
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button {
  border-radius: 9px;
  border: 1px solid rgba(232,237,242,0.16);
  background: var(--ts-nav-pill);
  color: rgba(232,237,242,0.9) !important;
  font-weight: 600;
  letter-spacing: 0.01em;
  font-size: 0.96rem;
  padding: 0.42rem 0.88rem !important;
  transition: all 140ms ease;
}
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button:hover {
  border-color: rgba(255,122,98,0.5);
  transform: translateY(-1px);
}
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button[aria-pressed="true"] {
  border-color: rgba(255,146,124,0.9);
  background: var(--ts-nav-pill-active);
  box-shadow: 0 8px 16px rgba(255,77,45,0.3), inset 0 1px 0 rgba(255,255,255,0.16);
  color: #fff !important;
}

/* ---- Sidebar primary navigation ---- */
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
  margin: 0.25rem 0 1rem;
}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
  width: 100%;
  align-items: stretch;
  flex-direction: column;
  gap: 0.32rem;
  border-radius: 12px;
  padding: 0.36rem;
  background: linear-gradient(180deg, rgba(8,14,24,0.94), rgba(11,17,28,0.9));
  border-color: rgba(232,237,242,0.16);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [data-baseweb="button-group"] button {
  width: 100%;
  justify-content: flex-start;
  min-height: 2.35rem;
  border-color: transparent;
  background: rgba(18,28,44,0.54);
  box-shadow: none;
  text-align: left;
}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [data-baseweb="button-group"] button:hover {
  transform: none;
  border-color: rgba(255,122,98,0.30);
  background: rgba(24,36,55,0.74);
}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [data-baseweb="button-group"] button[aria-pressed="true"] {
  border-color: rgba(255,122,98,0.42);
  background: linear-gradient(90deg, rgba(255,122,98,0.22), rgba(18,28,44,0.72));
  box-shadow: inset 3px 0 0 rgba(255,122,98,0.94);
}
[data-testid="stExpander"] {
  margin-top: 0.75rem;
  margin-bottom: 1.2rem;
  border: 1px solid rgba(232,237,242,0.14) !important;
  border-radius: 12px !important;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(17,24,38,0.9), rgba(13,19,32,0.86));
  box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}
[data-testid="stExpander"] details summary {
  font-weight: 650;
  letter-spacing: 0.01em;
}

/* ---- Sidebar controls contrast ---- */
[data-testid="stSidebar"] * {
  color: rgba(232,237,242,0.86) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: var(--ts-soft-light) !important;
  font-weight: 700 !important;
}

/* ---- Inputs (readable values + placeholders) ---- */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: linear-gradient(180deg, rgba(15,22,35,0.96), rgba(12,18,31,0.96)) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(232,237,242,0.14) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
[data-baseweb="select"] * {
  color: rgba(232,237,242,0.94) !important;
}
[data-baseweb="select"] input::placeholder {
  color: rgba(232,237,242,0.45) !important;
}
label {
  color: rgba(232,237,242,0.82) !important;
}

/* Sidebar Navigation selectbox polish */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border-radius: 14px !important;
  border: 1px solid rgba(232,237,242,0.18) !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] svg {
  color: rgba(232,237,242,0.75) !important;
}

/* Radio fallbacks used when segmented controls are not available. */
[data-testid="stRadio"] [role="radiogroup"] {
  display: flex;
  flex-wrap: wrap;
  gap: 0.38rem;
  padding: 0.28rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.14);
  background: linear-gradient(180deg, rgba(10,16,28,0.86), rgba(8,14,24,0.82));
}
[data-testid="stRadio"] [role="radiogroup"] label {
  min-height: 2.15rem;
  padding: 0.36rem 0.66rem;
  border-radius: 9px;
  border: 1px solid rgba(232,237,242,0.12);
  background: rgba(18,28,44,0.70);
}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  border-color: rgba(255,122,98,0.62);
  background: linear-gradient(135deg, rgba(255,122,98,0.26), rgba(255,77,45,0.20));
  font-weight: 700;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"],
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--ts-heat-soft), var(--ts-heat)) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 11px !important;
  font-weight: 680 !important;
  padding: 0.55rem 1.05rem !important;
  box-shadow: 0 10px 24px rgba(255,77,45,0.22) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #ff8a73, #ff5535) !important;
  box-shadow: 0 12px 28px rgba(255,77,45,0.30) !important;
}
.stButton > button[kind="primary"]:focus-visible,
.stButton > button[data-testid="stBaseButton-primary"]:focus-visible,
button[kind="primary"]:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible {
  outline: 2px solid rgba(255,122,98,0.6);
  outline-offset: 2px;
}


/* ---- Tables ---- */
[data-testid="stDataFrame"] {
  background: linear-gradient(180deg, rgba(20,29,44,0.72), rgba(15,23,35,0.72));
  border: 1px solid rgba(232,237,242,0.15);
  border-radius: 16px;
  padding: 0.35rem;
  box-shadow: 0 14px 34px rgba(0,0,0,0.34);
}

/* The grid itself */
[data-testid="stDataFrame"] [role="grid"] {
  background: var(--ts-panel-alt) !important;
  color: rgba(232,237,242,0.9) !important;
  border-radius: 12px;
}

/* ---- HTML tables (our controlled dark tables) ---- */
.rc-table {
  margin: 0.45rem 0 1.1rem;
  background: rgba(16,22,34,0.78);
  border: 1px solid var(--ts-border);
  border-radius: 12px;
  padding: 0.6rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  overflow: hidden;
}

.rc-table table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.92rem;
  color: rgba(232,237,242,0.9);
}

.rc-table thead th {
  background: var(--ts-panel);
  color: rgba(232,237,242,0.95);
  text-align: left;
  font-weight: 700;
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid rgba(255,255,255,0.10);
}

.rc-table tbody tr:nth-child(even) td {
  background: rgba(15,22,35,0.92);
}

.rc-table table th:first-child,
.rc-table table td:first-child {
  width: 64px;
  text-align: center;
  font-weight: 800;
  color: rgba(232,237,242,0.94);
}

.rc-table table td:nth-child(2) {
  font-weight: 700;
  letter-spacing: 0.2px;
}

.rc-table tbody tr:hover td {
  background: rgba(255,255,255,0.03);
}

.rc-table { overflow: auto; }

.rc-table table th:first-child,
.rc-table table td:first-child {
  position: sticky;
  left: 0;
  z-index: 3;
  background: var(--ts-panel-alt);
}

.rc-table table th:nth-child(2),
.rc-table table td:nth-child(2) {
  position: sticky;
  left: 64px; /* same as Pos width */
  z-index: 2;
  background: var(--ts-panel-alt);
}

/* Header row */
[data-testid="stDataFrame"] [role="columnheader"] {
  background: var(--ts-panel) !important;
  color: rgba(232,237,242,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}

/* Body cells */
[data-testid="stDataFrame"] [role="gridcell"] {
  background: var(--ts-panel-alt) !important;
  color: rgba(232,237,242,0.9) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Hover */
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
  background: rgba(255,255,255,0.03) !important;
}

[data-testid="stPlotlyChart"] {
  width: 100%;
  max-width: var(--ts-readable-max);
  box-sizing: border-box;
  margin: 0.45rem auto 1.05rem;
  background: linear-gradient(180deg, rgba(16,25,40,0.68), rgba(11,19,30,0.7));
  border: 1px solid rgba(232,237,242,0.13);
  border-radius: 16px;
  padding: 0.35rem 0.35rem 0.15rem;
  box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}

/* Development-over-time is a season-long time series: let it use the full page
   rail instead of the narrower readable cap, so the race axis has room. */
.st-key-ts-dev-over-time [data-testid="stPlotlyChart"] {
  max-width: var(--ts-page-max);
}

/* The movement ladder is a two-column slopegraph, so a shorter measure makes
   one-place changes legible and leaves the Finish-side hover card on-canvas. */
.st-key-ts-biggest-movers {
  max-width: var(--ts-readable-max);
  margin-inline: auto;
}
.st-key-ts-biggest-movers .ts-movement-chart-title {
  max-width: var(--ts-movement-chart-max);
  margin: 1.25rem auto 0;
  padding-inline: 0.4rem;
  color: rgba(232,237,242,0.94);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: 1rem;
  font-weight: 650;
  line-height: 1.3;
}
.st-key-ts-biggest-movers .ts-movement-chart-title span {
  margin-left: 0.4rem;
  color: rgba(139,148,158,0.96);
  font-size: 0.84rem;
  font-weight: 500;
}
.st-key-ts-biggest-movers [data-testid="stCaptionContainer"] {
  max-width: var(--ts-movement-chart-max);
  margin: 0.4rem auto 0;
  padding-inline: 0.4rem;
}
.st-key-ts-biggest-movers [data-testid="stPlotlyChart"] {
  max-width: var(--ts-movement-chart-max);
  margin: 0.65rem auto 1.25rem;
}
[data-testid="stAlert"] {
  border: 1px solid rgba(232,237,242,0.16) !important;
  border-radius: 14px !important;
  background: linear-gradient(180deg, rgba(17,24,38,0.9), rgba(14,21,34,0.88)) !important;
  box-shadow: 0 8px 22px rgba(0,0,0,0.22);
}
hr {
  border: 0 !important;
  border-top: 1px solid rgba(232,237,242,0.14) !important;
  margin: 1.15rem 0 1.15rem !important;
}

/* Shared content surfaces */
.panel {
  background: linear-gradient(180deg, rgba(16,24,38,0.42), rgba(11,18,30,0.32));
  border: 1px solid rgba(232,237,242,0.1);
  border-radius: 18px;
  padding: 1rem 1rem 0.65rem;
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.run-options-note {
  margin-top: 0.2rem;
  color: rgba(232,237,242,0.72);
  font-size: 0.9rem;
}
.surface-card {
  background: linear-gradient(180deg, var(--ts-card-strong), var(--ts-card));
  border: 1px solid var(--ts-card-border);
  border-radius: 18px;
  padding: 1rem 1.05rem;
  box-shadow: 0 12px 28px rgba(0,0,0,0.26);
}
.ts-hero-deck {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 1fr);
  gap: 1rem;
  align-items: stretch;
  margin: 0.25rem 0 1rem;
}
.ts-hero-deck__lead,
.ts-hero-deck__meta {
  min-width: 0;
}
.ts-hero-deck__lead .ts-surface-header {
  height: 100%;
  margin: 0;
}
.ts-hero-deck__meta .ts-stat-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-auto-rows: minmax(0, 1fr);
  gap: 0.9rem;
  margin: 0;
  height: 100%;
}
.ts-hero-deck__meta .ts-stat-card {
  min-height: 0;
  height: 100%;
}
.ts-run-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.9rem;
  margin: 0.35rem 0 0.8rem;
  padding: 0.72rem 0.9rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.12);
  background: linear-gradient(180deg, rgba(16,24,38,0.86), rgba(11,18,30,0.78));
  box-shadow: 0 10px 22px rgba(0,0,0,0.18);
}
.ts-run-summary--success {
  border-color: rgba(72,191,145,0.28);
}
.ts-run-summary--info {
  border-color: rgba(120,167,255,0.24);
}
.ts-run-summary__label {
  color: rgba(139,148,158,0.94);
  text-transform: uppercase;
  font-size: 0.72rem;
  font-weight: 700;
}
.ts-run-summary__value {
  color: rgba(232,237,242,0.94);
  font-size: 0.95rem;
  font-weight: 650;
  text-align: right;
}
.ts-session-overview {
  margin: 0.2rem 0 1rem;
  padding: 0.85rem;
  border: 1px solid rgba(232,237,242,0.12);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(15,23,36,0.78), rgba(10,17,28,0.68));
  box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}
.ts-session-overview__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
.ts-session-overview__label {
  color: rgba(255,122,98,0.96);
  text-transform: uppercase;
  font-size: 0.74rem;
  font-weight: 700;
}
.ts-session-overview__flow {
  color: rgba(139,148,158,0.92);
  font-size: 0.88rem;
  text-align: right;
}
.ts-session-track {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 0.65rem;
}
.ts-session-tile {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 0.72rem;
  align-items: start;
  min-height: 104px;
  padding: 0.78rem 0.82rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.10);
  background: rgba(15,24,38,0.72);
}
.ts-session-tile--result {
  border-color: rgba(72,191,145,0.28);
}
.ts-session-tile__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.65rem;
  height: 1.65rem;
  border-radius: 999px;
  background: rgba(255,122,98,0.16);
  border: 1px solid rgba(255,122,98,0.34);
  color: rgba(255,235,231,0.96);
  font-weight: 700;
  font-size: 0.8rem;
}
.ts-session-tile__title {
  color: rgba(232,237,242,0.96);
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: 0.98rem;
  line-height: 1.25;
}
.ts-session-tile__state {
  margin-top: 0.22rem;
  color: rgba(255,122,98,0.94);
  font-size: 0.84rem;
  font-weight: 650;
}
.ts-session-tile--result .ts-session-tile__state {
  color: rgba(118,211,179,0.94);
}
.ts-session-tile__meta {
  margin-top: 0.34rem;
  color: rgba(139,148,158,0.92);
  font-size: 0.84rem;
  line-height: 1.38;
}
.ts-surface-header {
  margin: 0.2rem 0 1rem;
  padding: 1.2rem 1.25rem 1.1rem;
  border-radius: 16px;
  border: 1px solid rgba(232,237,242,0.12);
  background:
    linear-gradient(135deg, rgba(18,29,45,0.92), rgba(11,18,30,0.84)),
    radial-gradient(circle at top right, rgba(255,77,45,0.12), rgba(255,77,45,0));
  box-shadow: 0 18px 36px rgba(0,0,0,0.28);
}
.ts-surface-header--default {
  border-color: rgba(232,237,242,0.12);
}
.ts-surface-header__eyebrow {
  margin-bottom: 0.45rem;
  color: rgba(255,122,98,0.96);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  font-weight: 700;
}
.ts-surface-header__title {
  margin: 0;
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: clamp(1.45rem, 1.4rem + 0.8vw, 2.3rem);
  line-height: 1.08;
}
.ts-surface-header__summary {
  margin: 0.7rem 0 0;
  max-width: 68ch;
  color: rgba(232,237,242,0.82);
  font-size: 0.98rem;
  line-height: 1.58;
}
.ts-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.8rem;
  margin: 0.2rem 0 1rem;
}
.ts-stat-grid--hero {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.ts-stat-grid--movement {
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.ts-stat-card {
  min-height: 112px;
  padding: 0.95rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.10);
  background: linear-gradient(180deg, rgba(15,24,38,0.94), rgba(11,18,30,0.86));
  box-shadow: 0 14px 26px rgba(0,0,0,0.22);
}
.ts-stat-card--accent {
  border-color: rgba(255,122,98,0.42);
  background:
    linear-gradient(180deg, rgba(28,24,24,0.96), rgba(20,17,22,0.90)),
    radial-gradient(circle at top right, rgba(255,122,98,0.14), rgba(255,122,98,0));
}
.ts-stat-card--warning {
  border-color: rgba(245,183,74,0.35);
}
.ts-stat-card--success {
  border-color: rgba(72,191,145,0.34);
}
.ts-stat-card__label {
  color: rgba(139,148,158,0.96);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.72rem;
  font-weight: 700;
}
.ts-stat-card__value {
  margin-top: 0.45rem;
  color: rgba(232,237,242,0.96);
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: 1.28rem;
  line-height: 1.15;
}
.ts-stat-card__meta {
  margin-top: 0.55rem;
  color: rgba(139,148,158,0.92);
  font-size: 0.9rem;
  line-height: 1.45;
}
.ts-notice {
  margin: 0.55rem 0;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.12);
  background: linear-gradient(180deg, rgba(16,24,38,0.9), rgba(11,18,30,0.82));
  box-shadow: 0 10px 22px rgba(0,0,0,0.18);
}
.ts-notice--info {
  border-color: rgba(120,167,255,0.22);
}
.ts-notice--warning {
  border-color: rgba(245,183,74,0.28);
  background: linear-gradient(180deg, rgba(38,33,18,0.92), rgba(25,22,14,0.84));
}
.ts-notice--success {
  border-color: rgba(72,191,145,0.26);
  background: linear-gradient(180deg, rgba(18,37,28,0.92), rgba(14,25,21,0.84));
}
.ts-notice__label {
  color: rgba(139,148,158,0.94);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 0.7rem;
  font-weight: 700;
}
.ts-notice__body {
  margin-top: 0.28rem;
  color: rgba(232,237,242,0.92);
  font-size: 0.94rem;
  line-height: 1.55;
}
.ts-matchup-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 0.7rem;
  margin: 0.6rem 0 0.15rem;
}
.ts-matchup-card {
  --matchup-accent: rgba(139,148,158,0.72);
  --matchup-fill: rgba(139,148,158,0.42);
  padding: 0.82rem 0.9rem;
  border-radius: 12px;
  border: 1px solid rgba(232,237,242,0.12);
  background: linear-gradient(180deg, rgba(15,24,38,0.92), rgba(10,17,28,0.84));
  box-shadow: 0 10px 20px rgba(0,0,0,0.18);
}
.ts-matchup-card--slight {
  --matchup-accent: rgba(245,183,74,0.84);
  --matchup-fill: rgba(245,183,74,0.55);
  border-color: rgba(245,183,74,0.26);
}
.ts-matchup-card--moderate {
  --matchup-accent: rgba(120,167,255,0.84);
  --matchup-fill: rgba(120,167,255,0.58);
  border-color: rgba(120,167,255,0.28);
}
.ts-matchup-card--clear,
.ts-matchup-card--strong {
  --matchup-accent: rgba(255,122,98,0.9);
  --matchup-fill: rgba(255,77,45,0.68);
  border-color: rgba(255,122,98,0.34);
}
.ts-matchup-card__top,
.ts-matchup-card__meta,
.ts-matchup-card__line {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.ts-matchup-card__top,
.ts-matchup-card__meta {
  justify-content: space-between;
}
.ts-matchup-card__identity {
  display: inline-flex;
  align-items: center;
  min-width: 0;
  gap: 0.42rem;
}
.ts-matchup-card__rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.62rem;
  height: 1.25rem;
  padding: 0 0.28rem;
  border-radius: 999px;
  color: rgba(232,237,242,0.96);
  background: rgba(232,237,242,0.08);
  border: 1px solid rgba(232,237,242,0.14);
  font-size: 0.68rem;
  font-weight: 760;
}
.ts-matchup-card__team {
  color: rgba(139,148,158,0.98);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.68rem;
  font-weight: 760;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ts-matchup-card__tag {
  color: rgba(232,237,242,0.92);
  border: 1px solid var(--matchup-accent);
  border-radius: 999px;
  padding: 0.12rem 0.42rem;
  font-size: 0.68rem;
  font-weight: 720;
  white-space: nowrap;
}
.ts-matchup-card__line {
  margin-top: 0.62rem;
  color: rgba(232,237,242,0.9);
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: 1.02rem;
  line-height: 1.25;
}
.ts-matchup-card__favorite {
  color: rgba(232,237,242,0.98);
  font-weight: 760;
}
.ts-matchup-card__vs {
  color: rgba(139,148,158,0.92);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: 0.78rem;
  font-weight: 650;
}
.ts-matchup-card__meta {
  margin-top: 0.5rem;
  color: rgba(139,148,158,0.94);
  font-size: 0.78rem;
}
.ts-matchup-card__meta--samples {
  margin-top: 0.34rem;
  font-size: 0.72rem;
}
.ts-matchup-card__advantage {
  color: var(--matchup-accent);
  font-weight: 760;
  text-align: right;
}
.ts-matchup-card__bar {
  margin-top: 0.55rem;
  position: relative;
  height: 7px;
  border-radius: 999px;
  background: rgba(232,237,242,0.08);
  overflow: hidden;
}
.ts-matchup-card__bar::after {
  content: "";
  position: absolute;
  top: -2px;
  bottom: -2px;
  left: 50%;
  width: 1px;
  background: rgba(232,237,242,0.35);
}
.ts-matchup-card__bar span {
  position: absolute;
  top: 0;
  left: 50%;
  display: block;
  height: 100%;
  border-radius: 0 999px 999px 0;
  background: var(--matchup-fill);
}
.ts-matchup-card__bar--favorite-left span {
  right: 50%;
  left: auto;
  border-radius: 999px 0 0 999px;
}
.ts-matchup-card__scale {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 0.45rem;
  margin-top: 0.34rem;
  color: rgba(139,148,158,0.94);
  font-size: 0.72rem;
  line-height: 1.2;
}
.ts-matchup-card__scale span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ts-matchup-card__scale span:first-child {
  color: var(--matchup-accent);
  font-weight: 760;
}
.ts-matchup-card__scale span:nth-child(2) {
  text-align: center;
}
.ts-matchup-card__scale span:last-child {
  text-align: right;
}
.ts-matchup-card__samples {
  margin-top: 0.22rem;
  color: rgba(139,148,158,0.78);
  font-size: 0.72rem;
  text-align: right;
}
.ts-stage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.8rem;
  margin: 0.4rem 0 1.15rem;
}
.ts-stage-card {
  position: relative;
  min-height: 118px;
  padding: 0.95rem 1rem 0.9rem;
  border-radius: 18px;
  border: 1px solid rgba(232,237,242,0.10);
  background: linear-gradient(180deg, rgba(15,23,36,0.92), rgba(10,17,28,0.86));
  box-shadow: 0 12px 24px rgba(0,0,0,0.2);
}
.ts-stage-card__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.9rem;
  height: 1.9rem;
  border-radius: 999px;
  background: rgba(255,122,98,0.18);
  border: 1px solid rgba(255,122,98,0.38);
  color: rgba(255,235,231,0.96);
  font-weight: 700;
  font-size: 0.84rem;
}
.ts-stage-card__title {
  margin-top: 0.75rem;
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  font-size: 1rem;
  color: rgba(232,237,242,0.96);
}
.ts-stage-card__state {
  margin-top: 0.35rem;
  color: rgba(255,122,98,0.94);
  font-size: 0.88rem;
  font-weight: 600;
}
.ts-stage-card__meta {
  margin-top: 0.5rem;
  color: rgba(139,148,158,0.9);
  font-size: 0.88rem;
  line-height: 1.45;
}
.section-kicker {
  margin: 0;
  color: rgba(255,122,98,0.92);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.76rem;
  font-weight: 700;
}

/* Hide/neutralize Streamlit spinner/status blocks that show as white bars */
[data-testid="stSpinner"] {
  background: transparent !important;
}
[data-testid="stSpinner"] > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.stCacheStatus, [data-testid="stStatusWidget"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* hide Streamlit footer/menu noise */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Cache + status widgets */
.stCacheStatus, [data-testid="stStatusWidget"] { display: none !important; }

/* Footer variants */
footer, [data-testid="stFooter"] { display: none !important; }
#MainMenu { visibility: hidden !important; }

/* Custom footer on same content rail */
.brand-footer {
  margin-top: 2.6rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(232,237,242,0.12);
  color: rgba(139,148,158,0.95);
  font-size: 0.96rem;
  letter-spacing: 0.005em;
  text-align: left;
}

/* Contact page */
.contact-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 0.3rem;
}
.contact-card {
  background: linear-gradient(180deg, rgba(16,25,40,0.86), rgba(12,19,31,0.78));
  border: 1px solid rgba(232,237,242,0.16);
  border-radius: 12px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.24);
  padding: 1rem 1.1rem;
  min-height: 236px;
}
.contact-card--full {
  margin-top: 1rem;
  min-height: 0;
}
.contact-card h3 {
  margin: 0 0 0.85rem 0;
  font-size: 1.34rem;
  font-family: "Sora", "IBM Plex Sans", sans-serif;
  color: rgba(232,237,242,0.95);
}
.contact-card p {
  margin: 0.2rem 0 0;
}
.contact-card ul {
  margin: 0.55rem 0 0;
  padding-left: 1.28rem;
}
.contact-card li {
  margin: 0.22rem 0;
}
.contact-muted {
  margin-top: 0.72rem !important;
  font-size: 0.93rem;
  color: rgba(139,148,158,0.94);
}
.contact-link-stack {
  display: grid;
  gap: 0.62rem;
  margin-top: 0.2rem;
}
.contact-link-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.8rem;
  border: 1px solid rgba(157,198,255,0.28);
  border-radius: 12px;
  padding: 0.66rem 0.78rem;
  background: rgba(20,31,47,0.64);
  color: #d4e5ff !important;
  text-decoration: none !important;
  transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
}
.contact-link-row:hover {
  border-color: rgba(197,221,255,0.56);
  background: rgba(26,39,58,0.78);
  transform: translateY(-1px);
}
.contact-link-row__label {
  font-weight: 650;
}
.contact-link-row__value {
  color: rgba(232,237,242,0.75);
  font-size: 0.9rem;
}

@media (max-width: 980px) {
  .ts-hero-deck {
    grid-template-columns: 1fr;
  }
  .ts-hero-deck__meta .ts-stat-grid,
  .ts-stat-grid--hero {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .contact-grid {
    grid-template-columns: 1fr;
  }
  .contact-card {
    min-height: 0;
  }
  .ts-surface-header {
    padding: 1rem 1rem 0.95rem;
  }
  .ts-stat-card,
  .ts-stage-card,
  .ts-session-tile {
    min-height: 0;
  }
  .ts-session-overview__head,
  .ts-run-summary {
    align-items: flex-start;
    flex-direction: column;
  }
  .ts-session-overview__flow,
  .ts-run-summary__value {
    text-align: left;
  }
}

@media (max-width: 700px) {
  .ts-hero-deck__meta .ts-stat-grid,
  .ts-stat-grid--hero {
    grid-template-columns: 1fr;
  }
  .ts-stat-grid--movement {
    grid-template-columns: 1fr;
  }
  .st-key-ts-biggest-movers .ts-movement-chart-title span {
    display: block;
    margin: 0.2rem 0 0;
  }
}

</style>
"""
