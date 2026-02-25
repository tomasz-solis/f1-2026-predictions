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
}

/* ---- App background ---- */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(120% 90% at 10% 0%, #101727 0%, #0B111D 38%, var(--ts-graphite) 72%);
    color: var(--ts-soft-light);
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
      linear-gradient(110deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.0) 18%, rgba(255,255,255,0.04) 36%, rgba(255,255,255,0.0) 54%),
      radial-gradient(80% 60% at 90% 8%, rgba(37,71,130,0.14), rgba(11,15,20,0));
    opacity: 0.5;
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
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1240px;
  padding-top: 1.25rem;
  padding-right: 2.6rem;
  padding-left: 2.6rem;
  padding-bottom: 2.2rem;
}

@media (max-width: 1080px) {
  [data-testid="stAppViewContainer"] .main .block-container {
    padding-right: 1.6rem;
    padding-left: 1.6rem;
  }
}

@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
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
[data-testid="stExpander"] {
  margin-bottom: 1.2rem;
  border: 1px solid rgba(232,237,242,0.14) !important;
  border-radius: 14px !important;
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

/* Primary button */
.stButton > button {
  background: linear-gradient(135deg, var(--ts-heat-soft), var(--ts-heat)) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 11px !important;
  font-weight: 680 !important;
  padding: 0.55rem 1.05rem !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #ff8a73, #ff5535) !important;
  box-shadow: 0 10px 24px rgba(255,77,45,0.28);
}
.stButton > button:focus-visible {
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
  background: rgba(16,22,34,0.78);
  border: 1px solid var(--ts-border);
  border-radius: 16px;
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
  background: linear-gradient(180deg, rgba(16,25,40,0.68), rgba(11,19,30,0.7));
  border: 1px solid rgba(232,237,242,0.13);
  border-radius: 16px;
  padding: 0.35rem 0.35rem 0.15rem;
  box-shadow: 0 12px 30px rgba(0,0,0,0.3);
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
  border-radius: 18px;
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
  .contact-grid {
    grid-template-columns: 1fr;
  }
  .contact-card {
    min-height: 0;
  }
}

</style>
"""
