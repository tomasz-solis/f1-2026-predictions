---
name: Trackside Labs
description: The pit-wall instrument for independent F1 forecasting — a calm graphite chassis with one hot readout.
colors:
  heat: "#FF4D2D"
  heat-soft: "#FF7A62"
  graphite: "#0B0F14"
  panel: "#111826"
  panel-alt: "#0F1623"
  soft-light: "#E8EDF2"
  steel: "#8B949E"
  success: "#48BF91"
  warning: "#F5B74A"
  info: "#78A7FF"
  link: "#9DC6FF"
  ck-pre: "#9DC6FF"
  ck-fp1: "#3671C6"
  ck-fp2: "#F4A7A3"
  ck-fp3: "#FF4D2D"
  ck-sq: "#F5B74A"
  ck-sprint: "#48BF91"
  ck-q: "#C084FC"
  ck-r: "#E8EDF2"
typography:
  display:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "clamp(1.45rem, 1.4rem + 0.8vw, 2.3rem)"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "normal"
  headline:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "2.1rem"
    fontWeight: 650
    lineHeight: 1.15
    letterSpacing: "0.015em"
  title:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "1.28rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "0.008em"
  body:
    fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif"
    fontSize: "1.03rem"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "0.01em"
  label:
    fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif"
    fontSize: "0.72rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.09em"
rounded:
  sm: "9px"
  md: "12px"
  lg: "16px"
  xl: "18px"
  pill: "999px"
spacing:
  xs: "0.4rem"
  sm: "0.65rem"
  md: "1rem"
  lg: "1.25rem"
  xl: "2.6rem"
components:
  button-primary:
    backgroundColor: "{colors.heat}"
    textColor: "#FFFFFF"
    rounded: "11px"
    padding: "0.55rem 1.05rem"
  button-primary-hover:
    backgroundColor: "{colors.heat-soft}"
    textColor: "#FFFFFF"
  nav-pill:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.sm}"
    padding: "0.42rem 0.88rem"
  nav-pill-active:
    backgroundColor: "{colors.heat}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: "0.42rem 0.88rem"
  surface-card:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.xl}"
    padding: "1rem 1.05rem"
  stat-card:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.md}"
    padding: "0.95rem 1rem"
  input:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.md}"
    padding: "0.5rem 0.75rem"
---

# Design System: Trackside Labs

## 1. Overview

**Creative North Star: "The Pit Wall"**

Trackside Labs looks like the race engineer's screen on the pit wall. A calm, near-black graphite chassis (#0B0F14) fills the frame; information sits in layered translucent panels lit from a faint ambient glow; and a single hot color — Brake Heat orange (#FF4D2D) — flares only where a decision or a live state actually lives. The data leads. Energy is earned, never sprayed across the surface. This is a serious analytics instrument that happens to be about racing, not a racing site with charts bolted on.

The system carries race-paced momentum through motion and the heat accent, but its core is disciplined and evidence-first, mirroring the product's one claim: *the F1 forecast that knows how much to trust each signal*. Depth comes from layered glass over a lit floor — panels float above a background wash of radial gradients (a warm heat bloom top-left, a cool blue bloom top-right) with soft, deep shadows and a blurred sticky header. Nothing is flat, but nothing shouts. Confidence without false confidence extends to the surface itself: it states what it knows and gives uncertainty room to breathe.

It explicitly rejects two aesthetics. It is **not a betting or gambling site** — no neon odds, no garish hype, none of the sportsbook grammar. And it is **not a generic SaaS dashboard** — no cream-and-blue template, no interchangeable hero-metric tiles, no anonymous AI-startup polish. It must also read as visibly independent, never borrowing the authority of official F1 or team liveries.

**Key Characteristics:**
- Near-black graphite base with one signature hot accent, used sparingly.
- Layered translucent panels over an ambient lit-floor background; soft deep shadows, backdrop blur.
- Two-family type system: Sora for engineered headings, IBM Plex Sans for dense, legible UI and prose.
- A dedicated checkpoint color scale that carries the model's evidence timeline (PRE → race) through every chart.
- Dark, high-contrast, WCAG 2.1 AA as the floor — legibility wins whenever it conflicts with energy.

## 2. Colors

A near-black dark theme built from cool graphite and steel neutrals, disciplined to a single warm accent, with a small semantic set for state and a distinct multi-hue scale reserved for data.

### Primary
- **Brake Heat** (#FF4D2D): The one voice. Primary actions, the active navigation pill, live-state indicators, section kickers, and the FP3/decision moment on charts. This is the glow of a hot brake disc — motorsport-specific, always meaningful.
- **Heat Soft** (#FF7A62): The lighter end of the accent, used for hover brightening, gradient tops, and softer accent borders.

### Neutral
- **Graphite** (#0B0F14): The body background. The pit-wall chassis everything sits on.
- **Panel** (#111826) and **Panel Alt** (#0F1623): The two dark surface layers for cards, tables, inputs, and panels — the second neutral layer that separates content from chassis.
- **Soft Light** (#E8EDF2): Primary ink. Headings and high-emphasis text (typically at 0.90–0.96 opacity over dark).
- **Steel** (#8B949E): Muted text — labels, meta, disclaimers, secondary captions. Never drops so low it fails AA on body copy.

### Tertiary — Semantic States
- **Success Green** (#48BF91): Completed sessions, positive movement, result-confirmed tiles.
- **Warning Amber** (#F5B74A): Caution notices, sprint-qualifying, slight-edge matchups.
- **Info Blue** (#78A7FF) / **Link Blue** (#9DC6FF): Informational notices and hyperlinks. Link blue is the interactive text color.

### Named Rules
**The One Voice Rule.** Brake Heat is the only warm color on the surface and appears on roughly ≤10% of any screen — primary action, current selection, and live state. Its rarity is what makes it read as "pay attention here." Never use it as a background fill for large areas or as decoration.

**The Two-Floor Rule.** Content never sits directly on graphite. It rests on Panel / Panel Alt surfaces, which rest on the lit graphite floor. Two layers of dark, never one.

**The Data Scale Is Sacred.** The checkpoint scale — PRE #9DC6FF, FP1 #3671C6, FP2 #F4A7A3, FP3 #FF4D2D, SQ #F5B74A, SPRINT #48BF91, Q #C084FC, R #E8EDF2 — encodes the evidence timeline and is used *only* in data visualization. Never repurpose these hues for UI chrome, and never recolor a checkpoint; the color IS the identity of that session across every chart.

## 3. Typography

**Display Font:** Sora (with 'IBM Plex Sans', sans-serif)
**Body Font:** IBM Plex Sans (with 'Segoe UI', sans-serif)

**Character:** A two-family pairing on a real contrast axis — Sora is a geometric, slightly technical display sans that gives headings an engineered, instrument-panel confidence; IBM Plex Sans is a humanist workhorse tuned for dense labels, tables, and prose. Sora states, Plex explains.

### Hierarchy
- **Display** (Sora, 600, clamp(1.45rem, 1.4rem + 0.8vw, 2.3rem), line-height 1.08): Surface headers at the top of a page or major section. The only place fluid sizing is allowed.
- **Headline** (Sora, 650, 2.1rem, letter-spacing 0.015em): The main page header / brand-level title.
- **Title** (Sora, 600, ~1rem–1.28rem, line-height 1.15): Card titles, stat values, session-tile titles. Sora carries emphasis at small sizes.
- **Body** (IBM Plex Sans, 400, 1.03rem, line-height 1.6): Prose, summaries, notice bodies. Capped at ~62–68ch for readable measure.
- **Label** (IBM Plex Sans, 700, 0.72rem, letter-spacing ~0.08–0.12em, UPPERCASE): Kickers, stat-card labels, run-summary labels, the section kicker in Brake Heat.

### Named Rules
**The Sora-For-Statement Rule.** Sora is for headings, titles, stat values, and kickers — moments of statement. It never runs as body copy or fills tables. IBM Plex Sans owns everything dense and everything read in paragraphs.

**The Fixed-Scale Rule.** Outside the single Display header, type sizes are fixed rem, not fluid. Users read this at consistent DPI; a heading that shrinks inside a panel looks worse, not better.

## 4. Elevation

Layered glass over a lit floor. Depth is real here: translucent panels (backgrounds in the 0.7–0.9 alpha range) float above an ambient background of radial gradients and faint vertical rule-lines, separated by soft, deep, low-opacity shadows and, on the sticky header, a 10px backdrop blur. Surfaces are never truly flat, but shadows stay diffuse and dark rather than crisp — the mood is a dim garage, not a bright card grid.

### Shadow Vocabulary
- **Card ambient** (`box-shadow: 0 12px 28px rgba(0,0,0,0.26)`): The default lift for surface cards, stat cards, and panels.
- **Header lift** (`box-shadow: 0 18px 36px rgba(0,0,0,0.28)`): The taller elevation for surface-header blocks at the top of a section.
- **Table lift** (`box-shadow: 0 14px 34px rgba(0,0,0,0.34)`): The deepest shadow, anchoring dense data tables.
- **Heat glow** (`box-shadow: 0 10px 24px rgba(255,77,45,0.22)`): The one colored shadow — under primary buttons and active accent elements only. This is the accent's halo, not a neutral shadow.

### Named Rules
**The Diffuse-Dark Rule.** Shadows are large-radius, low-opacity, and pure black — 24–36px blur at 0.2–0.34 alpha. Never small, tight, or high-contrast; a 2px hard drop shadow reads as a 2014 app and is forbidden.

**The Colored-Shadow-Is-Accent-Only Rule.** The only non-black shadow allowed is the Brake Heat glow, and only under interactive accent elements. No colored glows on neutral cards.

## 5. Components

### Buttons
- **Shape:** Softly rounded (11px radius).
- **Primary:** A Brake Heat gradient (Heat Soft → Heat), white text, weight ~680, padding 0.55rem 1.05rem, carrying the heat-glow shadow. This is the "go" control — the forecast trigger.
- **Hover / Focus:** Hover brightens the gradient and deepens the glow (to rgba(255,77,45,0.30)); `:focus-visible` shows a 2px Brake Heat outline at 0.6 alpha, offset 2px. Focus is always visible — required for AA.

### Cards / Containers
- **Corner Style:** 16–18px on major surfaces (surface cards, surface headers), 12px on stat/session tiles.
- **Background:** Translucent Panel / Panel Alt gradients (top-lighter to bottom-darker), never a solid flat fill.
- **Shadow Strategy:** Card ambient by default; Header lift for surface headers (which also carry a faint Brake Heat radial glow in the top-right corner).
- **Border:** A hairline `1px solid rgba(232,237,242,0.10–0.16)` — the ghost border that defines edges without a hard line.
- **Internal Padding:** ~1rem on cards, ~1.2rem on surface headers.

### Inputs / Fields
- **Style:** Dark gradient fill (Panel Alt), 12–14px radius, hairline border rgba(232,237,242,0.14–0.18), with a subtle inset top highlight.
- **Values & Placeholders:** Value text at ~0.94 alpha Soft Light; placeholders at 0.45 alpha — still legible, never invisible.
- **Focus:** Border shifts toward accent / lightens; keep a visible focus treatment on every field for AA.

### Navigation
- **Style:** A segmented-control pill group in a dark gradient trough with a hairline border. Top nav is horizontal and scrolls rather than wrapping on mobile; the sidebar variant stacks vertically full-width.
- **States:** Default pills are Panel-dark with muted Soft Light text; hover lifts 1px and borders warm toward heat; the **active** pill is a Brake Heat gradient with white text and heat glow. In the sidebar, the active item instead uses an inset Brake Heat left-edge bar (a nav indicator, not a card stripe).
- **Mobile:** Nav fills width; tap targets ≥2.35rem tall.

### Charts (Signature)
- Plotly figures render on **transparent** paper and plot backgrounds so they sit directly on the panel glass. Grid lines are Soft Light at very low alpha (rgba(232,237,242,0.08–0.18)); chart font is Soft Light (#E8EDF2) at ~13–14px. Each chart lives inside a bordered, 16px-radius panel capped at ~980px for a readable measure. Series color comes exclusively from the checkpoint scale.

### Matchup Card (Signature)
- A team head-to-head card with a center-anchored advantage bar that fills left or right from the midline. Advantage strength maps to color: steel (even) → amber (slight) → info blue (moderate) → Brake Heat (clear/strong). The bar magnitude and hue together read the confidence of the edge at a glance.

## 6. Do's and Don'ts

### Do:
- **Do** keep Brake Heat (#FF4D2D) rare — primary action, current selection, and live state only, ≤10% of a screen. Its scarcity is the point.
- **Do** rest all content on Panel / Panel Alt surfaces over the graphite floor; two layers of dark, never content directly on #0B0F14.
- **Do** use Sora for headings, titles, stat values, and kickers; IBM Plex Sans for everything dense and everything in paragraphs.
- **Do** use large, diffuse, dark shadows (24–36px blur, 0.2–0.34 alpha) and reserve the colored heat glow for interactive accent elements.
- **Do** keep the checkpoint color scale exclusively for data, one fixed hue per session code, PRE → R.
- **Do** hold WCAG 2.1 AA: body ≥4.5:1, visible focus rings on every control, a reduced-motion alternative for every transition. Legibility wins over energy.

### Don't:
- **Don't** drift toward a **betting or gambling site** — no neon odds, no garish hype CTAs, no sportsbook framing around the probabilities.
- **Don't** ship a **generic SaaS dashboard** — no cream-and-blue template, no interchangeable hero-metric tiles, no identical icon-heading-text card grids, nothing that reads as anonymous AI-startup polish.
- **Don't** borrow **official F1 or team branding** — no series/team liveries, marks, or color codes. The look must read as independent.
- **Don't** spray the heat accent as a background fill, gradient-text, or decoration; and never use it as a neutral shadow color.
- **Don't** use tight, high-contrast drop shadows. If a shadow looks like a 2014 app, the blur is too small and the color is too hard.
- **Don't** set body copy or table content in Sora, and don't let muted steel text fall below AA contrast to save "elegance."
