# Orion Hub Standalone Substrate Page

## Goal

Provide a low-blast-radius operator/debug page for substrate and review-loop inspection without expanding the busy main Hub shell.

## Isolation pattern

Implemented as:

- dedicated route: `/substrate`
- dedicated template: `templates/substrate.html`
- dedicated JS bundle: `static/js/substrate.js`

The main `index.html` and `static/js/app.js` are intentionally left unchanged.

## Data-source split

The page calls bounded read-only endpoints under `/api/substrate/*`.

- GraphDB-facing semantic sections:
  - `/api/substrate/overview`
  - `/api/substrate/hotspots`
- SQL-backed operational/control sections:
  - `/api/substrate/review-queue`
  - `/api/substrate/review-executions`
  - `/api/substrate/telemetry-summary`
  - `/api/substrate/calibration`

Each endpoint returns explicit source metadata (`kind`, `degraded`, `query`, `last_refreshed`).

## Operator posture

- read-only only
- bounded query limits
- clear degraded/fallback signaling
- manual refresh from the standalone page

No mutation actions or runtime autonomy expansion were added.

## Forward path

Future phases can replace degraded GraphDB placeholders with live query seams and optionally add richer tabular rendering, while preserving route/template/bundle isolation.
