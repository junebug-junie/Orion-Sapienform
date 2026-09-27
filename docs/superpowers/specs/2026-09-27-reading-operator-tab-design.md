# Reading operator tab (Hub)

## Problem

Orion's reading pipeline stores rich output per read (Stage 1 handoff: what was
learned, candidate priors, concept candidates, open threads, fetch evidence;
Stage 2 result: summary, priors tested, hops, round trips; journal landings).
Hub exposes only counts and wallet usage (`/world-pulse-read/api/status`), and
no page renders even that. The operator cannot see what a read produced, why a
read was rejected, or act on a stuck/failed read.

Live 2026-09-27 (read-only): 331 queue rows, 13 with a Stage 1 handoff, 4 with
a Stage 2 result, one finished Stage 2 whose output only exists in the journal.

## Design

A standalone `/reading` page, embedded as a Hub tab via iframe (the GPU pool
pattern: `gpu_pool.html` + `gpu_pool_tab.js`).

### Read endpoints (existing router, `/world-pulse-read`)

- `GET /api/reads?phase=&kind=&include_stale=&limit=&offset=` — newest first.
  `phase` ∈ `all|active|done|failed|skipped|with_output`. Stale digest items
  (`last_error = stale_digest_item`) are hidden unless `include_stale=1`.
- `GET /api/reads/{seed_id}` — row, parsed handoff, parsed Stage 2 result,
  durable bindings, aliases, and journal entries matched by the existing
  `world_pulse_read:<trace>` / `world_pulse_read_stage2:<trace>` source refs.

A handoff on a non-`done` row is shown as "rejected / earlier attempt", never
as learning.

### Controls

Guard: the GPU pool CSRF rule (`X-Requested-With: orion-hub` + JSON body). No
operator token (Juniper's call, 2026-09-27).

- `POST /api/reads` `{url, why_now, title}` — builds `ReadingRequestedV1`
  with new `invocation_context="operator"` → `requested_by="juniper"` and calls
  the existing `enqueue_reading` ingress (normalization, DNS validation,
  aliasing, commit-before-publish).
- `POST /api/reads/{seed_id}/cancel` — stage in play is Stage 1 when
  `status ∈ {pending, claimed}`, Stage 2 when `status = done` and
  `stage2_status ∈ {pending, claimed}`.
  - unconsumed durable binding → `POST {HUB_READING_DURABLE_URL}/runs/{run_id}/cancel`;
    the existing worker path (`ReadingCancelled` → `cancel_claim`) finishes it.
  - pending, no binding → skipped with `reading_cancelled_by_operator` directly.
  - claimed, no binding (worker between claim and bind) → 409, retry shortly.
  - No wallet charge in any branch.
- `POST /api/reads/{seed_id}/retry` `{stage: 1|2}` — only terminal rows
  (`failed|skipped`) that are not aliases and have no unconsumed binding for
  that stage. Stage 1 also refuses when another row is active for the same URL
  and when the skip reason is `stale_digest_item` (it would be re-skipped).
  Stage 2 requires Stage 1 `done` and non-empty `read_evidence`. Resets the
  stage to `pending`, attempts 0, error cleared. Spends a normal wallet slot
  when it runs. Previous handoff is kept until a new Stage 1 replaces it.

### Contract change

`ReadingContext` gains `"operator"` (provenance: `requested_by="juniper"`).
`ReadingToolBindingV1` is unchanged, so the model tool can never claim it.

## Non-goals

Editing outputs, promoting priors/concepts into memory by hand, bulk actions,
live streaming (page polls ~15 s), new env keys.

## Acceptance checks

- Disposable-PostgreSQL tests: list phases/stale filter, detail with journal
  and aliases, cancel (pending / bound / claimed-unbound), retry rules, submit
  with operator provenance.
- Route tests: guard (CSRF), error mapping, durable cancel call.
- JS rendering test and a browser smoke that loads `/reading` and opens detail.
- Live: `/world-pulse-read/api/reads` returns real rows after deploy (else
  UNVERIFIED).
