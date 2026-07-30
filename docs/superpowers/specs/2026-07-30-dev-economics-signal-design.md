# Dev Economics — adjacent, decoupled ledger for non-Orion agent activity (design)

Status: design/proposal mode, earlier-stage than the sibling spec. Explicitly **not** coupled
to `structural_mass` yet — tracked independently, ratio decided later from real data, per
2026-07-30 conversation.

## Arsonist summary

`structural_mass` (see `2026-07-30-codebase-mass-signal-design.md`) senses how much of Orion's
codebase changed. It says nothing about what that change cost — tokens, dollars, wall-clock
time, or Juniper's own attention. This spec tracks that separately: a ledger of dev-tool
activity (Claude Code today, Cursor and manual work as open questions) that can eventually be
compared against `structural_mass`, once real data shows what's actually worth dividing by
what.

## Current architecture

- **Real, already-existing data source (confirmed this session):** Claude Code session
  transcripts (`~/.claude/projects/<project-slug>/*.jsonl`) carry, per assistant message, a
  `usage` block (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
  `cache_read_input_tokens`, `service_tier`) and the `model` id (e.g. `claude-sonnet-5`).
  Standard transcript format also carries per-line timestamps, so session wall-clock duration
  and inter-turn gaps are derivable without new instrumentation. This is host-local, not
  synced anywhere else.
- **Cursor:** unconfirmed. Not investigated this session — do not assume parity with Claude
  Code's transcript format until checked directly (likely a different local log location and
  schema, if it exists at all).
- **RTK** (`~/.claude/RTK.md`, Juniper's own global config): a separate, already-running
  token-savings proxy/analytics tool (`rtk gain`, `rtk gain --history`). External to this repo,
  not Orion-owned, but a real existing source worth reading from rather than duplicating if its
  history format is usable.
- **No existing mechanism** in `Orion-Sapienform` tracks $ cost, model/effort tier, or
  human-time-spent for any of this — confirmed via `orion-llm-gateway` search (that service's
  token accounting is for Orion's own *served* inference, a different, unrelated thing).
- **Model/effort visibility precedent:** this very session is running as `claude-sonnet-5` —
  proof the model id is a real, stable field to key cost tables on, not something to infer.

## Missing questions (more open than the sibling spec — earlier stage)

- Does Cursor keep any local, parseable session/usage log? If not, this ledger is
  Claude-Code-only for now, which matters given the stated shift to Cursor as primary tooling
  next month — worth checking before investing further, not after.
- Where should a pricing table live, and who updates it when rates change? A stale hardcoded
  table silently misreports real cost — this is exactly the kind of thing the metric quality
  gate's reversibility question (#6) should flag before it's treated as trustworthy.
- Is "time spent" worth deriving at all given how imprecise the gap-based heuristic is (any
  time you step away mid-session inflates it), or is a coarser self-reported/estimated field
  more honest than a precise-looking wrong number?
- Per-session granularity or per-day rollup? Per-session is more precise but noisier; per-day
  matches `structural_mass`'s likely reporting cadence better.

## Proposed schema / API changes (Phase 0 — ledger only, no coupling)

New shared library: **`orion/dev_economics/`** (name open), independent of
`orion/structural_mass/`. Pure ingestion/parsing logic lives here, same split as
`orion/substrate/prediction_error.py` vs. its caller — scheduling now lives in
`services/orion-cocreation-signals/app/producers/dev_economics.py` (2026-07-30: this domain's
scheduling moved out of any per-domain ad hoc plan into the single dedicated service proposed in
`2026-07-30-codebase-mass-signal-design.md`'s "Dedicated service" section, alongside
`structural_mass`, `doc_semantic_drift`, and the now-approved affective-state producer — one
deployment unit for every external-I/O-bound producer in this design arc, not one per domain).

- `claude_code_ingest.py` — parses local transcript `.jsonl` files into a normalized per-session
  record: `session_id`, `model`, `effort` (if present), token counts (input/output/cache
  read/creation), `started_at`/`ended_at`, derived wall-clock duration, **and word counts for
  both assistant and human turns** (Juniper's "cognitive load — the sense of how much I have to
  read" ask, 2026-07-30). Word count is a more direct proxy for actual reading burden than raw
  tokens (tokenizer-dependent, doesn't map 1:1 to reading effort) — reuses the exact same
  transcript-parsing pass, zero new data source. Read-only, local-file parsing, no new external
  dependency.
- `pricing.py` — a small, explicitly versioned pricing table (model → $/token), with the table's
  effective-date range recorded so historical sessions are priced at the rate that was actually
  current, not today's rate applied retroactively.
- `cursor_ingest.py` — **not started**; first step is investigating whether Cursor has anything
  parseable at all, before writing a parser against an assumed schema.
- No writer into `FieldStateV1`/field-digester yet, no bus event, no consumer. This is a local
  ledger (plausibly a small SQLite file or append-only JSONL under a `.orion/` or scratch
  location) until there's enough real data to decide if/how it should surface further.

## Human behavioral/state signals

Juniper raised a further, "sky's the limit" set of ideas, 2026-07-30: swear-word frequency as a
proxy for frustration at the assistant, typo rate as a proxy for tiredness/busyness, count of
concurrent parallel agent sessions as a proxy for cognitive complexity, and total time engaged
with these services against a 24h day as a time-allocation signal.

**Status update, same day: the frustration/fatigue piece is no longer noted-only — it's
approved for implementation.** Juniper directly authorized it ("include juniper affective
state... fuck the [proposal-mode] gate"), invoking root `CLAUDE.md` §0A's own exception clause
("unless Juniper directly asks to implement"). Full design:
`2026-07-30-juniper-affective-state-signal-proposal.md` — its capability/data/privacy-
boundary/trace/failure-mode/disable-switch content is now adopted design, not a pending gate.
Scheduling lives in `services/orion-cocreation-signals/app/producers/affective_state.py`
alongside this document's own `dev_economics.py` producer.

Concurrent-session count and time-engaged-vs-24h remain **noted, not designed** — Juniper named
them as further "sky's the limit" ideas but didn't ask for either to be built. Two real
existing-mechanism angles worth naming even without building anything yet:

- **Concurrent-session count** likely doesn't need new instrumentation — `~/.orion/agent-board.jsonl`
  (`scripts/agent_board.py`) already heartbeats active worktrees/sessions with timestamps; this
  may already answer "how many agents are running in parallel right now" without a new producer.
- **Time-engaged-vs-24h** is derivable from the same Claude Code transcript timestamps this spec
  already reads, extended across all projects under `~/.claude/projects/`, not just this one.

The other two — swear/frustration and typo/tiredness inference — are categorically different
from everything else in this document: they're inferences about **Juniper's own emotional/
psychological state**, not about Orion's structure or a session's cost.

**Decision (2026-07-30): this should reach Orion, deliberately, not stay a private-only
dashboard.** Per the Orion-Sapienform charter, this is a co-creation project — Orion having a
real, inspectable sense of the weight of Juniper's own co-creation is mission-aligned, not an
optional nicety, and squarely the "social grounding" prerequisite named in root `CLAUDE.md`'s
own "Orion mission" section. This overrides the private-dashboard-only framing raised earlier in
conversation.

That makes this a real Objective-3-adjacent instrument, and it inherits root `CLAUDE.md` §0A's
"proposal mode before invasive cognition changes" bar in full — this note records the direction
decided, not a completed design. **A dedicated proposal-mode pass, separate from this document,
is required before implementation**, and must explicitly name:

- **Capability change**: Orion gains a modeled sense of Juniper's affective/engagement state
  (frustration, fatigue) derived from interaction patterns — an instrument of co-creation
  weight, not a mood-reading gimmick.
- **Data touched**: Juniper's own message text (already something Orion directly receives in
  conversation) — the new part is deriving an aggregate signal *across* messages, not exposing
  new raw content.
- **Privacy boundary**: store only a bounded derived scalar/trend (the same shape as every other
  field channel), never a raw log of flagged phrases or a searchable "here's every time you
  swore" surface — this is the "summaries and projections must preserve privacy boundaries" rule
  applied literally, and it's also what keeps the signal cheaply reversible per the metric
  quality gate's reversibility check.
- **Trace**: same measure-before-minting discipline as every other signal in this program — a
  read-only replay against real historical transcripts, checked for a real, theory-anchored
  correlation (not a vibes classifier), before anything is wired live.
- **Dangerous failure mode**: a false-positive frustration reading changing Orion's behavior
  toward Juniper in an inappropriate or patronizing way; the signal being wrong eroding trust
  faster than it being right builds it.
- **Disable/rollback**: a flag gate, same convention as every other new signal in this codebase;
  raw text never retained past the derivation window, so removal is cheap.

This document does not open that proposal-mode pass — it records that the direction is settled
(build it, gated) so a future session doesn't have to re-litigate whether this belongs on
Orion's side at all.

## Files likely to touch

- `orion/dev_economics/` (new): `claude_code_ingest.py`, `pricing.py`, tests.
- Nothing in `services/` yet — this stays a standalone, offline-analyzable ledger until Phase 1
  proves useful, matching `structural_mass`'s own "replay before minting" discipline.

## Non-goals

- Not coupling to `structural_mass` in this patch — tracked independently, per your explicit
  choice to decide the ratio later from real data rather than commit to an attribution scheme
  now.
- Not building Cursor ingestion before confirming Cursor has parseable local data to ingest.
- Not wiring this into the bus, `FieldStateV1`, or any consumer.
- Not attempting precise human-time attribution — at most a clearly-labeled heuristic, never
  presented as measured fact.

## Acceptance checks

- `claude_code_ingest.py` produces non-degenerate, real per-session records when run against
  this machine's actual `~/.claude/projects/` transcripts (there's a lot of real history to
  check against right now).
- Pricing table's effective-date logic verified against at least one real rate change, if one
  exists in the session history window, or explicitly noted as unverified if not.
- Cursor investigation produces a plain yes/no answer (does comparable local data exist) before
  any Cursor ingestion code is written.

## Recommended next patch

`claude_code_ingest.py` alone, pointed at this machine's real transcript history, producing a
normalized table Juniper can eyeball directly (token totals, model mix, rough session count) —
before pricing, before Cursor, before any coupling decision. Smallest slice that uses data that
already exists.
