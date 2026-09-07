# Orion Cockpit POV — Soft HUD turn sighting

**Date:** 2026-09-06
**Status:** Slice A+B landed; Rank-1 pre-motor progress hops live; C stubbed
**Branch intent:** `docs/cockpit-pov-design`
**Related:** Unified Orion turn (`docs/superpowers/specs/2026-07-05-unified-orion-turn-design.md`), fused Turn Trace (`services/orion-hub/scripts/chat_turn_trace_routes.py`), harness step bus (`orion:harness:run:step`)

---

## Arsonist summary

Juniper opens a **Cockpit** from a chat turn and sees the turn from Orion’s helmet: Soft HUD glass, hop rail, scrubber, and a thick inspector. Live turns stream hops as they happen; finished turns rewind the same recording. Today’s Turn Trace stays as a thin summary — Cockpit is the deep dive. The data spine is a **Turn Sighting** timeline (not a prettier paint over the incomplete fused join).

---

## Problem

Turn Trace under Hub chat is too thin: it summarizes stance / “N motor steps” / finalize flags and does not show the breadth of what Orion actually saw (stance input bundle, motor system/prefix, each FCC hop’s tool I/O, offboarding). There is no live mid-turn helmet view and no scrub/rewind of a full recording.

---

## Goals

- Soft HUD modal (pretty, frosted glass / gentle vignette) as Orion’s POV of a unified chat turn
- **Thick** coverage: ingress → association → stance inputs → stance decision → motor boot (full prefix/system prompt) → each FCC motor hop → draft/appraisal → reflect/finalize → offboarding/closure
- Dual surface: helmet tells the story; inspector drawer holds full raw payloads (copyable, searchable)
- **Both** live follow and post-turn playback/rewind
- Keep existing Turn Trace panel in place for now
- Operator Hub surface (not a public multi-user API)

## Non-goals (v1)

- Replacing Turn Trace
- Matching thickness for classic PlanRunner path (unified Orion turn first)
- Public / multi-user cockpit
- Pretty-only demo without a real timeline store
- Keyword-triggered conversational mode changes (out of scope; unrelated)

---

## Product shape

### Entry

- **Cockpit** control on or near the chat turn (near Turn Trace)
- Opens a full-screen Soft HUD modal

### Helmet (visor)

- Soft frosted glass, gentle edge vignette
- Shows the current hop’s short “what Orion saw” line plus a rolling stream of recent hop lines
- Live: advances with the turn
- Scrubbing: freezes to that moment in the recording
- Mid-turn open allowed: join wherever the stream currently is

### Hop rail

- One bead per hop (and visible gap beads when a stage was not recorded)
- Click bead → jump scrubber + load hop in inspector

### Scrubber

- Play / pause / step back / step forward / drag
- Live edge marker; scrubbing behind live pauses follow-mode; **Jump to live** resumes

### Inspector drawer

- Bound to selected hop
- Sections: Summary · Inputs · Prompt/Prefix · Tools · Output · Side-effects
- Full raw JSON for the hop (store keeps full payloads; UI may virtualize scroll but must not silently drop content)

### Turn Trace

- Remains under the chat turn as today’s summary
- Cockpit does not replace it in v1

---

## Architecture

### Turn Sighting timeline (single source of truth)

Every unified turn gets a timeline keyed by `correlation_id`. That timeline feeds:

1. Live cockpit (WS hop appends on the Hub chat session)
2. Rewind (HTTP fetch of stored hops, local scrub)

Do **not** treat the current fused `/api/chat/turn/{id}/trace` join as sufficient for Cockpit thickness. It remains useful for Turn Trace and as a secondary cross-check.

### Hop record (logical contract)

Each hop includes at least:

| Field | Role |
|--------|------|
| `correlation_id` | Turn identity |
| `seq` | Monotonic order in the timeline |
| `ts` | Event time |
| `stage` | Canonical stage id (see below) |
| `title` / `visor_line` | Short helmet text |
| `status` | `started` \| `ok` \| `failed` \| `skipped` \| `gap` |
| `summary` | Structured short fields for inspector Summary tab |
| `raw` | Full operator-debug payload for that hop |
| `producer` | Service / function that emitted the hop |

### Canonical stages

1. **ingress** — user message, attachments metadata, turn intake (real hop; observation molecule publish still deferred)
2. **pre_turn_appraisal** — Hub boundary progress (started / ok / failed / skipped) for repair_pressure etc.
3. **association** — attention / open loops / repair / trajectory slice (emitted as soon as built; hollow-fresh called out when `signal_count=0`)
4. **thought_rpc** — Thought stance RPC started + done (elapsed / failure); Mind quality when present on the reply
5. **mind_enrichment** — Mind quality flags when Hub has them; otherwise honest `mind_details_unavailable`
6. **stance_inputs** — full bundle fed into stance
7. **stance_decision** — proceed / defer / refuse + reasons + felt slice
8. **harness_dispatch** — governor contacted (pre-motor → motor handoff)
9. **motor_boot** — full system/prefix context given to the FCC motor
10. **motor_hop** — each tool/thought step (call + result); many rows per turn
11. **draft_appraisal** — draft text + substrate appraisal
12. **finalize** — reflection, Orion voice pass, compliance
13. **closure** — outcome / post-turn closure / inspectable side-effects

Missing instrumentation shows as an explicit **gap** bead — never a fabricated hop.

**Pre-motor progress (Rank-1, live):** Soft HUD streams honest Hub-boundary hops for dying appraisal / Thought / Mind / felt-state phases while the unified turn waits. Ingress is a real user-intake hop; remaining Slice C thickness is observation-molecule publish + optional thicker attachments — not another ingress gap.

### Producers (emit as the turn runs)

Prefer appending from real path code, reusing existing bus facts where they already exist (e.g. `orion:harness:run:step`, harness run artifacts, thought decision, finalize molecules) rather than a second shadow log of invented summaries.

Likely touch points (implementation plan will pin exact functions):

- `orion/hub/turn_orchestrator.py` — ingress, association handoff, WS mirror into cockpit frames
- Stance / thought path — stance_inputs + stance_decision
- `orion-harness-governor` / `orion/harness/fcc_motor.py` / runner — motor_boot + motor_hop
- `orion/harness/finalize.py` — draft_appraisal, finalize, closure
- `orion-sql-writer` — durable timeline persist
- `services/orion-hub` static JS — Cockpit modal UI

### Storage

- New durable timeline store keyed by `correlation_id` + `seq` (table or equivalent in the same Postgres Hub already reads for turn traces)
- Full `raw` retained for operator debug (same honesty class as unredacted harness turn-trace content)
- Complements `harness_turn_trace` / grammar atlas / thought_decision; does not delete them

### Hub API / live wire

- `GET /api/chat/turn/{correlation_id}/cockpit` → ordered timeline (optionally paged)
- Optional `GET .../cockpit/hops/{seq}` for one fat hop if bodies are paged
- WS frames on the chat session, e.g. `cockpit_hop`, `cockpit_timeline_complete`, scoped by `correlation_id`
- Open mid-turn: HTTP snapshot of hops so far, then follow WS appends
- Open after: HTTP full timeline; scrub client-side

### Failure / honesty

- Store/API down → cockpit shows degraded; Turn Trace may still partially work
- Zero hops → empty cockpit, not a fake “complete”
- Failed stage → hop with `status=failed` + error in `raw`
- No empty-shell success states

---

## UI aesthetic

**Soft HUD (chosen):** frosted glass, soft blue-gray vignette, calm readable type, inspector as the dense surface. Rejected alternatives for v1: Hard HUD (scanline terminal), Cinematic (beauty-first / thinner chrome).

Mockup references (brainstorm session): Soft / Hard / Cinematic stills under `.superpowers/brainstorm/` (gitignored).

---

## Shipping slices (still thick overall)

Slices are delivery cuts, not permission to ship a hollow helmet.

| Slice | Delivers |
|--------|----------|
| **A — Spine + Soft HUD** | Schema + store + Hub API + WS hop frames + Cockpit button/modal (visor, rail, scrubber, inspector). Instrument clean existing signals first (stance decision, existing motor steps, finalize/closure). Gaps visible as gap beads. |
| **B — Thick inputs** | Stance input bundle, full motor_boot prefix/system prompt, association reads |
| **C — Offboarding completeness** | Remaining post-motor side-effects + any missing motor hop kinds |

**Acceptance for “done” of the whole feature (not only Slice A):** every canonical stage either has real hops with raw payloads or an explicit gap that has an owned follow-up; live + rewind both work; Soft HUD is the entry experience.

---

## Privacy / ops

- Operator Hub debug surface only
- Retention: same class as harness turn-trace / grammar traces unless a shorter cockpit TTL is set later
- Do not invent a new public unauthenticated dump of prompts

---

## Tests / evals

**Gate tests**

- Producer emits hop with expected `stage` + required raw keys
- Timeline fetch returns strict `seq` order
- WS frame shape for `cockpit_hop`
- UI smoke: Cockpit control → modal → bead click loads inspector
- Regression: Turn Trace panel still mounts

**Eval / smoke**

- One live unified-turn smoke: open Cockpit mid-turn, confirm hops append; after complete, rewind to motor_boot and confirm prefix present (Slice B+)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Payload size blows up Hub WS / browser | Page fat `raw` via hop GET; virtualize inspector; keep visor lines thin |
| Instrumentation misses the real prompt Orion saw | motor_boot must record the exact prefix assembly site, not a reconstructed guess |
| Duplicate / conflicting logs vs grammar atlas | Timeline is append-only sighting; atlas remains substrate grammar truth |
| Soft HUD ships before thickness | Slice A must show gaps honestly; do not call the feature complete until B/C land |

---

## Files likely to touch (implementation)

- `orion/schemas/` — Turn Sighting / cockpit hop schema + registry
- `orion/bus/channels.yaml` — if a dedicated persist/stream channel is added
- `orion/hub/turn_orchestrator.py` — emit/mirror hops + WS
- `orion/harness/*` / `services/orion-harness-governor/` — motor_boot / motor_hop
- `orion/harness/finalize.py` — finalize / closure hops
- `services/orion-sql-writer/` — persist timeline
- `services/orion-hub/scripts/` — cockpit API routes
- `services/orion-hub/static/js/` — Cockpit Soft HUD modal
- Hub tests + harness/sql-writer tests

Exact file list locked in the implementation plan.

---

## Acceptance checks

1. From a Hub unified-turn chat bubble, Cockpit opens Soft HUD modal
2. Live: hops appear without refresh while the turn runs
3. After: scrubber can rewind to an earlier hop; inspector shows that hop’s raw payload
4. motor_boot (once Slice B lands) shows the real system/prefix text Orion was given
5. Missing stages show as gaps, not silent omission
6. Turn Trace still present and functional

---

## Recommended next step

Execute **Slice B** plan: `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b.md`. After B lands, invoke **writing-plans** for Slice C (`docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md`).
