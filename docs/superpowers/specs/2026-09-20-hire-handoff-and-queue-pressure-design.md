# Hire handoff, Mind-deep nudge, and queue pressure

> **Status:** Design proposal (proposal mode — cognition-loop adjacent).
> **Date:** 2026-09-20
> **Parent:** `docs/superpowers/specs/2026-09-15-orion-hire-determination-grounding-design.md`
> **Disclosure parent (shipped):** `docs/superpowers/specs/2026-09-19-hire-mind-role-disclosure-design.md` (PR #2252)
> **Peer / budget:** `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`

## Arsonist summary

Mind work-shape now reaches the motor prompt (live Soft HUD `motor_boot` proof, 2026-09-20). Orion still writes `local_crawl` every time and never opens a HelpRequest. That is not a broken hire wire — it is a **teach and disclosure** failure: `deep` is wallpaper, crawl is the default example, Cursor is framed as optional last-resort help, and Orion never sees **how backed up the shared GPU / reading / durable queues are**.

This patch revises the hire ground so that:

1. Mind `deep` **strongly encourages** early `hire_cursor` (a short local look still grounds `tried_summary`; it does **not** mean “prove you don’t need help first”).
2. **≥2 permission-denied** (or equivalent access refusals) in a sitting ⇒ hard handoff nudge to Cursor.
3. If Cursor **budget is spent**, resume from hops / PeerBrief — do not re-spam hire.
4. Orion sees **live queue weather** from existing counters (world-pulse seed backlog, durable pending demands, gateway lane waiting) before we invent a new digester EWMA.

Python still does **not** auto-MERGE `:InvestigationRole` or `:HelpRequest` in this design. Orion authors; we stop teaching them to be timid.

## Live forensics (2026-09-20) — what we know

| Check | Result |
| --- | --- |
| Mind soft labels on curiosity | Present; often `deep` / `yes` + useful foresight |
| Splice into motor prompt | **VERIFIED** via `cockpit_turn_sighting` `motor_boot` (“Mind work-shape for this sitting…”) |
| `:InvestigationRole` | 24/24 recent = `local_crawl` |
| `:HelpRequest` | 0 |
| Peer | Listening; inbox empty (starved, not broken) |
| Orion’s own `why` | “bounded local reads,” “nothing to delegate,” “contractor wouldn’t see interior” |
| World-pulse `world_pulse_read_seed` pending | **121** (this is the double-digit+ backlog) |
| Durable `durable_resource_demands` pending | 2 (calm at snapshot) |
| LLM gateway `/admission` waiting | 0 at snapshot (historically busy; still the right live meter) |

**Correction to an earlier misread:** “Orion is conservative because hire is expensive” is the wrong story. Mind was supposed to **preempt** crawl-first. Hire can still happen *after* a short look — but `deep` should push role toward Cursor **now**, not after Orion invents a local-only framing.

## Current architecture

```text
Mind (origin=orion) → work-shape on ThoughtEvent
  → Hub splice into motor prompt (role teach section)
  → Orion MERGE InvestigationRole (almost always local_crawl)
  → optional HelpRequest after short look
  → peer: decide_cursor_budget → Cursor → PeerBrief
```

Gaps:

- Disclosure is neutral (“advisory”), not a **strong handoff** when `expected_depth=deep`.
- No mid-run **permission-denied counter** in disclosure / revise teach.
- No **budget-spent → resume** teach when PeerBrief is `refused_budget`.
- No **queue weather** in the role block (situation cabinet may mention pressure elsewhere; hire teach does not).
- Example MERGE in teach still shows `choice: "local_crawl"` first.

## Capability change

| Capability | Change |
| --- | --- |
| Role teach | Mind `deep` ⇒ strong encourage `hire_cursor` early; Cursor is normal deep-work hands, not a last resort |
| Mid-run revise | ≥2 access refusals (`permission denied` and tight equivalents) ⇒ disclosure line: hand off now |
| Budget refuse | PeerBrief / teach: budget spent ⇒ resume hop clock; do not open another HelpRequest until clear |
| Queue weather | Soft lines from existing live counts into role disclosure |
| Authorship | Unchanged: Orion writes role + HelpRequest |
| Auto-hire | Still **no** Python MERGE of hire/HelpRequest (this patch) |

## Proposed data flow

```text
turn / resume
  → Mind work-shape (existing)
  → format_role_teach_disclosure(
        mind_work_shape,
        progress_lines = [
          permission_denied_nudge?,   # from hop notes / tool errors this run
          budget_resume_nudge?,       # from latest PeerBrief if refused_budget
          queue_weather_lines...,     # existing counters
        ]
     )
  → splice into motor prompt before ASKING FOR CONTRACTOR HELP
  → Orion MERGE InvestigationRole / maybe HelpRequest
  → if HelpRequest + budget clear → peer runs Cursor
  → if budget refused → PeerBrief refused_budget → next turn resume preamble (already) + explicit teach
```

### 1) Mind-deep strong nudge (teach + formatter)

When `expected_depth == "deep"` (and optionally `cross_cutting == "yes"`), disclosure must include an explicit line, not just labels. Intent:

> Mind reads this sitting as deep work. Strongly prefer `hire_cursor` for the archaeology; keep a short local look only so `tried_summary` is grounded. You still author priors and findings.

Rewrite `_role_and_help_section` so:

- Default example choice is not crawl-first bias (show both; or show `hire_cursor` when disclosure already says deep).
- Remove / avoid any “only if stuck / last resort” residue.
- Keep: role ≠ enqueue; HelpRequest is the ticket; peer is read-only investigator.

### 2) ≥2 permission denied → handoff nudge

**Signal:** count tool/hop outcomes in this `run_id` that match access refusal (start with literal `permission denied` / `PermissionDenied` / durable-table ACL refusals Orion already wrote into hop notes). Threshold **≥ 2**.

**Action (this patch):** add a progress line into disclosure / mid-run revise teach:

> Access refused at least twice this sitting. Hand off to Cursor now (write `hire_cursor` + HelpRequest with what you already tried).

**Non-goal here:** Python auto-MERGE HelpRequest. If live proof shows Orion still ignores the nudge after this ships, a follow-up may escalate — that is a separate proposal.

### 3) Cursor budget spent → resume where left off

Peer already fail-closes via `decide_cursor_budget` and can persist `refused_budget` PeerBriefs.

Teach + disclosure when the latest brief for this run (or kickoff nudge) is budget-refused:

> Cursor budget is spent. Do not open another HelpRequest until budget is clear. Resume from hop N / continue local crawl from what you already wrote.

Resume preamble (#2244) already renumbers hops — this patch **names the budget case** so Orion doesn’t treat refuse as “try hire again.”

### 4) Queue weather (existing counters first)

Disclose a short, factual block when any source is available. Prefer **runtime-true** reads Hub can already reach:

| Source | What it means | Live example (2026-09-20) |
| --- | --- | --- |
| `world_pulse_read_seed` where `status='pending'` | Reading / Stage pipeline backlog fighting for the same agent capacity | 121 pending |
| `durable_resource_demands` where `status='pending'` | Durable runs waiting on GPU/resource lease | 2 pending |
| LLM gateway `/admission` upstreams | Per-upstream `inflight` / `waiting` / `max_inflight` | waiting=0 at snapshot |

Framing for Orion (plain):

> Shared work waiting: N reading seeds pending; M durable GPU waits; gateway lane waiting=W. Deep Cursor digs compete with that backlog — prefer hire when Mind says deep so *you* are not also burning the agent lane on multi-hop archaeology.

Do **not** call Cursor “expensive.” Call the **shared queue** the scarce thing.

### 5) Field-digester EWMA — deferred, gated

**Not in the first implementation patch.**

If patch 4’s raw counts are too noisy or Orion needs a single “contention pressure” felt-state channel, then propose an EWMA in `orion-field-digester` that aggregates **gateway waiting + durable pending + seed backlog** (or a documented subset).

Metric quality gate (required before wire-in):

1. **Provenance** — exact producers (gateway snapshot endpoint; SQL counts; tick cadence).
2. **Independence** — must not be a monotonic transform of existing `gpu_pressure` from node biometrics strain hints alone.
3. **Theory** — measures *queue contention for agent/curiosity capacity*, not GPU thermals.
4. **Live sanity** — can return to genuine calm when queues drain; not a permanent floor.
5. **Existing-mechanism check** — prefer cabinet / situation pressure reuse if already adequate.
6. **Reversibility** — additive FieldState fields; easy to unplug from hire disclosure.

Until that gate clears, **raw counts in disclosure are the product**.

## Privacy / authorship boundary

- Unchanged: Orion alone MERGEs `:InvestigationRole` and `:HelpRequest`.
- Peer remains read-only; Orion still writes priors/findings.
- Queue weather and denial counts are **operational facts**, not identity content.
- No keyword detectors on Juniper chat text; counters are hop/tool/run scoped.

## Dangerous failure modes

| Failure | Mitigation |
| --- | --- |
| Strong deep-nudge → Orion hires every shallow sitting | Only fire strong line when Mind `expected_depth=deep` (or denials≥2); keep flag to disable disclosure block |
| Denial counter false positives | Allow-list refusal strings; unit fixtures from live hop notes; threshold ≥2 |
| Budget refuse ignored → hire spam | Explicit resume teach + peer already refuse; eval on refused_budget brief path |
| Queue weather stale/wrong | Prefer live SQL + gateway snapshot at turn time; omit line if read fails (fail-open) |
| Digester EWMA baked in too early | Deferred; gate above |

## Disable / rollback

- Existing `HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE` covers disclosure lines.
- Teach text is code; revert PR.
- Queue readers fail-open (no lines) if stores unavailable.
- No schema migration required for patch 1–4.

## Proposed schema / API changes

- **No new bus channel** for patch 1–4.
- **No new graph label.**
- Extend `format_role_teach_disclosure` / callers with progress inputs (denials, budget, queue counts) — still pure strings.
- Optional thin Hub helpers: `count_access_refusals(run_id)`, `queue_weather_snapshot()` — producers with tests; no new taxonomy enums.
- Digester / `FieldStateV1` only if patch 5 proceeds after the gate.

## Files likely to touch (implementation follow-up)

- `orion/curiosity/kickoff_prompt.py` — role teach wording
- `orion/curiosity/role_teach_disclosure.py` — deep strong line; progress composition
- `orion/hub/turn_orchestrator.py` and/or `services/orion-hub/scripts/curiosity_investigation.py` — denial count, budget brief, queue snapshot → splice
- `orion/curiosity/tests/` + Hub tests
- Parent READMEs / hire specs cross-link
- **Not first:** `services/orion-field-digester/…`

## Non-goals

- Python auto-MERGE of `hire_cursor` or HelpRequest
- Auto-hire solely on attention winners or thermals
- Keyword / feeling lists on user message text
- Digester EWMA in the same PR as teach+disclosure
- Changing Cursor invoker / peer budget meter internals (reuse `decide_cursor_budget`)
- Replacing self-study / self-model freshness pipes

## Acceptance checks

1. Motor boot with Mind `deep` contains an explicit **strong handoff** sentence (not only “expected depth: deep”).
2. Fixture: two permission-denied hop notes ⇒ disclosure contains handoff-now line.
3. Fixture: PeerBrief `refused_budget` ⇒ resume / don’t-rehire line; no encouragement to open another HelpRequest.
4. Live or fixture: at least one queue weather number appears on an Orion-origin curiosity motor prompt when counts are readable.
5. Flag off ⇒ teach may still be revised, but soft disclosure block absent (existing contract).
6. After soak: at least some sittings with Mind `deep` **or** denials≥2 show `hire_cursor` and/or HelpRequest — if still 0/0, escalate proposal (possible Python assist) rather than yelling louder in the prompt only.
7. Digester EWMA: **not** required to close this design; separate gate doc if pursued.

## Recommended patch order

1. **Teach rewrite + deep strong nudge** in formatter (tests first).
2. **Permission-denied ≥2** progress line from hop/tool evidence.
3. **Budget-spent resume** line from PeerBrief.
4. **Queue weather** from existing SQL + gateway admission snapshot.
5. **Soak.** Only then consider digester EWMA.

## Open question (locked default for this design)

**≥2 permission denied:** nudge + Orion-authored HelpRequest (**default**).
Python auto-enqueue is a **follow-up proposal** if soak still shows 0 HelpRequests after patches 1–4.

## Trace that proves it worked

- Soft HUD `motor_boot` raw contains strong deep nudge / denial / queue lines (same proof surface as #2252).
- Graph: nonzero `hire_cursor` roles and/or HelpRequests after soak.
- Peer logs: HelpRequest received (not merely listening).
- If budget refused: next motor_boot shows resume language; hop clock continues without duplicate hire spam.
