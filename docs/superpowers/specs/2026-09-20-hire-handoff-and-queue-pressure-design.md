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
4. Orion sees **one queue-pressure score (0–10, EWMA-normalized)**, not raw counts. Raw counts (`121` vs `2` vs `0`) mean nothing to Orion without a sense of what's normal — a single scored number Orion can act on (*"pressure is 8/10, elevated"*) does. **Revised 2026-09-20 (Juniper): go straight to the score. Do not ship raw-count disclosure first.** The score is an **official field-digester / FieldState metric** — not more Hub-local Redis EWMA shadow state. It must clear CLAUDE.md §0A **and** land in the metric semantic layer so the CI static gates (`check_metric_lineage.py --gate`, `check_definition_drift.py --gate`, `check_inner_state_registry.py`) actually see it.

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
| Queue pressure | Official digester `FieldStateV1` score (0–10, EWMA-normalized); Hub **reads** it for role disclosure — does not own the meter |
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
          queue_pressure_line?,       # read official FieldState score + driving source
        ]
     )
  → splice into motor prompt before ASKING FOR CONTRACTOR HELP
  → Orion MERGE InvestigationRole / maybe HelpRequest
  → if HelpRequest + budget clear → peer runs Cursor
  → if budget refused → PeerBrief refused_budget → next turn resume preamble (already) + explicit teach
```

**Score producer (official path — not Hub):**

```text
seed / durable / gateway counters (live reads)
  → orion-field-digester tick (EWMA baselines + max() score)
  → FieldStateV1 fields (+ glossary / inner-state / metric lock)
  → consumers: hire disclosure (Hub), later curiosity-supervisor (reserved)
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

### 4) Queue pressure score (official digester metric — EWMA, not Hub shadow state)

**Revised 2026-09-20 (Juniper, twice):**

1. Skip raw-count disclosure. A raw count means nothing to Orion without a sense of what's normal — `2` and `10000` both need context Orion does not have. A single scored number does: *"pressure is 8/10"* is actionable; *"2 pending"* is not.
2. **Do not put the EWMA in Hub.** No `orion:hire:queue_pressure:ewma:*` Redis keys, no Hub scheduler tick owning baselines, no one-off helper that only hire disclosure can see. That is more shadow instrumentation living in Hub. The score is an **official** `orion-field-digester` / `FieldStateV1` instrument — same class of thing as `sustained_load_pressure` / `tension_deviation_pressure`: produced on the digester tick, persisted on field state, registered so the metric semantic layer and CI static gates can reject a bad or unfinished wire.

**Sources (still the right three counters; digester reads them, Hub does not re-implement):**

| Source | What it means | Live example (2026-09-20) |
| --- | --- | --- |
| `world_pulse_read_seed` where `status='pending'` | Reading / Stage pipeline backlog fighting for the same agent capacity | 121 pending |
| `durable_resource_demands` where `status='pending'` | Durable runs waiting on GPU/resource lease | 2 pending |
| LLM gateway `/admission` upstreams | Per-upstream `inflight` / `waiting` / `max_inflight` | waiting=0 at snapshot |

**Why EWMA, not a fixed threshold:** each source has its own natural scale and its own normal range (the seed backlog runs in the hundreds; durable demand runs in the single digits) — one fixed cutoff can't mean the same thing for both. An EWMA baseline gives each source its own rolling "what's typical for you lately," so the score answers *"is this source worse than its own normal right now,"* not *"is this number big."*

**Shape of the computation** (lives in digester digestion code + `FieldStateV1`; pattern sibling to `sustained_load_pressure`):

1. For each of the 3 sources, maintain a decayed rolling baseline on **field state** (additive EWMA fields — same family as `dimension_precision_ewma*` / tension baselines — not Hub Redis). Updated on the digester tick cadence that already advances `FieldStateV1` (exact half-life tunable after gate item 4; proposed starting point ~24h half-life).
2. At score time: `ratio = current_count / max(ewma_baseline, floor)` (small floor, e.g. `1`, avoids divide-by-near-zero when a source is usually near 0).
3. Per-source sub-score: `clip(10 * (ratio - 1) / 4, 0, 10)` — ratio 1.0x → 0, ratio 5x → 10, linear between. Explainable on purpose.
4. **Overall score = `max()` of the three sub-scores, not an average.** One badly-backed-up queue is enough; averaging against two calm queues would hide the spike that should change the hire decision.
5. Persist at least: overall score (0–10 or normalized 0–1 — pick one and lock it in the FieldState field docstring), driving source id, and keep raw counts in digester diagnostics / tick logs for soak forensics (not shown to Orion in the hire line).
6. Hub hire disclosure **only reads** the latest field state (or the existing situation/cabinet path that already surfaces field pressures) and formats one line: *"Queue pressure: 8/10 (high) — durable GPU demand is running well above its normal level. Deep Cursor digs compete with that. Prefer hire when Mind says deep."* Never call Cursor "expensive" — the **shared queue** is the scarce thing.

**Metric quality gate (CLAUDE.md §0A) — blocking before wire-in, recorded in this doc / PR:**

1. **Provenance** — producers are the three table rows above; digester function + line that writes the FieldState fields must be named in the impl PR.
2. **Independence** — open until checked: all three may share the same circe GPU pool. `max()` over correlated sources can still be defensible, but correlation vs historical data is required — **TODO, first digester impl step**. Must also prove this is not a monotonic transform of existing `gpu_pressure` (node biometrics strain) or a rebadge of `sustained_load_pressure` / `cortex_exec_step_load`.
3. **Theory anchor** — measures *queue contention for shared agent/curiosity capacity* that a Cursor-vs-local decision trades against — not GPU thermals, not tension change-detection.
4. **Live-data sanity** — **NOT YET DONE, blocking.** Pull multi-day history per source; confirm each can return to genuine calm; drop/reweight a degenerate always-flat source before it sits inert in `max()`.
5. **Existing-mechanism check** — `rg` digester + field glossary + `PRESSURE_DIMENSIONS` / `field_pressures()`; reuse or extend rather than duplicate.
6. **Reversibility** — additive FieldState fields + digester module; retire by deleting producer **and** lock/registry/glossary entries together (kill means kill — no “excluded from one consumer but still ticking”).

**Semantic layer + CI static gate (mandatory — same patch that introduces the metric):**

This is not optional documentation. A new FieldState / inner-state / pressure instrument that hire disclosure (or any cognition consumer) reads must clear the **metric semantic layer** (`docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md`) and the **orion-static-gates** workflow, or it does not ship.

Required in the **same** implementation changeset as the digester producer:

| Surface | What must happen |
| --- | --- |
| `FieldStateV1` | Additive fields for score (+ driving source; EWMA state as needed) with real quiet-tick semantics documented |
| Digester producer | Tick path writes the fields; tests; README / channel glossary entry if it is a field pressure channel |
| `orion/inner_state_registry.py` | Entry naming producer service, cadence, composition, cognition consumer(s) — hire disclosure is a real consumer |
| `config/field/field_channel_glossary.v1.yaml` | If exposed as a field channel / pressure dimension — classify meaning so Hub/debug surfaces are not freehand |
| `config/metrics/metric_definitions.lock.json` | Regenerated (`check_definition_drift.py --update` or project equivalent) so R4 drift gate sees the new definition |
| `python scripts/check_metric_lineage.py --gate` | Clean — consumer existence / orphan ratchet must see Hub (or situation) as a discovered consumer, not an orphan |
| `python scripts/check_definition_drift.py --gate` | Clean |
| `python scripts/check_inner_state_registry.py` | Clean |
| CI | `.github/workflows/orion-static-gates.yml` steps above must pass on the PR |

Hub's job is **consume + format**. If digester is down or score absent, disclosure **fails open** (omit the queue line) — it must not invent a Hub-side EWMA fallback that re-creates the shadow meter this revision forbids.

**Sibling pattern to copy (not fork):** `sustained_load_pressure` — digester-computed, on `FieldStateV1`, consumed by Hub (`tension_outreach_trigger` style), registered in inner-state + metric lock, independent-theory docstring required.
## Privacy / authorship boundary

- Unchanged: Orion alone MERGEs `:InvestigationRole` and `:HelpRequest`.
- Peer remains read-only; Orion still writes priors/findings.
- Queue pressure score and denial counts are **operational facts**, not identity content.
- No keyword detectors on Juniper chat text; counters are hop/tool/run scoped.

## Dangerous failure modes

| Failure | Mitigation |
| --- | --- |
| Strong deep-nudge → Orion hires every shallow sitting | Only fire strong line when Mind `expected_depth=deep` (or denials≥2); keep flag to disable disclosure block |
| Denial counter false positives | Allow-list refusal strings; unit fixtures from live hop notes; threshold ≥2 |
| Budget refuse ignored → hire spam | Explicit resume teach + peer already refuse; eval on refused_budget brief path |
| Queue pressure stale/wrong | Digester fails open / Hub omits line if score absent; never Hub-side EWMA fallback |
| EWMA baseline uncalibrated at launch (half-life/floor picked, not fit to data) | Gate item 4 must run on real history before ship; score provisional until then |
| Slow half-life understates a fast, real spike | Persist/log raw counts beside score in digester diagnostics for soak |
| One always-near-baseline source permanently never fires | Gate item 4 degenerate-source check; drop/reweight rather than leave inert in `max()` |
| Metric ships without semantic-layer / static-gate registration | Block merge: lineage `--gate`, definition-drift `--gate`, inner-state registry must pass in the same PR |

## Disable / rollback

- Existing `HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE` covers disclosure lines (including omitting the queue line).
- Teach text is code; revert PR.
- Queue pressure line omitted if FieldState score absent / digester stale — **no Hub EWMA fallback**.
- Retiring the metric: delete digester producer + FieldState fields + glossary/inner-state/lock entries together; re-run static gates so orphans do not linger.

## Proposed schema / API changes

- **FieldStateV1:** additive fields for queue-pressure score, driving source, and per-source EWMA state as needed (quiet-tick semantics documented like `sustained_load_pressure`).
- **orion-field-digester:** producer module on the digestion tick; tests; README / glossary as required for a real pressure instrument.
- **Registries / lock (same PR):** `orion/inner_state_registry.py` entry; `config/metrics/metric_definitions.lock.json` refresh; field channel glossary if applicable.
- **Hub:** read-only consumer for hire disclosure (and later reserved supervisor consumer) — format string only.
- **No** Hub-owned Redis EWMA keys for this score.
- **No** new bus channel required unless an existing field-state read path is insufficient (prefer reuse).
- Extend `format_role_teach_disclosure` with progress inputs (denials, budget, official score reading) — still pure strings.
- Thin Hub helper ok: `count_access_refusals(run_id)` (hop-local, not a metric). Queue pressure is **not** a Hub helper that recomputes the score.

## Files likely to touch (implementation follow-up)

- `services/orion-field-digester/app/digestion/…` — EWMA + `max()` score producer
- `orion/schemas/field_state.py` — additive fields
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`, optionally `config/field/field_channel_glossary.v1.yaml`
- Digester tests + gate-clean CI on the PR
- `orion/curiosity/kickoff_prompt.py` — role teach wording
- `orion/curiosity/role_teach_disclosure.py` — deep strong line; progress composition
- `orion/hub/turn_orchestrator.py` and/or `services/orion-hub/scripts/curiosity_investigation.py` — denial count, budget brief, **read** FieldState score → splice
- `orion/curiosity/tests/` + Hub tests (consumer only)
- Parent READMEs / hire specs cross-link

## Non-goals

- Python auto-MERGE of `hire_cursor` or HelpRequest
- Auto-hire solely on attention winners or thermals
- Keyword / feeling lists on user message text
- Raw-count disclosure as a first step before the score (superseded 2026-09-20)
- **Hub-local Redis / Hub-scheduler EWMA for this score** (superseded 2026-09-20 — official digester metric only)
- Shipping the score without semantic-layer registration or with failing static gates
- Changing Cursor invoker / peer budget meter internals (reuse `decide_cursor_budget`)
- Replacing self-study / self-model freshness pipes

## Acceptance checks

1. Motor boot with Mind `deep` contains an explicit **strong handoff** sentence (not only “expected depth: deep”).
2. Fixture: two permission-denied hop notes ⇒ disclosure contains handoff-now line.
3. Fixture: PeerBrief `refused_budget` ⇒ resume / don’t-rehire line; no encouragement to open another HelpRequest.
4. Live or fixture: the queue pressure **score** (not raw counts) appears on an Orion-origin curiosity motor prompt when FieldState has a fresh reading, with the driving source named.
5. Flag off ⇒ teach may still be revised, but soft disclosure block absent (existing contract).
6. After soak: at least some sittings with Mind `deep` **or** denials≥2 show `hire_cursor` and/or HelpRequest — if still 0/0, escalate proposal (possible Python assist) rather than yelling louder in the prompt only.
7. Metric quality gate item 4 (live-data sanity, §4) recorded against real multi-day history **before** the score ships to Orion.
8. **Semantic layer / CI:** `python scripts/check_metric_lineage.py --gate`, `python scripts/check_definition_drift.py --gate`, and `python scripts/check_inner_state_registry.py` pass on the impl PR; lineage card for the new URN shows digester as producer and hire disclosure (Hub) as a consumer — not an orphan.
9. No Hub Redis key / Hub tick exists that recomputes this EWMA (grep acceptance in PR).

## Recommended patch order

1. **Teach rewrite + deep strong nudge** in formatter (tests first).
2. **Permission-denied ≥2** progress line from hop/tool evidence.
3. **Budget-spent resume** line from PeerBrief.
4. **Official queue pressure metric:** multi-day history + §0A gate → digester producer + FieldState + registry/lock/glossary → static gates green → Hub read-only disclosure line.
5. **Soak.**

## Open question (locked default for this design)

**≥2 permission denied:** nudge + Orion-authored HelpRequest (**default**).
Python auto-enqueue is a **follow-up proposal** if soak still shows 0 HelpRequests after patches 1–4.

## Follow-up: the same score, a second consumer (2026-09-20)

`docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`'s "arming the reading channel" patch (PR #2253, merged) left two things unbuilt on purpose: nothing yet watches curiosity runs and grades them automatically, and nothing decides *who* does that grading. In conversation with Juniper the same day this doc was revised, a second use for the **official** queue-pressure FieldState reading came up: Orion choosing whether to **grade its own hop notes itself, or hand that grading to Cursor** — the same self-vs-outsource decision this patch already builds for investigating, just applied to reviewing instead.

This is **not** part of the patch order above and is **not** being implemented alongside patches 1–4. It's recorded here because it's the same **registered** primitive, so building the digester metric with a second consumer declared (or at least reserved in the inner-state registry notes) avoids a near-term duplicate Hub reimplementation:

- Second consumer reads the **same** FieldState fields — no second EWMA, no second gate to clear, no Hub recomputation.
- The decision it feeds is different in kind from patches 1–3: hiring Cursor for *investigation* is already-approved territory; Orion grading its own reasoning and *writing that grade back onto a `Prior`* is belief-adjacent and needs its own proposal-mode doc. This section only reserves the shared instrument.
- Open question for that future doc: is outsourcing grading about *load relief* (queue score is the right trigger) or *independent second opinion* (load is the wrong trigger)? Answer before writing that doc.

## Trace that proves it worked

- Soft HUD `motor_boot` raw contains strong deep nudge / denial / queue **score** lines (same proof surface as #2252).
- Digester FieldState (or live field read) shows a non-orphan queue-pressure score with driving source; lineage card matches.
- Static gates green on the impl PR (lineage, definition drift, inner-state registry).
- Graph: nonzero `hire_cursor` roles and/or HelpRequests after soak.
- Peer logs: HelpRequest received (not merely listening).
- If budget refused: next motor_boot shows resume language; hop clock continues without duplicate hire spam.
