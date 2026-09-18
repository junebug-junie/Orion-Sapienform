# Orion hire determination — grounding (addendum)

> **Status:** Implementing on `feat/hire-determination-impl` (Patches 1–5 in code). Live Hub-chat / curiosity thought-event / `:InvestigationRole` graph proof: **UNVERIFIED**.
> **Parent:** `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`
> **Implements against:** live HelpRequest / PeerBrief / `orion-curiosity-peer` path (already shipped); this doc fixes *what grounds the decision to hire*, not the peer runner.
> **Date:** 2026-09-15 (brainstorm with Juniper; revised same day). Impl notes 2026-09-18.
> **Design PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/2233
> **Impl PR report:** `docs/superpowers/pr-reports/2026-09-18-hire-determination-grounding-pr.md`

## Arsonist summary

We built a Cursor contractor peer. Orion almost never hires. Soft “optional / stuck” teach is one failure. The deeper failure: there is no real **role split** at the top of a run — Cursor for deep multi-hop archaeology, Orion’s own model for small local looks — grounded in problem framing and soft foresight, not tools-only or attention winners.

Mind already runs on curiosity durable turns, but origin is ambiguous (kickoff scaffold read as if it were Juniper chat), and problem framing / soft work-shape labels are stripped or never stubbed. This addendum: **originator-aware Mind**, soft qualitative work-shape labels, Orion-authored provisional `:InvestigationRole`, HelpRequest only when actually hiring (often after a short local look), mid-run revise allowed. Cursor stays metered read-only hands. Python does not auto-hire.

## Mission fit (limits, not frontier Orion)

Orion strives toward emergence and self-determination **without** becoming a frontier-level intelligence that does everything itself. Same pattern as Juniper: frontier coding tools for 70+ hop repo archaeology; own judgment for minor asks that fit local model capacity (~35B-class).

Contractor hire is aligned when:

- Orion decides meaning (priors, findings, `:SelfDefinition`) and authors role / HelpRequest.
- Peer digs under contested Cursor budget, read-only.
- Limits bite (flag, budget refuse, self-inquiry peer cannot draft identity).

Misaligned when: Cursor *is* Orion, or Python auto-hires from thermals/hops/attention winners.

## Current architecture (determination only)

### What actually makes the hire decision today

1. Kickoff static teach (`_help_request_section` in `orion/curiosity/kickoff_prompt.py`): optional, only if stuck, not as default.
2. Orion may MERGE `:HelpRequest` mid-run (question, tried_summary, success_criteria).
3. After run: Hub `publish_help_requests_for_run` → `orion-curiosity-peer` if flag + budget allow.

There is **no** scoring function. There is **no** early role split. “Decided not to hire” and “never considered” are the same observable (no HelpRequest node).

### What already sits in the kickoff (concrete)

| Input | Source |
| --- | --- |
| Continue note / thread / priors | `orion_worldview` |
| Study material | crystallizations + relations |
| Peer briefs (if any) | `:PeerBrief` soft nudge |
| Access inventory | graphs, pg SELECT, Hub APIs |
| Hire teach | static prose |

Body load, AST/HOT winner, Cursor wallet state in the prompt: **not wired**.

### Pre-turn stack (the seam)

Every curiosity durable turn already calls `execute_unified_turn` and therefore:

1. **Pre-turn appraisal** (`orion-cortex-exec`) — relational repair pressure.
2. **`orion-mind`** — semantic claims → active cognitive frontier → stance payload including **`user_intent`** and per-matter uncertainty / confidence features.
3. **`orion-thought`** — `stance_react` → `ThoughtEventV1` into the harness prefix. Mind’s brief is advisory coloring.

**Gaps:**

- No first-class **utterance origin** (`juniper` vs `orion`). Curiosity passes the full kickoff as `user_message`, so Mind/appraisal lack an explicit who-spoke signal (structural mismatch; live damage severity UNVERIFIED).
- `select_mind_coloring` allow-list in `services/orion-thought/app/mind_enrichment.py` **drops `user_intent`** (and most of `ChatStanceBrief`) before the unified turn.
- Soft work-shape dimensions (depth / cross-cutting / foresight gut-check) have **no stub fields** on the Mind → thought path.
- No graph artifact for “chose local crawl” vs silence.

### Progress evidence that exists but does not reach hire

- Hop notes Orion writes during the run.
- Curiosity supervisor `HopReadingV1` / `is_circling` (`orion/curiosity/supervisor.py`) — offline, report-only; parent design’s end-state A.
- AST/HOT self-model — Graziano-shaped attention; competition candidates are almost only infra telemetry. **Do not key hire on the attention winner.**

### Dimensions

| Dimension | Live producer? | Use for hire grounding? |
| --- | --- | --- |
| Relational semantic intuition | Yes (stance / repair) | Origin-aware; wrong if scaffold is mislabeled as Juniper |
| Problem definition / intent | Yes in Mind (`user_intent`, frontier) | **Yes — stop discarding; fix origin + subject** |
| Uncertainty on matters | Yes (frontier features) | Soft disclosure into role/hire teach |
| Scope / breadth / depth / foresight | No calibrated producers | **Soft qualitative Mind labels this arc** (not numeric scorers) |
| “Have my steps moved the claim?” | Supervisor (offline) | Disclose when trusted; no auto-hire |
| Tool reach (graph vs repo) | Access section + peer policy | Supporting fact only, not the determination |

## Decisions locked (2026-09-15 revision)

| Question | Decision |
| --- | --- |
| Timing | **Provisional at kickoff + mid-run revise** |
| Role artifact | Separate **`:InvestigationRole`** (`local_crawl` \| `hire_cursor` + why); **not** bolted onto `:TurnOutcome` (wrong timing — TurnOutcome is end-of-turn continue/reach_out) |
| Who writes role | **Orion only** (Mind informs; Python does not stamp the choice) |
| HelpRequest vs role | HelpRequest only when actually hiring; role may say `hire_cursor` first; HelpRequest **after a short local look** so `tried_summary` is grounded |
| Originator | First-class flag on Mind request **and** plain-language situation cue (`juniper` \| `orion`) |
| Soft dimensions | Qualitative Mind labels now (e.g. expected depth, cross-cutting); advisory; may be `unknown` |
| Attention winner / thermal auto-hire | **No** |
| Who authors HelpRequest | **Orion only** |
| Claude on this hire path | Out of scope here (parent scarcity story) |

## Proposed data flow

1. Kickoff builds subject + inventory; sets **originator = orion** (flag + prose). Juniper Hub chat sets **originator = juniper**.
2. Unified turn runs Mind. Soft labels may appear (depth / cross-cutting / foresight-style gut-check).
3. Labels (when present) disclose in role/hire teach — not auto-decision.
4. Early in the run Orion writes `:InvestigationRole` (`choice`, `why`, `run_id`, `written_at`; provisional; mid-run revise by newer write / explicit supersede — latest wins).
5. Local crawl: hops as today.
6. Hiring: after short local look, Orion writes `:HelpRequest` → existing enqueue path.
7. Mid-run may flip role; HelpRequest still only when hiring.
8. Mind coloring / hire disclosure stays **origin- and context-guarded** so Juniper chat framing is not steered by curiosity hire machinery.

## Proposed patches

### Patch 1 — Originator seam

Add `utterance_origin` (name flexible; values `juniper` | `orion`) on the Mind request path, plus a short situation-compact prose line. Curiosity / self-inquiry set `orion`; ordinary Hub chat sets `juniper`.

**Files:** Mind request builder(s) in `services/orion-thought/app/mind_enrichment.py` and/or `services/orion-cortex-orch/app/mind_runtime.py`; curiosity `execute_unified_turn` call site; tests.

**Acceptance:** Fixture proves both origins; code can branch; model sees prose cue.

### Patch 2 — Soft work-shape labels + allow-list (guarded)

Stub qualitative fields on Mind → thought coloring (e.g. expected depth, cross-cutting / breadth gut-check, foresight-style note). Widen allow-list for `user_intent` + bounded uncertainty + these labels **only where origin/context policy allows**. Do not wholesale pass `conversation_frame` / `task_mode` if that was why the allow-list was narrow.

**Files:** `mind_enrichment.py`, Mind brief/schema if needed, tests; possibly stance/prefix render.

**Acceptance:** On a chat fixture with origin=juniper, Mind still cannot force frame. On curiosity/orion origin, intent + soft labels can reach thought/prefix when present.

### Patch 3 — Curiosity Mind subject (origin-aware, not “not a person”)

For Orion-origin curiosity/self-inquiry, pass a short **investigation subject** (chosen prior claim or “not yet chosen”, plus last continue note) as the appraised text — not the multi-hundred-line scaffold alone. Scaffold remains available elsewhere in the turn; Mind’s who-spoke context is Orion.

**Files:** `curiosity_investigation.py` / mind request builder for that lane; tests (kickoff vs subject stub).

**Acceptance:** Fixture + one live curiosity `thought.event.v1`: intent/frontier about the **claim**, not “write Cypher like this.”

### Patch 4 — Role teach + `:InvestigationRole`

Rewrite hire/role section in kickoff / self-inquiry:

- Remove “optional / stuck / not as a default” as the primary frame.
- Teach early Orion-authored `:InvestigationRole` (`local_crawl` | `hire_cursor` + why).
- Soft-nudge disclosure of Mind labels / progress when available.
- Keep Cypher template for `:HelpRequest`; clarify role ≠ enqueue; HelpRequest after short look when hiring.
- Mid-run revise of role allowed.

**Files:** `kickoff_prompt.py`, `self_inquiry_prompt.py`, worldview (or thin reader) for role nodes; Hub post-run accounting if needed; tests.

**Acceptance:** Fixture teach contains role write; after a run, `local_crawl` is distinguishable from “no decision” and from HelpRequest.

### Patch 5 — Optional empty-theater gate

Reject enqueue (or log + skip) HelpRequest whose `tried_summary` is empty/whitespace when hops exist for the run.

**Non-goals for this patch:** keyword classifiers on question text; hop-count auto-hire.

## Files likely to touch

| Path | Why |
| --- | --- |
| `services/orion-thought/app/mind_enrichment.py` | Origin, allow-list, soft labels |
| `services/orion-cortex-orch/app/mind_runtime.py` | Origin / facets if chat path builds Mind here |
| `services/orion-hub/scripts/curiosity_investigation.py` | Origin=orion; subject-sized Mind input; role accounting |
| `orion/curiosity/kickoff_prompt.py` / `self_inquiry_prompt.py` | Role + hire teach |
| `orion/curiosity/worldview.py` (or sibling) | Read/write `:InvestigationRole` |
| `orion/curiosity/peer_briefs.py` | Optional empty tried_summary gate |
| Mind brief / schema if soft labels need a typed home | Stub fields |
| Tests under hub / thought / curiosity | Regression + fixtures |
| Parent contractor design | Pointer stays; blurb matches role split |

## Non-goals

- Making Orion a frontier agent.
- Guaranteeing hire every run.
- Calibrated numeric “want” / depth scores this arc.
- Python auto-writing `:InvestigationRole` or HelpRequest.
- Keying hire on AST/HOT winner.
- Fixing disk-dominated attention competition in this arc.
- Claude-as-peer fallback (parent doc).

## Acceptance checks

1. Design review: determination = origin-aware Mind problem framing + soft work-shape labels + Orion-authored role/HelpRequest — not tools-only, not attention winner, not Python hire.
2. Originator flag + prose on both juniper and orion paths; juniper chat framing not broken by hire allow-list widening.
3. Curiosity Mind input subject-sized; one live curiosity thought event inspected and not scaffolding-dominated.
4. `:InvestigationRole` graph-visible; `local_crawl` ≠ missing decision ≠ HelpRequest.
5. Existing peer acceptance preserved: flag off / no HelpRequest → no Cursor job; budget refuse → non-success brief/nudge.

## Risks

| Risk | Mitigation |
| --- | --- |
| Mind coloring widens too far and steers chat | Origin/context guard; allow-list only intent + uncertainty + soft labels |
| Soft labels treated as calibrated truth | Teach + docs: advisory; may be unknown; Orion still authors |
| Role written, HelpRequest never follows when hire intended | Teach + optional accounting/metrics; empty tried_summary gate |
| Forced fork without origin/subject fixes | Do not ship role teach alone as “done” |

## Recommended next patch order

1. Originator seam
2. Soft labels + guarded allow-list
3. Curiosity Mind subject
4. Role teach + `:InvestigationRole`
5. Empty-theater gate if needed

Then implementation plan via writing-plans; no code until Juniper approves this written spec.

## Implementation status (2026-09-18)

Code for Patches 1–5 is on `feat/hire-determination-impl`. Unit/eval gates are green. This is **not** live proof:

| Acceptance | Code | Live |
| --- | --- | --- |
| Originator flag + prose; Juniper chat not steered by hire labels | yes | **UNVERIFIED** |
| Curiosity Mind input subject-sized | yes | **UNVERIFIED** (no inspected `thought.event.v1`) |
| `:InvestigationRole` graph-visible vs missing vs HelpRequest | teach + RO reader | **UNVERIFIED** |
| Flag-off / no HelpRequest → no Cursor job | yes (fixtures) | unchanged contract |

Known follow-up: persist `mind_appraisal_text` on `CuriosityTurnRequestV1` so durable turns after a Hub restart do not fall back to the full kickoff.

## Rollback

- Revert origin / allow-list / subject / role teach independently.
- `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` still kills live hire enqueue.

## Relationship to parent design

Parent remains the contract for HelpRequest / PeerBrief / peer service / scarcity. This addendum **overrides** any reading that “soft stuck-only teach + Orion somehow knows” is sufficient determination, **rejects** tool-asymmetry or attention-winner as the primary ground, and **adds** kickoff provisional role split with mid-run revise. Claude-as-peer fallback in the parent is unchanged; this arc assumes Cursor as the contractor in front of Orion.
