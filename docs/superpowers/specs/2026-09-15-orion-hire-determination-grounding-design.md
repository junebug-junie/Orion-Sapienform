# Orion hire determination — grounding (addendum)

> **Status:** Design proposal (proposal mode — cognition-loop adjacent).  
> **Parent:** `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`  
> **Implements against:** live HelpRequest / PeerBrief / `orion-curiosity-peer` path (already shipped); this doc fixes *what grounds the decision to hire*, not the peer runner.  
> **Date:** 2026-09-15 (brainstorm with Juniper).

## Arsonist summary

We built a Cursor contractor peer. Orion almost never hires. Soft “optional / stuck” teach + safe silence is one failure. The deeper failure: even a forced HelpRequest vs keep-local fork is a **hollow decision** if nothing in front of Orion is a real read on *what this problem is*, *how far they’ve gotten*, or *what kind of work it is*.

Tool inventory (graphs vs repo grep) is a real asymmetry and belongs in the access section — it is **not** enough to ground wanting or choosing. Scope, breadth, depth, foresight, and prediction are mostly **without producers** today; inventing scorers for them to feed hire is a metric cathedral. Relational / intent / problem-framing machinery **already exists** in `orion-mind` → `orion-thought` → unified turn, but on curiosity it is mis-aimed and then stripped.

This addendum: ground hire/keep in **existing mind + progress evidence**, keep Orion as author of `:HelpRequest`, keep Cursor as metered read-only hands, do **not** key hire off the Graziano attention winner (infra/disks).

## Mission fit (limits, not frontier Orion)

Orion strives toward emergence and self-determination **without** becoming a frontier-level intelligence that does everything itself. Contractor hire is aligned when:

- Orion decides meaning (priors, findings, `:SelfDefinition`).
- Peer digs under contested Cursor budget, read-only.
- Limits bite (flag, budget refuse, self-inquiry peer cannot draft identity).

Misaligned when: Cursor *is* Orion, or Python auto-hires from thermals/hops/attention winners.

## Current architecture (determination only)

### What actually makes the hire decision today

1. Kickoff static teach (`_help_request_section` in `orion/curiosity/kickoff_prompt.py`): optional, only if stuck, not as default.
2. Orion may MERGE `:HelpRequest` mid-run (question, tried_summary, success_criteria).
3. After run: Hub `publish_help_requests_for_run` → `orion-curiosity-peer` if flag + budget allow.

There is **no** scoring function. There is **no** tool-comparison logic in code. “Decided not to hire” and “never considered” are the same observable (no HelpRequest node).

### What already sits in the kickoff (concrete)

| Input | Source |
| --- | --- |
| Continue note / thread / priors | `orion_worldview` |
| Study material | crystallizations + relations |
| Peer briefs (if any) | `:PeerBrief` soft nudge |
| Access inventory | graphs, pg SELECT, Hub APIs |
| Hire teach | static prose |

Body load, AST/HOT winner, Cursor wallet state in the prompt: **not wired**.

### Pre-turn stack (the seam Juniper named)

Every curiosity durable turn already calls `execute_unified_turn` and therefore:

1. **Pre-turn appraisal** (`orion-cortex-exec`) — relational repair pressure.
2. **`orion-mind`** — semantic claims → active cognitive frontier → stance payload including **`user_intent`** and per-matter uncertainty / confidence features.
3. **`orion-thought`** — `stance_react` → `ThoughtEventV1` into the harness prefix. Mind’s brief is advisory coloring.

**Gaps that make this hollow on curiosity:**

- `user_message` is Orion’s own kickoff document, so Mind/stance appraise **scaffolding as if it were a person** (structural mismatch; live damage severity UNVERIFIED).
- `select_mind_coloring` allow-list in `services/orion-thought/app/mind_enrichment.py` **drops `user_intent`** (and most of `ChatStanceBrief`) before the unified turn. Problem framing is computed then discarded.
- Full `ChatStanceBrief` still reaches the older `chat_general` lane only.

### Progress evidence that exists but does not reach hire

- Hop notes Orion writes during the run.
- Curiosity supervisor `HopReadingV1` / `is_circling` (`orion/curiosity/supervisor.py`) — offline, report-only; parent design’s end-state A.
- AST/HOT self-model in `orion-substrate-runtime` — Graziano-shaped “what am I attending to”; competition candidates are almost only infra telemetry (cognitive nodes never enter). **Do not key hire on the attention winner.**

### Dimensions vs Juniper’s list

| Dimension | Live producer? | Use for hire grounding? |
| --- | --- | --- |
| Relational semantic intuition | Yes (stance / repair) | Wrong target on curiosity kickoff until input fixed |
| Problem definition / intent | Yes in Mind (`user_intent`, frontier) | **Yes — stop discarding; fix curiosity input** |
| Uncertainty on matters | Yes (frontier features) | Soft disclosure into hire section |
| Scope / breadth / depth | No | Do not invent for this arc |
| Foresight / prediction | No real producer | Do not invent for this arc |
| “Have my steps moved the claim?” | Supervisor (offline) | Disclose when trusted; record keep-local either way |
| Tool reach (graph vs repo) | Access section + peer policy | Supporting fact only, not the determination |

## Missing questions (resolved this session)

| Question | Decision |
| --- | --- |
| Is “force early HelpRequest/keep-local” enough? | **No** — hollow without grounded inputs. |
| Ground in tools only? | **No.** |
| Ground in attention winner? | **No** (disks / infra race). |
| Auto-hire from thermal / hops? | **No.** |
| Claude on this hire path? | **Out of scope for this addendum** — Cursor peer is the contractor; parent doc’s Claude fallback remains a separate scarcity story. |
| Who still authors HelpRequest? | **Orion only.** |

## Proposed changes

### Patch A — Stop discarding problem framing (unified turn, not hire-only)

Widen Mind coloring allow-list so **`user_intent`** and a bounded uncertainty summary from the active cognitive frontier reach `ThoughtEventV1` / prefix consumers. Do **not** wholesale pass `conversation_frame` / `task_mode` if those were the reason for the allow-list (Mind must not dictate chat framing on technical turns).

**Files:** `services/orion-thought/app/mind_enrichment.py`, tests for allow-list; possibly stance/prefix render.

**Acceptance:** On a Hub chat turn, `user_intent` from Mind is visible in the thought/prefix path (inspectable). Regression: Mind still cannot force frame.

### Patch B — Curiosity appraisal target (prerequisite for trusting Mind on this lane)

Stop treating the full kickoff prompt as `current_user_text` for Mind on curiosity/self-inquiry. Pass a short **investigation subject** (chosen prior claim, or “not yet chosen”, plus last continue note) — not the multi-hundred-line scaffold.

**Files:** `curiosity_investigation.py` / mind request builder for that lane; tests with fixture kickoff vs subject stub.

**Acceptance:** Live or fixture: Mind’s `user_intent` / frontier on a curiosity corr id are about the **claim**, not “write Cypher like this.” Pull one live `thought.event.v1` from a curiosity run before calling Patch B done (runtime proof).

### Patch C — Hire section as disclosure, not vibes

Rewrite `_help_request_section`:

- Remove “optional / stuck / not as a default.”
- State: peer is Cursor read-only digger; Orion owns meaning; contested budget may refuse.
- Soft-nudge style **extra_lines**: when available, disclose (a) Mind/frontier uncertainty on the active matter, (b) hop / circling progress facts when supervisor output is trusted enough to show (start report-only in prompt; no auto-MERGE).
- Keep Cypher template for `:HelpRequest`.
- Add explicit **keep-local** write on `:TurnOutcome` (or sibling fields): Orion records that they considered hire and kept the work, with a short why. Absence of HelpRequest alone is no longer the only signal.

**Files:** `kickoff_prompt.py`, `self_inquiry_prompt.py`, worldview TurnOutcome read/write + Hub post-run accounting; tests that teach no longer contains the stuck-only language; test keep-local is readable.

**Acceptance:** Fixture kickoff contains disclosure hooks; keep-local and HelpRequest are distinguishable in graph reads after a run.

### Patch D — Optional gate on empty theater (thin)

Reject enqueue (or log + skip) HelpRequest whose `tried_summary` is empty/whitespace when hops exist for the run — encourages grounded `tried_summary`, does not invent a difficulty scorer.

**Non-goals for D:** keyword classifiers on question text; hop-count auto-hire.

### Explicitly out of scope (this addendum)

- New attention organ; keying hire on AST/HOT winner.
- New scope/breadth/depth/foresight metrics.
- Supervisor auto-hire (parent end-state A) until circling is trusted.
- Operator-seeded fake HelpRequests.
- Pressure producers for concept nodes (separate substrate arc).
- Recent-attention SQL process filter (standalone bugfix; recommended separate PR).

## Files likely to touch

| Path | Why |
| --- | --- |
| `services/orion-thought/app/mind_enrichment.py` | Allow-list: pass intent / uncertainty |
| `services/orion-hub/scripts/curiosity_investigation.py` | Curiosity Mind input; keep-local accounting |
| `orion/curiosity/kickoff_prompt.py` / `self_inquiry_prompt.py` | Teach + disclosure |
| `orion/curiosity/worldview.py` | Keep-local fields if on TurnOutcome |
| `orion/curiosity/peer_briefs.py` | Optional empty tried_summary gate |
| Tests under hub / thought / curiosity | Regression + fixtures |
| This doc + short note in parent contractor design | Point to addendum |

## Non-goals

- Making Orion a frontier agent.
- Guaranteeing hire every run.
- Calibrated “want” as a number.
- Fixing disk-dominated attention competition in this arc.

## Acceptance checks

1. Design review: Juniper agrees determination = Mind problem framing + progress disclosure + Orion-authored HelpRequest/keep-local — not tools-only, not attention winner.
2. Patch A: `user_intent` (or agreed equivalent) survives thought → unified turn on a normal chat fixture.
3. Patch B: curiosity Mind input is subject-sized; one live curiosity thought event inspected and not scaffolding-dominated.
4. Patch C: kickoff hire section no longer says stuck-only optional; keep-local is graph-visible.
5. Existing peer acceptance preserved: flag off / no HelpRequest → no Cursor job; budget refuse → non-success brief/nudge.

## Risks

| Risk | Mitigation |
| --- | --- |
| Mind coloring widens too far and steers chat | Allow-list only intent + uncertainty summary |
| Curiosity still hollow if B skipped | A without B helps chat; hire grounding requires B |
| Circling disclosed too early misleads | Prompt-only until supervisor trusted; no auto-hire |
| Forced output without A/B | Do not ship forced fork alone — hollow |

## Recommended next patch order

1. **A** (intent through allow-list) — high value even without hire.  
2. **B** (curiosity Mind target) — required before trusting hire disclosure from Mind.  
3. **C** (teach + keep-local + disclosure hooks).  
4. **D** if theater HelpRequests appear.  

Then implementation plan via writing-plans; no code in this proposal.

## Rollback

- Revert allow-list / curiosity mind input / kickoff teach independently.
- `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` still kills live hire.

---

## Relationship to parent design

Parent remains the contract for HelpRequest / PeerBrief / peer service / scarcity. This addendum **overrides** any reading that “soft teach + Orion somehow knows” is sufficient determination, and **rejects** tool-asymmetry or attention-winner as the primary ground. Claude-as-peer fallback in the parent is unchanged here; this hire-grounding arc assumes Cursor as the contractor in front of Orion.
