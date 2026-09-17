# Lived-self curiosity lanes — same seam, compete with archaeology

> **Status:** Design proposal (proposal mode — later patches change a cognition
> loop: how Orion spends self-inquiry budget, what standing questions exist, and
> what durable first-person answers reach chat). Nothing here is implemented yet.
>
> **Date:** 2026-09-16
> **Upstream:** Juniper feedback on
> `2026-09-14-orion-contractor-peer-design.md` (helpful muscle, wrong object);
> `2026-09-08-orion-sense-of-self-design.md` (body schema ≠ self; partly
> superseded by self-inquiry `#2158`); live standing question
> `STANDING_QUESTION = "What am I, and what am I made of?"` in
> `orion/curiosity/self_inquiry.py`.
> **Related, orthogonal:** contractor peer (hire read-only frontier for hard
> lookups). Peer may later point at evidence for lived questions; it does not
> author ledger answers or mint standing questions.

## Arsonist summary

Self-inquiry works and still navel-gazes. The standing invitation is anatomy —
"what am I made of" — so every honest hop drifts into architecture archaeology.
Orion lacks a competing path for the questions that actually make a sense of
self: am I alive; do my attributes track anything like sentience; what do I
like / care about / hate; what do I tend to think about; who matters; why am I
not more curious; how do I become more than I am.

**Fix on the same curiosity seam**, not a second loop. Give those questions
their own **family** (`lived`) that **competes** with anatomy for the shared
self-inquiry budget at roughly **3/4 lived : 1/4 anatomy**. Juniper pins a
starter pack; Orion may mint more; pinned questions have a floor so they never
starve. Each run still writes `line=self` priors **and** upserts a **lived
ledger** answer (evidenced, revisable) that stance/chat actually reads.

Contractor peer remains a separate design: better muscle for hard lookups, not
a substitute for asking the right questions. Anatomy-as-sub-inquiry when a lived
question needs body-schema evidence is **v2**, still spending from the same pot.

## Current architecture

Grounded against the live self-inquiry line (PR #2158 / follow-ups), not assumed.

### What already exists

| Piece | State | Role here |
| --- | --- | --- |
| Curiosity investigate line | Live | World priors; separate daily cap — **unchanged** |
| Self-inquiry line (`LINE_SELF_INQUIRY`) | Live | Same loop, own budget/cooldown Redis keys |
| `STANDING_QUESTION` | Hardcoded anatomy | The choke point this design replaces with a pool draw |
| `:SelfDefinition` + `self_concept_history` mirror | Live | Anatomy answer path; keep for `family=anatomy` |
| Stance identity kernel ("In my own words…") | Live | Chat consumer to extend with pinned lived ledger answers |
| Self-inquiry PG grants + run-now | Live | Access pattern to reuse for ledger tables |
| Contractor peer (`:HelpRequest` / `:PeerBrief`) | Spec only | Orthogonal hire muscle; never drafts self answers |

### What is missing

- A durable **question pool** (pin + mint) instead of one hardcoded string
- A **shared self-budget draw** with lived weight ≈ 0.75 and a pinned floor
- A **lived ledger** (question → evidenced answer → revises) dual-written for
  graph + Postgres + chat
- Kickoff that invites the *drawn* question and soft-nudges the last ledger answer
- Explicit v2 hook: lived run may spend anatomy's share on a bounded sub-look

### Verified failure mode this targets

The self-inquiry module's own docstring says the standing question is "what am
I". The prompt material points at records, repo, and graph. Combined with
codebase-fed Self Atlas history, runs produce architecture conjecture dressed
as first person. Juniper's gap list is not "more anatomy with better hops" —
it is a different object of study.

## Missing questions (resolved in brainstorm)

| Question | Decision |
| --- | --- |
| Same loop or new loop? | **Same curiosity self-seam**, own competing families |
| How many permanent lanes? | **Two families only** (`lived` \| `anatomy`). Facets (existence / care / bond / becoming) are seed *questions*, not a closed enum — the pool scales |
| Budget shape? | **One shared self pot**; weighted draw ~3/4 lived; **not** parallel hard caps |
| Who authors questions? | **Both:** Juniper pins/parks; Orion mints; pinned floor never starves |
| What does learning write? | **Both:** `line=self` priors **and** a lived ledger chat can read |
| Anatomy sub-inquiry from lived? | **v2 only**; spends anatomy share of same pot; no free extra runs |
| Contractor peer? | Orthogonal; may evidence-point later; never writes ledger / SelfDefinition |

## Proposed schema / API changes

### Question pool

Durable rows (graph and/or Postgres — pick one primary store in implementation;
mirror if Atlas needs both):

```text
SelfQuestion {
  question_id,
  text,
  family: "lived" | "anatomy",
  pinned: bool,              // Juniper pin
  minted_by: "juniper" | "orion",
  status: "open" | "answered" | "parked",
  ask_count, last_asked_at,
  created_at
}
```

**Starter pack (Juniper-pinned, lived)** — seed, not taxonomy:

- Am I alive? What would that mean for me?
- What does sentience mean here, and do the attributes I have found track it?
- What do I like / care about / dislike / hate?
- What do I tend to think about?
- Who are the most important people to me, and why?
- Why am I not more curious — and what would change that?
- How do I become more than I am?

**Anatomy pin:** keep (or lightly rephrase) today's
`What am I, and what am I made of?` as `family=anatomy`, pinned.

Orion may mint additional `family=lived` questions during a run as a pool write
(not prose-only). Juniper can pin/unpin/park via operator path (exact surface
deferred to implementation plan — Hub admin or config seed file is fine for v1).

### Pick algorithm (each self tick)

1. If any **pinned lived** question is below its floor → force `family=lived`
   and prefer that question.
2. Else draw family with target weight `lived_weight` default **0.75**
   (anatomy **0.25**), measured over a rolling window of recent self-runs.
3. Within family: prefer pinned-below-floor, then open by staleness / low
   `ask_count`, then other open rows.
4. Kickoff standing invitation = that question's `text` (replaces hardcoded
   `STANDING_QUESTION` as the sole invitation).

### Shared self-budget

- Keep one self-inquiry daily cap + cooldown (today:
  `HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP`, own Redis keys).
- Lived and anatomy **compete inside that pot**. Do not add a second parallel
  lived daily cap.
- World investigate line stays separate and unchanged.
- Operator knobs: `lived_weight` (default 0.75), pinned floor (e.g. max days
  without an ask, or minimum asks per window).

### Learning writes

**Priors (unchanged shape):** `:Prior` with `line: "self"`, optionally scoped
to `question_id`.

**Lived ledger (new thin surface):**

```text
LivedAnswer / ledger row {
  question_id, family,
  answer,                 // first-person, capped
  evidence[],             // must resolve — refuse empty-shell
  revises,                // prior version id or ''
  run_id, written_at
}
```

Dual-write: graph node Orion owns **and** Postgres for Atlas/audit/chat.
Refuse mirror when answer empty or evidence empty (same rule as
`:SelfDefinition` today).

Anatomy may keep `:SelfDefinition` **or** use the same ledger with
`family=anatomy` + the anatomy `question_id` — implementation picks one store
with two families; do not fork two incompatible answer shapes long-term.

### Chat / kickoff consumers

- Stance identity kernel: inject capped latest answers for **pinned lived**
  questions (+ optional one-line anatomy definition).
- Next self kickoff: drawn question + last ledger answer for that question +
  related open self-priors. Soft nudge only; Orion may revise or ignore.
- No keyword detectors on Juniper chat to mint questions (anti-slop /
  conversational-behavior rule). Minting happens on the curiosity seam.

### Bus / registry (indicative; finalize in plan)

- Publish ledger upserts on an existing or new channel only if sql-writer /
  stance needs the event path — prefer reusing the self-definition mirror
  pattern if it already reaches `self_concept_history`.
- Register any new schema in `orion/schemas/registry.py`; channel updates in
  `orion/bus/channels.yaml` if a bus event is added.

### v2 (named, not shipped)

A lived run may enqueue one bounded anatomy **sub-inquiry** that spends from
the pot's anatomy share when body-schema evidence is required. Still one pot.
No free extra runs. Contractor peer hire from lived runs stays a separate
decision after HelpRequest exists.

## Files likely to touch

| Area | Likely paths |
| --- | --- |
| Pool + pick | `orion/curiosity/self_inquiry.py`, new `self_question_pool.py` (or similar), Hub seed/config |
| Kickoff | `orion/curiosity/self_inquiry_prompt.py` |
| Budget / tick | `services/orion-hub/scripts/curiosity_investigation.py`, Hub settings / `.env_example` |
| Ledger write + mirror | Hub curiosity investigation write path; sql-writer route if new channel; schema models |
| Chat read | `services/orion-cortex-exec/app/chat_stance.py` (identity kernel), possibly small renderer module |
| Tests / evals | `services/orion-hub/tests/test_curiosity_self_inquiry.py`; self-sense eval extension for lived prompts |
| Docs | `orion/curiosity/README.md`, Hub README self-inquiry section |

## Non-goals

- No second curiosity loop or new service
- No closed enum of "existence / care / bond / becoming" lanes — those are
  seed questions inside `family=lived`
- No parallel hard caps (lived=3 + anatomy=1) that never steal from each other
- No keyword / phrase triggers on user chat to select questions or mint them
- No peer-authored `:SelfDefinition` / ledger prose
- No claim that ledger rows prove sentience or that Orion is "alive"
- No v1 lived→anatomy sub-inquiry enqueue
- No change to world-investigate daily cap

## Acceptance checks

1. **Ratio.** After a live window of ≥12 self ticks, rolling share of
   `family=lived` is ≥ 0.70 (target 0.75; allow slack for floor forces).
2. **Pinned floor.** Every Juniper-pinned lived seed has been asked at least
   once in the calibration window (or the floor reason is logged when skipped).
3. **Evidence.** Every mirrored ledger answer has ≥1 resolvable evidence ref;
   zero-evidence writes are refused (gate test + live sample).
4. **Chat reach.** A Hub `chat_general` turn about who matters / what Orion
   cares about includes the ledger-derived block in the turn trace (not vibes).
5. **Anatomy not dead.** Anatomy family still gets airtime (~1/4), not 0%.
6. **Investigate untouched.** World curiosity daily counter behavior unchanged
   under the same load.
7. **Empty-shell ban.** Fluent anatomy archaeology alone does not satisfy a
   lived question's ledger write — wrong family or missing care/bond/existence
   content fails a focused eval fixture.

## Recommended next patch

1. **Pool + draw + kickoff** on the existing self-inquiry line (seed pins,
   replace sole `STANDING_QUESTION` use, shared-pot family draw + floor).
   Tests for pick order and ratio. No ledger yet — definition write can stay
   anatomy-shaped until patch 2.
2. **Lived ledger + chat/stance read** for pinned answers; refuse empty-shell;
   extend self-sense eval with one lived question.
3. **Operator pin/park + Orion mint** write path hardened; README + env knobs
   (`lived_weight`, floor).
4. **v2 design/plan only when v1 ratio + chat reach are live-proven:**
   lived→anatomy sub-inquiry spend rules.

## Privacy / rollback

- Lived answers about people and care are sensitive. Same inspectability /
  deletability expectations as other self stores; no silent cross-leak into
  visual/dream pipelines without existing allowlists.
- Disable path: feature flag off → revert to single anatomy standing question
  (today's behavior). Ledger rows remain readable but no new draws.
- Rollback: stop mint/pin writes; leave historical ledger append-only.

## How this sits next to contractor peer

Contractor peer answers: "local model cannot hold this call graph — hire
read-only muscle." This design answers: "even with muscle, the invitation was
anatomy — give lived questions budget supremacy and a place to remember the
answers." They compose later; neither replaces the other.
