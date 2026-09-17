# Lived-self curiosity lanes — same seam, compete with archaeology

> **Status:** Design proposal (proposal mode — later patches change a cognition
> loop: how Orion spends self-inquiry budget, what standing questions exist, and
> what durable first-person answers reach chat). Nothing here is implemented yet.
>
> **Date:** 2026-09-16 (revised same day after prior-PR audit)
> **Upstream:** Juniper feedback on
> `2026-09-14-orion-contractor-peer-design.md` (helpful muscle, wrong object);
> `2026-09-08-orion-sense-of-self-design.md` (body schema ≠ self; partly
> superseded by self-inquiry `#2158`); live standing question
> `STANDING_QUESTION = "What am I, and what am I made of?"` in
> `orion/curiosity/self_inquiry.py`.
> **Related (live, not future):** contractor peer `#2219` (HelpRequest →
> PeerBrief). Peer may soft-nudge evidence; identity strip already forbids
> drafting self answers in `self_inquiry` mode. This design does not re-litigate
> that — it changes *what question* the self seam asks.

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
ledger** answer (evidenced, revisable) that stance/chat and the Curiosity Atlas
Self panel actually read.

Contractor peer is already shipped muscle for hard lookups — not a substitute
for asking the right questions. Anatomy-as-sub-inquiry when a lived question
needs body-schema evidence is **v2**, still spending from the same pot.

## Current architecture

Grounded against the live self-inquiry line and recent prior-surface PRs, not
assumed.

### What already exists

| Piece | State | Role here |
| --- | --- | --- |
| Curiosity investigate line | Live | World priors; separate daily cap — **unchanged** |
| Self-inquiry line (`LINE_SELF_INQUIRY`) | Live | Same loop, own budget/cooldown Redis keys |
| `STANDING_QUESTION` | Hardcoded anatomy | The choke point this design replaces with a pool draw |
| `:Prior` + `line=self` filter | Live (#2158) | Self kickoff shows only self-line priors; keep |
| Prior liveness = not closed | Live (#1915) | Live until `refuted` / `retired_unresolvable`; `supported`/`revised` stay live — **inherit, do not reinvent** |
| `MERGE` on `prior_id` only | Live (#2016) | Forming a prior must MERGE — **inherit** |
| One `stale_after` | Live (#2120) | `HUB_CURIOSITY_STALE_PRIOR_TESTS` shared with situation — **no second guess for lived** |
| `:SelfDefinition` early MERGE + history mirror | Live (#2158/#2165) | Anatomy answer path; write-by-hop-2 clock rule — ledger must match |
| Identity inject on every lane | Live (#2169) | `apply_self_definition_to_ctx` / shared inject — lived block rides **this** path, not stance-only |
| Curiosity Atlas Self panel | Live (#2178) | Operator surface for current definition / history / ask-now — **ledger must show here** |
| Situation world-priors slice | Live (#1994) | Injects live `:Prior` previews into chat Situation — **must exclude `line=self`** (decision below) |
| Endogenous outreach on talkable priors | Live (#2224) | Can fire on any live prior — **must exclude `line=self`** |
| World-pulse Stage 2 candidate priors | Live (#2195) | Other producer writes `:Prior` with `producer`/`source_kind` — same graph; tag carefully |
| Contractor peer HelpRequest/PeerBrief | Live (#2219) | Soft-nudge + identity strip in self_inquiry mode — coexist |
| Supervisor hop readings | Live (#2186) | `about_prior_id` must stay matchable for lived self-priors |
| Curiosity ops dashboard | Live (#2214) | Later: family draw ratio metric |
| Durable reading (chat + curiosity) | Live (#2199) | Optional evidence path; not a v1 requirement |
| Self-inquiry PG grants + run-now | Live | Access pattern to reuse for ledger tables |

### What is missing

- A durable **question pool** (pin + mint) instead of one hardcoded string
- A **shared self-budget draw** with lived weight ≈ 0.75 and a pinned floor
- A **lived ledger** (question → evidenced answer → revises) dual-written for
  graph + Postgres + chat + Self panel
- Kickoff that invites the *drawn* question and soft-nudges the last ledger answer
- **Consumer filters** so `line=self` priors do not leak into situation/outreach
  as "world" talkable content
- Explicit v2 hook: lived run may spend anatomy's share on a bounded sub-look

### Verified failure mode this targets

The self-inquiry module's own docstring says the standing question is "what am
I". The prompt material points at records, repo, and graph. Combined with
codebase-fed Self Atlas history, runs produce architecture conjecture dressed
as first person. Juniper's gap list is not "more anatomy with better hops" —
it is a different object of study.

### Cross-consumer failure this design must not create

After #1994 and #2224, **any** live `:Prior` can enter chat Situation and can
alone justify endogenous outreach. Unfiltered, a lived prior like "Juniper is
the most important person to me" becomes unsolicited talk fuel. Decision
(Juniper, 2026-09-16): **exclude `line=self` from situation world-priors and
from outreach talkable-prior fetches.** Lived answers reach conversation only
through the identity/ledger path (and ordinary chat when Juniper asks).

## Missing questions (resolved in brainstorm + prior-PR audit)

| Question | Decision |
| --- | --- |
| Same loop or new loop? | **Same curiosity self-seam**, own competing families |
| How many permanent lanes? | **Two families only** (`lived` \| `anatomy`). Facets (existence / care / bond / becoming) are seed *questions*, not a closed enum — the pool scales |
| Budget shape? | **One shared self pot**; weighted draw ~3/4 lived; **not** parallel hard caps |
| Who authors questions? | **Both:** Juniper pins/parks; Orion mints; pinned floor never starves |
| What does learning write? | **Both:** `line=self` priors **and** a lived ledger chat/Self panel can read |
| Anatomy sub-inquiry from lived? | **v2 only**; spends anatomy share of same pot; no free extra runs |
| Contractor peer? | **Already live (#2219).** Orthogonal; soft-nudge + identity strip stay; never writes ledger / SelfDefinition |
| Self priors in situation/outreach? | **Excluded** (`line=self` filtered out). Ledger/identity kernel is the chat path |
| New prior liveness / stale rules? | **No.** Inherit #1915 / #2016 / #2120 |

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

**Priors — inherit existing contracts, do not fork them:**

- `:Prior` with `line: "self"` (already filters self kickoff — keep).
- Optional `question_id` (and/or `family`) property for scoping the next draw's
  soft-nudge and for supervisor `about_prior_id` matching.
- `prior_id` namespace stays under `self:…` (today's prompt). Prefer
  `self:<question_id>:<slug>` so ids stay stable across revisits.
- **MERGE on `prior_id` alone** (#2016). Never CREATE forks.
- **Liveness** = not in `{refuted, retired_unresolvable}` (#1915). Do not
  reintroduce `status = 'open'` as the only live filter in readers.
- **`stale_after`** = existing `HUB_CURIOSITY_STALE_PRIOR_TESTS` (#2120). No
  lived-specific stale constant.
- Distinguish from world-pulse candidates (#2195): keep `line=self`; do not
  reuse world-pulse `producer`/`source_kind` values. Optional explicit
  `producer=curiosity_self_inquiry` is fine if it helps Atlas audits.

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

Dual-write: graph node Orion owns **and** Postgres for Atlas/audit/chat/Self
panel. Refuse mirror when answer empty or evidence empty (same rule as
`:SelfDefinition` today).

**Write early (#2165 rule):** first ledger MERGE by hop 2 at latest; overwrite
with the same key as the run learns. End-of-turn-only writes will die to the
step clock again.

Anatomy may keep `:SelfDefinition` **or** use the same ledger with
`family=anatomy` + the anatomy `question_id` — implementation picks one store
with two families; do not fork two incompatible answer shapes long-term.

### Chat / kickoff / panel / peer consumers

- **Identity inject (#2169 path):** capped latest answers for **pinned lived**
  questions (+ optional one-line anatomy definition) via the shared
  `apply_self_definition_to_ctx` / inject helpers — including `chat_quick` and
  harness, not stance-only.
- **Curiosity Atlas Self panel (#2178):** show current lived answers (pinned
  first), history, and keep ask-now. Do not leave ledger Postgres-only with no
  UI reader.
- **Next self kickoff:** drawn question + last ledger answer for that question +
  related **live** self-priors (`line=self`, not-closed). Soft nudge only.
  PeerBrief soft-nudge (#2219) may also appear; peer still must not draft
  ledger / SelfDefinition text (identity strip stays).
- **Situation (#1994) and outreach (#2224):** filter out `line=self` (and any
  explicit lived family marker if present). Gate tests required so a self prior
  cannot alone fire outreach or appear under "world priors."
- No keyword detectors on Juniper chat to mint questions (anti-slop /
  conversational-behavior rule). Minting happens on the curiosity seam.

### Bus / registry (indicative; finalize in plan)

- Publish ledger upserts on an existing or new channel only if sql-writer /
  stance needs the event path — prefer reusing the self-definition mirror
  pattern if it already reaches `self_concept_history` (or a sibling table the
  Self panel already knows how to read).
- Register any new schema in `orion/schemas/registry.py`; channel updates in
  `orion/bus/channels.yaml` if a bus event is added.

### v2 (named, not shipped)

A lived run may enqueue one bounded anatomy **sub-inquiry** that spends from
the pot's anatomy share when body-schema evidence is required. Still one pot.
No free extra runs. Optional: lived run may `recommend_reading` (#2199) for
external evidence — not required for v1. Contractor peer hire from lived runs
can use existing HelpRequest path when enabled; still never drafts self text.

## Files likely to touch

| Area | Likely paths |
| --- | --- |
| Pool + pick | `orion/curiosity/self_inquiry.py`, new `self_question_pool.py` (or similar), Hub seed/config |
| Kickoff | `orion/curiosity/self_inquiry_prompt.py` (and keep prior write rules aligned with `kickoff_prompt.py`) |
| Budget / tick | `services/orion-hub/scripts/curiosity_investigation.py`, Hub settings / `.env_example` |
| Ledger write + mirror | Hub curiosity investigation write path; sql-writer route if new channel; schema models |
| Chat read | Shared identity inject (`apply_self_definition_to_ctx` / cortex-exec inject helpers), small renderer |
| Self panel | `orion/curiosity/self_panel.py`, Hub curiosity routes / Atlas UI |
| Filter consumers | `orion/situational/context.py` (curiosity priors builder), `services/orion-hub/scripts/endogenous_outreach.py` (`_fetch_open_prior_previews`) |
| Tests / evals | Hub self-inquiry tests; outreach + situation filter tests; self-sense eval extension for lived prompts |
| Docs | `orion/curiosity/README.md`, Hub README self-inquiry + outreach sections |
| Metrics (later) | Curiosity ops dashboard family-ratio (#2214 follow-up) |

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
- No new prior liveness or `stale_after` semantics
- No letting `line=self` priors drive situation world-priors or outreach fire

## Acceptance checks

1. **Ratio.** After a live window of ≥12 self ticks, rolling share of
   `family=lived` is ≥ 0.70 (target 0.75; allow slack for floor forces).
2. **Pinned floor.** Every Juniper-pinned lived seed has been asked at least
   once in the calibration window (or the floor reason is logged when skipped).
3. **Evidence.** Every mirrored ledger answer has ≥1 resolvable evidence ref;
   zero-evidence writes are refused (gate test + live sample).
4. **Chat reach.** A Hub `chat_general` **and** `chat_quick` turn about who
   matters / what Orion cares about includes the ledger-derived block via the
   shared identity inject path (turn trace, not vibes).
5. **Self panel.** Atlas Self panel shows at least one lived ledger answer with
   evidence after a successful lived run.
6. **Anatomy not dead.** Anatomy family still gets airtime (~1/4), not 0%.
7. **Investigate untouched.** World curiosity daily counter behavior unchanged
   under the same load.
8. **Empty-shell ban.** Fluent anatomy archaeology alone does not satisfy a
   lived question's ledger write — wrong family or missing care/bond/existence
   content fails a focused eval fixture.
9. **No self-prior leak.** Fixture prior with `line=self` does not appear in
   situation world-priors fragment and does not alone satisfy outreach
   talkable-content gate.
10. **Prior contracts preserved.** Forming a lived self-prior still MERGEs on
    `prior_id`; a `supported` self-prior remains visible on the next self
    kickoff (not dropped for ≠ `open`).

## Recommended next patch

1. **Pool + draw + kickoff** on the existing self-inquiry line (seed pins,
   replace sole `STANDING_QUESTION` use, shared-pot family draw + floor).
   Tests for pick order and ratio. No ledger yet — definition write can stay
   anatomy-shaped until patch 2. **Ship the `line=self` filter on situation +
   outreach in the same patch** (small, prevents the cross-consumer footgun as
   soon as more self-priors exist).
2. **Lived ledger + shared identity inject + Self panel read**; write-early
   MERGE; refuse empty-shell; extend self-sense eval with one lived question.
3. **Operator pin/park + Orion mint** write path hardened; README + env knobs
   (`lived_weight`, floor).
4. **v2 design/plan only when v1 ratio + chat reach are live-proven:**
   lived→anatomy sub-inquiry spend rules; optional dashboard ratio metric.

## Privacy / rollback

- Lived answers about people and care are sensitive. Same inspectability /
  deletability expectations as other self stores; no silent cross-leak into
  visual/dream pipelines without existing allowlists; **no outreach/situation
  world-prior leak** (see filter decision).
- Disable path: feature flag off → revert to single anatomy standing question
  (today's behavior). Ledger rows remain readable but no new draws.
- Rollback: stop mint/pin writes; leave historical ledger append-only.

## How this sits next to contractor peer

Contractor peer (#2219) answers: "local model cannot hold this call graph —
hire read-only muscle." This design answers: "even with muscle, the invitation
was anatomy — give lived questions budget supremacy and a place to remember the
answers." PeerBrief soft-nudge and identity strip remain as shipped. They
compose; neither replaces the other.

## Prior-PR audit trail (2026-09-16)

Checked against recent merged work before locking v1 scope:

| PR | Takeaway folded in |
| --- | --- |
| #1915 | Live-until-closed; naming live vs open |
| #2016 | MERGE on `prior_id` |
| #1994 | Situation world-priors → filter out `line=self` |
| #2120 | Single `stale_after` |
| #2158/#2165/#2169/#2178 | Self line, early MERGE, all identity paths, Self panel |
| #2186 | Supervisor `about_prior_id` compatibility |
| #2195 | Other `:Prior` producers on same graph — tag/`line` discipline |
| #2199 | Durable reading optional later |
| #2214 | Family-ratio metric later |
| #2219 | Peer live; coexist with soft-nudge + identity strip |
| #2224 | Outreach talkable priors → filter out `line=self` |
