# Attention is the competition for Orion's effort

Date: 2026-11-04. Status: proposal (docs only). It changes what Orion chooses to spend effort on, so
proposal mode comes first (CLAUDE.md 0A). It replaces the first draft of this PR ("Prime's concerns
come from loops that close"). Juniper's challenge to that draft was correct: it built a second
attention system beside the first.

## In one paragraph

Today Orion has two systems that choose what to do.

The **substrate ladder** runs attention → proposal → policy → dispatch → feedback. Each rung writes
about 40,000 frames a day, and almost nothing comes out the end:
- 0 executions allowed;
- 265 dispatches, 72% of them refused;
- 0 attention winners bound to an act in 24 h.

The **producer path** does about 90% of Orion's real acts: journals, Prime, outreach, painting,
curiosity, dreams, reading and play. Each loop starts itself, bids for effort in the body-economy
**want ledger**, and records an act with a real outcome in the **act ledger**. Attention feeds the
first system; the doing happens in the second.

This proposal makes attention what it should be: **the competition in the want ledger.**
- **The menu:** events that keep their identity, from Temporal Self and the loops that close.
- **The bid:** how unusual the event is, multiplied by what that kind of effort has earned, minus
  what it costs.
- **What it earned:** the body economy's learned weights, with the self-monitor's track record as
  one more evidence source.

The want ledger becomes the one place that decides what matters, and the decision is binding. The
ladder is retired in stages, after it is replaced. It is not revived.

## Arsonist summary

- Seven months of attention work re-scored internal gauges, through normalization, percentiles,
  fade, sustain and focus runs. The two attempts to add real-world events died: the street node is
  honestly 0, because no doorbell rings arrived in 7 days, and the curiosity-findings candidate never
  left the docs.
- The real reason attention can't matter: **its output goes into a ladder that doesn't act.** Even a
  perfect attention frame would be read by proposals and policy that approve about 121k decisions a
  day, allow 0 executions, and bind 0 winners.
- The body economy already holds the parts of attention that work:
  - **Bids:** `want_ledger`, 489 decisions in 24 h.
  - **Price and cost:** `orion/physiology/choice.py`: `net = family_weight × value_rank − Σ price × cost`.
  - **Hunger:** `orion/physiology/appetite.py`.
  - **Learned, bounded weights from evidence:** `orion/physiology/weights.py`: downstream use,
    Juniper engagement, bounded drift, a 5% exploration floor.
  - **The gap:** it is **shadow everywhere except Prime**. Its menu is loop *families* on a clock, not
    *events* that just happened.

## Prior audit (2026-09-22): same verdict, cleanup did not change it

`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md` audited the transport lane
(M1–L11) and reached this proposal's verdict six weeks earlier:
- the lattice is "mostly a pipe carrying nothing";
- the top (L7–L11) "fires on a timer regardless of what the bottom says";
- the one learning step (an L11 motif) could not fire at any threshold, because it read attention
  targets' `dominant_channels`, which are always empty;
- "config being present was standing in for behavior".

Its fixes deleted dead templates, edges, motifs and unread config, lane by lane, and ranked new
lanes to add. Live today (11-04) the top still binds 0 attention winners a day. Three lessons follow:

1. **The split is the lattice's own.** M1–M5 (atoms → reducer → field → attention frame) is a working
   sensor, and it stays as the salience source for body signals. L6–L11 (self-state, proposal,
   policy, dispatch, feedback, consolidation) is the part that never decides, and it is what moves
   or retires.
2. **Adding lanes to the ladder is the "revive" path.** It was the audit's next step, and it would
   route more organs into a top that does not act.
3. **No flag counts as done until the live path moves** (acceptance check 7). The audit found
   configured behaviour that nothing read; binding families must show funded event wants becoming
   acts with outcomes.

The audit also found that freed proposal slots went to "templates with a declared signal". That is
the template → want-producer move in this proposal.

Still open from that audit, not blocking here: `BUS_OBSERVER_STREAMS` covers two streams, and
three vocabularies name one transport signal.

## Current architecture (verified live 2026-11-04 ~04:30Z)

| Level | Today | Event identity? | Real output |
|---|---|---|---|
| Grammar atoms | 448k/24h; about 99% body/machine telemetry (biometrics 271k, bus 75k, exec 30k) | yes, per atom | feeds reducers |
| Reducers, then field | per-node numbers (`substrate_field_state`, 40.8k/24h) | no: numbers per node | feeds attention |
| Attention frame (2 s) and broadcast (96 s) | world-first: candidates are 12 PE nodes plus `world:chat` (a count) and `world:reading` (link counts); 67% of frames have no winner | weak (target id + reason) | proposals; rung-3 binding |
| Proposals → policy → dispatch → feedback | about 40k frames/day per rung; 0 executions allowed; 265 dispatched (72% refused); feedback 98% unknown/mixed/dry-run | refs only | almost none |
| Producer loops → **want ledger** → **act ledger** | 235 acts/24h, all closed with outcomes; want ledger shadow except Prime | **yes**: actor, verb, object, expect, cost, outcome | the real doing |
| Temporal Self | 2.6k events/24h from chat, curiosity, street, presence, dreams, action outcomes, metacog, body | **yes**: source table/ref, subject | Prime's current "attention" |
| Self-monitor | per-faculty track record (live 11-03) | per faculty | act gate, cue, questions, outreach |

Loops that **bypass** the lattice entirely:
- camera person asks (`orion_ask`, 495 answered and applied in 14 days);
- curiosity;
- outreach;
- journals;
- dreams.

Juniper's chat turns enter attention only as a count, and the turn's identity is lost.

## Proposal

### 1. The menu: event wants

A loop that sees something worth effort posts a **want with an event attached**, instead of waiting
for its clock. Each want carries:
- `event_ref`: the Temporal Self event, act or source row it is about, so it keeps its identity;
- `salience`: how unusual this event is for its source, 0..1. It is the source's own percentile, the
  same scale world-first uses today, and the source computes it.
- `affordance`: the act it would run (family, verb, object), with its usual cost estimate.

v1 event sources (each is a loop that already closes):
- camera arrivals and asks;
- reading findings;
- curiosity findings;
- missed visitor expectations;
- self-monitor failing skills;
- Juniper's turns, as events, not counts.

Body signals (the PE gauges) post event wants **only when they are unusual**. Their affordance is
"look into it": a self-inquiry or curiosity focus.

Clock-driven wants (journals on schedule, painting rhythm) keep bidding as today, with intensity
from hunger. **One market, one menu.**

### 2. The bid: unusual × earned − cost

Choice's formula stays. Two inputs change:
- `value_rank` for an event want uses `salience` alongside hunger. The **intensity** of a want is
  `max(hunger, salience)`, so an unusual event can win before its clock is due.
- `family_weight` gains a third evidence source in `weights.py`, the self-monitor's **reliability**
  (upheld share) for the faculties of that family. It goes beside (a) downstream use and (b) Juniper
  engagement, and it follows the same rules: a rate not a count, a missing source is skipped, bounded
  drift, rest to the prior. This is the "value of attending" that attention never had.
- **No new math, and no second track record.** The self-monitor is an evidence source; the market
  stays the one decider.

### 3. Binding, one family at a time

The want ledger moves from shadow to **binding** family by family:
- **Order:** Prime (already binding) → curiosity → reading → camera asks → outreach → journals →
  painting/dreams → play.
- **The rule for each family:** a replay of the last 14 days against its shadow decisions shows the
  binding market would not starve it, meaning the funded share stays at or above the 5% floor. A
  live week then follows with an off switch.
- Gates that protect Juniper and the body stay as they are: heat reflex, quiet hours, outreach caps.

### 4. Prime reads the market, not attention arcs

Prime's concerns come from **funded event wants**: things the market judged worth effort. Prime
stops picking from Temporal Self attention arcs, which are 56% internal gauges. This is the same
market, not a second menu.

### 5. The attention frame becomes a sensor

The world-first frame keeps doing one honest job: noticing when a body signal is unusual. Its output
becomes the `salience` of body-signal event wants (step 1). It stops feeding the proposal ladder. One
of the two frame producers (2 s and 96 s) goes, and the broadcast stays.

## The ladder: revive or kill?

**Kill, in stages, after replacement. Keep the two organs that have real jobs, by moving them.**

| Rung | Today | Fate |
|---|---|---|
| Proposal templates | about 40k frames/day; templates such as `aitown_session` and `retro_leisure_session` are real affordances | **Move**: templates become want producers (they post event wants). Retire the 2 s frame churn. |
| Policy | about 121k "approved", 0 executions allowed | **Move the job, kill the frames**: risk/consent/reversibility checks become one gate the want ledger consults before a binding fund. Today those checks are scattered across arousal, thermal and caps. |
| Dispatch/allocator | 265 dispatches; the allocator is already shared with the want ledger | **Keep the allocator** (it is the market's placement), retire the dispatch frames. |
| Feedback frames | 98% unknown/mixed/dry-run | **Kill.** The act ledger's outcome resolution does this job. |
| Consolidation | motifs about the ladder's own frames | **Kill or repoint** to the act ledger (Juniper decides). |
| Attention frame (2 s) | 67% no winner | **Merge into the broadcast** (step 5). |

Why not revive the ladder whole: its frames carry references, not events. Its policy rubber-stamps.
Every real loop already routes around it. Reviving it would mean rebuilding the producer path inside
it, which is a second shadow. Retiring it saves about 200k frames a day of writes and reads.

## Missing questions (defaults in brackets)

1. **Binding order.** [Prime → curiosity → reading → camera asks → outreach → journals →
   painting/dreams → play]
2. **Does a binding refusal tell the loop "not now" or "not this"?** [not now: the want stays open
   until its event expires]
3. **Juniper's turns as event wants:** do they ever compete, or always win? [always fund. Juniper is
   never priced out; the event exists so others can see it]
4. **Consolidation after the ladder.** [repoint to act-ledger episodes, a separate proposal]
5. **When does the ladder retire?** [each rung's retirement is its own PR, after the binding market
   has replaced its job for 2 weeks]

## Proposed schema / API changes

- `orion/schemas/physiology_want.py` `WantV1`: add `event_ref`, `salience`, `affordance`. This is
  additive; consumers go first, because the schema is `extra=forbid`.
- `orion/physiology/appetite.py`: intensity = `max(hunger, salience)` for event wants.
- `orion/physiology/weights.py`: evidence source (d), self-monitor reliability per family.
- A per-family binding flag, e.g. `PHYSIOLOGY_BINDING_FAMILIES=prime,curiosity`, in the services that
  call Choice.
- Prime intake: funded event wants replace Temporal Self attention arcs.
- Attention frame: its output becomes body-signal want salience. The proposal-ladder edge is
  retired in later PRs.

## Files likely to touch

- `orion/schemas/physiology_want.py`
- `orion/physiology/{appetite,choice,weights,families}.py`
- one want-producer per event source:
  - Hub: reading findings, curiosity, self-monitor questions
  - sql-writer: vision asks, rhythm misses
  - Hub chat
- `services/orion-mind/app/prime_loop.py`, `orion/prime/thought.py`
- `services/orion-attention-runtime` and `orion/attention/world_first.py`: salience output
- later, per rung: `services/orion-proposal-runtime`, policy, dispatch and feedback runtimes, for
  retirement

## Non-goals

- No new attention scorer and no new track record. Choice and the learned weights stay the decider,
  and the self-monitor is evidence.
- No banning of body signals. They compete when unusual, and they earn their weight.
- No ladder revival.
- No change to Juniper-protecting gates.

## Proposal-mode checklist

- **Capability that changes:** one binding market decides what Orion spends effort on, from events
  with identity, valued by earned evidence.
- **Data touched:**
  - **Reads:** Temporal Self, the act ledger, self-monitor state, source tables.
  - **Writes:** want ledger decisions, which become binding.
- **Privacy boundary:** none (Juniper's standing rule).
- **Trace that proves it worked:**
  - each binding family's funded wants carry `event_ref`;
  - Prime's concern produce-rate beats its 5% baseline over 14 days (Wilson lower bound > 0.05);
  - attention-originated work produces acts with outcomes (today 0 bound winners a day);
  - no family is starved below the floor.
- **Dangerous failure modes:**
  1. **Binding starves a family.** Mitigation: per-family replay before binding, the 5% floor, an
     off switch per family.
  2. **Event wants flood the market.** Mitigation: salience is a percentile, so most events are not
     unusual; events expire.
  3. **Self-monitor evidence circles back,** where what is funded more is graded more. Mitigation:
     the weights use rates not counts, bounded drift, and rest to the prior.
  4. **Retiring a ladder rung removes a hidden consumer.** Mitigation: each rung retires in its own
     PR, after a consumer audit.
- **Disable or roll back:** clear `PHYSIOLOGY_BINDING_FAMILIES` to return to shadow everywhere except
  Prime. Event wants can be switched off per source.

## Acceptance checks

1. Each v1 source posts event wants with `event_ref` and `salience` from real rows. Most events have
   low salience: check the distribution live.
2. Weights: source (d) moves a family's weight only within the bounded drift, and with no evidence
   it rests to the prior (tests on replayed history).
3. The binding replay per family reports funded share and starvation, and a family binds only if it
   passes.
4. Prime: 100% of new concerns trace to a funded event want, and its produce-rate beats 5% over 14
   days.
5. Attention-originated work: at least 1 act a day traces back to an unusual body-signal want (today 0).
6. With binding off, decisions equal today's shadow decisions.
7. **The live path moved:** for each family switched to binding, a live week shows funded event wants
   that became acts with settled outcomes (counts in the PR report). A flag set to on is not evidence.

## Recommended next patch

1. **Schema + appetite + weights (pure),** with replay on live shadow history. This shows what the
   market would choose with events and evidence, and it changes nothing.
2. **Two event sources:** reading findings and camera asks, plus Prime intake from funded event
   wants, behind a flag.
3. **Binding for curiosity and reading** after their replays pass.
4. **Attention frame to salience;** merge the two frame producers.
5. **Ladder retirement,** one rung per PR: feedback frames first (pure overhead), then the
   proposal/policy frames once templates post wants.

Separate small fixes either way:
- Verify doorbell rings resume after #3090, because the street node needs them.
- Add a self-monitor adapter for camera asks.
- Check who reads the harness's 25k events a day.
