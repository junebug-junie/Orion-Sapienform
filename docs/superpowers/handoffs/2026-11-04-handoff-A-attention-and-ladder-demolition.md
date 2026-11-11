# Handoff A: attention becomes the want-ledger competition; demolish the substrate ladder

Written 2026-11-04 at the end of a long session, for a fresh agent with no context. Read this, then
the spec. Do not re-derive the findings below; they were verified live on 11-04.

## What this session is for

Juniper's ask: **demolish the substrate ladder.** Do it by making attention the competition for
Orion's effort inside the body-economy want ledger, then retire the ladder's deciding half rung by
rung. The design is approved in direction. Juniper said "yeah I like your new proposal" to rev 2;
the PR itself is not merged.

**The spec:** PR #3089, branch `docs/prime-concern-sources`, worktree
`/mnt/scripts/Orion-Sapienform-prime-concern-sources`, file
`docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`. Read it in full first.
Its "Prior audit" section folds in `docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`,
which Juniper pointed to. That audit reached the same verdict six weeks earlier, and its lane-by-lane
cleanup did not change the outcome.

## The one-paragraph picture (live, 2026-11-04)

The lattice has two halves.

**M1–M5 is the sensing half:** grammar atoms → reducers → field → attention frame. It is healthy:
448k atoms a day, about 99% of them body and machine telemetry.

**L6–L11 is the deciding half:** proposal → policy → dispatch → feedback → consolidation. It writes
about 40k frames a day per rung and does almost nothing:
- 0 executions allowed out of about 121k "approved" decisions;
- 265 dispatches, 72% of them refused;
- 0 attention winners bound to an act in 24 h;
- feedback 98% unknown/mixed/dry-run;
- consolidation learns motifs about the ladder's own frames.

**About 90% of Orion's real acts bypass the ladder.** Journals, Prime, outreach, painting,
curiosity, dreams, reading and play each start themselves. Each bids in the body-economy
**`want_ledger`** (shadow everywhere except Prime) and records an act with an outcome in the
**act ledger** (`grammar_acts` + `substrate_action_outcomes`).

The design: attention = the `want_ledger` competition.
- **The menu:** event wants, each keeping its event identity.
- **The bid:** salience × earned family weight − price × cost. This is the existing
  `orion/physiology/choice.py` formula.
- **The evidence:** the self-monitor's reliability becomes evidence source (d) in
  `orion/physiology/weights.py`.
- **Binding:** one family at a time.
- **The ladder:** retired in stages, not revived.

## Decisions already made (do not reopen)

- **No shadow system.** Rev 1 of #3089 built a second attention menu beside the first. Juniper:
  "then what's the point of attention if we just build a shadow?" Rev 2 makes the want ledger the
  one place that decides.
- **Kill the ladder, don't revive it**, but move its two useful organs:
  - proposal templates become want producers;
  - policy's job (risk, consent, reversibility) becomes one gate the market consults.

  Keep the allocator; it is already shared with the want ledger. Kill the feedback frames.
  Consolidation: kill or repoint to the act ledger. **Juniper decides that one.**
- **Body signals (internal PE gauges) are not banned.** They compete when unusual, and they earn
  their weight.
- **Predictions live in `services/orion-prediction`** (Juniper's rule, 10-23). Any new bet goes
  there as a model, not as a side ledger.

## Open questions for Juniper (spec "Missing questions"; defaults in brackets)

1. **Binding order.** [Prime (already binding) → curiosity → reading → camera asks → outreach →
   journals → painting/dreams → play]
2. **A binding refusal means "not now" or "not this"?** [not now: the want stays open until its
   event expires]
3. **Juniper's turns as event wants.** [always fund; never priced out]
4. **Consolidation after the ladder.** [repoint to act-ledger episodes, a separate proposal]
5. **When each rung retires.** [its own PR, after the binding market has replaced its job for 2 weeks]

Ask these before building past patch 1. Patch 1 needs none of them.

## Recommended next patch (start here)

**Patch 1, pure, no live change:**
- `WantV1` gains `event_ref`, `salience` and `affordance`. Consumers go first, because the schema
  is `extra=forbid`.
- `appetite.py`: intensity = `max(hunger, salience)` for event wants.
- `weights.py`: evidence source (d), the self-monitor reliability per family.
- A **replay on the last 14 days of shadow `want_ledger` decisions**, showing what the market would
  have chosen.

Then patches 2–5 as listed in the spec:
2. two event sources plus Prime intake;
3. binding for curiosity and reading;
4. attention frame → salience, and merge the two frame producers;
5. retire ladder rungs one per PR, feedback frames first.

## Key code and data

- **Body economy:**
  - `orion/physiology/{choice,appetite,weights,families,budget,prices}.py`
  - `orion/schemas/physiology_want.py`
  - table `want_ledger` (cols include family, producer, verb, tier, outcome, reason, lane, net,
    binding_constraint, shadow, decision jsonb)
  - Design: `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`
- **Act ledger:** `orion/grammar/act_ledger.py`, tables `grammar_acts`, `substrate_action_outcomes`,
  `substrate_action_effect_posterior`.
- **Attention:**
  - `orion/attention/world_first.py`
  - `services/orion-attention-runtime/app/worker.py:241` (frame A, 2 s, 67% no winner)
  - `orion/substrate/attention_broadcast.py` and `services/orion-substrate-runtime/app/worker.py:3405`
    (broadcast B, 96 s)
  - `orion/schemas/attention_candidate.py`
- **Ladder:**
  - `services/orion-proposal-runtime` (`orion/proposals/builder.py`)
  - the policy runtime
  - `services/orion-execution-dispatch-runtime` (`orion/autonomy/allocator.py`, shared)
  - feedback-runtime
  - consolidation
  - rung-3 binding: `orion/autonomy/self_shed.py:58 bind_workspace_winner` (0 bound in 24 h)
- **Prime:**
  - `services/orion-mind/app/prime_loop.py` reads Temporal Self arcs
    (`GET /temporal-self/frame`, `orion/prime/thought.py:219`)
  - concerns live in Redis `orion:prime:state`
- **Temporal Self:** tables `temporal_self_event`, `temporal_self_arc` (orion-durable-runs). It is the
  best cross-source event-identity seam.
- **Self-monitor (live):** Redis `orion:self_monitor:latest`, `orion/self_monitor/` (see handoff B).
- **Liveness:** `make substrate-ladder-check` (GREEN 11-04).

## Facts that will save you time

- **Prime's concerns:** 42 in 7 days, 2 produced something, 40 dropped. About half came from
  internal PE gauges, half from body-economy "which lane binds" lines, and **0** from the world,
  Juniper, reading or curiosity. `act.prime.predict` is held unreliable (~5% vs ~11% stated). The
  act gate (#3081) now refuses new concern bets, with a re-test every 6 h.
- **Loops that bypass the lattice entirely:**
  - camera person asks (`orion_ask`, 495 answered and applied in 14 days, the most reliable loop,
    ungraded by the self-monitor);
  - curiosity;
  - outreach;
  - journals;
  - dreams.

  Juniper's chat enters attention only as a count (`world:chat`); the turn's identity is lost.
- **Street node.** `node:substrate.street` is honestly 0: no event passed its gate in 7 days.
  **No doorbell rings for 7 days.** #3090 (Ring push "Invalid EC key") merged but is UNVERIFIED to
  have restored rings. Check that before building on street events.
- **Harness events.** harness-governor writes about 25k grammar events a day that no reducer reads.
  UNVERIFIED that anything else reads them.
- **Past attempts.** Seven months of attention work, listed in #3089's history, mostly re-scored
  gauges. world-first #2583 is the one real improvement (67% of frames now have no winner).

## Rules this work must follow (learned the hard way)

- **Build the organ, not a feeder for a mechanism.** Rung 6 failed twice by feeding a single signal
  into existing machinery.
- **Kill means kill:** retire the old thing fully, no partial exclusion. Each rung retirement needs
  a consumer audit first.
- **No flag counts as done until the live path moves.** That is acceptance check 7 in the spec.
- **Proposal mode before cognition changes.** Binding changes what Orion does, so get Juniper's OK
  per family.
- **Deploys:**
  - only from the clean primary `main` checkout;
  - **wait until no chat turn is running** (Juniper's explicit instruction);
  - after a post-pull rebuild, check whether durable-runs was included: the rebuild script missed
    it once on 11-03.
- **Merging.** Juniper must explicitly authorize admin-override merges per PR. A blanket "override"
  doesn't carry to new PRs, and the permission classifier blocks it otherwise.
- **Plain English to Juniper,** STE100 style. Lead with what a thing does, identifiers after.
  Every critique carries a recommendation.
