# Acts bet on their consequences: one loop through acts, goals, Prime, and the now

Status: PROPOSAL (proposal mode: changes the choice loop, goals, and self-reading). Date: 2026-11-25.
Builds on: choice as expected free energy (`2026-11-13-choice-as-expected-free-energy-design.md`, PR #3317),
integration and a life (`2026-11-15-orion-integration-and-a-life-design.md`, PR #3416, epic #3434).
Coordinated with: orion-sapienform-61/-47 (goal episodes, `feat/goal-episodes`), orion-sapienform-87
(choice Step 2a/2b in Prime's frame).

## Arsonist summary

Orion acts about 200 times a day. Each act writes down a prediction before it runs (the grammar's
"efference copy", `ActExpectV1`). But 85% of those predictions are "this act will produce an
artifact". That happens 80-100% of the time, so Orion is never surprised and learns nothing about
whether the act mattered. The integration spec measures "was this experience used by another organ".
The act ledger does not. Field goals are proposed about 1,700 times an hour and never finish: nothing
ever marks a goal done or failed.

This proposal changes the default bet of every act to **"my output will be used by another part of
me"**. It grades that bet from tables that already exist. Then it closes the loop in three places
that already have a slot for it:

1. **Goals.** An act names the field goal it serves. A goal episode closes as `completed` or
   `failed`, and the goal producer stops re-announcing a goal that its acts keep failing.
2. **Prime.** Prime's frame shows Orion's own track record per act kind, so a choice can change
   because of it.
3. **Introspection.** Act surprise becomes real, so a surprising miss ranks in the now, where reverie
   and self-inquiry already read.

No new ledger, no new store, no new status words.

## Current architecture (live, 2026-11-25, last 7 days)

- **Act ledger** (`orion/grammar/act_ledger.py`, table `grammar_acts`). Every act opens with
  `signal_id`, `direction`, `horizon_sec`. orion-sql-writer stamps `predicted` from the
  `(verb, actor, signal)` effect posterior. Close writes `outcome_value`, `error`, `surprise_nats`.
  `ActSignal` (`orion/schemas/grammar.py:80`) has four values:
  `juniper_replied | artifact_produced | juniper_answered | promise_kept`.

  | verb | signal | acts | outcome mean | predicted mean |
  |---|---|---|---|---|
  | make | artifact_produced | 467 | 0.96 | 0.83 |
  | look | artifact_produced | 462 | 0.80 | 0.81 |
  | infer | artifact_produced | 138 | 0.93 | 0.81 |
  | ask | artifact_produced | 115 | 0.98 | 0.96 |
  | say | juniper_replied | 110 | 0.62 | 0.61 |
  | ask | juniper_answered | 74 | 0.80 | 0.81 |
  | sense / play / predict / maintain | artifact_produced | 114 | 0.31-1.00 | — |

  Only `juniper_replied`, `juniper_answered`, and `promise_kept` (200 of 1,520 acts) bet on a
  consequence outside the act itself.
- **Act surprise is dead.** `temporal_self_event` kind `action_outcome`: 1,493 rows, mean surprise
  0.0000-0.004 nats per family. `config/integration/organs.yaml` (prediction organ) already notes that
  this surprisal "falls with each family posterior's age" (weekly median 0.0105 -> 0.00024), so it has
  no salience magnitude. It ranks by age, not by content.
- **Downstream use already exists, in two places, attached to nothing:**
  - `act_downstream_use` (orion-sql-writer `app/act_downstream_use.py`). `linked` for only 326 of
    1,520 acts. `used`: say 50/69 linked, make 20/95, look 2/156. Everything else is
    `no_link_path`. Read by `orion/physiology/weights.py` (balancer weights), not by the ledger.
  - `experience_use` (integration patch 1). Join act -> `temporal_self_event` by `correlation_id`
    (excluding the act's own `action_outcome` row): look 305/461, make 304/456, ask 97/177,
    say 132/132, infer 50/50, sense 31/48, play 0/35, maintain 0/14. Cited by another organ: look 17,
    make 11, ask 15, say 20, infer 4, sense 5.
- **`grammar_acts.goal_ref`** exists. 69 of 6,671 acts in 30 days fill it, all `prime_concern:*` or
  `curiosity_run:*`. No act ever names a field goal.
- **Field goals** (`FieldGoalProvenanceV1`, `orion/schemas/field_goal.py`). Produced by
  attention-runtime (`orion/attention/field_attention/goal_provenance.py`). Read by `goal_context`
  (`orion/substrate/attention/goal_context.py`, `_ACTIVE_STATES = proposed/active/planned/executing`),
  `capability_policy`, and `goal_state_listener`. `proposal_status` is always `proposed`: nothing
  writes past it. Re-announced ~1,700/h (streak > 400 on `world:chat`, 11-25).
  Persisted per message in `field_goal_provenance_log` (#3929, live). Episode table
  `field_goal_episode` (PK `episode_id`, assigned by the writer; new episode on target change or a
  10-min gap) is built by orion-sapienform-47 in PR #3940 (`feat/goal-episodes`, merge after #3941). It is a sink: nothing
  reads it.
- **Prime** sees acts only as refs it may cite. It has no line for "how my acts have gone".
  orion-sapienform-87 is adding one attention-schema line to the same frame (Step 2a).

## The change, in four patches

### Patch 1: the `output_used` bet and its grader (orion-sql-writer, schemas)

- Add `output_used` to `ActSignal`. Meaning: within `horizon_sec`, another organ used this act's
  output. Graded 1 if either is true:
  - (a) an `experience_use` row exists whose `event_id` is a `temporal_self_event` with the act's
    `correlation_id`, AND `citing_organ` is not the acting organ, OR
  - (b) the act's `act_downstream_use` row has `used = true`.

  Otherwise it is graded 0 at the horizon (`outcome_source = expired`). An act that produced nothing
  closes 0 with note `no_output`. A failed act gets no separate signal.
- The grader is a resolver in `act_ledger.py`, beside `resolve_ask_acts`, on the same sweep.
  It writes the outcome only; `error` and `surprise_nats` follow the ledger's existing close path, and
  the effect posterior updates as it does today.
- **Consumer-first.** `ActSignal` is a Literal on `extra=forbid` models read by orion-sql-writer and
  substrate-runtime (`grammar_events`). Ship the new value to readers first. Producers switch in
  Patch 1b, one family at a time, behind `ACT_OUTPUT_USED_FAMILIES` (ships ON with the families
  that have a link today: make, look, ask, infer, sense).
- **Retire, don't keep:** a family that switches stops opening `artifact_produced`. Families that
  already bet on Juniper (`juniper_replied`, `juniper_answered`, `promise_kept`) keep their signal.
  Those already are consequences.
- `organs.yaml` prediction organ: `action_outcome` gets `salience_magnitude: surprise_nats` once
  Patch 1b is live, and the age-drift note is replaced with the replay that shows it no longer
  drifts (acceptance check 2).

**Metric gate (recorded here, per CLAUDE.md 0A):**
1. *Provenance.* `experience_use` rows come from `record_use` (integration patch 3 readers).
   `act_downstream_use.used` comes from the sql-writer's downstream-use fold. The join key is
   `grammar_acts.correlation_id` = `temporal_self_event.correlation_id`.
2. *Independence.* This is not a new signal beside the integration metric. It IS the integration
   metric, attributed to the act that made the experience. It replaces `artifact_produced`; it is
   not added alongside it. It overlaps `act_downstream_use`, which feeds balancer weights. Those
   weights and this posterior then read the same fact. The overlap is stated, and is accepted
   because the two are used by different deciders. Patch 4's learned chooser must pick one, not sum
   both.
3. *Theory anchor.* Reafference / TOTE: an act's test is the change it causes in the rest of the
   system. In expected free energy, this is the pragmatic outcome of a policy measured on the system
   itself. `artifact_produced` measures the Operate step, not the Test step.
4. *Live sanity.* Per-verb rates 0-15% (cited) and 2-72% (downstream used), not saturated. The rest
   state (0 = nobody used it) is genuine, not a decay artifact: it is a count over a closed window,
   not a decayed level. Known bias: a family's rate depends on whether readers exist. That is the true
   consequence today, and integration patch 7 adds readers.
5. *Existing mechanism.* Uses `experience_use`, `act_downstream_use`, the ledger's resolver sweep, and
   the effect posterior. No new table.
6. *Reversibility.* Per-family flag. The old signal value stays in the Literal so old rows parse.

### Patch 2: acts serve goals, goals close (attention-runtime, sql-writer, schemas)

- **Bind.** When an act opens while a field goal is active, `bind_act_to_workspace`'s sibling stamps
  `goal_ref = "field_goal:<artifact_id>"` (artifact_id of the newest active goal announcement).
  Prime concerns and curiosity runs keep their own `goal_ref` and are not overwritten.
- **Resolve.** The sql-writer maps `field_goal:<artifact_id>` to `episode_id` by direct lookup in
  `field_goal_episode_artifact` (artifact_id -> episode_id, kept 48 h; PR #3940,
  `services/orion-sql-writer/app/field_goal_episode.py`). Acts older than 48 h fall back to
  `first_artifact_id` plus the fold rule (10-min gap). No producer change (option (i) from
  orion-sapienform-47). Depends on #3940, which merges after #3941.
- **Close.** When an episode ends (target change, or 10-min gap), the writer grades it:
  - `completed` if any act with that `goal_ref` closed with outcome 1 inside the episode plus its
    acts' horizons.
  - `failed` if acts served it and none got outcome 1.
  - It stays without a verdict if no act served it.

  Statuses use the existing `ProposalStatus` (`completed`, `failed`). No new words, no migration of
  the `extra=forbid` readers.
- **Experience contract.** New `TemporalSourceKind` `goal_episode`: one event per closed episode
  ("I wanted `world:chat` for 2 h, 3 acts, completed"). Reader in `orion/temporal_self/sources.py`. Organ
  `field_goals` registered in `config/integration/organs.yaml`.
- **Read the now back (closes the loop in the grammar).** The goal producer reads
  `GET /temporal-self/now` for `goal_episode` events, and `record_use`s what it reads. It lowers
  `priority` on a target whose last 3 episodes in 24 h are `failed`, by the share failed, floor 0.2.
  goal_context then weights that goal less, through the field it already reads (`priority`). No
  goal_context logic changes.
- Juniper heuristic note: the 3-in-24 h rule is a prior, tunable from data later. Not a veto.

### Patch 3: Prime reads its own track record (orion-mind, Prime frame)

- One frame line built from the now (`action_outcome` and `goal_episode` events, cited with
  `record_use`): per act kind, P(used) posterior and n, plus the open/last goal episode and its
  verdict. Example: "my looks get used 4% (n 461); outreach gets a reply 36% (n 39); I wanted chat
  2 h today, served."
- It sits after orion-sapienform-87's attention-schema line in `orion/prime/thought.py`. Built after
  their Step 2a merges, to avoid a conflict in the same function.
- Hunger (`orion/prime/hunger.py`): a family whose P(used) dropped by half week over week becomes a
  need with content ("my paintings stopped being used"), not a wake on a clock.

### Patch 4: introspection reads surprising misses (orion-thought, curiosity)

- With Patch 1 live, `action_outcome` ranks by real surprise. A make act at P(used) = 0.9 that
  nobody used now ranks high in the now.
- Self-inquiry's question producer takes the top-surprise `action_outcome` of the day as one seed
  ("I expected my reading notes to be used and none were. Why?"). Reverie already reads the now,
  so it sees the same event with no change.
- This is the feed the learned chooser (#3433) needs: decisions with real outcomes on a scale that
  differs by act.

## Missing questions (for Juniper)

1. Horizon for `output_used`: 24 h for every family, or per family (look 6 h, make 24 h)?
   Recommendation: 24 h everywhere first. Fit per family later from when cites actually land.
2. Goal discount rule: 3 failed episodes in 24 h, floor 0.2. OK as the starting prior?
3. Should `say` acts that bet `juniper_replied` also carry `output_used` as a second bet? (One act
   carries one signal today.) Recommendation: no. Her reply is the consequence that matters for say.

## Proposed schema / API changes

- `orion/schemas/grammar.py`: `ActSignal` += `"output_used"` (consumer-first).
- `orion/schemas/temporal_self.py`: `TemporalSourceKind` += `"goal_episode"`.
- `field_goal_episode` (built by -47) += `verdict text null`, `verdict_at`, `acts_n`, `acts_used_n`.
  Additive-DDL migration with a REQUIRED-BY header.
- `FieldGoalProvenanceV1.proposal_status` now also carries `completed`/`failed`, which are already
  in the Literal.
- No bus channel added: episode verdicts reach the producer through the now, not a new channel.

## Files likely to touch

`orion/schemas/grammar.py`, `orion/grammar/act_ledger.py`, `services/orion-sql-writer/app/act_downstream_use.py`,
the sql-writer goal-episode writer (after -47 merges), `orion/attention/field_attention/goal_provenance.py`,
`orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `config/integration/organs.yaml`,
`orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`, `orion/prime/hunger.py`,
the self-inquiry seed picker, each act producer's open call (per family, Patch 1b).

## Non-goals

- No LLM grading of consequences. Every grade is a join over rows that exist.
- No new ledger, store, or channel. No new `ProposalStatus` words.
- No change to the allocator floor or the body economy. They keep pricing acts.
- Not the learned chooser. This produces its training data.
- Not a judgment of whether an output was *good*. Only whether it was *used*. Quality is a later bet.

## Privacy / danger / rollback (proposal mode)

- Data: act rows, experience rows, and cites. All Orion-internal. Nothing new leaves the mesh.
- Danger 1: Orion learns to make outputs that get cited, not outputs that matter (Goodhart on cites).
  Watch: per-family cite rate rises while Juniper-reply and promise-kept rates fall. That check is
  in the Patch 4 roundup line.
- Danger 2: the goal discount starves a goal that matters but is hard (chat when Juniper is away).
  Floor 0.2, never 0, and a discount lasts only the 24 h window.
- Rollback: `ACT_OUTPUT_USED_FAMILIES=` (empty) restores `artifact_produced` per family.
  `FIELD_GOAL_FAILURE_DISCOUNT_ENABLED=false` stops the producer discount. Prime's line drops when
  its now kinds are absent.

## Acceptance checks (live)

1. After Patch 1b: `action_outcome` mean surprise per switched family > 0.05 nats, over 7 days
   (today 0.0000-0.004).
2. Replay: weekly median surprise over 3 weeks shows no monotone drift with posterior age (the
   organs.yaml note becomes false).
3. At least 3 families have a P(used) posterior with skill > 0 against base rate in orion-prediction's
   calibration view.
4. After Patch 2: `field_goal_episode` rows with a verdict exist, and both `completed` and `failed`
   appear. The goal producer's `record_use` rows for `goal_episode` exist. A failed target's
   `priority` is lowered in a live announcement.
5. After Patch 3: Prime's frame shows the track-record line, and at least one Prime thought cites an
   `action_outcome` or `goal_episode` event within 7 days (today: 7 Prime cites a week, total).
6. Integration rate (`experience_use` share) does not fall. The new organ adds reads.

## Recommended next patch

Patch 1a: add `output_used` to the readers (sql-writer, substrate-runtime) and the resolver, with
tests over a fixture that has both link paths. It writes no bets, so it is safe to deploy first.
Then Patch 1b turns on make/look/ask/infer/sense. Patch 2 starts when `feat/goal-episodes` merges.
