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
  Persisted as episodes in `field_goal_episode` (PK `episode_id`, assigned by the writer; new episode
  on target change or a 10-min gap; PR #3940, live 2026-11-25). The per-message table
  `field_goal_provenance_log` (#3929) was retired and dropped on 2026-11-25. It is a sink: nothing
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

  It closes 1 at the first use (the 60 s resolver sweep, `SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC`), so a
  hit is fresh. It closes 0 only at the horizon (`outcome_source = expired`), so a miss is at least one
  horizon old by nature: "nobody used it in a day" takes a day to know.
- **Observable or not (review Q4).** Three cases, three treatments:
  - *observable*: the act's output landed as a `temporal_self_event` (same `correlation_id`) or has a
    `linked` `act_downstream_use` row. Graded 1 or 0 as above. Counts in the posterior.
  - `no_output`: the act ran and made nothing (`act_downstream_use.link_status = 'no_output'`, or the
    producer closes failed). A real 0: nothing was there to use. Counts.
  - `no_link_path`: output exists but no reader could ever be observed using it (1,185 of 1,511 rows
    in `act_downstream_use`, 7 d). Closes with `outcome_source = expired`, note `unobservable`, and is
    **masked out of the effect posterior and the self-monitor grade**, the way the reachable forecast
    masks untyped hours. Counting these as zeros would teach "nobody uses it" from acts that cannot be
    seen. A family switches in Patch 1b only when its observable share is >= 50% over 7 days (today:
    say 100%, infer 100%, make 67%, look 66%, sense 65%, ask 55%; play and maintain 0%, so they stay).
- The grader is a resolver in `act_ledger.py`, beside `resolve_ask_acts`, on the same sweep.
  It writes the outcome only; `error` and `surprise_nats` follow the ledger's existing close path, and
  the effect posterior updates as it does today.
- **Consumer-first.** `ActSignal` is a Literal on `extra=forbid` models read by orion-sql-writer and
  substrate-runtime (`grammar_events`). Ship the new value to readers first. Producers switch in
  Patch 1b, one family at a time, behind `ACT_OUTPUT_USED_FAMILIES` (ships ON with the families
  that pass the observable-share bar above).
- **Writer sweep (review Q2).** `"artifact_produced"` is written by 17 non-test source files (prime acts,
  durable-runs `run_acts.py`, Hub `act_emit.py`, curiosity replica, dig look, consent/intention/
  day_letter, town sessions). Patch 1b does NOT edit each call site's literal. It adds one helper,
  `act_signal_for(verb, actor)`, that returns `output_used` for switched families and
  `artifact_produced` otherwise, and every call site calls it. A site that is not swept keeps
  sending `artifact_produced`, which is still a valid Literal value: a silent fallthrough, not a
  rejection. So the patch carries a test that greps for a literal `"artifact_produced"` at an act-open
  call site and fails, and a live check that no switched (verb, actor) opened `artifact_produced`
  after deploy.
- **Series break (review Q1).** The self-monitor keys a belief on
  `grammar_acts:{verb}|{actor}|{signal_id}` (`orion/self_monitor/adapters.py:140`). A switch is a new
  claim, so a new key is correct: "my looks get used" is not "my looks finish". But the old key's
  grade must not silently vanish from the act gate and consults. Patch 1b: (a) writes a
  `series_break` row naming old key -> new key and the switch time; (b) the act gate reads the old
  key's grade until the new key has n >= 30 settled observable acts; (c) a test pins (b).
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

- **Bind (review Q7: attribution).** Binding by time alone is wrong: a goal is re-announced about
  1,100 times a day, so almost every act would be "serving" it, and almost every episode would read
  completed off incidental acts. An act binds `goal_ref = "field_goal:<artifact_id>"` only when the
  act's own target matches the goal's `field_target_id` (`world:chat` <- `act:say` on
  `hub.chat_reply`/`hub.outreach`; `capability:*` <- the matching capability act). The match table
  must come from the field-attention target registry (`FieldAttentionTargetV1.target_id`), not a
  hand list. Whether that registry already names act targets is UNVERIFIED. If it does not, Patch 2a
  builds that mapping first, with a test per pair. Acts with no match get no field goal. Prime
  concerns and curiosity runs keep their own `goal_ref` and are not overwritten.
- **Episode key (review Q6).** Episodes fold on `field_target_id` with a 10-min gap. `artifact_id` is
  unique per announcement (1,112 of 1,112 in 24 h), so it is only the lookup key, never the episode
  key. `field_goal_episode` is not in the live database yet. Merge order: #3941, then #3940, then
  Patch 2. Patch 2's acceptance checks cannot run before #3940 is deployed.
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
- **Shadow first.** Patch 2b writes verdicts and the `goal_episode` events only. The discount stays
  off until 7 days of verdicts show the completed/failed split per target. Expected split: UNKNOWN
  today. If more than 90% of episodes read `completed`, the bind rule is still too loose, and the
  discount does not ship.
- **Read the now back (closes the loop in the grammar; review Q8).** The goal producer reads
  `GET /temporal-self/now` for `goal_episode` events, and `record_use`s what it reads. It writes the
  discounted `priority` into each new announcement row, since priority lives per row. goal_context
  then weights that goal less, through the field it already reads. No goal_context logic changes.
  The discount is NOT a rolling window, because a window oscillates: lower priority -> fewer serving
  acts -> no verdicts -> the window rolls off -> priority returns, about every 24 h. Instead the
  discount is a state per target. It is set after 3 `failed` episodes in a row, and it lowers
  priority by the failed share, floor 0.2. It is cleared only by a `completed` episode. The floor
  keeps some acts flowing, so a recovery can be observed. If a target has no verdict in 72 h, the
  state decays by half, so a goal that cannot be served at all is not starved forever.
- Juniper heuristic note: the 3-failures rule is a prior, tunable from data later. Not a veto.

### Patch 3: Prime reads its own track record (orion-mind, Prime frame)

- One frame line built from the now (`action_outcome` and `goal_episode` events, cited with
  `record_use`): per act kind, P(used) posterior and n, plus the open/last goal episode and its
  verdict. Example: "my looks get used 4% (n 461); outreach gets a reply 36% (n 39); I wanted chat
  2 h today, served."
- It sits after orion-sapienform-87's attention-schema line in `orion/prime/thought.py`. Built after
  their Step 2a merges, to avoid a conflict in the same function.
- Hunger (`orion/prime/hunger.py`, review Q9): a family becomes a need with content ("my paintings
  stopped being used") only when its week-over-week drop passes a two-proportion test at p < 0.05.
  A rule like "P(used) halved" is about 2 SE at p ~ 4% and a few hundred acts, so it would fire on
  noise. The drop must also hold with the dominant reader removed. Today reverie wrote 992 of 1,065
  `experience_use` rows, so a reverie change alone would halve every family's rate. The need names
  which reader stopped using the family, so "one reader changed" reads differently from "my outputs
  stopped being useful". A drop confined to one reader becomes a need about that reader, not about
  the act family.

### Patch 4: introspection reads surprising misses (orion-thought, curiosity)

- With Patch 1 live, `action_outcome` ranks by real surprise. A make act at P(used) = 0.9 that
  nobody used now ranks high in the now.
- Self-inquiry's question producer takes the top-surprise `action_outcome` of the day as one seed
  ("I expected my reading notes to be used and none were. Why?"). Reverie already reads the now,
  so it sees the same event with no change.
- Staleness (review Q10): the resolver sweeps every 60 s. A hit closes at first use, but a miss
  closes at the 24 h horizon, so a miss seed is about a day old. That is intended: self-inquiry asks
  about a pattern ("my notes go unread"), not a moment. A surprising *hit* (an act at P(used) = 0.05
  that got used) closes within minutes and is the fresh seed.
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
  Watch (review Q5): `promise_kept` has 16 acts in its lifetime, so it cannot be a counter-metric yet.
  The measurable counters are `say -> juniper_replied` (110 acts/7 d, 62%) and
  `ask -> juniper_answered` (74 acts/7 d, 80%). Alarm when either falls by a two-proportion test
  p < 0.05 over 14 days while the cite rate rises. `promise_kept` joins at n >= 100. The check is in
  the Patch 4 roundup line.
- Danger 2: the goal discount starves a goal that matters but is hard (chat when Juniper is away).
  Floor 0.2, never 0, and a discount lasts only the 24 h window.
- Rollback: `ACT_OUTPUT_USED_FAMILIES=` (empty) restores `artifact_produced` per family.
  `FIELD_GOAL_FAILURE_DISCOUNT_ENABLED=false` stops the producer discount. Prime's line drops when
  its now kinds are absent.

## Acceptance checks (live)

1. After Patch 1b: per switched family, surprise is not flat. The 7-day spread of `surprise_nats`
   has IQR > 0.05 nats (today every family is 0.0000-0.004). Mean surprise alone is not the check,
   because at P(used) ~ 3% one graded-0 close already gives -ln 0.97 ~ 0.03, so base rate alone would
   pass a mean bar (review Q3).
2. Replay: weekly median surprise over 3 weeks shows no monotone drift with posterior age (the
   organs.yaml note becomes false).
3. Gate, with numbers (review Q3): at least 3 families with n >= 200 observable settled acts have
   per-act predictions that sort used from not-used with AUC >= 0.60 and sort p < 0.05 (the same
   statistic `/turn-self-bets/live` already reports as `sort_auc`/`sort_p`), and Brier skill vs the
   family base rate >= 0.02. A family-mean posterior alone cannot pass this: it has no per-act
   spread. So this check also proves the predicted value carries per-act information, for example
   from object or hour. If no family passes after 21 days, the posterior is only a base rate, and the
   chooser (#3433) gets it labelled as such.
3b. Self-monitor: no switched family's act-gate grade flips without a `series_break` row naming it.
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
