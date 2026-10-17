# Attention world candidates: let what Orion does compete for attention

Date: 2026-10-16. Status: PROPOSAL (design mode, no runtime change in this PR).
Replay script: `scripts/analysis/replay_attention_world_candidates.py` (read-only).

## Arsonist summary

Today only two kinds of thing can win Orion's attention: a body sensor that is
unusually upset, and Juniper talking more than usual. The things Orion actually
does in the world (curiosity runs, reading, paintings) cannot enter the contest
at all.

The obvious fix is wrong. If "Orion finished a run" competes the way chat does
(how many in the last 15 minutes), Orion attends to its own busyness. The
replay shows reading alone would win 253 ticks a day, push out Juniper's chat
node 51 times, and cut the no-winner share from 64% to 55%. That is Orion
watching itself work, and it says nothing true about what the work found.

The right question is "did this act change what Orion believes?" One source
already measures that honestly: each curiosity run records how far it moved
Orion's beliefs, in nats of Bayesian surprise (`realized_nats`). That passes
the metric gate. Reading and paintings do not yet: reading records no
confidences, and most paintings run on a timer, so their finishing is
predicted by construction.

**Recommendation:** ship one new candidate, `world:curiosity`. A finished
curiosity run competes for 300 seconds when its belief change is in the top
tenth of its own last 7 days. It carries the run id, the priors it moved
(before and after), and a capped excerpt of the finding, so reverie can say
something true and specific. It yields to Juniper: while chat is eligible,
it cannot win. Over the last 6 days it would have won 13 ticks (3 runs),
displaced nobody's chat, and never spawned new work (the paths that turn a
winner into work already drop non-`node:` winners). Small on purpose: it is
the honest size of "Orion learned something surprising".

## Current architecture

- **The contest.** `services/orion-substrate-runtime/app/worker.py:2943-3034`
  (`_attention_broadcast_tick`) calls
  `orion/substrate/attention_broadcast.py:179` `build_substrate_attention_frame`
  with `world_first=True`. `world_first_signals` (`:342-372`) admits only
  `node:substrate.*` nodes that carry a `prediction_error`. Other nodes with
  pressure go into the trace list `uncalibrated` and are never ranked
  (`:368-369`). Plus external candidates from
  `_world_first_external_candidates` (`worker.py:3036-3050`): today only
  `chat_candidate` (`orion/attention/world_first.py:631`).
- **The rule.** `orion/attention/world_first.py` `judge_candidate`: each
  candidate is scored against its own 7 days. Cuts `(0.5, 0.9, 0.99)`. External
  wins only at band `high`/`unusual` (top tenth). Event-written sources stop
  competing 300 s after the event (`EVENT_ORIENTING_WINDOW_SEC`), fading after
  60 s. Continuous body sources must stay hot more than 300 s of 600 s
  (`orion/attention/sustained.py`, PR #2718). No eligible candidate means no
  winner.
- **Who won, last 6 days** (`substrate_attention_broadcast_log`, 11,976 ticks,
  2026-10-10 05:57 .. 10-16 07:50 UTC, one every ~44 s):

  | winner | ticks | share | per day |
  |---|---:|---:|---:|
  | (no winner) | 7,685 | 64.2% | 1,264 |
  | node:substrate.bus_synaptic | 981 | 8.2% | 161 |
  | node:substrate.perception | 971 | 8.1% | 160 |
  | node:substrate.biometrics | 591 | 4.9% | 97 |
  | node:substrate.execution | 385 | 3.2% | 63 |
  | world:chat | 374 | 3.1% | 62 |
  | node:substrate.codebase | 349 | 2.9% | 57 |
  | node:substrate.cabinet | 267 | 2.2% | 44 |
  | node:substrate.chat | 210 | 1.8% | 35 |
  | node:substrate.route | 163 | 1.4% | 27 |

  The brief said world:chat wins ~23 ticks/day; this window says ~62/day. The
  difference is the window, not a contradiction worth chasing here.

- **What a winner carries.** A candidate's `label` becomes the open loop's
  `description`, cut to 120 characters (`orion/substrate/attention/scoring.py:130`).
  The loop id is a hash of that text. The signal's `evidence_refs` is only
  `[source_id]` (`attention_broadcast.py:421-424`): the candidate's own refs
  are dropped. Signal `provenance` is copied onto the loop
  (`scoring.py:179`). So today a world winner reaches consumers as one short
  sentence and an id.

### Every consumer of the winner, and what a `world:curiosity` winner does there

| consumer (file:line) | reads | with `world:curiosity` |
|---|---|---|
| Reverie narration, `services/orion-thought/app/broadcast_reader.py:65` -> `reverie.py:84-165, 295-320, 425-470` | `attended_node_ids`, `selected_description`, loop `description`, `provenance.world_first_reason`, `magnitude` | Narrates the label. Without new evidence it narrates "a curiosity run changed my mind" and nothing else: the garbage-in problem. Needs the evidence block below. |
| Reverie proposals, `orion/reverie/proposal.py:66` | `attended_node_ids` | Drops a proposal when every attended id is world (`is_world_source_id`). `world:` prefix means no render_scene or other proposal can come from this winner. This is the loop breaker. |
| Reverie semantic lift, `orion/reverie/semantic_lift.py:20-49` | loop `source_refs` | `world:curiosity` becomes a citable ref, like `world:chat` today. New refs (`curiosity_run:<id>`) must be checked against its prefix parser. |
| Thought chain key, `services/orion-thought/app/chain.py:94,154` | `selected_open_loop_id` | Works. The loop id is stable if the label is stable per run (no counts or ages in it). |
| Coalition / stance_react / felt state, `orion/hub/association.py:62-90`, `stance_react.py:185`, `services/orion-thought/app/bus_listener.py:149`, `mind_enrichment.py:239` | `attended_node_ids`, loop text | Allow-list and `situation_compact` text. Fine. |
| Relational concept node, `orion/substrate/relational/adapters/attention_ctx.py:59-90` | `selected_*` | Writes `attending:current_focus`. Fine. |
| Endogenous curiosity, `orion/substrate/endogenous_curiosity.py:370-375` | `open_loops` | Skips loops with no `node:` ref. Cannot seed new curiosity. Second loop breaker. |
| Attention self-model, `orion/substrate/attention_self_model.py:681-790` -> `orion:attention:schema` -> `recent_attention_cue.py:18` (chat stance cue) | `selected_*`, `reason_narrative` | Juniper's chat stance would see "recent attention: a curiosity run …". Fine and wanted. |
| System One appraisal, `orion/substrate/system_one_appraisal.py:170-176` | `selected_*` | Input only. Fine. |
| Hub spotlight, `brain_frame_producer.py:258` -> `services/orion-hub/static/js/self-brain.js:81-118` | `attended_node_ids` | Id not in `frame.nodes`; falls back to the label, as chat does. Fine. |
| World-action binding, `orion/autonomy/self_shed.py:56-98` (`bind_workspace_winner`), `orion/proposals/builder.py:198-260` | first `node:` ref | `winner_unbindable:node_not_bound`, same as chat. Third loop breaker. |
| Goal bridge, `services/orion-attention-runtime/app/store.py:697-750` | `source_refs LIKE 'node:%'` | Filtered out. Goals never target it. |
| Voluntary override, `attention_broadcast.py:580`, `top_down.py:135` | goal target vs loop refs | Goals are internal-only, so no bias either way. |
| Temporal Self attention arcs, `orion/temporal_self/broadcast.py:39-56` -> `arcs.py:433-470` -> dream replay `services/orion-dream/app/replay.py:~316` | selected loop `source_refs[0]`, description | An "attention" arc with `subject_ref=world:curiosity`. Dream replay prints "attention arc: <label>". Fine. |
| Chat stance ATTENTION FRAME, recall intent (`stance_react.j2:68`, `orion/memory/retrieval_intent.py`) | the per-turn chat frame, not the broadcast | Not affected. |
| Field contest, `services/orion-attention-runtime/app/worker.py:239-305` | its own candidates | Not affected by this patch (non-goal). |

No consumer breaks on a non-`node:` winner: `world:chat` already exercises
every one of these paths 62 times a day.

## Candidate sources and the metric gate

"How unusual is this right now" must follow the existing pattern: a reading
scored against the same source's own 7 days, judged by `judge_candidate`.

### Event source: one table for all three acts

`substrate_durable_run_state` holds a terminal row (`node='finish'`,
`status` completed/failed) for every curiosity, reading and painting run.
Published on `orion:durable:run:state` (`services/orion-durable-runs/app/runner.py:798`),
persisted by sql-writer (`services/orion-sql-writer/app/models/durable_run_state.py:15`).
Event-to-row latency: median 25-40 ms (`created_at - generated_at`, 7 days).
Live counts, completed per day, 10-10..10-15:

| workflow | completed/day | failed/7d | notes |
|---|---|---:|---|
| curiosity.investigate | 12-19 | 2 | 50 `investigate` + 53 `self_inquiry` in 7 d |
| reading.turn | 4-94, bursty (82 and 94 on 10-14/15) | 38 | stage 1 and stage 2 rows per seed |
| reverie.visual | 15-22 | 22 | 82 of 116 on the `visual_baseline.v1` timer |

### Temporal Self arcs: rejected as the live source

Arcs are the cleanest *summary* of these acts, but they reach the database too
late. `temporal_self_event.ingested_at - occurred_at` median is 360-456 s for
curiosity/visual/chat events (1,484 s for `curiosity_run`), and an arc for a
run that ended at 07:05:50 first appeared at 07:12:31. That is past the 300 s
orienting window: an arc can only report old news. Reading has no arc kind at
all. Use the arc id as a *citation* later, not as the trigger.

### Gate, per source

**1. `world:curiosity` — belief change of a finished curiosity run. PASSES.**

1. *Provenance.* `orion/curiosity/value.py:211-310` (`RunOutcome`): the sum over
   priors this run tested of `KL(Bern(after) || Bern(before))`, from Hub's
   own confidence snapshots at turn start and end, counting only priors
   stamped with this run id. Written at finish by
   `services/orion-hub/scripts/curiosity_offer_decisions.py:354`
   (`record_run_outcome`, upsert at `:102`), called from
   `curiosity_investigation.py:1373`, into `curiosity_run_outcomes`. The row
   lands ~0.7 s before the durable `finish` row (median over 119 runs).
2. *Independence.* Curiosity, reading and painting write no prediction error
   onto any `node:substrate.*` node (producer survey; only `act:make`/`act:look`
   write `substrate_action_outcomes`, not a node). But timing overlaps: in the
   300 s after a curiosity run finishes, `node:substrate.execution` is
   eligible on 13% of ticks against 4% otherwise, and `biometrics` 10% vs 6%.
   So the *event* already pokes the body. The *quantity* does not: execution's
   error is cortex-exec trajectory error (`worker.py:4368`), with no access to
   prior confidences. Nats are new signal, and the only one carrying what the
   run found. Not derived from `world:chat`.
3. *Theory anchor.* Bayesian surprise, posterior against prior (Itti & Baldi
   2009), the same quantity `orion.autonomy.prediction.bayesian_surprise_nats`
   scores for dispatched actions. Attention to surprising *outcomes* of one's
   own actions, not to the actions themselves: the action is predicted
   (efference copy), the result is not. Caveat, stated: Orion writes the
   confidences itself. The number is "how far Orion says its mind moved",
   not ground truth.
4. *Live sanity.* 52 outcome rows in 7 days, all with a value. 33 of 52 are
   exactly 0.0 (tested, moved nothing): it rests, and a real 0 is a result,
   not an absence. p75 0.010, p90 0.047, max 0.853 nats. Not saturated, not
   flat. The 7-day count at any tick in the window is 47-53. In the 6-day
   replay window 45 runs finished, 34 at 0.0, and 3 reached band `high`
   (0.20, 0.72, 0.45 nats). Gap: the `self_inquiry` line (53 runs) writes no
   outcome row, so it cannot compete. Correct, not a bug: it has no priors to
   move.
5. *Existing mechanism.* `realized_nats` already exists and is already read by
   Orion's Day (`orion/orion_day/gather.py:107`) and by curiosity's own value
   model. No new metric is built; it is a new *consumer*.
6. *Reversibility.* Cheap. One candidate builder, one store read, one flag.
   Nothing persisted in a new schema except an additive optional field.

**2. `world:activity.reading` (rate of finished reading turns). FAILS step 3.**
Rate of Orion's own acts against its own week measures how busy Orion is. Busy
is self-caused and predicted; there is no theory that makes it news. The
replay confirms the harm: 1,536 wins in 6 days (253/day), no-winner share
64.2% -> 55.3%, displacing `node:substrate.chat` 51 times and every body node.
Reading bursts (82-94 turns/day on 10-14/15) would own attention whenever
Orion reads. The belief-change alternative is not available: reading's
`priors_tested` holds verdict labels (254 supported, 66 revised, 6 refuted,
36 untested in 7 days) with no confidences, so no nats. Resonance
(`orion/world_pulse_read/resonance.py`) is eval-only and not persisted.

**3. `world:activity.painting` (rate of finished visual runs). FAILS step 3.**
82 of 116 paintings in 7 days were requested by the `visual_baseline.v1`
timer: their finishing is predicted by construction. The replay: 211 wins
(35/day), and it is the only variant that displaced `world:chat` (2 ticks).
The act-ledger surprise (`substrate_action_outcomes`, `dispatch_kind='act:make'`,
`surprise_nats`) is the right future measure, but it started 2026-10-16 and has
4 rows: fails step 4 (no history). Revisit at >= 20 rows in 7 days.

**4. `world:activity.curiosity` (rate of finished curiosity runs). FAILS step 3.**
Same busyness argument. Replay: 140 wins (23/day), mostly in the overnight
curiosity window.

**5. Vision presence transitions (`vision_presence_transition`). DEFERRED, fails step 4.**
Producer `services/orion-vision-window/app/presence.py:518-537`. Only since
2026-10-14 (no 7-day history). `identity_confirmed` is false on every row and
`label` is empty, so the winner could only say "someone". present<->recent
flapping is ~1,000 of 1,282 rows. Also overlaps `node:substrate.perception`
(camera surprise already competes). A real world event; needs identity and
debounce first.

**6. Juniper's message content.** Not a new candidate. `world:chat` already
competes on rate. The gap is what it carries (a count). Second patch: carry
her latest message excerpt as evidence on `world:chat` (below). No metric
change, so no gate needed beyond the evidence cap.

**7. Temporal Self arcs.** Rejected as the trigger (latency above). Kept as a
citation target.

## Replay (last 6 days, real `judge_candidate`)

Method: every stored broadcast tick keeps its traced candidates and verdicts
exactly. The new candidate is built at the tick's time from events that had
happened by then and judged with the real `judge_candidate`. The winner is
re-picked with `rank_candidates`' key. With no new candidate the script
reproduces the stored winner on 11,976 of 11,976 ticks. Each variant runs
alone. Not modelled: the yield-to-Juniper rule (it changes nothing here:
`world:curiosity` never coincided with an eligible `world:chat`).

| variant | gate | ticks eligible | ticks won | won/day | no-winner share | displaced |
|---|---|---:|---:|---:|---|---|
| baseline | — | — | — | — | 64.17% | — |
| **world:curiosity (nats, event, 300 s)** | **pass** | 21 | 13 | 2.1 | 64.11% | 7 no-winner, 5 execution, 1 biometrics |
| rate: curiosity runs / 15 min | fail | 143 | 140 | 23.0 | 63.19% | 117 no-winner, 9 biometrics, 8 execution, 3 perception, 2 bus_synaptic, 1 route |
| rate: reading turns / 15 min | fail | 1,791 | 1,536 | 252.7 | 55.29% | 1,064 no-winner, 118 perception, 84 execution, 80 biometrics, 60 bus_synaptic, 53 route, 51 node:substrate.chat, 19 cabinet, 7 codebase |
| rate: paintings / 15 min | fail | 218 | 211 | 34.7 | 63.07% | 132 no-winner, 27 bus_synaptic, 14 biometrics, 12 perception, 8 codebase, 7 cabinet, 6 execution, 3 route, **2 world:chat** |

`world:curiosity` wins came from 3 runs (10-13 x2, 10-16 x1). What Orion would
have attended to instead:

1. **2026-10-13 07:39:42**, instead of `node:substrate.execution`. Run
   `882aa3553cf9`, 0.20 nats: "Picked the one with a clock on it. The
   cast-alternation prior (`self:cast_keeps_alternating_not_settling_20261011`)
   had a pre-registered decider…"
2. **2026-10-13 09:09:31**, instead of no winner. Run `384ddfdb9132`, 0.72 nats:
   tested `self:chat_seat_demand_collapse_is_upstream_20261008` and dropped it
   0.65 -> 0.08. "All of it is in the graph and read back: role, three hops, the
   refutation with its revision trail (0.65 -> 0.08, open -> refuted)…"
3. **2026-10-16 07:06:06**, instead of `node:substrate.biometrics`. Run
   `c6fe2d16ca0e`, 0.45 nats: tested
   `self:rung6_and_covariates_merged_after_my_last_definition_20261016`,
   0.55 -> 0.10, and formed a new prior. "She pointed at the merged PRs… four
   merges (#2714-2717)… The twist: the prior was already stale when it formed."

For contrast, the reading-rate variant's wins are reading summaries ("Stage 2
confirmed and sharpened the Stage 1 blocker. The tiiuae/Falcon-ASR-1.6B repo is
gated…") winning over `node:substrate.chat` because Orion read 2 articles in
15 minutes. Content-rich, but not unusual in any sense the rule can defend.

Reproduce:

```bash
# exports: see the script docstring (SELECT-only)
/mnt/scripts/Orion-Sapienform/.venv/bin/python \
  scripts/analysis/replay_attention_world_candidates.py frames.jsonl events.jsonl
```

## What the winner carries downstream

A bare label repeats the reverie garbage-in problem. The candidate carries
evidence, capped, every item with a citation id:

| field | source | cap |
|---|---|---|
| `label` (becomes loop description, <=120 chars, stable per run) | "Curiosity <run_id[:8]> moved my belief that <prior short> from 0.65 to 0.08" (largest-nats prior) | 120 chars |
| `evidence_refs` | `world:curiosity`, `curiosity_run:<run_id>`, `journal:<journal_entry_id>`, `prior:<prior_id>` per moved prior | 8 refs |
| `world_evidence.priors` | `curiosity_run_outcomes.per_prior` rows with `kind`, `before`, `after`, `nats` | top 3 by nats |
| `world_evidence.finding_excerpt` | `substrate_durable_run_state.detail.finding_text`, whitespace collapsed | 400 chars |
| `world_evidence.realized_nats`, `completed_at`, `run_id` | outcome row | — |

Plumbing needed, all in the first patch:

- `AttentionCandidateV1.evidence: dict | None = None` (additive optional field on an
  `extra="forbid"` model; built and consumed in-process, so no cross-service
  rollout order beyond rebuilding `orion` in substrate-runtime and
  attention-runtime together).
- `world_first_signals` passes `cand.evidence_refs` (deduped, capped) instead of
  only `[cand.source_id]` (`attention_broadcast.py:421`), and puts
  `cand.evidence` into signal `provenance["world_evidence"]`, which
  `scoring.py:179` already copies onto the loop.
- Reverie `_open_loops_for_prompt` / `_loop_signal` (`services/orion-thought/app/reverie.py:295-330`)
  renders `world_evidence`: the moved priors with before/after, and the
  excerpt, marked as Orion's own recorded finding.

## Proposed schema / API changes

- `orion/schemas/attention_candidate.py`: `WORLD_CURIOSITY_SOURCE_ID = "world:curiosity"`;
  `AttentionCandidateV1.evidence: dict[str, Any] | None = None`;
  `SELF_CAUSED_WORLD_SOURCE_IDS = frozenset({WORLD_CURIOSITY_SOURCE_ID})`.
- `orion/attention/world_first.py`:
  - `CURIOSITY_MIN_RUNS_7D = 20` (7-day count is 47-53 today; below 20, band
    `insufficient_history`).
  - `CURIOSITY_OUTCOMES_SQL`: `curiosity_run_outcomes` left-joined to the
    `curiosity.investigate` finish row in `substrate_durable_run_state` for
    `finding_text` and `journal_entry_id`, `realized_nats IS NOT NULL`,
    `completed_at >= :since`.
  - `curiosity_candidate(outcomes | None, *, now, event_decay=True)`: newest
    outcome's nats against the 7-day outcomes, `compute_prediction_error_magnitude`
    + mid-rank, `source_kind="external"`, `value_kind="level"`,
    `event_window_sec=EVENT_ORIENTING_WINDOW_SEC`. `None` = read failed = absent,
    never calm.
  - `rank_candidates`: after judging, a candidate in `SELF_CAUSED_WORLD_SOURCE_IDS`
    is moved to ineligible with reason `"yields to Juniper (world:chat eligible)"`
    when `world:chat` is eligible. Traced like any other verdict.
- No bus channel, no table, no registry entry: the candidate is in-process
  and appears in the existing `frame.debug.world_first` trace.
- Env (substrate-runtime): `ATTENTION_WORLD_CURIOSITY_ENABLED=true` (default
  on, Juniper's ship-on rule). `false` = today's candidate set exactly.

## Files likely to touch (first patch)

- `orion/schemas/attention_candidate.py`: id constant, `evidence` field, self-caused set.
- `orion/attention/world_first.py`: SQL, `curiosity_candidate`, yield rule.
- `orion/substrate/attention_broadcast.py`: pass refs and evidence through.
- `services/orion-substrate-runtime/app/store.py`: `fetch_curiosity_outcomes(since)` (next to `fetch_chat_turn_times`, `:831`).
- `services/orion-substrate-runtime/app/worker.py:3036-3050`: append the candidate behind the flag.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `README.md`: flag.
- `services/orion-thought/app/reverie.py`: render `world_evidence`.
- Tests: `orion/attention/tests/` (or the existing world_first test module), substrate-runtime worker test, reverie prompt test.
- Eval: `scripts/analysis/replay_attention_world_candidates.py` (this PR) as the periodic eval.

## Non-goals

- No reading or painting candidate. They fail the gate today (see above).
- No rate-of-own-activity candidates, ever, without a new theory anchor.
- No field-contest (`orion-attention-runtime`) change. Its consumers already
  drop world targets (`goal_provenance.py:54-110`, `proposals/builder.py:54-95`).
- No change to band cuts, the 300 s window, or chat scoring.
- No Temporal Self change. Arcs stay a citation, not a trigger.
- No new action from a world winner. The three existing loop breakers stay.

## Proposal-mode items

- **Capability change.** Orion can attend to "I just learned something that
  surprised me", with the specific belief and finding in hand. Reverie and the
  chat "recent attention" cue can then speak about it.
- **Data touched.** Read-only: `curiosity_run_outcomes`, the curiosity finish
  row in `substrate_durable_run_state`. Written: nothing new; the existing
  broadcast log/projection rows carry one more candidate and its evidence.
- **Privacy boundary.** There is no privacy boundary between Juniper and Orion
  (Juniper's rule). What leaves the box: the finding excerpt (<=400 chars) can
  reach the reverie LLM prompt. Reverie runs on Orion's local lanes, the same
  path `finding_text` content already reaches via the journal. Nothing new goes
  to an external API. UNVERIFIED: which lane reverie narration is routed to
  today; check before deploy.
- **Trace that proves it worked.** In `substrate_attention_broadcast_log`:
  `frame.debug.world_first.candidates[]` contains `world:curiosity` with a
  reason on every tick; within 300 s of a top-decile run, `winner =
  'world:curiosity'` and the selected loop's `source_refs` includes
  `curiosity_run:<run_id>`; the next reverie thought for that tick cites the
  run id or prior id. A log line `substrate_world_first_curiosity
  run=<id> nats=<v> band=<b>` per new run.
- **Dangerous failure modes and their mechanisms.**
  1. *Orion's own activity crowds out Juniper.* Mechanism: the yield rule makes
     `world:curiosity` ineligible whenever `world:chat` is eligible. Tested.
     Replay: 0 chat displacements even without it.
  2. *Self-reinforcing loop: attending to curiosity spawns more curiosity.*
     Mechanisms already in code, each tested by the patch: reverie proposals
     drop world-only coalitions (`orion/reverie/proposal.py:66`); endogenous
     curiosity skips loops with no `node:` ref (`endogenous_curiosity.py:370-375`);
     world-action binding refuses non-node winners (`self_shed.py:56-98`); the
     goal bridge filters `node:%` (`orion-attention-runtime/app/store.py:697-750`).
     The candidate is event-written: one run can win only in the 300 s after it
     finishes, and cannot re-arm without a new finished run.
  3. *Goodhart on nats.* Orion writes the confidences. If narration made big
     swings feel rewarded, flip-flopping priors would win. No learning path
     connects attention to confidence writing today. Watch: share of
     `world:curiosity` wins whose top prior reverses a previous move by the
     same prior (`curiosity_run_outcomes.per_prior` history). Kill switch below.
  4. *Silent outcome writer.* If Hub stops writing outcomes, the candidate goes
     `insufficient_history` after the 7-day count drops below 20, and absent on
     read failure. Never "calm".
  5. *Stale label loop id.* The label must not contain counts or ages, or the
     loop id churns (the bug `world:chat` has today). Test it.
- **Disable / rollback.** `ATTENTION_WORLD_CURIOSITY_ENABLED=false` in
  `services/orion-substrate-runtime/.env`, then rebuild/restart
  substrate-runtime. Default ON. The flag removes the candidate entirely; the
  rest of the contest is untouched.

## Missing questions

1. Should Juniper always outrank Orion's own news? This doc says yes (the
   yield rule). It is a policy choice, not a measurement.
2. Is 3 wins in 6 days the size Juniper wants? It is the honest size under the
   top-tenth rule. Making it bigger means a lower cut for this source only,
   which is a knob, not a finding.
3. Should reading stage 2 write prior confidences (before/after), so reading
   can earn a nats measure? That is the path for reading; it is a reading
   prompt/schema change, out of scope here.
4. When `act:make`/`act:look` surprise reaches 20 rows a week, is it the
   painting/reading measure? Needs its own gate (what is its prediction?).

## Acceptance checks

- Unit: `curiosity_candidate` with an all-zero week and a new 0.0 run is
  band `quiet`, not eligible; a new run above the week's 90th percentile is
  eligible for 300 s and not at 301 s; read failure is absent; fewer than 20
  runs is `insufficient_history`.
- Unit: yield rule: with `world:chat` eligible, `world:curiosity` is
  ineligible with the stated reason; without it, the ranking is unchanged.
- Unit: the label is identical across ticks for the same run (stable loop id).
- Unit: `world_first_signals` carries `curiosity_run:<id>` in `evidence_refs`
  and `world_evidence` in loop provenance.
- Unit: a `world:curiosity`-only coalition yields no reverie proposal and no
  endogenous-curiosity seed.
- Reverie prompt test: a world:curiosity loop renders before/after and the
  excerpt, capped.
- Flag off: the frame is identical to today's on a fixed fixture.
- Eval: the replay script reproduces stored winners on 100% of ticks with the
  candidate off, and reports the candidate's wins with it on.
- Live (after deploy): within a week, at least one `world:curiosity` winner in
  `substrate_attention_broadcast_log` within 300 s of a top-decile run, and a
  reverie thought citing that run. Until then: UNVERIFIED.

## Recommended next patch

1. **First (this proposal):** `world:curiosity` in the substrate broadcast, with
   evidence pass-through, the yield rule and the flag. Smallest slice that lets
   one real act of Orion's compete on what it found.
2. **Second:** carry Juniper's latest message as evidence on `world:chat`
   (excerpt <=200 chars + `chat_history_log` id), and drop the count from its
   label so its loop id stops churning.
3. **Later, each behind its own gate:** reading (needs confidences), painting
   (needs act-ledger history), presence (needs identity + debounce).

## Gate record (summary)

| source | provenance | independent | theory | live sanity | exists already | reversible | verdict |
|---|---|---|---|---|---|---|---|
| curiosity nats | value.py:211-310, offer_decisions.py:354 | yes (timing overlaps execution; quantity new) | Bayesian surprise | rests at 0 (33/52), tail to 0.85 | yes, new consumer | cheap | **pass** |
| curiosity rate | durable finish rows | yes | none (busyness) | ok | — | — | fail (3) |
| reading rate | durable finish rows | yes | none (busyness) | bursty | — | — | fail (3), harmful in replay |
| painting rate | durable finish rows | yes | timer-driven | ok | — | — | fail (3) |
| act:make surprise | act_ledger.py:264 | yes | Bayesian surprise (UNVERIFIED what it predicts) | 4 rows | yes | — | fail (4), revisit |
| presence | presence.py:518 | overlaps perception | orienting to arrival | 2.5 days, no identity, flappy | — | — | deferred (4) |
| Temporal Self arcs | chronicle | — | — | 6-25 min latency | — | — | rejected as trigger |
