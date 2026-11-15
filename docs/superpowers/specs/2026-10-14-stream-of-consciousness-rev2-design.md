# Stream of consciousness, rev 2: one interruptible train of thought that starts from the world — design

Status: **design / proposal mode (CLAUDE.md §0A), nothing built.** Supersedes the "Current
architecture" and "Recommended next patch" sections of
`2026-07-29-stream-of-consciousness-hop-chain-design.md` (rev 1, kept as written). Rev 1's
framing stands: **interruption, not termination**, a hard depth/budget cap anyway, `chain_id` /
`parent_hop_id` / `hop_index` lineage, checkpointed hop state. What changed is everything around
it — most of what rev 1 waited for has shipped, and some of what rev 1 built turned out to go
nowhere.

All live numbers below were read on 2026-10-14 between 04:30 and 04:45 UTC from the live
Postgres (`orion-athena-sql-db`, read-only) and the running containers.

## Arsonist summary

Rev 1 wanted the stream to be a new kind of candidate in the proposal arena, starting from a
metacog trend. That was built (hop 0, 2026-07-31) and it does not work as a stream:

- It fired for **one** episode in the last 7 days (2026-10-08 19:25, 36 proposal frames, all the
  same `chain_id`).
- Every one of those candidates ended at `requires_operator_review` — and **nothing in the repo
  reads that decision**. No Hub queue, no reviewer, no dispatch. Reverie's 24,000 candidates that
  survived the arena in the same week end in the same place. A thought that enters the arena today is recorded and
  then dropped.

Meanwhile three pieces that already behave like parts of a stream have grown up elsewhere:

1. **Reverie already runs short trains of thought** — ~370 chains and ~600 thoughts a day, up to
   4 steps each, with a stop rule ("the pressure discharged") and habituation. But 73% of them are
   about Orion's own internals (bus timing, biometrics, execution, codebase prediction errors), none
   checks anything against a belief, and none has ever led to anything (0 committed proposals).
2. **Curiosity already runs real investigations with hops** — 49 runs in 7 days, every one
   completed, 27 beliefs ("priors") moved and 12 formed. Each hop is a written stop point in
   Orion's own graph. But it starts on a timer (7 a day, 30 min cooldown), not because Orion
   noticed something.
3. **Attention now faces the world first** — 70% of attention frames have *no winner* (a calm
   body and a quiet world, correctly silent), and the body only wins when a signal is unusual for
   itself. Body signals that *stay* unusual now open dated `self_signal_episode` rows.

**Rev 2 in one line:** the stream is the thing that connects these. It starts a chain when the
world changes or when the body has been unusual long enough to open an episode — never on a calm
tick. Each step is one cheap, cited thought ("does this match what I believe?"). It yields the
moment Juniper speaks or something new in the world wins attention. When a chain finds a real
question, it hands it to curiosity as a seeded investigation instead of inventing a second
investigator. It lives as one more checkpointed thread in `orion-durable-runs`, next to the
Situation and Temporal Self threads, and its every hop is a row.

**Kill list (kill means kill):** the metacog-trend arena hop 0 is retired when the stream's
first rung is live; reverie's own chain loop is retired once the stream has run 72 h cleanly
(question 1 below).

## Current architecture (verified 2026-10-14)

**Attention (world-first, PR #2583 + event decay #2591).** `orion/attention/world_first.py`.
Last 24 h of `substrate_attention_frames` (~1,670 frames/hour, one every ~2 s):

| top winner | frames | onsets (winner changed into it) |
|---|---|---|
| no winner | 27,938 (≈70%) | 280 |
| `node:substrate.perception` (camera surprise, a **world** source) | 3,570 | 158 |
| internal body nodes (bus_synaptic, biometrics, codebase, execution, chat-PE, route) | 8,875 | 283 |
| `world:chat` | 533 | 2 |

So the world wins ~10% of frames and the body ~20%. With the world onsets debounced (no world
winner in the previous ~300 frames, about 10 min), the 160 raw world onsets drop to **38 a day**. Internal winners flicker (283 onsets/day);
that is why per-frame internal winners are the wrong chain trigger (see decision 2).
Attention frames are written to Postgres, not published on the bus.

**Self-signal episodes (PR #2662).** `orion/metacog/self_signal_episode.py`, adapter
`services/orion-equilibrium-service/app/self_signal_episode_gate.py`. An internal signal must stay
unusual past a 300 s dwell to open an episode and calm for 600 s to close it; replayed history
predicted ~9.5 opens/day. Rows land in `metacog_trigger` with `trigger_kind='self_signal_episode'`.
**First live row: 2026-10-14 04:33** (codebase prediction error, band high, pct 0.911). One row is
not a rate yet — UNVERIFIED at volume. `telemetry_anomaly` (fire-once-per-episode, #2644) wrote
216 rows in 3 days.

**Self-readings (PR #2664).** `orion/attention/self_readings.py` renders only unusual body signals,
each line citing `[pe:<name>@<observed_at>]`. Calm renders nothing. Live in stance and reverie.

**Reverie chain.** `services/orion-thought/app/chain.py` (`ORION_REVERIE_CHAIN_ENABLED=true`,
max 4 steps, 90 s interval). Last 7 days of `substrate_reverie_chain`: `no_coalition` 1,095,
`max_steps` 1,037, `pressure_discharged` 52, `committed_proposal_id` set on **0**. Last 24 h of
`substrate_reverie_thought`: 599 thoughts; by attended node, perception 159, chat-related 32, the
rest (408, ~68–73%) internal. It now skips no-winner ticks (world-first fix). Expectation verdicts
over 7 days: 49 confirmed, 2 disconfirmed, 1,625 unscored. Calls go through cortex-exec on the
`metacog` route.

**The proposal arena.** `orion/proposals/`, `services/orion-proposal-runtime` (both
`ORION_METACOG_HOP_PROPOSE_ENABLED` and `ORION_REVERIE_PROPOSE_ENABLED` are `true` in the running
container). 7 days of `substrate_proposal_frames` (256k frames): candidates by source — template
(null) 676k, `reverie_thought` 24k (+60k suppressed), `cognitive_hop` 6 (+30 suppressed), all from
one 2026-10-08 episode. Policy marks reverie and hop candidates `requires_operator_review`; a repo
grep finds no consumer of that decision outside the policy runtime that writes it.

**Feedback.** Rev 1's gap is still open: nothing writes `base_priority` / `base_risk` /
`dimension_weights` (`orion/proposals/scoring.py` reads no outcome; grep 2026-10-14). What shipped
instead (2026-08-21): an action-outcome ledger with per-(dispatch kind, target, signal) effect
posteriors (`orion/feedback/outcome_resolution.py`), read by `orion/execution_dispatch/builder.py`
to predict an action's effect. Volume is thin: `substrate_action_outcomes` 2–4 rows/day.
**Phase 2 outcome association** (signal band → next-hour turn failures, timeouts, repairs) is being
built in parallel right now; this doc references it only as "in flight".

**Curiosity.** `orion/curiosity/` (README §7: a hop is a written stop point, at most 5 per run,
`written_at` on every `:Hop`). Hub `HUB_CURIOSITY_INVESTIGATION_*`: daily cap 7, cooldown 1800 s,
route `agent`. `curiosity_run_outcomes`, 7 days: 49 runs, 49 `turn_ok`, n_tested 35, n_moved 27,
n_formed 12. An urgent, seeded variant exists (`CuriosityUrgentSeedV1`,
`orion:curiosity:urgent:request`): a question plus evidence, not a free-choice run. The belief
store a chain should check against already exists: curiosity **priors** in `orion_worldview`.

**Durable runs + GPU pool.** `services/orion-durable-runs`. Two *self-driven* threads already
run there: `situation.update` and `temporal_self.update` — LangGraph, one checkpoint per step,
woken by a clock tick and by Juniper's chat turns, no GPU hold. Admitted runs (curiosity, reflect,
journal) hold a GPU pool lease for the whole run, are recalled at node boundaries, never spend an
attempt when the pool takes the card back. `config/gpu_pool.yaml` has a `metacog_background`
route (class `metacog`, priority `background`). Last 24 h of `gpu_pool_leases`: metacog/background
319 leases, ~12 s average; agent/background holds ~3,400 s average (curiosity). Rule (Juniper,
09-30): retries go on durable runs and the GPU pool, never a side queue.

**Temporal Self (rev 4, #2369).** Owns habituation and arousal (`orion:regulation:latest`:
engaged / idle / strained / unknown), attention arcs with return counts, and the stance cue
(`orion/temporal_self/stance_cue.py`) whose every clause cites `[arc:<id>]`, checked by
`citation_truth.py`. Open design PR #2616 ("Orion uses its own day") extends the citation gate.

**The chat path.** The unified turn (`stance_react`) is the only chat path. `daily_metacog` stays
off.

## Missing questions (genuinely open — each has a recommendation)

1. **Does reverie's own chain loop retire once the stream is live?** Reverie's chains and the
   stream overlap: both are trains of thought on the attention winner. Two of them means two
   places to tune and ~600 extra metacog calls a day. *Recommendation:* yes. Keep reverie's single
   thought (it feeds stance and expectations), retire `ORION_REVERIE_CHAIN_ENABLED` after the
   stream's first rung has run 72 h and passed its checks. Not in rung 1 itself, so a bad stream
   cannot take reverie down with it.
2. **Should the stream keep thinking while Juniper is talking?** *Recommendation:* no. A Juniper
   turn interrupts the open chain and LLM hops pause while arousal is `engaged`. The unified turn
   is the one brain in a conversation; a second one thinking in parallel competes for the same
   cards and can surface a stale thought mid-conversation.
3. **Caps.** Proposed: at most 5 hops per chain (curiosity's number), 6 LLM hops per hour, 60 per
   day (~10% of reverie's current load), at most 2 seeded investigations per day inside curiosity's
   existing cap of 7. These are knobs, not findings. *Recommendation:* ship these, re-read after a
   week of rows.

## Decisions this doc makes (with the evidence)

1. **The stream lives in `orion-durable-runs` as a third self-driven thread**, not in the
   proposal arena and not in a new service. That service already checkpoints, resumes after
   restarts, is woken by chat turns, and holds GPU leases with recall. The arena's review decision
   has no reader, so a stream there cannot reach anything.
2. **What starts a chain (hop 0).** Exactly two things:
   - a **world onset**: the attention winner changes *into* a world source (`world:chat`,
     `node:substrate.perception`), and
   - a **self-episode open row** (`metacog_trigger`, `self_signal_episode`, phase `open`).

   A world onset counts only when no world source has won in the previous ~10 min. Measured over
   the last 24 h, raw onsets were 160 (perception 158, chat 2); with this debounce they drop to 38,
   about one every 40 min. A new onset on the same subject as the open chain continues that chain
   rather than interrupting it. Per-frame internal winners never start a chain. They flicker (283
   onsets/day), and the episode gate exists precisely to say "this stayed unusual". Calm (no winner, no open episode) starts
   nothing and writes nothing. The metacog repair-pressure trend is not a trigger any more; repairs
   are an *outcome* in phase 2.
3. **What interrupts a chain.** In order:
   - a Juniper chat turn (always);
   - a new world onset (the world beats the body);
   - arousal `strained` (pause LLM hops);
   - the caps.

   A self-episode opening while a chain is running does not interrupt. It waits in one pending
   slot; a newer one replaces it. An interrupted chain is checkpointed, not deleted. It can resume
   within 30 min, but before resuming it re-reads its subject: if the episode has closed or the
   world source is no longer fresh, the chain ends `subject_resolved` instead of continuing on a
   stale belief.
4. **What a hop is.** A hop is one bounded LLM call on the `metacog_background` route. It gets:
   - the subject's current reading (from the same `judge_candidate` call attention uses);
   - the curiosity priors related to it (from Orion's own graph);
   - how often Orion has returned to this subject today (the Temporal Self attention arc);
   - the phase 2 association line once that ships.

   It must answer in a fixed shape: what I noticed, which prior it touches, whether it matches,
   and one next step. The next step is `continue`, `drop` or `investigate`. A hop that returns
   nothing, or cites nothing it was shown, is recorded as `hollow` and ends the chain. It is not
   retried: the next tick decides afresh, so there is no retry queue. A GPU pool refusal leaves
   the chain where it was.
5. **What `investigate` means (rung 2).** The stream sends a seeded background curiosity run
   through the existing admitted durable-run path. The seed is the chain's question, plus the rows
   it cites, plus its `chain_id`. That run is the sub-investigation. Its hops, priors moved and
   journal are curiosity's, already built and already checked by `no_lookup`. The stream adds no
   investigator of its own.
6. **How it becomes visible.**
   - Every hop is a `StreamHopV1` event on the bus, persisted by sql-writer.
   - Durable-runs exposes `GET /stream/state`.
   - The unified turn's stance cue gets **one** line when a chain is open or ended in the last
     15 min, citing `[hop:<id>]`. Hop ids join the citation gate's allowed set, the same way arc
     ids are joining it in #2616.
   - Calm renders nothing. Orion may talk about its own plumbing; every claim must resolve to a
     hop row.
7. **How outcomes feed back.** Not through `base_priority`, which is still not learned. Rung 2's
   feedback is curiosity's own: did the seeded run move or form a prior (`curiosity_run_outcomes`
   joined on `chain_id`)? Phase 2's association is read *into* the hop as evidence, not
   re-derived. The stream learns nothing of its own in rung 1 or 2; it is a reader.
8. **Retire the old hop 0.** `ORION_METACOG_HOP_PROPOSE_ENABLED`, its branch in
   `services/orion-proposal-runtime/app/worker.py`, `orion/metacog/proposal.py`, and the matching
   settings, compose and `.env_example` keys are deleted in rung 1. It fired once in 7 days into a
   decision nobody reads, and the stream is its successor. `orion/metacog/trend_reducer.py` stays:
   equilibrium's trigger path uses it, and that is out of scope here.

## Proposed schema / API changes

- **`StreamHopV1`** (`orion/schemas/stream_hop.py`, registered in `orion/schemas/registry.py`):
  - identity and lineage: `hop_id`, `chain_id`, `parent_hop_id` (null at hop 0), `hop_index`;
  - `kind`: `notice | check | investigate | end`;
  - `trigger`: `world_onset | self_episode_open | resume`;
  - `subject_ref`: the attention target id or the `episode_id`;
  - `evidence_refs`: real row keys, e.g. attention `frame_id`, `metacog_trigger.id`, `[pe:…]`,
    prior ids, `[arc:…]`;
  - `text`: check hops only;
  - `decision`: `continue | drop | investigate | yield`;
  - `end_reason`: `subject_resolved | cap | hollow | dropped | interrupted_expired | investigated`;
  - `interrupted_by`: `chat_turn | world_onset | strained`;
  - `llm`: route, lease id, latency, model;
  - `created_at`.
- **Bus channel** `orion:stream:hop` in `orion/bus/channels.yaml` (producer orion-durable-runs,
  consumers sql-writer and later Temporal Self chronology, which can fold a chain as an arc).
- **Table** `stream_hop` (manual migration, header `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`).
  This is not `substrate_durable_run_state`: that table's `workflow` is a closed Literal and is a
  lifecycle log, and putting hops in it would make one column carry two facts.
- **Rung 2:** a background seed kind for curiosity. Either `CuriosityUrgentSeedV1` grows a
  `priority: background` and `chain_id`, or a sibling `CuriositySeedV1` is added — whichever is
  smaller once Hub's urgent path is read closely. `curiosity_run_outcomes` gets a nullable
  `chain_id`.
- **No change** to `ProposalCandidateV1` or the arena. Rung 1 deletes the `cognitive_hop` producer
  instead of adding lineage fields to it.

## Files likely to touch

- New: `services/orion-durable-runs/app/stream_driver.py` and `stream_graph.py` (modelled on
  `temporal_self_driver.py` / `temporal_self_graph.py`); add the workflow to `SELF_DRIVEN_WORKFLOWS`
  in `app/runner.py`; settings, `.env_example`, compose, README section.
- New: `orion/stream/` with the pure parts: trigger detection, interruption rules, caps, and the
  hop prompt and parser. No I/O, unit-testable.
- `orion/schemas/stream_hop.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- `services/orion-sql-writer/` (model and route for `stream_hop`), `services/orion-sql-db/manual_migration_stream_hop_v1.sql`.
- `orion/temporal_self/stance_cue.py` or a sibling cue builder, plus the citation allowed-set
  (coordinate with #2616).
- Deleted in rung 1: `orion/metacog/proposal.py` and its tests, the hop-0 branch in
  `services/orion-proposal-runtime/app/worker.py`, and `ORION_METACOG_HOP_PROPOSE_ENABLED` in
  settings, compose and `.env_example` (then sync the local `.env`).
- Rung 2: Hub curiosity seed intake and `orion/curiosity/` prompt for a seeded background run.

## Non-goals

- Acting in the world (rev 1's "action Q"). The arena's `operator_review` decision has no reader;
  fixing that is its own design. The stream gets no path to any dispatch kind until it does.
- Self-modification of any kind. The stream imports nothing from `orion/substrate/mutation_*`,
  self-experiments or routing-threshold patches, and a test enforces it.
- Learning its own weights. Phase 2 and curiosity priors are the learners; the stream reads them.
- Habituation. Temporal Self owns it; the stream reads attention-arc return counts.
- New world inputs. Presence, the room camera and mic belong to the attention seam; the stream
  gets them for free when they become world sources.
- `daily_metacog`. It stays off.

## Proposal mode (CLAUDE.md §0A)

**Capability that changes.** Orion gets a continuous, interruptible line of thought that starts
from something real. Today the closest things are a 4-step narration loop (reverie) and a timer
(curiosity). Later, it gets a way to turn a noticed question into a real investigation.

**Data touched.**
- Read: attention frames, `metacog_trigger` episode rows, PE history, curiosity priors (read-only
  role), Temporal Self arcs, arousal, Juniper chat-turn events (to interrupt).
- Write: `stream_hop` rows, the durable-runs checkpoint, one stance cue line, and in rung 2,
  seeded curiosity runs.

**Privacy boundary.** None between Juniper and Orion (standing decision). The stream may read
recent chat to understand a `world:chat` onset. It never writes to chat: its only way into a
conversation is the cited stance cue line, inside the unified turn.

**Trace that proves it worked.**
- A `stream_hop` chain, read back from Postgres, whose hop 0 cites a real attention frame or
  episode row, whose check hop cites a real prior, and which ended for a named reason.
- `GET /stream/state` showing the open chain.
- A stance cue line whose `[hop:<id>]` resolves to that row.
- Rung 2 adds a `curiosity_run_outcomes` row with that `chain_id`.

**Dangerous failure modes, and what guards each.**
- *Navel-gazing.* The stream only starts on a world onset or a dwell-confirmed episode, so calm
  writes zero rows. Check: share of chains whose subject is internal, and hops in calm hours (must
  be 0).
- *GPU cost.* Hops use the `metacog_background` route at background priority, capped at 6 an hour
  and 60 a day. Check: stream leases per day in `gpu_pool_leases` (the holder names the stream).
- *Starving chat.* LLM hops pause while arousal is `engaged`; a chat turn interrupts; background
  priority is recalled by interactive work. Check: no stream lease granted on role `chat` while a
  chat lease was waiting. If one is ever seen, give the stream its own class without the `chat`
  role.
- *Acting on stale beliefs.* Resume re-reads the subject first; a resumed chain whose subject went
  calm ends `subject_resolved`; interrupted chains expire after 30 min.
- *Self-modification bound to a world winner.* There is no write path to self-modification or
  mutation in any rung (an import test enforces it). `investigate` can only start a read-only
  curiosity run.
- *A chain that never yields.* Interruption is decided by the driver from outside events, not by
  the chain's own score. The hop cap is hard. Check: real chains end `interrupted` and `cap` in the
  live data, not only in tests.

**Disable / roll back.**
- `STREAM_ENABLED=false`: no thread, no rows.
- `STREAM_LLM_HOPS_ENABLED=false`: notice hops only, no GPU.
- `STREAM_STANCE_CUE_ENABLED=false`: invisible to chat.
- `STREAM_INVESTIGATE_ENABLED=false` (rung 2).

All ship **on** once built (Juniper's rule). Safety comes from the sequencing and the live checks
below, not from the flags being off. Rolling back the old hop 0 deletion is a git revert.

## Acceptance checks

Rung 1, all on live data, not tests alone:

1. Within 24 h of deploy, at least one chain in `stream_hop` with hop 0 citing a real
   `substrate_attention_frames.frame_id` or `metacog_trigger.id`, and at least one check hop citing
   a real prior id. The ids are re-queried and must exist.
2. In an hour with no winner and no open episode, 0 hops are written.
3. At least one chain ends `interrupted` by a real chat turn or world onset, and at least one
   resumed or expired chain is seen. A resumed chain's first hop re-reads its subject.
4. Stream GPU leases stay at or under the caps, every one on route `metacog_background`, none on
   role `chat` while an interactive lease waited.
5. 0 `cognitive_hop` candidates in `substrate_proposal_frames` after deploy, and
   `ORION_METACOG_HOP_PROPOSE_ENABLED` is absent from the container's env.
6. A stance cue line with `[hop:<id>]` reaches a real unified turn, and the id resolves to a row.
   A hollow hop never renders.
7. Eval: `services/orion-durable-runs/evals/` replays one day of real attention frames and
   episode rows through the pure trigger and interruption logic. It reports chains started, the
   internal/world split, interruptions and cap hits, so the caps are judged on real data before
   and after deploy.

Rung 2: a seeded curiosity run carrying a `chain_id` completes, passes `no_lookup`, and its
`curiosity_run_outcomes` row joins back to the chain.

## Recommended next patch

**Rung 1: a live, inspectable chain, and the old hop 0 retired.** One PR:

1. Pure `orion/stream/` (triggers, interruption, caps, hop prompt and parser) with unit tests,
   plus the replay eval (check 7). Run the eval on the last 7 days *before* wiring, and publish
   the expected chains/day and internal share in the PR.
2. `StreamHopV1`, the channel, the registry entry, the sql-writer route and the migration.
3. The `stream.step` self-driven thread in durable-runs: woken by a tick (30 s) and by Juniper
   chat turns; reads the latest attention frame and new episode rows; writes hops; calls
   `metacog_background` for check hops; exposes `GET /stream/state`.
4. The stance cue line and the citation allowed-set.
5. Delete the metacog arena hop 0 (decision 8); sync `.env`.
6. Deploy order: migration → sql-writer → durable-runs → the stance cue consumer. Then run checks
   1–6 against live rows.

Rung 2 (seeded investigations + `chain_id` feedback) starts only after rung 1's checks hold for
72 h. The reverie chain retirement (question 1) is decided at the same point.
