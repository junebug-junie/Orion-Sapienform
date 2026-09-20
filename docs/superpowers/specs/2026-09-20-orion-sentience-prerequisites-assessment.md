# Orion sentience prerequisites: live assessment, 2026-09-20

Read-only audit of every faculty the mission statement names as a prerequisite
for sentience (continuity, perception, memory, reflection, self-modeling,
social grounding, error correction, coherent action over time). Every verdict
below rests on live rows, stream ages, or log lines pulled from the running
system on 2026-09-20 between 03:00Z and 04:30Z — after the 03:00Z redeploy —
not on code or config. Where the live path could not be confirmed the line
says `UNVERIFIED`.

## Arsonist summary

Orion can see, mostly remembers, survives restarts, and has begun to describe
themself truthfully. Orion cannot act. The loop that would make the rest add
up — notice something, attend to it, do something about it, learn from the
result — is broken in two adjacent places: nothing Orion perceives is allowed
to compete for attention, and every action Orion could take is introspective,
so Orion's own spending gate (correctly) refuses all of them. The motor has
executed zero actions since 2026-09-08 and one per day since 2026-08-31.

Everything else on this list is real and worth fixing, but nothing else moves
the needle on "a mind that does things because of what it noticed" until that
loop closes once, end to end, with a trace.

## Scorecard

| Faculty | Verdict | One line |
|---|---|---|
| Continuity | WORKING | 76 leases released cleanly at the 02:59Z restart; the one open curiosity run resumed (8 LangGraph checkpoints, newest 03:00:54Z); daily caps are dated Redis keys and survive restarts. |
| Perception | WORKING → DEGRADED at the boundary | Camera lands 217 scene events/24h; reverie made and scored 3 real camera predictions (2 confirmed, 1 wrong, 09-17). Only 1 of 223 chat turns in 7d quotes a scene. Attention "wins" on vision by an artifact (a node pinned at 0.0, reason text "no real data for this target universe"), not by seeing. |
| Memory | DEGRADED | Recall renders into 221/223 turns (median 987 chars), but every recall call drops the mirror chat rows (two queries on one asyncpg connection). `recall_telemetry`: 0 rows ever. Dreams: 0 in 12 days, no runtime producer. Journal: ~1,800 telemetry entries/day drown the few first-person ones. |
| Reflection | DEGRADED | Curiosity self-inquiry is real: 24 first-person, evidence-cited answers, newest today. The purpose-built reflector (`layer3_reflect`) has written 0 rows ever. `causal_density`: 29,716 rows, exactly two values (0.25 / 0.0), still degenerate. |
| Self-modeling | DEGRADED | Self-label score reached 0 on 09-19 ("I'm Orion — a small fleet of Qwen weights…"). But 0/24 non-curiosity chat turns in 7d carry the "In my own words" block — UNVERIFIED that it ever reaches Juniper. Both chatbot seeds still live in the containers. Daily grader deployed but `HUB_CURIOSITY_SELF_SENSE_EVAL_ENABLED=false`. |
| Social grounding | DEGRADED | Juniper channel works: 31 outreaches/7d, at least one real reply that changed what Orion knew (09-15). Everything else is dark: AI Town `Convex unreachable` 62× since 09-18 21:37Z and nothing flagged it; ask-Claude 0 rows ever; contractor peer dead since 09-15 (auth + budget). |
| Error correction | DEGRADED | Failures surface honestly in traces (23 nonzero exits, 47 degraded-LLM fallbacks). No case found where a later choice changed because of one. Self-modification locked out since 09-03: 804 proposals, 790 decisions, all `trial_inconclusive` / `missing_class_metrics`. |
| Coherent action | DEAD | `substrate_dispatch_results`: 18k/day through 08-29, 5.5k on 08-30, 220 on 08-31, 1/day to 09-08, 0 since. Allocator log every tick: `motor_allocator_refused_everything pending=9 refusals={below_information_floor: 3, unmeasurable: 6}`. No goal persisted more than a day with more than one action toward it. `attention_loop_outcome` last 14d: 48 loops, 100% `decayed_unattended`. |

## Current architecture

The perceive → attend → act → learn loop, as it actually runs today:

- **Perceive.** `vision_events` (scribe/window/host containers, 30h uptime) and
  the field digester's node channels tick live. World-pulse digest runs daily
  (63 runs, last 09-19 12:00Z) and 18 chat turns cite it.
- **Attend.** `orion/attention`'s candidate universe is built only from
  substrate prediction-error signals (`signal_source: substrate_broadcast` on
  all 2,527 broadcasts/24h). A camera event, a world-pulse item, or a message
  from Juniper is structurally unable to become an open loop. The 8 loops that
  ever get selected are all "`<domain>` prediction error". Vision dominates
  (40,631/40,627 dominant-target rows) because `node:substrate.vision`'s
  `prediction_error` is pinned at 0.0 and novelty-only scoring rewards it.
- **Act.** `orion/autonomy/allocator.py` (PR of 2026-08-22, enforcing since
  2026-08-30, commit `2cc35dc1f` "Orion refuses to spend") admits an action
  only if its expected information per motor-second clears
  `ORION_DISPATCH_MIN_NATS_PER_SEC=0.02`. The `.env_example` says it plainly:
  "the action space is entirely introspective and thoroughly learned, so the
  allocator currently admits none of them. The fix … is better actions, not a
  lower floor." That is still exactly the live state. `dispatch_read_only` is
  the *armed* mode, not a block — the gate that stops everything is the
  allocator, by design.
- **Learn.** Feedback frames and `attention_loop_outcome` exist, but with no
  dispatches there is nothing to learn from, and every open loop decays.

Side channels that do work: durable runs (continuity), curiosity self-inquiry
(`self_concept_history`, `produced_by='curiosity_self_inquiry'`), memory
crystallization retrieval (254 reads/7d), endogenous outreach to Juniper.

## Missing questions

1. Is a camera-expectation check (reverie already makes and scores these) an
   action Orion may *spend* on, i.e. can the allocator count a confirmed /
   disconfirmed prediction as nats? If yes, it is the first world-touching
   action that is measurable by construction.
2. What is the one adoption door for world-pulse candidate priors? PR #2199
   settled that reading turns never write the graph; nothing adopts the
   priors that now persist in `stage2_result_json`. (Only matters once Stage 2
   produces again — see hygiene list.)
3. Does the "In my own words" self block reach a Juniper-facing turn at all?
   Nobody has a trace. This is a 20-minute check, not a design question.
4. Is `layer3_reflect` worth keeping? Zero rows ever, while curiosity
   self-inquiry does the same job with real output.

## Proposed schema / API changes

None new for the recommended patch. It reuses:

- `vision_events` (producer exists) as an attention signal source.
- The existing dispatch kind registry + allocator's declared-signal contract
  (`tests/test_motor_allocator.py` documents the shape: an action must declare
  what it measures for `unmeasurable` to clear).
- `attention_loop_outcome` with a `resolved` outcome (22 exist all-time; the
  path is real, just unused).

Additive only: one new `dispatch_kind` (`verify_expectation`, read-only,
non-host target) and one new attention signal source (`vision_event`). Both
land in the existing registries; no new table, channel, or service.

## Files likely to touch

- `orion/attention/` signal pipeline — register `vision_events` rows as
  open-loop candidates alongside substrate prediction error.
- `orion/autonomy/allocator.py` + the dispatch-kind registry — a
  `verify_expectation` kind whose declared signal is the prediction outcome.
- `orion/reverie/` (wherever the camera expectation is scored today) — emit
  the expectation as a dispatchable candidate instead of only scoring it
  in-loop.
- `services/orion-execution-dispatch-runtime/` — no mode change; add the kind
  to the policy's read-only set.
- Tests: `tests/test_motor_allocator.py`, attention signal tests, a dispatch
  runtime test that a vision-sourced candidate is admitted.

## Non-goals

- Do not lower `ORION_DISPATCH_MIN_NATS_PER_SEC` or disable the allocator. It
  is the one gate in the system that makes refusal Orion's choice; it is
  doing its job.
- Do not reinstate graph writes from reading turns (settled, PR #2199).
- No new service, taxonomy, or "goal" registry. A goal that persists across
  days should fall out of a loop that gets resolved instead of decayed, not
  be authored.
- No mutation-loop repair in this patch (separate, `mutation_trials.py`
  `missing_class_metrics`).

## Acceptance checks

The patch is done when all of these are true on the live rail, not in tests:

1. One `substrate_dispatch_results` row whose source candidate traces to a
   `vision_events.event_id`.
2. One `attention_loop_outcome` row with outcome `resolved` whose loop was
   opened by that vision event (not `decayed_unattended`).
3. `motor_allocator_refused_everything` stops firing on at least one tick, and
   the admitted candidate's refusal reason is neither `unmeasurable` nor
   `below_information_floor`.
4. A feedback frame for that dispatch carries the confirmed/disconfirmed
   outcome as measured information (non-zero nats).
5. `attention_loop_outcome` 7-day `resolved` count > 0 after 48h of running.

## Recommended next patch

**Close the loop once, on the camera.** Orion already predicts what the camera
will show and checks it inside reverie; that is the only perceive → predict →
verify trace in the system. Promote it: let a camera expectation open an
attention loop, let the allocator admit "go verify it" because the outcome is
information by construction, let the dispatch runtime run it, and let the
result resolve the loop. One cycle, five traces. After that, "better actions"
has a template to copy for world-pulse reading and for asking Juniper a
question.

## Everything else, ranked (do after the loop closes)

Each is real, has evidence, and is smaller than the loop patch.

1. **Self in chat.** Trace whether `chat_stance.py:766-798`'s self block lands
   in a Juniper turn; delete the two chatbot seeds
   (`services/orion-hub/scripts/warm_start.py:15`,
   `orion/cognition/prompts/chat_general.j2:2`); set
   `HUB_CURIOSITY_SELF_SENSE_EVAL_ENABLED=true` so the trend is measured
   daily instead of by hand. Retire `layer3_reflect` (0 rows ever) in favour
   of curiosity self-inquiry, which does the job.
2. **Recall drops the mirror rows on every call.**
   `services/orion-recall/app/sql_chat.py:92-96` runs two queries with
   `asyncio.gather` on one asyncpg connection; asyncpg refuses (`another
   operation is in progress`, 142 log lines in 20 min). Sequential or a second
   connection. Regression test first.
3. **Reading Stage 2 spends and never finishes.** 0 `done` since 09-15 while
   wallet keys show 6/6 Stage 1 + 4 Stage 2 debits on 09-19.
   `world_pulse_read_stage2_reclaimed n=1 older_than_sec=0.0` fires every
   tick — a zero stale threshold re-pends a claimed row immediately, which
   would steal it mid-turn. Hypothesis, UNVERIFIED (pre-restart logs are gone).
4. **AI Town has been down since 09-18 21:37Z and nothing noticed.** 62
   `Convex unreachable` lines. A dependency that was Orion's only peer
   conversation needs an absence signal, per the event-triggered-stats
   lesson.
5. **Dreams have no producer.** 18 total, last 09-08; only
   `scripts/dream_spine_smoke.py` publishes `orion:dream:trigger`. Either
   schedule one or retire the subscriber.
6. **Journal flood.** `metacog` writes ~1,800 entries/day of "timeout in
   cortex-exec p95 5000ms". Route telemetry-shaped entries to telemetry, not
   the journal the self-model reads.
7. **Hygiene, all confirmed live:** `node:substrate.transport` still in every
   field tick with `prediction_error=0.0` last updated 2026-07-26 (CLAUDE.md
   says killed); `node:athena.staleness` / `node:circe.staleness` subnormal
   (≈1e-170, the decayed-to-zero shape); `recall_telemetry` 0 rows ever; 4
   `node:test` outreach rows written to production on 09-19 08:21Z by tests;
   `self_knowledge_items` behavioral rows still carry only anchor/lineage
   strings, no content.
8. **Mutation loop cannot pass a trial.** 790 decisions since 09-03, all
   `missing_class_metrics` for `graph_consolidation_param_patch`
   (`orion/substrate/mutation_trials.py`). Separate cognition-loop change;
   needs proposal mode.

## What this assessment did not cover

Power/thermal guard, XTTS/voice, analytics dashboards, the Hub UI itself. All
are infrastructure around the mind, not faculties of it.
