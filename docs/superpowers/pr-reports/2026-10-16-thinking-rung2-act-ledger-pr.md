# feat: act ledger, "doing on one track" (thinking architecture rung 2)

## Summary

- Orion's world-facing doing now writes one shared record per act: what it is about to do, what it expects, and later what happened. This is rung 2 of `docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md` (PR #2669, still open, not merged here).
- The record is an extension of the existing grammar, not a new contract: two new event kinds (`act_opened`, `act_closed`) and an `ActV1` payload on `GrammarEventV1.act` (`orion/schemas/grammar.py`).
- orion-sql-writer stores each act in a new `grammar_acts` table, closes it by `act_id`, and grades every close through the existing L10 learner (`substrate_action_outcomes` + `substrate_action_effect_posterior`, `EffectPosterior`). No new learner.
- Four first producers: paintings (`make`, orion-thought), chat replies to Juniper (`say`, hub), outreach (`say`/`ask`, hub), web reading Stage 1 (`look`, hub).
- A horizon resolver in the writer closes "did Juniper answer" acts from `chat_history_log`, and expires producer acts that never reported back.
- Debug surface `GET /act/ledger` on orion-sql-writer; calibration eval on 60 days of real outcome history.

## Outcome moved

Before: world-facing doing (painting, replying, outreach, reading) ran outside the predict-act-grade loop. `substrate_action_outcomes` got 25 rows in the last 7 days, all housekeeping; `substrate_action_ratings` has 0 rows ever.

After deploy (expected, UNVERIFIED until deployed): every painting, chat reply, outreach message and Stage 1 read opens an act with an expectation, and closes into `substrate_action_outcomes` with an error and a moved posterior. Rough volume from live 30-60 day rates: about 8 paintings/day, about 15 reads/day, about 4 outreach/day, plus chat replies.

## Rung 1 gate (read-only, live, 48 h to 2026-10-16 02:16 UTC)

Rung 1 ("attention faces the world", #2528 / #2583) is live enough to build on.

| Frame winner | Frames | Share |
|---|---|---|
| no winner | 44,210 | 54.2% |
| body (`source_kind:internal`) | 25,154 | 30.9% |
| world (`source_kind:external`) | 12,141 | 14.9% |

World winners: camera (`node:substrate.perception`) 7,229, chat (`world:chat`) 4,912. World candidates enter the contest constantly (perception 81.5k, chat 76.2k target rows). The design doc's baseline was 72% no-winner and no world input. Remaining rung-1 gap, not blocking rung 2: reading is not an attention candidate.

## Current architecture

- The grammar (`GrammarEventV1`) was a trace language: no verb, no expectation, no outcome.
- `substrate_action_outcomes` + `EffectPosterior` (`orion/autonomy/prediction.py`) graded only proposal-template dispatches from orion-feedback-runtime.
- Paintings (durable `reverie.visual`), chat replies, outreach and reading each kept their own tables and gates. None predicted anything checkable.

## Architecture touched

- Contract: `orion/schemas/grammar.py` (`ActV1`, `ActExpectV1`, `ActOutcomeV1`, `ActCostV1`, closed `ActVerb`, `ActLevel`, `ActSignal`), `orion/schemas/registry.py` (`ActV1`), `orion/bus/channels.yaml` (orion-thought added as a grammar producer).
- Writer: `orion/grammar/ledger.py` (hands freshly inserted act events to the act ledger), `orion/grammar/act_ledger.py` (persist, close, grade, resolve, summary), `services/orion-sql-writer/app/act_resolver_loop.py`, `GrammarActSQL` model, `/act/ledger` route.
- Learner: `orion/autonomy/prediction.py` gains binary-scale constants and `score_act_outcome` on the same `EffectPosterior`.
- Producers: `orion/grammar/acts.py` (builders), `services/orion-thought/app/visual_steps.py`, `orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/act_emit.py`, `endogenous_outreach.py`, `world_pulse_read_pipeline.py`.

## Files changed

- `orion/schemas/grammar.py`: act event kinds and `ActV1` payload, validators (expect required above `metabolic`, close needs outcome, binary value).
- `orion/schemas/registry.py`: register `ActV1`.
- `orion/bus/channels.yaml`: orion-thought listed as an `orion:grammar:event` producer.
- `config/metrics/metric_definitions.lock.json`: re-lock for that producer change.
- `orion/grammar/acts.py`: producer builders, deterministic `act_id`, chat-reply horizon and its derivation, literal provenance per producer.
- `orion/grammar/act_ledger.py`: writer side (open, close-by-id, L10 grading, resolver, summary).
- `orion/grammar/ledger.py`: route freshly inserted act events to the act ledger; kill switch.
- `orion/autonomy/prediction.py`: `ACT_BINARY_*` constants, `act_dispatch_kind`, `cold_act_posterior`, `score_act_outcome`.
- `orion/hub/turn_orchestrator.py`: `say` act after a Juniper chat reply is persisted.
- `services/orion-hub/scripts/act_emit.py`: outreach `say`/`ask` and reading `look` producers, horizons and derivations.
- `services/orion-hub/scripts/endogenous_outreach.py`, `world_pulse_read_pipeline.py`: call the producers.
- `services/orion-thought/app/visual_steps.py`: `make` act per durable painting run.
- `services/orion-sql-writer/app/{main.py,settings.py,act_resolver_loop.py,models/*}`, `docker-compose.yml`, `.env_example`, `README.md`: writer wiring, flags, docs.
- `services/orion-hub/{app/settings.py,.env_example}`, `services/orion-thought/{app/settings.py,.env_example}`: producer flags.
- `scripts/sync_local_env_from_example.py`: sync prefixes for the new hub and writer keys.
- Tests/evals: `tests/test_act_ledger_contract.py`, `services/orion-sql-writer/tests/test_act_ledger_{wiring,integration}.py`, `services/orion-sql-writer/evals/{act_ledger_calibration.py,test_act_ledger_calibration_eval.py,fixtures/act_outcome_sequences.json}`, `services/orion-thought/tests/test_painting_act.py`, `services/orion-hub/tests/test_act_ledger_producers.py`.
- CI: `.github/workflows/{orion-sql-writer-tests,visual-baseline,orion-reading-tests}.yml`.

## Schema / bus / API changes

- Added: `GrammarEventKind` `act_opened`, `act_closed`; `GrammarEventV1.act: ActV1 | None`; `ActV1` registry entry; table `grammar_acts` (SQLAlchemy `create_all` at writer start); `GET /act/ledger?days=7` on orion-sql-writer.
- Added rows (no schema change): `substrate_action_outcomes` / `substrate_action_effect_posterior` rows with `dispatch_kind = act:<verb>`, `target_id = <actor>`, `signal_id` in {`juniper_replied`, `artifact_produced`}. The `act:` prefix keeps them disjoint from proposal-template keys.
- Removed / renamed: none.
- Compatibility: `GrammarEventV1` is `extra="forbid"`, so orion-sql-writer (the only consumer that validates the model) must be rebuilt BEFORE the producers (orion-hub, orion-thought), or it rejects act events. orion-equilibrium-service and orion-heartbeat read the payload as a plain dict and only look at atoms (`app/service.py` in each), so they need no rebuild.

## Env/config changes

- Added keys: `SQL_WRITER_ACT_LEDGER_ENABLED=true`, `SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC=60` (orion-sql-writer); `HUB_ACT_LEDGER_ENABLED=true` (orion-hub); `THOUGHT_ACT_LEDGER_ENABLED=true` (orion-thought). All ship ON.
- Removed / renamed keys: none.
- `.env_example` updated: yes, all three services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; all four keys present in the primary checkout's `.env` files. The hub and writer keys first fell outside `SYNC_PREFIXES`; fixed by adding `HUB_ACT_LEDGER_` and `SQL_WRITER_ACT_` prefixes in this PR.
- Skipped keys requiring operator action: none.

## Metric quality gate: `expect` / `error`

1. **Provenance.** `predicted` = the `substrate_action_effect_posterior` mean for (`act:<verb>`, actor, signal), stamped at open by `orion/grammar/act_ledger.py:open_act`. `outcome` = producer close (paintings: `visual_steps.painting_act_transition`; reading: Stage 1 lifecycle in `world_pulse_read_pipeline.publish_lifecycle`) or resolver (`resolve_due_acts`, `chat_history_log` with the canonical `orion.regulation.juniper_turns` predicate). `error = outcome - predicted`, computed by the writer at close.
2. **Independence.** `artifact_produced` for paintings derives from the same event as the durable run's terminal state, and for reading from the Stage 1 lifecycle. It is a new view of an existing fact, not new signal. The new information is the comparison with an expectation, plus the posterior. `juniper_replied` is new for acts. It shares its source table with the attention contest's `world:chat` candidate (same `chat_history_log`), so it is not independent of chat salience.
3. **Theory anchor.** Efference copy (von Holst) and TOTE units, per the design doc §3.2. Grading is the existing Normal-Normal conjugate update with Bayesian surprise (Itti & Baldi 2009). The binary constants come from the scale itself, not from tuning: prior N(0.5, 0.25) is the max-entropy rate; observation variance 0.25 is max p(1-p).
4. **Live-data sanity.** Outcomes were rebuilt read-only from live history (eval `act_ledger_calibration.py --live`):
   - Chat reply at 1800 s: rate 0.49 (n=249).
   - Outreach at 36000 s: rate 0.485 (n=237).
   - Paintings: 254 of 289 produced (0.88, 30 d). Reading Stage 1 runs: 375 of 448 completed (0.84).
   - None of these is pinned at 0 or 1, and each can read 0 ("not answered", "not produced").
   - Honest limit: for say/ask the posterior learns a RATE, not a per-act forecast. Its Brier score is 0.255, against 0.250 for a constant 0.5 guess. Per-act `error` is about ±0.5 noise. It is meaningful only in aggregate, as a rate that moves over time.
   - Outreach at a 30-minute horizon would be 0.04, almost always "no". At the median horizon (10 h) it is dominated by Juniper's daily rhythm, not by the message itself. Separating those two needs the holdback arm in rung 3.
5. **Existing mechanism.** Reuses `EffectPosterior`, `substrate_action_outcomes`, `substrate_action_effect_posterior`, the grammar ledger, and `JUNIPER_TURN_PREDICATE`. Temporal Self already reads `substrate_action_outcomes` (`services/orion-durable-runs/app/temporal_self_sources.py` `action_outcome`), so closed acts reach Temporal Self with no change to `orion/temporal_self/sources.py`.
6. **Reversibility.** Cheap. The kill switches stop production and materialisation. Act rows are keyed `act:*` and can be deleted by prefix. `grammar_acts` is standalone. No schema change to existing tables.

Horizon derivations (all measured 2026-10-16, from code comments):
- Chat reply 1800 s: median gap between Juniper's turns, 2007-2015 s, 60 d.
- Outreach 36000 s: median time to Juniper's next turn after an outreach message, 36792 s.
- Paintings 5400 s: the run's own `deadline_at - requested_at`.
- Reading 3600 s: p90 of completed reading.turn runs, 2734 s, rounded up.

## Tests run

```text
tests/test_act_ledger_contract.py                                   13 passed
services/orion-sql-writer/tests/test_act_ledger_wiring.py            5 passed
services/orion-sql-writer/tests/test_act_ledger_integration.py       9 passed (scratch postgres:16, throwaway schema)
services/orion-thought: test_painting_act + test_visual_steps + test_bus_listener_visual_step   74 passed
orion/autonomy/tests/test_agency_episode_reader.py                   9 passed
services/orion-hub reading/outreach/turn subset + test_act_ledger_producers                     587 passed, 37 skipped
tests/test_grammar_event_producer_catalog.py, env parity/sync tests, migration drift, schema skew  118 passed
orion-sql-writer full suite: 14 failed / 806 passed (none act/grammar) (main checkout baseline: 15 failed, same files; pre-existing)
root: 4 pre-existing failures identical on main (expected_effect live policy, exec_result wildcard, field glossary, single-consumer gate)
hub: test_turn_orchestrator_utterance_origin::...mind_appraisal... fails identically on main
gates: check_env_template_parity PASS, check_metric_lineage --gate PASS, check_definition_drift --gate PASS (re-locked),
       compose parity N/A (env_file), async routes / chat-route poachers / system_health producers / stdlib shadow PASS
```

## Evals run

```text
services/orion-sql-writer/evals/test_act_ledger_calibration_eval.py   2 passed
python evals/act_ledger_calibration.py --live (read-only):
  say:hub.chat_reply (1800s)   n=249 empirical=0.49   final_mean=0.49   brier=0.2555 base=0.25
  say/ask:hub.outreach (36000s) n=237 empirical=0.4852 final_mean=0.4853 brier=0.2549 base=0.25
```

## Docker/build/smoke checks

```text
Not run: no deploy requested. Live path UNVERIFIED until the restart below.
```

## Review findings fixed

A code-review subagent reviewed `origin/main...HEAD`. It found no blockers and 8 "should" findings. All 8 are fixed.

- Finding: A painting replay that reports terminal/"produced" was graded 0. An "already_satisfied" refusal also closed at 0.
  - Fix: any outcome "produced" now closes at 1. A prepare that refuses before any work is not an act at all. "already_satisfied" never closes (`visual_steps.painting_act_transition`).
  - Evidence: `test_painting_act.py` has new transition rows for both replay cases (16 passed).
- Finding: A read retry reused the act after attempt 1 had already closed it at 0, so the retry's success was lost. Refusals (bad URL, blocked publisher, index page) were graded as failed reads.
  - Fix: read acts are now explicit in `_read_claimed`. The act opens only after the refusal checks. A stage-1 failure closes at 0 only when no retry is scheduled (`_fail_seed` now returns its outcome). A retried seed keeps one open act, decided by the retry or by the horizon.
  - Evidence: `test_read_failure_closes_only_when_no_retry_is_scheduled`, `test_read_act_opens_only_after_refusal_checks`.
- Finding: Solicited `_deliver` calls (collapse-mirror replies, the debug route) were graded as outreach.
  - Fix: only `unsolicited=True` deliveries publish an outreach act.
  - Evidence: `test_solicited_delivery_is_not_an_outreach_act`.
- Finding: The "Juniper replied" check uses the time the turn row was stored, which is after Orion's answer. This biases the 0s.
  - Fix: documented in the horizon basis string, in `orion/grammar/acts.py`, and under Risks below. Not re-sourced: no turn-start timestamp exists today that has both a correlation id and the unsolicited filter. Follow-up.
  - Evidence: the basis text is carried on every say act.
- Finding: The resolver ran up to 500 closes in one transaction. That risked deadlocks with grammar lanes. A close lost that way, or shed and replayed after the horizon, would be expired at 0.
  - Fix: each resolver close now runs in its own savepoint and is committed separately. Before expiring an act, the resolver checks `grammar_events` for a persisted `act_closed` on `act:<id>` and applies that close instead.
  - Evidence: `test_resolver_recovers_a_persisted_close_instead_of_expiring`.
- Finding: The agency-episode reader turned act rows into false motor episodes with integrity issues.
  - Fix: both reader queries now filter `dispatch_kind NOT LIKE 'act:%'`.
  - Evidence: `test_act_ledger_rows_are_not_read_as_motor_episodes`.
- Finding: The act SQL never ran against Postgres in CI.
  - Fix: a new CI step ("Run act ledger Postgres integration tests (must not skip)") sets `ORION_SQL_WRITER_TEST_DATABASE_URI` and fails on any SKIPPED. Locally the tests ran against scratch postgres:16.
  - Evidence: 9 integration + 5 wiring tests, 14 passed.
- Finding: A success reported after the horizon was graded 1. `closed_in_horizon` counted resolver and expiry zeros as coverage.
  - Fix: a success reported after the horizon is graded 0 with note `late:<s>`. `closed_in_horizon` now counts only producer reports and observed replies. Resolver zeros are reported separately as `resolved_absent`.
  - Evidence: `test_success_reported_after_the_horizon_grades_zero`, `test_resolver_zeros_are_not_counted_as_in_horizon_coverage`.
- Nits fixed:
  - The posterior row is seeded before `FOR UPDATE`, which closes the race between two concurrent first closes.
  - `outcome_row_id` is now BigInteger.
  - The difference between `error` (stamped at open) and nats (belief at close) is documented.
- Nits left as they are:
  - The `ask`/`say` split on "?". The design's verb set asks for both.
  - The painting horizon starts at prepare, not at requested_at. It is harmless and noted.
  - The chat-reply guard rests on `no_write`. The reviewer confirmed that all four non-Juniper callers set it.

## Restart required

The writer first (it validates GrammarEventV1, which is extra="forbid"), then the producers. Run these from the clean primary main checkout after the merge:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
curl -fsS "http://localhost:8220/act/ledger?days=1"
```

## Risks / concerns

- Severity: medium. Concern: deploy order. If a producer is rebuilt before orion-sql-writer, the old writer rejects act events: they go to bus_fallback_log, and the acts are lost. Mitigation: use the restart order above.
- Severity: medium. Concern: say/ask posteriors learn a rate only (Brier no better than 0.5), and outreach replies are confounded with Juniper's daily rhythm. Mitigation: documented; per-act discrimination needs rung 3's holdback arm and rung 5's situational goals.
- Severity: low. Concern: a read's retries and GPU waits count against its single 3600 s horizon, so a late but real read closes 0 with note `late:`. Mitigation: these are the intended "by the horizon" semantics, and they are visible in `/act/ledger`.
- Severity: medium. Concern: `chat_history_log.created_at` is when the merged turn row is stored, which happens after Orion's answer. So `juniper_replied` really means "Juniper sent a turn that Orion answered and stored by the horizon". A reply sent near the horizon with a slow turn reads 0, and a turn that errored reads 0. Mitigation: stated in the basis on every act. Follow-up: resolve from a turn-start timestamp.
- Severity: low. Concern: 7 of the 11 verbs have no producer yet. They are reserved by the design doc's closed vocabulary. Mitigation: a new producer must add a literal provenance branch and tests.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
