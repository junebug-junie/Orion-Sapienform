## Summary

Stacks on #3532 (fix/promise-finder-presence), now merged; this branch is rebased on main. #3532 must be **deployed** (governor) before any promise act opens: until then every stored item lacks `force`/`done_when` and is skipped by design.

When Orion tells Juniper "I'll do X later", that promise is now written down on the act ledger, and it closes later as kept, expired or void. This runs in **shadow**: nothing learns from the result and nothing steers Orion yet. Issue #3531 decides when that changes. Juniper approved this step on 2026-11-16 ("okay hit it").

- Each firm promise in a reply to one of Juniper's turns opens one act: actor `hub.promise`, signal `promise_kept`, 72 h. The act points at the promising turn's record and holds no copy of the quote.
- An offer ("if you want, I'll...") opens nothing until Juniper's next message accepts it. The judge sees the previous reply's offers and says which one she accepted.
- Before each of Juniper's turns, the 5b judge sees the open promises (up to 8). It can mark one kept only by quoting the words of the reply that deliver it. Plain code checks the quote is really in the reply.
- At 72 h, sql-writer closes an unkept promise as `expired` (Juniper wrote, no reply delivered it), or `void` (Juniper wrote nothing in the window, so no reply could have kept it).
- In shadow, sql-writer writes no outcome row and no posterior for these acts. The self-monitor builds no bet from them, and the act-chain forecaster leaves them out. The stance cue (D3) is built but ships **off**.
- A read-only report shows open, kept, expired, void and un-closeable promises, the self-reinforcement guard, and writes a blinded labelling sheet for #3531.

## Outcome moved

Before: a promise was found and stored on the turn record (D0), and Prime opened a concern for it, but nothing tracked whether Orion kept it. After: every promise to Juniper has a lifecycle on the act ledger with a graded-by-quote "kept" close, so #3531 can measure live precision and the keep rate before any of it reaches a track record or stance.

## Current architecture

- The 5b judge returns `promises_made` (gated by `orion/harness/promise_gate.py`, rules from #3532), `offers_accepted` and `promises_kept`, stored on `harness_turn_trace.run_artifact.reflection`. Live turns never passed `pending_offers` or `open_promises`, so `offers_accepted` and `promises_kept` were always empty.
- Prime's hand-back opens a concern per promise turn (`orion/prime/handback.py`) with a `promise_text` copy, 72 h TTL; Prime may take it on as an `intention.carry` run.
- The act ledger (`orion/grammar/act_ledger.py`) knew three signals: `juniper_replied`, `artifact_produced`, `juniper_answered`.

## Architecture touched

- Contract: `ActSignal += "promise_kept"` (consumer-first); `PromiseKeptV1`, `FinalizeReflectionV1.promises_kept_evidence`, `HarnessRunRequestV1.open_promises` / `pending_offers`, `PROMISE_JUDGE_VIEW_MAX = 8`.
- orion-hub: pre-turn read of open promises and pending offers (Juniper's turns only), promise act publish after delivery.
- orion-harness-governor: passes the two request fields to the finalize chain (judge only).
- orion-sql-writer (via `orion/grammar/act_ledger.py`): shadow signals not graded, no cost for promise acts, resolver void/expired branch.
- orion-durable-runs / any self-monitor host (`orion/self_monitor`): shadow and void closes are no bet.
- orion-prediction: act-chain read leaves shadow acts out.
- orion-thought: D3 cue behind an OFF flag.

## Files changed

- `orion/grammar/promise_acts.py` (new): the promise act (identity, ref, builder), `plan_promise_events` (opens, accepted offers, kept closes, counts), close kinds, `SHADOW_ACT_SIGNALS`.
- `orion/grammar/promise_view.py` (new): read-only SQL for open promises (quotes from `harness_turn_trace`) and the previous reply's offers.
- `orion/hub/promise_ledger.py` (new): bounded, fail-open view for the turn; publishes the turn's promise acts.
- `orion/hub/turn_orchestrator.py`: view before the request (Juniper's turns only); `_publish_promise_acts` runs as a background task from that same view.
- `orion/grammar/act_ledger.py`: not graded for shadow signals and void notes; no cost fill for promise acts; resolver void/expired branch and counts.
- `orion/harness/promise_gate.py`, `orion/harness/finalize.py`, `orion/cognition/prompts/harness_finalize_reflect.j2`: kept marks are `{id, quote}`, quote substring-gated on the draft and again on the final text; promise view threaded through the chain and the tool-retry re-reflection.
- `orion/schemas/grammar.py`, `orion/schemas/harness_finalize.py`: contract additions above.
- `orion/self_monitor/collect.py`, `orion/self_monitor/adapters.py`: shadow signals filtered in SQL; shadow and void closes are no bet.
- `services/orion-prediction/app/store.py`: act-chain read excludes shadow signals.
- `services/orion-harness-governor/app/bus_listener.py`: passes `open_promises` / `pending_offers` to the chain.
- `services/orion-thought/app/open_promises_cue.py` (new), `bus_listener.py`, `settings.py`, `orion/cognition/prompts/stance_react.j2`: D3 cue, off.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, `services/orion-thought/.env_example`, `scripts/sync_local_env_from_example.py`: flags.
- `scripts/report_promise_ledger.py` (new): read-only report and blinded labelling sheet.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: "Amendments (2026-11-16)", one answer per critique point.
- Tests: `orion/grammar/tests/test_promise_acts.py`, `orion/harness/tests/test_promise_gate.py`, `orion/self_monitor/tests/test_self_monitor_promise_shadow.py`, `services/orion-hub/tests/test_promise_ledger_hub.py`, `services/orion-thought/tests/test_open_promises_cue.py`, `services/orion-sql-writer/tests/test_promise_acts_integration.py` (+ conftest skip list), `tests/test_report_promise_ledger.py`.
- `services/orion-sql-writer/evals/promise_acts_mutations.py` (new): 10 mutants of the shadow, void, kept-quote and offer guards.
- CI: `.github/workflows/promise-ledger-tests.yml` (new; the promise gate tests had no CI job), `orion-sql-writer-tests.yml` runs the new Postgres test and the mutation check.

## Schema / bus / API changes

- Added: `promise_kept` to `ActSignal`; `PromiseKeptV1`; `FinalizeReflectionV1.promises_kept_evidence`; `HarnessRunRequestV1.open_promises`, `.pending_offers`; judge JSON schema `promises_kept` items are now `{id, quote}`.
- Removed / renamed: none.
- Behavior changed: a kept id needs a delivering quote in the reply (a bare id is dropped); acts on a shadow signal and void closes are never graded.
- Compatibility: `promises_kept` stays a list of ids (Prime's hand-back reads it). The request fields are ignored by an older governor. `ActSignal` is consumer-first: an older sql-writer rejects a `promise_kept` event, so sql-writer must run the new image before Hub publishes one (deploy order below). No new channel; `orion:grammar:event` already lists orion-hub as a producer. No registry change (no new bus kind).

## Env/config changes

- Added keys: `HUB_PROMISE_LEDGER_ENABLED=true` (orion-hub; shadow is the shipped mode), `THOUGHT_OPEN_PROMISES_CUE_ENABLED=false` (orion-thought; deliberately off until #3531).
- Removed / renamed: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (added `HUB_PROMISE_LEDGER_` to the sync prefixes, since no generic `HUB_` prefix syncs).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/harness orion/prime orion/self_monitor/tests orion/grammar/tests tests/test_report_promise_ledger.py
  1411 passed
pytest services/orion-hub/tests/test_promise_ledger_hub.py test_act_ledger_producers.py test_route_engine_reuse.py test_chat_history_no_raw_publish.py
  49 passed
pytest services/orion-hub/tests (full, before review fixes): 4979 passed, 2 failed, 4 errors -- all host-env:
  fcc_model_labels (PermissionError ~/.fcc/.env), consult_belief_store_sql (CREATE INDEX CONCURRENTLY in a
  transaction, local DB), and test_route_engine_reuse (MINE: per-call engine; fixed, now passes)
services/orion-thought: test_open_promises_cue.py + test_self_track_record_cue.py   18 passed
services/orion-harness-governor/tests   106 passed, 1 failed (PermissionError ~/.fcc/.env, host-env)
services/orion-prediction -k act_chain   17 passed (other failures: statsmodels/prophet not installed locally)
sql-writer Postgres integration (throwaway postgres:16 container), incl. test_promise_acts_integration.py (4),
  test_act_ledger / camera_ask / workspace_binding   all pass
sql-writer unit lane (orion/grammar/tests/test_ledger.py + services/orion-sql-writer/tests)   1331 passed, rest skipped
scripts/check_integration_seam.py   OK
orion-static-gates.yml, every run step locally (after rebase)   42/42 OK
```

## Evals run

```text
services/orion-sql-writer/evals/promise_acts_mutations.py (throwaway Postgres): 10/10 mutants CAUGHT
  (shadow graded, shadow set emptied, promise cost, void never void, void/expired swapped, self-monitor bets
  on shadow acts, kept without quote, re-promise as kept, offer opens early, old row opens).
No new model eval: this patch does not change what the finder lists (the #3532 replay numbers stand).
The kept-quote prompt change is unmeasured on replay; #3531's live labelling covers it.
scripts/report_promise_ledger.py --days 14 against live (read-only): 0 promise acts (not deployed),
79 judged Juniper turns, 0 firm promises, 12 stored items skipped as pre-#3532 (no force/done_when).
```

## Docker/build/smoke checks

```text
No docker build run. Live read-only SQL smoke against conjourney: fetch_promise_view runs without error;
EXPLAIN ANALYZE of the previous-turn read 3.1 ms and the open-promise read 0.9 ms (partial index
idx_grammar_acts_open_horizon). scripts/report_promise_ledger.py --days 14 runs live.
```

## Review findings fixed

Code review ran in a subagent (a first run died on a usage limit and was rerun). No blockers.

- Finding: promise acts were published inline before the final frame, with a second DB read of the offers.
  - Fix: one pre-turn read feeds both the judge and the ledger; the publish runs as a background task.
  - Evidence: `test_delivered_juniper_turn_opens_and_closes_promise_acts` passes the view, no re-read.
- Finding: the second offers read could see a different "previous row" than the judge did.
  - Fix: same as above (one read).
- Finding: acts opened on `source_label == "hub_orion"`, the view on `utterance_origin == "juniper"`; an Orion-origin turn could open acts.
  - Fix: acts open only when the turn got the pre-turn view (Juniper's turns).
  - Evidence: `test_no_view_means_no_acts_even_on_a_hub_orion_turn`.
- Finding: the read had no DB-side limit; a timed-out worker thread keeps running.
  - Fix: statement_timeout 1.5 s, connect_timeout 2 s, pool_timeout 1 s on a cached 2+2 pool. Live timings 1-3 ms, so no new index.
  - Evidence: `test_view_read_fails_open`; live EXPLAIN ANALYZE above.
- Finding: the motor-blindness test could not fail (the prompt builder was never given the promises).
  - Fix: the test now drives `HarnessRunner.run` with open promises and offers on the request and checks every motor kwarg.
- Finding: no test ran the view SQL on Postgres; no fail-open test.
  - Fix: `test_the_hub_view_sql_reads_open_promises_and_the_previous_replys_offers`, `test_view_read_fails_open`.
- Finding: a kept quote could just repeat the promise; the cap tested raw length.
  - Fix: refused as `dropped_kept_repeats_promise`; cap on normalized length. Mutant caught.
- Finding: report judged-turn join could duplicate; report turn clock differs from the resolver's.
  - Fix: DISTINCT ON correlation id; clock difference stated in the report docstring.
- Finding: the new ActSignal value must reach sql-writer before hub publishes it.
  - Fix: deploy order stated below. An event published early is dropped (shadow only; logged).
- Not changed: AI-town rows counting as "Juniper wrote" in the void rule. Checked live: 0 non-hub_orion Juniper-predicate rows in 14 days. Same predicate as the reply-act resolver.
- Not changed: no test of the `execute_unified_turn` wiring or the tool-retry re-reflection carrying the promise view (see Risks).

## Restart required

```bash
# After #3532 is merged and deployed: the post-pull rebuild picks these up. Order matters once:
# sql-writer (accepts promise_kept) and the grammar-event consumers before orion-hub publishes one.
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: live precision of the finder is still unproven (held-out 10/12, Wilson 0.55-0.95). Mitigation: shadow; #3531 labels the first ~60 stored items blind with `report_promise_ledger.py --sample`.
- Severity: medium. Concern: needs-juniper defaults -- void rule, recall floor, Prime concern following the act at go-live, carry runs not counting as keeps (spec amendments 3, 6, 7, 10). Mitigation: listed in the PR description with recommendations.
- Severity: low. Concern: if orion-hub restarts on the new image before orion-sql-writer, the first promise events are rejected by the old writer and lost. Mitigation: deploy order below; shadow only.
- Severity: low. Concern: the kept-quote prompt change is unmeasured on replay. Mitigation: #3531 live labels; the gate fails closed.
- Severity: low. Concern: no direct test that the tool-retry re-reflection carries open promises, nor of the `execute_unified_turn` request wiring. Mitigation: both are plain argument threading, covered by review; follow-up issue opened.
- Severity: low. Concern: one more read per Juniper turn before the motor starts (bounded at about 2 s worst case, 1-3 ms live).
- Issues opened: #3560 (wiring + tool-retry tests). Issues closed: none. #3531 stays open (it decides go-live).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
