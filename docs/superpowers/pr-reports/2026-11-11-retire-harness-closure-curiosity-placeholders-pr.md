# Retire step 3: curiosity skips non-measurement nodes; harness_closure retired

## Summary

- Orion's self-started curiosity no longer chases a prediction-error node unless the glossary says that node's number is a real measurement. Placeholder, bucket, and unknown (no glossary entry) nodes are skipped. This is the same rule Orion's self readings already use (`orion/attention/self_readings.py`), now shared through one helper, `node_prediction_error_is_measurement`.
- `node:substrate.harness_closure` is retired. It wrote a fixed 0.65 every time a harness turn ended "unresolved". The number was never a measured size. The write is removed from `handle_post_turn_closure` (`services/orion-substrate-runtime/app/worker.py`). The turn referent row that reverie reads is still written; it is a different artifact.
- Kill means kill: the glossary entry and metric lock row are removed. The unused `harness_closure_signal` argument of the attention self-model reducer is removed. The Hub attention-organ page no longer lists it as a known domain. The live-sanity eval no longer expects it.
- Two stale tests that already failed on main were fixed in passing (glossary entry count; `street` missing from the event-written table).

## Outcome moved

Before (7 days to 2026-11-11 ~03:30 UTC), harness_closure was the top curiosity seed:

| lead seed (slot 0) | candidate sets led | invoked runs |
|---|---|---|
| node:substrate.harness_closure | 495 | **130** |
| node:substrate.codebase | 112 | 71 |
| node:substrate.chat | 99 | 63 |
| node:substrate.execution | 88 | 32 |
| node:substrate.strain | 44 | 32 |
| node:substrate.perception_walkway | 38 | 13 |
| node:substrate.perception | 45 | 12 |

- 1,575 invoked curiosity runs in total; 130 (8.3%) were led by the fixed 0.65.
- harness_closure took 818 of 1,523 prediction-error candidate slots (54%).
- Its history: 426 rows in 7 days, every one exactly 0.650.

Which nodes are non-measurement kinds today (glossary + live FalkorDB graph):

- placeholder: `node:substrate.harness_closure` only (now retired; no entry).
- bucket: none.
- unknown (carries `prediction_error` in the graph, no glossary entry): `node:substrate.vision` (0.0, last written 2026-10-02; never crossed 0.55, so it never seeded). After this patch, also the lingering harness_closure node until it is deleted.

Expected after deploy: 0 candidate slots and 0 invoked runs led by harness_closure, and by any placeholder/bucket/unknown node. **UNVERIFIED until deploy.**

Re-measure after deploy (replace the timestamp with the deploy time):

```sql
-- PE-sourced curiosity seeds by focal node, and invoked runs they led, since deploy
select c->'focal_node_refs'->>0 as node,
       count(*) as candidate_slots,
       count(*) filter (where ord = 1 and s.gate_json->>'evaluator_outcome' = 'invoke') as invokes_led
from substrate_endogenous_curiosity_candidates s,
     jsonb_array_elements(s.candidates_json) with ordinality as t(c, ord)
where s.created_at > timestamptz '2026-11-12 00:00:00+00'
  and c->'notes' ? 'source:prediction_error'
group by 1 order by 2 desc;
```

Pass: no row for `node:substrate.harness_closure` or `node:substrate.vision`.
Also: `select count(*) from substrate_node_prediction_error_history where node_id='node:substrate.harness_closure' and recorded_at > <deploy time>;` must be 0.

## Proposal-mode facts (approved 2026-11-11, "yeah on 2")

- **Capability change:** curiosity can no longer be started by a node whose number is not a measurement. Orion loses one fake reason to be curious (the fixed 0.65); real seed sources (execution, chat, codebase, strain, perception, walkway, cabinet triggers, repair pressure, open loops, coverage gaps) are unchanged.
- **Data touched:** code stops writing `node:substrate.harness_closure` (FalkorDB `orion_substrate`) and its rows in `substrate_node_prediction_error_history`. Nothing is deleted by code. The operator command below deletes the one stored node.
- **Privacy boundary:** none crossed; internal self-signals only.
- **Trace that proves it worked:** the SQL above (no harness_closure seeds since deploy); worker log `substrate_endogenous_curiosity_seed_sources` never names `harness_closure`; no `post_turn_closure_prediction_error_write` log lines after deploy.
- **Dangerous failure:** if the glossary cannot be read, curiosity gets no prediction-error seeds for that tick (fail closed). Other seed sources still run. A broken glossary is already loud (self readings and attention log `*_glossary_unreadable`).
- **Rollback:** revert this PR and redeploy orion-substrate-runtime. The node starts being written again on the next unresolved turn.

## Current architecture

- `_prediction_error_candidates` iterated every substrate graph node with `metadata.prediction_error`, decayed it by age, and seeded curiosity above 0.55. It had no value-kind check, so the fixed 0.65 counted as "sustained prediction error".
- `handle_post_turn_closure` wrote 0.65 to `node:substrate.harness_closure` on every unresolved turn, then persisted the turn referent.
- Self readings, world-first attention, the episode gate, and the outcome rollup already excluded placeholder kinds via the glossary.

## Architecture touched

- `orion/substrate/endogenous_curiosity.py` (consumer filter).
- `orion/attention/self_readings.py` (shared helper; same behaviour).
- `services/orion-substrate-runtime/app/worker.py` (producer removed).
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json` (metric removed).
- `orion/substrate/attention_self_model.py` (dead argument removed).
- `services/orion-hub/scripts/attention_organ_routes.py`, `outreach_vocabulary.py` (domain list; comments).

## Every reader of harness_closure and what happened to it

| reader | what it read | action |
|---|---|---|
| `orion/substrate/endogenous_curiosity.py` | node's `prediction_error` (generic scan) | now skipped: not a measurement (no glossary entry) |
| `orion/substrate/attention_self_model.py` `harness_closure_signal` | node metadata dict | argument and branch removed; no caller ever passed it |
| `orion/attention/world_first.py` judge / `orion/substrate/attention_broadcast.py` / `services/orion-attention-runtime/app/worker.py` | generic PE nodes + 7-day history | `judge_candidate` changed from a deny-list (placeholder, bucket) to an allow-list (level, count, trigger, binary) for internal sources. The lingering node (value_kind None) is refused as "unknown: not a measurement" even when fresh (review finding 1; test added). |
| `orion/attention/self_readings.py` | glossary-listed nodes only | already excluded placeholders; now no entry at all |
| `services/orion-equilibrium-service/app/self_signal_outcome_rollup.py` | allowlist of measurement kinds | already excluded; unknown also excluded |
| `services/orion-equilibrium-service/app/self_signal_episode_gate.py` | history magnitudes | judged by the same `judge_candidate`: "unknown", never "hot", at any age (test updated) |
| `scripts/trace_unified_turn.py` | `post_turn_closure_prediction_error_write` log line | dead hop parser removed; test now asserts the line no longer classifies |
| `services/orion-substrate-runtime/app/brain_frame_producer.py` | any node's `dynamic_pressure` / `prediction_error` | shows the lingering node until it is deleted -> operator command below |
| `services/orion-hub/scripts/attention_organ_routes.py` | known-domain list | removed from the list; a lingering node renders as an unlisted extra row with its real age until deleted |
| `services/orion-hub/scripts/outreach_vocabulary.py` | the name only | kept on purpose: a fabricated "harness_closure" mention is still caught as ungrounded; comments updated |
| `services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py` | expected-node list | removed |
| `orion/reverie/referent_loader.py`, `semantic_lift.py`, `services/orion-substrate-runtime/app/turn_referent_store.py` | `harness_closure:<corr>` turn-referent ids in `substrate_turn_referent` | unrelated artifact (turn referent, not the PE node); still written; unchanged |
| `orion/schemas/field_state.py`, `orion/field/significance.py`, `orion/substrate/falkor_codec.py`, hub outreach comments | comments about the 2026-09-07 fabrication incident | history; falkor_codec consumer comment updated |
| `scripts/analysis/replay_attention_eligibility.py` | labels for historical replay rows | kept on purpose with a comment (reads history only) |
| `services/orion-substrate-runtime/README.md`, `services/orion-hub/README.md` | docs that described the producer / reducer argument as live | rewritten as retired |
| `services/orion-field-digester/app/digestion/decay.py` | — | never touched it: harness_closure was FalkorDB-only, not a FieldStateV1 node vector, so there is no fake decay-to-zero path |

## Files changed

- `orion/substrate/endogenous_curiosity.py`: value-kind filter in `_prediction_error_candidates`.
- `orion/attention/self_readings.py`: `MEASUREMENT_VALUE_KINDS`, `node_prediction_error_is_measurement` (shared rule).
- `services/orion-substrate-runtime/app/worker.py`: producer and constant removed; docstrings.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: entry removed, lock regenerated.
- `orion/substrate/attention_self_model.py`: `harness_closure_signal` removed.
- `orion/substrate/falkor_codec.py`: comment.
- `services/orion-hub/scripts/attention_organ_routes.py`, `services/orion-hub/scripts/outreach_vocabulary.py`.
- `services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py`.
- Tests: `orion/substrate/tests/test_endogenous_curiosity.py`, `test_attention_self_model.py`, `test_attention_broadcast.py`; `services/orion-substrate-runtime/tests/test_post_turn_closure_listener.py`, `test_worker_prediction_error_node.py`, `test_worker_pe_history.py`, `test_field_channel_anomaly_listener.py`, `test_prediction_error_domain_map.py`, `test_prediction_error_label.py`; `services/orion-hub/tests/test_field_channel_glossary_routes.py`, `test_outreach_vocabulary.py`; `tests/test_attention_world_first.py`, `test_attention_event_decay.py`, `test_metric_prompt_semantics.py`, `test_field_channel_glossary.py`; `orion/metacog/tests/test_self_signal_episode.py`.

## Schema / bus / API changes

- Added: none.
- Removed: metric `metric://field_channel/orion-substrate-runtime/node:substrate.harness_closure.prediction_error`; Python argument `reduce_attention_self_model(harness_closure_signal=...)` (no callers).
- Renamed: none.
- Behavior changed: curiosity skips non-measurement PE nodes; no harness_closure writes.
- Compatibility notes: no bus channel or schema registry change. `HarnessPostTurnClosureV1` is still consumed (turn referents).

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/test_endogenous_curiosity.py + walkway fence + broadcast world-first: pass (25 in curiosity file, 3 new)
services/orion-substrate-runtime touched tests (8 files): 77 passed
services/orion-substrate-runtime full tests+evals (minus DB-only integration file): 577 passed;
  14 failed + 9 errors, identical set on clean origin/main (env/DB/operator-token tests)
tests/ attention+glossary+metric suites, orion/metacog, orion/reverie, orion/substrate/tests, orion/attention:
  1643 passed after review fixes; remaining fails = 4 felt_state tests, identical on clean origin/main
services/orion-attention-runtime tests (minus env-only test_heartbeat_chassis.py): 55 passed, 6 skipped
services/orion-equilibrium-service tests+evals: 280 passed
tests/test_trace_unified_turn.py, tests/test_metric_rest_range.py: pass
CI-covered suites (self-readings, street-attention, regulation): 137 passed
services/orion-hub touched tests (5 files) + test_endogenous_outreach: pass
services/orion-equilibrium-service rollup tests + evals: 39 passed
services/orion-thought reverie/self-readings tests: 101 passed, 3 skipped
python scripts/check_metric_lineage.py --gate: PASS
python scripts/check_definition_drift.py --gate: PASS (lock re-generated: 1 high "removed")
```

## Evals run

```text
services/orion-substrate-runtime/evals (incl. test_pe_magnitude_live_sanity.py): pass
services/orion-equilibrium-service/evals/test_self_signal_*_replay_eval.py: pass
```

## Docker/build/smoke checks

```text
Not run. No dependency, compose, or env change. Live after-check UNVERIFIED until deploy.
```

## Review findings fixed

Reviewer: orion-repo-agent subagent, target `git diff origin/main...HEAD`.

- Finding (should): `world_first.judge_candidate` used a deny-list (placeholder, bucket). After the glossary entry was removed, a fresh harness_closure reading (value_kind None) would be judged as a real level. Only the staleness rule kept it out, and two tests had been weakened to old ages.
  - Fix: allow-list `MEASUREMENT_VALUE_KINDS` in `orion/attention/world_first.py`, which self readings now imports (one definition). Both tests are back to fresh age and assert "not a measurement".
  - Evidence: `tests/test_attention_world_first.py::test_internal_node_without_glossary_entry_is_never_a_measurement`, `orion/metacog/tests/test_self_signal_episode.py` harness case; attention/metacog/attention-runtime/equilibrium suites pass.
- Finding (should): `scripts/trace_unified_turn.py` still parsed the retired `post_turn_closure_prediction_error_write` line, and a test pinned it.
  - Fix: hop removed from the regex table, the hop order, and the labels; test inverted.
  - Evidence: `tests/test_trace_unified_turn.py` passes.
- Finding (should): substrate-runtime and Hub READMEs described the producer and `harness_closure_signal` as live.
  - Fix: rewritten as retired.
  - Evidence: diff.
- Finding (nit): stale comments in `tests/test_metric_rest_range.py` and `scripts/analysis/replay_attention_eligibility.py`.
  - Fix: comments updated; replay key kept on purpose for historical rows.
- Finding (nit): fail-closed test patches the semantics function, not the glossary exception path.
  - Fix: test now makes `resolve_channel_entry` raise and clears the semantics cache.
  - Evidence: `test_unreadable_glossary_fails_closed_for_prediction_error_only` passes.

## Restart required

Deploy rebuilds orion-substrate-runtime (worker.py, glossary, orion/ package), orion-hub, orion-attention-runtime, orion-equilibrium-service, orion-thought (shared `orion/`, `config/`) through the normal post-pull rebuild from clean main.

Then delete the stored node. Stop the runtime first, because its in-memory graph cache could write the node back:

```bash
docker stop orion-athena-substrate-runtime && docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "MATCH (n:SubstrateNode {node_id:'node:substrate.harness_closure'}) DETACH DELETE n" && docker start orion-athena-substrate-runtime
```

Known caveat: `services/orion-substrate-runtime/README.md` (transport retirement, 2026-07-31) records an open defect where `orion-cortex-exec-background` writes deleted substrate nodes back from a stale snapshot. `node:substrate.transport` is still in the graph today for that reason. It came back **without** `prediction_error`, because that key is excluded from that writer's merge. So the delete may not stick, but the 0.65 should not come back.

Check (pass = 0 rows, or one row with an empty `prediction_error`):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (n {node_id:'node:substrate.harness_closure'}) RETURN n.prediction_error, n.observed_at"
```

The code does not depend on the delete. Every prediction-error reader now refuses this node because it is not a measurement. The delete only removes it from the brain frame and the Hub attention-organ page. The 7-day history rows age out on their own (the runtime prunes the table to 7 days).

## Risks / concerns

- Severity: low. Concern: an unreadable glossary turns off prediction-error seeds for that tick. Mitigation: same fail-closed rule as self readings; other seed sources still run; glossary failures already log.
- Severity: low. Concern: until the node is deleted, the brain frame and Hub attention-organ page still show it at 0.65 with its real (growing) age. Mitigation: operator command above.
- Severity: low. Concern: world-first attention now refuses any internal node with no glossary entry. A future internal PE node shipped without its glossary entry would be silently out of the attention contest. Mitigation: that is the intended rule (the semantic layer decides what is a measurement), and it matches self readings and curiosity. The current 11 measured nodes all have entries (checked live).
- Severity: low. Concern: the stored node may be resurrected by the known cortex-exec-background round-trip defect after deletion. Mitigation: it comes back without `prediction_error`, and every reader refuses it anyway.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3184
