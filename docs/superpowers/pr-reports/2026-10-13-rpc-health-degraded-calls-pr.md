## Summary

- A reply where the replying service **declined or skipped the work** (a GPU-pool refusal, an unusable request, a canned metacog fallback produced without asking the model, a `system.error`) is now **counted, never timed**. The service that degraded marks its own reply; every caller's `rpc_request()` records it in a new `degraded_count` instead of as a fast success. A failure *after real work* (backend timeout, 5xx, junk model output) is not degraded: its wait was real, and it is the slowness the gate must see.
- The marker is one reason string in the reply envelope's open `trace` dict (`trace["rpc_degraded"]`), or the repo's existing failure kind (`system.error`). No envelope schema change, no taxonomy, nothing guessed from payload content by the caller.
- Producers that mark: **orion-llm-gateway** (the inference contract's refusal and unusable-request outcome classes), **orion-cortex-exec** (a step whose gateway call was refused, a metacog draft that was skipped or refused and fell back to the template, an unusable request), **orion-cortex-orch** (relays a degraded exec/verb reply into its own reply; workflow sub-verbs do not relay).
- The transport baseline never learns speed from a degraded reply: excluded from `fast`/`level`/`floor`; a pool where degraded replies outnumber real calls is tainted and **may not move the floor down**; a degraded-only window breaks the traffic-time chain; it opens no condition (refusals plus a timeout are not `zero_success`).
- Contract: `degraded_count` added (additive, default 0) to `RpcChannelLatencyV1`, `RpcHealthSnapshotV1` and `TransportBaselineHourlyV1`; sql-writer column added at boot. Delivery consumers (signal-gateway organ, substrate rpc_delivery) count degraded replies as *delivered*.
- Replay of the stored 2026-10-08 shed: the pre-fix control reproduces the live collapse (14.4 s -> 1.39 s, live 13.9 s -> 1.26 s); with the fix the floor holds and normal traffic afterwards reads ratio 1.00.

## Outcome moved

The root cause behind PR #2629's false "saturated" readings. #2629 made the gate *state and re-seed* a poisoned floor within hours; this stops the poisoning.

What happened, from stored rows (read-only queries on the live DB):

- `cognition_traces`: on 2026-10-08 every `journal.compose` call in 15:00-19:00 UTC (263/263) and 41 of 46 at 06:00 failed with `gpu_pool_unavailable`. Exec-side duration of a refusal: ~0.44-0.47 s median (some were 115 s `deadline` waits -- slow refusals, also not a speed reading).
- `transport_baseline_hourly`, cortex-orch -> `orion:cortex:exec:request:background`: floor 13,937 ms -> 6,185 ms at 06:00, -> 1,263 ms by 18:00. Hop call counts per hour equal the journal.compose counts (36/69/67/66/68 vs 36/68/63/65/67).
- Correction to the brief: the poisoning calls on that hop were **journal.compose refusals**, not metacog drafts. The metacog "(degraded)" rows (`orion_metacog`, 59/69..66/68 per hour) are the same shed seen from metacog's own lane (`...#log_orion_metacognition`, excluded from the gate). Every metacog trigger is paired 1:1 with an actions `journal.compose` on the unlabelled lane (same correlation id), which is why the two hops' counts match.

Replay (`services/orion-equilibrium-service/evals/run_transport_baseline_degraded_shed_replay.py`, stored counts 2026-10-07 12:00 -> 2026-10-09 23:00):

| | control: refusals counted as successes (pre-fix) | fixed: refusals counted as degraded |
|---|---|---|
| floor before the shed | 14.4 s | 14.4 s |
| floor at 2026-10-08 19:00 | **1.39 s** (live: 1.26 s) | **14.4 s** |
| lowest floor | 1.39 s | 9.7 s (the real post-shed level, 9.4 s) |
| median saturation ratio after the shed | 6.29 (live: ~6x) | **1.00** |
| episodes opened | saturation open + escalate + regime_shift | none |
| refusals counted in degraded_count | 0 | 386 / 386 |

## Which marker defines "degraded", and why

`orion/core/bus/rpc_degraded.py`:

The rule: **degraded = the replier declined or skipped the work.** A failure after real work stays timed.

1. **`reply.trace["rpc_degraded"] = "<reason>"`, set by the service that degraded.** It is the only party that knows its answer is a refusal or a template. Generic (any service can set it), carried in `BaseEnvelope.trace` (an open dict), so old callers ignore it and no envelope schema migration is needed. One reason string, bounded to 120 chars, first cause wins.
2. **`reply.kind in SYSTEM_ERROR_KINDS`** -- already the repo-wide "this reply is a failure" convention.

How a service marks: the Rabbit chassis opens a per-request collector (a contextvar); anything the handler runs calls `note_degraded(reason)` and the chassis stamps the reply. Services that publish their own reply (cortex-exec's manual exec result, its Hunter-based `orion:verb:request`) stamp with `mark_degraded`. Relays are explicit: `rpc_request()` does **not** propagate on its own, because a degraded side lookup (a state fetch, an optional recall) does not make the caller's own answer fake. A marker merely echoed from the request's trace is ignored (rpc_request) and stripped (Rabbit, cortex-exec `_derive_envelope`).

Rejected alternatives: the caller sniffing payload content (`ok: false`, empty text) -- guesses, and payload shapes differ per service; a new first-class `BaseEnvelope` field -- `extra="forbid"` on every consumer in the mesh, a mesh-wide consumer-first migration for one string.

## Gate semantics: excluded, not a failure

Decision: a degraded reply is excluded from speed and **opens no condition**. A degraded reply proves the bus hop delivered and the downstream compute refused -- the GPU pool's domain, already observed by `gpu_pool_events` / `gpu_pool_orion_shed`. Calling it a transport `timeout`/`zero_success` would be a new false *transport* trigger, the failure class this fixes. It stays visible: `KeyObservation.degraded_count` per window, `degraded_count` per hourly row, the `[rpc] reply degraded` log line with the reason.

Beyond exclusion: a pool where degraded replies **outnumber** the real successes is tainted (no `fast` fold, as timeouts already do) and the floor may not move *down* on it -- while the hop is mostly refusing, its few surviving real calls are not evidence of a new best speed. A stray refusal does not taint: at rest there are ~1 gateway refusal/day across all cortex traces (`cognition_traces`, 2026-10-11..13; 486 on 2026-10-08), so a trickle cannot freeze `fast` or the floor. Refusals in windows between pools count toward the next pool. Timeouts keep their existing floor behaviour (unchanged here; they censor too, noted as a follow-up). Calls-per-minute counts degraded replies (they were real load). `zero_success` requires zero *delivered* replies, so refusals plus a timeout do not read as total loss.

## Current architecture

`OrionBusAsync.rpc_request()` recorded every reply as a success with its elapsed time (`RpcHealthAggregator.record_success`), timeouts separately. Per-hop sufficient statistics (`RpcChannelLatencyV1`) were published on `orion:rpc_health:snapshot` and folded by the transport baseline reducer (`orion/metacog/transport_baseline.py`, in orion-equilibrium-service). orion-llm-gateway already classified its own refusals (`grammar_emit.classify_outcome`) and its docstring recorded that "caller-side RPC health counts it as a success"; cortex-exec already failed a refused step by name (`gateway_error_step_failure`) but replied `ok=True`.

## Architecture touched

- Bus core: `orion/core/bus/rpc_degraded.py` (new), `rpc_health.py`, `rpc_health_publish.py`, `async_service.py`, `bus_service_chassis.py` (Rabbit collector + stamp).
- Contracts: `orion/schemas/telemetry/rpc_health.py`, `orion/schemas/telemetry/transport_baseline_hourly.py`.
- Reducer: `orion/metacog/transport_baseline.py` (+ `KeyState.pend_degraded_n`, `KeyObservation.degraded_count`; state schema unchanged, absent field loads as 0; old code ignores it on rollback).
- Consumers: orion-equilibrium-service (hourly row, obs log), orion-sql-writer (column), `orion/signals/adapters/rpc_health.py` (signal-gateway), `orion/substrate/rpc_delivery.py` (substrate-runtime).
- Producers: orion-llm-gateway, orion-cortex-exec, orion-cortex-orch.

## Files changed

- `orion/core/bus/rpc_degraded.py`: the marker, reader, stamp, collector.
- `orion/core/bus/rpc_health.py`: `degraded_count` (pooled + per hop), `record_degraded`, `record_hop_degraded`, `HopLatency.merge`.
- `orion/core/bus/async_service.py`: `rpc_request()` classifies the reply (worker path reuses the envelope `_handle_rpc_result` already decoded; inline path decodes); `record_hop_degraded`.
- `orion/core/bus/bus_service_chassis.py`: Rabbit opens a collector per request, stamps the reply, strips an echoed marker.
- `orion/core/bus/rpc_health_publish.py`: publishes pooled `degraded_count`; merge via `HopLatency.merge`.
- `orion/schemas/telemetry/rpc_health.py`, `transport_baseline_hourly.py`: `degraded_count`.
- `orion/metacog/transport_baseline.py`: exclusion, taint, floor hold, chain break, calls.
- `orion/signals/adapters/rpc_health.py`, `orion/substrate/rpc_delivery.py`: degraded = delivered.
- `services/orion-equilibrium-service/app/transport_baseline_hourly.py`, `app/transport_baseline_gate.py`, `README.md`.
- `services/orion-sql-writer/app/models/transport_baseline_hourly.py`, `app/main.py` (boot `ADD COLUMN IF NOT EXISTS degraded_count INTEGER`).
- `services/orion-llm-gateway/app/grammar_emit.py` (`DEGRADED_OUTCOMES` = refusal + unusable-request classes, `degraded_outcome`), `app/main.py`.
- `services/orion-cortex-exec/app/executor.py` (`gateway_refusal`, notes), `app/main.py` (stamps its own replies).
- `services/orion-cortex-orch/app/clients.py`, `app/orchestrator.py` (relay), `app/workflow_runtime.py` (sub-verbs do not relay).
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: `floor_ms` rest semantics; notes on `degraded_count`.
- Tests: `orion/core/bus/tests/test_rpc_degraded.py`, `orion/metacog/tests/test_transport_baseline_degraded.py`, appended to the signal adapter, rpc_delivery, equilibrium hourly, gateway (`test_degraded_reply_marker.py`), cortex-exec (gateway refusal, metacog fallback, verb reply), cortex-orch (`test_rpc_health_hops.py`).
- Evals: `orion/metacog/evals/run_rpc_degraded_mutations.py`, `services/orion-equilibrium-service/evals/run_transport_baseline_degraded_shed_replay.py` + `fixtures/background_lane_gpu_shed_2026-10-08.jsonl` (60 stored hours).
- `.github/workflows/transport-baseline-tests.yml`: runs the new tests, mutation eval and replay; path filters for the bus core files.
- `docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md`: addendum.

## Schema / bus / API changes

- Added: `RpcChannelLatencyV1.degraded_count: int = 0`; `RpcHealthSnapshotV1.degraded_count: int = 0`; `TransportBaselineHourlyV1.degraded_count: int = 0` (ge=0); SQL column `transport_baseline_hourly.degraded_count INTEGER NULL`; envelope trace key `rpc_degraded` (convention, not schema); `OrionBusAsync.record_hop_degraded`, `RpcHealthAggregator.record_degraded/record_hop_degraded`, `SharedRpcHealthSink.record_hop_degraded`.
- Removed / renamed: none. No new bus channel.
- Behavior changed: `success_count` (pooled and per hop) no longer includes degraded replies; outcomes are disjoint (calls = success + timeout + degraded). `success_latency_ms_*` and log sums exclude them.
- Compatibility: all fields additive with default 0. **No consumer model-validates these payloads on receive** (checked: signal adapter, rpc_delivery, reducer, evidence_map, capture_replay all read dicts; `_validate_payload` runs on publish with the producer's own build) -- so a newer producer never makes an old consumer reject a snapshot. The one strict receiver is sql-writer for `TransportBaselineHourlyV1` (`extra="forbid"`), hence it goes first. `resolve("RpcChannelLatencyV1"/"RpcHealthSnapshotV1"/"TransportBaselineHourlyV1")` returns the edited classes (asserted in tests).

## Deploy order (consumer-first)

1. `orion-sql-writer` -- adds the column at boot; must precede equilibrium (strict validation of the hourly row).
2. `orion-equilibrium-service`, `orion-signal-gateway`, `orion-substrate-runtime` -- read `degraded_count`.
3. `orion-llm-gateway`, `orion-cortex-exec`, `orion-cortex-orch` -- mark and classify.
4. Optional, any order: other callers of cortex-orch/cortex-exec (`orion-actions`, `orion-execution-dispatch-runtime`, `orion-harness-governor`) pick up the classifier in shared `orion/core/bus` on their next rebuild; until then they keep timing degraded replies as before (no worse than today).

Out-of-order is safe in every direction (old consumer + new producer: degraded calls simply vanish from its view, which is the fix for the reducer and a small delivery undercount for the organ; new consumer + old producer: field absent = 0).

## Env/config changes

- Added / removed / renamed keys: none. No feature flag: the marker is inert until a producer sets it, and the exclusion is the bug fix.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## Metric quality gate (`degraded_count`; changed meaning of `success_count` / `floor_ms`)

1. Provenance: `RpcHealthAggregator.record_degraded` / `record_hop_degraded` (`orion/core/bus/rpc_health.py`), called from `OrionBusAsync._record_rpc_reply` when `degraded_reason(reply)` is non-None; per-window `KeyObservation.degraded_count` in `fold_snapshot`; hourly sum in `transport_baseline_hourly.py:_row`.
2. Independence: not new signal -- a re-partition of calls already counted (it moves refusals out of `success_count`). It is independent of latency by construction (never timed).
3. Theory anchor: a latency baseline must be estimated over round trips of the service being measured; a refusal or template is a different process with a different latency distribution (spec guard 1, "cannot learn busy as normal", generalised to "cannot learn refusing as fast").
4. Live data: 2026-10-08 stored rows show the degenerate case the metric exists for (refusals 263/263 for four hours); at rest it is 0 on every hop (no producer marked before this PR). Rest point: 0, reached whenever no downstream refuses -- it is a count, so it can return to 0 within one window.
5. Existing mechanism: `check_metric_lineage.py --metric degraded_count` -> UNREGISTERED before and after; the lineage resolver addresses **float** fields only, so this int is documented in the `transport_baseline_hourly.v1` notes instead (a semantics key on an int field is refused by the resolver). `floor_ms` rest semantics updated and re-locked. `config/field/field_channel_glossary.v1.yaml` has no transport/rpc entries (it covers field channels), unchanged.
6. Reversibility: additive fields; reverting producers stops marking (counts return to success). The SQL column is nullable and can stay.

Semantic-layer trace (`check_metric_lineage.py --metric floor_ms`, after): written by `transport_baseline_hourly.py:161`; blast radius 11 (evidence_map.py:409, grader, gate, hourly) -- all read it as "best recent speed", which is what this fix restores.

## Tests run

```text
orion/core/bus/tests + orion/metacog + orion/signals + tests/test_bus_async_rpc_worker.py
  + tests/test_rpc_delivery_reducer.py + scripts/analysis grader tests           490 passed
orion/core/bus/tests/test_rpc_degraded.py (new)                                     18 passed
orion/metacog/tests/test_transport_baseline_degraded.py (new)                        9 passed
services/orion-equilibrium-service tests + evals                                   192 passed
services/orion-llm-gateway tests                                                   455 passed
services/orion-cortex-orch tests: 31 failed / 217 passed -- identical set on a detached origin/main
  (diffed); tests/test_rpc_health_hops.py 9 passed
services/orion-cortex-exec: full-suite collection is broken on main ("Verb already registered");
  ran all 150 files one process each on origin/main and on this branch -- identical results except
  the files with added tests (test_llm_gateway_overloaded_reply 18->19, test_metacog_two_pass_draft
  10->12, test_verb_runtime_reply_channel 1->2)
services/orion-signal-gateway app/tests                                              38 passed
services/orion-substrate-runtime tests/test_worker_rpc_delivery_tick.py              7 passed
services/orion-sql-writer tests/test_transport_baseline_hourly_sql_shape.py          6 passed
actions 249, mind 119, durable-runs 391, harness-governor 77, execution-dispatch 107 passed;
  thought 1 failed / embodiment 14 failed / substrate-runtime collection error -- identical on origin/main
static gates: check_metric_lineage --gate PASS, --prompt-semantics PASS,
  check_definition_drift --gate PASS (after --update, 1 change), check_inner_state_registry OK (22),
  tests/test_agent_trace_schema_registry.py + schema-skew + substrate ladder + migration drift +
  metric prompt semantics + compose journald: 190 passed
```

## Evals run

```text
PYTHONPATH=. python orion/metacog/evals/run_rpc_degraded_mutations.py
22/22 mutants killed (aggregator: degraded recorded as success [the live bug], merge/absorb drop
degraded; marker: echoed request marker honoured, system.error not degraded, note dropped;
rpc_request: worker/inline ignore marker, stash leak, foreign future stashed; Rabbit never stamps /
hands the handler the request marker; reducer: ignores degraded_count, floor may move down,
mostly-degraded pool not tainted, any single refusal taints, refusals between pools forgotten,
refusals + timeout open zero_success, degraded-only window keeps the chain, not counted as calls,
pend_degraded_n not restored / not reset).
First run (18 mutants): 17/18 -- "worker path leaks stashed reply envelopes" SURVIVED; added
test_rpc_request_worker_path_never_leaks_a_stashed_reply, then KILLED. Review added 4 more.

python services/orion-equilibrium-service/evals/run_transport_baseline_degraded_shed_replay.py
all 5 checks pass (table above)
PYTHONPATH=. python orion/metacog/evals/run_transport_baseline_pause_mutations.py   (unchanged, #2629)
mesh eval + sparse-hop replay: pass unchanged
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh <svc> build (temporary .env symlinks, removed after), all Built:
  orion-sql-writer, orion-equilibrium-service, orion-llm-gateway, orion-cortex-exec (3 images),
  orion-cortex-orch, orion-signal-gateway, orion-substrate-runtime
in-image (docker run --rm, every image): orion.core.bus.rpc_degraded present; degraded_count in
  RpcChannelLatencyV1 and TransportBaselineHourlyV1; sql-writer model has the column
NOT deployed.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD` (read-only; findings 1 and 5 reproduced in scratch scripts). Every finding below is now covered by a test and a mutant.

- Finding (should-fix, high): refusals plus one timeout opened `zero_success` -- with refusals out of `success_count`, a shed window read as total loss: a new false transport alarm in exactly the incident scenario.
  - Fix: `zero_success` requires `degraded == 0` (zero *delivered* replies).
  - Evidence: `test_refusals_plus_a_timeout_are_not_total_loss`; mutant "refusals plus a timeout open zero_success" KILLED.
- Finding (should-fix, high): `_rpc_reply_envelopes` leak -- orion-hub's `harness_governor_client` registers its own futures in `_pending_rpc` and pops only that dict, so every governor reply (often a large agent payload) would be held for the life of the process.
  - Fix: stash only for keys `rpc_request()` itself registered (`_rpc_classify_keys`).
  - Evidence: `test_a_foreign_future_in_pending_rpc_never_stashes_its_reply`; mutant "a foreign pending future's reply is stashed (leak)" KILLED.
- Finding (should-fix): slow failures were hidden from latency -- the gateway marked all backend-failure classes (incl. `upstream_timeout`), cortex-exec marked every metacog fallback (incl. a model that ran and returned junk) and every failed verb, orch marked terminalized timeouts. During real saturation those long replies would have left the statistics: the gate blind to the slowness it exists to see.
  - Fix: one rule, "degraded = the replier declined or skipped the work". Gateway: refusal + unusable-request classes only. cortex-exec: `gateway_refusal()` (refusal classes only) at the step and the metacog draft, plus drafts skipped for prompt size; no blanket `exec_failed`/`exec_error`. orch: no longer marks timeouts/errors.
  - Evidence: `test_a_backend_failure_stays_timed` (gateway), `UPSTREAM_500_REPLY` -> not degraded (cortex-exec), metacog junk-output case -> not degraded, failed verb -> not degraded.
- Finding (should-fix): a trickle of degraded replies could freeze `fast` and the floor (any one refusal tainted the pool); the registry's "0 at rest" claim was unverified.
  - Fix: taint/hold only when degraded replies outnumber real successes in the pool. Verified the at-rest rate live: ~1 gateway refusal/day across all cortex traces 2026-10-11..13 (486 on 10-08); registry note now cites it.
  - Evidence: `test_a_stray_refusal_does_not_freeze_the_baseline`; mutant "any single refusal taints the pool" KILLED.
- Finding (should-fix, latent): a reply that copied a marked request trace kept the inherited reason (first cause wins), which the caller then discarded as an echo -- a real degradation read as success.
  - Fix: Rabbit hands the handler a request without the marker, and drops an inherited marker before applying the noted one.
  - Evidence: `test_rabbit_handler_never_sees_a_request_marker_and_its_own_note_wins`; mutant "Rabbit hands the handler the request's marker" KILLED.
- Finding (deploy order): sql-writer validates `TransportBaselineHourlyV1` (`extra="forbid"`) and must go before equilibrium.
  - Fix: already the stated order; restated in "Deploy order" and the restart line.
- Finding (nit): orch relayed any workflow sub-verb's degradation (some are side fetches) to the whole workflow reply.
  - Fix: `_run_workflow_subverb` runs inside a nested collector that drops the relay; the sub-verb's own `verb:` hop still counts it.
  - Evidence: `test_a_workflow_subverb_does_not_relay_its_degradation_to_the_workflow_reply`.
- Finding (nit): a degraded-only window with no pool open tainted nothing.
  - Fix: refusals between pools count toward the next pool.
  - Evidence: `test_refusals_between_pools_count_toward_the_next_pool`; mutant "refusals between pools are forgotten" KILLED.
- Finding (nit): cortex-exec's self-published reply is stamped before its `finally` grammar flush; a note added there would miss it. Not changed: no note is added after that point today.

## Restart required

From the primary checkout on main after merge, consumer-first, one line:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-equilibrium-service orion-signal-gateway orion-substrate-runtime orion-llm-gateway orion-cortex-exec orion-cortex-orch; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

Expect, after a shed or any refusal: `[rpc] reply degraded corr_id=... reason=gpu_pool_unavailable` in cortex-exec's log and `degraded_count > 0` in `transport_baseline_hourly` for the affected hop, with its `floor_ms` unchanged.

## Risks / concerns

- Severity: medium. Concern: the floor already absorbed on 2026-10-08 is not repaired here (#2629 states the shift and re-seeds; live both hops now read floor ~4 s vs level ~6-7 s). Mitigation: this stops the next shed from doing it again; #2629's regime_shift handles the residue.
- Severity: medium. Concern: a hop whose replies are *all* degraded for long has no latency evaluation at all (nothing real to measure). Mitigation: visible as `degraded_count` per window/hour and in the `[rpc] reply degraded` log; deliberately not a transport trigger (see "Gate semantics").
- Severity: low. Concern: a *fast* backend failure (e.g. an upstream 5xx returned quickly -- the 2026-10-02..06 agent-worker 500s may have been; their durations were not checked here) is still timed and can still pull a floor down. Deliberate: the rule is "declined work", and a backend failure's latency is not knowable as fake vs real without a per-call duration rule. Follow-up candidate.
- Severity: low. Concern: only services that mark are covered. Other refusal sources (any service returning a fallback without marking) still read as fast successes. Mitigation: the marker is generic and one call (`note_degraded`) away for any Rabbit service; the three producers cover the live incident (gateway refusal -> exec -> orch -> actions).
- Severity: low. Concern: timeouts also censor a window (slow calls become timeouts) but still may move the floor down. Unchanged here, to keep this PR's floor change scoped to degraded replies. Follow-up candidate.
- Severity: low. Concern: replay reconstruction (stated in the eval): real round trip 14.0 s / 9.4 s, refusal = exec-side median + 800 ms. Validated by the control reproducing the stored collapse (1.39 s vs 1.26 s) and the post-shed ratio (6.3x vs ~6x).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2640

🤖 Generated with [Claude Code](https://claude.com/claude-code)
