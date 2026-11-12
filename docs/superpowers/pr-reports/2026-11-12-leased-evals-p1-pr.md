# Leased evals, Phase 1: evals take a GPU seat at the back of the line and call only through the gateway

Design: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md` (PR #3237).
Juniper approved all 9 defaults on 2026-11-12. This PR builds Phase 1 only. Phases 2 and 3 (the lane
door in front of each model server) are not in it.

## Summary

- Replay evals no longer post straight to a model server. Each one asks the GPU pool for a seat and
  sends every call through orion-llm-gateway under that seat (`orion/evals/leased_llm.py`).
- The pool has new "operator" work classes for this work. They sit at the lowest priority
  (`background`) and wait behind all of Orion's own background work. Orion's work takes the seat
  back when it needs it. Chat's card is used only while Juniper has lent it. The heat reflex sheds
  this work on the same lines as Orion's background work.
- When the pool takes a seat back mid-run, the eval stops, writes a partial report, and exits 3.
  Nothing resumes on its own.
- Orion's body economy does not count this work as Orion's: not in lane use, not in hunger, not in
  arousal strain, not in Orion's energy (it is reported as `sponsored_wh`). The heat price is
  unchanged (answer 3).
- The CI gate for direct circe worker addresses now scans `evals/` and `bench/` folders. On main
  it would have flagged three of the replay evals. A verb may not declare an `operator*` lane.

## Outcome moved

The 11-12 failure mode was a replay eval running on chat's card and the agent card at the same time
as Orion's own work, invisible to the pool. After this PR those evals:

- appear in `gpu_pool_events` (`work_class like 'operator%'`, `priority='background'`);
- wait for a free seat instead of sharing one;
- yield the seat when Orion needs it.

## Current architecture

Replay evals held hard-coded circe URLs (`http://<circe>:8011/8015/8016`) and posted to them
directly. The model-replay eval took control-verb holds at `interactive` priority (never shed by
heat) and then called the worker's URL directly. `check_circe_worker_refs.py` skipped `evals/` and
`bench/`. The lane price counted every lease as Orion's.

## Architecture touched

- GPU pool: new classes and routes, an "operator last" ordering rule in the queue, a refusal of
  operator leases that bid or that ask above `background`.
- Shared helper `orion/evals/leased_llm.py`: `operator_lane` (async), `operator_lane_sync`,
  `OperatorLanes` (one hold per lane per run, taken on first use).
- Evals: promise-ledger, judge-crash, judge-obedience, perception-claims (orion/harness/evals),
  cortex-exec current-turn disclosure, vLLM smoke, model replay.
- Physiology (durable-runs), energy attribution (sql-writer), arousal S2.
- CI gates: `check_circe_worker_refs.py`, `check_verb_physiology.py`.

## Files changed

- `config/gpu_pool.yaml`: classes `operator`, `operator_chat`, `operator_agent`, `operator_agent_gpu2`,
  `operator_agent_deep`, `operator_metacog`, `operator_fast`. Each has a route of the same name at
  `background`. No role owns them.
- `orion/gpu_pool/value_order.py`: `is_operator_class`.
- `orion/gpu_pool/scheduler.py`: rule 6, operator background leases sort after every other
  background lease (`operator_last`, default on).
- `orion/gpu_pool/shed.py`: a queued operator hold is not Orion's deferrable self-work (hunger).
- `services/orion-gpu-pool/app/{runtime,main,settings}.py`, `.env_example`, `docker-compose.yml`:
  `GPU_POOL_OPERATOR_LAST`. `acquire` refuses `operator_class_never_bids` and
  `operator_class_background_only`.
- `orion/evals/leased_llm.py` (new) and `orion/evals/tests/test_leased_llm.py` (new).
- `orion/harness/evals/{promise_ledger,judge_crash,judge_obedience,perception_claims}_replay_eval.py`:
  moved to the helper. Each has a `--lane`/`--lanes` option and a `HOLDER = "operator:..."`.
- `services/orion-cortex-exec/evals/run_current_turn_disclosure_live_eval.py`: `--url` replaced by
  `--lane` (default `chat`, the probe's route).
- `services/orion-vllm-host/evals/smoke_glm.py`: `--url/--route` replaced by `--lane` (default
  `agent-deep`). Runbook and README updated.
- `orion/evals/model_replay/{pool_hold,runner,agent_loop}.py` and `scripts/run_bonsai_replay_eval.py`:
  - `LeaseHoldTransport` takes heartbeated, non-retryable `background` holds on `operator_agent` and
    `operator_agent_gpu2`.
  - Calls go to the gateway with the hold ref.
  - A recall cuts the task and stops the run.
  - `attach_factory` and `AttachedClient` are deleted. The gateway does the attach.
- `scripts/gpu_pool_clear_gpu2.py`: keeps its interactive control-verb holds through `RESERVE_SEATS`.
  It is a reservation Juniper starts by hand, like the Hub button (see Risks).
- `orion/physiology/body.py`, `services/orion-durable-runs/app/{physiology_store,settings,main}.py`,
  `.env_example`, `docker-compose.yml`: `ORION_PHYSIOLOGY_OPERATOR_LOAD_SPONSORED`. An operator
  lease's slot time is left out of the lane's level and taken out of its budget.
- `orion/energy/act_energy.py`, `orion/grammar/act_energy_fill.py`,
  `services/orion-sql-writer/app/{orion_energy_attribution,settings}.py`, `.env_example`, README:
  `ORION_ENERGY_ATTRIBUTION_OPERATOR_SPONSORED`. An operator lease's card share goes to
  `sponsored_wh`. The operator part is also returned as `sponsored_lease_wh` and is not stored.
- `orion/regulation/arousal.py`: S2 leaves operator classes out of the queue.
- `scripts/check_circe_worker_refs.py`: scans `evals/` and `bench/`. One ALLOW entry covers the body
  atlas's recorded pool-state fixtures (data, not calls).
- `scripts/check_verb_physiology.py`: refuses `operator*` lanes.
- `scripts/sync_local_env_from_example.py`: syncs the `ORION_ENERGY_ATTRIBUTION_` prefix.
- Tests across `orion/gpu_pool/tests`, `services/orion-gpu-pool/tests`, `orion/evals/**/tests`,
  `orion/harness/evals`, `orion/physiology/tests`, `orion/energy/tests`, `services/orion-sql-writer/tests`,
  `services/orion-cortex-exec/tests`, `services/orion-vllm-host/tests`, `tests/`.

## Schema / bus / API changes

- Added: pool work classes and gateway routes `operator*` (config). No schema or bus change. No new
  `Priority` value. `GpuLeaseRequestV1` is unchanged.
- Removed: `pool_hold.attach_factory`, `agent_loop.AttachedClient`. Evals lose their `--url` /
  `--llm-url` options (no direct-URL mode, by design).
- Behavior changed: the pool refuses an operator class that bids or asks above `background`.
- Compatibility notes:
  - The gateway reads the routes from its own copy of `config/gpu_pool.yaml`. Until the gateway and
    the pool are rebuilt, an eval's hold is refused with `unknown_class:operator_*`. It fails closed
    and calls nothing (verified live, see Smoke).
  - The circe lane controller parses the same file. The new classes have the existing `ClassSpec`
    shape and no launch digest moved (`check_gpu_pool_config`: digest unchanged, the digest test
    passes).

## Env/config changes

- Added keys:
  - `GPU_POOL_OPERATOR_LAST=true` (orion-gpu-pool)
  - `ORION_PHYSIOLOGY_OPERATOR_LOAD_SPONSORED=true` (orion-durable-runs)
  - `ORION_ENERGY_ATTRIBUTION_OPERATOR_SPONSORED=true` (orion-sql-writer)

  All three ship ON.
- Removed / renamed keys: none.
- `.env_example` updated: yes, all three, plus compose and settings where the service lists env.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. All three keys are in
  the primary checkout's live `.env` files. The sql-writer key needed a new sync prefix, which this
  PR adds.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/gpu_pool/tests                                  571 passed
services/orion-gpu-pool/tests                         202 passed, 15 skipped
orion/evals (helper, model replay)                    127 passed (with operator-class tests)
orion/harness/evals                                    47 passed
orion/physiology/tests                                310 passed
orion/energy + orion/grammar                          225 passed
services/orion-durable-runs/tests                     890 passed, 136 skipped
services/orion-sql-writer/tests                      1229 passed, 214 skipped
services/orion-llm-gateway/tests                      487 passed
services/orion-vllm-host/tests                         38 passed
tests/test_regulation_arousal.py                       80 passed
tests/ (circe gate, verb physiology, env sync, clear_gpu2, power bench, env key)  214 passed
services/orion-hub/tests                             4692 passed, 4 failed. The same 4 fail on main
                                                     (attention_loops_ui_smoke x2, check_async_routes,
                                                     fcc_model_labels).
services/orion-cortex-exec/tests                     14 collection errors ("Verb already registered").
                                                     Identical on main. The new test file passes alone
                                                     (3 passed).
Gates: check_circe_worker_refs PASS, check_verb_physiology OK, check_gpu_pool_config ok,
check_env_key_single_source OK, check_settings_defaults orion-gpu-pool OK,
check_service_env_compose_parity orion-durable-runs OK, check_body_economy_contract OK,
check_chat_route_poachers PASS, check_definition_drift PASS, check_metric_lineage PASS.
```

## Evals run

```text
No live eval was run. Under this design an eval must hold an operator seat, and the live pool does
not know the operator classes until it is redeployed (see Smoke).
Eval-path coverage instead: each migrated eval's call path is pinned by tests:
- no direct worker URL in source;
- a failed call becomes a row error;
- a recall propagates and stops the run.
Gate evidence: the new circe gate, run against main's tree, flags judge_crash, judge_obedience and
perception_claims (the evals behind the 11-12 incident class).
```

## Docker/build/smoke checks

```text
Live smoke (ORION_BUS_URL=redis://<athena tailnet>:6379/0, operator_lane_sync("agent",
holder="operator:leased-evals-p1-smoke")):
  REFUSED agent: the GPU pool did not grant operator_agent (unknown_class:operator_agent)
The helper reached the live pool over the real bus, took no seat and called nothing.
The granted path is UNVERIFIED until the pool and gateway are rebuilt.
No docker build was run; nothing was deployed.
```

## Review findings fixed

A code-review subagent reviewed the diff against origin/main.

- Finding: a queued eval hold counts as load in arousal S2 (blocker). It would make Orion read
  strained while an eval waits.
  - Fix: `gpu_queue_evidence` leaves operator classes out of the queue, held and deferrable maps.
  - Evidence: `test_gpu_evidence_queued_evals_are_never_strain`.
- Finding: Ctrl+C during a model-replay task released the cards before the model threads stopped.
  - Fix: `_run_or_stop` cancels and drains the pair on an outer cancel.
  - Evidence: `test_cancelling_the_run_drains_the_pair_before_returning`.
- Finding: Ctrl+C while queued left the queued hold for up to 6 h.
  - Fix: the interrupt cancels the acquire task, so `gpu_lease` withdraws the hold.
  - Evidence: `test_ctrl_c_while_queued_withdraws_the_queued_hold`.
- Finding: an operator class could be asked at `interactive` by a direct RPC.
  - Fix: the pool refuses `operator_class_background_only`.
  - Evidence: `test_an_operator_class_lease_that_carries_a_bid_is_refused`.
- Finding: the disclosure eval could fall back to its localhost placeholder bus.
  - Fix: it exits 2 with a clear message when `ORION_BUS_URL` is unset.
- Finding: a gateway "hold gone" refusal in model replay was scored as a model failure.
  - Fix: `HoldGone` plus the transport's recall stops the run.
  - Evidence: two tests in `test_runner.py`.
- Nits fixed:
  - `is_operator_class` matches `operator` and `operator_*` exactly.
  - Hold-gone markers narrowed.
  - Ctrl+C releases as `cancelled`.
  - A "waiting in the queue" line after 10 s.
  - The release docstring now explains the `cancelled` outcome.
- Not fixed, listed under Risks: the Phase 1 caller gaps (finding 6) and the idle first lane while a
  second lane waits (finding 7). For finding 7 the waiting line now makes it visible, and
  promise-ledger defaults to one lane.

## Restart required

Deploy order: the pool first, so it knows the classes. The gateway next, so the routes exist. The
two readers can follow in any order. On the primary `main` checkout, after merge:

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) should pick these up from the diff.

The circe lane controller parses `config/gpu_pool.yaml`. It needs no rebuild (shape unchanged, no
launch digest moved), but circe should `git pull` on its normal schedule.

Answer 7 (kev :8009 behind the circe firewall, athena only) is a host change. This PR does not make
it. One line for Juniper. It is not persistent: the gate unit re-applies its own port list on
restart. Making it durable is the Phase 3 edit to `scripts/ops/circe_llm_port_gate.sh` `PORTS`.

```bash
ssh -t circe@circe 'sudo iptables -w -t mangle -I PREROUTING 1 -p tcp --dport 8009 -m addrtype --dst-type LOCAL -j ORION-LLM-GATE && sudo ip6tables -w -t mangle -I PREROUTING 1 -p tcp --dport 8009 -m addrtype --dst-type LOCAL -j ORION-LLM-GATE'
```

## Risks / concerns

- Severity: medium.
  - Concern: Phase 1 callers not moved. None of these is caught by the circe gate: the first is
    localhost on circe, and the others are in the worker's own zone or use a tunnel.
    - `scripts/gpu_power_bench.py` (B12): needs raw llama.cpp `/completion` timings and slot pinning on
      127.0.0.1. It takes no pool hold. Its companion `gpu_pool_clear_gpu2.py` keeps the interactive
      reservation, because a background borrower is recalled by the card's owners and the bench could
      never get a clean card.
    - The worker probes `probe_slot_bleed.py`, `probe_structured_output.py` and the two
      `verify_*thinking_off*` scripts (A6-A9, `services/orion-llamacpp-host/scripts/`): no
      `--via-gateway` option yet.
    - `services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py`: posts through an SSH tunnel
      (127.0.0.1:18013). It is not in the spec's inventory.
  - Mitigation: Phase 2's door counts these as unkeyed calls by source. Before Phase 3 enforces on a
    lane, each one gets `--via-gateway`, or the PR for that lane names it as a holdout.
- Severity: low.
  - Concern: `OperatorLanes` keeps a granted first lane idle while a second lane waits (finding 7).
  - Mitigation: promise-ledger defaults to `--lanes agent`. The helper prints a waiting line after
    10 s.
- Severity: low.
  - Concern: chat-lane evals (judge-obedience, perception-claims and the disclosure eval default to
    `chat`) wait until Juniper lends gpu0 (answer 4). `--lane agent` runs the same model on gpu1.
  - Mitigation: documented in each eval's docstring. A waiting line is printed.
- Severity: low.
  - Concern: the heat price still charges an eval's watts to Orion's heat share (answer 3, revisit
    after 14 days of operator lease data).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3242

🤖 Generated with [Claude Code](https://claude.com/claude-code)
