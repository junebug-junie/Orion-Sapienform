# Leased lane callers: committed evals call a model only under a lease (#3921)

## Summary

- Two committed evals sent model calls straight to a circe lane, with no lease. The lane door on the agent seat (:8015) counted them as unleased. Once the door enforces, it would refuse them with 401.
- Both evals now take one GPU pool operator seat through `orion/evals/leased_llm.py`. Each call goes to orion-llm-gateway under that seat. There is no direct-URL mode left.
- A new CI gate, `scripts/check_leased_lane_callers.py`, fails when a committed eval or script makes a model call without the lease helper, unless a reviewer has named it as a gateway caller.
- The old address gate (`check_circe_worker_refs.py`) only sees circe address literals. Both bugs took a bare `--url` argument, so it never saw them. The new gate covers that shape.

## Outcome moved

The agent lane's door can be flipped to enforce without breaking a committed eval. Any new committed eval that calls a model must be classified before it merges.

## Current architecture

- `orion/evals/leased_llm.py` already gave evals an operator hold plus gateway-only calls. The gateway passthrough forwards the body unchanged and adds the hold's lease id for the lane door.
- `services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py` took `--url http://<strata-agent-host>:8015` and POSTed with urllib, no lease header. Its docstring said "run under an operator lease". This is the 11-17 batch of 77 unleased calls.
- `scripts/analysis/eval_curiosity_register_judge.py` took `--lane name=<llama.cpp base url>` and POSTed with httpx, no lease. The 11-25 batch of 239 unleased calls came from a scratchpad copy of this judge (`run_judge.py`, not committed).
- `scripts/check_circe_worker_refs.py` scans for literal circe addresses only.

## Architecture touched

- Operator tooling only (evals, scripts). No service runtime, no contract change.
- CI: `orion-static-gates.yml` (new gate + its tests), `leased-evals-tests.yml` (migration tests + path triggers).

## Files changed

- `services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py`: `--url` replaced by `--lane` (default `agent`); calls go through `operator_lane_sync` + `lane.chat`. Logprobs, `logit_bias` and the Strata take-over body are sent unchanged.
- `scripts/analysis/eval_curiosity_register_judge.py`: `--lane` is now an operator lane name; one hold per lane; rows are judged one at a time (one hold = one slot); `--concurrency` removed. A failed call is a row error; a recall stops the run with exit 3.
- `scripts/check_leased_lane_callers.py`: new gate.
- `tests/test_check_leased_lane_callers.py`: gate tests (real tree passes, the bug shape fails, allow list cannot cover a file that names a lane port, stale entries fail, scope).
- `orion/evals/tests/test_lane_callers_3921_leased.py`: the two evals call only through a lane; no URL flag, no worker address.
- `.github/workflows/orion-static-gates.yml`, `.github/workflows/leased-evals-tests.yml`: wiring.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the two evals' CLIs. `eval_json_task_turn_end_after_brace.py --url` is gone (use `--lane`). `eval_curiosity_register_judge.py --lane name=url` is now `--lane <operator lane>`; `--concurrency` is gone.
- Compatibility notes: both need `ORION_BUS_URL` (the pool) and the gateway (`ORION_LLM_GATEWAY_URL`, default `http://127.0.0.1:8210`).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=$PWD .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_check_leased_lane_callers.py \
  orion/evals/tests/test_lane_callers_3921_leased.py tests/test_check_circe_worker_refs.py \
  orion/evals/tests/test_leased_llm.py orion/harness/evals
-> 204 passed in 23.15s

python3 scripts/check_leased_lane_callers.py
-> PASS (5 leased caller(s), 17 gateway caller(s), 1 reviewed direct; lane ports 8011,...,8099)
python3 scripts/check_circe_worker_refs.py
-> PASS
Regression check: the gate run against origin/main's copies of both evals gives
  eval_json_task_turn_end_after_brace.py -> names_lane_port (line 20, :8015)
  eval_curiosity_register_judge.py       -> unclassified (line 15, /v1/chat/completions)
```

## Evals run

```text
No eval harness change. The two migrated scripts are evals; their pure parts are covered by
orion/evals/tests/test_lane_callers_3921_leased.py. Full runs need private envelopes / live SQL and were not run.
```

## Docker/build/smoke checks

```text
Live smoke: UNVERIFIED. One helper call (python -m orion.evals.leased_llm chat, holder
operator:smoke-3921-leased-lane-callers) was tried on two lanes. Both refused before any call was sent:
  --lane agent       --queue-wait-sec 300 -> REFUSED: the GPU pool did not grant operator_agent (deadline)
  --lane agent-gpu2  --queue-wait-sec 600 -> REFUSED: the GPU pool did not grant operator_agent_gpu2 (deadline)
Operator work waits behind all of Orion's own work by design, and both seats stayed busy.

Agent door (/door/stats on :8015), shadow mode, read during this session:
  unleased 239 -> 244 -> 265, all from 100.92.216.81 (athena) on /v1/chat/completions.
None of these came from this smoke (its calls were refused before sending). Some other athena caller,
not committed to the repo, is still posting directly. Commented on #3921.
No Docker build needed: no service code changed.
```

## Review findings fixed

The code review ran in a subagent against commit abd76f5d3.

- Finding: the gate missed f-string shapes (`f"{base}/completion"`, `/chat/completions` without `/v1`, `/infill`, OpenAI client calls).
  - Fix: `MODEL_CALL_RE` widened. The port gate's reviewed `DIRECT_CALLERS` (`scripts/gpu_power_bench.py`) are now their own kind, `direct`. The wider pattern found 3 more callers. All 3 go through the gateway, and all 3 are now classified.
  - Evidence: `test_every_model_call_shape_counts`, `test_near_misses_are_not_model_calls`, `test_reviewed_direct_caller_passes_even_naming_a_port`.
- Finding: any import of `orion.evals.leased_llm` counted as leased, even an import of an exception class only. A commented-out shell CLI line also counted.
  - Fix: an AST check now requires an import of `operator_lane_sync`, `operator_lane`, `OperatorLanes` or `gateway_call`. Shell comment lines are ignored. `orion/evals/model_replay/agent_loop.py` imported only `_hold_gone`, so it no longer counts as leased. It is now a reviewed gateway caller: it posts to the gateway with the task's hold ref.
  - Evidence: `test_importing_only_an_exception_is_not_a_lease`, `test_a_commented_shell_cli_call_is_not_a_lease`.
- Finding: two allow-list reasons said `:8222` was the gateway's port. Nothing listens there (`curl 127.0.0.1:8222/health` gets no connection), and the gateway publishes only 8210.
  - Fix: the defaults and docs in `orion/integration/evals/run_now_ablation_eval.py` and `services/orion-hub/evals/run_reading_resonance_eval.py` now use `:8210`. The reasons are corrected.
- Finding: in the judge, a recall partway through a lane threw away every row already judged.
  - Fix: rows are added one at a time. On a recall the judge writes `labels_<lane>.partial.jsonl` and a report with `recalled`/`judged`/`of`, then exits 3.
  - Evidence: `test_judge_recall_writes_a_partial_report`.
- Finding: on `operator_metacog` the gateway adds `top_p 0.90` / `presence_penalty 0.20` when the body leaves them unset (`with_gpu3_route_defaults`). That changes the judge's measurement.
  - Fix: the judge body sets `top_p: 1.0` and `presence_penalty: 0` explicitly.
  - Evidence: `test_judge_pins_sampling_the_gateway_would_otherwise_default`.
- Finding (nit): the brace eval crashed with a traceback on an unknown lane or a failed call.
  - Fix: `--lane` accepts only the operator lane names. A failed reading is recorded as a string, and a recall still stops the run. The help text says the token id assumes the flash-next vocabulary.
- Finding (nit): scope was limited to `evals/`, `scripts/` and `bench/`.
  - Fix: `tools/` and `experiments/` are added. The docstring names the boundary and the text-only limit.
- Gateway passthrough check (no change needed): the reviewer confirmed that `logprobs`, `top_logprobs`, `logit_bias` and `chat_template_kwargs` reach the worker unchanged on `operator_agent`. The passthrough does not run the Strata take-over a second time, so the brace eval's number means the same thing it did before.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: the gate classifies by file text. A script that builds the model-call path from pieces (for example `"/v1/" + "chat/completions"`) would not be seen.
  - Mitigation: the lane door is the runtime fence; under enforce it refuses such a call. The gate catches the shape that actually happened.
- Severity: low
  - Concern: the gateway callers named in `GATEWAY_CALLERS` call by route without an operator hold, so their work runs on Orion's own classes, not `operator_*`. That is outside this issue (the gateway leases each call, so the door sees them leased).
- Issues: Fixes #3921. Opened: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3930

🤖 Generated with [Claude Code](https://claude.com/claude-code)
