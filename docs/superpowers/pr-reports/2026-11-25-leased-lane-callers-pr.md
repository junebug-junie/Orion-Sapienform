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
SMOKE_TESTS_PLACEHOLDER
```

## Evals run

```text
No eval harness change. The two migrated scripts are evals; their pure parts are covered by
orion/evals/tests/test_lane_callers_3921_leased.py. Full runs need private envelopes / live SQL and were not run.
```

## Docker/build/smoke checks

```text
LIVE_SMOKE_PLACEHOLDER
```

## Review findings fixed

REVIEW_PLACEHOLDER

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

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
