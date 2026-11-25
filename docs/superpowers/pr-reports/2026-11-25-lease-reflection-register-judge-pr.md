## Summary

- Main already moved the reflection-register judge and its three callers onto an operator lease (`4da8d4f27`, after this PR was opened). This PR is now narrowed to the one gap that commit left open.
- **The gap:** the lane sends `model=operator_agent`, and the gateway rewrites `model` only for vLLM grants. A llama.cpp/strata seat may echo `operator_agent` back. `judge_one` would then reject every reply as `model_not_evaluated: operator_agent`, so a full run would yield zero labels.
- **The fix:** `served_model(lane, body)`. A reply model that is the lane's route, or any `operator*` name, is not a model. The pool's grant record (`lease.grant.model_file`) names the seat's model instead. When nothing is known, the row stays `model_not_evaluated: unknown` and is never a label. A real served name in the reply still wins.
- Tests for the judge itself (pinned sampling, call error, recall, unevaluated model, route echo), plus a source check that none of the four files has a direct URL mode. They are wired into `leased-evals-tests.yml` along with the four paths.

Relates to #3921.

## Outcome moved

A leased judge run on a strata seat produces labels instead of 100 % `model_not_evaluated`. This is UNVERIFIED live: no leased judge call has run against the real agent seat yet.

## Current architecture

On main, `judge_one(lane, item)` calls `lane.chat(...)` and trusts `body["model"]` for the `PASSED_JUDGE_MODELS` check.

## Architecture touched

`scripts/analysis/reflection_register_sources.py` only (operator scripts). No service, bus or schema changes.

## Files changed

- `scripts/analysis/reflection_register_sources.py`: `served_model`, used by `judge_one`.
- `orion/evals/tests/test_reflection_register_judge_leased.py`: the judge-level tests.
- `.github/workflows/leased-evals-tests.yml`: runs the new test, with the four judge paths as triggers.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest orion/evals/tests/ -> 47 passed
python3 scripts/check_leased_lane_callers.py -> PASS (5 leased, 17 gateway, 1 reviewed direct)
```

## Evals run

None: there is no live seat call yet. Before a full run, check one row: `judge_anchor_set.py score` on a one-row file.

## Docker/build/smoke checks

Not applicable (operator scripts).

## Review findings fixed

- Finding (medium, from the first review of this branch): a route-name echo turns every row into `model_not_evaluated`.
  - Fix: `served_model`.
  - Evidence: `test_a_reply_that_echoes_the_route_reads_the_grants_model`.
- The other findings from that review (partial reports on recall, reading before holding a seat) were about this branch's own caller rewrite. Main's `4da8d4f27` replaced that rewrite, so they are dropped here with it.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. If strata does return its real model name, `served_model` is a no-op.
- Issues opened/closed: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
