# Lease the reflection-register judge (unblocks main's static gates)

## Summary

- The reflection-register judge no longer posts straight to a model seat. It now calls only through an operator lane: one GPU pool hold, every call sent to orion-llm-gateway under it (`orion/evals/leased_llm.py`).
- `judge_one(lane, item)` replaces `judge_one(client, url, item)`. It keeps its return tuple `(label, reason, error, model)` and its passed-model check. A failed call is still an error tuple. A recall of the hold propagates.
- The three callers take `--lane <operator lane>` (default `agent`, the seat they used) instead of `--judge-url`. Each takes one hold per run. On a recall each writes what it judged so far and exits 3; on a refused hold it exits 2.
- The judge body pins `top_p` 1.0 and `presence_penalty` 0, so the gateway's operator defaults do not change the sampling.
- This unblocks main's "Static repo gates" workflow (step "Committed model callers are leased or gateway callers (lane door)"). Relates to #3921.

## Outcome moved

Main's static gates go from red to green. The path that sent 286+ unleased calls to the circe agent seat on 2026-11-25 (`run_self_question_door_replay.py --live --judge-url http://<circe>:8015`) no longer exists; the pool now sees and can recall every judge call.

## Current architecture

`scripts/analysis/reflection_register_sources.py` (self-question-door work, merged 2026-11-25 before #3930's gate) POSTed to `<url>/v1/chat/completions` with an `httpx.Client` and no lease. Three scripts passed it a seat URL from `--judge-url`.

## Architecture touched

Operator scripts and evals only. No service code, no bus, no schema, no env.

## Files changed

- `scripts/analysis/reflection_register_sources.py`: `judge_one` takes a lane; docstring fixed; `DEFAULT_LANE`, `JUDGE_TIMEOUT_SEC`.
- `scripts/analysis/eval_reflection_register_share.py`: `--lane`, one hold (none when every organ reads stored labels), partial `items.partial.jsonl` / `report.partial.md` on recall.
- `scripts/analysis/judge_anchor_set.py`: `score --lane`, one hold, partial score printed on recall, `--export` not written then.
- `services/orion-hub/evals/run_self_question_door_replay.py`: `--live --lane`, one hold, `questions.partial.jsonl` on recall. Fixture mode unchanged.
- `orion/evals/tests/test_reflection_register_judge_leased.py`: new gate tests (fake lane).
- `.github/workflows/leased-evals-tests.yml`: runs the new tests; path filters for the four files.

## Schema / bus / API changes

- Added: none
- Removed: `--judge-url` on the three scripts (replaced by `--lane`)
- Renamed: none
- Behavior changed: judge calls go through an operator hold and the gateway
- Compatibility notes: operator command lines must switch to `--lane agent` (default) and set `ORION_BUS_URL`

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
python3 scripts/check_leased_lane_callers.py
  before: FAIL  scripts/analysis/reflection_register_sources.py [unclassified] line 9: /chat/completions
  after:  PASS (5 leased caller(s), 17 gateway caller(s), 1 reviewed direct; ...)
python3 scripts/check_circe_worker_refs.py   PASS
pytest orion/evals/tests/   60 passed (new file: 20 tests)
```

## Evals run

```text
PYTHONPATH=. python -P services/orion-hub/evals/run_self_question_door_replay.py   PASS: every door gate holds
```

The live modes (`--live`, share eval, anchor score) were not run: they take a real pool hold. UNVERIFIED live.

## Docker/build/smoke checks

```text
Not applicable: no service image or compose change.
Every run step of .github/workflows/orion-static-gates.yml was run locally (see PR comment / CI).
```

## Review findings fixed

A code review ran in a subagent on `origin/main...HEAD`. It found nothing that blocks the patch. It raised four points, and three are fixed here.

- Finding (medium): the lane sends `model: operator_agent`. The gateway rewrites `model` only for vLLM grants, and the agent seat is strata/llama.cpp. If the seat echoes the request's model name, every reply reads `model_not_evaluated: operator_agent`, and the door replay maps every question to `held`.
  - Fix: `served_model(lane, body)`. A reply model that is the route name (or any `operator*` name) is not a model. Then the judge reads the seat's model from the pool's grant record (`lease.grant.model_file`). When nothing is known it stays `model_not_evaluated: unknown`, never a label. A real served name in the reply still wins.
  - Evidence: `test_a_reply_that_echoes_the_route_reads_the_grants_model`. Which name strata actually returns is UNVERIFIED live.
- Finding (low): the share eval's partial report left out the organ that the recall cut off.
  - Fix: on a recall, that organ gets a summary row named `<organ> (cut off by recall)`.
  - Evidence: `test_share_partial_report_counts_the_organ_cut_off`.
- Finding (low): the replay took the GPU hold before its SELECTs ran. A psql failure therefore held a seat for nothing.
  - Fix: both SELECTs run first. The hold covers only the judging.
  - Evidence: `test_replay_reads_before_it_holds_a_seat`.
- Not changed: the reviewer noted that two tests match strings only. They are kept as cheap guards against the old URL mode coming back. The behavior tests cover the rest.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: anyone with a saved `--judge-url` command line gets an argparse error.
- Mitigation: the error is loud and the docstrings show the new command.
- Severity: medium (UNVERIFIED live)
- Concern: nobody has run a live leased judge call yet. So nobody has seen whether strata returns its real model name or echoes `operator_agent`.
- Mitigation: `served_model` handles both cases. Before you trust a full run, run `judge_anchor_set.py score` on a one-row file and check the `model` value.
- Issues: none opened, none closed. Relates to #3921.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3964
