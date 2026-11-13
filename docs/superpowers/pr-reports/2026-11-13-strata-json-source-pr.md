## Summary

- Root cause of the Strata "model did not return valid JSON" 502s, read in Strata's own code on circe: Strata 0.2.7 never forces JSON. `response_format` only adds an instruction to the prompt, then runs a strict check on the finished reply. On failure it returns a 502 and the reply is lost. Grammar-style fields are ignored (four tried live).
- On Strata lanes the gateway now does both steps itself. It sends Strata's exact instruction and runs Strata's exact check, so valid replies come back unchanged.
- One exact repair: a Python-style dict (`{'key': ...}`) is converted only when its meaning can't change.
- One corrective retry: the model sees its own reply and the parser's error. If the retry itself fails, callers still get the original JSON error.
- The raw bad reply is logged (`strata_json_rejected`), so the prompts that cause bad JSON can finally be fixed.
- Removed the finalize judge's own retry (#3292). It's superseded, and keeping it would have stacked retries.

## Outcome moved

Juniper's chat turn (2026-11-13 23:01Z, "el Coco just had his butt stitches removed") was deferred and never answered. Its stance step got this 502. Over 15 hours there were 21 of them, all on the Strata lanes: agent 11, Prime 7, judge 2, chat 1.

- **6 single quotes:** Python-style dicts. Now repaired with no extra call.
- **13 bare quotes:** a quote inside text. Now get one corrective retry.
- **2 unterminated strings:** now get one corrective retry.
- **Anything still failing:** leaves its raw text in the log.

## Current architecture

`_execute_openai_chat` sent `response_format` to Strata, and Strata's 502 became `upstream_http_5xx:http_502: model did not return valid JSON`. Only the finalize judge retried, once, a full round through cortex. Stance, Prime and the agent lane failed outright.

## Architecture touched

- `orion-llm-gateway`: `app/strata_structured.py` is new and is wired into `_execute_openai_chat`. Other backends are unchanged.
- `orion/harness/finalize.py`: the judge-only retry is removed.

## Files changed

- `services/orion-llm-gateway/app/strata_structured.py`: instruction, check, repair, corrective messages.
- `services/orion-llm-gateway/app/llm_backend.py`: takeover; retry loop; a failed retry returns the original failure; time budget.
- `services/orion-llm-gateway/tests/test_strata_structured.py`: new tests. Unit tests for each failure kind, a golden test of Strata's instruction text, and request-path tests against a fake Strata.
- `services/orion-llm-gateway/tests/test_llm_backend.py`: two Strata payload tests now pin the new behavior.
- `services/orion-llm-gateway/README.md`: new section.
- `orion/harness/finalize.py`: retry, its constants, and the unused `deadline_monotonic` parameter removed.
- `orion/harness/tests/test_finalize_judge_json_502_retry.py`: deleted.
- `services/orion-harness-governor/README.md`: the retry paragraph replaced.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Strata lanes no longer receive `response_format`; the gateway sends the same instruction and checks the reply itself. Failure messages are unchanged.
- Compatibility notes: Strata's image has no `jsonschema`, so Strata checked only "one JSON object", and the gateway does the same. Requests Strata used to refuse with a 400 (a malformed schema spec) now go out and fail as a JSON 502 instead. No caller sends those today.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-llm-gateway/tests -> 539 passed
pytest orion/harness/tests              -> 848 passed
gates: diff --check, async routes, env-key single source, hostname refs, definition drift,
       metric lineage, personal terms                                 -> all PASS
```

## Evals run

```text
Live probes against circe-worker-chat (:8011), 2026-11-13:
- bait for a single-quoted dict, with response_format json_object or json_schema -> HTTP 502
  "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)", byte-identical
  to Juniper's stance failure
- grammar, guided_json, top-level json_schema, extra_body.guided_json -> ignored (still single quotes)
- the module live: single quote -> repaired on the first try; normal request -> unchanged; the
  bare-quote bait did not reproduce (the model escaped its quotes)
UNVERIFIED live: the corrective retry on real traffic (#3289).
```

## Docker/build/smoke checks

```text
No new dependency, no compose change. The gateway and governor images rebuild on the post-pull run.
```

## Review findings fixed

- Finding: the Python-literal repair could change string content (`\alpha` became a control character; `\x41`, octal and `\/` also changed).
  - Fix: a repair is refused for any escape Python reads differently from JSON, for prefixed or triple-quoted strings, for adjacent strings and for repeated keys.
  - Evidence: `test_a_repair_never_changes_what_a_string_says` (six cases) and `test_adjacent_strings_are_never_joined`.
- Finding: a failure on the retry (overflow, 5xx) replaced the JSON error. An overflow would also make main.py re-lease and re-run the whole call.
  - Fix: a retry failure returns the original JSON failure and logs `strata_json_retry_failed`.
  - Evidence: `test_a_failed_retry_hands_back_the_original_json_failure` (400 overflow and 503).
- Finding: the retry had no time budget.
  - Fix: the retry runs only when the first call used at most half the read timeout.
  - Evidence: `test_no_retry_when_the_first_call_used_the_budget`.
- Finding: the tests compared the instruction against the gateway's own copy.
  - Fix: a golden test against Strata's literal text.
- Finding: a dead `deadline_monotonic` was left after the judge retry was removed.
  - Fix: removed with its three call-site kwargs.
- Finding: tickets and docs were stale.
  - Fix: #3289 re-scoped to verify the gateway path live; gateway README updated.
- Also from review: a missing `finish_reason` is not treated as cut off (test); a schema containing NaN is left to Strata's 400; the error text matches Strata's.

## Restart required

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The post-pull rebuild after merge covers both.

## Risks / concerns

- Severity: low
  - Concern: logprobs on a reply the gateway rewrote still describe the model's raw tokens, unlike Strata's alignment.
  - Mitigation: only the uncertainty summary reads them, and it only aggregates.
- Severity: low
  - Concern: `strata_json_rejected` logs up to 1,500 characters of the raw reply, which may include chat content.
  - Mitigation: gateway logs are local; this matches existing debug snippets.
- Severity: low
  - Concern: after a retry, `usage` counts only the second call.
  - Mitigation: none in this PR.
- Issues opened: none.
- Issues re-scoped: #3289 (now: verify the gateway path live).
- Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3355

🤖 Generated with [Claude Code](https://claude.com/claude-code)
