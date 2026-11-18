# Held-seat Mind time limit + repair probe reads bare YES/NO replies

## Summary

- Mind phases on a held seat now get 150 s each, not 60 s. The gpu2 seat runs Bonsai-2-27B, which thinks before it answers. Since PR #3238 sent held turns there, 18 of 59 of its Mind calls stopped at the 60 s cap. (`MIND_LLM_HELD_TIMEOUT_SEC`, `services/orion-mind/app/engine.py`)
- Unheld runs (Juniper's chat turns) keep 60 s. The run wall (180 s from orion-thought) still bounds the whole held run, so nothing that waits on Mind changes its worst case.
- The repair-pressure check now reads a reply of seven bare `YES`/`NO` lines. The prompt says "exactly seven lines, one per kind, YES or NO only", and models sometimes obey it literally and drop the `kind:` label. Line i is kind i of the prompt's list. (`probe_answer_spans`, `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`)
- No hollow fallback: six lines, eight lines, prose, or a non-answer line still fail the check as `probe_unparseable`. Text before the last `</think>` is never read as an answer.
- Bare replies are scored from their own answer tokens' logprobs (same character-position match as labeled replies), not from the fixed text-fallback numbers.

## Outcome moved

- Held Mind phases on Bonsai can finish instead of timing out at 60 s.
- Held repair-pressure checks that got a bare-line answer now produce a real reading instead of `failed_paradigms=repair_pressure:probe_unparseable`.

## Current architecture

- `orion-mind` `run_mind` runs up to three LLM phases (semantic, appraisal, stance). Each phase timeout was `min(MIND_LLM_TIMEOUT_SEC=60, wall remaining)`. That number goes to the gateway as `gateway_read_timeout_sec` (gateway honors up to 900 s) and is also the bus RPC wait.
- Who waits on Mind for a held turn: Hub `ThoughtClient.react` (400 s, `TIMEOUT_SEC`) -> orion-thought `run_mind_for_thought` (HTTP 210 s, sends wall 180 s) -> orion-mind phases.
- The repair probe (cortex-exec `pre_turn_appraisal`) parsed only `kind: YES|NO` lines (`_LINE_RE`). A bare `NO\nNO\n...` reply parsed to nothing and the check failed.

## Architecture touched

- `services/orion-mind`: new setting, held-run branch in the phase timeout, startup warning.
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py` (used by cortex-exec): reply parser.
- No bus, schema, or API change.

## Files changed

- `services/orion-mind/app/engine.py`: held runs use `MIND_LLM_HELD_TIMEOUT_SEC` as the phase cap.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`: new key, default 150.
- `services/orion-mind/app/main.py`: startup log names the held cap; warns if it is not below the caller's Mind HTTP timeout.
- `services/orion-mind/README.md`: held-run time limit.
- `services/orion-mind/tests/test_mind_held_timeout.py`: regression tests.
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: `probe_answer_spans`, `parse_probe_answers`; `_aligned_answer_entries` uses the same spans.
- `tests/test_repair_probe_bare_lines.py`, `tests/fixtures/repair_probe_replies/*.json`: parser tests and 7 captured live replies. The fixtures hold reply tokens and logprobs only, not the thread text.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: held Mind phases may run up to 150 s; a bare seven-line probe reply is a measurement.
- Compatibility notes: none. Deploy order does not matter.

## Env/config changes

- Added keys: `MIND_LLM_HELD_TIMEOUT_SEC=150` (orion-mind)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes, shipped at the live intended value (150).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-mind`; it also added the unrelated missing `PRIME_DAY_LETTER_BID_WINDOW_HOURS=18`).
- skipped keys requiring operator action: none. (`ORION_MIND_RELATIONAL_NAMES` is a pre-existing intentional divergence, untouched.)

## Why 150 s

Pulled from `llm_call_usage` (holder `orion-mind`, pool_role `agent-gpu2`, 11-10..11-13, including spill calls before 11-12):

- Served: 41 calls, median 33 s, p90 49 s, max 56 s. Up to 1,537 completion tokens.
- `upstream_timeout`: 18 calls, all at ~60 s total (wait + generation). The served distribution is cut at 60 s, so the true tail is longer than 56 s.
- Bonsai speed on those calls: median ~26 tok/s, slow end ~14 tok/s (two outliers at 7 tok/s).
- The semantic phase may write 2,048 tokens (`MIND_LLM_MAX_TOKENS_SEMANTIC`). At 14 tok/s that is ~146 s. 150 s covers it.
- On `agent` (Strata) all held Mind calls were served in 11-26 s, so 150 s changes nothing there.

Downstream check: the held run still stops at its 180 s wall. orion-thought waits 210 s for Mind; Hub waits 400 s for the whole stance step. Measured since 11-12, Hub's stance step on Bonsai turns took up to 237 s with Mind at 60 s, so stance work after Mind was up to ~177 s. Worst case now: 180 + ~177 = ~357 s, under 400 s. A slow first phase leaves the later phases less time; they are skipped with `wall_budget_insufficient`, not run past the wall.

## Root cause of `probe_unparseable`

The models sometimes drop the label. The prompt says "Reply with exactly seven lines, one per kind, YES or NO only". Sometimes the model's first sampled token is a bare `NO` or `YES` instead of `specific...`. Once that happens, it writes seven bare lines: `NO\nNO\nNO\nNO\nNO\nNO\nNO`. That is 14 tokens (7 answers + 7 newlines), the same count as all four failed checks. The old parser read only `kind: YES|NO` lines, so it found no answer. The cause is not thinking tokens, not a `<think>` block, not an answer missing from top-k, and not a new logprob shape.

Evidence:

- cortex-exec log: `repair_pressure_probe_no_evidence corr=2bd79f21-... text_head='NO\nNO\nNO\nNO\nNO\nNO\nNO'`.
- Captured live: 11 probes went through the gateway bus path with the same options as `_llm_probe_call` and route `agent`. The `agent-gpu2` and `agent` probes also carried an operator hold on that seat (`orion.evals.leased_llm`); 5 more were unleased one-offs on `agent`. Raw JSON: `/tmp/claude-1000/-mnt-scripts-Orion-Sapienform/5f4e08ea-a4a4-4511-9664-a67c4a299d6c/scratchpad/raw{,_bare}/`.
- First answer token, top alternatives:
  - Bonsai (agent-gpu2, llama.cpp `completion_probabilities`): `specific` -0.16, `NO` -2.6, `YES` -3.5. About 10% of Bonsai replies start bare.
  - Strata (agent, `choices[0].logprobs.content`): `specific` -0.07, `NO` -4.2. About 1-3% start bare.
- One Bonsai capture came back bare on the third small-thread probe: `YES\nYES\nNO\nNO\nNO\nNO\nNO`, 14 tokens.

Gate: I re-scored the 7 captured replies (now `tests/fixtures/repair_probe_replies/`) with the new parser:

- The bare Bonsai reply had no answer under the old parser (`{}` -> `probe_unparseable`). The new parser scores all 7 kinds from their own answer-token logprobs (`logprob_probe_v2`). Scores: 0.299, 0.088, 0.568, 0.042, 0.463, 0.567, 0.162. They vary. None is the fixed text-fallback value (0.087 at confidence 0.65).
- The 6 labeled replies (4 Bonsai, 2 Strata) still score from logprobs, unchanged. Example (Bonsai, thread 112afc17): 0.384, 0.032, 0.064, 0.005, 0.142, 0.108, 0.168.
- `test_captured_replies_score_from_real_logprobs` checks every fixture this way. `test_captured_replies_cover_the_bare_shape` requires at least one bare fixture and one labeled fixture.

Still a failed check (`probe_unparseable`) on purpose:

- 6 or 8 lines.
- Prose.
- Any line that is not YES/NO.
- Numbered lines that are not 1..7 in order.
- Some lines numbered and some not.
- Reasoning that opened `<think>` and never closed.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q services/orion-mind/tests
  273 passed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_repair_probe_bare_lines.py \
  tests/test_repair_pressure_v2_paradigm.py tests/test_logprob_probe_runner.py tests/test_held_turn_lane_rule.py
  all passed (bare-line tests: 27, incl. 7 captured-fixture re-scores)
Regression proof: test_mind_held_timeout.py fails 2/5 against the pre-patch engine.py.
python scripts/check_env_template_parity.py -> PASS
python scripts/check_env_key_single_source.py -> OK
python scripts/check_service_env_compose_parity.py orion-mind -> N/A (env_file)
Note: test_prime_self_monitor_gate.py::test_settings_ship_the_gate_on fails only when root tests/ and
services/orion-mind/tests run in ONE pytest call; same on origin/main without this PR's files. CI
runs them separately (green).
```

## Evals run

```text
No eval harness covers the repair probe parser or Mind phase timeouts. The captured-reply re-score
in tests/test_repair_probe_bare_lines.py is the gate (see above).
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-mind config  -> refused in a worktree: no root .env here (env
template parity PASS before that). Not run; the change is a settings key + Python only.
```

## Review findings fixed

Code review ran in a subagent.

- Finding: there were no captured fixtures, so the "real reply" gate skipped silently.
  - Fix: committed 7 captured live replies, including a real bare Bonsai reply. Added `test_captured_fixtures_exist`, so an empty fixture directory now fails the run.
  - Evidence: fixture tests pass.
- Finding: numbered bare lines like `3. YES\n1. NO...` were mapped by position, so answers could land on the wrong kind.
  - Fix: numbered lines must be numbered 1..7 in order. A mix of numbered and unnumbered lines is refused.
  - Evidence: added cases to `test_anything_else_answers_nothing`.
- Finding: an unclosed `<think>` was read as the answer.
  - Fix: if the last `<think>` comes after the last `</think>`, the reply has no answer.
  - Evidence: added the same tests.
- Finding (nit): `YES.` / `NO.` lines failed to parse.
  - Fix: one trailing period is now allowed.
- Finding (nit): no startup note said that three 150 s held phases do not fit the 180 s wall.
  - Fix: `mind_held_wall_note` log line in `main.py`.
- Not taken: reserve wall time for stance by shrinking the held semantic cap.
  - Why: semantic is the phase every other phase depends on. If it is cut, the run produces nothing. A finished semantic phase with a skipped stance phase still leaves partial output. The trade-off is listed under Risks.
- Not taken (nit, pre-existing): add an RPC margin over `gateway_read_timeout_sec`.
  - Why: the outcome is the same failure; only the gateway's typed reason is lost. It would also change timing for unheld chat-turn Mind calls. Left as is.

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

(`orion-cortex-exec` carries the probe parser via the shared `orion/` package; the post-pull rebuild also restarts its lanes and cortex-orch.)

Post-deploy check (run after a few held turns, e.g. 6 h later; replace the timestamp with the deploy time):

```sql
-- Fix 1: Bonsai Mind calls should be served past 60 s, with few or no upstream_timeout at ~60 s.
select outcome, count(*), max(duration_ms), percentile_cont(0.9) within group (order by duration_ms)
from llm_call_usage
where recorded_at > '<deploy time>' and holder = 'orion-mind' and pool_role = 'agent-gpu2'
group by 1;

-- Fix 2: no new probe_unparseable on held turns.
select status, summary->>'error', count(*)
from cockpit_turn_sighting
where stage = 'pre_turn_appraisal' and status in ('ok','failed') and created_at > '<deploy time>'
group by 1, 2 order by 3 desc;
```

## Risks / concerns

- Severity: high (pre-existing, not caused by this PR)
  - Concern: every held Mind run since 11-12 ends as `fallback_contract_only`. 120 of 120 runs on `agent` and 39 of 46 on `agent-gpu2` failed with `semantic_schema_invalid`; the rest timed out. Fix 1 lets Bonsai phases finish, but a finished semantic phase still fails its schema check, so Mind will still add no coloring to held turns until that is fixed.
  - Mitigation: none in this PR. Next step: capture one held semantic reply and see why it fails `_validate_semantic_raw` (`services/orion-mind/app/synthesis.py`).
- Severity: low
  - Concern: a held run whose semantic phase takes ~150 s leaves ~30 s for appraisal and stance; they are skipped.
  - Mitigation: honest skip telemetry (`wall_budget_insufficient`). Raising the held wall would also need orion-thought's 210 s and Hub's 400 s waits raised; not done here.
- Severity: low
  - Concern: bare-line mapping trusts the prompt's kind order.
  - Mitigation: only an exact seven-line, all-YES/NO reply maps; `test_prompt_order_is_the_bare_line_order` pins the order to `_EVIDENCE_KINDS`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3253

🤖 Generated with [Claude Code](https://claude.com/claude-code)
