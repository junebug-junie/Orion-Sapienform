## Summary

PR #3773 (merged and deployed 2026-11-24 03:37Z) put Prime's experience cites into the JSON reply template, as the
last key. A live model sample then showed that template breaks Prime's JSON: the model drops the final closing brace.
This patch moves the cite line before `"move"`, so the template ends exactly as it did before #3773.

- `"refs": [] | [{"kind": "experience", ...}]` now sits between `"concerns"` and `"move"`.
- The template's ending (`"id": "..."}]}}`) is unchanged from before #3773; an import-time assert pins the move line.
- The cite paragraph says "the one before "move"".
- Regression test fails on main (refs last) and on the pre-#3773 code (no refs in the template).

Refs #3425. Refs #3434. Follow-up to #3773.

## Outcome moved

Prime's answers stay valid JSON while the template carries the cite.

Live sample, same model as Prime (qwen3.8-flash-next on circe, agent lane, operator hold), frames rebuilt from
Prime's real now reads of 11-23/11-24, temperature 0.4, thinking off, max_tokens 700:

| template | answers | invalid JSON | `refs` key present | answers with a kept cite |
|---|---|---|---|---|
| before #3773 (no refs in template) | 15 | 0 | 0 | 0 |
| #3773 (refs last), no JSON mode | 15 | 6 | 9 | 0 |
| #3773 (refs last), JSON mode as live | 1 | 1 (gateway 502 `structured_output_failed`) | - | - |
| this PR (refs before move), JSON mode as live | SAMPLE_PENDING |

Probe of one invalid #3773 answer: `... "move": null, "refs": [] ` and then end of output. The last `}` is missing.

Live gateway path: Prime sends `return_json`, the gateway sets `response_format: json_object` and refuses invalid
JSON with a 502; Prime retries a 5xx once, then the thought fails.

## Why Prime almost never cites (diagnosis, read-only)

1. Shown rows stored: yes. 73 `experience_shown` rows for Prime, 11-16..11-24, 70 distinct thoughts.
2. Frame lists the items: yes (`prime_now_read ... items=12`; frame rebuilt from a real shown row lists all 12).
3. Model answers carry cites: no. Before #3773, 15 of 15 sampled answers had no `refs` key at all. Live
   `prime_thought` lines all say `now_use=none` with no cite warning.
4. Parser and "kept only when shown" filter: not the break. No warnings; the one live cite (11-21) parsed and stored.
5. `_record_now_use` and the server: work. The 11-21 cite stored against a `since=` read (`advances_cursor=false`),
   so the cursor concern from the #3501 review does not block cites.

Second cause, upstream: the now rarely shows Prime anything that bears on its concerns. Kinds with no salience
(reading_turn, vision_percept, attention_row) sort after every kind that has one, even a prediction surprise at 0.00.
Two MemOnce reading turns were recorded while Prime held a MemOnce concern and were never shown. Filed #3783.

## Files changed

- `orion/prime/thought.py`: `_REPLY_MOVE` / `_REPLY_CITES_THEN_MOVE` replace `_REPLY_CLOSE` / `_REPLY_CLOSE_WITH_CITES`.
- `orion/prime/tests/test_prime_now_cites.py`: template test pins refs before move and the unchanged ending.
- `orion/prime/evals/run_prime_frame_eval.py`: `now` check requires refs before move and the unchanged ending.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `services/orion-mind/tests/test_prime_replay.py`:
  re-recorded; vs the pre-#3773 golden, 1 part changed, 0 after mapping the prompt change back.
- `orion/integration/evals/fixtures/now_ablation_cases.jsonl`: rebuilt.

## Schema / bus / API changes

None. Prompt text only.

## Env/config changes

None.

## Tests run

```text
pytest orion/prime orion/integration services/orion-mind/tests -q   -> 805 passed
template regression test with main's thought.py                    -> 1 failed (refs last)
template regression test with pre-#3773 thought.py                 -> 1 failed (no refs)
```

## Evals run

```text
run_prime_frame_eval.py -> all PASS (now: ... cite in reply template=True)
run_prime_block_eval, run_prime_hunger_replay, run_concern_evidence_clock_replay, run_intention_take_on_replay -> exit 0
run_now_ablation_eval --offline -> PASS
live sample -> table above
```

## Docker/build/smoke checks

```text
Not run: prompt text only.
```

## Review findings fixed

From the #3773 review (all fixed there): "draws on" wording, single swap before paragraphs plus an import-time
assert, unambiguous key wording, characterization-test docstring. This PR keeps all four.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: high until this deploys. Concern: main runs the refs-last template; Prime thoughts can fail on invalid
  JSON. Mitigation: merge and deploy this PR.
- Severity: medium. Concern: even with a valid template the model cited nothing in the sample; the now's content
  rarely bears on Prime's concerns. Mitigation: #3783.
- Issues opened: #3783. Issues closed: none.

## Live check after deploy (UNVERIFIED until run)

```bash
docker logs orion-mind --since 24h 2>&1 | grep -E "prime_thought |prime_try_no_model" | grep -oE "now_use=[^ ]*|kind=[a-z_0-9]+" | sort | uniq -c
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*) from experience_use where citing_organ = 'prime' and cited_at > '<deploy time>'"
```

Pass: no `prime_try_no_model` with a JSON or structured-output error, and Prime thoughts keep landing.
Prime cites: expected rare until #3783.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
