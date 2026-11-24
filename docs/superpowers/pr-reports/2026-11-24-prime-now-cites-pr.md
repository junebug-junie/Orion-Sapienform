## Summary

Orion's Prime reads what the other organs lived before each thought, but almost never says which of those
experiences it used. The cause is the prompt: the JSON reply template had no place for those cites.

- The JSON reply template now ends with a top-level `"refs"` key for experience cites whenever the frame shows
  WHAT YOUR ORGANS LIVED (arc mode and event mode alike).
- The cite paragraph now points at that key, and asks for a cite when the reading draws on an experience.
- The template swap happens once, before any later paragraph is appended, and an import-time assert fails if the
  template's closing text moves.
- Regression test, Prime frame eval check, re-recorded Prime replay golden and now-ablation fixture.

Refs #3425. Refs #3434.

## Outcome moved

Prime thoughts that cite an experience. Live 2026-11-16..11-24: 70 Prime thoughts were shown a now (73
`experience_shown` rows), and 1 thought cited (2 `experience_use` rows, 2026-11-21 20:24Z). Reverie made 958 cites in the
same window. Integration patch 3's acceptance check (#3425) needs `experience_use` rows from Prime.

## Current architecture

- `services/orion-mind/app/prime_loop.py` `_read_now` reads `GET /temporal-self/now?organ=prime&since=...`; the
  frame lists up to 12 items as `[experience:<event_id>]`.
- `orion/prime/thought.py` `system_prompt(..., experiences=True)` appended `EXPERIENCE_PROMPT` as the last
  paragraph. The JSON template it refers to ("Reply with ONE JSON object and nothing else") had only the move's
  `refs` (arc/act, "Any other id refuses the move"), no top-level `refs`.
- `apply_answer` keeps shown cites; `_record_now_use` POSTs them to `/temporal-self/now/use`.

## Where the chain broke (live evidence, read-only)

1. Shown rows stored: yes. 73 `experience_shown` rows for `reader_organ='prime'` since 11-16 00:10Z, 12 items on most.
2. Frame includes the items: yes. `prime_now_read shown_id=... items=12` on each thought in `docker logs orion-mind`;
   the frame rebuilt from a live shown row lists all 12 `[experience:...]` lines.
3. Model answer carries cites: no. Every `prime_thought` log line since the last restart says `now_use=none` and
   carries no `experience cite(s) not shown` or `refs:` warning, so the answers held no experience refs at all
   (not dropped ones). Raw answers are not stored anywhere (cockpit_turn_sighting holds Hub turns only).
4. Parsing / shown filter: not the break. No warnings from either path; the one 11-21 cite parsed and stored.
5. `_record_now_use` / server: works. The 11-21 cite was stored against shown row `dbd5606126b5401a8a12e1e4911072dd`,
   which has `advances_cursor = false` (a `since=` read). So the earlier review worry is disproven: the server's
   `/temporal-self/now/use` checks the shown row's event ids, not whether the read moved the cursor.

## Architecture touched

`orion/prime/thought.py` prompt text only. No schema, bus, API, env or runtime code change.

## Files changed

- `orion/prime/thought.py`: `_REPLY_CLOSE` / `_REPLY_CLOSE_WITH_CITES`, template swap in `system_prompt`, reworded `EXPERIENCE_PROMPT`.
- `orion/prime/tests/test_prime_now_cites.py`: regression test (fails on the old code) and a characterization test.
- `orion/prime/evals/run_prime_frame_eval.py`: the `now` check also requires the cite in the reply template.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `services/orion-mind/tests/test_prime_replay.py`:
  re-recorded; 1 part changed (`now_section_and_one_cite.llm_calls`), 0 after mapping the prompt change back.
- `orion/integration/evals/fixtures/now_ablation_cases.jsonl`: rebuilt (the two Prime cases carry the new prompt).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: Prime's system prompt (only when the now is in the frame).
- Compatibility notes: the parser already accepted top-level `refs`; no consumer changes.

## Env/config changes

- None. `.env_example` not touched; no local `.env` sync needed.

## Tests run

```text
pytest orion/prime orion/integration services/orion-mind/tests -q   -> 802 passed
new regression test on origin/main's thought.py                    -> FAILS (missing top-level experience refs)
```

## Evals run

```text
orion/prime/evals/run_prime_frame_eval.py        -> all PASS (now: ... cite in reply template=True)
run_prime_block_eval, run_prime_hunger_replay, run_concern_evidence_clock_replay, run_intention_take_on_replay -> exit 0
python -m orion.integration.evals.run_now_ablation_eval --offline   -> PASS
```

Live model sample (old prompt vs new prompt, same live-rebuilt frames, same model qwen3.8-flash-next on the agent
lane under an operator hold): see "Live sample" below.

## Live sample

LIVE_SAMPLE_PLACEHOLDER

## Docker/build/smoke checks

```text
Not run: prompt-text change only, no dependency, compose or boot config change.
```

## Review findings fixed

- Finding: "If your reading mentions something ... cite it" ties a cite to a mention, inviting hollow cites.
  - Fix: "If your reading draws on something from that section, cite it."
  - Evidence: `orion/prime/thought.py` EXPERIENCE_PROMPT.
- Finding: the `str.replace` ran after later paragraphs were appended and replaced every match; a template edit could make it a silent no-op.
  - Fix: swap once (`count=1`) before any paragraph is appended; module-level assert that `SYSTEM_PROMPT` holds `_REPLY_CLOSE` exactly once.
  - Evidence: `system_prompt`; tests pass.
- Finding: "the last key of the reply above" was ambiguous with the regulate/take-on snippets in between.
  - Fix: "the top-level "refs" key at the end of the JSON reply template".
- Finding: the second new test passes on old code.
  - Fix: docstring says it is a characterization test.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

(From the clean primary main checkout after merge; the post-pull rebuild does this.)

## Risks / concerns

- Severity: medium. Concern: cites may now be hollow (cited, not used). Mitigation: the now-ablation eval
  (`run_now_ablation_eval --live`) measures hollow-cite share; run it after a week of live cites.
- Severity: low. Concern: `refs` is the last key; an answer cut at `PRIME_MAX_TOKENS=700` loses the whole JSON.
  Mitigation: live readings run ~300 chars; the move was already last before.
- Issues opened: none. Issues closed: none (#3425 stays open until live Prime cites are seen).

## Live check after deploy (UNVERIFIED until run)

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(distinct s.reader_run_ref), count(distinct u.shown_id) from experience_shown s left join experience_use u on u.shown_id = s.shown_id where s.reader_organ = 'prime' and s.shown_at > '<deploy time>'"
docker logs orion-mind --since 24h 2>&1 | grep -o "now_use=[^ ]*" | sort | uniq -c
```

Pass: some Prime thoughts after the deploy show `now_use=stored N/N`, and `experience_use` gains `citing_organ='prime'` rows.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
