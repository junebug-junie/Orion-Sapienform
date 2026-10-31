## Summary

- Broken-JSON failures since the 2026-10-30/31 Strata swap come from three different causes. The swap caused one of them. The other two were already happening.
- **Reading turns (main cause, present before the swap):** a long answer goes over the output limit (16,384 tokens, mostly spent on thinking). The Claude Code CLI then asks the model to continue in a new response. The harness kept only the last response, so the JSON started in the middle of a string. A new `OutputCapStitcher` joins the pieces again in both stream readers (`fcc_motor`, the hub's `fcc_claude_bridge`).
- **Episode distill (caused by the swap):** Qwen3.8-Flash-Next writes longer answers than Qwen3.8-27B. One episode needed about 4,040 tokens, and all 12 calls on it stopped at the 4,096 limit. The budget is now 8,192. A cut answer now fails as `distill_truncated_at_max_tokens`, not as the misleading `distill_json_invalid`.
- **Stance (Strata, not fixed here):** Strata does not constrain decoding to valid JSON. It checks the answer after it is written (`method: prompt_and_validate`) and returns HTTP 502 when the answer is invalid. This happened on 2 of 164 Strata stance calls since the deploy. Both answers were genuinely malformed, not truncated. See Risks.

## Outcome moved

- On the governor's 83 real transcripts that hit the output cap (2026-10-20..31), 22 cut JSON answers failed to parse. Replayed through the stitcher, 14 of them now parse. All 24 that already parsed still parse, so nothing regressed.
- Since the 08:01Z deploy, 2 of the 3 cap-split reading failures recover on replay (10:53Z `c5df42da`, 17:00Z `832767bd`). The 18:54Z one still fails because the model itself retracted and rewrote a fragment.
- Episode distill: the stuck episode (`41d53643`) gets ~2x the budget it needs. **UNVERIFIED** until the restart, because the probe ran the prompt directly on the engine, not through the service.

## Current architecture

- Reading turn: Hub -> harness governor -> `fcc_motor` runs `claude -p` in stream-json mode. The draft is "last extracted text wins" (`extract_final_from_stream_event`), then `world_pulse_read_stage2` calls `parse_json_object(outcome.text)`.
- Episode distill: orion-durable-runs calls the gateway directly on route `memory_distill` with `max_tokens=4096`, thinking off and `structured_output_method=json_object_only`. Without a schema the gateway sends no `response_format` (`apply_structured_output_to_payload` returns early), so distill gets no JSON check from the engine.
- Stance: orion-thought -> cortex-exec -> gateway -> Strata, with `response_format` set. Strata validates afterward and returns 502 `structured_output_failed`.

## Evidence

Rates (`substrate_durable_run_state`, reading.turn, `Could not parse JSON object`):

| window | turns | parse failures | rate |
|---|---|---|---|
| 10-23..10-30 (before swap) | 759 | 26 | 3.4% |
| 10-31 all day | 242 | 13 | 5.4% |
| 10-31 since 08:01Z deploy | 165 | 7 | 4.2% |

On 10-31, 6 of the 13 failures were Flash-Next `<tool_call>` text leaking into the answer between 00:41 and 03:41Z, before the deploy. None appeared after it. The 7 since the deploy: 3 output-cap splits (Bonsai), 3 prose-only answers where the agent gave up and never wrote JSON (Bonsai), and 1 Flash-Next answer that used single quotes for strings. Before the swap the same families appear: 9 mid-JSON tails (Qwen3.8-27B and Bonsai) and 17 prose-only answers.

Output cap, from the governor's Claude Code transcripts: 83 of about 4,900 sessions since 10-20 hit the cap. Bonsai had 46, Qwen3.8-27B had 34, Flash-Next had 1. The cut responses held 6.18M characters of thinking and 0.21M characters of answer text. Turn `c5df42da`: three responses of 16,384 output tokens each, about 60k characters of thinking per response; answer text was 3,876 chars in the second and 7,993 chars in the third.

CLI behaviour, reproduced with the host `claude` 2.1.316 against a fake Messages server that cut the first response with `stop_reason: max_tokens`: the stream had `stop_reason: null` on both assistant events, the resume instruction arrived as a plain `user` event, and `result` held only the tail. The resume instruction is therefore the only reliable marker.

Distill (`llm_call_usage`, route `memory_distill`): before the swap, Qwen3.8-27B wrote at most 3,872 tokens (median about 1,850). There was 1 earlier failure on 10-26 (not inspectable, logs rotated). On 10-31, Flash-Next got 12 calls for episode `41d53643` (8,303 prompt tokens); all 12 stopped at 4,096 tokens with `finish_reason=length` (gateway log). The rendered prompt was sent to circe:8015 directly with `max_tokens=12000`: it finished at 4,037 tokens with `finish_reason=stop`, and the answer parses (6 memories, 2 questions).

Stance (`llm_call_usage` + gateway log): 2 of 164 Strata stance calls since 08:01Z returned 502 (`Expecting property name...` at char 315; `Unterminated string` at char 721). Both had `max_tokens=8000`, took about 15 s, and Strata did not report `finish_reason=length` (it has a separate message for that). So these were malformed, not truncated. The Strata server source (`/opt/strata/terminal/serve/structured.py`) and its capabilities report `structured_output.method = "prompt_and_validate"`. A direct probe on :8011 confirmed it: asked to echo `{'a': 1, b: True}`, the engine returned 502 instead of producing constrained JSON.

## Architecture touched

- `orion/fcc/output_cap_continuation.py` (new): one stitcher, shared by both FCC stream readers.
- `orion/harness/fcc_motor.py`, `services/orion-hub/scripts/fcc_claude_bridge.py`: feed each event to the stitcher; draft = stitched text. New metadata `fcc_output_cap_continuations`.
- orion-durable-runs: distill call returns `finish_reason`; graph raises `distill_truncated_at_max_tokens:<n>` on `length`; default budget 8192.

## Files changed

- `orion/fcc/output_cap_continuation.py`: the stitcher. A piece cut before the resume instruction becomes the head. The next assistant text continues it. A tool result or any other user event ends the continuation. A piece that opens with the same first 32 characters as the head is a rewrite and replaces the head. The resume instruction itself never becomes the draft.
- `orion/harness/fcc_motor.py`: wire the stitcher; add the metadata count.
- `services/orion-hub/scripts/fcc_claude_bridge.py`: same wiring.
- `orion/harness/tests/test_fcc_motor_output_cap_continuation.py`: unit tests for the stitcher, plus an end-to-end `run_fcc_turn` test that uses the stream shape the real CLI produced. The end-to-end test fails on main.
- `services/orion-hub/tests/test_fcc_claude_bridge_run.py`: the same end-to-end case for the bridge.
- `services/orion-durable-runs/app/runner.py`: `_call_memory_distill_llm` returns `finish_reason`.
- `services/orion-durable-runs/app/episode_distill_graph.py`: a truncated answer is a named failed attempt.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `MEMORY_EPISODE_DISTILL_MAX_TOKENS` 4096 -> 8192.
- `services/orion-durable-runs/tests/test_episode_distill_graph.py`, `tests/test_runner_memory_distill_finish_reason.py`: regression tests.
- `.github/workflows/orion-fcc-warm-pool-tests.yml`: runs the new motor test.

## Schema / bus / API changes

- Added: none on the bus. New FCC final-frame metadata key `fcc_output_cap_continuations` (int).
- Removed / Renamed: none.
- Behavior changed: an FCC draft that the output cap split now holds the whole answer instead of the last piece. A distill attempt cut at the budget logs `distill_truncated_at_max_tokens:<n>`.
- Compatibility notes: an uncut turn's draft is byte-identical.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- Changed default: `MEMORY_EPISODE_DISTILL_MAX_TOKENS` 4096 -> 8192 (settings, `.env_example`, compose default).
- `.env_example` updated: yes (`services/orion-durable-runs`).
- local `.env` synced: `scripts/sync_local_env_from_example.py` adds keys only and does not change values. The primary checkout's `services/orion-durable-runs/.env` was edited by hand from 4096 to 8192. Without that edit, the old `.env` value would override the new compose default.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/harness/tests (warm pool, motor cancel/served-model/write-stamp/output-cap/repeat-breaker,
  context ceiling, chat memory, undeclared tool, call usage) + orion/llm/tests/test_call_usage.py -> 135 passed
orion/fcc/tests -> all pass except test_claude_spawn::test_no_undiscovered_root_callers_of_claude_permission_argv,
  which also fails on main (it names files this PR does not touch)
services/orion-hub: test_fcc_claude_bridge_{run,parse,mcp}.py, test_agent_claude_chat_history.py -> 21 passed
services/orion-durable-runs: tests -> 627 passed, 117 skipped
regression check: test_motor_final_frame_carries_the_whole_answer fails with main's fcc_motor.py
python3 scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
Replay eval (ad hoc, real data): the stitcher run over the governor's 83 transcripts that hit the cap.
  JSON-shaped drafts: 46. Parse before -> after: fail->ok 14, ok->ok 24, fail->fail 8, ok->fail 0.
  The 8 remaining failures are model-side continuation mistakes (overlapping text, a reopened object,
  a retracted fragment), not stitching errors.
Direct engine probe (circe :8015), episode 41d53643's real distill prompt, thinking off:
  finish_reason=stop, 4,037 completion tokens, parses (6 memories, 2 questions).
```

No checked-in eval harness covers FCC drafts. The replay above is a one-off run, not a periodic eval. Follow-up: a periodic "cap-hit transcripts parse rate" eval over the governor's transcripts.

## Docker/build/smoke checks

```text
Not run. Deploy is out of scope for this task. Live effect is UNVERIFIED until the services restart.
```

## Review findings fixed

(filled in after review)

## Restart required

```bash
./scripts/rebuild_services_from_git_diff.sh
```

Or per service, from the clean primary main checkout after merge: `scripts/safe_docker_build.sh orion-harness-governor up -d --build`, `scripts/safe_docker_build.sh orion-hub up -d --build`, `scripts/safe_docker_build.sh orion-durable-runs up -d --build`.

## Risks / concerns

- Severity: medium. Concern: thinking still uses about 97% of the output budget on cut FCC responses (6.18M chars of thinking vs 0.21M chars of answer). That is about 12 minutes of GPU per cut response on Bonsai. This PR recovers the answer but does not stop the burn. Mitigation / follow-up: proposal to lower reasoning effort or turn thinking off for the reading stage-2 final write-up. That change touches how Orion reasons, so it needs proposal mode.
- Severity: low-medium. Concern: Strata has no grammar-constrained decoding, so `response_format` is only checked after the answer is written. About 1.3% of Strata stances return 502, and the reading turn defers and retries. Before the swap, llama.cpp's grammar made malformed JSON impossible. Mitigation / follow-up: ask upstream whether Strata can constrain decoding, or keep JSON-critical verbs on a llama.cpp lane.
- Severity: low. Concern: distill sends `structured_output_method=json_object_only` without a schema, so the gateway sends no `response_format`. The code comment says "JSON-object output", but no JSON-object check happens. This PR does not change that, because a Strata post-check would only turn a parse failure into a 502. Follow-up: correct the comment, or send a schema.
- Severity: low. Concern: the restart rule compares the first 32 characters. A model that continues by repeating the answer's first 32 characters inside the cut piece would be treated as a rewrite. That did not happen in any of the 83 transcripts.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
