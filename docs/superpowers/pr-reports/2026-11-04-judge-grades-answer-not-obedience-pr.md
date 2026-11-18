## Summary

- Unified-turn self-prediction step **A2**. The finalize "5b" self-check (`harness_finalize_reflect.j2`) now judges only two things:
  - does the draft answer the message;
  - is it grounded in what the turn actually saw (tool results, receipts, situation notes).
- Stance's imperative is context for tone, never the standard. A draft that corrects the imperative from evidence is aligned.
- New rule: a draft that is a tool call, or tool-call markup, does not count as a reply.
- **A failed self-check is no longer reported as a misaligned reply.** When the 5b call fails (bad JSON, no GPU slot), the harness writes a fallback "misaligned" verdict. That happened on 111 of 1,439 turns in 7 days. The metacog chat-turn gate counted every one as a misaligned reply. It now fires `reflection_failed` and passes `finalize_degraded_reason=reflection_failed` with no verdict, and the repair reason is `reflection_failed`.
- New opt-in eval: it replays real turns through the old and the new prompt.

## Outcome moved

Before/after replay of real turns from the last 7 days, on the chat lane with the same model settings for both prompts. 33 turns loaded; 7 lacked stored inputs.

| bucket | turns | judged not aligned, old -> new | notes citing the imperative, old -> new |
|---|---|---|---|
| **obedience**: the stored verdict was misaligned or uncertain and cited the imperative | 14 | 62% -> 54% | 6 -> 1 |
| **control**: the stored verdict was aligned | 19 | 5% -> 5% (the same turn) | 10 -> 2 |

- Notes that cite the imperative fell from 16 to 3. The 3 remaining are, at most, mentions in passing.
- No control turn changed verdict.
- **Most "obedience" turns were marked down correctly.** In 7 of the 14, the draft was a raw tool call, an unfinished agent loop, and not a reply. Their old notes cited the imperative, but the real fault was the missing answer.
- In one such turn, the new prompt first called a tool-call draft aligned, because the call "addresses" the question. The tool-call rule was added after that.
- I then re-judged the 9 turns the old prompt marked down: all 8 that returned a readable verdict are misaligned, for the right reasons (tool-call draft, or unsupported claims). None cites the imperative. One turn is unreadable under both prompts.
- Raw verdicts and notes: `/tmp/judge-obedience-replay/report.json` and `recheck.json`.

## Current architecture

- The INTEGRATIVE CHECK said to "compare draft_text against thought_event.imperative, tone, and strain_refs" and to judge alignment with "the task, felt stance, and substrate".
- A failed 5b call produced `alignment_verdict="misaligned"` with `reflection_source="degraded_llm_failure_fallback"`. Every consumer read that as a real verdict.

## Architecture touched

- Prompt: `harness_finalize_reflect.j2`.
- Shared schema: the constant `REFLECTION_SOURCE_JUDGE_FAILED` in `orion/schemas/harness_finalize.py` (no shape change).
- Harness: `finalize.py` uses the constant, and `response_repair_reason_for` returns `reflection_failed`.
- orion-equilibrium-service: `chat_turn_metacog_gate.py`.

## Files changed

- `orion/cognition/prompts/harness_finalize_reflect.j2`: two questions; the imperative becomes context; the tool-call-is-not-a-reply rule; world-contact cross-references reworded to match.
- `orion/cognition/prompts/tests/test_imperative_first_prompts.py`: contract tests for the new framing. Grounding and world contact are still required.
- `orion/schemas/harness_finalize.py`: `REFLECTION_SOURCE_JUDGE_FAILED`, with a comment on who handles it.
- `orion/harness/finalize.py`: the constant, and the `reflection_failed` repair reason.
- `orion/harness/tests/test_response_repair_gate.py`: covers the repair reason.
- `services/orion-equilibrium-service/app/chat_turn_metacog_gate.py`: a failed judge fires `reflection_failed` and does not pass the verdict upstream.
- `services/orion-equilibrium-service/tests/test_chat_turn_metacog_gate.py`: 3 tests, including evidence_map reading it as "finalize degraded", not "alignment misaligned".
- `orion/harness/evals/judge_obedience_replay_eval.py` and its test: the replay eval.

## Schema / bus / API changes

- Added: the constant `REFLECTION_SOURCE_JUDGE_FAILED`; the upstream value `finalize_degraded_reason="reflection_failed"`; the repair reason `reflection_failed`.
- Removed: none
- Renamed: none
- Behavior changed: 5b verdicts no longer weigh obedience to the imperative, and a tool-call draft is not a reply. A failed judge reaches metacog as degraded, with no verdict.
- Compatibility notes: no payload shape change. Response repair still runs on a failed judge, as before.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no change needed
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no env change
- skipped keys requiring operator action: none

## Tests run

```text
orion/harness/tests orion/cognition/prompts/tests orion/metacog + eval test -> 975+ passed (705 harness/prompt subset re-run after review fixes)
services/orion-equilibrium-service: pytest tests -q                         -> 256 passed
```

## Evals run

```text
python -m orion.harness.evals.judge_obedience_replay_eval --per-bucket 20   (40 picked, 33 loaded, 66 calls)
re-judge of the 9 marked-down obedience turns with the final prompt          (9 calls)
Results in "Outcome moved".
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy. After pull: the 5b prompt is read per call (no rebuild of the template); the
equilibrium service rebuilds. Check: within a day, chat-turn metacog triggers show reflection_failed=true
where a judge failed, and no alignment_verdict=misaligned with reflection_source=degraded_llm_failure_fallback.
```

## Review findings fixed

- Finding (medium): a failed judge still reached the response-repair reason as "misaligned".
  - Fix: `response_repair_reason_for` returns `reflection_failed`.
  - Evidence: `test_a_failed_judge_repairs_with_reason_reflection_failed_not_misaligned`.
- Finding (medium): replayed receipts carried the grammar atom's "Harness step N: tool=X, " prefix.
  - Fix: the prefix is stripped, and the docstring says tool results are clipped at 500 characters for both prompts.
  - Evidence: `test_receipt_is_rebuilt_from_a_grammar_atom`.
- Finding (low): the control bucket could include quick-gate or unknown-source reflections.
  - Fix: the eval picks only `substrate_informed_pass`.
- Finding (low): the "do not rule misaligned for no world contact when tools ran" clause had been dropped.
  - Fix: restored.
- Finding (low): stale cross-references in the WORLD-CONTACT OPPORTUNITY section.
  - Fix: reworded to match the new rules.
- Finding (low): after this merges, `--old-ref origin/main` equals the new prompt.
  - Fix: the docstring says to pass the pre-merge sha.
- Self-found in the replay: a tool-call draft was judged aligned.
  - Fix: the tool-call-is-not-a-reply rule, plus a contract test. Re-judged above.

## Restart required

```text
Post-pull rebuild handles it (orion-equilibrium-service; services importing orion/harness or the schema).
```

## Risks / concerns

- Severity: medium
  - Concern: the turn-outcome molecule, the Hub thought-process badge, and the hourly self-signal outcome rollup still show a failed judge as "misaligned" (the molecule carries no `reflection_source`).
  - Mitigation: documented at the constant. It needs its own patch across Hub and the rollup.
- Severity: medium
  - Concern: the root cause of most judge failures is still open. The 5b model continues the reply's task and writes `what_i_learned` or `summary` instead of `FinalizeReflectionV1`. That accounts for 65 of the 111 failures.
  - Mitigation: the next patch.
- Severity: low
  - Concern: the replay has 14 + 19 turns, with thinking off.
  - Mitigation: the decision rests on two things: no control flips, and imperative citations falling from 16 to 3. Both are large effects at this size.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3097

🤖 Generated with [Claude Code](https://claude.com/claude-code)
