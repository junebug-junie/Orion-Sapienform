## Summary

- Juniper's chat turns (warm pool) now start without the CLI's 15 built-in coding skills in context.
- Same switch as #2706 (`HARNESS_FCC_DISABLE_SKILLS`, default true). Warm pool gets `CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1` in its env, because #2706's `--disable-slash-commands` breaks the pool's `/clear` reset.
- Flag check factored into `_skills_disabled()` in `orion/harness/fcc_motor.py`, used by both paths.

## Outcome moved

Chat turns on the 35B chat lane (`circe:8011`, `n_ctx 65536`) compact every 15-25 tool calls. Measured on turn `a9128e7a` (2026-10-15 23:46Z): first request already 23,983 tokens; compaction fires at ~32.5k; after it, the turn restarts near 24k. Working room is about 8k tokens.

#2706 dropped the skill listing for spawned turns only. Chat replies take the warm path (`fcc_warm_path_decision decision=try_warm chat_reply=True`), so they still carried it. This patch removes ~1.1k tokens (built-in skills) from that floor: about 13% more working room per compaction cycle.

Not removed: context-mode's 8 plugin skills (~940 tokens). `skillOverrides` (`off`, `user-invocable-only`, `name-only`) via `--settings` did not change their listing (`/context` probe, 939 tokens every variant).

## Current architecture

`_build_subprocess_env` builds env for both spawned and warm-pool `claude` processes. `build_claude_argv` added `--disable-slash-commands` only when `prompt is not None`.

## Architecture touched

`orion/harness/fcc_motor.py` env builder. No schema, bus or compose change.

## Files changed

- `orion/harness/fcc_motor.py`: `_skills_disabled()`; set `CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1` in subprocess env when on.
- `orion/harness/tests/test_task_anchor.py`: default on for chat env; off switch (4 values).
- `services/orion-harness-governor/.env_example`: comment now describes both paths. No key change.

## Schema / bus / API changes

None.

## Env/config changes

- Added/removed/renamed keys: none. `HARNESS_FCC_DISABLE_SKILLS` meaning extended (comment updated).
- local `.env` sync: not needed (no key change). Env parity: PASS (94 services).

## Tests run

```text
orion/harness/tests: 568 passed
services/orion-harness-governor/tests: 85 passed
mutation: env line removed -> 5 new tests fail
```

## Evals run

No eval harness covers subprocess env. Live proof in the governor container (CLI 2.1.301), stream-json `/clear` as the warm pool sends it:

```text
off: ... conversation_reset ... init(skills=26,slash=56) success
on:  ... conversation_reset ... init(skills=11,slash=41) success
/context Skills row: 2k -> 939 tokens
```

## Docker/build/smoke checks

Not rebuilt in the worktree. Post-deploy check: new chat transcripts' `skill_listing` attachment has no `claude-api`/`dataviz`/`keybindings-help`; warm pool logs no `clear_failed`.

## Review findings fixed

- Finding (minor): with the flag off, an inherited `CLAUDE_CODE_DISABLE_BUNDLED_SKILLS` in the governor env would still reach the subprocess, so the off switch would not be exact.
  - Fix: `env.pop(...)` when off.
  - Evidence: `test_off_switch_wins_over_an_inherited_bundled_skill_drop`.
- Reviewer checked every `_build_subprocess_env` caller (spawn, warm pool `_spawn`; hub bridge has its own builder), `warm_signature` respawn on flag change (hashes `os.environ`, includes the flag), and env leakage (only a nested `claude` inside a turn would inherit; none exists). No other findings.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Warm slots respawn with `signature_mismatch` only if the governor env changes; a rebuild restarts the container, so all slots respawn anyway.

## Risks / concerns

- Severity: low. Bigger lever untouched: the CLI reserves 33k of the 64k window (20k output + 13k margin) regardless of `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` (probed 70/92/100: same 33k). `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` would shrink it to 21k, but 577 of 18,088 steps since 10-01 produced >8,192 output tokens (mostly thinking), so that needs its own decision.
- Remaining fixed floor: kickoff prompt (~26 KB on chat), MCP instructions (~10 KB), deferred tool list (~11 KB), context-mode hook text, Orion's AutoMem (~16 KB, proposal mode).
