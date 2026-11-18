## Summary

- Spawned `claude` turns now pass `--disable-slash-commands`. The CLI stops listing its own coding skills in every turn.
- Flag `HARNESS_FCC_DISABLE_SKILLS` (default true) turns it off.
- Tests in `orion/harness/tests/test_task_anchor.py`: default on, warm pool excluded, off switch (4 values).

## Outcome moved

Orion's turn context carried a 6,645-character skill listing: 8 context-mode skills plus 12 built-in Claude Code skills (`claude-api`, `keybindings-help`, `code-review`, `loop`, `dataviz` ...). Orion has no skills of its own. 0 Skill calls in 5,056 governor transcripts. The lane window is 64k tokens, and a self-inquiry turn already holds about 85k characters of fixed overhead before its first tool call (see the 2026-10-15 killed run: resume refused with `transcript_exceeds_lane_window`, 63 compactions in 48 h).

## Current architecture

`build_claude_argv` (`orion/harness/fcc_motor.py`) builds argv for spawn and warm-pool turns. It had no skills switch. A per-skill deny rule (`permissions.deny: Skill(name)`) was tested live and does NOT change the listing the model receives (5,981 chars both runs). The only CLI switch is `--disable-slash-commands`.

## Architecture touched

`orion/harness/fcc_motor.py` (one flag in the shared argv builder, spawn path only). Governor `.env_example` and compose.

## Files changed

- `orion/harness/fcc_motor.py`: add `--disable-slash-commands` unless `HARNESS_FCC_DISABLE_SKILLS` is false.
- `services/orion-harness-governor/.env_example`, `docker-compose.yml`: new key.
- `orion/harness/tests/test_task_anchor.py`: default on, warm pool keeps slash commands, off switch.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: `HARNESS_FCC_DISABLE_SKILLS=true` (governor).
- `.env_example` updated: yes. Local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes (verified key present). Env parity: PASS (94 services).
- Skipped keys: none from this change.

## Tests run

```text
orion/harness/tests: 562 passed (main baseline 556)
services/orion-harness-governor: 85 passed
```

## Evals run

No eval harness covers argv. Live proof instead, in the governor container, same prompt twice:

```text
no flag:                 skills=26, skill_listing=6645 chars, context-mode tools=11, mcp connected
--disable-slash-commands skills=0,  skill_listing=absent,    context-mode tools=11, mcp connected, hook context 4902 unchanged
```

## Docker/build/smoke checks

Not rebuilt in the worktree. Deploy from clean primary main after merge. Post-deploy: new transcripts in the `harness-claude-config` volume must have no `skill_listing` attachment.

## Review findings fixed

- Finding (blocker): `--disable-slash-commands` on the warm pool breaks its `/clear` handshake. With the flag the CLI treats `/clear` as plain text and sends no `conversation_reset`; `_clear` would time out on every reset and slots could bleed context between turns.
  - Fix: the flag is added only when `prompt is not None` (spawn path). Warm pool (`prompt=None`) keeps slash commands and the listing.
  - Evidence: reproduced live, same stream-json `/clear` input. No flag: `conversation_reset`, `init`, `result`. With flag: `init`, synthetic `assistant`, `result`, no reset. Regression test `test_warm_pool_argv_keeps_slash_commands_so_clear_still_resets`.
- Finding (minor): off-switch test passed even with the flag removed. Fix: it now asserts the flag is present by default, then absent for `false/0/no/off`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Any future skill (user or plugin) is also hidden while the flag is on. Mitigation: flag off, or ship Orion skills another way. Orion's real instructions already live in tool descriptions and the kickoff write section.
- Not covered: warm-pool chat processes still carry the listing (needs `/clear`).
- Not done: the kickoff (about 52 KB) and the 15.6 KB auto-memory file are the larger weights. Auto-memory is Orion's own notes and needs Juniper's decision.

## PR link

(see PR)
