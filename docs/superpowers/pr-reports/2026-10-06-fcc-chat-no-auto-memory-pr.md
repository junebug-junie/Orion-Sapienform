## Summary

- Hub chat replies no longer load Claude Code's auto-memory: the notes folder that curiosity, urgent, self-inquiry and mutation runs write into the shared sandbox checkout. Investigation runs keep it, and nothing is deleted.
- Hub now tells the governor who started each turn (`HarnessRunRequestV1.utterance_origin`: `juniper` / `orion` / none).
- The governor spawns `claude` with `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` only for `juniper` turns. This is gated by `HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY`, which ships on.
- Implements L7 of `docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md` (approved 2026-10-06).

## Outcome moved

This removes one of the two known sources of "watching the prediction errors dance on the bus" talk in Hub replies: the ungoverned side channel from investigation notes into chat.

**Baseline (BEFORE, read-only, 2026-10-06, last 7 days).** "Phrase hits" means a `final_text` matching `prediction[ _-]?error|bus[ _-]?synaptic`:

| scope | turns | turns with phrase | phrase count |
|---|---|---|---|
| Hub chat replies (`chat_history_log.source='hub_orion'`) | 19 | 2 | 2 |
| Outreach messages in chat (empty source, `unsolicited`) | 28 | 7 | 7 |
| All `harness_turn_trace` turns | 665 | 105 | 190 |

The query is in `services/orion-harness-governor/README.md` ("Chat replies do not read Claude Code auto-memory"). Re-run it about 7 days after deploy and compare against the `hub_orion` row.

## Current architecture

Every FCC turn runs `claude -p` with `cwd=/mnt/orion-fcc/repo`. Claude Code keys auto-memory by working directory (`/root/.claude/projects/-mnt-orion-fcc-repo/memory/`), so every turn read and wrote one shared memory, and nothing gates or reviews it. `--setting-sources` does not control it. The CLI (2.1.288, checked read-only inside `orion-athena-harness-governor`) reads `CLAUDE_CODE_DISABLE_AUTO_MEMORY`; a truthy value turns memory off before the `autoMemoryEnabled` setting is consulted. The env var was chosen because it is per-process and needs no per-turn settings file.

## Architecture touched

orion-hub (request producer), orion-harness-governor (runner + FCC motor), and the shared schema `HarnessRunRequestV1`.

## Files changed

- `orion/schemas/harness_finalize.py`: optional `utterance_origin` field.
- `orion/hub/turn_orchestrator.py`: copies `utterance_origin` onto the harness request.
- `orion/harness/runner.py`: adds `is_chat_reply_request`; passes `chat_reply=True` only for `juniper` turns, and `default_fcc_runner` forwards it.
- `orion/harness/fcc_motor.py`: `run_fcc_turn(chat_reply=...)`, `_build_subprocess_env` sets or pops the env var, adds `chat_auto_memory_disabled()`.
- `services/orion-harness-governor/{.env_example,app/settings.py,docker-compose.yml,README.md}`: the new flag (default true) and the proof query.
- `orion/harness/tests/test_fcc_chat_auto_memory.py`, `services/orion-hub/tests/test_turn_orchestrator_utterance_origin.py`: tests.

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.utterance_origin: str | None = None`.
- Removed / Renamed: none.
- Behavior changed: the governor's claude env for chat-reply turns.
- Compatibility notes: the model uses pydantic's default `extra="ignore"`. A new Hub talking to an old governor has the field dropped. An old Hub talking to a new governor sends None, which means "not a chat reply" (today's behavior). Both deploy orders are safe. No channel changes.

## Env/config changes

- Added keys: `HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY=true` (orion-harness-governor).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The primary checkout's `services/orion-harness-governor/.env` now has `HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY=true`.
- Skipped keys requiring operator action: none for this key. The sync reported a pre-existing divergence, `HARNESS_FCC_INTROSPECT_ENABLED` (local true, example false), which was left untouched.

## Tests run

```text
pytest orion/harness/tests/test_fcc_chat_auto_memory.py          15 passed
pytest orion/harness/tests                                        438 passed
pytest services/orion-harness-governor/tests                      64 passed
pytest services/orion-hub/tests/test_turn_orchestrator_utterance_origin.py \
       services/orion-hub/tests/test_turn_orchestrator_ws_frames.py   60 passed, 1 failed (pre-existing on main)
scripts/check_env_template_parity.py                              PASS
scripts/check_env_key_single_source.py                            OK
git diff --check                                                  clean
```

The failure, `test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness` (the role-teach work-shape preamble gets spliced into `user_message`), also fails on main at f9309426b. It is unrelated to this change.

## Evals run

```text
No eval harness covers this seam. The post-deploy proof is the documented 7-day phrase-count query (README) against the baseline above.
```

## Docker/build/smoke checks

```text
Not run (not deployed, per task). Read-only container check: claude 2.1.288 reads CLAUDE_CODE_DISABLE_AUTO_MEMORY.
```

## Review findings fixed

- Finding: no committed PR report.
  - Fix: this file.
  - Evidence: `docs/superpowers/pr-reports/2026-10-06-fcc-chat-no-auto-memory-pr.md`.
- Finding: Hub "agent-claude" mode spawns claude directly from Hub (`services/orion-hub/scripts/fcc_claude_bridge.py`) and does not go through the governor.
  - Fix: left out of scope on purpose. That is Juniper driving Claude Code directly, not Orion's reply writer. Listed under risks.
  - Evidence: the review's caller enumeration.
- Finding: one Hub test fails.
  - Fix: confirmed it is pre-existing on main. Not touched.
  - Evidence: the same failure on the primary checkout at f9309426b.
- Finding: mutation runs are named as memory writers but their spawn path was not traced.
  - Fix: none needed. They reach the motor as non-`juniper` requests and keep their memory. Marked UNVERIFIED.
  - Evidence: `HarnessRunRequestV1(` has one producer and the governor has one claude spawn site.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build && ./scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

(Deploy from the primary checkout on main after merge.)

## Risks / concerns

- Severity: low. Concern: a chat reply that relied on a recent investigation finding loses it. Mitigation: investigation findings reach recall through the journal. Rollback: `HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY=false`, then restart the governor.
- Severity: low. Concern: outreach messages (7/28 phrase hits, the bigger chat-visible leak) still load auto-memory, because they are not chat replies under this spec. Mitigation: setting `utterance_origin` on the outreach caller would be a one-line follow-up; that is Juniper's call.
- Severity: low. Concern: Hub agent-claude mode is not covered (see above).
- UNVERIFIED: that a live chat turn's claude process has the variable set and its transcript shows no memory index load. That needs a post-deploy check.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
