## Summary

- AI Town NPCs now leave overlong chats mechanically (`conversation.leave`) instead of asking the LLM for a goodbye that can hang forever.
- Orion's embodiment walks out of dead participating chats after `EMBODIMENT_CONVERSATION_ABANDON_SEC` (default 180s): stale last line, or never spoke (using transcript evidence so a restart cannot false-positive).
- Live 2026-09-14 Mara↔Orion `c:186863` was stuck nine days; operator leave cleared it. This patch prevents the same deadlock class.

## Outcome moved

Stuck AI Town conversations that never end when leave-via-LLM fails, and Orion bodies that wait forever after speaking last.

## Current architecture

NPC duration/message-cap leave used `agentGenerateMessage` type=`leave`. Embodiment turn-taking waited for a partner reply with no abandon timer.

## Architecture touched

- `services/orion-ai-town` patch stack + README
- `orion/embodiment` dead-chat helper, perception `created_ms`, `leave_conversation` client
- `services/orion-embodiment` worker/settings/env/compose

## Files changed

- `services/orion-ai-town/patches/orion-mechanical-leave.patch`: mechanical leave
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: register patch last
- `orion/embodiment/dead_chat.py`: abandon decision
- `services/orion-embodiment/app/worker.py`: wire abandon + leave
- tests, READMEs, `.env_example`, compose, settings

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: town leave path; embodiment may send `leaveConversation`
- Compatibility notes: requires AI Town rebuild/redeploy to apply patch; embodiment restart to pick up abandon

## Env/config changes

- Added keys: `EMBODIMENT_CONVERSATION_ABANDON_SEC` (default `180`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: none for this change

## Tests run

```text
PYTHONPATH=services/orion-embodiment:. pytest \
  services/orion-ai-town/tests/test_mechanical_leave_patch.py \
  services/orion-ai-town/tests/test_npc_answer_first_patch.py \
  orion/embodiment/tests/test_dead_chat_abandon.py \
  orion/embodiment/tests/test_perception.py \
  orion/embodiment/tests/test_aitown_client.py \
  services/orion-embodiment/tests/test_worker_dead_chat_abandon.py -q
# 55 passed
```

## Evals run

```text
No new eval harness for this seam; gate tests cover the contract.
```

## Docker/build/smoke checks

```text
Not redeployed in this session. After merge: rebuild orion-ai-town (apply patches) and restart orion-embodiment.
```

## Review findings fixed

- Finding: corrupt patch hunk line counts
  - Fix: corrected `@@ -207,19 +207,12 @@`; added hunk-count gate test
  - Evidence: `test_mechanical_leave_patch_hunk_line_counts_match` passes
- Finding: restart false-positive on never-spoke counter
  - Fix: transcript `author_id == own` counts as already spoke
  - Evidence: `test_no_abandon_when_transcript_shows_orion_spoke_but_counter_zero`
- Finding: abandon during in-flight speech
  - Fix: skip while `cid in _speaking_conversations`
  - Evidence: `test_engage_skips_abandon_while_speaking`

## Restart required

```bash
# Circe — AI Town (applies mechanical-leave patch at build)
cd /mnt/scripts/Orion-Sapienform/services/orion-ai-town
# from a worktree via scripts/safe_docker_build.sh orion-ai-town ...
docker compose --env-file ../../.env --env-file .env -f docker-compose.yml up -d --build

# Athena — embodiment
cd /mnt/scripts/Orion-Sapienform
scripts/safe_docker_build.sh orion-embodiment up -d --build
```

## Risks / concerns

- Severity: should
- Concern: abandon only runs when `EMBODIMENT_SOCIAL_ENABLED=true` (same gate as engage)
- Mitigation: live Athena already has social enabled; document in README

## PR link

(filled after open)
