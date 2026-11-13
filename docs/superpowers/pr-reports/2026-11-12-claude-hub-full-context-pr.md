## Summary

This is part 2 of 2 for Patch 1a of the Claude-in-Hub design (`docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`). Part 1 is #3249 (the companion). It must merge and deploy first.

- When Juniper invites Claude, Hub now sends the whole conversation, not 12 turns. It goes back to 160,000 characters of text, newest rows kept first, and it never cuts a row in half. One line says how many older turns were left out.
- Every line has its real time and a short turn ref. Orion's unprompted messages are labelled. Claude's own earlier side exchanges are included and marked.
- On a session's first turn, Claude also gets a short index of what is open in Hub: the tab Juniper is on, open consults, open asks, and how many concerns Prime holds. It has names and counts only, never content.
- Open question 11: Hub now stores Juniper's `@claude` prompt with Claude's reply. When Orion's chat reconnects, Orion sees the newest 3 side exchanges as one labelled block that says Orion was not asked to reply. Each side is trimmed to 300 characters.
- orion-sql-writer never turns a side exchange with Claude into memory.
- The spec now records Juniper's "yes" to all ten open questions, and adds question 11 with her answer.

## Outcome moved

- Live today, for Juniper's main session, Claude's first turn would get 89 turns (111 lines, 159,106 characters) back to 11-07, plus "[Earlier: 368 turns from 2026-08-20 to 2026-11-07 not shown.]". Before, it got at most 12 turns from the last 48 hours, with every time stamped "now", and with none of Orion's unprompted messages. Those are 353 of the session's 457 rows.
- Orion can now read Juniper's side conversations with Claude, in one compact block that is never addressed to Orion.

## Current architecture

`/api/room/claude/invite` called `_recent_room_transcript`: 12 turns, 48 h, through `chat_history_rehydrate`. That reader drops unprompted and Claude rows and passes no row times. Juniper's `@claude` text was never stored. Claude's reply was stored as a response-only `room_claude` row. Orion's rehydrate excluded that row, so Orion never saw any side exchange.

## Architecture touched

- orion-hub: the invite route, a new context builder, the relay, rehydrate, and the continuity builder.
- orion-sql-writer: one guard in `_memory_turn_initiator`.
- The schema fields come from #3249.
- No bus channels change. There are 4 new Hub env keys.

## Files changed

- `services/orion-hub/scripts/room_claude_context.py` (new): the window walk, the 12,000-character row rule, the "Earlier" note, the hub index, and the Prime count. The Prime count follows the stale-means-absent rule.
- `services/orion-hub/scripts/api_routes.py`: the invite route uses the full window by default. `HUB_ROOM_CLAUDE_CONTEXT_MODE=recent` is the rollback.
- `services/orion-hub/scripts/room_claude_relay.py`: stores the prompt and `invited_by` in `client_meta.room_claude`, not in the `prompt` column.
- `services/orion-hub/scripts/chat_history_rehydrate.py`: the side-exchange fetch, the block render, and time-ordered insertion. The block is never added at index 0, and never without a restored conversation.
- `services/orion-hub/scripts/cortex_request_builder.py`: drops the block from every lane except the unified turn.
- `services/orion-hub/scripts/websocket_handler.py`: passes the two side-exchange settings to rehydrate.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: 4 new keys.
- `services/orion-hub/tests/test_room_claude_full_context.py` (new, 30 tests).
- `services/orion-hub/evals/room_claude_context_live_eval.py` (new): a read-only live eval.
- `services/orion-sql-writer/app/worker.py`, `tests/test_room_claude_never_memory.py`: a `room_claude` row is never memory.
- `.github/workflows/room-claude-context-tests.yml`: adds the Hub tests.
- `docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`: status, the answers, Q11, and the build notes.

## Schema / bus / API changes

- Added: Hub fills `turn_ref`, `unprompted`, `side_exchange`, `hub_index`, `transcript_note`, and the real `created_at` (fields from #3249). `client_meta.room_claude.prompt` and `.invited_by` are added to stored Claude rows. The invite response gains `transcript_chars` and `transcript_note`.
- Removed / renamed: none.
- Behavior changed: Claude's transcript (above). Orion's rehydrated history may hold one `system` block marked `kind: "side_exchange"`.
- Compatibility notes: needs #3249's companion live first, because the request is `extra="forbid"`. Old Claude rows without a stored prompt render the reply only.

## Env/config changes

- Added keys (orion-hub): `HUB_ROOM_CLAUDE_CONTEXT_MODE=full`, `HUB_ROOM_CLAUDE_CONTEXT_MAX_CHARS=160000`, `HUB_HISTORY_SIDE_EXCHANGE_MAX=3`, `HUB_HISTORY_SIDE_EXCHANGE_TRIM_CHARS=300`.
- Removed / renamed: none.
- `.env_example` updated: yes. `settings.py` and the compose fallbacks match (`check_settings_defaults.py orion-hub --example-drift`: OK).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-hub`: yes, all 4 keys were added.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion-hub, every test file touching room_claude / rehydrate / websocket_handler / continuity + test_route_engine_reuse.py
  -> 711 passed
orion-hub test_room_claude_full_context.py test_room_claude_relay.py test_chat_history_rehydrate.py
  -> passed in a clean venv with the CI workflow's install list
orion-room-companion tests -> 83 passed
orion-sql-writer tests (not integration) -> 1233 passed, 158 skipped
tests/test_no_personal_terms.py tests/test_check_settings_defaults.py -> 31 passed
check_env_key_single_source.py -> OK; check_service_env_compose_parity.py orion-room-companion -> OK
```

## Evals run

```text
POSTGRES_URI=... python services/orion-hub/evals/room_claude_context_live_eval.py 9c91d646-...   (read-only)
  lines 111, turns 89, text_chars 159,106, request 179,348 bytes
  transcript_note "[Earlier: 368 turns from 2026-08-20 to 2026-11-07 not shown.]"
  hub_index: 4 open consults (id, status, turns), 5 open asks
  checks within_cap / nonempty / oldest_first / every_line_has_turn_ref / hub_index_within_800: all true
Prime count against the live bus Redis (read-only): "- Prime holds 2 concern(s)"
Q11 block against live rows (read-only): 0 side exchanges in the last 48 h. With a 30-day window: 1 exchange,
  rendered as a 379-character block.
```

## Docker/build/smoke checks

```text
No image built. No deploy (Juniper merges; the post-pull rebuild deploys).
```

## Review findings fixed

The code review ran in a subagent against `origin/main...feat/claude-hub-full-context`.

- Finding (blocker): the schema and Hub in one deploy break an old companion. The rebuild planner orders Hub first.
  - Fix: split into #3249 (companion first) and this PR. Unset fields are left out of the payload, so `recent` mode is a real rollback.
  - Evidence: `test_unset_new_fields_are_left_out_of_the_payload`.
- Finding: the time mark loses Orion's reply when Claude is invited mid-turn, because the reply is merged into a row that already has an older `created_at`.
  - Fix: dedupe by per-line key in the companion (#3249).
  - Evidence: `test_orions_reply_merged_into_a_half_written_row_is_still_sent`.
- Finding: two Hub sessions share one mark.
  - Fix: the same keys.
  - Evidence: `test_a_second_hub_session_does_not_hide_its_older_lines`.
- Finding: the side block was a mid-list system message on the legacy chat lanes.
  - Fix: tag it `kind=side_exchange`, and drop it unless `keep_timestamps` (the unified turn) is set.
  - Evidence: `test_side_block_reaches_only_the_unified_turn`.
- Finding: the "Earlier" count included one shown row when the fetch limit ended the walk.
  - Fix: `<` in that case, `<=` when the cap stopped it.
  - Evidence: `test_earlier_count_...`, `test_fetch_conversation_against_a_fake_connection`.
- Nits fixed:
  - `NULLS LAST` on the window query.
  - The Prime count is stale-means-absent (`test_prime_count_is_shown_fresh_and_hidden_stale`).
  - Trim 0 means no trim.
  - No duplicate block on a second hello.
  - CI paths cover `requirements.txt` and the continuity builder.
  - Comments now give the real prompt size: about 165,000 characters including labels.
- CI finding (static gates, schema-skew discovery): Hub now reads `PrimeStateV1`, and its writer was not visible to the check.
  - Fix: declare orion-mind (`app/prime_loop.py`, Redis `orion:prime:state`) in `orion/schema_skew_discovery.py` `DECLARED_WRITERS`.
  - Evidence: `tests/scripts/test_schema_skew_discovery.py` 18 passed. CI green (27 checks).
- Not changed:
  - Hub still sends the whole window on resumed invites, about 180 KB per bus message. The companion discards what was already sent. This is acceptable at current invite rates.

## Restart required

Deploy order. The schema is `extra="forbid"`, so order matters:

```bash
# 1. After #3249 merges (companion first):
scripts/safe_docker_build.sh orion-room-companion up -d --build
# 2. After this PR merges:
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Run each from the clean primary `main` checkout. The post-pull rebuild runs them. sql-writer can go in any order. Its change only adds a guard.

## Risks / concerns

- Severity: medium. Concern: a live invite with the new window is UNVERIFIED. No real Claude turn ran on the new code. The only real call was the stdin size check. Mitigation: the companion log line `room_companion_prompt ... prompt_chars=` and the invite response's `transcript_chars` show it on the first live invite. The spec's spend projection is $0.20 to $0.30 for a full first turn of the day.
- Severity: low. Concern: a side exchange that happens while Orion's tab stays open reaches Orion only after the next reconnect. The live socket history does not append Claude's replies. Mitigation: this matches "not too much chatter". Appending live would be its own small patch, if wanted.
- Severity: low. Concern: Orion seeing the side block is UNVERIFIED live. There have been no side exchanges in the last 48 h. Mitigation: unit tests cover render, placement and lane filtering. The first `@claude` after deploy, followed by a reconnect, will show it as a "System" line in Orion's RECENT CONVERSATION.

## PR link

See the PR this file was opened with. Part 1: https://github.com/junebug-junie/Orion-Sapienform/pull/3249

🤖 Generated with [Claude Code](https://claude.com/claude-code)
