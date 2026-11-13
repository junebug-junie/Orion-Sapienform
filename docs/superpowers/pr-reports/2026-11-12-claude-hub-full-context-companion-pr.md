## Summary

This is the first of two PRs for Patch 1a of the Claude-in-Hub design. It changes only the side that runs Claude. It must merge and deploy before the Hub PR (feat/claude-hub-full-context).

- Claude's prompt goes to `claude -p` over stdin, not on the command line. Linux refuses one argument over 131,072 bytes. The whole-conversation prompt is about 165,000 characters.
- The Hub room's Claude session starts fresh once a day, at the first invite after 04:00 Denver time. Consult rooms do not rotate.
- A resumed session gets only lines it has not seen yet. The companion stores a key for each line it sent (`<turn ref>|<speaker>`).
- The request schema gets new optional fields. A payload that leaves them unset stays byte-identical to the old one, so a companion that has not been rebuilt still accepts it.

## Outcome moved

- A first-turn prompt can now be as large as the whole conversation. Before, a prompt over 131,072 bytes would fail with E2BIG.
- Before, a resumed session got the same 12 turns again on every invite. Now it gets only new lines.
- Claude's per-turn context, and so its cost, is bounded to one day.

## Current architecture

The companion ran `claude -p <prompt>` with the prompt in argv. It kept one Claude session per room for its whole life. On `--resume` it sent every transcript line Hub included, so Claude got duplicates.

## Architecture touched

- `orion-room-companion`: the argv and stdin, the session store, prompt rendering, and turn assembly.
- `orion/schemas/room_claude.py`: new optional fields (consumer first).
- No bus channel changes and no env changes.

## Files changed

- `orion/schemas/room_claude.py`: `RoomTranscriptEntryV1.turn_ref`, `.unprompted`, `.side_exchange`; `RoomClaudeRequestV1.hub_index`, `.transcript_note`. Unset values are left out of the serialized payload.
- `services/orion-room-companion/app/claude_session.py`: the prompt goes in over stdin (`input=`). It is not in argv.
- `services/orion-room-companion/app/session_store.py`: the room day (04:00 Denver), `SessionSlot`, sent-line keys (at most 5,000), the time mark kept for lines with no turn ref, and legacy entries that rotate once.
- `services/orion-room-companion/app/room_prompt.py`: line labels (`Speaker [turn:xxxxxxxx] (Thu Nov 12 09:16): text`), the hub index and the "Earlier" note on a session's first turn only, and `lines_after_mark`.
- `services/orion-room-companion/app/main.py`: wires the slot and the filter, and logs the prompt size.
- `services/orion-room-companion/README.md`: session continuity, stdin, and what Claude receives.
- `services/orion-room-companion/tests/`: `test_full_context.py` is new. The argv tests are updated.
- `.github/workflows/room-claude-context-tests.yml`: runs the companion tests. Before this, no CI ran them.

## Schema / bus / API changes

- Added: the five optional fields above.
- Removed / renamed: none.
- Behavior changed: none for a producer that does not set the fields.
- Compatibility notes: the schema is `extra="forbid"`. This PR goes first. Hub starts setting the fields in the next PR.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed.
- Skipped keys: none.

## Tests run

```text
services/orion-room-companion: pytest tests -> 83 passed
  (also 83 passed in a clean venv built from requirements.txt only, which is what CI installs)
services/orion-curiosity-peer: pytest tests -> 79 passed (another producer of RoomClaudeRequestV1)
services/orion-hub: test_room_claude_relay.py test_social_room.py test_hub_direct_social_room_isolation.py -> 50 passed
```

## Evals run

```text
Live stdin check (one real call, before any code was written):
  a 150,101-byte prompt was piped to `claude -p` in the running orion-room-companion
  container (CLI 2.1.239, claude-haiku-4-5, --tools "", --no-session-persistence).
  Result: "tangerine-42" (the marker on the prompt's last line). Cost $0.113.
Real-subprocess test: a 200,000-character prompt reaches a fake `claude` whole (test_full_context.py).
```

## Docker/build/smoke checks

```text
Not built in this PR. Live behavior after deploy: UNVERIFIED (see Risks).
```

## Review findings fixed

The review ran in a subagent against both PRs together.

- Finding: deploying the schema and Hub together makes an old companion reject every request. The rebuild planner orders Hub first.
  - Fix: split into two PRs, companion first. Unset new fields are left out of the payload, so `HUB_ROOM_CLAUDE_CONTEXT_MODE=recent` is a real rollback.
  - Evidence: `test_unset_new_fields_are_left_out_of_the_payload`.
- Finding: a time mark loses Orion's reply. That happens when Juniper invites Claude while Orion is still answering, because the reply is merged into a row whose `created_at` is already at or before the mark.
  - Fix: dedupe by per-line key.
  - Evidence: `test_orions_reply_merged_into_a_half_written_row_is_still_sent`.
- Finding: all Hub sessions share one Claude session, so session B's older lines were hidden behind session A's mark.
  - Fix: the same keys.
  - Evidence: `test_a_second_hub_session_does_not_hide_its_older_lines`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-room-companion up -d --build
```

Run it from the clean primary `main` checkout after merge. The post-pull rebuild does this.

## Risks / concerns

- Severity: low. Concern: the first invite after deploy starts a new Claude session, because old entries carry no day. Mitigation: this is intended. The session reseeds from the conversation Hub sends.
- Severity: low. Concern: a resumed turn with the real CLI and stdin is UNVERIFIED live. Only a fresh `-p` call over stdin was run. Mitigation: `--resume` takes the prompt from the same `-p` input path. The first live resumed turn will show it in the companion log (`room_companion_prompt ... resume=True`).

## PR link

See the PR this file was opened with.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
