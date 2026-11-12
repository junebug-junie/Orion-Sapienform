## Summary

- Design only, no code: Claude in Hub gets the whole conversation (to a 160,000-character cap) and read-only tools over every Hub surface, each behind Orion's own permission.
- Orion holds one standing permission per category (allow / ask / deny), all starting at "ask". An ask goes to a short consent call in Prime's process; Orion answers yes or no with a reason.
- Every ask, answer and lookup becomes an audit event, a footer line under Claude's reply in chat, and two Hub endpoints.
- Three build-changing facts found: a 160,000-character prompt cannot go on the command line (131,072-byte per-argument limit), resumed Claude turns already get duplicate transcripts, and raw Hub JSON is far too big (cockpit 2.5 MB, body economy 412 KB).
- Recommends two small first patches: 1a whole conversation, 1b the consent rail plus `prime_now` and `body_economy`.

## Outcome moved

None at runtime. This is a design artifact that awaits Juniper's answers to 10 open questions.

## Current architecture

`services/orion-room-companion` runs `claude -p` with `--tools ""`; Hub seeds 12 turns (48 h max) from `chat_history_log`; `room_prompt.py` says Claude gets "nothing from Orion's interior".

## Architecture touched

Docs only: `docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`.

## Files changed

- `docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`: the design.
- `docs/superpowers/pr-reports/2026-11-12-claude-hub-tools-orion-consent-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed in the spec only).
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: proposed `RoomTranscriptEntryV1` fields are a consumer-first migration (model is `extra="forbid"`).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
None: docs only. Live data for the sizes and costs came from read-only queries against conjourney (chat_history_log, orion_consult_turn, hub_attention_minute), read-only Hub GETs on localhost:8080, and read-only dig commands in orion-harness-governor.
```

## Evals run

```text
None: docs only.
```

## Docker/build/smoke checks

```text
None: docs only.
```

## Review findings fixed

- No code review: design-only PR.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: consent latency on circe chat lane 0 is unmeasured.
- Mitigation: Patch 1b acceptance check 8 records p50/p90 over the first 20 asks.

## PR link

(see GitHub)
