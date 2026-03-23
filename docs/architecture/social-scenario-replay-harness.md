# Social scenario replay harness

This harness provides a compact, repeatable way to replay realistic social-room scenarios through the existing social-room seams without standing up a new runtime.

## Fixture model

The scenario pack lives at `tests/fixtures/social_room/scenario_replay.json` and validates against three typed models:

- `SocialScenarioFixtureV1`
  - identifies the platform, room, participant, seeded state, transcript turn fixtures, and expectations for a scenario
- `SocialScenarioExpectationV1`
  - defines bounded assertions for routing, context selection, deliberation, repair, floor behavior, epistemic framing, prompt content, and safety visibility
- `SocialScenarioEvaluationResultV1`
  - records pass/fail, mismatch reasons, the seams exercised, observed outcomes, and explicit safety observations

Transcript fixtures can mix:

- `social_turn`
  - replayed through the real social-memory service so summary, deliberation, floor, freshness, and context-window logic are exercised from stored-turn inputs
- `bridge_message`
  - replayed through the real social-room bridge policy/service path so routing, repair, epistemic signaling, and Hub payload assembly are exercised from inbound room messages

## Replay path

For each fixture, the harness runs these real seams in order:

1. seed optional room / participant / stance / style / ritual state into an in-memory social-memory database
2. replay any `social_turn` fixtures through `SocialMemoryService.process_social_turn(...)`
3. fetch the real social-memory summary and inspection snapshot
4. replay the live `bridge_message` through `SocialRoomBridgeService.process_callsyne_message(...)` with fake transport clients so the real policy output is still produced
5. feed the captured Hub payload into `build_chat_request(...)`
6. render `orion/cognition/prompts/chat_social_room.j2` with the real metadata

This keeps the harness close to the production decision path while remaining fast enough for targeted regression checks.

## What it catches

The starter pack focuses on regressions such as:

- wrong-thread or wrong-audience routing
- bridge summaries firing when a plain peer reply is better
- clarifying questions not appearing when ambiguity genuinely requires one
- stale consensus, snapshots, or ritual/style hints outranking fresher disagreement or commitments
- repair and epistemic signals failing to propagate into Hub metadata and prompt grounding
- pending artifact dialogue incorrectly becoming active continuity
- private / blocked material leaking into summary, inspection, or prompt-facing evaluation surfaces
- leave-open / handoff / closure preferences drifting back toward over-management

## Developer entry pattern

Run the replay pack with pytest:

```bash
pytest -q tests/test_social_room_scenario_replay.py
```

To add a new regression case, append a fixture to `tests/fixtures/social_room/scenario_replay.json` and define only the bounded expectations that matter for that scenario.
