# orion-social-memory

Relational continuity synthesizer for social-room turns (`orion:chat:social:stored`).

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of the service's own bus connection.

## Database migration (required on existing installs)

`Base.metadata.create_all()` does **not** add columns to existing tables. After pulling hub-social-room-ops-v1, run:

```bash
psql "$DATABASE_URL" -f services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql
```

Use the same DSN as `DATABASE_URL` in this service's `.env` (default database: `conjourney`).

## Ingest turn

`POST /ingest-turn` publishes a `SocialRoomTurnV1` onto `orion:chat:social:turn`
(`social.turn.v1`). It does **not** call `process_social_turn` directly.

Requires `Authorization: Bearer $SOCIAL_MEMORY_INGEST_TOKEN`. An empty
`SOCIAL_MEMORY_INGEST_TOKEN` is fail-closed (401).

## Town continuity

`GET /town-continuity` reads raw `social_room_turns` rows and returns speaker-grounded
pair + town utterances (`TownContinuityReadV1`). Required query params: `platform`,
`room_id`, `thread_id`, `speaker_id`. Blank `thread_id` or `speaker_id` is 400
`thread_id_and_speaker_id_required`. Blank platform/room is 400
`platform_and_room_id_required`. Missing `social_room_turns` returns empty lists.

Embodiment `GET /summary` is unchanged.

## Smoke checks

```bash
curl -fsS 'http://localhost:8765/health'
curl -fsS 'http://localhost:8765/summary?platform=hub&room_id=hub-direct&participant_id=juniper'
curl -fsS 'http://localhost:8765/town-continuity?platform=aitown&room_id=aitown-town&thread_id=juniper-feld--nico-sable&speaker_id=nico-sable'
curl -fsS 'http://localhost:8765/inspection?platform=hub&room_id=hub-direct&participant_id=juniper'
curl -fsS -X POST 'http://localhost:8765/ingest-turn' \
  -H "Authorization: Bearer $SOCIAL_MEMORY_INGEST_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"the urn is dying","response":"I will pull a spare from the back"}'
```

Restart `orion-social-memory` after migration if it was crash-looping on schema errors.
