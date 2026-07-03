# orion-social-memory

Relational continuity synthesizer for social-room turns (`orion:chat:social:stored`).

## Database migration (required on existing installs)

`Base.metadata.create_all()` does **not** add columns to existing tables. After pulling hub-social-room-ops-v1, run:

```bash
psql "$DATABASE_URL" -f services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql
```

Use the same DSN as `DATABASE_URL` in this service's `.env` (default database: `conjourney`).

## Smoke checks

```bash
curl -fsS 'http://localhost:8765/health'
curl -fsS 'http://localhost:8765/summary?platform=hub&room_id=hub-direct&participant_id=juniper'
curl -fsS 'http://localhost:8765/inspection?platform=hub&room_id=hub-direct&participant_id=juniper'
```

Restart `orion-social-memory` after migration if it was crash-looping on schema errors.
