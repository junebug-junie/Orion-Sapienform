# Orion Topic Rail

## Run modes

### Daemon (default)
```bash
python services/orion-topic-rail/app/main.py
```

### Train-only
```bash
TOPIC_RAIL_MODE=train python services/orion-topic-rail/app/main.py
```

### Backfill
```bash
TOPIC_RAIL_MODE=backfill python services/orion-topic-rail/app/main.py
```

## Summaries and drift (optional)

Enable summary/drift outputs (disabled by default):
```bash
TOPIC_RAIL_SUMMARY_ENABLED=true TOPIC_RAIL_DRIFT_ENABLED=true \
python services/orion-topic-rail/app/main.py
```

Enable bus publishing of summary/shift events:
```bash
TOPIC_RAIL_BUS_PUBLISH_ENABLED=true \
python services/orion-topic-rail/app/main.py
```

## Lifecycle + health

Refit policy (default: never):
```bash
TOPIC_RAIL_REFIT_POLICY=ttl TOPIC_RAIL_ALLOW_REFIT_IN_DAEMON=true \
python services/orion-topic-rail/app/main.py
```

Enable health endpoint:
```bash
TOPIC_RAIL_HTTP_ENABLED=true TOPIC_RAIL_HTTP_PORT=8610 \
python services/orion-topic-rail/app/main.py
```

## Smoke test
```bash
python scripts/smoke_topic_rail_chat.py
```

### Notes
- `TOPIC_RAIL_RUN_ONCE=true` runs a single iteration then exits.
- `TOPIC_RAIL_FORCE_REFIT=true` retrains even if artifacts exist.
