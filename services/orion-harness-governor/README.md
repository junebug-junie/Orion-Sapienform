# orion-harness-governor

Bus worker for unified Hub turns. Listens on `orion:harness:run:request`, runs fcc motor + three-beat finalize (5a/5b/5c), replies with `HarnessRunV1`, and publishes audit artifacts.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_HARNESS_RUN_REQUEST` | `orion:harness:run:request` | RPC intake from Hub |
| `CHANNEL_HARNESS_RESULT_PREFIX` | `orion:harness:run:result:` | Reply channel prefix |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Audit publish after each run |
| `CHANNEL_FINALIZE_APPRAISAL_REQUEST` | `orion:substrate:finalize_appraisal:request` | 5a draft molecule RPC |
| `CHANNEL_POST_TURN_CLOSURE` | `orion:substrate:post_turn_closure` | Step 7 learning closure |

## Flow

```text
LISTEN orion:harness:run:request
  → validate HarnessRunRequestV1 + thought disposition
  → HarnessRunner.run() — fcc motor + grammar receipts + draft_text
  → run_harness_finalize_chain() — 5a substrate / 5b reflect / 5c voice / 6b outcome
  → REPLY HarnessRunV1
  → PUBLISH orion:harness:run:artifact
  → emit_post_turn_closure (step 7)
```

## Local checks

```bash
PYTHONPATH=services/orion-harness-governor:. ./orion_dev/bin/python -m pytest services/orion-harness-governor/tests/ -v
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/harness/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml config
```

## Health

`GET http://localhost:7156/health`
