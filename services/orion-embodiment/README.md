# orion-embodiment

Mind-to-sprite bridge: gives Orion a persistent AI Town body driven by its own state. This service is the **sole Convex actuator/perceiver**. Producers only publish semantic `EmbodimentIntentV1` events; this worker arbitrates (deliberate preempts involuntary for a hold window), resolves semantic intents to `{x,y}`, actuates via Convex `sendInput(moveTo)`, and emits outcomes + perception.

**All flags default off.** With `ORION_EMBODIMENT_ENABLED=false` the worker connects nothing and actuates nothing.

## Inputs

- `orion:embodiment:intent` — `EmbodimentIntentV1` (from substrate-runtime, harness-governor, cortex-exec)

## Outputs

- `orion:embodiment:outcome` — `EmbodimentOutcomeV1`
- `orion:embodiment:perception` — `WorldPerceptionV1`

## Port

`8130` (`EMBODIMENT_PORT`)

## Bus / Redis

| Env | Default | Purpose |
|-----|---------|---------|
| `ORION_BUS_URL` | `${ORION_BUS_URL}` (pass-through from root `.env`; Tailscale node) | Redis URL for Orion bus |
| `ORION_BUS_ENABLED` | `true` | Disable bus connect/publish when `false` |
| `ORION_EMBODIMENT_ENABLED` | `false` | Master switch — worker does nothing unless `true` |
| `EMBODIMENT_CHANNEL_INTENT` | `orion:embodiment:intent` | Intent subscribe channel |
| `EMBODIMENT_CHANNEL_OUTCOME` | `orion:embodiment:outcome` | Outcome publish channel |
| `EMBODIMENT_CHANNEL_PERCEPTION` | `orion:embodiment:perception` | Perception publish channel |

Never hardcode `bus-core` / `redis://redis`. `ORION_BUS_URL` passes through from the root `.env`.

## Secrets (`~/.fcc/.env`)

`AITOWN_CONVEX_URL`, `AITOWN_ADMIN_KEY`, `AITOWN_WORLD_ID`, `AITOWN_ORION_PLAYER_ID`, `AITOWN_ORION_AGENT_ID` are loaded from `~/.fcc/.env` (mounted read-only at `/root/.fcc/.env`), **not** from this service `.env`.

## Bootstrap (create/update Orion's body)

```bash
python services/orion-embodiment/scripts/bootstrap_orion_agent.py           # dry-run
python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write    # persist ids to ~/.fcc/.env
```

The persona is a privacy-filtered projection of Orion's live self-model.

## Run

```bash
cd services/orion-embodiment
cp -n .env_example .env
docker compose --env-file ../../.env --env-file .env up -d --build
curl -fsS http://localhost:8130/health
```
