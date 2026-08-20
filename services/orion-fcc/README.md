# orion-fcc

Anthropic-compatible **FCC proxy** (`free-claude-code` `fcc-server`) as a managed Orion service. Hub Agent Claude and Orion harness motor both point `ANTHROPIC_BASE_URL` at this service on host port **8082**.

## One secrets file — not two

| File | Purpose |
|------|---------|
| **`~/.fcc/.env`** | **Single operator contract** — `MODEL_*`, `GITHUB_PAT`, `FIRECRAWL_API_KEY`, `AITOWN_*`, `ANTHROPIC_AUTH_TOKEN`, etc. Template: `config/fcc.env_example` |
| **`services/orion-fcc/.env`** | **Thin service surface only** — published port, config mount path, container networking override for `LLAMACPP_BASE_URL`. **No secrets.** |

`fcc-server` loads `~/.fcc/.env` from the mounted volume. Compose `environment:` vars (e.g. `FCC_LLAMACPP_BASE_URL` → `LLAMACPP_BASE_URL`) override file values **only inside the container** so bridge-network routing to `llm-gateway` (orion-llm-gateway's compose service key) works without editing your secrets file.

The operator model selection keys now default to the `harness` lane rather than `chat` (as of 2026-08-20). `harness` is its own entry in the gateway's route table, split off `chat` for the same reason `agent` was split off `chat` on 2026-08-14: `chat` carries live Hub chat traffic with zero admission throttling and a single-slot worker, so a long-running FCC turn sharing that route could stall a real chat reply, or vice versa. As shipped, `harness` is an interim alias of the same worker `chat` uses -- a labeling/observability seam, not yet physical isolation. See the gateway README's route-table docs before assuming this buys latency isolation today.

## Topology

```text
claude CLI (Hub / harness)  →  orion-fcc :8082  →  orion-llm-gateway :8210/v1  →  Atlas llama.cpp
```

Consumers:

| Consumer | URL |
|----------|-----|
| Hub (host network) | `HUB_FCC_SERVER_URL=http://127.0.0.1:8082` |
| Harness governor | `HARNESS_FCC_SERVER_URL=http://host.docker.internal:8082` |

## Run

```bash
# 1. Operator secrets (once)
mkdir -p ~/.fcc
cp config/fcc.env_example ~/.fcc/.env
# edit ~/.fcc/.env — MODEL_*, tokens, AITOWN_*, etc.

# 2. Service operator surface
cp services/orion-fcc/.env_example services/orion-fcc/.env

# 3. Start (after orion-llm-gateway is up)
docker compose \
  --env-file services/orion-fcc/.env \
  -f services/orion-fcc/docker-compose.yml \
  up -d --build

curl -fsS http://127.0.0.1:8082/health
```

## Env keys (service `.env` only)

| Key | Default | Meaning |
|-----|---------|---------|
| `FCC_PORT` | `8082` | Host-published port |
| `FCC_CONFIG_DIR` | `${HOME}/.fcc` | Mount source for operator FCC config |
| `FCC_LLAMACPP_BASE_URL` | `http://llm-gateway:8210/v1` | Container upstream override |
| `FCC_OPEN_BROWSER` | `false` | Suppress admin UI browser open |
| `FCC_LOG_FILE` | `/tmp/fcc-server.log` | Server log path (writable; config mount stays ro) |

## Health

- `GET http://127.0.0.1:8082/health`
- Admin UI (local): `http://127.0.0.1:8082/admin`
