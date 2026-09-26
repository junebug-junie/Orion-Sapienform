# orion-fcc

Anthropic-compatible **FCC proxy** (`free-claude-code` `fcc-server`) as a managed Orion service. Hub Agent Claude and Orion harness motor both point `ANTHROPIC_BASE_URL` at this service on host port **8082**.

## One secrets file — not two

| File | Purpose |
|------|---------|
| **`~/.fcc/.env`** | **Single operator contract** — `MODEL_*`, `GITHUB_PAT`, `FIRECRAWL_API_KEY`, `AITOWN_*`, `ANTHROPIC_AUTH_TOKEN`, etc. Template: `config/fcc.env_example` |
| **`services/orion-fcc/.env`** | **Thin service surface only** — published port, config mount path, container networking override for `LLAMACPP_BASE_URL`. **No secrets.** |

`fcc-server` loads `~/.fcc/.env` from the mounted volume. Compose `environment:` vars (e.g. `FCC_LLAMACPP_BASE_URL` → `LLAMACPP_BASE_URL`) override file values **only inside the container** so bridge-network routing to `llm-gateway` (orion-llm-gateway's compose service key) works without editing your secrets file.

As of 2026-08-20, the operator model selection keys are MEANT TO move to a new `harness`
lane rather than `chat` -- but as of this note, `config/fcc.env_example`'s `MODEL`/
`MODEL_SONNET`/`MODEL_OPUS` still literally say `chat`, and any live `~/.fcc/.env` needs the
same by-hand edit (`llamacpp/harness` instead of `llamacpp/chat`; leave `MODEL_HAIKU` alone if
it already points elsewhere). This is not yet done: the `.env`/`.env_example` change is
config-only and was applied separately from the code in this patch. `harness` is its own entry
in the gateway's route table, split off `chat` for the same reason `agent` was split off `chat`
on 2026-08-14: `chat` carries live Hub chat traffic with zero admission throttling and a
single-slot worker, so a long-running FCC turn sharing that route could stall a real chat
reply, or vice versa. Even once applied, `harness` is shipped as an interim alias of the same
worker `chat` uses -- a labeling/observability seam, not yet physical isolation. See the
gateway README's route-table docs before assuming this buys latency isolation.

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

## Messages transport compatibility

The pinned FCC 2.4.4 Messages route always returned SSE, including requests
with `stream: false`. Claude's WebFetch fallback rejects that HTTP 200 as a
malformed non-streaming response. Silent upstream waits also exceeded Claude's
stream-idle watchdog before source summarization could finish.

The image applies `install_transport_patch.py` to the audited upstream route
and error emitter. SHA-256 guards fail the build if either source changes; an FCC upgrade must
review/remove the patch explicitly. The adapter leaves provider routing,
generation settings, recovery, and deadlines unchanged:

- `stream: true`: immediate and 15-second idle SSE `ping` events, with original
  provider events preserved. Pings are transport liveness, not reading progress.
- False, null, or omitted `stream`: assemble a complete Anthropic JSON message,
  preserving content blocks, tool JSON, citations, stop metadata, and usage.
  Incomplete/error streams return HTTP 502, never a partial successful message.
- Disconnect: cancel the pending read and finish async provider cleanup under
  a cancellation shield, including buffered non-streaming requests.
- Provider failures: emit real SSE errors, not error prose in successful
  assistant messages. Non-streaming assembly preserves the error payload in a
  502 response. Existing local optimization responses are unchanged.

There are no new env keys, bus events, or schema registry entries. Upstream
`config.settings.Settings` still owns FCC settings; this wrapper has no local
`settings.py` or application bus consumer.

### Checks

Build in a worktree, using a separate Compose project to avoid replacing the
deployment image tag:

```bash
scripts/safe_docker_build.sh orion-fcc -p orion-fcc-transport-test build
docker run --rm --entrypoint python \
  -v "$PWD/services/orion-fcc/tests:/tests:ro" \
  orion-fcc-transport-test-fcc -m unittest discover -s /tests -v
```

`.github/workflows/orion-fcc-tests.yml` also installs the pinned upstream,
applies the guard, and exercises its real patched route. The route regression
fails against the original unpatched image.

Opt-in live eval (uses model capacity and public web access; no queue/memory
writes), after starting an isolated candidate on `app-net`:

```bash
scripts/safe_docker_build.sh orion-fcc -p orion-fcc-transport-test \
  run -d --no-deps --name orion-fcc-transport-canary fcc
docker exec -i orion-athena-harness-governor python3 - \
  --base-url http://orion-fcc-transport-canary:8082 \
  < services/orion-fcc/evals/webfetch_smoke.py
docker stop orion-fcc-transport-canary
docker rm orion-fcc-transport-canary
```

The eval permits exactly one WebFetch of arXiv 2310.19279, checks the source
title in both the tool receipt and final reply, and enforces a 900-second total subprocess deadline.
This proves the tool path, not Stage 2 journal landing. Use the reading verifier
after an explicitly approved queue retry to check end-to-end completion.
