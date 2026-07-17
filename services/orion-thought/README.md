# orion-thought

Bus worker service for unified Hub turns. Listens on `orion:thought:request`, runs cortex `stance_react`, applies stance quality and disposition policy, replies with `ThoughtEventV1`, and publishes audit artifacts.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_THOUGHT_REQUEST` | `orion:thought:request` | RPC intake from Hub |
| `CHANNEL_THOUGHT_RESULT_PREFIX` | `orion:thought:result:` | Reply channel prefix (Hub sets `reply_to`) |
| `CHANNEL_THOUGHT_ARTIFACT` | `orion:thought:artifact` | Audit publish after each thought |
| `CHANNEL_CORTEX_EXEC_REQUEST` | `orion:cortex:exec:request` | Cortex exec plan RPC |
| `CHANNEL_CORTEX_EXEC_RESULT_PREFIX` | `orion:exec:result` | Cortex exec reply prefix |

## Flow

```text
LISTEN orion:thought:request
  → validate StanceReactRequestV1
  → run_stance_react() — cortex verb stance_react (brain tier)
  → parse_stance_react_payload + apply_stance_react_pipeline
  → REPLY ThoughtEventV1 on reply_to
  → PUBLISH orion:thought:artifact
```

## Local checks

```bash
PYTHONPATH=services/orion-thought:. ./orion_dev/bin/python -m pytest services/orion-thought/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml config
```

## Health

`GET http://localhost:7155/health`

## Reverie semantic lift

Set `ORION_REVERIE_SEMANTIC_LIFT_ENABLED=true` (default `false`) to lift coalition
`harness_closure:{corr}` pointers into human `ConcernCardV1` text from
`substrate_turn_referent` before `reverie_narrate`. Requires referent rows from
unresolved post-turn closures and routes narration through the background/metacog lane.

## Resonance health monitor (Phase H+)

`ORION_REVERIE_RESONANCE_ALERT_ENABLED` (default `true`) already runs an observation-only
tripwire (`orion.reverie.resonance.detect_resonance`) after every completed reverie chain,
persisting a `ResonanceAlertV1` to `substrate_reverie_resonance_alert` whenever a theme
re-ignites faster than its own refractory bound allows. Until now that alert reached
Postgres and a Hub debug panel only — nobody got paged.

`app/resonance_monitor.py` closes that gap the same way the field-digester /
attention-runtime / self-state-runtime health monitors already do (all three merged,
running in production): an edge-triggered check that pages via `orion-notify`'s
`POST /attention/request` (surfacing in Hub's existing Pending Attention panel) only on
a healthy→unhealthy transition, retries a failed delivery until it actually succeeds, and
checks `orion-notify`'s own pending list before suppressing a first-observation alert (so
a process restart mid-incident can't go permanently silent).

**"Unhealthy" here is not "an alert exists."** A 2026-07-12 investigation confirmed a real
historical resonance burst had already self-resolved by the time it was investigated, but
the detector kept re-reporting the same old `violation_count`/`min_gap_sec` for ~20 hours
afterward, because those old rows hadn't yet aged out of `detect_resonance`'s 200-row
lookback window (`ORION_REVERIE_RESONANCE_WINDOW`). Paging on "an alert exists" would have
paged for ~20 hours about an already-resolved problem. Instead, a theme is only considered
unhealthy when its `violation_count` strictly **increases** across its last 2 persisted
samples — i.e. the loop is actually getting worse right now, not just echoing history.

Reuses `NOTIFY_BASE_URL`/`NOTIFY_API_TOKEN` (same values as the other three services'
health monitors) — no new settings beyond those two.

## Computed salience v2 (shadow-first)

Flags (default-off): `ORION_ATTENTION_SALIENCE_V2_ENABLED`,
`ORION_ATTENTION_HABITUATION_ENABLED`, `ORION_ATTENTION_SALIENCE_WEIGHTS` (JSON override).

Shared module: `orion/substrate/attention/salience.py`. When `SALIENCE_V2` is on,
`score_loop`/`derive_salience` read the combiner salience and the reverie tick
emits `AttentionSalienceTraceV1` on `orion:attention:salience:trace` (persisted to
`attention_salience_trace`).

Migrations (apply before enabling):
`psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql`

## Mind stance enrichment (unified turn)

Set `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true` (default `false`) to run
`orion-mind` before `stance_react` and inject an advisory self/attention
`mind_coloring` block into the verb context. `stance_react` stays the sole
author of `ThoughtEventV1` and reconciles the coloring (existing inputs win —
it never forces chat framing on technical/agent turns).

Module: `app/mind_enrichment.py` (snapshot builder, fail-open HTTP client,
allow-list coloring selector, artifact publisher, bounded `drive_state_compact`
Postgres fetch).

Flags:

| Env key | Default | Role |
|---------|---------|------|
| `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` | `false` | Master switch |
| `ORION_MIND_BASE_URL` | `http://orion-mind:6611` | Mind endpoint |
| `ORION_THOUGHT_MIND_TIMEOUT_SEC` | `210` | HTTP read timeout (must exceed `WALL_MS/1000`) |
| `ORION_THOUGHT_MIND_WALL_MS` | `180000` | Mind policy wall time (must be ≥ `3 × MIND_LLM_TIMEOUT_SEC × 1000`) |
| `ORION_THOUGHT_MIND_ROUTER_PROFILE` | `default` | Mind router profile |
| `ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES` | `2000000` | Response body cap |
| `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED` | `false` | Publish `mind_runs` artifact (`mode=orion`) |
| `ORION_THOUGHT_MIND_COLORING_MAX_ITEMS` | `3` | Coloring list cap |
| `ORION_THOUGHT_MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC` | `0.4` | Bounded single-row `drive_audits` lookup timeout |

**`drive_state_compact` facet:** before building the Mind request,
`_maybe_build_mind_coloring` (`app/bus_listener.py`) fetches the latest
`drive_audits` row (`subject = 'orion'`) via `fetch_drive_state_facet_for_thought`,
bounded by `ORION_THOUGHT_MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC`. On timeout,
connection failure, missing table, no row, or a row with no meaningful content
(a quiet DriveEngine tick), it degrades to `None` and the facet is simply
omitted — this never raises or slows the turn beyond the bounded timeout.
Mirrors `orion-cortex-orch`'s `drive_state_compact` facet
(`services/orion-cortex-orch/app/mind_runtime.py`), adapted to `orion-thought`'s
sync `psycopg2`/`SQLAlchemy` DB seam (reuses `app/store.py`'s lazy engine —
`orion-thought` has no `asyncpg` dependency) instead of that service's asyncpg
pool.

**Preconditions (silent no-op if unmet):**
1. `orion-mind` must have `MIND_LLM_SYNTHESIS_ENABLED=true` — `meaningful_synthesis`
   (the only quality that fires coloring) is produced only by Mind's LLM path.
2. `orion-thought` must be able to reach `ORION_MIND_BASE_URL`.

**Budget invariant:** `orion-mind` runs 3 sequential LLM phases, each capped by
`MIND_LLM_TIMEOUT_SEC` (default 60s on the `orion-mind` service). If
`ORION_THOUGHT_MIND_WALL_MS` is below ~`3 × MIND_LLM_TIMEOUT_SEC × 1000` (180000)
synthesis is cut off mid-pipeline and the Mind always degrades to `contract_only`
(the coloring never fires) — an empty-shell no-op. The default `180000` allows
all three phases; `ORION_THOUGHT_MIND_TIMEOUT_SEC` (HTTP read) must
exceed `WALL_MS/1000` so Mind's own fail-open result is returned instead of the
client aborting. `orion-thought` logs a `mind_enrichment_config` warning at boot
if either invariant is violated while enrichment is enabled.

The min-viable wall is derived from `MIND_LLM_TIMEOUT_SEC_ASSUMED` (60s) in
`app/settings.py`. That mirrors `orion-mind`'s `MIND_LLM_TIMEOUT_SEC` (a separate
service, not readable from here) — if you change the per-phase timeout on
`orion-mind`, re-derive `MIND_ENRICHMENT_MIN_VIABLE_WALL_MS` and the wall default
here, or the boot warning will under-fire.

**Latency caveat:** enrichment runs synchronously before `stance_react`. A stuck
Mind LLM now fails open after ~3×60s of phase clamping (bounded by the 210s HTTP
read) rather than the old ~12s, i.e. up to ~180–210s added to a live user turn in
the worst case. Pair any rollout (`ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true`)
with a turn-level budget/circuit-breaker on the caller.

Everything fails open: Mind unconfigured / unreachable / slow / low-quality →
byte-identical to today's stance behavior.
