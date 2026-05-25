# PR: Retire legacy topic rail; wire notify digest to Topic Foundry

**Branch:** `feat/retire-topic-rail`  
**Base:** `main`  
**Worktree:** `.worktrees/feat-retire-topic-rail`

## Summary

Fully decouples and removes legacy `orion-topic-rail` and Landing Pad topic proxy routes. **Topic Foundry** is now the only active topic/drift HTTP surface for notify digest (and Hub Topic Studio, unchanged).

- Deleted `services/orion-topic-rail/` (entire service)
- Removed Landing Pad `GET /api/topics/summary` and `GET /api/topics/drift` plus `topic_rail` reducer
- Notify digest calls Topic Foundry directly via `TOPIC_FOUNDRY_URL`
- Removed dead Hub JS that still referenced legacy endpoints (DOM was already gone)
- Retired bus channels/schemas: `orion:topic:summary.v1`, `orion:topic:shift.v1`, `orion:topic:rail:assigned.v1`, `TopicSummaryEventV1`, `TopicShiftEventV1`, `TopicRailAssignedV1`

## Replacement Topic Foundry endpoints

| Digest need | Endpoint | Notes |
|-------------|----------|-------|
| Latest completed run | `GET /runs?format=wrapped&status=complete&limit=1` (+ optional `model_name`) | Resolves `run_id` for topics |
| Top topics | `GET /topics?run_id={uuid}&limit={TOPICS_MAX_TOPICS}` | Maps `items[].label` / `topic_id`, `count` → `TopicItem` |
| Drift highlights | `GET /drift?model_name={name}&limit={TOPICS_DRIFT_MAX_RECORDS}` | Maps `records[].js_divergence` → `DriftItem.score`; filtered by `TOPICS_WINDOW_MINUTES` on `window_end` |

Hub proxy (unchanged): `{HUB}/api/topic-foundry/...` when `TOPIC_FOUNDRY_URL` points at Hub.

## Legacy references removed

- Service tree: `services/orion-topic-rail/**`
- Landing Pad: `/api/topics/summary`, `/api/topics/drift`, `app/reducers/topic_rail.py`, `tests/test_topic_rail_queries.py`
- Schemas: `orion/schemas/topic.py`; registry entries; three `orion:topic:*` rail channels in `orion/bus/channels.yaml`
- Hub: ~249 lines of orphaned Topic Rail JS (`fetchTopicRailData`, legacy endpoint fetches)
- README tables/flow text referencing `orion-topic-rail` / “topic rail”
- Notify digest `LANDING_PAD_URL` as topics source (replaced by `TOPIC_FOUNDRY_URL`; deprecated env logs a warning if still set)

## Files changed (high level)

| Area | Files |
|------|-------|
| Notify digest | `app/digest.py`, `app/main.py`, `app/settings.py`, `docker-compose.yml`, `.env_example`, `README.md`, `tests/*` |
| Landing Pad | `app/web/api.py` (routes removed) |
| Hub | `static/js/app.js`, `app/settings.py`, `README.md` |
| Bus/schemas | `orion/bus/channels.yaml`, `orion/schemas/registry.py` |
| Docs/smokes | `README.md`, `scripts/smoke_no_topic_rail.sh`, `scripts/smoke_topic_digest.sh`, `scripts/smoke_topic_drift_alert.sh` |
| SQL comment only | `services/orion-sql-db/manual_migration_chat_topic.sql` |

**Local sync (not committed):** `services/orion-notify-digest/.env` updated on host to match `.env_example` (`TOPIC_FOUNDRY_URL`, etc.).

## Intentionally retained

- **DB migrations / manual SQL** for `chat_topic`, `chat_topic_summary`, `chat_topic_session_drift` in `services/orion-sql-db/` — historical tables may still exist in deployed Postgres; dropping them is out of scope. Comments updated where they tripped the smoke grep.
- **Topic Foundry** behavior unchanged (no router changes).

## Env / compose changes

**Notify digest** (`.env_example` / `docker-compose.yml`):

```bash
TOPIC_FOUNDRY_URL=http://orion-topic-foundry:8615
TOPIC_FOUNDRY_MODEL_NAME=
TOPICS_DRIFT_MAX_RECORDS=50   # replaces TOPICS_DRIFT_MAX_SESSIONS
# LANDING_PAD_URL — deprecated; startup warns if set without TOPIC_FOUNDRY_URL
```

## Tests and smokes

| Command | Result |
|---------|--------|
| `bash scripts/smoke_no_topic_rail.sh` | **PASS** |
| `pytest services/orion-notify-digest/tests -q` | **4 passed** |
| `pytest services/orion-landing-pad/tests -q` | **2 passed** |
| `pytest services/orion-topic-foundry/tests -q` | **1 failed** (pre-existing: `test_chat_corpus_builder_stages` — unrelated to this PR) |

## Operational notes

- **Drift semantics:** Alerts use Topic Foundry **JS divergence**, not legacy session `switch_rate` / `entropy`. Retune `DRIFT_ALERT_THRESHOLD` after deploy.
- **Topic labels in digest:** Foundry `/topics` often returns `label=null`; digest falls back to `Topic {id}` unless enrichment labels are added server-side later.
- **Run selection:** Digest picks latest completed run by Foundry list order (`created_at DESC`); may differ from drift daemon’s `completed_at` ordering.

## Test plan

- [ ] Set `TOPIC_FOUNDRY_URL` on notify-digest (direct or Hub proxy)
- [ ] Confirm daily digest includes Top Topics / Drift Highlights (or graceful unavailable text)
- [ ] With `DRIFT_ALERTS_ENABLED=true`, confirm alerts fire only for recent drift records within `TOPICS_WINDOW_MINUTES`
- [ ] Hub Topic Studio still loads via `/api/topic-foundry/*`
- [ ] `bash scripts/smoke_no_topic_rail.sh` in CI
