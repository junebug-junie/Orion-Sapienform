# PR: Fix Hub Mind modal live verification

**Branch:** `fix/mind-hub-modal-live-verification`

## Summary

The Mind modal was empty because Hub read APIs required an exact `session_id` match while the browser’s active `orion_sid` often differed from the session stored on `mind_runs` for the same `correlation_id`. Rows existed in Postgres; `/api/mind/runs?correlation_id=…` returned `[]` when sessions diverged.

This change keeps `/api/mind/runs/recent` strictly session-scoped, relaxes **correlation-specific** lookups with safe fallbacks (`session_match`, `session_id IS NULL`, optional `context_session_id` from the message turn), improves modal empty/error diagnostics, and adds orch publish logging for `session_id`.

## Root cause

| Layer | Finding |
|-------|---------|
| **DB** | `mind_runs` has rows; no NULL `session_id` in recent data. Example: `correlation_id=e7256fd6…` stored under `session_id=eb6d05c4…` while browser used `723d819d…`. |
| **API** | `WHERE correlation_id = $1 AND session_id = $2` returned `[]` on session mismatch. |
| **UI** | Modal showed generic empty state; no correlation/session diagnostics. |

## Changes

| File | Change |
|------|--------|
| `services/orion-hub/scripts/mind_routes.py` | Correlation list/detail: allow current session, NULL session, or `context_session_id`; add `session_visibility`; recent empty `diagnostics.hint`. |
| `services/orion-hub/static/js/app.js` | Stamp `sessionId` on message meta; pass `context_session_id` from turn; richer modal empty/error copy; recent tab shows diagnostics hint. |
| `services/orion-cortex-orch/app/mind_runtime.py` | Log `mind_run_artifact_publish` with `mind_run_id`, `correlation_id`, `session_id`, etc. |
| `services/orion-hub/tests/test_mind_routes.py` | Session match, NULL fallback, context session, recent diagnostics. |
| `services/orion-hub/tests/test_mind_hub_tab.py` | Static asserts for new JS helpers. |
| `services/orion-cortex-orch/tests/test_mind_orch.py` | Artifact publish includes request `session_id`. |

## Tests

```bash
PYTHONPATH=. ./scripts/test_service.sh orion-hub \
  services/orion-hub/tests/test_mind_routes.py \
  services/orion-hub/tests/test_mind_hub_tab.py -q
# 12 passed

PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-orch/tests/test_mind_orch.py::test_publish_mind_run_artifact_includes_request_session_id -q
# 1 passed
```

## Live verification

### Phase 1 — DB (Postgres `orion-athena-sql-db`, user `postgres`, db `conjourney`)

- `mind_runs`: 97+ rows; latest `e7256fd6…` / `session_id=eb6d05c4…`
- Session distribution: multiple distinct sessions; dominant `723d819d…` (89 rows)

### Phase 2 — API (Hub `http://127.0.0.1:8080`)

| Request | Result |
|---------|--------|
| `GET /api/mind/runs?correlation_id=e7256fd6…` + header `723d819d…` | `[]` (0 rows) |
| Same + `context_session_id=eb6d05c4…` | **1 row**, `session_visibility=session_other` |
| `GET /api/mind/runs/recent` + new session | `items: []`, `diagnostics.hint` present |

### Phase 6 — New Mind-enabled turn

- `POST /api/chat` with `mind_enabled: true` → `correlation_id=4a070376…`, `session_id=723d819d…`
- **No `mind_runs` row** for that correlation: orch log `mind_http_failed … Temporary failure in name resolution` (Mind service DNS from recreated orch container). Infra blocker for end-to-end new-run proof, not a Hub modal read-path bug.

### Hub redeploy

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d hub-app
# HTTP 200 on /
```

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| Correlation API returns rows for modal when message session provided | **Pass** (live curl) |
| Modal can show real run details (existing row) | **Pass** (API returns full `result_jsonb`) |
| Recent tab session-scoped | **Pass** (SQL unchanged) |
| New turn creates `mind_runs` with non-null `session_id` | **Blocked** (Mind HTTP DNS failure on orch during test) |
| Tests pass | **Pass** |

## Follow-up: Orch → Mind service discovery (same branch)

### Root cause

`ORION_MIND_BASE_URL=http://orion-athena-mind:6611` in orch, but Mind container is `orion-mind` (`PROJECT=orion` in `services/orion-mind/.env`). After orch recreate, DNS for `orion-athena-mind` failed (`Temporary failure in name resolution`).

### Fix

| Change | Purpose |
|--------|---------|
| `services/orion-cortex-orch/.env_example` → `http://orion-mind:6611` | Match live container hostname |
| `services/orion-cortex-orch/.env` (local, gitignored) | Same |
| `services/orion-mind/docker-compose.yml` | `app-net` aliases `orion-athena-mind`, `orion-mind` |
| `mind_runtime.log_mind_http_failure` | Logs `mind_base_url`, `session_id`, `trigger`, `exc_type`, `status_code` |

### Live proof (2026-05-17)

| Step | Evidence |
|------|----------|
| Orch → Mind health | `curl -sf http://orion-mind:6611/health` OK; `getent hosts orion-athena-mind` resolves after alias |
| Mind-enabled chat | `correlation_id=d6521171-242d-4b6d-9c8e-f746744d48ac` |
| Orch publish | `mind_run_artifact_publish mind_run_id=ff35c6a9-… session_id=723d819d… ok=True` |
| DB | 1 row, non-null `session_id`, `ok=t` |
| API | `GET /api/mind/runs?correlation_id=…` → 1 row, `session_visibility=session_match` |
| Recent tab | New run appears in `/api/mind/runs/recent` for session `723d819d…` |

## Remaining limitations

1. **Mind tab “open run”** does not pass `turnMeta`; fine while recent list is session-scoped only.
2. **`session_id IS NULL` fallback** widens correlation lookup for legacy rows; prod has no NULL sessions today—consider removing after backfill.
3. **PROJECT mismatch**: orch stack uses `PROJECT=orion-athena`; Mind may run with `PROJECT=orion` — keep `ORION_MIND_BASE_URL` aligned with actual Mind `container_name` or use compose aliases.

## Test plan (manual)

1. Hard refresh Hub; confirm `orion_sid` in localStorage.
2. Open a message with Mind button from a prior session; modal should load runs when message `sessionId` differs from active session.
3. Open Mind tab; confirm recent runs for current session only; empty state shows diagnostics hint.
4. Send Mind-enabled chat; verify `mind_runs` row and modal after orch/Mind DNS is healthy.
