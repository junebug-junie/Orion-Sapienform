# PR: Hub Mind modal visibility + Orch→Mind service discovery

**Branch:** `fix/mind-hub-modal-live-verification`

## Summary

Two live-verified fixes on one branch:

1. **Hub Mind modal empty** — `mind_runs` rows existed but `/api/mind/runs?correlation_id=…` returned `[]` when the browser `orion_sid` did not match the row’s `session_id`. Correlation-scoped reads now allow safe fallbacks; the frontend passes `context_session_id` from the message turn and shows session/correlation diagnostics on empty/error.

2. **No new `mind_runs` after orch recreate** — `ORION_MIND_BASE_URL` pointed at `orion-athena-mind:6611` while the live Mind container is `orion-mind` (`PROJECT=orion`). Orch logged `mind_http_failed` / DNS resolution failure. URL and compose network aliases were corrected; a new Mind-enabled turn now persists and appears in Hub API/UI.

## Commits (this work)

| Commit | Description |
|--------|-------------|
| `8c58daa1` | Hub modal session visibility (routes + `app.js` + tests) |
| `62fb0ea9` | Orch `mind_run_artifact_publish` structured logging + publish test |
| `91fb854d` | Orch Mind base URL + failure diagnostics + Mind compose aliases |
| `70eb64cc` | This PR report |

## Root causes

### A. Modal / API session mismatch

| Layer | Finding |
|-------|---------|
| **DB** | Rows present; e.g. `correlation_id=e7256fd6…` under `session_id=eb6d05c4…` while browser had `723d819d…`. |
| **API** | `WHERE correlation_id = $1 AND session_id = $2` returned `[]` on mismatch. |
| **UI** | Generic “Select a run.” / “No runs yet.” with no session diagnostics. |

### B. Orch → Mind DNS

| Layer | Finding |
|-------|---------|
| **Config** | `ORION_MIND_BASE_URL=http://orion-athena-mind:6611` in orch `.env` / `.env_example`. |
| **Runtime** | Mind `container_name` = `orion-mind` (`PROJECT=orion` in `services/orion-mind/.env`). |
| **Orch logs** | `mind_http_failed … Temporary failure in name resolution` when calling Mind after container recreate. |
| **From orch** | `getent hosts orion-mind` OK; `orion-athena-mind` failed until alias added. |

## Files changed

| File | Change |
|------|--------|
| `services/orion-hub/scripts/mind_routes.py` | Correlation list/detail: current session, `NULL` session, or `context_session_id`; `session_visibility` on rows; recent empty `diagnostics.hint`. |
| `services/orion-hub/static/js/app.js` | `sessionId` on message meta; `context_session_id` query param; `mindRunsEmptyDiagnostics`; improved modal/API error copy; recent tab diagnostics hint. |
| `services/orion-hub/tests/test_mind_routes.py` | Session match, NULL fallback, context session, exclusion of other sessions, recent diagnostics. |
| `services/orion-hub/tests/test_mind_hub_tab.py` | Static asserts for new JS helpers. |
| `services/orion-cortex-orch/app/mind_runtime.py` | `mind_http_base_url`, `log_mind_http_failure`, `mind_run_artifact_publish` logging. |
| `services/orion-cortex-orch/app/orchestrator.py` | Call structured `log_mind_http_failure` on Mind HTTP errors. |
| `services/orion-cortex-orch/.env_example` | `ORION_MIND_BASE_URL=http://orion-mind:6611` + hostname comment. |
| `services/orion-cortex-orch/tests/test_mind_orch.py` | Publish `session_id` test; base URL; failure log fields; HTTP client URL test. |
| `services/orion-mind/docker-compose.yml` | `app-net` aliases: `orion-athena-mind`, `orion-mind`. |

**Operator note:** Copy `ORION_MIND_BASE_URL` from `.env_example` into local `services/orion-cortex-orch/.env` (gitignored).

## Tests

```bash
# Hub (12 passed)
PYTHONPATH=. ./scripts/test_service.sh orion-hub \
  services/orion-hub/tests/test_mind_routes.py \
  services/orion-hub/tests/test_mind_hub_tab.py -q

# Orch (16 passed)
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-orch/tests/test_mind_orch.py -q
```

## Live verification

### Hub modal / read path

**DB** (`orion-athena-sql-db`, `postgres` / `conjourney`):

- `mind_runs`: 97+ rows; multiple sessions; no NULL `session_id` in recent sample.

**API** (`http://127.0.0.1:8080`):

| Request | Result |
|---------|--------|
| `GET /api/mind/runs?correlation_id=e7256fd6…` + `X-Orion-Session-Id: 723d819d…` | `[]` |
| Same + `context_session_id=eb6d05c4…` | **1 row**, `session_visibility=session_other` |
| `GET /api/mind/runs/recent` + unknown session | `items: []`, `diagnostics.hint` present |

**Hub redeploy:**

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d hub-app
# HTTP 200
```

### Orch → Mind / write path

**DNS / health (inside `orion-athena-cortex-orch`):**

```
ORION_MIND_BASE_URL=http://orion-mind:6611
getent hosts orion-athena-mind → 172.18.0.30  (after compose alias)
curl -sf http://orion-mind:6611/health → {"ok":true,...}
```

**Mind-enabled chat:**

```bash
POST /api/chat  (mind_enabled: true, X-Orion-Session-Id: 723d819d…)
# correlation_id=d6521171-242d-4b6d-9c8e-f746744d48ac
# session_id=723d819d-e4a3-4b19-96db-62e96faf83f5
# (LLM reply may timeout independently; Mind path succeeded)
```

**Orch log:**

```
mind_run_artifact_publish mind_run_id=ff35c6a9-7eb5-4ff9-bd5b-e30fd5901529
  correlation_id=d6521171-242d-4b6d-9c8e-f746744d48ac
  session_id=723d819d-e4a3-4b19-96db-62e96faf83f5 trigger=user_turn ok=True
```

**DB:**

```sql
SELECT mind_run_id, correlation_id, session_id, ok, trigger, created_at_utc
FROM mind_runs
WHERE correlation_id = 'd6521171-242d-4b6d-9c8e-f746744d48ac';
-- 1 row: ff35c6a9-…, session_id=723d819d-…, ok=t, trigger=user_turn
```

**Hub API:**

```bash
GET /api/mind/runs?correlation_id=d6521171-…
# 1 row, session_visibility=session_match

GET /api/mind/runs/recent?hours=168&limit=5
# New run first in list for session 723d819d…
```

**Services restarted:**

```bash
# Mind (aliases)
cd services/orion-mind && docker compose --env-file .env up -d

# Orch (rebuild + env)
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
```

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| Correlation API returns rows for modal (with message session when needed) | **Pass** |
| Modal can show real run details (`result_jsonb` in list response) | **Pass** (API) |
| Recent Mind tab session-scoped | **Pass** |
| New Mind-enabled turn creates `mind_runs` with non-null `session_id` | **Pass** (`d6521171…`) |
| Orch → Mind HTTP succeeds | **Pass** (`/health`, artifact publish) |
| Hub + orch tests pass | **Pass** (12 + 16) |

## Remaining limitations

1. **Mind tab row click** does not pass `turnMeta` to the modal (OK while recent list is session-scoped).
2. **`session_id IS NULL` fallback** on correlation lookup is for legacy rows only; prod sample had no NULL sessions.
3. **`PROJECT` mismatch** between stacks (`orion-athena` vs `orion`) — keep `ORION_MIND_BASE_URL` aligned with Mind `container_name` or use compose aliases.
4. **Browser UI** not automated here; hard-refresh Hub and open Mind on `d6521171…` message to confirm modal rendering.

## Manual test plan

1. Hard refresh Hub; note `orion_sid` in localStorage.
2. Open Mind on an older message whose `sessionId` differs from active session — runs should load via `context_session_id`.
3. Send Mind-enabled chat; confirm Mind button opens modal with run id, ok, timestamps, request/result JSON.
4. Open `#mind` tab; confirm recent table includes the new run for the current session.
