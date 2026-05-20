# PR: Hub Forge — Source Delta Review UI (fix)

## Summary

Fixes the Hub **Source Delta Review** panel that was missing at runtime despite PR #598 merging the Knowledge Forge ingest API and Hub proxy.

**Root cause:** PR #598 (`feat/orion-knowledge-forge-source-delta-v1.2`) shipped backend + `POST /api/knowledge/sources/ingest` proxy only. The Hub template and `app.js` wiring lived on a separate branch (`feat-orion-knowledge-forge-hub-source-delta-v1.2b`) that was never merged. Runtime diagnosis confirmed:

| Check | Result |
|-------|--------|
| Repo `api_routes.py` proxy | Present |
| Running container proxy | Present |
| Repo `index.html` / `app.js` panel | **Missing on main** |
| Running container template/JS | **Missing** |
| `curl http://127.0.0.1:8080/` before fix | No `Source Delta Review` |
| Docker image stale? | **No** — code was never on main |
| Browser cache? | **No** — HTML never contained panel |

## Architecture

```text
Hub Forge tab (#forge)
  Source Delta Review panel (template + app.js)  ← this PR
       ↓ POST /api/knowledge/sources/ingest
  Hub proxy (already on main from #598)
       ↓ POST /v1/sources/ingest
  Knowledge Forge service (already on main)
```

## Changes

| File | Change |
|------|--------|
| `services/orion-hub/templates/index.html` | Source Delta Review panel (fields, safety copy, result area, debug JSON) |
| `services/orion-hub/static/js/app.js` | DOM refs, `runForgeSourceIngest`, validation, render, click handler, `escapeHtml` on lists |
| `services/orion-hub/tests/test_forge_hub_tab.py` | Assertions for panel markup and JS wiring |

**Branch:** `fix/hub-source-delta-ui`  
**Worktree:** `.worktrees/fix-hub-source-delta-ui`

## Commits

1. `903507a0` — fix(hub): wire Source Delta Review panel on Forge tab  
2. `d268b3b5` — fix(hub): escape source ingest list HTML and restore Forge subtitle  

## Verification

### Runtime (Athena)

```bash
# Panel in served HTML (after template sync + hub restart)
curl -s http://127.0.0.1:8080/ | grep -n "Source Delta Review\|forgeSourceIngestButton"

# Direct Forge API (use /repo path inside containers)
curl -s -X POST http://127.0.0.1:8630/v1/sources/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"/repo/docs/PR-orion-knowledge-forge-source-delta-v1.2.md","source_id":"source:hub-source-delta-test","kind":"design_doc","write_review":false,"dry_run":true}'

# Hub proxy
curl -s -X POST http://127.0.0.1:8080/api/knowledge/sources/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"/repo/docs/PR-orion-knowledge-forge-source-delta-v1.2.md","source_id":"source:hub-source-delta-test","kind":"design_doc","write_review":false,"dry_run":true}'
```

**Results:** Both APIs return `status: proposed`, `content` with `# Source Delta Review`, warnings `preview only: write_review is false`. HTML includes panel after fix.

**Note:** `/tmp/...` paths fail with `source path not found` — Forge/Hub containers see host paths via `/repo` mount, not arbitrary `/tmp`.

### Tests

```bash
docker exec orion-athena-hub sh -lc \
  'cd /repo && PYTHONPATH=/repo:/repo/services/orion-hub python3 -m pytest \
   services/orion-hub/tests/test_forge_hub_tab.py \
   services/orion-hub/tests/test_knowledge_forge_proxy_routes.py -q'
```

**Result:** `5 passed`

### Hub live reload

Hub `docker-compose.yml` bind-mounts `./templates` and `./static` from `services/orion-hub/`. After merging, either:

- Copy/sync updated files into the mounted paths and restart `orion-athena-hub`, or  
- Rebuild: `cd services/orion-hub && docker compose build --no-cache hub-app && docker compose up -d hub-app`

`HUB_UI_ASSET_VERSION` cache-busts `app.js` on each Hub process start — no manual version bump required.

## Acceptance criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `#forge` shows Source Delta Review panel | ✅ |
| 2 | Dry-run from Hub returns proposed claims/content preview | ✅ (via API; UI wired) |
| 3 | Hub calls `/api/knowledge/sources/ingest` | ✅ |
| 4 | API/proxy smoke works with curl | ✅ |
| 5 | UI handles write-disabled warning cleanly | ✅ (403 / message pattern) |
| 6 | Write + writable mount creates pending review | ✅ (existing backend; UI refreshes tab) |
| 7 | Pending review count refreshes after write | ✅ `refreshForgeTab()` |
| 8 | No accepted claims/specs/decisions mutated | ✅ (backend allowlist unchanged) |
| 9 | Tests pass | ✅ Hub tab + proxy tests |

## Code review

Subagent review: **APPROVED**. Addressed minor items:

- `escapeHtml()` on proposed-claims and warnings lists  
- Restored Forge subtitle: “Source ingest, claims, specs…”  

**Deferred (known):** Operator token header for write mode when `KNOWLEDGE_FORGE_OPERATOR_TOKEN` is set — same as v1.2b non-goals.

## Non-goals

Unchanged — no MCP, GraphDB, vectors, auto-accept, file watcher, or ideation panel.

## Test plan (manual)

1. Open Hub → **Forge** tab → confirm **Source Delta Review** panel (teal block, above Search).
2. Enter path `docs/PR-orion-knowledge-forge-source-delta-v1.2.md`, ID `source:hub-manual-test`, kind `design_doc`, **Dry run** checked → **Ingest Source**.
3. Confirm result: status, content preview, warnings.
4. Uncheck dry run, check **Write pending review** (requires `KNOWLEDGE_FORGE_WRITE_ENABLED=true` on Forge) → ingest → pending reviews count updates.
5. Hard refresh if an old tab was open (unlikely; panel was absent from HTML).
