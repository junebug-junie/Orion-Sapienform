# PR: Hub Forge — Source Delta Review panel (v1.2b)

## Summary

Adds the **Source Delta Review** panel to the Hub Forge tab so operators can run the Knowledge Forge source-ingest loop from the UI instead of only via CLI/API.

```text
Design doc changed
  → Hub: ingest source (dry-run or write pending review)
  → Forge proposes source-delta review
  → Hub shows pending review + proposed claims
  → later: accept/reject/apply (existing flows)
  → compile context pack
```

This is the **input valve** for the compiler loop: Sources → Reviews → Claims/Specs → Context Packs.

Depends on Knowledge Forge v1.2 (`feat/orion-knowledge-forge-source-delta-v1.2`): `POST /v1/sources/ingest` and Hub proxy `POST /api/knowledge/sources/ingest`.

## Changes

| File | Change |
|------|--------|
| `services/orion-hub/templates/index.html` | Source Delta Review panel (fields, safety copy, result area) |
| `services/orion-hub/static/js/app.js` | `runForgeSourceIngest`, validation, result render, refresh on write |
| `services/orion-hub/tests/test_forge_hub_tab.py` | Panel + JS wiring assertions |

No Knowledge Forge service changes. No new env vars (reuses existing `KNOWLEDGE_FORGE_BASE_URL` proxy).

## UI behavior

**Panel:** Source Delta Review (teal-bordered block, placed after operator takeaway, before search)

**Fields:**

| Field | Default |
|-------|---------|
| Source path | empty (placeholder example) |
| Source ID | empty (placeholder `source:my-design-doc`) |
| Kind | `design_doc` |
| Dry run | checked |
| Write pending review | unchecked |

**Button:** Ingest Source → `POST /api/knowledge/sources/ingest`

**Request body:**

```json
{
  "path": "...",
  "source_id": "source:...",
  "kind": "design_doc",
  "dry_run": true,
  "write_review": false
}
```

**Result:** status, source_id, source_path, review_path, proposed claims (count + list), possibly affected specs, warnings, review content preview, debug JSON under Forge debug details.

**After successful write** (`write_review=true`, `dry_run=false`): refreshes Forge status strip and pending reviews list.

**Safety copy:** “Source ingest creates pending review artifacts. It does not mutate accepted claims, execution-ready specs, or decisions.”

**Error handling:** missing path, invalid source ID (client), unreachable Forge, write disabled, API validation errors.

## Non-goals (unchanged)

- No accepted-claim editor, auto-apply, MCP, GraphDB, vector search, file watcher, or background ingestion daemon
- No operator token UI (writes with `KNOWLEDGE_FORGE_OPERATOR_TOKEN` require follow-up header wiring)

## Test results

```bash
PYTHONPATH=services/orion-hub pytest \
  services/orion-hub/tests/test_knowledge_forge_proxy_routes.py \
  services/orion-hub/tests/test_forge_hub_tab.py -q
# 5 passed
```

## Manual smoke

```bash
cat > /tmp/kf-hub-test-design.md <<'EOF'
# Hub Source Delta Test

## Requirements

- Hub should let operators ingest design docs.
- Source ingest should create pending reviews, not accepted claims.
- The result should show proposed claims and warnings.

## Non-goals

- No auto-accept.
- No vector search.
EOF
```

1. Open Hub → **Forge** tab → **Source Delta Review**
2. Source path: `/tmp/kf-hub-test-design.md`
3. Source ID: `source:hub-source-delta-test`
4. Kind: `design_doc`
5. Dry run: on, Write pending review: off → **Ingest Source** → preview claims, no corpus writes
6. Dry run: off, Write pending review: on → **Ingest Source** → pending review under corpus if `KNOWLEDGE_FORGE_WRITE_ENABLED=true`

## Panel layout (description)

```
┌─ Source Delta Review ─────────────────────────────┐
│ Source path [________]  Source ID [________]        │
│ Kind [design_doc ▼]  ☑ Dry run  ☐ Write pending   │
│ ⚠ Safety: no mutation of accepted claims/specs    │
│ [ Ingest Source ]                                   │
│ Status / paths / claims / warnings / preview        │
└─────────────────────────────────────────────────────┘
```

## Branch

- **Worktree:** `.worktrees/feat-orion-knowledge-forge-hub-source-delta-v1.2b`
- **Branch:** `feat/orion-knowledge-forge-hub-source-delta-v1.2b`
- **Base:** `feat/orion-knowledge-forge-source-delta-v1.2`

## Test plan

- [ ] Forge tab shows Source Delta Review above search
- [ ] Dry-run ingest returns proposed claims without `reviews/pending/` writes
- [ ] Write ingest creates pending review only under allowlisted paths
- [ ] Pending reviews list refreshes after write
- [ ] Accepted claims / execution-ready specs unchanged after ingest
- [ ] Existing compile/search/claims panels still work
