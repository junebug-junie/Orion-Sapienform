# GitHub compactor — daily repo development digest

**Date:** 2026-07-08  
**Status:** Approved for implementation planning  
**Problem:** Orion has no durable, always-on sense of what has recently been built in the repo. GitHub PR intelligence exists as a read-only skill (`skills.repo.github_recent_prs.v1`) but does not fetch PR bodies, does not compact descriptions into narrative, and does not persist into memory surfaces Orion actually uses at chat time.

---

## Arsonist summary

Once per day, ingest merged PR **descriptions** from the configured GitHub repo, compact them into a bounded digest of “what we’ve been building,” and persist:

1. **One superseding memory card** — Orion’s gut sense at chat time (`high_recall` inject).
2. **One append-only journal entry** — lineage and fuel for later `concept_induction_pass` synthesis.

Implementation is a new named cognition workflow `github_compactor_pass` in cortex-orch, scheduled daily via orion-actions — same pattern as `journal_pass` and `concept_induction_pass`. No new memory service.

---

## Goals

- Orion knows, in plain language, what development work landed recently (from PR descriptions, not just titles/paths).
- Digest is **bounded** (char budgets, fail loud) and **idempotent** (same calendar day + repo → same journal entry id).
- Card supersession gives a single active “repo dev snapshot” without accumulating stale cards.
- Reuses existing GitHub env (`GITHUB_TOKEN`, `ORION_ACTIONS_GITHUB_OWNER`, `ORION_ACTIONS_GITHUB_REPO`) and `RECALL_PG_DSN` on cortex-orch.

## Non-goals (v1)

- Multi-repo support
- Open/unmerged PR triage or issue compaction
- Direct write to `orion-self-state-runtime` or production self-model surfaces (constitution forbids)
- New bus channel for concept induction (journal lineage is enough; induction reads on demand)
- Keyword/phrase taxonomies for PR themes
- Hub Memory tab UI beyond existing card/journal surfaces

---

## Current architecture

| Piece | Today |
|-------|--------|
| `skills.repo.github_recent_prs.v1` | Fetches merged PR metadata (title, paths, labels, inferred services). **No PR body.** |
| `RepoPullRequestDigestItemV1` | Has optional `short_summary` field — unused by fetch verb |
| `skills.mesh.mesh_ops_round.v1` | Can include PR digest + optional journal write; ops-oriented, not daily compactor |
| `orion-actions` scheduler | Durable workflow schedules for `journal_pass`, `concept_induction_pass`, etc. |
| Memory cards | Postgres via `orion/core/storage/memory_cards.py`; cortex-orch injects `high_recall` / `always_inject` at chat |
| Journal | Append-only `journal.entry.write.v1` on `orion:journal:write` |
| Concept induction | `orion-spark-concept-induction` consumes bus intake channels; does not auto-ingest journal in v1 |

**Gap:** No daily job turns PR descriptions → compact narrative → card + journal.

---

## Proposed architecture

```text
[Daily schedule] orion-actions claims due schedule
  → orion:actions:trigger:workflow.v1 (workflow_id=github_compactor_pass)
    → cortex-orch workflow lane (_execute_github_compactor_pass)

Step 1 — Fetch
  → skills.repo.github_recent_prs.v1 (extended: body + short_summary)
  → lookback_days from workflow policy (default 1)

Step 2 — Compact
  → bounded LLM step
  → input: structured PR list JSON
  → output: { card_summary, journal_title, journal_body, pr_refs[] }
  → char budgets enforced before persist (fail loud, no blind truncation)

Step 3 — Memory card (supersede)
  → find active card where subschema.compactor_slot = "repo_dev_snapshot"
  → if found: change_card_status → superseded
  → insert_card (new active, priority=high_recall, anchor_class=project)
  → subschema: { compactor_slot, window_start, window_end, merged_pr_count, source_repo }

Step 4 — Journal (append-only)
  → build_write_payload → journal.entry.write.v1
  → source_ref: github_compactor_pass:{date}:{repo}
  → stable entry_id per (workflow_id, date, repo)

Step 5 — Result
  → workflow metadata: pr_count, card_id, journal entry_id, compact preview
  → optional notify via schedule policy
```

---

## Schema / API changes

### Extend PR digest item

**File:** `orion/schemas/actions/mesh_ops.py`

Add to `RepoPullRequestDigestItemV1`:

- `body: Optional[str] = None` — truncated PR description (max 2000 chars at fetch)
- `short_summary` — populated by compactor LLM step (optional at fetch time)

### Memory card supersession slot

**Contract:** `subschema.compactor_slot = "repo_dev_snapshot"`

- One producer: `github_compactor_pass`
- One consumer: cortex-orch recall inject (existing `memory_inject.py` path)
- Tag mirror: `tags: ["repo_dev_snapshot"]` for Hub filtering

**Provenance:** add `"repo_compactor"` to `MemoryProvenance` literal (preferred over overloading `auto_extractor`).

### Workflow registration

**File:** `orion/cognition/workflows/registry.py`

```python
WorkflowDefinition(
    workflow_id="github_compactor_pass",
    display_name="GitHub Compactor",
    description="Daily compaction of merged PR descriptions into a repo development digest (memory card + journal).",
    aliases=[
        "run github compactor",
        "compact recent prs",
        "github compactor pass",
        "what have we been building",
    ],
    user_invocable=True,
    autonomous_invocable=True,
    persistence_policy="Supersede one active memory card (repo_dev_snapshot slot) and append one journal entry per run.",
    result_surface="Return PR count, card summary preview, and journal entry id.",
    ...
)
```

### Compactor LLM output schema (new, local to workflow)

```python
class GithubCompactorDigestV1(BaseModel):
    card_summary: str          # max ~800 chars
    journal_title: str         # max ~120 chars
    journal_body: str          # max ~4000 chars
    pr_refs: List[str]         # e.g. ["#1234", "#1235"]
```

Enforce budgets in Python before any persist call; exceed → `WorkflowExecutionError("compactor_output_over_budget")`.

---

## Files likely to touch

| File | Why |
|------|-----|
| `orion/cognition/workflows/registry.py` | Register workflow |
| `services/orion-cortex-orch/app/workflow_runtime.py` | `_execute_github_compactor_pass` |
| `services/orion-cortex-exec/app/verb_adapters.py` | Fetch PR `body` in GitHub skill |
| `orion/schemas/actions/mesh_ops.py` | `body` field on digest item |
| `orion/core/contracts/memory_cards.py` | `repo_compactor` provenance literal |
| `orion/core/storage/memory_cards.py` | `find_active_by_compactor_slot`, `supersede_compactor_card` helpers |
| `services/orion-cortex-orch/tests/test_workflow_lane.py` | Workflow regression tests |
| `services/orion-cortex-exec/tests/test_skill_verbs.py` | PR body fetch test |
| `services/orion-hub/static/js/app.js` | Workflow catalogue entry (optional, match other workflows) |
| `services/orion-hub/templates/index.html` | Workflow dropdown option |
| `services/orion-actions/README.md` | Document workflow |

No changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` in v1.

---

## Error handling

| Condition | Behavior |
|-----------|----------|
| GitHub unavailable / token missing | Workflow fails; previous card stays active; schedule attention signal |
| Zero merged PRs in window | Skip card supersession; write journal stub (“No merges in window”); status `completed` |
| LLM output over char budget | Fail workflow; no partial persist |
| `RECALL_PG_DSN` unset on orch | Fail at card step with explicit error; journal may still write if bus path healthy |
| Re-run same calendar day | Idempotent journal entry_id; card superseded again with fresh content |

---

## Env / config

Reuses existing keys — **no new env keys in v1:**

- `GITHUB_TOKEN`
- `ORION_ACTIONS_GITHUB_API_URL`
- `ORION_ACTIONS_GITHUB_OWNER`
- `ORION_ACTIONS_GITHUB_REPO`
- `ORION_ACTIONS_MESH_DEFAULT_LOOKBACK_DAYS` (override via workflow policy; default 1 for daily)
- `RECALL_PG_DSN` on cortex-orch (memory card writes)

Operator creates daily schedule via Hub workflow schedule UI (cadence `daily`, notify optional).

---

## Testing

### Gate tests

- GitHub skill returns truncated `body` when GitHub API provides it
- Compactor workflow (mocked PR list): old card → `superseded`, new card → `active`, journal published
- Idempotent re-run same day: same journal `entry_id`, exactly one active card in slot
- Zero PRs: no card mutation, journal stub written
- Missing GitHub config: fails with `github_repo_not_configured`

### Acceptance checks (operator)

1. Schedule `github_compactor_pass` daily in Hub.
2. After run: one active card tagged `repo_dev_snapshot` with compact digest.
3. Journal shows dated entry with PR refs and service groupings.
4. Chat turn can show `memory_used=true` when card injects (cortex-orch recall).
5. Re-run same day is idempotent.

---

## Risks / mitigations

| Severity | Risk | Mitigation |
|----------|------|------------|
| Low | PR bodies are empty/low quality | Fall back to title + touched paths in compact prompt |
| Medium | Large PR volume blows LLM context | Cap PR count (e.g. 20, already in skill); truncate bodies at fetch |
| Low | Card inject too verbose | Hard 800-char card budget |
| Medium | Accidental duplicate cards if slot lookup fails | Unit test slot uniqueness; use transaction for supersede+insert |

---

## Recommended next patch

1. Extend GitHub PR skill to fetch `body` (truncated).
2. Add `github_compactor_pass` workflow with mocked LLM in tests.
3. Wire memory card supersession helpers.
4. Hub catalogue + actions README.
5. Operator smoke: one manual run, verify card + journal, then schedule daily.

---

## Self-review (2026-07-08)

- [x] No TBD/TODO placeholders
- [x] Architecture matches feature description (workflow → card + journal)
- [x] Scoped for single implementation plan (no decomposition needed)
- [x] Ambiguity resolved: lookback default 1 day; quiet day skips card; v1 single repo
- [x] No keyword cathedral; `compactor_slot` has producer, consumer, test
- [x] No self-state-runtime write; no new memory service
