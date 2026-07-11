# Chat history compactor — indexed conversation digests

**Date:** 2026-07-09  
**Status:** Approved for implementation planning  
**Problem:** Hub chat lands every turn in SQL `chat_history_log` and vector memory, but Orion has no durable, indexed digest of *what was discussed* that surfaces at chat time via memory cards. Existing paths (`skills.chat.discussion_window.v1`, `journal_discussion_window_pass`) extract transcripts on demand for journaling — they do not produce bounded, idempotent, recall-oriented memory cards.

---

## Arsonist summary

Compact bounded windows of `chat_history_log` into **indexed memory cards** (not superseding slots) so Orion has a searchable digest layer at chat time. Optional light journal append for lineage.

Implementation: new workflow `chat_history_compactor_pass` in cortex-orch, thin shared primitives under `orion/cognition/compactor/` for window keys + upsert-by-index, Skill Runner + daily schedule at 06:00 `America/Denver`. Reuses `skills.chat.discussion_window.v1` for SQL fetch. No new memory service, no bus/registry changes in v1.

**Distinct from GitHub compactor:** GitHub uses `compactor_slot` + supersede (one live snapshot). Chat uses `compactor_index` + upsert (many coexisting window cards).

---

## Goals

- Orion has plain-language digests of recent Hub conversation available as `high_recall` memory cards.
- Cards are **indexed by stable window keys** — reruns update the same card; different windows coexist.
- **Global scope** (all Hub chat in the window) for v1.
- **Dual trigger:** Skill Runner (on-demand, configurable lookback) + daily schedule (06:00 MST, yesterday's calendar day).
- Digest LLM runs **chat/brain route first**, falls back to **quick** on empty/invalid output.
- **Extensibility seam** for future Postgres sources (e.g. journals with field filters) without implementing journals in v1.
- Reuses existing env: `DATABASE_URL` / `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL` on cortex-exec, `RECALL_PG_DSN` on cortex-orch.

## Non-goals (v1)

- Journal table compaction or field-filtered journal sources
- `chat_history_log` row deletion, archival, or tier mutation (table hygiene)
- Per-user or per-source scoped cards (policy hooks reserved, not shipped)
- Superseding / single-slot snapshot semantics (`compactor_slot` pattern)
- New bus channels or schema registry event kinds
- Keyword/phrase taxonomies for conversation themes
- Hub Memory tab UI beyond existing card/journal surfaces

---

## Current architecture

| Piece | Today |
|-------|--------|
| `chat_history_log` | Append-only SQL sink for Hub turns (prompt/response, `user_id`, `source`, `correlation_id`, …) |
| Vector path | `chat.history.message.v1` → embeddings for semantic recall — not compaction |
| `skills.chat.discussion_window.v1` | Bounded SQL read + contiguous-cluster transcript from `chat_history_log` |
| `journal_discussion_window_pass` | On-demand: discussion window → journal compose → write; no memory card |
| `github_compactor_pass` | PR fetch → digest → **supersede** `repo_dev_snapshot` slot + journal |
| Memory cards recall | `orion-recall` embedding scoring + cortex-orch `always_inject` path |
| Workflow schedules | Durable store in orion-actions; Hub schedule inventory UI |

**Gap:** No scheduled or Skill Runner workflow turns `chat_history_log` → bounded digest → **indexed** memory card.

---

## Proposed architecture

```text
[Trigger paths]
  A) Hub Skill Runner → workflow chat_history_compactor_pass
       (Cognitive workflows optgroup; digest subverb: chat route → quick fallback)
  B) Daily schedule → orion-actions claims due @ 06:00 America/Denver
       → orion:actions:trigger:workflow.v1 (workflow_id=chat_history_compactor_pass)
       (bootstrap seed if schedule absent — visible in Hub schedule inventory)

[Workflow — cortex-orch _execute_chat_history_compactor_pass]

Step 1 — Resolve window
  scheduled: yesterday's calendar day in America/Denver (00:00–23:59:59.999)
  on-demand: lookback_hours from prompt/policy (default 24), rolling window ending now

Step 2 — Fetch
  → skills.chat.discussion_window.v1
  → map window to lookback_seconds + end_time_utc
  → global scope (no user_id / source filter in v1)
  → bounded max_turns + input trim before LLM

Step 3 — Compact
  → chat_history_compactor_digest_v1 (brain lane)
  → llm_route: chat/metacog first; retry on quick if empty/invalid JSON
  → output: ChatHistoryCompactorDigestV1
       { card_summary, journal_title?, journal_body?, turn_refs[] }
  → char budgets enforced before persist (fail loud)

Step 4 — Memory card (indexed upsert)
  → find active card where subschema.compactor_index = <window_key>
  → if found: UPDATE summary, evidence, time_horizon, subschema counters, updated_at
  → else: INSERT new active card (priority=high_recall, provenance=chat_compactor)
  → subschema: { compactor_index, compactor_kind, window_mode, turn_count, window_start, window_end }

Step 5 — Journal (light, optional)
  → if journal_body present and non-quiet: journal.entry.write.v1
  → stable entry_id from (workflow_id, compactor_index)

Step 6 — Result
  → workflow metadata: turn_count, card_id, compactor_index, card_summary_preview, journal_entry_id?
```

---

## Window key contract (scheme D)

| Mode | When | Key shape | Example |
|------|------|-----------|---------|
| **day** | Scheduled 06:00 `America/Denver` | `chat_compactor:day:{YYYY-MM-DD}` | `chat_compactor:day:2026-07-08` (yesterday when run on 2026-07-09) |
| **rolling** | On-demand Skill Runner / policy `lookback_hours` | `chat_compactor:rolling:{N}h:{start_iso}` | `chat_compactor:rolling:6h:2026-07-09T04:00:00-06:00` |

- `start_iso` is window start in ISO-8601 with offset (floor to minute for stability).
- Reruns with the same key update the same card — idempotent.
- Different keys produce separate active cards — **indexed history**, not supersede.

**Shared primitives** (`orion/cognition/compactor/index.py`):

```python
def build_compactor_index(
    *,
    kind: str,  # "chat_history_log" in v1
    mode: Literal["day", "rolling"],
    calendar_date: str | None = None,  # mode=day
    lookback_hours: int | None = None,
    window_start: datetime | None = None,
) -> str: ...
```

Future `kind="journal"` adds field-filter dimensions to the key — not implemented in v1.

---

## Schema / API changes

### Compactor digest output (new)

**File:** `orion/schemas/actions/chat_history_compactor.py`

```python
class ChatHistoryCompactorDigestV1(BaseModel):
    card_summary: str          # max CARD_SUMMARY_MAX_CHARS (default 800)
    journal_title: str         # max JOURNAL_TITLE_MAX_CHARS (default 120); optional for quiet paths
    journal_body: str          # max JOURNAL_BODY_MAX_CHARS (default 4000); may be empty
    turn_refs: List[str]       # correlation_ids or turn ids from source rows
```

Enforce budgets in Python before persist; exceed → `WorkflowExecutionError("compactor_output_over_budget")`.

### Memory card indexed upsert

**Contract:** `subschema.compactor_index = <window_key>` (unique among active cards per index)

- **Producer:** `chat_history_compactor_pass`
- **Consumers:** cortex-orch recall inject (existing paths); `orion-recall` cards embedding adapter
- **Tag mirror:** `tags: ["chat_dev_digest"]` for Hub filtering
- **Provenance:** add `"chat_compactor"` to `MemoryProvenance` literal

**DAL** (`orion/core/storage/memory_cards.py`):

- `find_active_card_by_compactor_index(pool, index) -> Optional[MemoryCardV1]`
- `upsert_indexed_compactor_card(pool, *, index, card, actor) -> UUID`
  - Transaction: lookup by index → update in place or insert
  - Write `memory_card_history` op `update` or `create` (not supersede)

**Card defaults:**

```python
MemoryCardCreateV1(
    types=["fact"],
    anchor_class="event",
    status="active",
    priority="high_recall",
    provenance="chat_compactor",
    title=f"Chat digest ({window_label})",
    summary=digest.card_summary,
    tags=["chat_dev_digest"],
    time_horizon=TimeHorizonV1(kind="era_bound", start=window_start_iso, end=window_end_iso),
    evidence=[EvidenceItemV1(source=ref, ts=window_label) for ref in turn_refs[:12]],
    subschema={
        "compactor_index": index_key,
        "compactor_kind": "chat_history_log",
        "window_mode": "day" | "rolling",
        "turn_count": N,
        "window_start": iso,
        "window_end": iso,
    },
)
```

### Workflow registration

**File:** `orion/cognition/workflows/registry.py`

```python
WorkflowDefinition(
    workflow_id="chat_history_compactor_pass",
    display_name="Chat History Compactor",
    description="Compact bounded chat_history_log windows into indexed memory cards (optional journal).",
    aliases=[
        "compact chat history",
        "compact last 24 hours of chat",
        "chat history compactor",
        "what have we been talking about",
    ],
    user_invocable=True,
    autonomous_invocable=True,
    execution_mode="sync",
    may_call_actions=False,
    persistence_policy="Upsert one indexed memory card per window key; optional append-only journal entry.",
    result_surface="Return turn count, compactor_index, card summary preview, card_id.",
    steps=[
        WorkflowStepDefinition(step_id="fetch_window", description="Fetch discussion window from SQL.", adapter="verb:skills.chat.discussion_window.v1"),
        WorkflowStepDefinition(step_id="compact", description="LLM compact digest JSON.", adapter="verb:chat_history_compactor_digest_v1"),
        WorkflowStepDefinition(step_id="persist_card", description="Upsert indexed memory card.", adapter="memory_cards:compactor_index"),
        WorkflowStepDefinition(step_id="persist_journal", description="Optional journal append.", adapter="journaler:append_only_write"),
    ],
    planner_hints=["Use when the operator asks to compact recent Hub chat into a durable memory digest."],
)
```

### Brain-lane digest verb

- **Prompt:** `orion/cognition/prompts/chat_history_compactor_digest_v1.j2`
- **Verb:** `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml`
- **Router:** allowlist in `services/orion-cortex-exec/app/router.py`
- **Route policy:** primary `llm_route=chat` (or metacog per executor convention); on empty/invalid structured output, one retry with `llm_route=quick` before fail

Input metadata key: `chat_history_compactor_input` (trimmed discussion window JSON).

---

## Skill Runner & schedule surfaces

### Skill Runner (Hub)

Add to `services/orion-hub/scripts/skill_runner_catalogue.py` and `templates/index.html` (Cognitive workflows optgroup):

| Catalogue prompt | Workflow |
|------------------|----------|
| `Compact the last 24 hours of chat into a memory digest.` | `chat_history_compactor_pass` (default rolling 24h) |
| `Compact the last 6 hours of chat into a memory digest.` | `chat_history_compactor_pass` (policy `lookback_hours=6`) |

Also register in `services/orion-hub/static/js/app.js` workflow catalogue and `workflow-ui.js` label map.

**Lane behavior:** Workflow invocation uses normal chat routing for the digest subverb — **not** the deterministic `skills.*`-only Skill Runner lane. Digest step: chat → quick fallback.

### Daily schedule bootstrap

**File:** `services/orion-actions/app/workflow_schedule_bootstrap.py` (new, thin)

On orion-actions startup (after schedule store init), if no recurring schedule exists for `chat_history_compactor_pass`:

- cadence: `daily`
- time: `06:00`
- timezone: `America/Denver`
- `notify_on`: match existing workflow schedule defaults (e.g. failure + completion attention)
- workflow_request metadata: `{ "window_mode": "day" }` (yesterday calendar day)

Idempotent: skip if schedule already present (match by `workflow_id` + recurring daily signature).

Document in `services/orion-actions/README.md` under `### chat_history_compactor_pass`.

---

## Files likely to touch

| File | Why |
|------|-----|
| `orion/cognition/compactor/index.py` | Shared window key builders |
| `orion/cognition/compactor/budget.py` | Generic budget assertion |
| `orion/cognition/compactor/tests/` | Pure key/budget tests |
| `orion/cognition/chat_history_compactor/constants.py` | Char budgets, tag, max turns |
| `orion/cognition/chat_history_compactor/digest.py` | Input trim, quiet-day stub, journal entry id |
| `orion/cognition/chat_history_compactor/tests/` | Digest helper tests |
| `orion/schemas/actions/chat_history_compactor.py` | Output schema |
| `orion/core/contracts/memory_cards.py` | `chat_compactor` provenance |
| `orion/core/storage/memory_cards.py` | `find_active_card_by_compactor_index`, `upsert_indexed_compactor_card` |
| `tests/test_indexed_compactor_memory_cards.py` | DAL unit tests (mocked asyncpg) |
| `orion/cognition/prompts/chat_history_compactor_digest_v1.j2` | LLM prompt |
| `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml` | Verb registration |
| `services/orion-cortex-exec/app/router.py` | Allowlist + route fallback hook |
| `services/orion-cortex-exec/app/executor.py` | chat→quick retry for digest verb (if not generic) |
| `services/orion-cortex-orch/app/chat_history_compactor_memory.py` | Pool + upsert wrapper |
| `services/orion-cortex-orch/app/workflow_runtime.py` | `_execute_chat_history_compactor_pass` |
| `orion/cognition/workflows/registry.py` | Register workflow |
| `services/orion-hub/scripts/skill_runner_catalogue.py` | Catalogue entries |
| `services/orion-hub/templates/index.html` | Skill Runner options |
| `services/orion-hub/static/js/app.js` | Workflow catalogue |
| `services/orion-hub/static/js/workflow-ui.js` | Display label |
| `services/orion-actions/app/workflow_schedule_bootstrap.py` | Default 06:00 MST schedule seed |
| `services/orion-actions/app/main.py` | Call bootstrap on startup |
| `services/orion-actions/README.md` | Operator docs |
| `docs/operator_skill_prompt_catalogue.md` | Sync catalogue prompts |

No changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` in v1.

---

## Error handling

| Condition | Behavior |
|-----------|----------|
| `DATABASE_URL` missing on cortex-exec | Fail at fetch: `discussion_window_skill_unavailable` |
| Zero turns in window | Quiet day — skip card upsert; optional short journal stub; status `completed` |
| LLM output over char budget | Fail workflow; no partial persist |
| Brain digest empty/invalid JSON | One retry on `quick` route; then `chat_compactor_digest_failed` |
| `RECALL_PG_DSN` unset on orch | Fail card step with `recall_pg_dsn_unavailable`; journal may still write if bus healthy |
| Re-run same `compactor_index` | Upsert updates same `card_id` (idempotent) |
| Schedule bootstrap race | Idempotent upsert; second writer no-ops |

---

## Env / config

Reuses existing keys — **no new env keys in v1:**

- `DATABASE_URL` or `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL` on cortex-exec (discussion window skill)
- `RECALL_PG_DSN` on cortex-orch (memory card writes)
- `ORION_BUS_URL` (journal write path, existing)

Operator may edit/pause the bootstrapped schedule in Hub. Override lookback via workflow execution policy metadata: `lookback_hours`, `window_mode`.

---

## Testing

### Gate tests

```bash
pytest orion/cognition/compactor/tests -q
pytest orion/cognition/chat_history_compactor/tests -q
pytest tests/test_indexed_compactor_memory_cards.py -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k chat_history_compactor -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -k chat_history_compactor -q
pytest services/orion-actions/tests -k chat_history_compactor_schedule_bootstrap -q
```

**Cases:**

- Window key builders: day vs rolling determinism
- Upsert: same index → update, not second card; different index → two active cards
- Workflow (mocked window + digest): card upserted, journal published when body present
- Quiet day: no card, optional stub journal
- Over-budget digest: fail without persist
- Brain fail → quick retry succeeds
- Schedule bootstrap: creates once, second call no-op

### Acceptance checks (operator)

1. Skill Runner "Compact the last 24 hours of chat…" → active card with `compactor_index` in Hub Memory.
2. Rerun same window → same `card_id`, updated `summary`.
3. Run 6-hour window → second card coexists (not superseded).
4. Hub schedule inventory shows daily 06:00 `America/Denver` entry after actions restart.
5. Scheduled run produces yesterday's `chat_compactor:day:{date}` card.
6. Chat recall can surface card (`high_recall`, embedding path) when query overlaps digest topic.

---

## Risks / mitigations

| Severity | Risk | Mitigation |
|----------|------|------------|
| Medium | Large transcripts blow LLM context | Cap `max_turns`; trim per-turn chars in `digest.py` before verb |
| Low | Contiguous-cluster selection drops older same-day turns | Document selection strategy in workflow metadata; tune `max_turns` |
| Medium | Too many indexed cards over months | v2: archival policy (`status=archived` after N days); out of v1 scope |
| Low | Global scope mixes users | v2 per-user keys; v1 documented as global house digest |
| Low | Quick fallback lowers digest quality | Log route used in workflow metadata; operator can re-run |

---

## Recommended next patch

1. Shared `compactor_index` helpers + DAL upsert (tests first).
2. `ChatHistoryCompactorDigestV1` schema + digest helpers + LLM verb.
3. `_execute_chat_history_compactor_pass` workflow + registry.
4. Hub Skill Runner + workflow catalogue entries.
5. orion-actions schedule bootstrap + README.
6. Operator smoke: manual 24h run → verify card; wait for / force schedule → verify day key.

---

## Self-review (2026-07-09)

- [x] No TBD/TODO placeholders
- [x] Architecture matches goals (indexed cards, not supersede; global v1; dual trigger)
- [x] Scoped for single implementation plan; journal source explicitly deferred
- [x] Ambiguity resolved: scheduled = yesterday calendar day MST; on-demand = rolling with hour param; quiet day skips card
- [x] No keyword cathedral; `compactor_index` has producer, consumer, DAL, test
- [x] Distinct from GitHub `compactor_slot` supersede pattern documented
- [x] No self-state-runtime write; no new memory service; no bus/registry v1 changes
