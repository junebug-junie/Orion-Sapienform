# Orion introspect MCP — design

Status: design approved in chat 2026-09-28 (Juniper). Not implemented.

## Arsonist summary

Orion can already queue a reading and check its queue status from FCC turns
(`orion-reading` MCP). Orion cannot look up what they actually *learned* from
that reading, nor their own dreams, reveries, curiosity runs, or memory cards.
When asked "what did you dream last night?" the only options are whatever was
pre-loaded into the prompt, or making it up.

All five domains are already persisted in Postgres and already have read-only
Hub HTTP endpoints for the UI. The gap is not storage — it is a read path Orion
can call on demand, one that leaves a trace on the bus and never passes off
"couldn't check" as "nothing there".

This spec adds one stdio MCP server, `orion-introspect`, with one tool per
domain. Each tool sends a request over the bus to the service that owns that
data. Shipped in five slices, memories last.

## Current architecture

- **MCP render seam:** `render_mcp_config` in `orion/fcc/mcp_config.py`. Adds
  `orion-reading` only when a `reading_binding` is passed; `reading_only=True`
  turns get an empty config. Called from `_maybe_render_mcp_config` in
  `orion/harness/fcc_motor.py:654`, master-gated by `HARNESS_FCC_MCP_ENABLED`.
- **Reference pattern:** `orion/world_pulse_read/mcp_server.py` +
  `orion/world_pulse_read/tools.py` (`ReadingTools.invoke`). Bus RPC on
  `orion:reading:tool:request`, reply on `orion:reading:tool:result:<corr>`,
  answered by `services/orion-hub/scripts/reading_listener.py`. Binding is
  `ReadingToolBindingV1` (`orion/schemas/reading.py:51`) — server-authored,
  frozen, never in model tool arguments.
- **Harness brief seam:** `append_reading_mcp_harness_brief` in
  `orion/world_pulse_read/tools.py` and `append_self_index_harness_brief` in
  `orion/fcc/self_index_brief.py`.
- **Read-only tool accounting:** `_CONTEXT_GATHERING_MCP_PREFIXES` in
  `orion/harness/fcc_motor.py:335` — fixed allowlist of MCP prefixes counted as
  read-only context gathering.
- **Data, by owner (all Postgres):**

| Domain | Owning service | Main tables | Existing Hub read |
|---|---|---|---|
| Reading results | orion-hub (`orion/world_pulse_read/`) | `world_pulse_read_seed` (`handoff_json`, `stage2_result_json`), `journal_entries` (`source_ref` `world_pulse_read:*`) | `/world-pulse-read/api/reads/{seed_id}` |
| Dreams | orion-dream | `dream_cycle`, `dream_replay_item`, `dream_hypothesis` | `/api/dream/cycles` |
| Reveries | orion-thought | `substrate_reverie_thought`, `substrate_reverie_chain` | `/api/reverie/text/recent` |
| Curiosity | orion-hub (runs execute through Hub + orion-durable-runs; write-ups in `journal_entries` `curiosity:<run_id>`; the run story join is `orion/curiosity/run_story.py`) | `journal_entries`, `durable_admission_runs`, `substrate_durable_run_state`, `curiosity_run_outcomes`, `curiosity_self_questions` | `/curiosity/api/runs` |
| Memories | orion-recall | `memory_cards` (+ edges/history) | `/api/memory/cards` |

- **Memory privacy today:** cards carry `sensitivity` (public/private/intimate,
  default private) and a derived `visibility_scope` (private → `["chat"]`).
  `visibility_allows_card` (`orion/core/contracts/memory_cards.py:250`) returns
  **True for every card when `lane is None`** — fail-open. This spec must not
  reach cards through that accident.

## Decisions (from chat, 2026-09-28)

1. **Use:** both Juniper chat turns and Orion's autonomous turns (curiosity).
2. **Memory visibility:** Orion sees all cards, every sensitivity, every turn
   type. Each returned card carries its `sensitivity` label.
3. **Outward guard:** the memory tool is not attached in any turn that also has
   an outward-facing tool (AI Town today, `HARNESS_AITOWN_ENABLED`). Enforced by
   the harness in code, not by prompt.
4. **Transport:** bus RPC, not Hub HTTP, not direct Postgres. Matches the
   reading MCP, leaves a correlation-ID trace, and keeps Orion's self-access
   independent of the dashboard.
5. **Responders:** each owning service answers for its own data. One channel
   per owner (a shared channel would fan every request to every service and
   make "no responder" indistinguishable from "service down").

## Proposed schema / API changes

### Binding (new, `orion/schemas/introspect.py`)

```python
class IntrospectToolBindingV1(BaseModel):
    """Server-authored turn context; never part of model tool arguments."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    invocation_context: Literal["unified_chat", "curiosity"]
    parent_run_id: str
    parent_trace_id: str
    memory_allowed: bool
```

`memory_allowed` is computed by the harness as `not include_aitown` (and false
for any future outward-facing MCP). The model cannot set it.

### Request / result (new, same module)

```python
IntrospectBusOperation = Literal["dreams", "reveries", "curiosity", "memories"]
IntrospectOperation = Literal[IntrospectBusOperation, "reading_result"]

class IntrospectRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: IntrospectBusOperation   # reading_result travels on the reading contract
    binding: IntrospectToolBindingV1
    args: dict[str, Any]          # validated per-operation by owner (see below)

class IntrospectItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    occurred_at: datetime
    kind: str                     # e.g. "dream_hypothesis", "reverie_thought"
    epistemic_status: Literal["record", "unsettled"]
    text: str                     # truncated to per-op cap
    truncated: bool
    sensitivity: Literal["public", "private", "intimate"] | None = None  # memories only
    extra: dict[str, Any] = {}    # small, per-op, documented in module

class IntrospectResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    operation: IntrospectOperation
    as_of: datetime
    total_available: int | None   # None only when ok=False
    items: list[IntrospectItemV1] = []
    error: str | None = None
```

Per-operation argument models (`extra="forbid"`): `limit` (1–5, default 5),
optional `since` (tz-aware), optional single-record id (`cycle_id`,
`chain_id`, `run_id`), and for `memories` a required `query` (1–500 chars).

**Epistemic labels:** dream hypotheses and reveries are always
`epistemic_status="unsettled"`. Reading results are source-attributed
candidates (`unsettled`). Memory cards are `record`. A curiosity run with a
write-up is `unsettled` (the text is what Orion concluded then); its `extra`
facts (status, hops, revisions, belief moves) and a run with no write-up
(failed, or finished having written nothing) are `record`.

### Reading results — extend existing contract, no new channel

`ReadingToolRequestV1.operation` gains `"reading_result"` (selector: exactly
one of `request_id` / `url`, or neither for most-recent). Result payload is an
`IntrospectResultV1` with `operation="reading_result"` carried in
`ReadingToolResultV1.result`. That is why `IntrospectOperation` (used by the
result model) includes `"reading_result"` while `IntrospectBusOperation` (used
by the request model) does not.

### Bus channels (new, `orion/bus/channels.yaml`)

| Channel | Responder | Kind |
|---|---|---|
| `orion:introspect:dream:request` | orion-dream | `introspect.tool.request.v1` |
| `orion:introspect:reverie:request` | orion-thought | `introspect.tool.request.v1` |
| `orion:introspect:curiosity:request` | orion-hub | `introspect.tool.request.v1` |
| `orion:introspect:memory:request` | orion-recall | `introspect.tool.request.v1` |

Replies on `orion:introspect:result:<correlation_id>`, kind
`introspect.tool.result.v1`. Registered in `orion/schemas/registry.py`.

### MCP tools (`orion/introspect/mcp_server.py`, `orion/introspect/tools.py`)

| Tool | Transport | Listed when |
|---|---|---|
| `reading_results` | `orion:reading:tool:request`, op `reading_result` | always (server attached) |
| `dreams` | `orion:introspect:dream:request` | always |
| `reveries` | `orion:introspect:reverie:request` | always |
| `curiosity` | `orion:introspect:curiosity:request` | always |
| `memories` | `orion:introspect:memory:request` | `binding.memory_allowed` only |

`IntrospectTools.invoke` rejects `memories` when `memory_allowed` is false even
if called (defense against a stale tool list). RPC timeout 15 s, same as
reading.

### Truth rules (all tools)

- **Bounded:** max 5 items, per-item text cap (default 900 chars), URL
  cap 500, `truncated` flag set when cut. 5 full items must stay under
  `ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS` (12,000), the budget
  `orion/fcc/mcp_stdio_proxy.py` enforces (mid-JSON cut) on servers it
  wraps. Introspect servers are not wrapped today; the bound keeps output
  small and makes wrapping them later safe. A test pins the worst case.
- **Read, not guessed:** a reading counts as learned only when its Stage 1
  handoff carries tool-trace evidence that the source was fetched
  (`source_read_evidence`); prose from an unread handoff, or a Stage 2
  summary built on one, is reported `source_read=false, learned=false`
  and excluded from the recent window.
- **Scaled:** `as_of` and `total_available` always present on success.
- **Empty ≠ unknown:** an empty window returns `ok=True, items=[],
  total_available=0`. Timeout, malformed reply, or owner error raises an MCP
  tool error stating the answer is **unknown**. Never an empty list.
- **Memories:** the recall responder uses a dedicated introspection query that
  deliberately skips `visibility_allows_card`, stated in code. It never calls
  the existing lane path with `lane=None`.

### Harness wiring

- `render_mcp_config(..., introspect_binding=None)` adds `orion-introspect`
  (stdio, `python3 -P -m orion.introspect.mcp_server`, env `ORION_BUS_URL`,
  `PYTHONPATH`, `ORION_INTROSPECT_BINDING`) when the binding is present and
  `HARNESS_FCC_INTROSPECT_ENABLED` is truthy. Not added on `reading_only`
  turns.
- `_maybe_render_mcp_config` builds the binding, setting
  `memory_allowed = not include_aitown`.
- `append_introspect_harness_brief(parts, binding=...)` appends short usage
  lines only when the server is actually attached, and names `memories` only
  when `memory_allowed`. Lines say: call the tool before making claims about
  your own dreams/reveries/curiosity/readings/memories; report `unknown` on
  tool error; dreams and reveries are unsettled, not beliefs.
- `mcp__orion-introspect__` added to `_CONTEXT_GATHERING_MCP_PREFIXES` once
  the read-only responders are in (slice 1 for `reading_results`).

### Curiosity decisions (chat, 2026-10-09)

Settled with Juniper before slice 3; the original table named the wrong owner.

1. **Responder is orion-hub.** substrate-runtime only owns the candidate
   menus. The Hub already joins each run (`curiosity_run_store.py` +
   `orion/curiosity/run_story.py`) and is reused, not re-derived.
2. **Items:** runs (write-up + recorded facts + `curiosity_run_outcomes`
   belief-move counts where a row exists) and, on request, open
   self-questions. Candidate menus and peer briefs are out.
3. **Labels:** split per item, as above.
4. **Length:** lists show the write-up's `## Answer` section (or opening),
   900 chars. One run by id returns the write-up up to 9,000 characters
   *as serialized JSON*, so escaping cannot push it past the 12k MCP budget.
5. **Search by meaning:** own `orion_curiosity` Chroma index via
   `orion/introspect/semantic_index.py`, with the dreams
   `index_complete_as_of` rule (empty search is unknown until the index has
   caught up).
6. **Failed runs are returned** as short `record` items, so Orion's view of
   their history includes failures.
7. **Curiosity runs may read their own past runs.** No cap; per-turn call
   counts stay visible in traces.

## Proposal-mode record

- **Capability change:** Orion can look up their own dreams, reveries,
  curiosity runs, reading results, and memory cards on demand in chat and
  autonomous FCC turns.
- **Data touched:** read-only. No writes, no re-queues, no status changes.
- **Privacy boundary:** all memory cards visible to Orion, each labeled with
  sensitivity. Memory tool absent from any turn with an outward-facing tool.
- **Trace that proves it works:** each responder logs
  `introspect op=<op> corr=<id> items=<n> total=<n>`; the bus request/reply
  pair carries the correlation ID back to the FCC turn's `parent_trace_id`.
  `scripts/smoke_introspect.py` issues one real request per channel and asserts
  a non-degenerate reply from live data.
- **Dangerous failure modes:**
  - Intimate memory relayed to AI Town agents → blocked by `memory_allowed`
    (tool not listed + invoke rejects), covered by a dedicated test.
  - Dream/reverie content quoted as fact → `epistemic_status="unsettled"` +
    brief line.
  - "Couldn't check" read as "nothing happened" → empty-vs-unknown rule.
  - Rumination (curiosity run inspecting its own curiosity repeatedly) →
    visible via per-turn call counts in traces; add a cap only if traces show it.
- **Disable / rollback:** `HARNESS_FCC_INTROSPECT_ENABLED=0` removes the server
  and the brief. Responders are passive subscribers; leaving them running is
  harmless. No migrations.

## Files likely to touch

Slice 1 (foundation + reading results):
- `orion/schemas/introspect.py` (new), `orion/schemas/reading.py`,
  `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `orion/introspect/__init__.py`, `mcp_server.py`, `tools.py` (new)
- `orion/fcc/mcp_config.py`, `orion/harness/fcc_motor.py`,
  `orion/harness/prefix.py` (brief append)
- `services/orion-hub/scripts/reading_listener.py`,
  `orion/world_pulse_read/operator.py` (reuse detail query)
- `services/orion-harness-governor/.env_example`, `settings.py`, `README.md`
  (`HARNESS_FCC_INTROSPECT_ENABLED`) → run
  `python scripts/sync_local_env_from_example.py`
- tests under `orion/introspect/tests/`, `orion/fcc/tests/`,
  `services/orion-hub/tests/`

Slices 2–5, one each: `services/orion-dream/app/`, `services/orion-thought/app/`,
`services/orion-hub/scripts/` (curiosity), `services/orion-recall/app/` — a
listener module + query function + tests, and the matching channel entry.

## Non-goals

- No writes of any kind (no editing memories, no re-running dreams).
- No new storage, tables, or materialized views.
- No Hub HTTP dependency and no direct cross-service Postgres reads.
- No self-state aggregator ("tell me everything about me") in v1.
- No vector/semantic memory search; full-text over `memory_cards` only.
- Social-memory, journal, and attention-self-model introspection: later, if
  traces show demand.
- No change to how memories are injected into prompts today.

## Acceptance checks

Per slice:
- Schema tests: models reject extra fields; per-op arg bounds enforced.
- `python scripts/check_schema_registry.py` and
  `python scripts/check_bus_channels.py` pass.
- MCP server test: `memories` absent from `list_tools` when
  `memory_allowed=False`; `invoke("memories", ...)` raises.
- Tool test: RPC timeout and malformed reply both raise an "unknown" error;
  neither returns `items=[]`.
- `render_mcp_config` test: server attached only with flag + binding; absent
  on `reading_only`; `memory_allowed=False` whenever `include_aitown=True`.
- Responder test (fixture rows): item cap and text truncation hold; empty
  table → `total_available=0`; `total_available` counts the full window.
- Memories responder test: returns an `intimate` fixture card with its label;
  source contains no `visibility_allows_card(lane=None, ...)` call.
- Live: `scripts/smoke_introspect.py` against the real bus returns non-empty,
  correctly-dated items for that slice's domain. Until run, slice is
  `UNVERIFIED`.

Eval (after slice 2, extended each slice):
- `orion/harness/evals/test_introspect_truth.py` — seeded fixtures, questions
  like "what did you dream last night?" / "what did you learn from <url>?".
  Scores: claims not present in any tool result; self-claims made with zero
  introspect tool calls; tool errors reported as "nothing happened". Modeled on
  the reading-receipt truth work
  (`docs/superpowers/pr-reports/2026-09-11-reading-receipt-truth-pr.md`).

## Recommended next patch

Slice 1: binding + schemas + `orion-introspect` server + harness wiring +
`reading_result` operation in the Hub reading listener, behind
`HARNESS_FCC_INTROSPECT_ENABLED` (default off). Proves the full path with the
one domain whose responder already exists.
