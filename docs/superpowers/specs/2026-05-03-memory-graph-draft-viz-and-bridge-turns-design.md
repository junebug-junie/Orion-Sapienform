# Memory graph draft visualization + bridge turn coverage — design

**Date:** 2026-05-03  
**Status:** Draft — operator review  
**Related:** [Memory graph chat entry bridge (2026-05-02)](./2026-05-02-memory-graph-chat-entry-bridge-design.md), [Memory graph annotator Hub](./2026-05-02-memory-graph-annotator-hub-design.md)

**Alignment with 2026-05-02 bridge:** The older doc’s “capped (e.g. **5** turns)” in the chain selector was an **illustrative default for token control**, not a hard product maximum. This doc is **normative** for **default depth**, **persisted operator max**, **hard upper cap**, **“Select last K”**, and **user-bubble ids** (`hub-utterance:*`); the 2026-05-02 doc is updated to point here.

---

## 1. Problem statement

1. **Readability:** The Memory tab annotator and the chat **Memory graph** modal show draft JSON and validate output in very small monospace areas; operators cannot *see* entity / situation / edge structure while editing.
2. **Turn coverage:** The bridge lists every DOM turn that has `data-turn-id`. **User** bubbles often lack ids (`appendMessage('You', …)` historically omitted meta), so the chain skews toward **assistant-only** despite role-agnostic collection logic. Operators want **user and assistant** turns visible and selectable, with **at least N** prior turns available for the `memory_graph_suggest` evidence block.

---

## 2. Goals and non-goals

### Goals

1. **Shared draft graph UI (Hub only):** Both the chat bridge modal and the Memory tab **Memory graph (optional)** block use the same behavior: parse draft JSON → **Cytoscape** graph (already loaded for card neighborhoods) + **detail panel** for the selected node/edge slice, with **hybrid** editing (see §4).
2. **Readable drafts:** Minimum comfortable font size (~12px) for the draft textarea and validate output; structured layout (draft | graph | detail) instead of a single tiny `<pre>`.
3. **Honest parse semantics:** Graph reflects **one** successfully parsed JSON object; conservative salvage (first Markdown code fence with info string `json`, or first top-level `{…}` by brace scan) only with an **explicit banner** (see §2.1); no silent “last good” graph.
4. **Bridge turn list:** Include **user and assistant** turns back to **configurable N** (persisted preference, hard upper cap), each with a **stable utterance id** suitable for `SuggestDraftV1.utterance_ids` / `utterance_text_by_id`.
5. **Data contract unchanged:** Validate / Approve bodies remain **`SuggestDraftV1`**; no API or DTO change required for “more turns” or client-issued ids (strings are opaque to RDF projection aside from consistency inside the draft).

### Non-goals

- Replacing GraphDB / validation rules or ontology contents.
- Full free-form graph drawing as the source of truth (no arbitrary sketch-to-RDF).
- Persisting Cytoscape layout coordinates into JSON in v1 (optional later; default **layout does not write back** to reduce JSON churn).

### 2.1 Parse, salvage, and draft-shape validation

- **Strict path:** If the textarea is valid JSON and decodes to a **single** top-level object, use it **without** a salvage banner.
- **Salvage path:** If strict parse fails, optionally try (in order): extract the first fenced block whose opening fence is labeled `json`; else brace-scan for the first balanced top-level `{…}`. **Any** salvage MUST show a banner that **cannot be dismissed until the draft re-parses without salvage**, e.g. “Graph from **salvaged** JSON — verify source.” If multiple `{` regions could match, prefer the **largest** balanced object or refuse with banner “Ambiguous JSON — edit manually” (implementer picks one rule and documents in PR).
- **Salvage default:** First ship may leave salvage **on** with banner; if operators paste noisy content often, a “Strict JSON only” toggle is optional later.
- **After JSON parse:** If the object **fails** lightweight `SuggestDraftV1`-shape checks (missing expected top-level keys or empty graph-relevant lists), show a **validation banner**, **empty or stub graph** (no fake nodes), **detail panel disabled** until shape passes re-parse; do not silently show a prior graph.

---

## 3. Id strategy for user and assistant turns (decision)

**Chosen approach: hybrid “C” — prefer server meta when present; otherwise stable client ids.**

| Source | Rule |
|--------|------|
| **Assistant** | Continue using existing meta (`message_id`, `turn_id`, linkage) via `canonicalTurnIdForMemoryGraph(meta)` when present. |
| **User** | If future WS/API provides `message_id` / `turn_id` / correlation-backed ids in meta passed into `appendMessage('You', text, meta)`, use the same canonical helper. |
| **User (today)** | On render, if no id after meta resolution, assign **`hub-utterance:<uuid>`** once per bubble and store it on `data-turn-id`. Never clear it for that DOM node. |

**Interaction with existing backfill:** `backfillLatestUserTurnIdForGraph` only sets `${correlationId}:user` when `dataset.turnId` is missing. Once user rows have `hub-utterance:*`, backfill skips them — acceptable; correlation-suffixed ids are optional sugar, not required for the memory-graph contract.

**Provenance / RDF:** `hub-utterance:<uuid>` is **session-local Hub identity**: a new Hub session or full page reload may mint **new** ids for the same historical utterance unless server meta arrives later. RDF / `utterance_uri` derivations remain deterministic **per id string** inside a draft; longitudinal dedup across sessions is **not** a v1 goal—server-provided `message_id` / `turn_id` remains the upgrade path for stable cross-session provenance.

**Logs / sensitivity:** Ids are not secrets but are **correlatable** in Hub and server logs alongside evidence text already sent in `memory_graph_suggest`—same sensitivity class as today’s suggest payloads (no additional PII in the id itself).

**Acceptance — backfill and DOM lifecycle**

- No **duplicate** turn rows for one logical bubble (backfill does not add a second id if `hub-utterance:*` already set).
- **First paint:** User rows eligible for the list get a stable `data-turn-id` before the bridge turn list runs its scan (assign `hub-utterance:*` on render when missing; avoid a visible flash of assistant-only list where avoidable).
- **DOM replacement:** If a bubble node is **recreated** (new element), treat it as a **new** utterance surface: assign a **new** `hub-utterance:*` (text must not keep an old uuid). If the product later preserves bubble identity across recreation, that is an explicit follow-on.

**Rationale:** Unblocks the bridge and **N**-turn selection immediately without coordinating a WS schema change; server-aligned ids remain a drop-in improvement when Hub starts echoing user message ids.

---

## 4. Draft graph UI (hybrid)

- **Preview:** Graph elements derived from parsed `SuggestDraftV1`-shaped JSON (entities, situations, edges; dispositions default **panel-only** when node count would exceed ~**40** graph elements—otherwise nodes OK; implementer documents the threshold in PR).
- **Navigation:** Pan, zoom, tap selection; optional node drag **for readability only** (no serialize to JSON unless explicitly deferred to a later “layout sync” feature). **Accessibility:** Best-effort keyboard focus order to/from the graph container and detail panel; Cytoscape is pointer-heavy—document known SR limitations in PR if any.
- **Detail panel:** Shows fields for the selected entity / situation / edge / disposition; **Apply** re-serializes the **entire** draft object (single `JSON.stringify` of the root after in-memory mutation) and replaces the textarea content **atomically** so it does not race partial string edits—operators should not edit the raw JSON in the textarea concurrently with Apply (or Apply wins last).
- **Structural add/remove:** Explicit buttons or JSON-only edits — not implicit from canvas gestures alone.

**Libraries:** Reuse **Cytoscape.js** already on the Hub page; implement mapping in a **shared** script included by `index.html`, invoked from `app.js` (bridge) and `memory.js` (Memory tab). **Cache-busting:** When the shared module changes, bump the script query string or asset version in `index.html` so operators do not load a stale graph mapper.

**Handoff (Memory tab):** v1 continues to hand off **draft JSON string** only (sessionStorage + `CustomEvent` per 2026-05-02). The graph and layout are **rebuilt** from JSON on the Memory tab—no serialized Cytoscape state in v1.

---

## 5. Bridge: configurable depth and selection

- Replace the fixed `40` with **operator-configured max turns** (input + **hard upper cap**, e.g. **80**), stored in `localStorage`. **Default** for first visit (and reset control): **5** turns visible/considered in the window—matches the 2026-05-02 illustrative token default; operators may raise up to the cap.
- **Default checkboxes:** Keep “clicked assistant + immediate prior user” when detectable; extend with a control **“Select last K turns in window”** (K ≤ loaded count) to avoid manual ticking for long chains. **Interaction:** “Select last K” **replaces** the current checkbox set with exactly the **K** most recent turns in the loaded window (including roles per §3); it does not union with prior picks. A separate **“Clear selection”** or re-open modal resets to the default checkbox rule.
- **Suggest payload:** Unchanged shape — `buildMemoryGraphSuggestUserContent(selected)` already lists `role` and `id` per turn; ensure **every** included row has a non-empty `turnId` (§3).

---

## 6. Data model impact

**None on backend DTOs.** `utterance_ids` and `utterance_text_by_id` already support arbitrary string keys and multiple turns. RDF `utterance_uri` remains a function of that string. Adding user lines only increases evidence size — manage via **N cap** and token awareness in the modal (hint text).

---

## 7. Testing

**Automated / smoke (minimum)**

- Bridge + Memory tab: new control element ids and shared script present (no load errors).
- Turn list after mixed user+assistant thread: **both** roles appear; **each** selected row has non-empty `turnId` in composed suggest payload.
- `localStorage` **max** cannot exceed hard cap (e.g. 80); default-on-first-visit behavior for max/window matches §5.
- Parse: strict JSON → no salvage banner; forced salvage fixture → banner visible; shape-invalid object after parse → validation banner and **no** misleading populated graph.

**Manual**

- Open bridge from an assistant bubble after several exchanges; run Suggest; confirm concatenated evidence and draft graph render; optional handoff to Memory tab rebuilds graph from pasted JSON.

---

## 8. Self-review checklist

- [x] No “TBD” on core decisions (ids, parse honesty, Cytoscape reuse, N configurable).
- [x] Consistent with 2026-05-02 bridge doc (handoff JSON-only, validate/approve rails); 2026-05-02 updated for depth defaults and cross-link.
- [x] Scope bounded (Hub client + optional prompt tweaks only if evidence format changes — not required for ids on user rows).
