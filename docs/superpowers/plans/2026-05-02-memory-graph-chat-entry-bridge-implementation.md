# Memory graph chat entry bridge — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Hub chat → memory-graph **Suggest** entry with structured turn evidence and Memory-tab handoff, and sequence **parent-spec gap closure** (§5, §12, appendices, downstream §9.x) as phased work.

**Architecture:** Reuse `resolveFeedbackLinkage` / message meta for stable ids; compose a structured **user_message** for `memory_graph_suggest` (prompt already lists Appendix C keys). Sync draft to Memory via `sessionStorage` + `CustomEvent`. Parallel track: harden ontology / SHACL / projector / tests already started under `ontology/memory/` and `tests/test_memory_graph_*`.

**Tech stack:** Orion Hub (`templates/index.html`, `static/js/app.js`, `static/js/memory.js`), FastAPI Hub routes (`services/orion-hub/scripts/memory_graph_routes.py`), `orion/memory_graph/*`, Jinja `orion/cognition/prompts/memory_graph_suggest_prompt.j2`, pytest.

---

## File map (bridge)

| File | Role |
|------|------|
| `services/orion-hub/templates/index.html` | Modal markup + optional `data-memory-graph-event-listener` hook |
| `services/orion-hub/static/js/app.js` | Turn chain UI, Suggest POST, open/close modal |
| `services/orion-hub/static/js/memory.js` | Listen for `orion-hub-memory-graph-draft-import`; prefill `#memoryGraphDraftJson`; optional tab switch via existing patterns |
| `orion/cognition/prompts/memory_graph_suggest_prompt.j2` | Optional: explicit subsection for machine-readable evidence block |

---

### Task 1: Canonical turn id helper (app.js)

**Files:**
- Modify: `services/orion-hub/static/js/app.js` (place near `resolveFeedbackLinkage` / `feedbackTargetKey`)

- [ ] **Step 1: Add helper after `resolveFeedbackLinkage`**

```javascript
function canonicalTurnIdForMemoryGraph(meta = {}) {
  const linkage = resolveFeedbackLinkage(meta);
  const id =
    (linkage && linkage.targetMessageId) ||
    meta.messageId ||
    meta.message_id ||
    meta.turnId ||
    meta.turn_id ||
    linkage?.targetTurnId ||
    '';
  const s = String(id || '').trim();
  return s || null;
}
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): canonical turn id helper for memory graph bridge"
```

---

### Task 2: Compose structured user message for Suggest

**Files:**
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: Add builder used only by memory-graph modal**

```javascript
function buildMemoryGraphSuggestUserContent(turns) {
  const lines = [];
  lines.push('Structured transcript evidence for memory graph extraction (do not invent turns).');
  lines.push('');
  turns.forEach((t, i) => {
    const id = t.turnId;
    const role = t.role || 'unknown';
    lines.push(`--- turn ${i + 1} id=${id} role=${role} ---`);
    lines.push(t.text || '');
    lines.push('');
  });
  lines.push('Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts.');
  return lines.join('\n');
}
```

`turns` items must be `{ turnId, role, text }` where `turnId` is the canonical id string.

- [ ] **Step 2: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): structured memory graph suggest user message builder"
```

---

### Task 3: Modal DOM + open/close

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: Append modal markup before `</body>` or inside `#hub` panel host — minimal**

```html
<div id="memoryGraphBridgeModal" class="hidden fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-4" aria-hidden="true">
  <div class="bg-gray-900 border border-gray-700 rounded-xl max-w-lg w-full max-h-[90vh] overflow-hidden flex flex-col shadow-xl">
    <div class="flex items-center justify-between px-4 py-2 border-b border-gray-800">
      <div class="text-sm font-semibold text-white">Memory graph from chat</div>
      <button type="button" id="memoryGraphBridgeModalClose" class="text-gray-400 hover:text-white text-lg leading-none">&times;</button>
    </div>
    <div class="p-4 overflow-y-auto flex-1 space-y-3 text-xs text-gray-300">
      <div id="memoryGraphBridgeTurnList" class="space-y-2"></div>
      <textarea id="memoryGraphBridgeDraft" class="w-full h-36 bg-gray-950 border border-gray-700 rounded p-2 font-mono text-[10px]" placeholder="Draft JSON appears here after Suggest…"></textarea>
      <pre id="memoryGraphBridgeStatus" class="text-[10px] text-gray-500 whitespace-pre-wrap"></pre>
    </div>
    <div class="flex flex-wrap gap-2 px-4 py-3 border-t border-gray-800">
      <button type="button" id="memoryGraphBridgeSuggest" class="px-3 py-1 rounded border border-indigo-700 bg-indigo-900/40 text-indigo-100 text-xs">Suggest draft</button>
      <button type="button" id="memoryGraphBridgeToMemory" class="px-3 py-1 rounded border border-gray-600 bg-gray-800 text-gray-200 text-xs">Continue on Memory tab</button>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Wire close + backdrop click + ESC in `app.js` `DOMContentLoaded`**

```javascript
function setupMemoryGraphBridgeModal() {
  const modal = document.getElementById('memoryGraphBridgeModal');
  const closeBtn = document.getElementById('memoryGraphBridgeModalClose');
  if (!modal || !closeBtn) return;
  function close() {
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
  }
  closeBtn.addEventListener('click', close);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) close();
  });
}
```

Call `setupMemoryGraphBridgeModal()` from existing startup/init path next to other modal setups.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js
git commit -m "feat(hub): memory graph bridge modal shell"
```

---

### Task 4: Conversation history slice for chain selection

**Files:**
- Modify: `services/orion-hub/static/js/app.js`

Implementation detail depends on how turns are stored today. **Locate** the structure pushed when rendering each bubble (search `chatMessages` vs in-conversation **scroll buffer**). If only DOM exists, add **`data-turn-id`** and **`data-role`** on each message root when appending to `#conversation`, storing **`canonicalTurnIdForMemoryGraph(meta)`** and role (`You` / `Orion`).

- [ ] **Step 1: When creating each conversation row `div`, set**

```javascript
const tid = canonicalTurnIdForMemoryGraph(meta);
if (tid) div.dataset.turnId = tid;
div.dataset.role = sender === 'Orion' ? 'assistant' : 'user';
```

- [ ] **Step 2: Implement `collectConversationTurnsUpTo(anchorEl, maxTurns)`** walking previous siblings / backward in DOM, collecting `{ turnId, role, text }` until cap.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): data-turn-id on chat bubbles for graph chain picking"
```

---

### Task 5: Assistant bubble button + modal populate

**Files:**
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: In Orion assistant branch where feedback buttons attach (~5749), append**

```javascript
const graphBtn = document.createElement('button');
graphBtn.type = 'button';
graphBtn.className = 'rounded-full border border-violet-500/40 bg-violet-500/10 px-2 py-1 text-[10px] font-semibold text-violet-200 hover:bg-violet-500/20';
graphBtn.textContent = 'Memory graph';
graphBtn.addEventListener('click', () => openMemoryGraphBridgeModal(div));
actionRow.appendChild(graphBtn);
```

Adjust `openMemoryGraphBridgeModal` to receive the bubble root `div`.

- [ ] **Step 2: Implement `openMemoryGraphBridgeModal(anchorDiv)`**

Populate `#memoryGraphBridgeTurnList` with checkboxes defaulting to anchor + prior user message checked; call `collectConversationTurnsUpTo`.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): Memory graph button opens bridge modal"
```

---

### Task 6: Suggest draft from modal

**Files:**
- Modify: `services/orion-hub/static/js/app.js` (mirror `memory.js` fetch)

Reuse API base constant (`API_BASE_URL`), headers `sessionHeader()` if exported — else duplicate minimal `{ "Content-Type": "application/json", "X-Orion-Session-Id": ... }` consistent with `memory.js`.

- [ ] **Step 1: On `#memoryGraphBridgeSuggest` click**

```javascript
const content = buildMemoryGraphSuggestUserContent(selectedTurns);
const payload = {
  mode: 'brain',
  verbs: ['memory_graph_suggest'],
  messages: [{ role: 'user', content: content }],
  use_recall: false,
  no_write: true,
};
```

Parse JSON from response; write string into `#memoryGraphBridgeDraft`; status note matching memory.js (“trim prose wrapper”).

- [ ] **Step 2: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): memory graph suggest from bridge modal"
```

---

### Task 7: Handoff to Memory tab + integration test (manual checklist)

**Files:**
- Modify: `services/orion-hub/static/js/memory.js`
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: On `#memoryGraphBridgeToMemory` click**

```javascript
const raw = document.getElementById('memoryGraphBridgeDraft')?.value || '';
sessionStorage.setItem('orion_memory_graph_draft_import', raw);
window.dispatchEvent(new CustomEvent('orion-hub-memory-graph-draft-import', { detail: { source: 'bridge' } }));
```

- [ ] **Step 2: In `memory.js` `DOMContentLoaded` block, add listener**

```javascript
window.addEventListener("orion-hub-memory-graph-draft-import", () => {
  const v = sessionStorage.getItem("orion_memory_graph_draft_import");
  if (v && draftTa) draftTa.value = v;
  sessionStorage.removeItem("orion_memory_graph_draft_import");
  graphSetOut(
    { ok: true, note: "Draft loaded from chat bridge. Review JSON then Validate." },
    false
  );
});
```

- [ ] **Step 3: Optional — switch nav to Memory** using same mechanism other tabs use (search `data-panel="memory"` click).

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/js/memory.js services/orion-hub/static/js/app.js
git commit -m "feat(hub): sync memory graph draft from bridge to Memory tab"
```

---

### Task 8: Prompt template tightening (optional follow-up)

**Files:**
- Modify: `orion/cognition/prompts/memory_graph_suggest_prompt.j2`

- [ ] **Step 1: After `User turn / context:` add**

```jinja2
Structured evidence (when present) lists turn ids and bodies — preserve those ids in utterance_ids and utterance_text_by_id.
```

- [ ] **Step 2: Commit**

```bash
git add orion/cognition/prompts/memory_graph_suggest_prompt.j2
git commit -m "docs(prompt): clarify memory_graph_suggest evidence ids"
```

---

### Task 9: Hub regression test (optional)

**Files:**
- Create: `services/orion-hub/tests/test_memory_graph_bridge_helpers.py` **only if** helpers are extracted to importable Python — **otherwise** skip and rely on manual QA checklist in Task 7.

If no Python extraction, replace with:

- [ ] **Manual QA:** Open Hub → send chat → Memory graph on reply → Suggest → Continue on Memory → Validate fixture `tests/fixtures/memory_graph/joey_cats_draft.json` path documented.

---

## Parent-spec gap closure (sequenced backlog)

These tasks implement **unfinished parent-spec items**; they may land in separate PRs.

### Task 10: §5 — Ontology + SHACL completeness

**Files:**
- Modify: `ontology/memory/orionmem-v2026-05.ttl`
- Modify: `ontology/memory/shapes-orionmem-v2026-05.ttl`
- Reference: parent spec **§4**, **Appendix B**

- [ ] **Step 1:** SHACL shapes cover **§4.7 Situation shape** and entity kinds per Appendix B table.

- [ ] **Step 2:** Run existing memory-graph validation tests:

```bash
pytest tests/test_memory_graph_validate.py tests/test_memory_graph_json_to_rdf.py -v
```

Expected: PASS (extend tests if new constraints added).

- [ ] **Step 3: Commit**

```bash
git add ontology/memory/
git commit -m "feat(ontology): align orionmem SHACL with memory graph annotator spec"
```

---

### Task 11: §5 — Projector mapping artifact

**Files:**
- Create or modify: `orion/memory_graph/projector_mapping.yaml` (or JSON colocated with package — match repo convention when added)

- [ ] **Step 1:** Encode **§4.9–§4.10** literal mapping notes from parent spec into YAML consumed by `project.py` (refactor projector to read file if currently hardcoded).

- [ ] **Step 2:** Run:

```bash
pytest tests/test_memory_graph_project.py -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(memory-graph): versioned projector mapping for Appendix D"
```

---

### Task 12: §5 — Golden exemplar + approve round-trip

**Files:**
- Existing: `tests/fixtures/memory_graph/joey_cats_draft.json`
- Modify: `tests/test_memory_graph_approve.py`

- [ ] **Step 1:** Ensure fixture matches **Appendix A** narrative and dual-write expectations.

- [ ] **Step 2:**

```bash
pytest tests/test_memory_graph_approve.py tests/test_memory_graph_graphdb_mocked.py -v
```

- [ ] **Step 3: Commit** when assertions match parent golden intent.

---

### Task 13: §12 — Compensation strategy (GraphDB ok, Postgres fails)

**Files:**
- Modify: `orion/memory_graph/approve.py`
- Modify: docs in parent spec or ADR under `docs/` only if requested

- [ ] **Step 1:** Implement single documented strategy (delete batch triples vs tombstone) per parent **§7**.

- [ ] **Step 2:** Add test simulating Postgres failure after GraphDB write.

- [ ] **Step 3:**

```bash
pytest tests/test_memory_graph_approve.py -v
```

---

### Task 14: §12 — memory-graph deploy shape

**Files:**
- Evaluate: `services/orion-hub/scripts/memory_graph_routes.py` vs standalone service

- [ ] **Step 1:** Document chosen phase (**library-in-Hub** vs FastAPI split) in repo `README` fragment under `services/orion-hub/` **only if** operators need runbook — avoid unsolicited markdown per team norms; prefer code comment on router.

---

### Task 15: §12 — Optional `MemoryProvenance.operator_graph`

**Files:**
- Modify: `orion/core/contracts/memory_cards.py` (if enum extended)

- [ ] **Step 1:** Add enum value + projector stamp when approve originates from graph annotator.

---

### Task 16: §8.3 — Distiller (D) behind flag

**Files:**
- TBD service wiring — gate **D** per parent **§8.3** after **P** stable.

- [ ] **Step 1:** Settings flag `MEMORY_GRAPH_DISTILLER_ENABLED` default false.

---

### Task 17: §8.5 / §9.1 — Recall routing sibling track

**Files:**
- `services/orion-recall/` (separate plan recommended)

- [ ] **Step 1:** Open follow-on plan `docs/superpowers/plans/YYYY-MM-DD-recall-memory-graph-routing.md` when bridge + §5 stable — **not** part of bridge PR.

---

### Task 18: §9.4–§9.6 — Journals / metacog UNION / spark envelope

**Files:**
- Per parent **§9.4–9.6** — schedule after **§9.1–§9.3** consumer paths.

- [ ] **Step 1:** Link implementation milestones to `journal_compose_prompt.j2` and introspection envelopes in dedicated specs.

---

## Plan self-review

| Parent-spec section | Covered by task |
|---------------------|-----------------|
| §6 turn-led flow | Tasks 1–7 |
| Appendix C | Tasks 6–8 |
| Appendix D | Tasks 11–12 |
| Appendix B / §5 ontology | Task 10 |
| §7 compensation | Task 13 |
| §12 items | Tasks 13–17 |
| §9 downstream | Tasks 17–18 |

Placeholder scan: no TBD steps in bridge Tasks 1–8; backlog Tasks 16–18 name explicit follow-on behavior.

---

## Execution handoff

**Plan complete** at `docs/superpowers/plans/2026-05-02-memory-graph-chat-entry-bridge-implementation.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — dispatch a fresh subagent per task; review between tasks.
2. **Inline execution** — run tasks in this session with checkpoints.

Which approach depends on your workflow; both require subagent-driven-development or executing-plans skills during execution.
