# Kids Story Chat Verb — Fast Lane, SQL-Heavy Recall, Firewall-Backed Listeners

**Date:** 2026-05-14  
**Status:** Draft — pending operator review before implementation plan  
**Verb name (canonical):** `chat_kids_story`

---

## Problem

Household storytelling over Orion wants:

- **Same operational shape as `chat_quick`:** single LLM pass after metacog hydration, FAST
  (`quick`) LLM routing, bounded Recall RPC wait, no `chat_general` stance-brief pass (no
  “heavy brain lane” for this use case).
- **Recall without vector/Chroma:** avoid embedding retrieval for latency and operator
  preference; still use **SQL chat + timeline** (and optionally **memory cards**) for
  continuity and structured “listener” facts.
- **No child PII in the public repo:** names, ages, and interest lists live only in
  operator-controlled stores behind the firewall (Recall Postgres / cards / RDF as deployed),
  not in git-tracked prompts or fixtures.

`chat_quick` already implements most of this machinery but is the wrong **semantic** trace
and wrong **default recall profile** contract for a dedicated kids-story lane. Adding a
**sibling verb** reuses behavior without overloading Hub “quick” semantics.

---

## Goals

1. Introduce **`chat_kids_story`**: a **two-step** plan mirroring `chat_quick`
   (`collect_metacog_context` → `llm_chat_kids_story`), with dedicated Jinja and verb-level
   `recall_profile` defaulting to a **vector-off** profile tuned for story continuity.
2. **Execution and LLM routing:** treat `chat_kids_story` identically to `chat_quick` for
   orch **execution lane** (`chat`), exec **quick route**, **metacog short timeout**, **Recall
   bus RPC cap**, **reply-context prep** (lightweight path), and **autonomy fanout** (bounded
   like plain `chat_quick`, not `chat_quick_full_stance`).
3. **Recall profile (in repo):** YAML defines only **knobs** (e.g. `vector_top_k: 0`,
   `sql_chat` / `sql_timeline` enabled, optional `cards_top_k` with narrow fusion weights).
   **No** hardcoded household data.
4. **Recall-backed “listener concept areas” (out of repo):** operators maintain structured
   facts (recommended: **`memory_cards`** with a documented tag/type convention, plus
   **SQL chat/timeline** for prior story turns). Hub may pass **non-PII selectors** (e.g.
   active listener ids) that resolve inside Recall.
5. **Prompt behavior:** Orion voice + playful storyteller depth appropriate for **roughly
   ages 6–9**; **multi-listener fairness** when several listeners are active; for interests
   such as “scary movies,” stories stay **spooky adventure, age-appropriate, no gore or
   adult horror** (honor the interest without violating child-safety tone).

---

## Non-Goals

- Replacing or deprecating `chat_quick` / `chat_general`.
- New GPU pool, new bus channel, or new execution lane type beyond extending existing **verb
  → lane** mapping.
- Mandatory Hub UI in v1 (verb override / API is sufficient); dedicated “Story time” toggle
  can follow.
- Vector/Chroma re-enablement for this verb (operators who want semantic search use another
  verb or explicit profile override outside this design).
- Storing names, ages, or interest strings inside this repository (including tests: use
  **synthetic** fixtures only).

---

## Architecture

### Verb and prompts (git)

| Artifact | Purpose |
|----------|---------|
| `orion/cognition/verbs/chat_kids_story.yaml` | Same step topology as `chat_quick`; `recall_profile` → e.g. `chat.story.kids.v1`; `prompt_template` → kids story Jinja; `personality_file` aligned with household Orion identity (same as quick/general unless a minimal override is introduced later). |
| `orion/cognition/prompts/chat_kids_story.j2` | Instructions for story tone, anchors, digest use, listeners, safety rails; **no** real child data. |
| `orion/recall/profiles/chat.story.kids.v1.yaml` | Vector off; SQL chat + timeline on; optional cards rail with conservative `cards_top_k`; `render_budget_tokens` sized for bedtime latency. |

### Centralized “fast single-pass chat” verb set (exec / orch / recall / fanout)

Introduce a single **frozenset** `FAST_SINGLE_PASS_CHAT_VERBS` containing at least:

- `chat_quick`
- `chat_kids_story`

**Replace** scattered `verb == "chat_quick"` checks that mean “fast interactive chat” with
**membership in this set**, except where semantics are truly quick-only:

- **`chat_quick_full_stance`** and Hub “stance variant” remain **`chat_quick` only**.

Sites that must include the new verb (non-exhaustive; implementation plan will grep):

- `services/orion-cortex-orch/app/execution_lanes.py` — map verb → `chat` lane.
- `services/orion-cortex-exec/app/executor.py` — token caps, metacog timeout, transcript
  compaction for Jinja, LLM route selection, Recall profile resolution path.
- `services/orion-cortex-exec/app/recall_utils.py` — generalize
  `apply_chat_quick_recall_profile_clamp` to **verbs in the frozenset** (rename to
  `apply_fast_chat_recall_profile_clamp` or equivalent); generalize
  `resolve_recall_bus_wait_sec` so **`chat_kids_story`** uses the same lane cap as
  `chat_quick` unless a dedicated setting is introduced.
- `services/orion-cortex-exec/app/router.py` — `prepare_chat_quick_reply_context` (or
  renamed helper) for lightweight reply context when verb is in the set.
- `orion/autonomy/fanout_policy.py` — treat `chat_kids_story` like plain `chat_quick`
  (bounded fanout), not like `chat_quick_full_stance`.

### Settings and env

- **`CHAT_KIDS_STORY_RECALL_PROFILE`** (new): default `chat.story.kids.v1`. Passed into the
  generalized clamp as the default profile for **`chat_kids_story`** (parallel to
  `CHAT_QUICK_RECALL_PROFILE` for `chat_quick`).
- **Recall RPC timeout:** extend the existing **`chat_quick`** cap logic to include
  **`chat_kids_story`** using **`CHAT_QUICK_RECALL_TIMEOUT_SEC`** (no new env key) unless
  product later needs a separate `CHAT_KIDS_STORY_RECALL_TIMEOUT_SEC`.

Document new keys in `services/orion-cortex-exec/.env_example` beside the existing
`CHAT_QUICK_*` entries.

### Hub and tracing

- **v1:** Hub sends `verbs: ["chat_kids_story"]` (or equivalent single-verb override) like
  other explicit verbs; ensure `validate_single_verb_override` and trace metadata accept the
  name.
- **Follow-up:** `services/orion-hub/static/js/app.js` branches that hardcode `chat_quick`
  for audio / lane UI should gain a **small shared helper** or sibling condition for
  `chat_kids_story` so story lane does not regress to wrong verb payloads.

### Verb activation

`orion/cognition/verbs/active.yaml` uses empty default `allow` (all discovered verbs except
`deny`). **No manifest change** required unless a node explicitly uses a non-empty allow list;
then add `chat_kids_story` to that node’s allow list.

---

## Data contract (firewall — not in git)

### Listener / concept areas

Operators define stable listener context using a **documented convention**, for example:

- **Memory cards:** type or tag such as `listener_profile` / `story_listener`, JSON body
  fields like `display_name`, `age_band`, `interests[]`, `notes` — maintained only in
  operator Postgres.
- **Continuity:** prior story sessions appear via **sql_chat** / **sql_timeline** in the
  digest when the user’s anchors and thread align.

### Hub selectors (optional)

Optional request options (names illustrative): `story_active_listener_ids: string[]` —
resolved server-side against Recall; **never** required to contain free-form PII in the
repo’s examples (tests use synthetic ids).

---

## Error handling and degradation

- If Recall times out or returns empty: prompt instructs graceful storytelling from
  **anchors + transcript** only; no apology loop that claims “no memory system.”
- If cards rail errors: existing Recall worker behavior (log, continue with SQL-only) applies;
  profile keeps **cards_top_k** modest so failure is non-fatal.

---

## Testing strategy

- **Contract tests** cloned from `chat_quick` coverage: execution lane, LLM route, Recall RPC
  cap, profile clamp when profile is not explicit, explicit profile preserved.
- **YAML/plumbing test:** `chat_kids_story` plan has two steps in correct order.
- **Prompt test:** template contains required placeholders (`message_history`, recall digest
  hooks) and **child-safety / multi-listener** clauses without embedding real names.
- **Synthetic only:** no real household strings in assertions.

---

## Implementation sequencing (high level)

1. Add recall profile YAML + verb YAML + Jinja prompt.
2. Introduce frozenset and thread through orch, exec, recall_utils, router, fanout_policy.
3. Add settings + `.env_example` + supervisor/router wiring for new profile default.
4. Add tests; run targeted pytest modules touched above.
5. Hub: document verb override; optional JS follow-up in a separate small change.

---

## Open decisions (resolved for this spec)

| Decision | Resolution |
|----------|------------|
| New verb vs overload `chat_quick` | **New verb** `chat_kids_story` for traces and recall defaults. |
| Vector recall | **Off** for default profile (`vector_top_k: 0`). |
| Listener storage | **Recall-backed** (cards + SQL), **not** git. |
| Lane wiring lift | **Bounded:** shared frozenset + extend existing quick paths; no new lane type. |

---

## References (code)

- `orion/cognition/verbs/chat_quick.yaml` — plan topology template.
- `services/orion-cortex-orch/app/execution_lanes.py` — verb → lane.
- `services/orion-cortex-exec/app/executor.py`, `router.py`, `recall_utils.py` — quick special cases.
- `orion/autonomy/fanout_policy.py` — `chat_quick` fanout.
- `orion/recall/profiles/assist.light.v1.yaml` — vector-off quick precedent.
