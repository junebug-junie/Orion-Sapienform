# Hub REPL Self-Introspection Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hub natural-language self-introspection reliably reach the context-exec (`investigation_v2`) "REPL" lane and answer from live repo/runtime evidence, instead of silently falling back to a `chat_general` reply that grounds a shallow answer on stale conversational memory.

**Architecture:** Fix three real seams found from live logs of the complained-about turn (corr `43d42f69…` ran `mode=brain context_exec_lane=False memory_used=True recall_count=16`). (1) The Hub MODE dropdown is not persisted (only COMPUTE is), so it silently reverts to `grounded_small`/`brain` on reload — the turn never entered the agent lane. (2) Inside `investigation_v2`, `memory` is a grounding source (`STRONG_HIT_SOURCES`) even for repo/runtime/mixed technical questions, so a stale memory hit can produce `answered_grounded` with zero live code/trace evidence. (3) Diagnostic prompts like "what would happen if we changed the orion-hub runtime?" don't trip `_REPO_HINTS`/`_RUNTIME_HINTS`, so the contract classifies them `conceptual` → `not_needed` → the repo/trace probes never run. We do **not** build the stale `interrogate_runtime_state` verb; we reuse the existing `investigation_v2` pipeline.

**Tech Stack:** Python 3.12, Pydantic v2 (`orion-context-exec`), pytest, vanilla JS/localStorage (`orion-hub` static UI).

**Non-goals:** No new `interrogate_runtime_state` verb, no `orion/runtime_interrogation/` package, no `orion-agent-chain` capability pin, no second introspection path, no keyword→mode trigger lists in prompts (structural contract/grounding changes only, per `.cursor/rules/conversational-behavior-anti-slop.mdc`). Runtime-log probe wiring and finalize fail-fast on a slow compute lane are captured as Phase 2 follow-ups, not this plan's core.

---

## Root cause evidence (from live logs, 2026-07-01)

- `hub_ingress_result corr=43d42f69… mode=brain status=success final_len=253 memory_used=True recall_count=16 context_exec_lane=False` — the shallow reply; ran `chat_general`, not the agent lane.
- `hub_ingress_result corr=e50b7df5… mode=agent memory_used=False recall_count=0 context_exec_lane=True` — a later turn did reach the lane, but the `chat` compute lane LLM gateway timed out 2×30s → 140-char deterministic stub (Phase 2 finding).

## File map

| File | Role |
|---|---|
| `services/orion-hub/static/js/app.js` | Persist/restore the Hub MODE selection (`orion_hub_mode`) so `mode=agent` survives reloads |
| `services/orion-context-exec/app/investigation_v2_reducers.py` | Make grounding sources contract-aware: memory does not ground repo/runtime/mixed technical answers |
| `services/orion-context-exec/tests/test_investigation_v2_reducers.py` | Unit tests for contract-aware grounding |
| `orion/cognition/answer_contract_normalize.py` | Route diagnostic/introspection prompts to `mixed_required` so repo+runtime probes fire |
| `services/orion-context-exec/tests/test_investigation_probe_plan.py` | Regression: introspection prompt warrants repo+traces |

---

### Task 1 — Persist the Hub MODE dropdown (stop silent revert to `brain`)

**Files:**
- Modify: `services/orion-hub/static/js/app.js:53` (`currentMode` init), `:9378-9398` (`applyHubModeSelection`), `:9486-9491` (init block)

The MODE dropdown has no persistence; on reload the `<select>` resets to `grounded_small` (`services/orion-hub/templates/index.html:238`) while COMPUTE persists via `localStorage['orion_llm_route']`. Result: a user who selected "Agent" earlier, reloaded, and re-picked compute is silently back on `brain` — the exact turn that produced the shallow memory reply. This is a JS/DOM change; `orion-hub` has no JS unit harness, so it is verified manually (consistent with `docs/superpowers/pr-reports/2026-06-15-hub-mode-compute-dropdowns-pr.md` and the social-room UI task).

- [ ] **Step 1.1: Initialize `currentMode` from localStorage**

In `services/orion-hub/static/js/app.js`, change line 53:

```js
let currentMode = "brain";
```

to:

```js
const HUB_MODE_STORAGE_KEY = 'orion_hub_mode';
let currentMode = "brain";
```

- [ ] **Step 1.2: Persist on selection**

In `applyHubModeSelection` (starts at `:9378`), add the persistence write right after `currentMode = spec.mode;` (line 9381):

```js
  function applyHubModeSelection(modeKey, { silent = false } = {}) {
    const key = String(modeKey || 'grounded_small').trim().toLowerCase();
    const spec = HUB_MODE_SPECS[key] || HUB_MODE_SPECS.grounded_small;
    currentMode = spec.mode;
    try { localStorage.setItem(HUB_MODE_STORAGE_KEY, key); } catch (_err) { /* storage disabled */ }
    modeVerbOverride = spec.verb;
```

- [ ] **Step 1.3: Restore on load**

In the init block (`:9486-9491`), change:

```js
  if (hubModeSelect) {
    hubModeSelect.addEventListener('change', () => {
      applyHubModeSelection(hubModeSelect.value);
    });
    applyHubModeSelection(hubModeSelect.value || 'grounded_small', { silent: true });
  }
```

to:

```js
  if (hubModeSelect) {
    hubModeSelect.addEventListener('change', () => {
      applyHubModeSelection(hubModeSelect.value);
    });
    let storedHubMode = null;
    try { storedHubMode = localStorage.getItem(HUB_MODE_STORAGE_KEY); } catch (_err) { storedHubMode = null; }
    const initialHubMode = (storedHubMode && HUB_MODE_SPECS[storedHubMode]) ? storedHubMode : (hubModeSelect.value || 'grounded_small');
    hubModeSelect.value = initialHubMode;
    applyHubModeSelection(initialHubMode, { silent: true });
  }
```

- [ ] **Step 1.4: Manual verification**

1. Reload Hub, set MODE = Agent, send a message → routing debug shows `context_exec_lane: true`; Hub logs show `hub_ingress_result … mode=agent … context_exec_lane=True`.
2. Reload the page → the MODE dropdown still reads **Agent** (not Grounded Small), and the next send is still `mode=agent`.
3. Set MODE = Grounded Small, reload → it stays Grounded Small (persistence is symmetric, no forced agent).

- [ ] **Step 1.5: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/static/js/app.js
git commit -m "fix(hub-ui): persist MODE selection so agent mode survives reloads"
```

---

### Task 2 — Memory must not ground repo/runtime/mixed technical answers

**Files:**
- Modify: `services/orion-context-exec/app/investigation_v2_reducers.py:446` (constant), `:481-540` (`_compose_answer_status` grounding)
- Test: `services/orion-context-exec/tests/test_investigation_v2_reducers.py`

Today `STRONG_HIT_SOURCES = {"repo", "traces", "memory"}` (`:446`) and grounding counts any strong hit from a warranted source (`:510-511`, `:538-540`). Because `run_memory = bool(perms.read_memory)` is always true for the agent profile, a namespace-memory hit alone yields `grounded_sources=["memory"]` → `answered_grounded`/`partial_grounding` with no live repo/trace evidence. For technical request kinds we require code/runtime evidence to ground; memory may still appear as context (`hit_sources`) but must not by itself satisfy the grounding verdict.

- [ ] **Step 2.1: Write the failing test**

Append to `services/orion-context-exec/tests/test_investigation_v2_reducers.py` (use the file's existing import pattern; if it has none, mirror `test_investigation_probe_plan.py`'s `_ctx_modules()` helper):

```python
def test_memory_hit_alone_does_not_ground_repo_technical_turn() -> None:
    _ctx_modules()
    from app.investigation_v2_reducers import _compose_answer_status, reduce_sections
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        EvidenceBundle,
        SourceResult,
        SourceStatus,
        context_exec_permissions_for_llm_profile,
    )

    request = ContextExecRequestV1(
        text="what would happen if we changed the orion-hub runtime?",
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=AnswerContract(
            request_kind="mixed",
            asks_for_explanation=True,
            requires_repo_grounding=True,
            requires_runtime_grounding=True,
            preferred_render_style="steps",
        ),
    )
    bundle = EvidenceBundle(
        memory=SourceResult(source="memory", status=SourceStatus.hit, summary="prior chat: you were shallow"),
    )
    sections = reduce_sections(bundle, request_text=request.text)
    status, _failed, _blocked, _unavail, grounded, _hits, _sources = _compose_answer_status(bundle, request, sections)
    assert "memory" not in grounded
    assert status != "answered_grounded"


def test_memory_still_grounds_personal_or_conceptual_turn() -> None:
    _ctx_modules()
    from app.investigation_v2_reducers import _compose_answer_status, reduce_sections
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        EvidenceBundle,
        SourceResult,
        SourceStatus,
        context_exec_permissions_for_llm_profile,
    )

    request = ContextExecRequestV1(
        text="what do you remember about me?",
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=AnswerContract(request_kind="personal", asks_for_explanation=True),
    )
    bundle = EvidenceBundle(
        memory=SourceResult(
            source="memory",
            status=SourceStatus.hit,
            summary="known fact",
            findings=["you prefer terse answers"],
        ),
    )
    sections = reduce_sections(bundle, request_text=request.text)
    status, _f, _b, _u, grounded, _h, _s = _compose_answer_status(bundle, request, sections)
    assert "memory" in grounded
```

(`EvidenceBundle`, `SourceResult`, `SourceStatus` are all exported from `orion.schemas.context_exec` — confirmed against `investigation_v2_reducers.py:10-15`.)

- [ ] **Step 2.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_v2_reducers.py -k "memory_hit_alone or memory_still_grounds" -q --tb=short`
Expected: FAIL — `memory` is currently in `grounded` for the `mixed` turn.

- [ ] **Step 2.3: Make grounding sources contract-aware**

In `services/orion-context-exec/app/investigation_v2_reducers.py`, keep the base constant (`:446`) and add a request-scoped resolver just below it:

```python
STRONG_HIT_SOURCES = frozenset({"repo", "traces", "memory"})
IMPORTANT_SOURCES = frozenset({"repo", "traces", "recall", "memory"})

# Request kinds whose answers must be grounded on live code/runtime evidence,
# not stale conversational memory. Memory may still surface as context, but a
# memory hit alone must not satisfy the grounding verdict for these turns.
_EVIDENCE_FIRST_REQUEST_KINDS = frozenset({"repo_technical", "runtime_debug", "mixed"})


def _strong_hit_sources_for(request: ContextExecRequestV1) -> frozenset[str]:
    contract = getattr(request, "answer_contract", None)
    request_kind = str(getattr(contract, "request_kind", "") or "").strip().lower()
    if request_kind in _EVIDENCE_FIRST_REQUEST_KINDS:
        return STRONG_HIT_SOURCES - {"memory"}
    return STRONG_HIT_SOURCES
```

Then, in `_compose_answer_status`, replace the two uses of the module constant with the per-request set. Change the hit loop (`:507-511`):

```python
    strong_sources = _strong_hit_sources_for(request)
    for name, result in _bundle_entries(bundle):
        if result is None:
            continue
        status_key = result.status.value
        sources[name] = status_key
        if result.status == SourceStatus.hit:
            hit_sources.append(name)
            ran_sources.append(name)
            if name in strong_sources:
                grounded_sources.append(name)
```

(The existing warranted-source filter at `:538-540` stays unchanged; it still narrows `grounded_sources` to contract-warranted sources.)

- [ ] **Step 2.4: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_v2_reducers.py -k "memory_hit_alone or memory_still_grounds" -q --tb=short`
Expected: PASS

- [ ] **Step 2.5: Run the full reducers + epistemic suites for regressions**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_v2_reducers.py services/orion-context-exec/tests/test_investigation_v2_epistemic.py services/orion-context-exec/tests/test_investigation_v2.py -q --tb=short`
Expected: PASS (existing personal/conceptual memory-grounding behavior preserved by `_strong_hit_sources_for`).

- [ ] **Step 2.6: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-context-exec/app/investigation_v2_reducers.py services/orion-context-exec/tests/test_investigation_v2_reducers.py
git commit -m "fix(context-exec): memory no longer grounds repo/runtime/mixed technical answers"
```

---

### Task 3 — Route diagnostic/introspection prompts to repo+runtime grounding

**Files:**
- Modify: `orion/cognition/answer_contract_normalize.py:38-65` (hints), `:73-118` (`heuristic_answer_contract`)
- Test: `services/orion-context-exec/tests/test_investigation_probe_plan.py`

"what would happen if we changed the orion-hub runtime?" contains `orion` but not `file`/`import`/`code`, and `runtime` is not in `_RUNTIME_HINTS`, so it classifies `conceptual` → `not_needed` → repo/trace probes never run. Add an explicit introspection/self-diagnostic signal that forces `mixed_required` (repo + runtime grounding). This is a structural contract change, not a prompt keyword→mode trigger.

- [ ] **Step 3.1: Write the failing test**

Append to `services/orion-context-exec/tests/test_investigation_probe_plan.py`:

```python
def test_self_introspection_prompt_warrants_repo_and_traces() -> None:
    _ctx_modules()
    from app.investigation_probe_plan import probe_plan_for_request
    from orion.cognition.answer_contract_normalize import heuristic_answer_contract
    from orion.schemas.context_exec import ContextExecRequestV1, context_exec_permissions_for_llm_profile

    text = "what would happen if we changed the orion-hub runtime?"
    contract = heuristic_answer_contract(text)
    assert contract.request_kind in {"mixed", "repo_technical", "runtime_debug"}
    assert contract.requires_repo_grounding or contract.requires_runtime_grounding

    req = ContextExecRequestV1(
        text=text,
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=contract,
    )
    plan = probe_plan_for_request(req)
    assert plan.investigation_status in {"mixed_required", "repo_required", "runtime_required"}
    assert plan.run_repo is True or plan.run_traces is True


def test_diagnostic_assessment_prompt_is_evidence_first() -> None:
    _ctx_modules()
    from orion.cognition.answer_contract_normalize import heuristic_answer_contract

    contract = heuristic_answer_contract("do a diagnostic assessment of the codebase")
    assert contract.request_kind in {"mixed", "repo_technical"}
    assert contract.requires_repo_grounding is True
```

- [ ] **Step 3.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_probe_plan.py -k "self_introspection or diagnostic_assessment" -q --tb=short`
Expected: FAIL — current classification is `conceptual` / `not_needed`.

- [ ] **Step 3.3: Add an introspection signal and route it to `mixed`**

In `orion/cognition/answer_contract_normalize.py`, add after `_RUNTIME_HINTS` (`:65`):

```python
_INTROSPECTION_HINTS = (
    "runtime",
    "your own code",
    "your codebase",
    "diagnose",
    "diagnostic",
    "introspect",
    "assess the codebase",
    "assess your",
    "how do you route",
    "how are you wired",
    "what happens when you",
    "what would happen if",
    "why did you route",
    "trace your",
    "inspect your",
)
```

Then in `heuristic_answer_contract`, add the `introspective` flag and route it to `mixed` (repo + runtime grounding). Change lines 74-99:

```python
    t = " " + " ".join((user_text or "").lower().split()) + " "
    personal = any(h in t for h in _PERSONAL_HINTS)
    repoish = any(h in t for h in _REPO_HINTS) or ("orion" in t and ("file" in t or "import" in t or "code" in t))
    runtimeish = any(h in t for h in _RUNTIME_HINTS)
    introspective = any(h in t for h in _INTROSPECTION_HINTS)

    if introspective and not (personal and not (repoish or runtimeish)):
        return AnswerContract(
            request_kind="mixed",
            asks_for_explanation=True,
            requires_repo_grounding=True,
            requires_runtime_grounding=True,
            allow_unverified_specifics=False,
            max_unverified_claims=0,
            preferred_render_style="steps",
        )

    if personal and not (repoish or runtimeish or introspective):
        return AnswerContract(
            request_kind="personal",
            asks_for_explanation=True,
            requires_repo_grounding=False,
            requires_runtime_grounding=False,
            requires_user_artifact_grounding=False,
            allow_inference=True,
            allow_unverified_specifics=False,
            max_unverified_claims=0,
            preferred_render_style="answer",
        )

    if repoish and runtimeish:
```

(The remaining `repoish and runtimeish` / `repoish` / `runtimeish` / conceptual branches are unchanged.)

- [ ] **Step 3.4: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_probe_plan.py -k "self_introspection or diagnostic_assessment" -q --tb=short`
Expected: PASS

- [ ] **Step 3.5: Regression — the existing shallow-complaint turn must still skip repo**

The existing `test_conceptual_complaint_skips_repo_probe` ("why do you give shallow responses like this?") must still pass — it has no introspection hint, so it stays `not_needed`.

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_investigation_probe_plan.py -q --tb=short`
Expected: PASS (all, including prior cases). If `test_conceptual_complaint_skips_repo_probe` breaks, tighten `_INTROSPECTION_HINTS` so a bare "shallow responses" complaint does not match.

- [ ] **Step 3.6: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add orion/cognition/answer_contract_normalize.py services/orion-context-exec/tests/test_investigation_probe_plan.py
git commit -m "feat(context-exec): route self-introspection/diagnostic prompts to repo+runtime grounding"
```

---

### Task 4 — Final verification

- [ ] **Step 4.1: Targeted suite**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_investigation_v2_reducers.py \
  services/orion-context-exec/tests/test_investigation_v2_epistemic.py \
  services/orion-context-exec/tests/test_investigation_v2.py \
  services/orion-context-exec/tests/test_investigation_probe_plan.py \
  -q --tb=short
```
Expected: all PASS.

- [ ] **Step 4.2: Compile-check touched Python**

```bash
cd /mnt/scripts/Orion-Sapienform
./orion_dev/bin/python -m compileall -q \
  services/orion-context-exec/app/investigation_v2_reducers.py \
  orion/cognition/answer_contract_normalize.py
```
Expected: exits 0.

- [ ] **Step 4.3: Live acceptance (requires running orion-hub + orion-context-exec stack)**

1. In Hub, set MODE = Agent, COMPUTE = quick (see Phase 2 note re: `chat` lane latency), ask: "what would happen if we changed the orion-hub runtime?"
2. Confirm Hub logs: `hub_ingress_result … mode=agent … context_exec_lane=True`.
3. Confirm context-exec produces an `InvestigationReportV2` whose **repo section cites `repo_grep` hits** (probe actually ran) and whose `answer_status` is not `answered_grounded` solely from a memory hit.
4. Ask the same question again after a stale "you were shallow" memory exists — confirm the answer does not collapse into an apology grounded only on that memory.

---

## Phase 2 follow-ups (out of core scope; separate commits)

- [ ] **Wire the stubbed `runtime_probe`** (`services/orion-context-exec/app/investigation_v2.py:300-314`, currently returns `SourceStatus.skipped` "not wired in PR2"). Concrete first cut: pass the `OrganRuntime` into `runtime_probe` (mirror `health_probe`'s signature) and surface real hop/readiness signal by reusing `check_recall_bus_ready` / `effective_llm_gateway_ready`, returning `hit`/`no_hit`/`unavailable` instead of `skipped`. Requires reading the `run_investigation_v2` call site to update the invocation. This would also have surfaced finding (C).

- [ ] **Fail-fast finalize/synthesis on a slow compute lane** (`services/orion-context-exec/app/llm_tools.py:82`, `settings.py:166` `CONTEXT_EXEC_LLM_TIMEOUT_SEC=30`). On the confirmed agent turn the `chat` route timed out 2×30s → 140-char stub. Options: lower the finalize/synthesis timeout so it degrades quickly to the deterministic report, and/or route the finalize pass through the `quick` lane regardless of the selected COMPUTE lane so a heavy `chat` model can't nuke the user-facing answer. Needs a product call on which lane finalize should use.

---

## Self-review

1. **Spec coverage:** UI mode-persistence (Task 1) ✅; memory-not-grounding technical turns (Task 2) ✅; diagnostic/introspection prompt routing (Task 3) ✅; runtime_probe + finalize latency captured as Phase 2 ✅.
2. **Anti-slop compliance:** No user-message keyword→mode triggers; changes are to the `AnswerContract` classifier and the grounding verdict (structural choke points), per `.cursor/rules/conversational-behavior-anti-slop.mdc`. `_INTROSPECTION_HINTS` selects an *evidence contract*, not a canned reply.
3. **Placeholder scan:** Tasks 1-3 contain concrete code + exact commands; Phase 2 items are explicitly deferred with a named approach, not stand-ins inside core tasks.
4. **Type consistency:** `_strong_hit_sources_for(request)` returns `frozenset[str]`; `_EVIDENCE_FIRST_REQUEST_KINDS` matches `AnswerContract.request_kind` literals (`repo_technical`/`runtime_debug`/`mixed`). `_INTROSPECTION_HINTS` is a `tuple[str, ...]` like the sibling hint tuples.
