# Substrate Effect UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Branch:** stay on `feat/repair-pressure-v1`. No worktree. Commit after each task. Push at the very end.

**Goal:** Make the repair_pressure appraisal pipeline visible per chat turn through a small "Substrate Effect" chip below each assistant message that opens a human-readable causal-receipt modal. The operator must be able to answer "what happened, why, with what evidence, what state emerged, and what behavior changed" without reading raw JSON.

**Architecture:** The appraisal package already exists at `orion/substrate/appraisal/` and is purely a library — nothing currently invokes it. This plan (a) wires the appraiser into the HTTP chat handler so a per-turn snapshot is captured, (b) adds a presentation-only view model owned by the backend that translates internal enums into operator-readable strings, (c) exposes one lazy GET endpoint per turn, (d) adds a small isolated JS module that renders the chip from a `substrate_effect_summary` field embedded in the chat response and fetches the full modal payload on click. No React rewrite. No new DB persistence — the per-turn snapshot lives in an in-memory LRU keyed by `turn_id`. The Substrate tab gets a thin "Recent effects" panel as Phase 5; not blocking the per-turn proof.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, FastAPI router (already mounted), plain vanilla JS (Hub is not React), Tailwind classes already in use, existing `style.css`. Reuses `orion.substrate.appraisal.*`, `orion.signals.models`, `orion.mind.substrate_emit`.

---

## Arsonist constraints

These are non-negotiable. Every task below preserves them.

1. **Backend owns translation.** Frontend must not parse `RepairPressureAppraisalV1`, `RepairEvidenceV1`, or `OrionSignalV1` directly. It receives `SubstrateEffectViewV1` with human strings already filled in. The reducer formula and raw evidence vector are NOT exposed in the primary view.
2. **No JSON-first UI.** Raw payload is only available under a collapsed "Developer Payload" section. Every other section uses operator labels. Field names like `dimensions`, `appraisal_kind`, `causal_parents`, `notes` never appear in primary view labels.
3. **Empty effect is a valid view, not 404.** Turns where no evidence fired return a populated `SubstrateEffectViewV1` whose outcome summary plainly says "No substrate effect was recorded for this turn." 404 is reserved for unknown `turn_id`.
4. **Distinguish four states explicitly.** The view always carries: evidence detected (count), appraisal produced (yes/no), signal emitted (yes/no), behavior changed (yes/no). The UI is failing if it conflates them.
5. **Lazy modal fetch.** Chip payload (`substrate_effect_summary`) is small and embedded in the chat response. The full view (`SubstrateEffectViewV1`) is fetched on click. Do not embed the full causal payload in every chat response.
6. **No new molecule kinds, no new signal kinds, no new substrate gradients.** The appraisal package is reused as-is.
7. **In-memory only.** The per-turn snapshot store is an in-memory `OrderedDict` LRU with bounded `max_entries`. No new DB tables, no new bus topics, no new schema_kernel atoms. If the process restarts, history is lost — that is acceptable for v1 because the source of truth lives elsewhere (chat history, signal bus, etc.).
8. **Pipeline failure must not break chat.** Every exception in the appraiser wiring is caught, logged, and the chat response goes out without a chip. Substrate visibility is a courtesy, not a precondition.

---

## File structure

| File | Status | Responsibility |
| --- | --- | --- |
| `orion/substrate/appraisal/view_model.py` | Create | Pydantic `SubstrateEffectViewV1` + supporting models. Label mappers `pressure_label`, `strength_label`, `confidence_label`, `KIND_LABELS`. Builder `build_substrate_effect_view(...)` that consumes a captured snapshot and returns the view. |
| `orion/substrate/appraisal/__init__.py` | Modify | Re-export `SubstrateEffectViewV1`, `build_substrate_effect_view`, and the label helpers. |
| `services/orion-hub/scripts/substrate_effect_cache.py` | Create | In-memory `OrderedDict` keyed by `turn_id` storing `SubstrateEffectSnapshot` (raw appraisal + signal + contract delta + window text). LRU cap. Module-level singleton. Read-only `get`, write `store`, listing `recent(limit)`. |
| `services/orion-hub/scripts/substrate_effect_pipeline.py` | Create | One pure orchestration function `run_substrate_effect_pipeline(turn_id, user_text, source_id, contract_before)` returning `(summary_dict, snapshot)`. Calls `emit_observation` → `select_recent_chat_molecules` → `appraise_repair_pressure` → `repair_appraisal_to_signal` → `apply_repair_pressure_contract`. Catches all internal exceptions, returns a `None` summary on failure. Holds a tiny per-session ring of recent observation molecules so windowing has prior turns to work with. |
| `services/orion-hub/scripts/api_routes.py` | Modify | (a) Call `run_substrate_effect_pipeline` inside `handle_chat_request` after the user prompt is known, before returning the result; merge the summary into the chat result under key `substrate_effect_summary`. (b) Add `GET /api/chat/turn/{turn_id}/substrate-effect`. (c) Add `GET /api/substrate-effect/recent` for Phase 5. |
| `services/orion-hub/static/js/substrate-effect-ui.js` | Create | Vanilla JS module. Exposes `window.SubstrateEffectUI = { renderChip, openModal, init }`. Functions: `renderSubstrateEffectChip`, `openSubstrateEffectModal`, `renderSubstrateEffectModal`, `renderOutcome`, `renderBehaviorDelta`, `renderCausalChain`, `renderEvidenceCards`, `renderScorecard`, `renderMoleculeSummaries`, `renderRawDebug`. |
| `services/orion-hub/static/css/substrate-effect.css` | Create | Scoped styles for the chip, the modal layout, scorecard bars. Loaded from `index.html`. |
| `services/orion-hub/static/js/app.js` | Modify | Inside `appendMessage(...)` assistant branch: if `meta.substrateEffectSummary` (or `meta.substrate_effect_summary`) is present, call `window.SubstrateEffectUI.renderChip(...)` and append the chip to `actionRow` (or a dedicated row below the body). |
| `services/orion-hub/templates/index.html` | Modify | (a) `<link>` the new CSS, (b) `<script>` the new JS module, (c) optionally add a tiny "Recent Substrate Effects" card inside the existing `#substrate` panel for Phase 5. |
| `tests/test_substrate_effect_view_model.py` | Create | High / medium / none / no-raw-dependency cases. |
| `services/orion-hub/tests/test_substrate_effect_cache.py` | Create | LRU eviction, get/store roundtrip, recent ordering. |
| `services/orion-hub/tests/test_substrate_effect_pipeline.py` | Create | High-pressure input produces a summary with `behavior_applied="repair_concrete"`; benign input produces a summary with `changed_behavior=False`; pipeline exception returns `None` summary and does not raise. |
| `services/orion-hub/tests/test_substrate_effect_endpoint.py` | Create | Unknown turn → 404; known turn → 200 with view fields populated; "no effect" turn → 200 with empty-but-valid view. |
| `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py` | Create | `handle_chat_request` result contains `substrate_effect_summary` for a repair-pressure prompt; benign prompt yields a summary with `changed_behavior=False`. |

**Do NOT modify** (architecture-locked):
- `orion/substrate/appraisal/models.py` — appraisal data contract is frozen.
- `orion/substrate/appraisal/evidence.py` / `repair_pressure.py` / `signal_bridge.py` / `contract.py` / `windowing.py` — pure library, do not touch.
- `orion/signals/models.py` / `orion/signals/registry.py` — signal envelope and registry are shared.
- `orion/substrate/molecules.py` — molecule shape is shared.
- `orion/mind/substrate_emit.py` — emit helper already correct.

---

## Shared test helper

Several backend tests build a chat observation molecule. Put this at the top of each test file that needs it.

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.mind.substrate_emit import emit_observation
from orion.substrate.molecules import SubstrateMoleculeV1


def make_chat_observation(text: str, *, source_id: str = "msg-test") -> SubstrateMoleculeV1:
    return emit_observation(surface_text=text, source_id=source_id)
```

---

## Phase 0 — Wire the appraiser into the chat path

### Task 1: In-memory turn snapshot cache

**Files:**
- Create: `services/orion-hub/scripts/substrate_effect_cache.py`
- Test:   `services/orion-hub/tests/test_substrate_effect_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# services/orion-hub/tests/test_substrate_effect_cache.py
from __future__ import annotations

from scripts.substrate_effect_cache import (
    SubstrateEffectCache,
    SubstrateEffectSnapshot,
)


def _make_snapshot(turn_id: str) -> SubstrateEffectSnapshot:
    return SubstrateEffectSnapshot(
        turn_id=turn_id,
        message_id=None,
        user_text="hi",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
        causal_molecule_ids=[],
    )


def test_store_then_get_returns_same_snapshot():
    cache = SubstrateEffectCache(max_entries=4)
    snap = _make_snapshot("t1")
    cache.store(snap)
    assert cache.get("t1") is snap


def test_unknown_turn_returns_none():
    cache = SubstrateEffectCache(max_entries=4)
    assert cache.get("missing") is None


def test_lru_eviction_drops_oldest():
    cache = SubstrateEffectCache(max_entries=2)
    cache.store(_make_snapshot("a"))
    cache.store(_make_snapshot("b"))
    cache.store(_make_snapshot("c"))
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_recent_returns_newest_first():
    cache = SubstrateEffectCache(max_entries=4)
    cache.store(_make_snapshot("a"))
    cache.store(_make_snapshot("b"))
    cache.store(_make_snapshot("c"))
    recent_ids = [s.turn_id for s in cache.recent(limit=10)]
    assert recent_ids == ["c", "b", "a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.substrate_effect_cache'`

- [ ] **Step 3: Implement the cache**

```python
# services/orion-hub/scripts/substrate_effect_cache.py
"""In-memory LRU cache of per-turn substrate effect snapshots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.signals.models import OrionSignalV1
from orion.substrate.appraisal.models import (
    RepairEvidenceV1,
    RepairPressureAppraisalV1,
)


@dataclass
class SubstrateEffectSnapshot:
    turn_id: str
    message_id: str | None
    user_text: str
    appraisal: RepairPressureAppraisalV1 | None
    signal: OrionSignalV1 | None
    evidence: list[RepairEvidenceV1] = field(default_factory=list)
    contract_before: dict[str, Any] = field(default_factory=dict)
    contract_after: dict[str, Any] = field(default_factory=dict)
    causal_molecule_ids: list[str] = field(default_factory=list)
    stored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SubstrateEffectCache:
    def __init__(self, *, max_entries: int = 256) -> None:
        self._max = max(1, int(max_entries))
        self._entries: "OrderedDict[str, SubstrateEffectSnapshot]" = OrderedDict()

    def store(self, snapshot: SubstrateEffectSnapshot) -> None:
        if snapshot.turn_id in self._entries:
            self._entries.move_to_end(snapshot.turn_id)
        self._entries[snapshot.turn_id] = snapshot
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def get(self, turn_id: str) -> SubstrateEffectSnapshot | None:
        return self._entries.get(turn_id)

    def recent(self, *, limit: int = 25) -> list[SubstrateEffectSnapshot]:
        items = list(self._entries.values())
        items.reverse()  # newest first
        return items[: max(0, int(limit))]


# Module-level singleton used by the Hub.
substrate_effect_cache = SubstrateEffectCache(max_entries=256)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_cache.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/substrate_effect_cache.py services/orion-hub/tests/test_substrate_effect_cache.py
git commit -m "feat(hub): in-memory LRU cache for substrate effect snapshots"
```

---

### Task 2: Chat-turn appraisal pipeline

**Files:**
- Create: `services/orion-hub/scripts/substrate_effect_pipeline.py`
- Test:   `services/orion-hub/tests/test_substrate_effect_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# services/orion-hub/tests/test_substrate_effect_pipeline.py
from __future__ import annotations

from scripts.substrate_effect_cache import SubstrateEffectCache
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


def test_high_pressure_prompt_yields_repair_concrete_summary():
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-A",
        message_id=None,
        user_text=(
            "you gave me garbage directions, again. stop being vague — "
            "build me a design spec for claude, nuts and bolts, "
            "arsonist pov only"
        ),
        source_id="conv-A",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is not None
    assert summary["appraisal_kind"] == "repair_pressure"
    assert summary["level_label"] in {"HIGH", "MEDIUM"}
    assert summary["evidence_count"] >= 3
    assert summary["changed_behavior"] is True
    assert summary["behavior_applied"] in {"repair_concrete", "concrete_bias"}
    assert cache.get("turn-A") is snapshot
    assert snapshot.contract_after["mode"] in {"repair_concrete", "concrete_bias"}


def test_benign_prompt_yields_no_behavior_change():
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-B",
        message_id=None,
        user_text="what's the weather like in Paris?",
        source_id="conv-B",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is not None
    assert summary["changed_behavior"] is False
    assert summary["behavior_applied"] is None
    assert snapshot.contract_after.get("mode") == "default"


def test_pipeline_handles_internal_failure_without_raising(monkeypatch):
    import scripts.substrate_effect_pipeline as mod

    def boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("forced")

    monkeypatch.setattr(mod, "appraise_repair_pressure", boom)
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-C",
        message_id=None,
        user_text="anything",
        source_id="conv-C",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is None
    assert snapshot is None
    assert cache.get("turn-C") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the pipeline**

```python
# services/orion-hub/scripts/substrate_effect_pipeline.py
"""Orchestrate the repair_pressure appraisal pipeline for one chat turn.

Failure must never propagate to the chat handler. On any exception we log
and return (None, None). The chat response then ships without a chip.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal import (
    REPAIR_PRESSURE_DEBUG_KEY,
    apply_repair_pressure_contract,
    appraise_repair_pressure,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)
from orion.substrate.molecules import SubstrateMoleculeV1

from .substrate_effect_cache import (
    SubstrateEffectCache,
    SubstrateEffectSnapshot,
    substrate_effect_cache,
)

logger = logging.getLogger("orion-hub.substrate_effect_pipeline")

# Per-source rolling buffer of observation molecules. Keyed by `source_id`
# (we use session_id or correlation_id). Bounded so memory stays flat.
_RECENT_OBSERVATIONS: dict[str, deque[SubstrateMoleculeV1]] = {}
_RECENT_MAX = 32


def _push_observation(source_id: str, mol: SubstrateMoleculeV1) -> list[SubstrateMoleculeV1]:
    buf = _RECENT_OBSERVATIONS.setdefault(source_id, deque(maxlen=_RECENT_MAX))
    buf.append(mol)
    return list(buf)


def _summary_dict(
    *,
    turn_id: str,
    appraisal,
    signal,
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
    evidence_count: int,
) -> dict[str, Any]:
    from orion.substrate.appraisal.view_model import pressure_label

    level = float(appraisal.dimensions.get("level", 0.0)) if appraisal else 0.0
    confidence = float(appraisal.confidence) if appraisal else 0.0
    level_lbl = pressure_label(level)
    before_mode = str(contract_before.get("mode") or "")
    after_mode = str(contract_after.get("mode") or "")
    changed = before_mode != after_mode
    behavior_applied = after_mode if changed else None
    chip_label = (
        f"{behavior_applied or 'no behavior change'} · "
        f"{level_lbl} repair pressure · "
        f"{evidence_count} evidence driver{'s' if evidence_count != 1 else ''}"
    )
    return {
        "turn_id": turn_id,
        "appraisal_kind": "repair_pressure" if appraisal else "none",
        "level": level,
        "level_label": level_lbl,
        "confidence": confidence,
        "behavior_applied": behavior_applied,
        "evidence_count": evidence_count,
        "changed_behavior": changed,
        "chip_label": chip_label,
    }


def run_substrate_effect_pipeline(
    *,
    turn_id: str,
    message_id: str | None,
    user_text: str,
    source_id: str,
    contract_before: dict[str, Any],
    cache: SubstrateEffectCache | None = None,
) -> tuple[dict[str, Any] | None, SubstrateEffectSnapshot | None]:
    """Run the appraiser end-to-end. Stash a snapshot in `cache`. Return summary."""

    store_in = cache if cache is not None else substrate_effect_cache
    try:
        if not (user_text or "").strip():
            return None, None

        mol = emit_observation(surface_text=user_text, source_id=source_id)
        window = select_recent_chat_molecules(_push_observation(source_id, mol), source_id=source_id)
        evidence = extract_repair_evidence(window)
        appraisal = appraise_repair_pressure(window, window_id=f"win-{turn_id}")
        signal = repair_appraisal_to_signal(appraisal)
        contract_after = apply_repair_pressure_contract(contract_before, signal)

        snapshot = SubstrateEffectSnapshot(
            turn_id=turn_id,
            message_id=message_id,
            user_text=user_text,
            appraisal=appraisal,
            signal=signal,
            evidence=list(evidence),
            contract_before=dict(contract_before),
            contract_after=dict(contract_after),
            causal_molecule_ids=list(appraisal.causal_molecule_ids),
        )
        store_in.store(snapshot)

        summary = _summary_dict(
            turn_id=turn_id,
            appraisal=appraisal,
            signal=signal,
            contract_before=contract_before,
            contract_after=contract_after,
            evidence_count=len(evidence),
        )
        return summary, snapshot
    except Exception:  # noqa: BLE001
        logger.warning("substrate_effect_pipeline_failed turn_id=%s", turn_id, exc_info=True)
        return None, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_pipeline.py -v`
Expected: 3 PASS.

Note: this step imports `orion.substrate.appraisal.view_model.pressure_label`, which Task 4 will create. Until then, replace the import with a local copy. The cleaner approach is to commit Task 2 first using a private inline label function and switch to the shared helper in Task 4. Use this temporary inline function:

```python
def _tmp_pressure_label(value: float) -> str:
    if value >= 0.75: return "HIGH"
    if value >= 0.45: return "MEDIUM"
    if value >= 0.25: return "LOW"
    return "NONE"
```

Replace `pressure_label(level)` with `_tmp_pressure_label(level)`. Remove the import. Switch back in Task 4.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/substrate_effect_pipeline.py services/orion-hub/tests/test_substrate_effect_pipeline.py
git commit -m "feat(hub): per-turn substrate effect pipeline with failure isolation"
```

---

### Task 3: Wire pipeline into the HTTP chat handler

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py` (function `handle_chat_request`)
- Test:   `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py`

- [ ] **Step 1: Write the failing test**

```python
# services/orion-hub/tests/test_handle_chat_request_substrate_effect.py
from __future__ import annotations

import asyncio

import pytest

from scripts import api_routes
from scripts.substrate_effect_cache import substrate_effect_cache


class _FakeCortexResult:
    def __init__(self, text: str, correlation_id: str) -> None:
        self.text = text
        self.correlation_id = correlation_id
        self.mode = "brain"
        self.use_recall = True
        self.routing_debug = {}
        self.raw = {"metadata": {}}
        self.memory_digest = None
        self.metacog_traces = []


class _FakeCortex:
    async def chat(self, req, correlation_id: str):  # noqa: ANN001
        return _FakeCortexResult(text="ok", correlation_id=correlation_id)


@pytest.fixture(autouse=True)
def _reset_cache():
    # Hub uses a process-singleton cache; isolate per-test.
    substrate_effect_cache._entries.clear()  # type: ignore[attr-defined]
    yield


def _make_payload(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_handle_chat_request_attaches_repair_summary_for_high_pressure(monkeypatch):
    # Bypass the rest of the gateway path; force the cortex layer to be a stub.
    monkeypatch.setattr(api_routes, "validate_single_verb_override", lambda *a, **k: None)
    cortex = _FakeCortex()
    payload = _make_payload(
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    result = asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-x", no_write=True))
    summary = result.get("substrate_effect_summary")
    assert summary is not None, "chat result must carry substrate_effect_summary"
    assert summary["appraisal_kind"] == "repair_pressure"
    assert isinstance(summary["chip_label"], str) and summary["chip_label"]
    assert summary["evidence_count"] >= 1


def test_handle_chat_request_summary_marks_no_change_for_benign(monkeypatch):
    monkeypatch.setattr(api_routes, "validate_single_verb_override", lambda *a, **k: None)
    cortex = _FakeCortex()
    payload = _make_payload("what's the weather like in Paris?")
    result = asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-y", no_write=True))
    summary = result.get("substrate_effect_summary")
    assert summary is not None
    assert summary["changed_behavior"] is False
    assert summary["behavior_applied"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/orion-hub && pytest tests/test_handle_chat_request_substrate_effect.py -v`
Expected: FAIL (`substrate_effect_summary` missing from result).

- [ ] **Step 3: Wire pipeline into `handle_chat_request`**

Open `services/orion-hub/scripts/api_routes.py`. Near the existing imports (top of file), add:

```python
from .substrate_effect_pipeline import run_substrate_effect_pipeline
```

Inside `handle_chat_request(cortex_client, payload, session_id, no_write)` — right after `corr_id = str(uuid4())` and before the call to `cortex_client.chat(...)` — extract the user prompt and run the pipeline. Use `session_id` as the `source_id` so the rolling observation buffer is per-session. Use the freshly minted `corr_id` as the `turn_id` since the HTTP path already uses it for that purpose downstream.

```python
        # ─── Substrate effect (best-effort, never blocks chat) ───────────────
        substrate_summary, _ = run_substrate_effect_pipeline(
            turn_id=corr_id,
            message_id=None,
            user_text=user_prompt,
            source_id=session_id,
            contract_before={"mode": "default"},
        )
```

Place that block before `resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=corr_id)`. After the chat reply has been assembled into the return dict (the function builds and returns a dict near the end), set:

```python
        if substrate_summary is not None:
            result["substrate_effect_summary"] = substrate_summary
```

Where `result` is the dict the function returns. If the existing implementation returns a literal dict rather than a `result` variable, lift it into a variable first, then assign, then `return result`. Make this the only edit to the return path.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-hub && pytest tests/test_handle_chat_request_substrate_effect.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run the broader chat handler tests to check no regression**

Run: `cd services/orion-hub && pytest tests/test_handle_chat_request_turn_effect.py tests/test_http_chat_spark_meta.py -v`
Expected: PASS (or skipped if env-dependent).

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_handle_chat_request_substrate_effect.py
git commit -m "feat(hub): wire repair_pressure pipeline into HTTP chat handler"
```

---

## Phase 1 — Backend view model

### Task 4: Label mappers and view-model schemas

**Files:**
- Create: `orion/substrate/appraisal/view_model.py`
- Modify: `orion/substrate/appraisal/__init__.py`
- Test:   `tests/test_substrate_effect_view_model.py` (label helpers section only — builder tests follow in Task 5)

- [ ] **Step 1: Write the failing tests for label mappers**

```python
# tests/test_substrate_effect_view_model.py
from __future__ import annotations

from orion.substrate.appraisal.view_model import (
    KIND_LABELS,
    confidence_label,
    pressure_label,
    strength_label,
)


def test_pressure_label_buckets():
    assert pressure_label(0.90) == "HIGH"
    assert pressure_label(0.75) == "HIGH"
    assert pressure_label(0.50) == "MEDIUM"
    assert pressure_label(0.30) == "LOW"
    assert pressure_label(0.10) == "NONE"


def test_strength_label_buckets():
    assert strength_label(0.95) == "Very strong"
    assert strength_label(0.70) == "Strong"
    assert strength_label(0.50) == "Medium"
    assert strength_label(0.30) == "Low"
    assert strength_label(0.10) == "Very low"


def test_confidence_label_buckets():
    assert confidence_label(0.95) == "Very high"
    assert confidence_label(0.70) == "High"
    assert confidence_label(0.50) == "Medium"
    assert confidence_label(0.30) == "Low"
    assert confidence_label(0.10) == "Very low"


def test_kind_labels_translate_internal_enums():
    assert KIND_LABELS["specificity_demand"] == "Specificity demand"
    assert KIND_LABELS["trust_rupture"] == "Trust rupture"
    assert KIND_LABELS["repair_pressure"] == "Repair pressure"
    assert KIND_LABELS["repair_concrete"] == "Repair concrete mode"
    assert KIND_LABELS["normal_chat"] == "Normal chat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_substrate_effect_view_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the view-model module**

```python
# orion/substrate/appraisal/view_model.py
"""Presentation-only view model for the Substrate Effect UI.

Lives next to the appraiser so backend owns translation.  Frontend renders
this view as-is; it must not re-derive labels from raw appraisal/signal
objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.signals.models import OrionSignalV1

from .contract import REPAIR_PRESSURE_DEBUG_KEY
from .models import RepairEvidenceV1, RepairPressureAppraisalV1


# ── Label maps ──────────────────────────────────────────────────────────

KIND_LABELS: dict[str, str] = {
    "repair_pressure": "Repair pressure",
    "specificity_demand": "Specificity demand",
    "trust_rupture": "Trust rupture",
    "coherence_gap": "Coherence gap",
    "repetition_failure": "Repetition failure",
    "operational_block": "Operational block",
    "explicit_repair_command": "Explicit repair command",
    "assistant_accountability_demand": "Assistant accountability demand",
    "salience": "Substrate salience",
    "contradiction": "Substrate contradiction",
    "coherence": "Substrate coherence",
    "novelty": "Substrate novelty",
    "level": "Level",
    "confidence": "Confidence",
    "repair_concrete": "Repair concrete mode",
    "concrete_bias": "Concrete bias",
    "normal_chat": "Normal chat",
    "none": "None",
}


def pressure_label(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MEDIUM"
    if value >= 0.25:
        return "LOW"
    return "NONE"


def strength_label(value: float) -> str:
    if value >= 0.85:
        return "Very strong"
    if value >= 0.65:
        return "Strong"
    if value >= 0.45:
        return "Medium"
    if value >= 0.25:
        return "Low"
    return "Very low"


def confidence_label(value: float) -> str:
    if value >= 0.85:
        return "Very high"
    if value >= 0.65:
        return "High"
    if value >= 0.45:
        return "Medium"
    if value >= 0.25:
        return "Low"
    return "Very low"


# ── View-model schemas ──────────────────────────────────────────────────


class SubstrateOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appraisal_kind: str
    level: float
    level_label: str
    confidence: float
    confidence_label: str
    behavior_applied: str | None = None
    summary: str


class BehaviorDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_before: str | None = None
    contract_after: str | None = None
    changed: bool
    rules_activated: list[str] = Field(default_factory=list)
    explanation: str | None = None


class CausalChainStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    title: str
    description: str
    detail: str | None = None
    linked_ids: list[str] = Field(default_factory=list)


class EvidenceCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_kind: str
    label: str
    strength_label: str
    score: float
    confidence: float
    source_span: str | None = None
    explanation: str
    meaning: str
    source_molecule_id: str | None = None


class ScorecardItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    label: str
    value: float
    value_label: str | None = None
    contribution: str | None = None


class ScorecardV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    items: list[ScorecardItemV1]
    final_label: str
    explanation: str | None = None


class MoleculeSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    molecule_id: str
    label: str
    explanation: str
    molecule_kind: str
    provenance_label: str | None = None


class SubstrateEffectViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_id: str
    message_id: str | None = None
    outcome: SubstrateOutcomeV1
    why: str | None = None
    behavior_delta: BehaviorDeltaV1 | None = None
    causal_chain: list[CausalChainStepV1] = Field(default_factory=list)
    evidence_cards: list[EvidenceCardV1] = Field(default_factory=list)
    scorecard: ScorecardV1 | None = None
    molecule_summaries: list[MoleculeSummaryV1] = Field(default_factory=list)
    raw_debug: dict[str, Any] | None = None
```

- [ ] **Step 4: Re-export from package `__init__.py`**

Open `orion/substrate/appraisal/__init__.py` and extend the imports + `__all__`:

```python
from .view_model import (
    KIND_LABELS,
    SubstrateEffectViewV1,
    SubstrateOutcomeV1,
    BehaviorDeltaV1,
    CausalChainStepV1,
    EvidenceCardV1,
    MoleculeSummaryV1,
    ScorecardV1,
    ScorecardItemV1,
    confidence_label,
    pressure_label,
    strength_label,
)
```

Add each of those names to `__all__`.

- [ ] **Step 5: Switch the pipeline back to the shared helper**

In `services/orion-hub/scripts/substrate_effect_pipeline.py`, remove `_tmp_pressure_label` and import `pressure_label` from `orion.substrate.appraisal.view_model`. Replace the call site accordingly.

- [ ] **Step 6: Run tests to verify**

Run: `pytest tests/test_substrate_effect_view_model.py -v`
Expected: 4 PASS.

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_pipeline.py -v`
Expected: still 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add orion/substrate/appraisal/view_model.py orion/substrate/appraisal/__init__.py \
        services/orion-hub/scripts/substrate_effect_pipeline.py \
        tests/test_substrate_effect_view_model.py
git commit -m "feat(substrate): SubstrateEffectViewV1 schemas and label mappers"
```

---

### Task 5: View-model builder

**Files:**
- Modify: `orion/substrate/appraisal/view_model.py`
- Test:   `tests/test_substrate_effect_view_model.py` (append builder tests)

- [ ] **Step 1: Append failing builder tests**

Append to `tests/test_substrate_effect_view_model.py`:

```python
from orion.substrate.appraisal.view_model import build_substrate_effect_view
from orion.substrate.appraisal import (
    appraise_repair_pressure,
    apply_repair_pressure_contract,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)
from orion.mind.substrate_emit import emit_observation


def _run_pipeline(text: str, *, source_id: str = "src"):
    mol = emit_observation(surface_text=text, source_id=source_id)
    window = select_recent_chat_molecules([mol], source_id=source_id)
    appraisal = appraise_repair_pressure(window, window_id="w")
    signal = repair_appraisal_to_signal(appraisal)
    contract_before = {"mode": "default"}
    contract_after = apply_repair_pressure_contract(contract_before, signal)
    evidence = extract_repair_evidence(window)
    return appraisal, signal, evidence, contract_before, contract_after


def test_high_pressure_view_carries_repair_concrete_delta():
    appraisal, signal, evidence, before, after = _run_pipeline(
        "you gave me garbage directions, stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    view = build_substrate_effect_view(
        turn_id="t-high",
        message_id="m-1",
        user_text="...",
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        contract_before=before,
        contract_after=after,
    )
    assert view.outcome.level_label == "HIGH"
    assert view.behavior_delta is not None
    assert view.behavior_delta.changed is True
    assert view.behavior_delta.contract_after == "repair_concrete"
    assert any("Behavior changed" in step.title for step in view.causal_chain)
    assert view.evidence_cards, "evidence cards must populate"
    labels = [card.label for card in view.evidence_cards]
    assert any(label and label[0].isupper() for label in labels)
    # Frontend must not need raw_debug for primary fields.
    primary = view.model_dump(exclude={"raw_debug"})
    assert primary["outcome"]["summary"]
    assert primary["behavior_delta"]["explanation"]


def test_medium_view_does_not_overstate():
    # craft a prompt that scores in the 0.45–0.74 band — one strong specificity
    # phrase, no trust_rupture/coherence_gap noise.
    appraisal, signal, evidence, before, after = _run_pipeline(
        "give me a concrete design spec for the next step"
    )
    view = build_substrate_effect_view(
        turn_id="t-med",
        message_id=None,
        user_text="...",
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        contract_before=before,
        contract_after=after,
    )
    # Don't lock the exact band; just assert the labels are coherent.
    assert view.outcome.level_label in {"LOW", "MEDIUM", "HIGH"}
    if view.outcome.level_label == "MEDIUM":
        assert view.outcome.behavior_applied in {"concrete_bias", None}


def test_no_effect_returns_valid_empty_view():
    view = build_substrate_effect_view(
        turn_id="t-empty",
        message_id=None,
        user_text="",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
    )
    assert view.outcome.appraisal_kind == "none"
    assert view.outcome.level == 0.0
    assert view.outcome.level_label == "NONE"
    assert view.outcome.behavior_applied is None
    assert "No substrate effect" in view.outcome.summary
    assert view.causal_chain == []
    assert view.evidence_cards == []
    assert view.molecule_summaries == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_substrate_effect_view_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_substrate_effect_view'`.

- [ ] **Step 3: Implement the builder**

Append to `orion/substrate/appraisal/view_model.py`:

```python
# ── Builder ────────────────────────────────────────────────────────────────

_EVIDENCE_MEANING: dict[str, str] = {
    "specificity_demand": (
        "The response should stop exploring and produce a usable implementation handoff."
    ),
    "trust_rupture": (
        "The user is signalling that prior assistant output was unreliable; "
        "the next response should acknowledge this and not repeat the failure mode."
    ),
    "coherence_gap": (
        "The response has been drifting or contradicting itself; "
        "the next response should converge to one explicit stance."
    ),
    "repetition_failure": (
        "The user is repeating a request the assistant has already received; "
        "the next response should address it directly instead of restating context."
    ),
    "operational_block": (
        "The user needs the answer to plug into another builder or pipeline; "
        "the answer must be structured enough to hand off directly."
    ),
    "explicit_repair_command": (
        "The user is constraining the response style explicitly; "
        "the next response must obey that constraint, not negotiate it."
    ),
    "assistant_accountability_demand": (
        "The user is holding the assistant accountable for prior turns; "
        "the next response should briefly acknowledge that, not deflect."
    ),
}


def _evidence_explanation(ev: RepairEvidenceV1) -> str:
    label = KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind)
    if ev.span:
        return f"Detected {label.lower()} from: \"{ev.span}\""
    return f"Detected {label.lower()} in the recent chat window."


def _evidence_card_label(ev: RepairEvidenceV1) -> str:
    return f"{KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind)} — {strength_label(ev.score)}"


def _build_outcome(
    appraisal: RepairPressureAppraisalV1 | None,
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
) -> SubstrateOutcomeV1:
    if appraisal is None:
        return SubstrateOutcomeV1(
            appraisal_kind="none",
            level=0.0,
            level_label="NONE",
            confidence=0.0,
            confidence_label="Very low",
            behavior_applied=None,
            summary="No substrate effect was recorded for this turn.",
        )
    level = float(appraisal.dimensions.get("level", 0.0))
    confidence = float(appraisal.confidence)
    before_mode = str(contract_before.get("mode") or "")
    after_mode = str(contract_after.get("mode") or "")
    changed = before_mode != after_mode
    behavior_applied = after_mode if changed else None
    lvl_lbl = pressure_label(level)
    if behavior_applied:
        summary = (
            f"Repair pressure was {lvl_lbl}, so Orion switched into "
            f"{KIND_LABELS.get(behavior_applied, behavior_applied)}."
        )
    else:
        summary = (
            f"Repair pressure was {lvl_lbl}. Orion did not change the response contract."
        )
    return SubstrateOutcomeV1(
        appraisal_kind="repair_pressure",
        level=level,
        level_label=lvl_lbl,
        confidence=confidence,
        confidence_label=confidence_label(confidence),
        behavior_applied=behavior_applied,
        summary=summary,
    )


def _build_behavior_delta(
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
) -> BehaviorDeltaV1:
    before_mode = contract_before.get("mode")
    after_mode = contract_after.get("mode")
    changed = before_mode != after_mode
    rules = list(contract_after.get("rules") or []) if changed else []
    if changed:
        explanation = (
            "The response contract was switched because repair pressure crossed "
            "the threshold defined by apply_repair_pressure_contract."
        )
    else:
        explanation = (
            "No response contract change was applied because repair pressure was "
            "below threshold."
        )
    return BehaviorDeltaV1(
        contract_before=str(before_mode) if before_mode is not None else None,
        contract_after=str(after_mode) if after_mode is not None else None,
        changed=changed,
        rules_activated=rules,
        explanation=explanation,
    )


def _build_why(evidence: list[RepairEvidenceV1]) -> str | None:
    if not evidence:
        return None
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)[:3]
    fragments: list[str] = []
    for ev in ranked:
        label = KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind).lower()
        if ev.span:
            fragments.append(f"{label} (\"{ev.span}\")")
        else:
            fragments.append(label)
    return "Detected: " + "; ".join(fragments) + "."


def _build_causal_chain(
    user_text: str,
    appraisal: RepairPressureAppraisalV1 | None,
    signal: OrionSignalV1 | None,
    evidence: list[RepairEvidenceV1],
    behavior_changed: bool,
    behavior_applied: str | None,
) -> list[CausalChainStepV1]:
    if appraisal is None:
        return []
    steps: list[CausalChainStepV1] = [
        CausalChainStepV1(
            index=1,
            title="Chat turn observed",
            description=(user_text[:160] + "…") if len(user_text) > 160 else user_text,
        ),
        CausalChainStepV1(
            index=2,
            title="Substrate created an observation molecule",
            description="This turn became shared substrate evidence.",
            linked_ids=list(appraisal.causal_molecule_ids),
        ),
    ]
    if evidence:
        top = sorted(evidence, key=lambda e: e.score, reverse=True)[:3]
        bullets = [
            f"{strength_label(e.score)} {KIND_LABELS.get(e.evidence_kind, e.evidence_kind).lower()}"
            for e in top
        ]
        steps.append(
            CausalChainStepV1(
                index=3,
                title="Repair evidence was detected",
                description="; ".join(bullets) if bullets else "no evidence",
            )
        )
    steps.append(
        CausalChainStepV1(
            index=len(steps) + 1,
            title="Appraiser reduced the evidence",
            description=(
                f"repair_pressure level={appraisal.dimensions.get('level', 0.0):.2f}, "
                f"confidence={appraisal.confidence:.2f}"
            ),
            linked_ids=[appraisal.appraisal_id],
        )
    )
    if signal is not None:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Signal emitted",
                description=f"{signal.organ_id} / {signal.signal_kind}",
                linked_ids=[signal.signal_id],
            )
        )
    if behavior_changed:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Behavior changed",
                description=(
                    f"The response used {KIND_LABELS.get(behavior_applied or '', behavior_applied or 'updated')}."
                ),
            )
        )
    else:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Behavior unchanged",
                description="No response contract switch was applied.",
            )
        )
    return steps


def _build_evidence_cards(evidence: list[RepairEvidenceV1]) -> list[EvidenceCardV1]:
    cards: list[EvidenceCardV1] = []
    for ev in evidence:
        cards.append(
            EvidenceCardV1(
                evidence_kind=ev.evidence_kind,
                label=_evidence_card_label(ev),
                strength_label=strength_label(ev.score),
                score=float(ev.score),
                confidence=float(ev.confidence),
                source_span=ev.span,
                explanation=_evidence_explanation(ev),
                meaning=_EVIDENCE_MEANING.get(ev.evidence_kind, ""),
                source_molecule_id=ev.source_molecule_id,
            )
        )
    return cards


_SCORECARD_KEYS: tuple[str, ...] = (
    "specificity_demand",
    "operational_block",
    "explicit_repair_command",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "assistant_accountability_demand",
    "salience",
    "contradiction",
    "coherence",
)


def _build_scorecard(appraisal: RepairPressureAppraisalV1 | None) -> ScorecardV1 | None:
    if appraisal is None:
        return None
    items: list[ScorecardItemV1] = []
    for key in _SCORECARD_KEYS:
        value = float(appraisal.dimensions.get(key, 0.0))
        items.append(
            ScorecardItemV1(
                key=key,
                label=KIND_LABELS.get(key, key),
                value=value,
                value_label=strength_label(value),
            )
        )
    items.sort(key=lambda item: item.value, reverse=True)
    level = float(appraisal.dimensions.get("level", 0.0))
    final = pressure_label(level)
    top_two = [item.label for item in items[:2] if item.value > 0.0]
    if top_two:
        explanation = (
            f"The score was {final.lower()} mostly because "
            + " and ".join(top_two)
            + " were the strongest contributors."
        )
    else:
        explanation = "No dimension exceeded zero."
    return ScorecardV1(
        title="Repair Pressure Scorecard",
        items=items,
        final_label=f"Repair pressure is {final}.",
        explanation=explanation,
    )


def _molecule_label(mol_id: str, evidence: list[RepairEvidenceV1]) -> str:
    if any(ev.source_molecule_id == mol_id for ev in evidence):
        return "Repair evidence"
    return "Chat observation"


def _molecule_explanation(mol_id: str, evidence: list[RepairEvidenceV1]) -> str:
    hits = [ev for ev in evidence if ev.source_molecule_id == mol_id]
    if not hits:
        return "Created from the current user turn."
    kinds = [KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind) for ev in hits]
    spans = [f'"{ev.span}"' for ev in hits if ev.span]
    if spans:
        return f"Captured {', '.join(kinds).lower()} from {spans[0]}."
    return f"Captured {', '.join(kinds).lower()}."


def _build_molecule_summaries(
    appraisal: RepairPressureAppraisalV1 | None,
    evidence: list[RepairEvidenceV1],
) -> list[MoleculeSummaryV1]:
    if appraisal is None:
        return []
    seen: set[str] = set()
    summaries: list[MoleculeSummaryV1] = []
    for mol_id in appraisal.causal_molecule_ids:
        if mol_id in seen:
            continue
        seen.add(mol_id)
        summaries.append(
            MoleculeSummaryV1(
                molecule_id=mol_id,
                label=_molecule_label(mol_id, evidence),
                explanation=_molecule_explanation(mol_id, evidence),
                molecule_kind="observation",
                provenance_label="chat_turn",
            )
        )
    return summaries


def build_substrate_effect_view(
    *,
    turn_id: str,
    message_id: str | None,
    user_text: str,
    appraisal: RepairPressureAppraisalV1 | None,
    signal: OrionSignalV1 | None,
    evidence: list[RepairEvidenceV1],
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
    include_raw_debug: bool = True,
) -> SubstrateEffectViewV1:
    outcome = _build_outcome(appraisal, contract_before, contract_after)
    delta = _build_behavior_delta(contract_before, contract_after)
    chain = _build_causal_chain(
        user_text=user_text,
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        behavior_changed=delta.changed,
        behavior_applied=outcome.behavior_applied,
    )
    raw_debug: dict[str, Any] | None = None
    if include_raw_debug:
        raw_debug = {
            "appraisal": appraisal.model_dump(mode="json") if appraisal else None,
            "signal": signal.model_dump(mode="json") if signal else None,
            "evidence": [ev.model_dump(mode="json") for ev in evidence],
            "contract_before": contract_before,
            "contract_after": contract_after,
        }
    return SubstrateEffectViewV1(
        turn_id=turn_id,
        message_id=message_id,
        outcome=outcome,
        why=_build_why(evidence),
        behavior_delta=delta,
        causal_chain=chain,
        evidence_cards=_build_evidence_cards(evidence),
        scorecard=_build_scorecard(appraisal),
        molecule_summaries=_build_molecule_summaries(appraisal, evidence),
        raw_debug=raw_debug,
    )
```

Also re-export `build_substrate_effect_view` from `orion/substrate/appraisal/__init__.py` and add it to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_substrate_effect_view_model.py -v`
Expected: 7 PASS (4 label tests + 3 builder tests).

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/view_model.py orion/substrate/appraisal/__init__.py tests/test_substrate_effect_view_model.py
git commit -m "feat(substrate): build_substrate_effect_view assembles human-readable view"
```

---

## Phase 2 — API endpoint

### Task 6: `GET /api/chat/turn/{turn_id}/substrate-effect`

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Test:   `services/orion-hub/tests/test_substrate_effect_endpoint.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-hub/tests/test_substrate_effect_endpoint.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.substrate_effect_cache import (
    SubstrateEffectSnapshot,
    substrate_effect_cache,
)
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


@pytest.fixture()
def client():
    from scripts.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_cache():
    substrate_effect_cache._entries.clear()  # type: ignore[attr-defined]
    yield


def test_unknown_turn_returns_404(client):
    response = client.get("/api/chat/turn/does-not-exist/substrate-effect")
    assert response.status_code == 404


def test_high_pressure_turn_returns_view_with_repair_concrete(client):
    summary, _ = run_substrate_effect_pipeline(
        turn_id="turn-high",
        message_id=None,
        user_text=(
            "you gave me garbage directions, stop, build me a design spec for claude, "
            "arsonist pov only, nuts and bolts"
        ),
        source_id="conv-high",
        contract_before={"mode": "default"},
    )
    assert summary is not None  # sanity: pipeline did run
    response = client.get("/api/chat/turn/turn-high/substrate-effect")
    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-high"
    assert body["outcome"]["appraisal_kind"] == "repair_pressure"
    assert body["outcome"]["level_label"] in {"HIGH", "MEDIUM"}
    assert body["behavior_delta"]["changed"] in (True, False)
    # primary fields populated without needing raw_debug
    assert isinstance(body["outcome"]["summary"], str) and body["outcome"]["summary"]
    assert isinstance(body["causal_chain"], list)
    assert isinstance(body["evidence_cards"], list)


def test_known_turn_with_no_effect_returns_valid_empty_view(client):
    # Manually inject a snapshot whose appraisal is None — simulates a turn
    # where the pipeline was wired but produced no appraisal at all (rare; we
    # still want a 200 response with a coherent empty view).
    snap = SubstrateEffectSnapshot(
        turn_id="turn-empty",
        message_id=None,
        user_text="",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
        causal_molecule_ids=[],
    )
    substrate_effect_cache.store(snap)
    response = client.get("/api/chat/turn/turn-empty/substrate-effect")
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"]["appraisal_kind"] == "none"
    assert body["outcome"]["level"] == 0.0
    assert body["evidence_cards"] == []
    assert body["causal_chain"] == []
    assert "No substrate effect" in body["outcome"]["summary"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_endpoint.py -v`
Expected: FAIL with 404 on the success cases (route not yet defined).

- [ ] **Step 3: Implement the endpoint**

Open `services/orion-hub/scripts/api_routes.py`. Near the other imports add:

```python
from orion.substrate.appraisal.view_model import build_substrate_effect_view
from .substrate_effect_cache import substrate_effect_cache
```

Place this route near the other `/api/chat/...` routes (search for `@router.get("/api/chat/messages")` and add this nearby):

```python
@router.get("/api/chat/turn/{turn_id}/substrate-effect")
def api_chat_turn_substrate_effect(turn_id: str) -> Dict[str, Any]:
    """Return the operator-readable Substrate Effect view for a chat turn.

    404 only when the turn_id is unknown. Turns where the appraiser ran but
    produced no behavior change still return 200 with a valid (empty-summary)
    view.
    """
    snap = substrate_effect_cache.get(turn_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="turn not found")
    view = build_substrate_effect_view(
        turn_id=snap.turn_id,
        message_id=snap.message_id,
        user_text=snap.user_text,
        appraisal=snap.appraisal,
        signal=snap.signal,
        evidence=snap.evidence,
        contract_before=snap.contract_before,
        contract_after=snap.contract_after,
    )
    return view.model_dump(mode="json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_endpoint.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_effect_endpoint.py
git commit -m "feat(hub): GET /api/chat/turn/{turn_id}/substrate-effect"
```

---

## Phase 3 — Chat chip

### Task 7: CSS for chip and modal

**Files:**
- Create: `services/orion-hub/static/css/substrate-effect.css`
- Modify: `services/orion-hub/templates/index.html`

- [ ] **Step 1: Write the stylesheet**

```css
/* services/orion-hub/static/css/substrate-effect.css */

.substrate-effect-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  border-radius: 9999px;
  border: 1px solid rgba(99, 102, 241, 0.40);
  background: rgba(99, 102, 241, 0.10);
  padding: 0.2rem 0.6rem;
  font-size: 11px;
  font-weight: 600;
  color: #c7d2fe;
  cursor: pointer;
  line-height: 1.2;
}
.substrate-effect-chip:hover { background: rgba(99, 102, 241, 0.20); }
.substrate-effect-chip[data-level="HIGH"] {
  border-color: rgba(244, 63, 94, 0.45);
  background: rgba(244, 63, 94, 0.12);
  color: #fecdd3;
}
.substrate-effect-chip[data-level="MEDIUM"] {
  border-color: rgba(234, 179, 8, 0.45);
  background: rgba(234, 179, 8, 0.12);
  color: #fde68a;
}
.substrate-effect-chip[data-level="NONE"] {
  border-color: rgba(107, 114, 128, 0.45);
  background: rgba(107, 114, 128, 0.10);
  color: #d1d5db;
}

.substrate-effect-modal-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.80);
  z-index: 2147483646;
}
.substrate-effect-modal {
  position: fixed; inset: 2rem; max-width: 56rem; margin: 0 auto;
  background: #0b1020; color: #e5e7eb;
  border: 1px solid #1f2937; border-radius: 1rem;
  z-index: 2147483647;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.substrate-effect-modal header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.25rem; border-bottom: 1px solid #1f2937;
}
.substrate-effect-modal header h2 {
  font-size: 14px; font-weight: 700; letter-spacing: 0.02em;
}
.substrate-effect-modal .body {
  padding: 1rem 1.25rem; overflow: auto; display: flex; flex-direction: column; gap: 1rem;
}
.substrate-effect-section h3 {
  font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
  color: #93c5fd; margin-bottom: 0.4rem;
}
.substrate-effect-section .lede { font-size: 13px; line-height: 1.45; color: #e5e7eb; }
.substrate-effect-section .secondary { font-size: 11px; color: #9ca3af; }

.substrate-effect-bar {
  display: grid; grid-template-columns: 14rem 1fr 3rem; gap: 0.5rem;
  align-items: center; font-size: 12px;
}
.substrate-effect-bar .fill {
  height: 0.5rem; border-radius: 9999px; background: rgba(99,102,241,0.7);
}
.substrate-effect-bar .track {
  height: 0.5rem; border-radius: 9999px; background: rgba(75,85,99,0.4);
  overflow: hidden;
}

.substrate-effect-card {
  border: 1px solid #1f2937; border-radius: 0.75rem;
  padding: 0.6rem 0.8rem; background: rgba(15,23,42,0.6);
}
.substrate-effect-card h4 { font-size: 12px; font-weight: 700; margin-bottom: 0.25rem; }
.substrate-effect-card .span { font-style: italic; color: #cbd5e1; }
.substrate-effect-card .meta { font-size: 11px; color: #9ca3af; }

.substrate-effect-chain { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.4rem; }
.substrate-effect-chain li {
  border-left: 2px solid #6366f1; padding-left: 0.6rem;
}
.substrate-effect-chain li strong { display: block; font-size: 12px; }
.substrate-effect-chain li .desc { font-size: 12px; color: #d1d5db; }

.substrate-effect-raw {
  border: 1px dashed #374151; border-radius: 0.5rem; padding: 0.5rem 0.7rem;
  background: rgba(17,24,39,0.7);
}
.substrate-effect-raw summary {
  cursor: pointer; font-size: 11px; color: #9ca3af; user-select: none;
}
.substrate-effect-raw pre {
  font-size: 10px; line-height: 1.3; color: #cbd5e1; overflow: auto; max-height: 24rem;
  margin-top: 0.5rem; white-space: pre-wrap; word-break: break-word;
}
```

- [ ] **Step 2: Reference the stylesheet in `index.html`**

Open `services/orion-hub/templates/index.html`. Find the existing `<link rel="stylesheet" href="/static/css/style.css">` (or equivalent) and add directly after it:

```html
<link rel="stylesheet" href="/static/css/substrate-effect.css">
```

- [ ] **Step 3: Visual sanity check**

Run: `cd services/orion-hub && python -m pytest tests/test_substrate_effect_endpoint.py -v`
Expected: still PASS (no behavior changed).

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/css/substrate-effect.css services/orion-hub/templates/index.html
git commit -m "feat(hub): substrate effect chip and modal styles"
```

---

### Task 8: JS module — chip + modal scaffold

**Files:**
- Create: `services/orion-hub/static/js/substrate-effect-ui.js`
- Modify: `services/orion-hub/templates/index.html`

- [ ] **Step 1: Implement the module**

```javascript
// services/orion-hub/static/js/substrate-effect-ui.js
// Substrate Effect UI — chip + modal.
// Owns nothing about appraiser semantics. Renders SubstrateEffectViewV1 as-is.

(function () {
  const API_BASE = (window.ORION_HUB_API_BASE || window.location.origin).replace(/\/+$/, '');

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    for (const child of [].concat(children)) {
      if (child == null) continue;
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
    return node;
  }

  function renderSubstrateEffectChip(summary, { onClick } = {}) {
    if (!summary || typeof summary !== 'object') return null;
    const level = String(summary.level_label || 'NONE').toUpperCase();
    const text = summary.chip_label || 'Substrate Effect';
    const chip = el('button', {
      type: 'button',
      class: 'substrate-effect-chip',
      dataset: { level, turnId: summary.turn_id || '' },
    }, [`Substrate Effect: ${text}`]);
    chip.addEventListener('click', () => {
      const turnId = summary.turn_id;
      if (!turnId) return;
      if (typeof onClick === 'function') onClick(turnId);
      else openSubstrateEffectModal(turnId);
    });
    return chip;
  }

  async function openSubstrateEffectModal(turnId) {
    if (!turnId) return;
    let view;
    try {
      const resp = await fetch(`${API_BASE}/api/chat/turn/${encodeURIComponent(turnId)}/substrate-effect`);
      if (!resp.ok) {
        view = { _error: `${resp.status} ${resp.statusText}` };
      } else {
        view = await resp.json();
      }
    } catch (err) {
      view = { _error: String((err && err.message) || err) };
    }
    document.body.appendChild(renderSubstrateEffectModal(view));
  }

  function renderSubstrateEffectModal(view) {
    const backdrop = el('div', { class: 'substrate-effect-modal-backdrop' });
    const modal = el('div', { class: 'substrate-effect-modal', role: 'dialog' });
    const close = () => {
      backdrop.remove();
      modal.remove();
      document.removeEventListener('keydown', onKey);
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    backdrop.addEventListener('click', close);

    const closeBtn = el('button', { type: 'button', class: 'text-gray-400 hover:text-white text-xs' }, ['Close']);
    closeBtn.addEventListener('click', close);
    modal.appendChild(el('header', {}, [el('h2', {}, ['Substrate Effect for This Turn']), closeBtn]));

    const body = el('div', { class: 'body' });
    if (view && view._error) {
      body.appendChild(el('div', { class: 'substrate-effect-section' }, [
        el('h3', {}, ['Error']),
        el('p', { class: 'lede' }, [`Failed to load substrate effect view: ${view._error}`]),
      ]));
    } else {
      body.appendChild(renderOutcome(view));
      if (view.why) body.appendChild(renderWhy(view));
      if (view.behavior_delta) body.appendChild(renderBehaviorDelta(view.behavior_delta));
      if (view.causal_chain && view.causal_chain.length) body.appendChild(renderCausalChain(view.causal_chain));
      if (view.evidence_cards && view.evidence_cards.length) body.appendChild(renderEvidenceCards(view.evidence_cards));
      if (view.scorecard) body.appendChild(renderScorecard(view.scorecard));
      if (view.molecule_summaries && view.molecule_summaries.length) body.appendChild(renderMoleculeSummaries(view.molecule_summaries));
      if (view.raw_debug) body.appendChild(renderRawDebug(view.raw_debug));
    }
    modal.appendChild(body);

    const root = el('div', {});
    root.appendChild(backdrop);
    root.appendChild(modal);
    return root;
  }

  function renderOutcome(view) {
    const o = view.outcome || {};
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Outcome']));
    section.appendChild(el('p', { class: 'lede' }, [o.summary || '']));
    section.appendChild(el('p', { class: 'secondary' }, [
      `Level: ${Number(o.level || 0).toFixed(2)} (${o.level_label || ''}) · Confidence: ${Number(o.confidence || 0).toFixed(2)} (${o.confidence_label || ''})`,
    ]));
    return section;
  }

  function renderWhy(view) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Why']));
    section.appendChild(el('p', { class: 'lede' }, [view.why]));
    return section;
  }

  function renderBehaviorDelta(delta) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['What changed']));
    const before = delta.contract_before || '—';
    const after = delta.contract_after || '—';
    const lede = delta.changed
      ? `Before: ${before} · After: ${after}`
      : `Before: ${before} · After: ${after} — no contract change.`;
    section.appendChild(el('p', { class: 'lede' }, [lede]));
    if (delta.explanation) {
      section.appendChild(el('p', { class: 'secondary' }, [delta.explanation]));
    }
    if (delta.rules_activated && delta.rules_activated.length) {
      const list = el('ul', { class: 'list-disc list-inside text-xs text-gray-200' });
      delta.rules_activated.forEach((rule) => list.appendChild(el('li', {}, [String(rule)])));
      section.appendChild(list);
    }
    return section;
  }

  function renderCausalChain(steps) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Causal chain']));
    const list = el('ol', { class: 'substrate-effect-chain' });
    steps.forEach((step) => {
      list.appendChild(el('li', {}, [
        el('strong', {}, [`${step.index}. ${step.title}`]),
        el('span', { class: 'desc' }, [step.description || '']),
      ]));
    });
    section.appendChild(list);
    return section;
  }

  function renderEvidenceCards(cards) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Evidence']));
    const wrap = el('div', { class: 'grid grid-cols-1 md:grid-cols-2 gap-2' });
    cards.forEach((card) => {
      const node = el('div', { class: 'substrate-effect-card' }, [
        el('h4', {}, [card.label]),
        card.source_span
          ? el('p', { class: 'span' }, [`"${card.source_span}"`])
          : null,
        el('p', { class: 'meta' }, [`Score ${Number(card.score).toFixed(2)} · Confidence ${Number(card.confidence).toFixed(2)}`]),
        card.meaning ? el('p', {}, [card.meaning]) : null,
      ]);
      wrap.appendChild(node);
    });
    section.appendChild(wrap);
    return section;
  }

  function renderScorecard(scorecard) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, [scorecard.title || 'Scorecard']));
    scorecard.items.forEach((item) => {
      const row = el('div', { class: 'substrate-effect-bar' });
      row.appendChild(el('span', {}, [item.label]));
      const track = el('div', { class: 'track' });
      const fill = el('div', { class: 'fill', style: `width:${Math.round(Math.max(0, Math.min(1, item.value)) * 100)}%` });
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('span', { class: 'secondary' }, [Number(item.value).toFixed(2)]));
      section.appendChild(row);
    });
    if (scorecard.final_label) section.appendChild(el('p', { class: 'lede' }, [scorecard.final_label]));
    if (scorecard.explanation) section.appendChild(el('p', { class: 'secondary' }, [scorecard.explanation]));
    return section;
  }

  function renderMoleculeSummaries(summaries) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Molecules used']));
    summaries.forEach((mol) => {
      section.appendChild(el('div', { class: 'substrate-effect-card' }, [
        el('h4', {}, [`${mol.label}`]),
        el('p', {}, [mol.explanation || '']),
        el('p', { class: 'meta' }, [mol.molecule_id]),
      ]));
    });
    return section;
  }

  function renderRawDebug(rawDebug) {
    const section = el('details', { class: 'substrate-effect-raw' });
    section.appendChild(el('summary', {}, ['Developer payload']));
    section.appendChild(el('pre', {}, [JSON.stringify(rawDebug, null, 2)]));
    return section;
  }

  window.SubstrateEffectUI = {
    renderChip: renderSubstrateEffectChip,
    openModal: openSubstrateEffectModal,
    renderModal: renderSubstrateEffectModal,
  };
})();
```

- [ ] **Step 2: Load the script from `index.html`**

In `services/orion-hub/templates/index.html`, just before the existing `<script src="/static/js/app.js">` (search for it), add:

```html
<script src="/static/js/substrate-effect-ui.js"></script>
```

- [ ] **Step 3: Manual smoke (optional — UI not yet wired)**

Run the Hub locally (if available) and confirm `window.SubstrateEffectUI` exists in console:
```bash
# in a browser console pointed at the hub:
typeof window.SubstrateEffectUI === 'object'
window.SubstrateEffectUI.openModal('does-not-exist')   // should render an Error section
```
If you cannot run Hub locally, skip this step.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/js/substrate-effect-ui.js services/orion-hub/templates/index.html
git commit -m "feat(hub): substrate-effect-ui module (chip + modal renderer)"
```

---

### Task 9: Render the chip below each assistant message

**Files:**
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: Find the assistant branch of `appendMessage`**

Open `services/orion-hub/static/js/app.js`. Locate the function `function appendMessage(sender, text, colorClass = 'text-white')` (around line 7132). Find the assistant branch — specifically the `actionRow` that already exists (around line 7218), inside `if (sender === 'Orion')`.

- [ ] **Step 2: Append the chip to `actionRow`**

Inside that `if (sender === 'Orion')` block, right after the existing inspect / mind / memory buttons are appended (look for the end of the action-row population, before `if (actionRow.childNodes.length) headerRow.appendChild(actionRow);`), add:

```javascript
// Substrate Effect chip — only renders when summary is present on meta.
const substrateSummary = meta.substrateEffectSummary || meta.substrate_effect_summary;
if (substrateSummary && window.SubstrateEffectUI && typeof window.SubstrateEffectUI.renderChip === 'function') {
  const chip = window.SubstrateEffectUI.renderChip(substrateSummary);
  if (chip) actionRow.appendChild(chip);
}
```

- [ ] **Step 3: Propagate the summary from the chat response into meta**

Search `app.js` for the HTTP chat fallback path that constructs `meta` for `appendMessage('Orion', ...)`. Around line 10514:

```javascript
appendMessage('Orion', displayText || '', 'text-white', {
  // existing fields
});
```

Add to that meta object:

```javascript
  substrateEffectSummary: d.substrate_effect_summary || null,
```

Repeat in the WS path (around line 10188) and any other `appendMessage('Orion', ...)` call sites:

```bash
grep -n "appendMessage('Orion'" services/orion-hub/static/js/app.js
```

For each call, add the same `substrateEffectSummary: d.substrate_effect_summary || null` field. Where the variable is named differently (e.g. `data` rather than `d`), adapt accordingly.

- [ ] **Step 4: Manual smoke (best-effort)**

Run the Hub locally and send a chat message containing a high-repair-pressure phrase such as:
> "you gave me garbage directions, stop, build me a design spec for claude, arsonist pov only, nuts and bolts"

A chip "Substrate Effect: …" should appear below the assistant response. Click it — the modal opens with Outcome / Why / What changed / Causal chain / Evidence / Scorecard / Molecules used / Developer payload (collapsed).

If you cannot run Hub locally, validate by curling the endpoint:
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"you gave me garbage directions, stop, build me a design spec for claude"}]}' \
  | jq '.substrate_effect_summary'
```
Expect a non-null object with `chip_label`, `level_label`, `evidence_count`, etc.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/app.js
git commit -m "feat(hub): render Substrate Effect chip below assistant messages"
```

---

## Phase 4 — Modal polish

The modal is already implemented in Task 8. This phase is a check-and-fix pass against the acceptance criteria.

### Task 10: Acceptance-criteria sweep

- [ ] **Step 1: Walk through each criterion**

For each item below, open the modal (or read the rendered HTML in `substrate-effect-ui.js`) and verify:

  1. Assistant messages can show a Substrate Effect chip. — chip renders when `substrate_effect_summary` is in the assistant meta.
  2. Clicking the chip opens a readable modal. — `openSubstrateEffectModal` fetches and appends modal nodes.
  3. The modal shows Outcome, Why, What changed, Causal chain, Evidence, Scorecard, Molecules used, Developer payload — in that order.
  4. Raw JSON collapsed by default — `<details>` element starts closed.
  5. Operator can understand effect without opening raw JSON — primary sections render full text.
  6. UI distinguishes evidence detected vs appraisal produced vs signal emitted vs behavior changed — the causal chain enumerates each as its own step; the chip summary carries `evidence_count`, `appraisal_kind`, `behavior_applied`, `changed_behavior`.
  7. If no behavior changed, UI says so plainly — Outcome lede says "Orion did not change the response contract"; causal chain step "Behavior unchanged".
  8. Chip does not require full modal payload to render — chip is built from the summary only.
  9. Modal fetches full view lazily — `openSubstrateEffectModal` fetches on click.
  10. Implementation is not graph-first — no graph viz code anywhere.

- [ ] **Step 2: Fix any drift inline**

If a criterion fails, edit `substrate-effect-ui.js` (or related file) to fix it. Add an explanatory secondary line if necessary.

- [ ] **Step 3: Commit (only if changes were made)**

```bash
git add -A
git commit -m "fix(hub): substrate effect modal acceptance pass"
```

If no changes were needed, skip the commit.

---

## Phase 5 — Substrate tab "Recent effects" card

This phase is secondary per the spec but inexpensive once the cache exists.

### Task 11: Recent-effects endpoint

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Test:   `services/orion-hub/tests/test_substrate_effect_endpoint.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_recent_effects_endpoint_returns_newest_first(client):
    for text, turn in [
        ("first benign", "t1"),
        ("you gave me garbage directions, stop, build me a design spec for claude", "t2"),
        ("third benign", "t3"),
    ]:
        run_substrate_effect_pipeline(
            turn_id=turn,
            message_id=None,
            user_text=text,
            source_id=f"conv-{turn}",
            contract_before={"mode": "default"},
        )
    response = client.get("/api/substrate-effect/recent?limit=10")
    assert response.status_code == 200
    rows = response.json().get("rows")
    assert isinstance(rows, list)
    assert [r["turn_id"] for r in rows][:3] == ["t3", "t2", "t1"]
    row_t2 = next(r for r in rows if r["turn_id"] == "t2")
    assert row_t2["chip_label"]
    assert row_t2["level_label"] in {"HIGH", "MEDIUM", "LOW", "NONE"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_endpoint.py::test_recent_effects_endpoint_returns_newest_first -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Implement the route**

In `services/orion-hub/scripts/api_routes.py`, near the per-turn endpoint, add:

```python
@router.get("/api/substrate-effect/recent")
def api_substrate_effect_recent(limit: int = Query(default=25, ge=1, le=100)) -> Dict[str, Any]:
    """Lightweight recent-effects feed for the Substrate tab.

    Returns the per-turn summary shape, not the full view. The full view is
    only fetched on demand via /api/chat/turn/{turn_id}/substrate-effect.
    """
    rows: List[Dict[str, Any]] = []
    for snap in substrate_effect_cache.recent(limit=limit):
        appraisal = snap.appraisal
        level = float(appraisal.dimensions.get("level", 0.0)) if appraisal else 0.0
        from orion.substrate.appraisal.view_model import pressure_label
        level_lbl = pressure_label(level)
        before_mode = str(snap.contract_before.get("mode") or "")
        after_mode = str(snap.contract_after.get("mode") or "")
        changed = before_mode != after_mode
        evidence_count = len(snap.evidence)
        rows.append({
            "turn_id": snap.turn_id,
            "stored_at": snap.stored_at.isoformat(),
            "appraisal_kind": "repair_pressure" if appraisal else "none",
            "level": level,
            "level_label": level_lbl,
            "behavior_applied": after_mode if changed else None,
            "evidence_count": evidence_count,
            "changed_behavior": changed,
            "chip_label": (
                f"{(after_mode if changed else 'no behavior change')} · "
                f"{level_lbl} repair pressure · "
                f"{evidence_count} evidence driver{'s' if evidence_count != 1 else ''}"
            ),
            "turn_summary": (snap.user_text[:120] + "…") if len(snap.user_text) > 120 else snap.user_text,
        })
    return {"rows": rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-hub && pytest tests/test_substrate_effect_endpoint.py -v`
Expected: 4 PASS (3 original + new).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_effect_endpoint.py
git commit -m "feat(hub): GET /api/substrate-effect/recent for substrate tab"
```

---

### Task 12: "Recent Substrate Effects" panel inside the existing `#substrate` tab

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/substrate-effect-ui.js` (extend with a tab-renderer hook)

- [ ] **Step 1: Add a placeholder card inside the substrate tab in `index.html`**

Find the `<section id="substrate" ...>` block. Just inside it (before the existing `<iframe id="substratePanelFrame"`), add:

```html
<div id="substrateEffectRecentPanel" class="bg-gray-900/40 border border-gray-700 rounded-xl p-3 space-y-2">
  <div class="flex items-center justify-between">
    <h2 class="text-sm font-semibold text-gray-200">Recent Substrate Effects</h2>
    <button id="substrateEffectRecentRefresh" type="button" class="text-xs text-indigo-300 hover:text-indigo-200">Refresh</button>
  </div>
  <div id="substrateEffectRecentBody" class="text-xs text-gray-300">Loading…</div>
</div>
```

- [ ] **Step 2: Add the tab-renderer hook to `substrate-effect-ui.js`**

Append before the final `window.SubstrateEffectUI = { ... };` block:

```javascript
async function loadRecentEffects(container) {
  try {
    const resp = await fetch(`${API_BASE}/api/substrate-effect/recent?limit=25`);
    if (!resp.ok) {
      container.textContent = `Failed to load recent effects: ${resp.status} ${resp.statusText}`;
      return;
    }
    const data = await resp.json();
    const rows = (data && data.rows) || [];
    container.innerHTML = '';
    if (!rows.length) {
      container.appendChild(el('p', { class: 'text-gray-400' }, ['No substrate effects recorded yet.']));
      return;
    }
    const list = el('div', { class: 'grid grid-cols-1 gap-2' });
    rows.forEach((row) => {
      const card = el('button', {
        type: 'button',
        class: 'substrate-effect-card text-left',
      }, [
        el('h4', {}, [row.chip_label]),
        el('p', { class: 'meta' }, [`${row.stored_at} · ${row.turn_summary}`]),
      ]);
      card.addEventListener('click', () => openSubstrateEffectModal(row.turn_id));
      list.appendChild(card);
    });
    container.appendChild(list);
  } catch (err) {
    container.textContent = `Failed to load recent effects: ${String((err && err.message) || err)}`;
  }
}

function initSubstrateEffectTab() {
  const body = document.getElementById('substrateEffectRecentBody');
  const refresh = document.getElementById('substrateEffectRecentRefresh');
  if (!body) return;
  loadRecentEffects(body);
  if (refresh) refresh.addEventListener('click', () => loadRecentEffects(body));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSubstrateEffectTab);
} else {
  initSubstrateEffectTab();
}
```

- [ ] **Step 3: Manual smoke (optional)**

Open the Substrate tab in the Hub UI. The "Recent Substrate Effects" card should populate after a chat turn that has substrate effect data.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/substrate-effect-ui.js
git commit -m "feat(hub): substrate tab — recent effects card"
```

---

## Phase 6 — Finish

### Task 13: Final test sweep + push

- [ ] **Step 1: Run the full backend test suite for substrate-effect**

Run:
```bash
pytest tests/test_substrate_effect_view_model.py tests/test_repair_pressure_e2e.py -v
cd services/orion-hub && pytest \
  tests/test_substrate_effect_cache.py \
  tests/test_substrate_effect_pipeline.py \
  tests/test_substrate_effect_endpoint.py \
  tests/test_handle_chat_request_substrate_effect.py -v
```
Expected: all PASS.

- [ ] **Step 2: Run a broader smoke to catch regressions**

Run:
```bash
cd services/orion-hub && pytest -q --maxfail=5 \
  tests/test_handle_chat_request_turn_effect.py \
  tests/test_http_chat_spark_meta.py \
  tests/test_chat_turn_spark_meta_turn_effect.py
```
Expected: all PASS (or pre-existing skips). If something fails, fix root cause — do NOT bypass.

- [ ] **Step 3: Quick local UI smoke if possible**

If a local Hub is runnable, post a chat message containing repair-pressure phrasing and confirm the chip + modal render. Otherwise, hit the endpoints with curl as in Task 9 Step 4.

- [ ] **Step 4: Push**

```bash
git push -u origin feat/repair-pressure-v1
```

---

## Self-review notes

- **Spec coverage:**
  - §1 Goal: Phase 0 wiring + Phases 1–4 deliver the per-turn causal receipt.
  - §2 UX layers: Layer 1 (chip) in Task 9; Layer 2 (modal) in Task 8; Layer 3 (Substrate tab) in Tasks 11–12.
  - §3 UI naming: chip text uses "Substrate Effect:" prefix; section labels match spec §12.
  - §4 Chip behavior: chip only renders when summary is present in meta; gate enforced in Task 9.
  - §5–§6 Modal sections: all sections rendered in Task 8 in the spec-mandated order.
  - §7 Data contract: Task 4–5 deliver the exact `SubstrateEffectViewV1` shape.
  - §8 Endpoint: Task 6 implements the per-turn endpoint with 404-vs-empty-view distinction.
  - §10 Response metadata hook: Task 3 attaches `substrate_effect_summary` to the chat response.
  - §11 Substrate tab: Tasks 11–12 deliver the recent-effects card.
  - §13 Label mappings: Task 4 implements `KIND_LABELS`, `strength_label`, `pressure_label`, plus `confidence_label` (extra but consistent).
  - §15 Tests: Tasks 1–6, 11 cover high / medium / no-effect / no-raw-dependency / cache eviction / endpoint coverage.
  - §16 Acceptance criteria: Task 10 sweep.
  - §17 Implementation order: Phases 0–5 mirror the spec phases (Phase 0 added so the appraiser actually runs).

- **Placeholder scan:** no "TBD", no "implement later". Every step contains the actual code.

- **Type consistency:** `SubstrateEffectSnapshot` field names line up across cache, pipeline, endpoint. `substrate_effect_summary` is the key everywhere (chat result, app.js meta both forms supported). `chip_label` field name is consistent across pipeline summary and recent-effects rows.
