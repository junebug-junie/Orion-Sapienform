# Repair Pressure Speech Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Hub substrate `contract_after` into cortex-exec `compile_speech_contract` so HIGH/MEDIUM repair pressure changes the TURN CONTRACT on the same chat turn.

**Architecture:** Hub attaches `repair_pressure_contract` to `CortexChatRequest.metadata` when `apply_repair_pressure_contract` changes mode; cortex-exec reads `ctx["metadata"]` and merges repair overlay into the existing deterministic speech contract with repair-wins / bias-blends precedence.

**Tech Stack:** Python 3.12, Pydantic v2, pytest; changes in `orion-hub`, `orion-cortex-exec`, shared `orion/substrate/appraisal/`.

**Spec:** `docs/superpowers/specs/2026-07-03-repair-pressure-speech-wiring-design.md`

---

## File map

| File | Role |
|------|------|
| `orion/substrate/appraisal/contract.py` | Add `REPAIR_PRESSURE_CONTRACT_METADATA_KEY` constant |
| `orion/substrate/appraisal/__init__.py` | Re-export metadata key |
| `services/orion-hub/scripts/repair_pressure_wiring.py` | **Create** — `attach_repair_pressure_contract(req, snapshot, enabled)` |
| `services/orion-hub/scripts/api_routes.py` | Capture snapshot; call attach after `build_chat_request` |
| `services/orion-hub/scripts/websocket_handler.py` | Call attach after pipeline, before cortex |
| `services/orion-hub/app/settings.py` | `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` |
| `services/orion-hub/.env_example` | Document flag |
| `services/orion-cortex-exec/app/chat_stance.py` | Extend `compile_speech_contract` with repair branch |
| `services/orion-cortex-exec/app/executor.py` | Pass repair metadata into compiler |
| `services/orion-cortex-exec/app/settings.py` | Same flag |
| `services/orion-cortex-exec/.env_example` | Document flag |
| `services/orion-hub/tests/test_repair_pressure_wiring.py` | **Create** — attach helper tests |
| `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py` | Extend — metadata on high-pressure turn |
| `services/orion-cortex-exec/tests/test_chat_relational_stance.py` | Repair precedence tests |
| `services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py` | **Create** — executor metadata → contract |

---

## Task 1 — Metadata key constant

**Files:**
- Modify: `orion/substrate/appraisal/contract.py`
- Modify: `orion/substrate/appraisal/__init__.py`
- Test: `services/orion-hub/tests/test_repair_pressure_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_repair_pressure_wiring.py`:

```python
from __future__ import annotations

from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY


def test_metadata_key_is_stable_string() -> None:
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY == "repair_pressure_contract"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py::test_metadata_key_is_stable_string -v
```

Expected: FAIL — `ImportError: cannot import name 'REPAIR_PRESSURE_CONTRACT_METADATA_KEY'`

- [ ] **Step 3: Add constant**

In `orion/substrate/appraisal/contract.py`, after `REPAIR_PRESSURE_DEBUG_KEY`:

```python
REPAIR_PRESSURE_CONTRACT_METADATA_KEY = "repair_pressure_contract"
```

In `orion/substrate/appraisal/__init__.py`, add to imports and `__all__`:

```python
from .contract import REPAIR_PRESSURE_DEBUG_KEY, REPAIR_PRESSURE_CONTRACT_METADATA_KEY, apply_repair_pressure_contract
```

```python
    "REPAIR_PRESSURE_CONTRACT_METADATA_KEY",
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py::test_metadata_key_is_stable_string -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/contract.py \
        orion/substrate/appraisal/__init__.py \
        services/orion-hub/tests/test_repair_pressure_wiring.py
git commit -m "feat(substrate): add repair_pressure_contract metadata key constant"
```

---

## Task 2 — Hub attach helper

**Files:**
- Create: `services/orion-hub/scripts/repair_pressure_wiring.py`
- Test: `services/orion-hub/tests/test_repair_pressure_wiring.py`

- [ ] **Step 1: Write failing tests**

Append to `services/orion-hub/tests/test_repair_pressure_wiring.py`:

```python
from orion.schemas.cortex.contracts import CortexChatRequest
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts.repair_pressure_wiring import attach_repair_pressure_contract
from scripts.substrate_effect_cache import SubstrateEffectSnapshot


def _snapshot(*, before_mode: str, after_mode: str, rules: list[str] | None = None) -> SubstrateEffectSnapshot:
    return SubstrateEffectSnapshot(
        turn_id="t1",
        message_id=None,
        user_text="x",
        appraisal=None,
        signal=None,
        contract_before={"mode": before_mode},
        contract_after={"mode": after_mode, "rules": rules or ["be more specific"]},
    )


def test_attach_skips_when_disabled() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    attach_repair_pressure_contract(req, _snapshot(before_mode="default", after_mode="repair_concrete"), enabled=False)
    assert req.metadata is None or REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (req.metadata or {})


def test_attach_skips_when_mode_unchanged() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    attach_repair_pressure_contract(req, _snapshot(before_mode="default", after_mode="default"), enabled=True)
    assert req.metadata is None or REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (req.metadata or {})


def test_attach_writes_contract_when_mode_changed() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain", metadata={"source": "hub_http"})
    snap = _snapshot(before_mode="default", after_mode="repair_concrete", rules=["include tests/acceptance checks"])
    attach_repair_pressure_contract(req, snap, enabled=True)
    assert isinstance(req.metadata, dict)
    payload = req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]
    assert payload["mode"] == "repair_concrete"
    assert "include tests/acceptance checks" in payload["rules"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py -v -k "attach"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.repair_pressure_wiring'`

- [ ] **Step 3: Implement helper**

Create `services/orion-hub/scripts/repair_pressure_wiring.py`:

```python
"""Wire substrate repair contract into Hub → Cortex chat metadata."""

from __future__ import annotations

from typing import Any

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY

from .substrate_effect_cache import SubstrateEffectSnapshot


def _contract_changed(snapshot: SubstrateEffectSnapshot) -> bool:
    before = str((snapshot.contract_before or {}).get("mode") or "")
    after = str((snapshot.contract_after or {}).get("mode") or "")
    return before != after


def attach_repair_pressure_contract(
    req: CortexChatRequest,
    snapshot: SubstrateEffectSnapshot | None,
    *,
    enabled: bool,
) -> None:
    """Mutate req.metadata in place when repair pressure changed behavior."""
    if not enabled or snapshot is None or not _contract_changed(snapshot):
        return
    meta: dict[str, Any] = dict(req.metadata or {})
    meta[REPAIR_PRESSURE_CONTRACT_METADATA_KEY] = dict(snapshot.contract_after)
    req.metadata = meta
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/repair_pressure_wiring.py \
        services/orion-hub/tests/test_repair_pressure_wiring.py
git commit -m "feat(hub): attach repair_pressure_contract to CortexChatRequest metadata"
```

---

## Task 3 — Extend `compile_speech_contract`

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 1: Write failing tests**

Append to `services/orion-cortex-exec/tests/test_chat_relational_stance.py`:

```python
def test_compile_speech_contract_repair_concrete_overrides_relational() -> None:
    brief = _relational_brief(interaction_regime="relational", companion_closing_move="end_with_a_wondering")
    repair = {
        "mode": "repair_concrete",
        "rules": [
            "no broad architecture wandering",
            "include tests/acceptance checks",
        ],
    }
    contract = compile_speech_contract(brief, repair_contract=repair)
    assert "companion turn" not in contract.lower()
    assert "include tests/acceptance checks" in contract
    assert "no broad architecture wandering" in contract


def test_compile_speech_contract_concrete_bias_appends_to_instrumental() -> None:
    brief = _instrumental_brief(task_mode="direct_response")
    repair = {
        "mode": "concrete_bias",
        "rules": ["be more specific", "include next concrete action"],
    }
    contract = compile_speech_contract(brief, repair_contract=repair)
    assert contract.startswith("Answer directly.")
    assert "be more specific" in contract
    assert "include next concrete action" in contract


def test_compile_speech_contract_ignores_default_repair_contract() -> None:
    brief = _instrumental_brief()
    contract = compile_speech_contract(brief, repair_contract={"mode": "default"})
    assert contract == "Answer directly."
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=services/orion-cortex-exec:. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "repair_concrete_overrides or concrete_bias_appends or ignores_default_repair" -v
```

Expected: FAIL — `unexpected keyword argument 'repair_contract'`

- [ ] **Step 3: Implement compiler extension**

In `services/orion-cortex-exec/app/chat_stance.py`, add helper before `compile_speech_contract`:

```python
def _compile_repair_speech_overlay(repair_contract: dict[str, Any] | None) -> str | None:
    if not isinstance(repair_contract, dict):
        return None
    mode = str(repair_contract.get("mode") or "")
    rules = repair_contract.get("rules") or []
    if mode not in {"repair_concrete", "concrete_bias"} or not rules:
        return None
    intro = (
        "Repair turn: answer concretely and operationally."
        if mode == "repair_concrete"
        else "Add concrete specificity this turn."
    )
    return intro + " " + "; ".join(str(r) for r in rules) + "."


def compile_speech_contract(
    brief: "ChatStanceBrief",
    *,
    repair_contract: dict[str, Any] | None = None,
) -> str:
```

At end of `compile_speech_contract`, replace bare `return " ".join(parts)` paths with a merge pattern. After computing regime text into variable `regime_text`:

```python
    overlay = _compile_repair_speech_overlay(repair_contract)
    if overlay is None:
        return regime_text
    mode = str((repair_contract or {}).get("mode") or "")
    if mode == "repair_concrete":
        return overlay
    return f"{regime_text} {overlay}"
```

Apply this pattern to all three regime branches (`minimal`, `relational`, `instrumental`) — extract each branch's return into `regime_text` then use the merge block above.

Ensure `from typing import Any` is present at top of file (likely already imported).

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
PYTHONPATH=services/orion-cortex-exec:. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py -v
```

Expected: all tests pass (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(cortex-exec): merge repair pressure into compile_speech_contract"
```

---

## Task 4 — Cortex-exec executor wiring + settings

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py`
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`
- Create: `services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py`:

```python
from __future__ import annotations

from app.chat_stance import compile_speech_contract
from orion.schemas.chat_stance import ChatStanceBrief
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY


def _resolve_speech_contract(metadata: dict | None, *, enabled: bool = True) -> str:
    """Mirror executor merge logic under test."""
    brief = ChatStanceBrief(
        conversation_frame="mixed",
        task_mode="direct_response",
        user_intent="fix this",
        self_relevance="operational",
        juniper_relevance="practical",
        answer_strategy="direct",
        stance_summary="repair",
    )
    repair_contract = None
    if enabled and isinstance(metadata, dict):
        raw = metadata.get(REPAIR_PRESSURE_CONTRACT_METADATA_KEY)
        if isinstance(raw, dict):
            repair_contract = raw
    return compile_speech_contract(brief, repair_contract=repair_contract)


def test_metadata_repair_concrete_reaches_speech_contract() -> None:
    md = {
        REPAIR_PRESSURE_CONTRACT_METADATA_KEY: {
            "mode": "repair_concrete",
            "rules": ["include file/module boundaries"],
        }
    }
    text = _resolve_speech_contract(md)
    assert "include file/module boundaries" in text
    assert "Repair turn" in text


def test_metadata_absent_keeps_instrumental_contract() -> None:
    text = _resolve_speech_contract({})
    assert text == "Answer directly."
```

- [ ] **Step 2: Run test — should pass after Task 3**

Run:

```bash
PYTHONPATH=services/orion-cortex-exec:. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py -v
```

Expected: PASS (tests compiler path; executor wiring is next)

- [ ] **Step 3: Add settings flag**

In `services/orion-cortex-exec/app/settings.py`, add near other feature flags:

```python
    repair_pressure_speech_wiring_enabled: bool = Field(
        True,
        alias="ENABLE_REPAIR_PRESSURE_SPEECH_WIRING",
    )
```

In `services/orion-cortex-exec/.env_example`, add:

```bash
# When true, Hub repair_pressure_contract metadata merges into chat TURN CONTRACT.
ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=true
```

- [ ] **Step 4: Wire executor**

In `services/orion-cortex-exec/app/executor.py`, replace line ~4252:

```python
                    ctx["speech_contract"] = compile_speech_contract(parsed_brief)
```

with:

```python
                    _md = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
                    _repair_contract = None
                    if settings.repair_pressure_speech_wiring_enabled:
                        _raw = _md.get(REPAIR_PRESSURE_CONTRACT_METADATA_KEY)
                        if isinstance(_raw, dict):
                            _repair_contract = _raw
                    ctx["speech_contract"] = compile_speech_contract(
                        parsed_brief,
                        repair_contract=_repair_contract,
                    )
```

Add import at top with other chat_stance imports:

```python
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
```

- [ ] **Step 5: Sync local env**

Run from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
        services/orion-cortex-exec/app/settings.py \
        services/orion-cortex-exec/.env_example \
        services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py
git commit -m "feat(cortex-exec): read repair_pressure_contract from ctx metadata"
```

---

## Task 5 — Hub HTTP wiring

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py`

- [ ] **Step 1: Write failing test**

Update `_FakeCortex` in `test_handle_chat_request_substrate_effect.py`:

```python
class _FakeCortex:
    last_req = None

    async def chat(self, req, correlation_id=None):
        self.last_req = req
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="ok",
            correlation_id=correlation_id or "corr-substrate",
        )
        return CortexChatResult(cortex_result=result, final_text="ok")
```

Append test:

```python
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY


def test_handle_chat_request_attaches_repair_contract_metadata_on_high_pressure():
    cortex = _FakeCortex()
    payload = _make_payload(
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-x", no_write=True))
    assert cortex.last_req is not None
    md = cortex.last_req.metadata or {}
    contract = md.get(REPAIR_PRESSURE_CONTRACT_METADATA_KEY)
    assert isinstance(contract, dict)
    assert contract.get("mode") in {"repair_concrete", "concrete_bias"}
    assert contract.get("rules")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_handle_chat_request_substrate_effect.py::test_handle_chat_request_attaches_repair_contract_metadata_on_high_pressure -v
```

Expected: FAIL — `contract` is None

- [ ] **Step 3: Add Hub settings flag**

In `services/orion-hub/app/settings.py`:

```python
    ENABLE_REPAIR_PRESSURE_SPEECH_WIRING: bool = Field(
        default=True,
        alias="ENABLE_REPAIR_PRESSURE_SPEECH_WIRING",
    )
```

In `services/orion-hub/.env_example`:

```bash
# Attach substrate repair contract to cortex chat metadata when behavior changes.
ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=true
```

- [ ] **Step 4: Wire HTTP path**

In `services/orion-hub/scripts/api_routes.py`:

1. Import:

```python
from .repair_pressure_wiring import attach_repair_pressure_contract
```

2. Change pipeline capture (~line 2102):

```python
    substrate_summary, substrate_snapshot = run_substrate_effect_pipeline(
```

3. After `build_chat_request` succeeds (~line 2146), before cortex call:

```python
    attach_repair_pressure_contract(
        req,
        substrate_snapshot,
        enabled=bool(getattr(settings, "ENABLE_REPAIR_PRESSURE_SPEECH_WIRING", True)),
    )
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_handle_chat_request_substrate_effect.py -v
```

Expected: all pass

- [ ] **Step 6: Sync env + commit**

```bash
python scripts/sync_local_env_from_example.py
git add services/orion-hub/scripts/api_routes.py \
        services/orion-hub/app/settings.py \
        services/orion-hub/.env_example \
        services/orion-hub/tests/test_handle_chat_request_substrate_effect.py
git commit -m "feat(hub): wire repair contract into HTTP chat metadata"
```

---

## Task 6 — Hub WebSocket wiring

**Files:**
- Modify: `services/orion-hub/scripts/websocket_handler.py`
- Test: `services/orion-hub/tests/test_repair_pressure_wiring.py`

- [ ] **Step 1: Write failing integration-style test**

Append to `test_repair_pressure_wiring.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts.repair_pressure_wiring import attach_repair_pressure_contract
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


@pytest.mark.asyncio
async def test_ws_path_can_attach_after_pipeline_before_cortex():
    """WS builds chat_req before pipeline; attach must run after pipeline."""
    from orion.schemas.cortex.contracts import CortexChatRequest

    high = (
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    _, snap = run_substrate_effect_pipeline(
        turn_id="ws-turn",
        message_id=None,
        user_text=high,
        source_id="sess-ws",
        contract_before={"mode": "default"},
    )
    req = CortexChatRequest(prompt=high, mode="brain")
    attach_repair_pressure_contract(req, snap, enabled=True)
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY in (req.metadata or {})
```

Note: if pytest-asyncio not configured for hub tests, drop `@pytest.mark.asyncio` and make this a sync test (as written — it is sync).

- [ ] **Step 2: Wire websocket_handler**

In `services/orion-hub/scripts/websocket_handler.py`:

1. Import:

```python
from scripts.repair_pressure_wiring import attach_repair_pressure_contract
```

2. Change ~line 890:

```python
            substrate_summary, substrate_snapshot = run_substrate_effect_pipeline(
```

3. Immediately after pipeline logging block, before grammar emit:

```python
            attach_repair_pressure_contract(
                chat_req,
                substrate_snapshot,
                enabled=bool(getattr(settings, "ENABLE_REPAIR_PRESSURE_SPEECH_WIRING", True)),
            )
```

- [ ] **Step 3: Run tests**

Run:

```bash
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py -v
```

Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/scripts/websocket_handler.py \
        services/orion-hub/tests/test_repair_pressure_wiring.py
git commit -m "feat(hub): wire repair contract into WebSocket chat metadata"
```

---

## Task 7 — Gate checks and regression

**Files:** (none — verification only)

- [ ] **Step 1: Run focused test suites**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_repair_pressure_wiring.py \
  services/orion-hub/tests/test_handle_chat_request_substrate_effect.py \
  services/orion-hub/tests/test_substrate_effect_pipeline.py -q

PYTHONPATH=services/orion-cortex-exec:. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py -q
```

Expected: all pass

- [ ] **Step 2: Run repo gates**

```bash
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
git diff --check
```

Expected: clean

- [ ] **Step 3: Optional live smoke (if Hub + cortex-exec running)**

Send high-pressure message via Hub; inspect cortex debug for `speech_contract` containing `Repair turn` or rule substring. Mark UNVERIFIED if containers not running.

- [ ] **Step 4: Commit any env sync only if needed**

Do not commit `.env` files.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Hub attaches `contract_after` when changed | Task 2, 5, 6 |
| Metadata key `repair_pressure_contract` | Task 1 |
| `repair_concrete` wins over relational | Task 3 test + implementation |
| `concrete_bias` blends with regime | Task 3 test + implementation |
| Per-turn `contract_before=default` | Unchanged in pipeline calls |
| Feature flag rollback | Task 4, 5 |
| Fail-open on pipeline failure | Existing behavior; attach skips on `snapshot is None` |
| No bus/schema changes | None planned |
| High-pressure fixture test | Task 5, pipeline tests |
| Benign turn no metadata | Task 2 test + Task 5 existing test |

Placeholder scan: none.

---

## Restart required

After deploy:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build orion-cortex-exec
```

Adjust service names to match local compose if different.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-03-repair-pressure-speech-wiring.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
