# Runtime Trace → Signal Nexus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Join existing cognition execution truth (`CognitionTracePayload`) into the signal nexus (`OrionSignalV1` + Hub) without duplicating trace truth — enabling turn drill-down (Milestone A) and a useful live runtime mesh (Milestone B).

**Architecture:** Milestone A extends the Organ Signal Gateway with a multi-emission adapter contract, `CognitionTraceAdapter` (1 run + N step signals), Hub `CognitionTraceCache` + correlation index on `SignalsInspectCache`, and correlation-scoped Organ Signals UI. Milestone B replaces stub organ adapters per signal-gateway phase 2 M1–M4, adds runtime service adapters, and ships layer filters on the live mesh.

**Tech Stack:** Python 3.12, Pydantic v2, Redis bus (`OrionBusAsync`), OpenTelemetry spans in `orion-signal-gateway`, Hub FastAPI + static JS (Cytoscape), pytest (`PYTHONPATH=.` or `./scripts/test_service.sh <service>`).

**Design source:** [docs/superpowers/specs/2026-05-20-runtime-trace-signal-nexus-design.md](../specs/2026-05-20-runtime-trace-signal-nexus-design.md)

**Depends on:** [Organ Signal Gateway phase 1](../specs/2026-05-01-organ-signal-gateway-design.md), [phase 2 Hub inspect](../specs/2026-05-01-organ-signal-gateway-phase-2-design.md)

**Worktree:** `git worktree add .worktrees/feat-runtime-trace-signal-nexus -b feat/runtime-trace-signal-nexus`

---

## Milestone map (all phases)

| Phase | Milestone | Outcome | Primary services |
|-------|-----------|---------|------------------|
| **A0** | A | Gateway cognition-channel preflight verified | `orion-signal-gateway` |
| **A1** | A | Multi-emission adapter contract + processor flatten | `orion/signals`, `orion-signal-gateway` |
| **A2** | A | `CognitionTracePayload.metadata` enrichment at publish | `orion-cortex-exec` |
| **A3** | A | Registry runtime nodes + `CognitionTraceAdapter` | `orion/signals` |
| **A4** | A | Hub `CognitionTraceCache` + `GET /api/cognition/trace/{id}` | `orion-hub` |
| **A5** | A | Correlation index + `GET /api/signals/correlation/{id}` | `orion-hub` |
| **A6** | A | Correlation ID propagation gate (§5.8) | `orion-hub`, `orion-cortex-exec` |
| **A7** | A | Chat Execution Steps panel | `orion-hub` static JS |
| **A8** | A | Organ Signals correlation view + stub hiding | `orion-hub` static JS |
| **A9** | A | Acceptance tests + staging smoke | all above |

| Phase | Milestone | Outcome | Primary services |
|-------|-----------|---------|------------------|
| **B0** | B | Shared stub detection + layer taxonomy | `orion-hub` |
| **B1** | B | Real `equilibrium` adapter (signal-gateway M1) | `orion/signals` |
| **B2** | B | Real `recall` adapter (M3) | `orion/signals` |
| **B3** | B | Real `chat_stance` adapter (M4) | `orion/signals` |
| **B4** | B | Real `autonomy` + `spark_introspector` adapters | `orion/signals` |
| **B5** | B | Runtime service adapters (`hub`, `cortex_gateway`, `cortex_orch`, writers) | `orion/signals`, emitters |
| **B6** | B | Remaining stub organ adapters (incremental) | `orion/signals` |
| **B7** | B | Layer filter UI + live mesh acceptance | `orion-hub` |

**Subagent rule:** One task = one commit. Do not start A3 adapter work until A1 processor tests pass. Do not mark A9 complete without §5.8 correlation gate evidence.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/signals/adapters/base.py` | `AdapterResult` type + `normalize_adapter_result()` |
| `orion/signals/adapters/cognition_trace.py` | **A3** — multi-emission cognition trace adapter |
| `orion/signals/adapters/__init__.py` | Register `CognitionTraceAdapter` before stubs |
| `orion/signals/registry.py` | `cortex_exec`, `llm_gateway`, `mind`, runtime nodes |
| `orion/signals/stub_detection.py` | **B0** — `is_stub_signal()` shared by Hub + tests |
| `services/orion-signal-gateway/app/processor.py` | Flatten list emissions; per-signal OTEL publish |
| `services/orion-signal-gateway/app/settings.py` | OTEL allowlist for cognition dimensions |
| `services/orion-signal-gateway/app/tests/test_processor_multi_emission.py` | **A1** gate tests |
| `services/orion-cortex-exec/app/main.py` | Trace metadata enrichment block |
| `services/orion-cortex-exec/tests/test_cognition_trace_metadata.py` | **A2** metadata tests |
| `services/orion-hub/scripts/cognition_trace_cache.py` | **A4** — bus subscriber + correlation index |
| `services/orion-hub/scripts/signals_inspect_cache.py` | **A5** — `correlation_id` secondary index |
| `services/orion-hub/scripts/api_routes.py` | New cognition + correlation API routes |
| `services/orion-hub/scripts/main.py` | Start/stop `CognitionTraceCache` |
| `services/orion-hub/app/settings.py` | `COGNITION_TRACE_*` settings |
| `services/orion-hub/static/js/thought-process.js` | **A7** — Execution Steps panel |
| `services/orion-hub/static/js/organ-signals-graph-ui.js` | **A8/B7** — correlation mode + layer filter |
| `services/orion-hub/tests/test_cognition_trace_api.py` | **A4/A5** Hub API tests |
| `orion/signals/adapters/tests/test_cognition_trace_adapter.py` | **A3** adapter unit tests |
| `orion/signals/adapters/tests/fixtures/cognition_trace_chat_general.json` | 3-step chat_general fixture |

---

# Milestone A — Runtime trace fusion

## Phase A0 — Gateway cognition-channel preflight

### Task A0-1: Verify subscription includes cognition trace

**Files:**
- Test: `services/orion-signal-gateway/app/tests/test_cognition_channel_subscription.py` (new)

- [ ] **Step 1: Write failing test**

```python
"""Gateway must subscribe to cognition trace channel (spec §5.4 preflight)."""
from app.settings import settings


def test_organ_channels_includes_cognition_trace() -> None:
    patterns = settings.ORGAN_CHANNELS
    assert any(
        p in ("orion:cognition:trace", "orion:cognition:*") or p.endswith("cognition:*")
        for p in patterns
    ), f"ORGAN_CHANNELS missing cognition pattern: {patterns}"
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=services/orion-signal-gateway:../../.. ./scripts/test_service.sh orion-signal-gateway app/tests/test_cognition_channel_subscription.py -v`

Expected: PASS (channel already present: `orion:cognition:*` in `app/settings.py`)

- [ ] **Step 3: Document preflight in gateway README**

Add a short note under operator checklist in `services/orion-signal-gateway/README.md`:

```markdown
### Cognition trace preflight (Runtime Trace Nexus A0)
- `ORGAN_CHANNELS` must include `orion:cognition:*` (or `orion:cognition:trace`).
- Staging: one `chat_general` turn → gateway logs show envelope on `orion:cognition:trace` reaching `SignalProcessor.handle_envelope`.
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-signal-gateway/app/tests/test_cognition_channel_subscription.py services/orion-signal-gateway/README.md
git commit -m "test(gateway): document cognition trace channel preflight"
```

---

## Phase A1 — Multi-emission adapter contract

### Task A1-1: Extend adapter base type

**Files:**
- Modify: `orion/signals/adapters/base.py`
- Create: `orion/signals/adapters/result.py`

- [ ] **Step 1: Write failing import test**

Create `orion/signals/adapters/tests/test_adapter_result.py`:

```python
from orion.signals.adapters.result import AdapterResult, normalize_adapter_result
from orion.signals.models import OrganClass, OrionSignalV1
from datetime import datetime, timezone


def test_normalize_single_signal() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="a" * 64,
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.5},
        observed_at=now,
        emitted_at=now,
    )
    out = normalize_adapter_result(sig)
    assert out == [sig]


def test_normalize_list() -> None:
    assert normalize_adapter_result([]) == []
    assert normalize_adapter_result(None) == []
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/signals/adapters/tests/test_adapter_result.py -v`

Expected: `ModuleNotFoundError: orion.signals.adapters.result`

- [ ] **Step 3: Implement result helpers**

Create `orion/signals/adapters/result.py`:

```python
from __future__ import annotations

from typing import List, Optional, Union

from orion.signals.models import OrionSignalV1

AdapterResult = Union[OrionSignalV1, List[OrionSignalV1], None]


def normalize_adapter_result(value: AdapterResult) -> List[OrionSignalV1]:
    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in value if s is not None]
    return [value]
```

Update `orion/signals/adapters/base.py`:

```python
from typing import ClassVar, Dict, List, Optional, Union

from orion.signals.adapters.result import AdapterResult
# ...

    @abstractmethod
    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> AdapterResult:
        """Return one signal, a list of signals, or None to drop."""
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/signals/adapters/result.py orion/signals/adapters/base.py orion/signals/adapters/tests/test_adapter_result.py
git commit -m "feat(signals): multi-emission AdapterResult contract"
```

### Task A1-2: Processor flattens list emissions

**Files:**
- Modify: `services/orion-signal-gateway/app/processor.py`
- Create: `services/orion-signal-gateway/app/tests/test_processor_multi_emission.py`

- [ ] **Step 1: Write failing processor test**

```python
"""Multi-emission adapter path (spec §5.1)."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.adapters.result import AdapterResult
from orion.signals.models import OrganClass, OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


class _ListEmitAdapter(OrionSignalAdapter):
    organ_id = "cortex_exec"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return channel == "orion:cognition:trace"

    def adapt(self, channel, payload, registry, prior_signals, norm_ctx) -> AdapterResult:
        now = datetime.now(timezone.utc)
        base = dict(
            organ_class=OrganClass.endogenous,
            dimensions={"success": 1.0},
            causal_parents=[],
            source_event_id="corr-test",
            observed_at=now,
            emitted_at=now,
        )
        return [
            OrionSignalV1(signal_id="run1", organ_id="cortex_exec", signal_kind="cognition_run", **base),
            OrionSignalV1(signal_id="step1", organ_id="graph_cognition", signal_kind="cognition_step", **base),
        ]


@pytest.fixture
def memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.mark.asyncio
async def test_gateway_processor_multi_emission(monkeypatch, memory_exporter):
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.signal_window import SignalWindow
    import app.processor as proc_mod

    monkeypatch.setattr(proc_mod, "ADAPTERS", [_ListEmitAdapter()])
    bus = AsyncMock()
    proc = SignalProcessor(
        bus=bus,
        signal_window=SignalWindow(30.0),
        norm_state=NormalizationStateRegistry(),
        output_channel_prefix="orion:signals",
        passthrough_pattern="orion:signals:*",
        service_ref=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="n"),
    )
    env = BaseEnvelope(
        kind="orion:cognition:trace",
        source=ServiceRef(name="orion-cortex-exec", version="0.1.0", node="n"),
        correlation_id="corr-test",
        payload={"verb": "chat_general", "mode": "brain", "steps": []},
    )
    await proc.handle_envelope(env)
    assert bus.publish.await_count == 2
    channels = [c.args[0] for c in bus.publish.await_args_list]
    assert "orion:signals:cortex_exec" in channels
    assert "orion:signals:graph_cognition" in channels
```

- [ ] **Step 2: Run test — expect FAIL** (single-emission path only publishes once)

- [ ] **Step 3: Update processor**

At top of `processor.py` add:

```python
from orion.signals.adapters.result import normalize_adapter_result
```

Replace the adapt block in `handle_envelope` (lines ~181–198):

```python
                raw = adapter.adapt(
                    channel=env.kind or "",
                    payload=payload,
                    registry=ORGAN_REGISTRY,
                    prior_signals=prior,
                    norm_ctx=norm_ctx,
                )
                signals = normalize_adapter_result(raw)
                if not signals:
                    continue
                for signal in signals:
                    if self._should_suppress_adapted(signal):
                        logger.debug(
                            "Suppressing adapted signal organ=%s kind=%s",
                            signal.organ_id,
                            signal.signal_kind,
                        )
                        continue
                    signal = with_missed_parent_notes(signal, prior, ORGAN_REGISTRY)
                    await self._emit_traced(signal, prior=prior)
                break
```

- [ ] **Step 4: Extend OTEL dimension allowlist**

In `services/orion-signal-gateway/app/settings.py`, append to `_DEFAULT_OTEL_DIMENSION_ALLOWLIST`:

```python
    "success",
    "error_present",
    "recall_used",
    "reasoning_present",
    "final_text_present",
    "step_count",
    "service_count",
```

- [ ] **Step 5: Run test — expect PASS**

Run: `./scripts/test_service.sh orion-signal-gateway app/tests/test_processor_multi_emission.py -v`

- [ ] **Step 6: Commit**

```bash
git add services/orion-signal-gateway/app/processor.py services/orion-signal-gateway/app/settings.py services/orion-signal-gateway/app/tests/test_processor_multi_emission.py
git commit -m "feat(gateway): publish each signal from multi-emission adapters"
```

---

## Phase A2 — CognitionTracePayload metadata enrichment

### Task A2-1: Enrich trace metadata at cortex-exec publish

**Files:**
- Modify: `services/orion-cortex-exec/app/main.py` (trace publish block ~321–337)
- Create: `services/orion-cortex-exec/tests/test_cognition_trace_metadata.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from orion.schemas.cortex.schemas import PlanExecutionResult, StepExecutionResult
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload


def _build_trace_metadata(res: PlanExecutionResult, *, req_extra: dict | None = None) -> dict:
    from app.main import build_cognition_trace_metadata

    return build_cognition_trace_metadata(res, req_extra=req_extra or {})


def test_metadata_includes_routing_and_presence_flags() -> None:
    res = PlanExecutionResult(
        verb_name="chat_general",
        status="success",
        mode="brain",
        steps=[
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="collect_metacog_context",
                order=0,
                result={"MetacogContextService": {}},
                latency_ms=10,
            ),
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="synthesize_chat_stance_brief",
                order=1,
                result={"LLMGatewayService": {"stance_brief": "secret"}},
                latency_ms=20,
            ),
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=2,
                result={"LLMGatewayService": {"content": "hello"}},
                latency_ms=30,
            ),
        ],
        final_text="hello",
        memory_used=True,
        metadata={"recall_profile": "chat.general.v1"},
    )
    meta = _build_trace_metadata(
        res,
        req_extra={
            "route_intent": "none",
            "session_id": "sess-1",
            "message_id": "msg-1",
        },
    )
    assert meta["verb"] == "chat_general"
    assert meta["mode"] == "brain"
    assert meta["recall_profile"] == "chat.general.v1"
    assert meta["chat_stance_debug_present"] is True
    assert meta["final_text_present"] is True
    assert meta["canonical_final_step_name"] == "llm_chat_general"
    assert meta["session_id"] == "sess-1"
    assert meta["message_id"] == "msg-1"
    assert "secret" not in str(meta)
```

- [ ] **Step 2: Run test — expect FAIL** (`build_cognition_trace_metadata` missing)

- [ ] **Step 3: Implement helper in `main.py`**

Add near trace publish:

```python
def build_cognition_trace_metadata(
    res: PlanExecutionResult,
    *,
    req_extra: dict | None = None,
) -> dict[str, Any]:
    extra = req_extra or {}
    steps = res.steps or []
    last_step = steps[-1].step_name if steps else None
    stance_present = any(
        s.step_name == "synthesize_chat_stance_brief" and s.status == "success" for s in steps
    )
    reasoning_present = bool(res.reasoning_content or res.reasoning_trace or res.metacog_traces)
    thinking_present = bool(res.inline_think_content or res.thinking_source)
    thought_source = None
    if res.reasoning_trace:
        thought_source = "reasoning_trace"
    elif res.metacog_traces:
        thought_source = "metacog_traces"
    elif res.inline_think_content:
        thought_source = "inline_think"
    return {
        "request_id": res.request_id,
        "status": res.status,
        "verb": res.verb_name,
        "mode": res.mode,
        "route_intent": extra.get("route_intent") or extra.get("routeIntent"),
        "recall_profile": (res.metadata or {}).get("recall_profile") or extra.get("recall_profile"),
        "chat_stance_debug_present": stance_present,
        "mind_handoff_quality": (res.metadata or {}).get("mind_handoff_quality"),
        "reasoning_present": reasoning_present,
        "thinking_present": thinking_present,
        "canonical_final_step_name": last_step,
        "canonical_thought_source": thought_source,
        "session_id": extra.get("session_id") or extra.get("sessionId"),
        "message_id": extra.get("message_id") or extra.get("messageId"),
        "root_correlation_id": extra.get("root_correlation_id"),
    }
```

Wire into `CognitionTracePayload(..., metadata=build_cognition_trace_metadata(res, req_extra=req_env.payload.args.extra or {}))`.

- [ ] **Step 4: Run test — expect PASS**

Run: `./scripts/test_service.sh orion-cortex-exec tests/test_cognition_trace_metadata.py -v`

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/main.py services/orion-cortex-exec/tests/test_cognition_trace_metadata.py
git commit -m "feat(cortex-exec): enrich CognitionTracePayload metadata for signal nexus"
```

---

## Phase A3 — Registry + CognitionTraceAdapter

### Task A3-1: Add runtime registry entries

**Files:**
- Modify: `orion/signals/registry.py`

- [ ] **Step 1: Write failing registry test**

Create `orion/signals/adapters/tests/test_runtime_registry_entries.py`:

```python
from orion.signals.registry import ORGAN_REGISTRY


def test_cortex_exec_registry_entry() -> None:
    entry = ORGAN_REGISTRY["cortex_exec"]
    assert "cognition_run" in entry.signal_kinds
    assert "orion:cognition:trace" in entry.bus_channels


def test_llm_gateway_registry_entry() -> None:
    assert ORGAN_REGISTRY["llm_gateway"].organ_id == "llm_gateway"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Add entries** (before closing `}` of `ORGAN_REGISTRY`):

```python
    "cortex_exec": OrionOrganRegistryEntry(
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=["cognition_run", "cognition_step"],
        canonical_dimensions=["success", "step_count", "latency_level", "recall_used", "confidence"],
        causal_parent_organs=["autonomy", "graph_cognition"],
        bus_channels=["orion:cognition:trace"],
        notes=["Runtime trace fusion: PlanRunner publishes CognitionTracePayload."],
    ),
    "llm_gateway": OrionOrganRegistryEntry(
        organ_id="llm_gateway",
        organ_class=OrganClass.endogenous,
        service="orion-llm-gateway",
        signal_kinds=["cognition_step", "completion"],
        canonical_dimensions=["success", "latency_level", "confidence"],
        causal_parent_organs=["chat_stance", "cortex_exec"],
        bus_channels=["orion:exec:result:LLMGatewayService"],
        notes=[],
    ),
    "mind": OrionOrganRegistryEntry(
        organ_id="mind",
        organ_class=OrganClass.endogenous,
        service="orion-mind",
        signal_kinds=["mind_handoff", "cognition_step"],
        canonical_dimensions=["success", "confidence"],
        causal_parent_organs=["chat_stance", "recall"],
        bus_channels=["orion:mind:run:complete"],
        notes=["Milestone B: mind_handoff adapter."],
    ),
    "cortex_gateway": OrionOrganRegistryEntry(
        organ_id="cortex_gateway",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-gateway",
        signal_kinds=["route_decision"],
        canonical_dimensions=["confidence"],
        causal_parent_organs=[],
        bus_channels=["orion:cortex:gateway:route"],
        notes=["Milestone B adapter."],
    ),
    "cortex_orch": OrionOrganRegistryEntry(
        organ_id="cortex_orch",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-orch",
        signal_kinds=["plan_resolution"],
        canonical_dimensions=["confidence"],
        causal_parent_organs=["cortex_gateway"],
        bus_channels=["orion:cortex:orch:plan"],
        notes=["Milestone B adapter."],
    ),
    "hub": OrionOrganRegistryEntry(
        organ_id="hub",
        organ_class=OrganClass.exogenous,
        service="orion-hub",
        signal_kinds=["chat_turn"],
        canonical_dimensions=["confidence"],
        causal_parent_organs=[],
        bus_channels=["orion:hub:chat:turn"],
        notes=["Milestone B: optional exogenous chat_turn root."],
    ),
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

### Task A3-2: CognitionTraceAdapter + fixture

**Files:**
- Create: `orion/signals/adapters/cognition_trace.py`
- Create: `orion/signals/adapters/tests/fixtures/cognition_trace_chat_general.json`
- Create: `orion/signals/adapters/tests/test_cognition_trace_adapter.py`
- Modify: `orion/signals/adapters/__init__.py`

- [ ] **Step 1: Add fixture** `cognition_trace_chat_general.json` (minimal 3-step trace; **no PII in fields used for assertions**):

```json
{
  "mode": "brain",
  "verb": "chat_general",
  "recall_used": true,
  "final_text": "USER_SECRET_SHOULD_NOT_APPEAR_IN_SIGNAL",
  "steps": [
    {"status": "success", "verb_name": "chat_general", "step_name": "collect_metacog_context", "order": 0, "result": {"MetacogContextService": {}}, "latency_ms": 120},
    {"status": "success", "verb_name": "chat_general", "step_name": "synthesize_chat_stance_brief", "order": 1, "result": {"LLMGatewayService": {}}, "latency_ms": 200},
    {"status": "success", "verb_name": "chat_general", "step_name": "llm_chat_general", "order": 2, "result": {"LLMGatewayService": {}}, "latency_ms": 522}
  ],
  "metadata": {"recall_profile": "chat.general.v1", "reasoning_present": false}
}
```

- [ ] **Step 2: Write failing adapter tests**

```python
import json
from pathlib import Path

import pytest

from orion.signals.adapters.cognition_trace import CognitionTraceAdapter
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

FIXTURE = Path(__file__).parent / "fixtures" / "cognition_trace_chat_general.json"
CORR = "corr-chat-general-fixture"


@pytest.fixture
def adapter() -> CognitionTraceAdapter:
    return CognitionTraceAdapter()


def test_cognition_trace_adapter_chat_general(adapter: CognitionTraceAdapter) -> None:
    payload = json.loads(FIXTURE.read_text())
    norm = NormalizationContext()
    out = adapter.adapt("orion:cognition:trace", payload, ORGAN_REGISTRY, {}, norm)
    assert isinstance(out, list)
    assert len(out) == 4
    run = next(s for s in out if s.signal_kind == "cognition_run")
    assert run.organ_id == "cortex_exec"
    assert run.source_event_id == CORR or run.source_event_id  # set in test via envelope mock
    steps = [s for s in out if s.signal_kind == "cognition_step"]
    assert {s.organ_id for s in steps} == {"graph_cognition", "chat_stance", "llm_gateway"}
    run_id = make_signal_id("cortex_exec", f"{CORR}:run")
    assert run.signal_id == run_id
    for s in steps:
        assert run.signal_id in s.causal_parents


def test_cognition_trace_adapter_no_pii_in_signal(adapter: CognitionTraceAdapter) -> None:
    payload = json.loads(FIXTURE.read_text())
    out = adapter.adapt("orion:cognition:trace", payload, ORGAN_REGISTRY, {}, NormalizationContext())
    blob = json.dumps([s.model_dump(mode="json") for s in out])
    assert "USER_SECRET" not in blob
    assert "final_text" not in blob
```

Update tests to pass `correlation_id` via adapter API — implement adapter to accept optional envelope correlation in `can_handle`/`adapt` by reading `payload` only; tests inject `correlation_id` into payload for `source_event_id` derivation:

```python
payload["correlation_id"] = CORR  # test-only; production uses envelope in gateway wrapper
```

**Gateway note:** `CognitionTraceAdapter` must prefer envelope `correlation_id` passed as kwarg or embedded at top level when gateway calls adapt — implement `adapt` to use `payload.get("correlation_id")` last, and add optional `envelope_correlation_id` parameter via a thin wrapper in processor (cleanest: merge into payload dict before adapt in processor for `orion:cognition:trace` only).

- [ ] **Step 3: Implement `cognition_trace.py`** (core logic per spec §5.3):

```python
"""CognitionTracePayload → OrionSignalV1 run + step signals (spec §5.3)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.adapters.result import AdapterResult
from orion.signals.models import OrganClass, OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_STEP_NAME_ORGAN = {
    "collect_metacog_context": "graph_cognition",
    "synthesize_chat_stance_brief": "chat_stance",
    "llm_chat_general": "llm_gateway",
}

_SERVICE_PREFIX_ORGAN = [
    ("RecallService", "recall"),
    ("Mind", "mind"),
    ("LLMGatewayService", "llm_gateway"),
    ("MetacogContextService", "graph_cognition"),
    ("AgentChainService", "agent_chain"),
    ("PlannerReactService", "planner"),
]


def _step_services(step) -> List[str]:
    if isinstance(step.result, dict) and step.result:
        return list(step.result.keys())
    return []


def _map_step_organ(step_name: str, services: List[str]) -> tuple[str, List[str]]:
    if step_name in _STEP_NAME_ORGAN:
        return _STEP_NAME_ORGAN[step_name], []
    if services:
        for prefix, organ in _SERVICE_PREFIX_ORGAN:
            if services[0].startswith(prefix) or prefix in services[0]:
                return organ, [f"step_organ_fallback:{step_name}"]
    return "cortex_exec", [f"step_organ_fallback:{step_name}"]


class CognitionTraceAdapter(OrionSignalAdapter):
    organ_id = "cortex_exec"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "cognition:trace" in channel:
            return True
        return payload.get("verb") is not None and "steps" in payload

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> AdapterResult:
        try:
            trace = CognitionTracePayload.model_validate(payload)
        except Exception:
            return None

        corr = str(payload.get("_envelope_correlation_id") or trace.correlation_id or "").strip()
        if not corr:
            corr = f"{trace.verb}:{int(trace.timestamp)}"
            synthetic_note = "synthetic_correlation_id"
        else:
            synthetic_note = None

        now = datetime.now(timezone.utc)
        steps = trace.steps or []
        all_ok = trace.metadata.get("status", "success") == "success" and all(
            s.status == "success" for s in steps
        )
        total_ms = sum(int(s.latency_ms or 0) for s in steps)
        meta = trace.metadata or {}
        reasoning = bool(meta.get("reasoning_present")) or bool(trace.recall_debug)

        run_entry = registry.get("cortex_exec") or ORGAN_REGISTRY["cortex_exec"]
        run_parents = [
            prior_signals[p].signal_id
            for p in (run_entry.causal_parent_organs or [])
            if p in prior_signals
        ]
        run_id = make_signal_id("cortex_exec", f"{corr}:run")
        run_notes: List[str] = []
        if synthetic_note:
            run_notes.append(synthetic_note)
        if not steps:
            run_notes.append("no_steps_in_trace")

        run_sig = OrionSignalV1(
            signal_id=run_id,
            organ_id="cortex_exec",
            organ_class=OrganClass.endogenous,
            signal_kind="cognition_run",
            dimensions={
                "success": 1.0 if all_ok else 0.0,
                "step_count": clamp01(len(steps) / 20.0),
                "latency_level": clamp01(min(total_ms, 120_000) / 120_000.0),
                "recall_used": 1.0 if trace.recall_used else 0.0,
                "reasoning_present": 1.0 if reasoning else 0.0,
                "final_text_present": 1.0 if (trace.final_text or "").strip() else 0.0,
            },
            causal_parents=run_parents,
            source_event_id=corr,
            observed_at=now,
            emitted_at=now,
            summary=(
                f"verb={trace.verb} mode={trace.mode} steps={len(steps)} "
                f"recall={int(trace.recall_used)} latency={total_ms}ms"
            ),
            notes=run_notes[:5],
        )

        out: List[OrionSignalV1] = [run_sig]
        for step in sorted(steps, key=lambda s: s.order):
            services = _step_services(step)
            organ_id, extra_notes = _map_step_organ(step.step_name, services)
            step_parents = [run_id]
            reg_entry = registry.get(organ_id) or ORGAN_REGISTRY.get(organ_id)
            if reg_entry:
                for p in reg_entry.causal_parent_organs or []:
                    if p in prior_signals:
                        step_parents.append(prior_signals[p].signal_id)
                        break
            step_id = make_signal_id(organ_id, f"{corr}:step:{step.order}:{step.step_name}")
            out.append(
                OrionSignalV1(
                    signal_id=step_id,
                    organ_id=organ_id,
                    organ_class=OrganClass.endogenous,
                    signal_kind="cognition_step",
                    dimensions={
                        "success": 1.0 if step.status == "success" else 0.0,
                        "latency_level": clamp01(min(int(step.latency_ms or 0), 60_000) / 60_000.0),
                        "error_present": 1.0 if (step.error or "").strip() else 0.0,
                        "service_count": clamp01(min(len(services), 5) / 5.0),
                    },
                    causal_parents=list(dict.fromkeys(step_parents)),
                    source_event_id=corr,
                    observed_at=now,
                    emitted_at=now,
                    summary=f"step={step.step_name} status={step.status} latency={step.latency_ms or 0}ms",
                    notes=extra_notes[:5],
                )
            )
        return out
```

- [ ] **Step 4: Inject envelope correlation in processor**

In `handle_envelope`, before `adapter.adapt` for cognition traces:

```python
        if (env.kind or "").startswith("orion:cognition:trace") and env.correlation_id:
            payload = {**payload, "_envelope_correlation_id": env.correlation_id}
```

- [ ] **Step 5: Register adapter first in `__init__.py`**

```python
from .cognition_trace import CognitionTraceAdapter

ADAPTERS: List[OrionSignalAdapter] = [
    CognitionTraceAdapter(),
    BiometricsAdapter(),
    # ... rest unchanged
]
```

- [ ] **Step 6: Run adapter tests — expect PASS**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/signals/adapters/tests/test_cognition_trace_adapter.py -v`

- [ ] **Step 7: Commit**

```bash
git add orion/signals/adapters/cognition_trace.py orion/signals/adapters/__init__.py orion/signals/registry.py orion/signals/adapters/tests/
git commit -m "feat(signals): CognitionTraceAdapter multi-emission for chat_general"
```

---

## Phase A4 — Hub CognitionTraceCache + API

### Task A4-1: CognitionTraceCache module

**Files:**
- Create: `services/orion-hub/scripts/cognition_trace_cache.py`
- Modify: `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`
- Modify: `services/orion-hub/scripts/main.py`

- [ ] **Step 1: Write failing API test**

Create `services/orion-hub/tests/test_cognition_trace_api.py`:

```python
def test_cognition_trace_cache_api_redacted_default() -> None:
    import asyncio
    from datetime import datetime, timezone
    from scripts.cognition_trace_cache import CognitionTraceCache
    from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
    from orion.schemas.cortex.schemas import StepExecutionResult

    cache = CognitionTraceCache(
        enabled=True,
        subscribe_channel="orion:cognition:trace",
        max_entries=10,
        ttl_sec=300.0,
        api_debug=False,
    )
    trace = CognitionTracePayload(
        correlation_id="corr-1",
        mode="brain",
        verb="chat_general",
        final_text="SECRET",
        steps=[
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=0,
                result={"LLMGatewayService": {}},
                latency_ms=1,
            )
        ],
        recall_used=True,
    )

    async def _run():
        await cache.put("corr-1", trace, otel_trace_id=None)
        return await cache.get_redacted("corr-1")

    body = asyncio.run(_run())
    assert body is not None
    assert body["final_text_present"] is True
    assert "SECRET" not in str(body)
    assert body["steps"][0]["error_present"] is False
```

- [ ] **Step 2: Implement cache** (mirror `SignalsInspectCache` patterns):

Key methods: `start(bus)`, `stop()`, `put(correlation_id, trace, otel_trace_id)`, `get_redacted(correlation_id)`, `get_debug(correlation_id)` (gated).

Redacted step shape per spec §5.5 — derive `services` from `list(step.result.keys())`.

- [ ] **Step 3: Add settings**

```python
    COGNITION_TRACE_CACHE_ENABLED: bool = Field(default=True, alias="COGNITION_TRACE_CACHE_ENABLED")
    COGNITION_TRACE_CACHE_MAX: int = Field(default=200, alias="COGNITION_TRACE_CACHE_MAX")
    COGNITION_TRACE_CACHE_TTL_SEC: float = Field(default=300.0, alias="COGNITION_TRACE_CACHE_TTL_SEC")
    COGNITION_TRACE_SUBSCRIBE_CHANNEL: str = Field(
        default="orion:cognition:trace", alias="COGNITION_TRACE_SUBSCRIBE_CHANNEL"
    )
    COGNITION_TRACE_API_DEBUG: bool = Field(default=False, alias="COGNITION_TRACE_API_DEBUG")
```

Wire in `main.py` startup alongside `SignalsInspectCache`.

- [ ] **Step 4: Add API route in `api_routes.py`**

```python
@router.get("/api/cognition/trace/{correlation_id}")
async def api_cognition_trace(correlation_id: str) -> Dict[str, Any]:
    import scripts.main as hub_main

    cache = getattr(hub_main, "cognition_trace_cache", None)
    if cache is None or not cache.enabled:
        raise HTTPException(status_code=503, detail="cognition_trace_cache_disabled")
    debug = bool(getattr(settings, "COGNITION_TRACE_API_DEBUG", False))
    body = await cache.get_debug(correlation_id) if debug else await cache.get_redacted(correlation_id)
    if body is None:
        raise HTTPException(status_code=404, detail="cognition_trace_not_cached")
    return body
```

- [ ] **Step 5: Run test — expect PASS**

Run: `./scripts/test_service.sh orion-hub tests/test_cognition_trace_api.py -v`

- [ ] **Step 6: Commit**

---

## Phase A5 — Correlation index on SignalsInspectCache

### Task A5-1: correlation_id → signal chain

**Files:**
- Modify: `services/orion-hub/scripts/signals_inspect_cache.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Extend: `services/orion-hub/tests/test_signals_inspect_api.py`

- [ ] **Step 1: Write failing test**

```python
def test_signals_inspect_correlation_index() -> None:
    import asyncio
    from datetime import datetime, timezone
    from orion.signals.models import OrganClass, OrionSignalV1

    SignalsInspectCache = _get_signals_inspect_cache_class()
    now = datetime.now(timezone.utc)
    corr = "corr-index-test"
    sigs = [
        OrionSignalV1(
            signal_id=f"s{i}",
            organ_id="cortex_exec",
            organ_class=OrganClass.endogenous,
            signal_kind="cognition_step",
            dimensions={"success": 1.0},
            source_event_id=corr,
            observed_at=now,
            emitted_at=now,
        )
        for i in range(2)
    ]

    cache = SignalsInspectCache(
        enabled=True,
        subscribe_pattern="orion:signals:*",
        window_sec=60.0,
        trace_enabled=False,
        trace_max_traces=10,
        trace_ttl_sec=600.0,
        trace_max_signals_per_trace=8,
    )

    async def _run():
        for s in sigs:
            await cache._ingest_signal(s)
        return await cache.get_correlation(corr)

    out = asyncio.run(_run())
    assert out is not None
    assert out["correlation_id"] == corr
    assert len(out["chain"]) == 2
```

Add `_ingest_signal` as a test hook or public `record_signal_for_tests`.

Implement `get_correlation(correlation_id)` returning:

```python
{
  "correlation_id": correlation_id,
  "chain": [...],
  "complete": True,
  "gaps": [],
  "hidden_stubs": 0,  # populated in A8 JS; API can compute via stub_detection
}
```

Index on `source_event_id == correlation_id` in `_handle_message`.

- [ ] **Step 2: Add route**

```python
@router.get("/api/signals/correlation/{correlation_id}")
async def api_signals_correlation(correlation_id: str) -> Dict[str, Any]:
    import scripts.main as hub_main

    cache = getattr(hub_main, "signals_inspect_cache", None)
    if cache is None or not cache.enabled:
        raise HTTPException(status_code=503, detail="signals_inspect_cache_disabled")
    body = await cache.get_correlation(correlation_id)
    if body is None:
        raise HTTPException(status_code=404, detail="correlation_not_cached")
    return body
```

- [ ] **Step 3: Run tests — expect PASS**

- [ ] **Step 4: Commit**

---

## Phase A6 — Correlation ID propagation gate (§5.8)

### Task A6-1: Ensure Hub turn metadata exposes canonical correlation_id

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py` (chat HTTP + WS response payload)
- Create: `services/orion-hub/tests/test_correlation_id_propagation.py`

- [ ] **Step 1: Write failing test**

```python
def test_chat_response_metadata_includes_canonical_correlation_id() -> None:
    """Hub must expose the same corr id used for cortex + trace APIs (spec §5.8)."""
    from scripts.api_routes import _chat_turn_trace_linkage

    out = _chat_turn_trace_linkage(
        hub_corr_id="hub-corr-abc",
        cortex_corr_id="hub-corr-abc",
        root_correlation_id=None,
    )
    assert out["correlation_id"] == "hub-corr-abc"
    assert out["root_correlation_id"] is None

    out2 = _chat_turn_trace_linkage(
        hub_corr_id="hub-corr-abc",
        cortex_corr_id="cortex-different",
        root_correlation_id="hub-corr-abc",
    )
    assert out2["correlation_id"] == "hub-corr-abc"
    assert out2["root_correlation_id"] == "hub-corr-abc"
```

- [ ] **Step 2: Implement helper and wire into chat JSON**

```python
def _chat_turn_trace_linkage(
    *,
    hub_corr_id: str,
    cortex_corr_id: str | None,
    root_correlation_id: str | None = None,
) -> dict[str, str | None]:
    canonical = hub_corr_id
    if cortex_corr_id and cortex_corr_id != hub_corr_id:
        return {
            "correlation_id": canonical,
            "root_correlation_id": root_correlation_id or hub_corr_id,
            "cortex_correlation_id": cortex_corr_id,
        }
    return {"correlation_id": canonical, "root_correlation_id": None, "cortex_correlation_id": cortex_corr_id}
```

Include `trace_linkage` (or flatten keys) in assistant message metadata returned to UI.

- [ ] **Step 3: Manual staging verification checklist** (document in test module docstring)

One live `chat_general` turn — assert same id across:

1. Hub chat response `correlation_id`
2. `CognitionTracePayload` envelope on bus (log)
3. `orion:signals:cortex_exec` run signal `source_event_id`
4. `GET /api/cognition/trace/{id}`
5. `GET /api/signals/correlation/{id}`

- [ ] **Step 4: Commit**

---

## Phase A7 — Chat Execution Steps panel

### Task A7-1: thought-process.js Execution Steps

**Files:**
- Modify: `services/orion-hub/static/js/thought-process.js`
- Modify: `services/orion-hub/static/js/thought-process.test.js`
- Modify: `services/orion-hub/tests/test_chat_stance_debug_panel.py` (assert panel hook exists)

- [ ] **Step 1: Write failing JS test**

In `thought-process.test.js`:

```javascript
const thoughtProcess = require('./thought-process.js');

test('buildExecutionStepsPanel returns collapsible section', () => {
  const panel = thoughtProcess.buildExecutionStepsPanel({
    correlationId: 'corr-1',
    apiBaseUrl: 'http://localhost:8080',
    trace: {
      verb: 'chat_general',
      steps: [{ step_name: 'collect_metacog_context', order: 0, status: 'success', latency_ms: 10, services: ['MetacogContextService'] }],
      complete: true,
    },
  });
  expect(panel).toContain('Execution Steps');
  expect(panel).toContain('collect_metacog_context');
  expect(panel).toContain('/organ-signals');
  expect(panel).toContain('correlation_id=corr-1');
});
```

- [ ] **Step 2: Implement `buildExecutionStepsPanel` + `fetchCognitionTrace`**

```javascript
  async function fetchCognitionTrace(apiBaseUrl, correlationId) {
    const base = String(apiBaseUrl || '').replace(/\/$/, '');
    const res = await fetch(`${base}/api/cognition/trace/${encodeURIComponent(correlationId)}`);
    if (res.status === 404) return { error: 'trace_not_cached' };
    if (!res.ok) return { error: `http_${res.status}` };
    return { body: await res.json() };
  }

  function buildExecutionStepsPanel({ correlationId, apiBaseUrl, trace, debug }) {
    // render <details><summary>Execution Steps</summary>… per step… footer link
  }
```

Resolve correlation via `root_correlation_id || correlation_id` from turn metadata.

- [ ] **Step 3: Hook into existing thought-process render path** where `correlationId` is already available in `app.js` (~6163 `createAgentTracePanel`).

- [ ] **Step 4: Run JS test**

Run: `cd services/orion-hub/static/js && npm test -- thought-process.test.js` (or project’s documented JS test command)

- [ ] **Step 5: Commit**

---

## Phase A8 — Organ Signals correlation view + stub hiding

### Task A8-1: Stub detection helper

**Files:**
- Create: `orion/signals/stub_detection.py`
- Test: `orion/signals/tests/test_stub_detection.py`

- [ ] **Step 1: Write failing test**

```python
from orion.signals.stub_detection import is_stub_signal
from orion.signals.models import OrganClass, OrionSignalV1
from datetime import datetime, timezone

def test_is_stub_signal_detects_placeholder_dimensions() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="x",
        organ_id="recall",
        organ_class=OrganClass.endogenous,
        signal_kind="recall_result",
        dimensions={"level": 0.5, "confidence": 0.5},
        observed_at=now,
        emitted_at=now,
        notes=["stub adapter — not yet implemented"],
    )
    assert is_stub_signal(sig) is True
```

- [ ] **Step 2: Implement per spec §5.9**

- [ ] **Step 3: Commit**

### Task A8-2: organ-signals-graph-ui.js correlation mode

**Files:**
- Modify: `services/orion-hub/static/js/organ-signals-graph-ui.js`
- Create: `services/orion-hub/tests/test_organ_signals_correlation_mode.py`

- [ ] **Step 1: Write failing static contract test**

```python
def test_organ_signals_graph_supports_correlation_query_param() -> None:
    src = (REPO_ROOT / "services/orion-hub/static/js/organ-signals-graph-ui.js").read_text()
    assert "correlation_id" in src
    assert "buildCorrelationGraphElements" in src
    assert "signal_id" in src
```

- [ ] **Step 2: Implement**

- Parse `?correlation_id=` from `window.location.search`
- Fetch `/api/signals/correlation/{id}`
- `buildCorrelationGraphElements(chain)` — node `id` = `signal_id`, label = `organ_id + signal_kind`, edges from `causal_parents` directly
- Filter stubs by default; show `Hidden stubs: N` + toggle
- Linear layout (`rankDir: 'LR'`) for correlation mode

- [ ] **Step 3: Run tests — expect PASS**

- [ ] **Step 4: Commit**

---

## Phase A9 — Milestone A acceptance

### Task A9-1: Targeted test sweep + staging smoke

- [ ] **Step 1: Run gateway tests**

`./scripts/test_service.sh orion-signal-gateway`

- [ ] **Step 2: Run hub tests**

`./scripts/test_service.sh orion-hub`

- [ ] **Step 3: Run adapter tests**

`PYTHONPATH=. ./venv/bin/python -m pytest orion/signals/adapters/tests/ -q`

- [ ] **Step 4: Staging smoke** (compose up)

1. Send Hub `chat_general` message
2. Confirm Execution Steps panel shows 3 steps with latency
3. Open Organ Signals `?correlation_id=<id>` — nodes are `signal_id`, chain `graph_cognition → chat_stance → llm_gateway`
4. Confirm stub warning + toggle
5. Confirm no PII in `orion:signals:*` payloads (grep bus or Hub inspect)

- [ ] **Step 5: PR report** — `docs/superpowers/pr-reports/2026-05-20-runtime-trace-signal-nexus-milestone-a.md` with §5.8 evidence

- [ ] **Step 6: Commit docs only**

**Milestone A acceptance checklist (from spec §6.1):**

- [ ] Multi-emission processor tests pass before adapter merge order verified
- [ ] Gateway preflight: cognition envelopes reach processor
- [ ] Hub 3-step timeline
- [ ] Correlation graph uses `signal_id` nodes
- [ ] Stub hiding + toggle
- [ ] Correlation ID gate §5.8
- [ ] No PII in signals / default cognition API
- [ ] `./scripts/test_service.sh orion-signal-gateway` and `orion-hub` pass

---

# Milestone B — Live mesh expansion

## Phase B0 — Stub detection + layer taxonomy

### Task B0-1: Layer map for Organ Signals filter

**Files:**
- Create: `orion/signals/layers.py`
- Modify: `services/orion-hub/static/js/organ-signals-graph-ui.js`

- [ ] **Step 1: Define layer membership**

```python
ORGAN_LAYER: dict[str, str] = {
    "cortex_exec": "runtime",
    "llm_gateway": "runtime",
    "cortex_gateway": "runtime",
    "cortex_orch": "runtime",
    "hub": "runtime",
    "graph_cognition": "cognition",
    "chat_stance": "cognition",
    "recall": "cognition",
    "mind": "cognition",
    "spark_introspector": "cognition",
    "biometrics": "infra",
    "equilibrium": "infra",
    # ... map remaining registry organs to memory|social|vision|persistence|infra
}
```

- [ ] **Step 2: Export `GET /api/signals/layers` optional** (or embed in active response)

- [ ] **Step 3: Commit**

---

## Phase B1 — Real `equilibrium` adapter (signal-gateway M1)

### Task B1-1: Replace equilibrium stub with EquilibriumSnapshotV1

**Files:**
- Modify: `orion/signals/adapters/equilibrium.py`
- Modify: `orion/signals/adapters/tests/test_equilibrium_adapter.py`

- [ ] **Step 1: Write failing test using real kind**

```python
def test_equilibrium_adapter_snapshot_v1(adapter, norm_ctx) -> None:
    from datetime import datetime, timezone
    payload = {
        "correlation_id": "eq-corr",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mesh_health": 0.82,
        "service_states": {},
    }
    signal = adapter.adapt("equilibrium.snapshot.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "mesh_health"
    assert signal.dimensions["level"] > 0.5
    assert "stub adapter" not in " ".join(signal.notes)
```

- [ ] **Step 2: Implement `can_handle` for `equilibrium.snapshot.v1` and `orion:equilibrium:snapshot`**

- [ ] **Step 3: Map `mesh_health` / distress fields → normalized dimensions**

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

---

## Phase B2 — Real `recall` adapter (M3)

### Task B2-1: Recall exec result adapter

**Files:**
- Modify: `orion/signals/adapters/recall.py`
- Create: `orion/signals/adapters/tests/test_recall_adapter.py`

- [ ] **Step 1: Capture fixture from `orion:exec:result:RecallService` representative payload** (redacted) in `tests/fixtures/recall_exec_result.json`

- [ ] **Step 2: Write failing tests** — `signal_kind` = `recall_result`, dimensions from hit count / quality, parents from `prior_signals`

- [ ] **Step 3: Implement** — no stub note; degrade with low `confidence` + note when payload partial

- [ ] **Step 4: Commit**

---

## Phase B3 — Real `chat_stance` adapter (M4)

### Task B3-1: Document bus contract subsection

**Files:**
- Create: `docs/superpowers/specs/2026-05-20-chat-stance-signal-adapter-contract.md` (channels, field paths, degradation)

- [ ] **Step 1: Write contract from cortex-exec chat stance publish paths** (`services/orion-cortex-exec/app/chat_stance.py`, bus kinds)

- [ ] **Step 2: Review with operator**

- [ ] **Step 3: Commit docs**

### Task B3-2: Implement chat_stance adapter

**Files:**
- Modify: `orion/signals/adapters/chat_stance.py`

- [ ] **Step 1–5: TDD** per contract — `signal_kind=chat_stance`, no stance brief text in `summary`/`dimensions`

- [ ] **Commit**

---

## Phase B4 — `autonomy` + `spark_introspector`

### Task B4-1: autonomy adapter

- [ ] Replace stub in `orion/signals/adapters/autonomy.py` using `orion:cortex:exec:request` / autonomy summary payloads from `orion/autonomy/summary.py` shapes

### Task B4-2: spark_introspector adapter

- [ ] Replace stub in `orion/signals/adapters/spark.py` (organ_id `spark_introspector`) for `orion:spark:signal` / telemetry kinds

Each task: failing test → implement → `./scripts/test_service.sh` not required; use `pytest orion/signals/adapters/tests/ -q`

---

## Phase B5 — Runtime service adapters

### Task B5-1: Hub `chat_turn` exogenous signal (optional per spec §11)

- [ ] Emit `hub.chat_turn` at start of Hub chat dispatch on `orion:hub:chat:turn` (or document deferral)

### Task B5-2: `cortex_gateway` route_decision

- [ ] Adapter on `orion:cortex:gateway:route` (verify channel in `orion/bus/channels.yaml`)

### Task B5-3: `cortex_orch` plan_resolution

- [ ] Adapter on orch plan publish channel

### Task B5-4: Persistence writer persist events

- [ ] Adapters for `sql_writer`, `rdf_writer`, `vector_writer` **persist** signal kinds (presence + latency only, no row contents)

---

## Phase B6 — Remaining stub organs

### Task B6-N: One organ per subagent

For each remaining stub adapter file in `orion/signals/adapters/` still returning `notes=["stub adapter — not yet implemented"]`:

- [ ] Verify `ORGAN_REGISTRY[*].bus_channels` against `orion/bus/channels.yaml`
- [ ] Add fixture + failing test
- [ ] Implement minimal real `adapt`
- [ ] Commit per organ

**Priority order after B4:** `collapse_mirror`, `journaler`, `social_memory`, `world_pulse`, then long tail.

---

## Phase B7 — Layer filter UI + Milestone B acceptance

### Task B7-1: Layer filter dropdown in Organ Signals

**Files:**
- Modify: `services/orion-hub/static/js/organ-signals-graph-ui.js`
- Modify: `services/orion-hub/templates/index.html` (dropdown hook)

- [ ] **Step 1: Add `<select id="organ-signals-layer-filter">`** with options: All, Runtime, Cognition, Memory, Infra, Social, Vision, Persistence

- [ ] **Step 2: Filter nodes in live mode** using `ORGAN_LAYER` map from API or inlined JS constant generated from `orion/signals/layers.py`

- [ ] **Step 3: Staging smoke**

- Live mesh ≥8 non-stub nodes during chat
- Runtime filter shows `cortex_exec`, `llm_gateway`, gateway, exec
- Cognition filter shows `chat_stance`, `recall`, `mind`, `spark_introspector`
- Missed causal parent rate <20% on primary chain (manual count from `/api/signals/active`)

- [ ] **Step 4: PR report** — `docs/superpowers/pr-reports/2026-05-20-runtime-trace-signal-nexus-milestone-b.md`

- [ ] **Commit**

**Milestone B acceptance (spec §6.2):**

- [ ] ≥8 non-stub nodes during normal chat
- [ ] Layer filters Runtime + Cognition correct
- [ ] Missed parent rate <20% on staging smoke

---

## Self-review (plan author)

| Spec section | Task(s) |
|--------------|---------|
| §5.1 multi-emission | A1-1, A1-2 |
| §5.2 metadata | A2-1 |
| §5.3 CognitionTraceAdapter | A3-2 |
| §5.4 registry + preflight | A0-1, A3-1 |
| §5.5 CognitionTraceCache | A4-1 |
| §5.6 correlation index | A5-1 |
| §5.7 step timeline | A7-1 |
| §5.8 correlation gate | A6-1 |
| §5.9 Organ Signals UI | A8-2, B7-1 |
| §6.2 Milestone B adapters | B1–B6 |
| §7 error handling | Covered in adapter/cache implementations (synthetic corr, 404, stubs, empty steps) |
| §8 test names | Mapped in tasks above |

**Placeholder scan:** No TBD steps. **Type consistency:** `AdapterResult`, `normalize_adapter_result`, `_envelope_correlation_id` used consistently.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-runtime-trace-signal-nexus-implementation.md`.

**Recommended: Subagent-Driven Development** — dispatch one subagent per task (A0-1, A1-1, …), with code review between tasks. Milestone A phases A0–A9 ship first; Milestone B starts only after A9 acceptance.

**Alternative: Inline Execution** — use `executing-plans` in this session with batch checkpoints after A1, A3, A5, A9.

Which approach do you want?
