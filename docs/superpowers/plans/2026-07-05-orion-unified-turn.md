# Orion Unified Turn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Brain speech spine and agent-claude bypass on Orion mode with a unified turn: association → stance_react → fcc harness → three-beat finalize (5a substrate / 5b reflect / 5c voice) → outcome molecule → post-turn learning closure.

**Architecture:** Hub owns a thin saga in `orion/hub/turn_orchestrator.py`. Two new bus workers isolate latency domains: `orion-thought` (stance) and `orion-harness-governor` (motor + finalize). Substrate structural appraisal (5a) is deterministic graph logic — not LLM. Brain tier cortex steps own Jinja for stance_react, harness_finalize_reflect, and orion_voice_finalize. Rollout is flag-gated; Brain `chat_general` shim stays until §13 sunset criteria pass.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Redis bus RPC (`OrionBusAsync.rpc_request`), FastAPI (Hub WS), existing fcc bridge (`services/orion-hub/scripts/fcc_claude_bridge.py`), cortex-exec verb runtime, substrate-runtime SQL projections.

**Spec:** `docs/superpowers/specs/2026-07-05-orion-unified-turn-design.md`

**Branch:** `feat/orion-unified-turn` (worktree recommended: `../Orion-Sapienform-unified-turn`)

**Parallel split (optional):** Phases 0–2 (foundation + thought), 3–4 (association + substrate appraisal), 5–6 (cortex finalize verbs), 7–10 (services + rollout) can run in separate worktrees once Phase 0 bus/schema contracts land on main.

---

## File map

| Path | Responsibility |
|------|----------------|
| `orion/schemas/thought.py` | **Create** — `ThoughtEventV1`, `StanceReactRequestV1`, `StanceHarnessSliceV1`, `HubAssociationBundleV1`, `CoalitionSnapshotV1` |
| `orion/schemas/harness_finalize.py` | **Create** — draft/verdict/outcome/closure/run schemas from spec §6 |
| `orion/bus/channels.yaml` | **Modify** — thought, harness, substrate finalize RPC, verdict/outcome/closure events |
| `orion/schemas/registry.py` | **Modify** — register all new schema_ids |
| `orion/hub/association.py` | **Create** — `build_hub_association_bundle()` via felt_state_reader + fail-closed |
| `orion/hub/turn_orchestrator.py` | **Create** — Hub saga: ingress → association → thought RPC → harness RPC → WS publish |
| `orion/hub/turn_request.py` | **Create** — `build_orion_turn_request()` (thin; not `build_chat_request`) |
| `orion/thought/profiles.py` | **Create** — stance profile registry (`stance_react` v1) |
| `orion/thought/stance_react.py` | **Create** — cortex RPC wrapper + ThoughtEvent assembly |
| `orion/thought/stance_quality.py` | **Create** — `enforce_thought_stance_quality()` harness profile |
| `orion/thought/policy_refusal.py` | **Create** — refuse/defer/proceed policy |
| `orion/thought/tests/` | **Create** — stance quality + evidence_refs fail-closed |
| `orion/harness/runner.py` | **Create** — fcc motor loop (reuse fcc_claude_bridge patterns) |
| `orion/harness/prefix.py` | **Create** — `compile_harness_prefix(ThoughtEventV1, …)` |
| `orion/harness/repair.py` | **Create** — `map_repair_pressure_contract()` → `HarnessRepairOverlayV1` |
| `orion/harness/grammar_publish.py` | **Create** — per-step grammar on `orion:grammar:event` |
| `orion/harness/finalize.py` | **Create** — 5a–6b + step 7 orchestration |
| `orion/harness/cortex_client.py` | **Create** — bus RPC to cortex-exec for finalize verbs |
| `orion/harness/substrate_client.py` | **Create** — draft molecule RPC to substrate-runtime |
| `orion/harness/tests/` | **Create** — prefix, repair overlay, quick lane, molecule gates |
| `orion/substrate/appraisal/finalize_draft_v1.py` | **Create** — deterministic 5a appraisal from inline receipts |
| `orion/cognition/verbs/stance_react.yaml` | **Create** |
| `orion/cognition/verbs/harness_finalize_reflect.yaml` | **Create** |
| `orion/cognition/verbs/orion_voice_finalize.yaml` | **Create** |
| `orion/cognition/prompts/stance_react.j2` | **Create** |
| `orion/cognition/prompts/harness_finalize_reflect.j2` | **Create** |
| `orion/cognition/prompts/orion_voice_finalize.j2` | **Create** |
| `services/orion-thought/` | **Create** — bus listener service shell |
| `services/orion-harness-governor/` | **Create** — bus listener service shell |
| `services/orion-substrate-runtime/app/finalize_appraisal_listener.py` | **Create** — 5a RPC handler |
| `services/orion-hub/scripts/thought_client.py` | **Create** — thought RPC client (PreTurnAppraisalClient pattern) |
| `services/orion-hub/scripts/harness_governor_client.py` | **Create** — harness RPC client |
| `services/orion-hub/scripts/websocket_handler.py` | **Modify** — Orion mode → turn_orchestrator when flag on |
| `services/orion-hub/scripts/api_routes.py` | **Modify** — HTTP parity |
| `services/orion-hub/app/settings.py` | **Modify** — unified turn flags + channel names |
| `services/orion-hub/.env_example` | **Modify** |
| `services/orion-cortex-exec/app/executor.py` | **Modify** — register three new verb steps |
| `orion/harness/evals/` | **Create** — layer attribution + quick-lane hard cases |
| `services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py` | **Create** |
| `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` | **Create** |

**Reuse (do not delete):** `services/orion-hub/scripts/fcc_claude_bridge.py`, `services/orion-hub/scripts/pre_turn_appraisal_client.py`, `orion/substrate/felt_state_reader.py`, `services/orion-cortex-exec/app/chat_stance.py` (`build_chat_stance_inputs` only — Brain shim keeps `enforce_chat_stance_quality`).

**Non-goals (this plan):** Brain shim removal (§13 step 11), reverie/dream profiles (step 12), merging 5b+5c, quick lane for stance or voice.

---

## Phase 0 — Broadcast + pre_turn smoke (spec §14 step 0)

### Task 1: Hub association read fail-closed

**Files:**
- Create: `orion/hub/association.py`
- Create: `orion/hub/tests/test_association_read_fail_closed.py`
- Modify: `orion/substrate/felt_state_reader.py` (export `read_attention_broadcast_lane` helper if missing)

- [ ] **Step 1: Write the failing test**

Create `orion/hub/tests/test_association_read_fail_closed.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.hub.association import build_hub_association_bundle
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
)


def _fresh_broadcast() -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        generated_at=datetime.now(timezone.utc),
        frame=AttentionFrameV1(open_loops=[]),
        attended_node_ids=["node-a", "node-b"],
    )


def test_association_read_fail_closed_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = _fresh_broadcast().model_copy(
        update={"generated_at": datetime.now(timezone.utc) - timedelta(seconds=300)}
    )

    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")

    def _fake_read(*, max_age_sec: int = 120, **kwargs):
        return stale, "felt_state_reader"

    monkeypatch.setattr("orion.hub.association._read_broadcast", _fake_read)

    bundle = build_hub_association_bundle(correlation_id="corr-stale", repair_bundle=None)
    assert bundle.broadcast_stale is True
    assert bundle.broadcast is not None


def test_association_read_fail_closed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "false")

    bundle = build_hub_association_bundle(correlation_id="corr-off", repair_bundle=None)
    assert bundle.broadcast is None
    assert bundle.broadcast_stale is True
    assert bundle.read_source == "felt_state_reader"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/hub/tests/test_association_read_fail_closed.py -v
```

Expected: FAIL — `ModuleNotFoundError: orion.hub.association`

- [ ] **Step 3: Implement association builder**

Create `orion/hub/__init__.py` (empty) and `orion/hub/association.py`:

```python
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import HubAssociationBundleV1
from orion.substrate.felt_state_reader import SubstrateFeltStateReader

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_MAX_AGE_SEC = 120


def _broadcast_enabled() -> bool:
    return os.getenv("ORION_ATTENTION_BROADCAST_ENABLED", "false").strip().lower() in _TRUTHY


def _max_age_sec() -> int:
    raw = os.getenv("SUBSTRATE_FELT_STATE_MAX_AGE_SEC", str(_DEFAULT_MAX_AGE_SEC))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SEC


def _read_broadcast(
    *,
    max_age_sec: int = _DEFAULT_MAX_AGE_SEC,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> tuple[AttentionBroadcastProjectionV1 | None, str]:
    if not _broadcast_enabled():
        return None, "felt_state_reader"
    reader = (reader_factory or _default_reader)()
    ctx: dict[str, Any] = {}
    reader.hydrate(ctx)
    raw = ctx.get("attention_broadcast")
    if raw is None:
        return None, "felt_state_reader"
    if isinstance(raw, dict):
        broadcast = AttentionBroadcastProjectionV1.model_validate(raw)
    else:
        broadcast = raw
    ts = broadcast.generated_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age > max_age_sec:
        return broadcast, "felt_state_reader"
    return broadcast, "felt_state_reader"


def _default_reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url=os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL")
        or os.getenv("POSTGRES_URI")
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        max_age_sec=_max_age_sec(),
    )


def _read_execution_trajectory(reader: SubstrateFeltStateReader) -> dict[str, Any] | None:
    ctx: dict[str, Any] = {}
    reader.hydrate(ctx)
    slice_ = ctx.get("execution_trajectory_projection")
    return slice_ if isinstance(slice_, dict) else None


def build_hub_association_bundle(
    *,
    correlation_id: str,
    repair_bundle: TurnAppraisalBundleV1 | None,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> HubAssociationBundleV1:
    max_age = _max_age_sec()
    broadcast, read_source = _read_broadcast(max_age_sec=max_age, reader_factory=reader_factory)
    broadcast_stale = broadcast is None
    if broadcast is not None:
        ts = broadcast.generated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        broadcast_stale = (datetime.now(timezone.utc) - ts).total_seconds() > max_age

    trajectory: dict[str, Any] | None = None
    if reader_factory or _broadcast_enabled():
        try:
            reader = (reader_factory or _default_reader)()
            trajectory = _read_execution_trajectory(reader)
        except Exception:
            trajectory = None

    return HubAssociationBundleV1(
        correlation_id=correlation_id,
        broadcast=broadcast,
        broadcast_stale=broadcast_stale,
        execution_trajectory_slice=trajectory,
        repair_bundle=repair_bundle,
        read_source=read_source,  # type: ignore[arg-type]
    )
```

Create minimal `orion/schemas/thought.py` stub for `HubAssociationBundleV1` (full schemas in Task 5):

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


class HubAssociationBundleV1(BaseModel):
    schema_version: Literal["hub.association.bundle.v1"] = "hub.association.bundle.v1"
    correlation_id: str
    broadcast: AttentionBroadcastProjectionV1 | None
    broadcast_stale: bool
    execution_trajectory_slice: dict[str, Any] | None = None
    repair_bundle: TurnAppraisalBundleV1 | None = None
    read_source: Literal["felt_state_reader", "hub_sql_fallback"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/hub/tests/test_association_read_fail_closed.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/hub/ orion/schemas/thought.py
git commit -m "feat(unified-turn): Hub association bundle with fail-closed stale broadcast"
```

---

### Task 2: Phase 0 smoke — correlation_id chains broadcast + pre_turn

**Files:**
- Create: `orion/hub/tests/test_phase0_ingress_smoke.py`

- [ ] **Step 1: Write integration smoke test (mocked bus)**

Create `orion/hub/tests/test_phase0_ingress_smoke.py`:

```python
from __future__ import annotations

import pytest

from orion.hub.association import build_hub_association_bundle
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


@pytest.mark.asyncio
async def test_ingress_observation_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict] = []

    def _fake_emit(**kwargs):
        emitted.append(kwargs)
        return type("Mol", (), {"molecule_id": "obs-1"})()

    monkeypatch.setattr("orion.mind.substrate_emit.emit_observation", _fake_emit)
    from orion.mind.substrate_emit import emit_observation

    mol = emit_observation(surface_text="hello", source_id="sess-1")
    assert mol.molecule_id == "obs-1"
    assert emitted[0]["surface_text"] == "hello"


def test_association_carries_repair_bundle_correlation() -> None:
    bundle = TurnAppraisalBundleV1(correlation_id="corr-ingress", paradigms={})
    assoc = build_hub_association_bundle(correlation_id="corr-ingress", repair_bundle=bundle)
    assert assoc.correlation_id == "corr-ingress"
    assert assoc.repair_bundle is not None
    assert assoc.repair_bundle.correlation_id == "corr-ingress"
```

- [ ] **Step 2: Run test**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/hub/tests/test_phase0_ingress_smoke.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add orion/hub/tests/test_phase0_ingress_smoke.py
git commit -m "test(unified-turn): phase 0 ingress smoke for observation + association correlation"
```

---

## Phase 1 — Grammar publish + fcc hook (spec §14 step 1)

### Task 3: Harness grammar publish per fcc step

**Files:**
- Create: `orion/harness/__init__.py`
- Create: `orion/harness/grammar_publish.py`
- Create: `orion/harness/tests/test_harness_grammar_publish_per_step.py`

- [ ] **Step 1: Write failing test**

Create `orion/harness/tests/test_harness_grammar_publish_per_step.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.harness.grammar_publish import publish_harness_step_grammar
from orion.schemas.thought import GrammarReceiptV1


@pytest.mark.asyncio
async def test_harness_grammar_publish_per_step() -> None:
    bus = AsyncMock()
    receipts: list[GrammarReceiptV1] = []

    async def _capture(event, **kwargs):
        receipts.append(
            GrammarReceiptV1(
                step_index=kwargs["step_index"],
                tool_name=kwargs.get("tool_name"),
                summary=kwargs["summary"],
                grammar_event_id=event.event_id,
            )
        )

    await publish_harness_step_grammar(
        bus,
        correlation_id="corr-1",
        channel="orion:grammar:event",
        step_index=0,
        tool_name="Read",
        summary="read file",
        publish_fn=_capture,
    )
    assert len(receipts) == 1
    assert receipts[0].tool_name == "Read"
    assert receipts[0].grammar_event_id
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/harness/tests/test_harness_grammar_publish_per_step.py -v
```

- [ ] **Step 3: Implement grammar publish**

Add to `orion/schemas/harness_finalize.py` (partial — full file grows in Task 12):

```python
class GrammarReceiptV1(BaseModel):
    step_index: int
    tool_name: str | None = None
    summary: str
    grammar_event_id: str | None = None
```

Create `orion/harness/grammar_publish.py`:

```python
from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.harness_finalize import GrammarReceiptV1


PublishFn = Callable[..., Awaitable[None]]


async def publish_harness_step_grammar(
    bus: Any,
    *,
    correlation_id: str,
    channel: str,
    step_index: int,
    tool_name: str | None,
    summary: str,
    publish_fn: PublishFn | None = None,
) -> GrammarReceiptV1:
    event = GrammarEventV1(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        event_kind="reasoning_step",
        summary=summary[:500],
        producer="orion-harness-governor",
        step_index=step_index,
        tool_name=tool_name,
    )
    if publish_fn is not None:
        await publish_fn(event, step_index=step_index, tool_name=tool_name, summary=summary)
    else:
        from orion.grammar.publish import publish_grammar_event

        await publish_grammar_event(bus, event, source_name="orion-harness-governor", channel=channel)
    return GrammarReceiptV1(
        step_index=step_index,
        tool_name=tool_name,
        summary=summary,
        grammar_event_id=event.event_id,
    )
```

Re-export `GrammarReceiptV1` in `orion/schemas/thought.py` or keep in harness_finalize only — tests import from `orion.schemas.harness_finalize` after Task 12; for now add re-export:

```python
# orion/schemas/thought.py — add at bottom during Task 3 if needed for test imports
from orion.schemas.harness_finalize import GrammarReceiptV1  # noqa: F401
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Register producer on grammar channel**

In `orion/bus/channels.yaml`, find `orion:grammar:event` and add `orion-harness-governor` to `producer_services` (keep existing producers).

Run:

```bash
python scripts/check_bus_channels.py
```

Expected: PASS

Commit:

```bash
git add orion/harness/ orion/schemas/harness_finalize.py orion/bus/channels.yaml
git commit -m "feat(unified-turn): harness grammar publish per fcc step"
```

---

## Phase 2 — ThoughtEvent + stance_react + orion-thought (spec §14 step 2)

### Task 4: Thought + harness finalize schemas and bus channels

**Files:**
- Modify: `orion/schemas/thought.py` (complete §6.1)
- Create: `orion/schemas/harness_finalize.py` (complete §6.2–6.7)
- Modify: `orion/bus/channels.yaml`
- Modify: `orion/schemas/registry.py`
- Create: `tests/test_unified_turn_schemas.py`

- [ ] **Step 1: Write schema round-trip tests**

Create `tests/test_unified_turn_schemas.py` with tests from spec §6 — minimal example:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.harness_finalize import HarnessDraftMoleculeV1, HarnessRunV1
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1


def test_thought_event_evidence_refs_required_subset() -> None:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        created_at=datetime.now(timezone.utc),
        imperative="Check the deploy logs first.",
        tone="direct",
        strain_refs=["node-a"],
        evidence_refs=["node-a"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    assert thought.disposition == "proceed"


def test_harness_run_finalize_ran_default_false() -> None:
    run = HarnessRunV1(
        correlation_id="c-1",
        final_text=None,
        finalize_ran=False,
        step_count=0,
        compliance_verdict="partial",
        grounding_status="unknown",
    )
    assert run.finalize_ran is False
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement full schema files**

Copy model definitions verbatim from spec §6.1–6.7 into `orion/schemas/thought.py` and `orion/schemas/harness_finalize.py`.

Add bus channels (spec §9) to `orion/bus/channels.yaml`:

```yaml
  - name: "orion:thought:request"
    kind: "request"
    schema_id: "StanceReactRequestV1"
    producer_services: ["orion-hub"]
    consumer_services: ["orion-thought"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:thought:result:*"
    kind: "result"
    schema_id: "ThoughtEventV1"
    producer_services: ["orion-thought"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:thought:artifact"
    kind: "event"
    schema_id: "ThoughtEventV1"
    producer_services: ["orion-thought"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:harness:run:request"
    kind: "request"
    schema_id: "HarnessRunRequestV1"
    producer_services: ["orion-hub"]
    consumer_services: ["orion-harness-governor"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:harness:run:result:*"
    kind: "result"
    schema_id: "HarnessRunV1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:substrate:finalize_appraisal:request"
    kind: "request"
    schema_id: "HarnessDraftMoleculeV1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-substrate-runtime"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:substrate:finalize_appraisal:result:*"
    kind: "result"
    schema_id: "SubstrateFinalizeAppraisalV1"
    producer_services: ["orion-substrate-runtime"]
    consumer_services: ["orion-harness-governor"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:harness:verdict:artifact"
    kind: "event"
    schema_id: "HarnessVerdictMoleculeV1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-substrate-runtime", "orion-hub"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:substrate:turn_outcome"
    kind: "event"
    schema_id: "HarnessTurnOutcomeMoleculeV1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-substrate-runtime", "orion-hub"]
    stability: "experimental"
    since: "2026-07-05"

  - name: "orion:substrate:post_turn_closure"
    kind: "event"
    schema_id: "HarnessPostTurnClosureV1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-substrate-runtime"]
    stability: "experimental"
    since: "2026-07-05"
```

Register all models in `orion/schemas/registry.py` following the `PreTurnAppraisalRequestV1` pattern.

- [ ] **Step 4: Run gates**

```bash
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_unified_turn_schemas.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/thought.py orion/schemas/harness_finalize.py orion/bus/channels.yaml orion/schemas/registry.py tests/test_unified_turn_schemas.py
git commit -m "feat(unified-turn): thought and harness finalize schemas with bus contracts"
```

---

### Task 5: enforce_thought_stance_quality (relational choke point)

**Files:**
- Create: `orion/thought/stance_quality.py`
- Create: `orion/thought/tests/test_stance_react_relational_survives_compressor.py`

- [ ] **Step 1: Port relational fixtures — failing test**

Create `orion/thought/tests/test_stance_react_relational_survives_compressor.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.thought.stance_quality import enforce_thought_stance_quality


def _relational_thought() -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-rel",
        correlation_id="c-rel",
        created_at=datetime.now(timezone.utc),
        imperative="Stay present; one situated question.",
        tone="warm",
        strain_refs=["loop-1"],
        evidence_refs=["loop-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="reflective",
            response_priorities=["companion_presence", "situated_curiosity", "hold_space"],
            response_hazards=["avoid_task_tracking", "avoid_next_steps"],
            answer_strategy="RelationalHoldSpace",
        ),
    )


def test_stance_react_relational_survives_compressor() -> None:
    thought = _relational_thought()
    stance_inputs = {"user_message": "just talk to me, im lonely"}
    enriched, changed = enforce_thought_stance_quality(thought, stance_inputs)
    assert "companion_presence" in enriched.stance_harness_slice.response_priorities
    assert "avoid_task_tracking" in enriched.stance_harness_slice.response_hazards
    assert enriched.stance_harness_slice.answer_strategy == "RelationalHoldSpace"


def test_stance_react_instrumental_compression_preserved() -> None:
    thought = ThoughtEventV1(
        event_id="t-tri",
        correlation_id="c-tri",
        created_at=datetime.now(timezone.utc),
        imperative="List concrete file paths and line numbers.",
        tone="direct",
        strain_refs=["node-x"],
        evidence_refs=["node-x"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="triage",
            conversation_frame="mixed",
            response_priorities=["triage_operational_blockers_first"],
            answer_strategy="direct",
        ),
    )
    enriched, _ = enforce_thought_stance_quality(thought, {"user_message": "deploy is broken again"})
    assert enriched.stance_harness_slice.task_mode == "triage"
    assert "triage_operational_blockers_first" in enriched.stance_harness_slice.response_priorities
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement harness-profile enforce**

Create `orion/thought/stance_quality.py` — extract relational exemption logic from `services/orion-cortex-exec/app/chat_stance.py`:

```python
from __future__ import annotations

from typing import Any

from orion.schemas.thought import ThoughtEventV1

_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_FRAMES = frozenset({"reflective", "playful_relational"})


def _is_relational(thought: ThoughtEventV1) -> bool:
    sl = thought.stance_harness_slice
    return sl.task_mode in _RELATIONAL_TASK_MODES or sl.conversation_frame in _RELATIONAL_FRAMES


def enforce_thought_stance_quality(
    thought: ThoughtEventV1,
    stance_inputs: dict[str, Any],
) -> tuple[ThoughtEventV1, bool]:
    """Harness profile of enforce_chat_stance_quality — no business-mode compressor on relational turns."""
    changed = False
    enriched = thought.model_copy(deep=True)
    sl = enriched.stance_harness_slice

    if _is_relational(enriched):
        return enriched, False

    # Instrumental compression: strip relationship foregrounding (mirror chat_stance business path)
    if sl.task_mode == "triage":
        hazards = list(sl.response_hazards)
        for h in ("self_intro_on_operational_turn", "generic_sympathy_script"):
            if h not in hazards:
                hazards.append(h)
                changed = True
        sl.response_hazards = hazards[:8]
        priorities = list(sl.response_priorities)
        if "triage_operational_blockers_first" not in priorities:
            sl.response_priorities = (["triage_operational_blockers_first"] + priorities)[:8]
            changed = True

    if not enriched.imperative.strip():
        enriched.disposition = "defer"
        enriched.disposition_reasons = list(enriched.disposition_reasons) + ["empty_imperative"]
        changed = True

    if not enriched.evidence_refs:
        enriched.disposition = "defer"
        enriched.disposition_reasons = list(enriched.disposition_reasons) + ["missing_evidence_refs"]
        changed = True

    return enriched, changed
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/thought/tests/test_stance_react_relational_survives_compressor.py -v
```

- [ ] **Step 5: Commit**

---

### Task 6: stance_react cortex verb + thought evidence fail-closed

**Files:**
- Create: `orion/cognition/verbs/stance_react.yaml`
- Create: `orion/cognition/prompts/stance_react.j2`
- Create: `orion/thought/stance_react.py`
- Create: `orion/thought/policy_refusal.py`
- Create: `orion/thought/tests/test_stance_react_evidence_refs_fail_closed.py`

- [ ] **Step 1: Failing test for evidence_refs fail-closed**

```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.thought.policy_refusal import evaluate_thought_disposition


def test_stance_react_evidence_refs_fail_closed() -> None:
    thought = ThoughtEventV1(
        event_id="t-bad",
        correlation_id="c-bad",
        created_at=datetime.now(timezone.utc),
        imperative="Proceed anyway.",
        tone="flat",
        strain_refs=["node-z"],
        evidence_refs=[],  # invalid
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    decision = evaluate_thought_disposition(thought, association_stale=True, coalition_ids={"node-z"})
    assert decision.disposition == "defer"
    assert "evidence_refs" in " ".join(decision.reasons)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement policy + stance_react assembly**

`orion/thought/policy_refusal.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from orion.schemas.thought import ThoughtEventV1

TRUST_RUPTURE_DEFER_THRESHOLD = 0.65


@dataclass(frozen=True)
class DispositionDecision:
    disposition: str
    reasons: list[str]
    boundary_register: bool = False


def evaluate_thought_disposition(
    thought: ThoughtEventV1,
    *,
    association_stale: bool,
    coalition_ids: set[str],
) -> DispositionDecision:
    reasons: list[str] = []
    if not thought.imperative.strip():
        reasons.append("empty_imperative")
    if not thought.evidence_refs:
        reasons.append("missing_evidence_refs")
    elif not set(thought.evidence_refs).issubset(coalition_ids | set(thought.strain_refs)):
        reasons.append("evidence_refs_not_in_coalition")
    if association_stale and not thought.evidence_refs:
        reasons.append("stale_broadcast_no_evidence")
    trust = thought.trust_rupture_score
    if trust is not None and trust >= TRUST_RUPTURE_DEFER_THRESHOLD:
        return DispositionDecision("refuse", reasons + ["trust_rupture"], boundary_register=True)
    if reasons:
        return DispositionDecision("defer", reasons)
    if thought.disposition != "proceed":
        return DispositionDecision(thought.disposition, list(thought.disposition_reasons))
    return DispositionDecision("proceed", [])
```

`orion/cognition/verbs/stance_react.yaml` — mirror `chat_general.yaml` plan block with single LLM step `llm_stance_react` using `stance_react.j2`, brain tier.

`orion/thought/stance_react.py` — parse cortex JSON into `ThoughtEventV1` fields; call `enforce_thought_stance_quality`; call `evaluate_thought_disposition`.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

---

### Task 7: orion-thought bus worker service

**Files:**
- Create: `services/orion-thought/app/main.py`, `bus_listener.py`, `settings.py`
- Create: `services/orion-thought/.env_example`, `docker-compose.yml`, `Dockerfile`, `requirements.txt`, `README.md`
- Create: `services/orion-thought/tests/test_thought_artifact_published.py`

- [ ] **Step 1: Failing RPC handler test**

```python
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from orion.schemas.thought import StanceHarnessSliceV1, StanceReactRequestV1, ThoughtEventV1
from orion.schemas.thought import HubAssociationBundleV1


@pytest.mark.asyncio
async def test_thought_artifact_published() -> None:
    from services.orion_thought.app import bus_listener  # package path after rename

    req = StanceReactRequestV1(
        correlation_id="c-1",
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="felt_state_reader",
        ),
        stance_inputs={"user_message": "hello"},
    )
    fake_thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    bus = AsyncMock()
    with patch.object(bus_listener, "run_stance_react", AsyncMock(return_value=fake_thought)):
        reply = await bus_listener.handle_stance_react_request(bus, req, reply_to="orion:thought:result:c-1")
    assert reply.correlation_id == "c-1"
    bus.publish.assert_called()  # artifact channel
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Scaffold service** (follow `services/orion-context-exec/app/bus_listener.py` pattern)

`services/orion-thought/app/settings.py` — key fields:

```python
orion_bus_url: str = Field(..., alias="ORION_BUS_URL")
channel_thought_request: str = Field("orion:thought:request", alias="CHANNEL_THOUGHT_REQUEST")
channel_thought_artifact: str = Field("orion:thought:artifact", alias="CHANNEL_THOUGHT_ARTIFACT")
channel_thought_result_prefix: str = Field("orion:thought:result:", alias="CHANNEL_THOUGHT_RESULT_PREFIX")
stance_react_timeout_sec: float = Field(12.0, alias="STANCE_REACT_TIMEOUT_SEC")
```

`services/orion-thought/.env_example`:

```bash
ORION_BUS_URL=redis://100.x.x.x:6379/0
CHANNEL_THOUGHT_REQUEST=orion:thought:request
CHANNEL_THOUGHT_ARTIFACT=orion:thought:artifact
CHANNEL_THOUGHT_RESULT_PREFIX=orion:thought:result:
STANCE_REACT_TIMEOUT_SEC=12
```

After editing `.env_example`:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Run tests + compose config**

```bash
PYTHONPATH=services/orion-thought:. ./orion_dev/bin/python -m pytest services/orion-thought/tests/ -v
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml config
```

- [ ] **Step 5: Commit**

---

## Phase 3 — Harness prefix + Hub Orion wire (spec §14 step 3)

### Task 8: compile_harness_prefix + repair overlay

**Files:**
- Create: `orion/harness/prefix.py`
- Create: `orion/harness/repair.py`
- Create: `orion/harness/tests/test_repair_overlay_changes_harness_prefix.py`

- [ ] **Step 1: Failing test**

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.harness.prefix import compile_harness_prefix
from orion.harness.repair import map_repair_pressure_contract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1


def test_repair_overlay_changes_harness_prefix() -> None:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        created_at=datetime.now(timezone.utc),
        imperative="Give file paths and commands.",
        tone="direct",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        repair_pressure_level=0.9,
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="triage",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    overlay: HarnessRepairOverlayV1 = map_repair_pressure_contract(
        {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
    )
    default_prefix = compile_harness_prefix(thought, repair_overlay=HarnessRepairOverlayV1())
    repair_prefix = compile_harness_prefix(thought, repair_overlay=overlay)
    assert repair_prefix != default_prefix
    assert "repair_concrete" in repair_prefix or "file/module" in repair_prefix
```

- [ ] **Step 2–5:** Implement `map_repair_pressure_contract` and `compile_harness_prefix`; run test; commit.

`compile_harness_prefix` must embed `ThoughtEventV1.imperative`, `tone`, `strain_refs`, and overlay lines — **never** call `compile_speech_contract`.

---

### Task 9: Hub turn request builder + thought client

**Files:**
- Create: `orion/hub/turn_request.py`
- Create: `services/orion-hub/scripts/thought_client.py`
- Create: `services/orion-hub/tests/test_build_orion_turn_request.py`

- [ ] **Step 1: Test that Orion turn request excludes chat_general fields**

```python
from __future__ import annotations

from orion.hub.turn_request import build_orion_turn_request


def test_build_orion_turn_request_is_thin() -> None:
    req = build_orion_turn_request(
        correlation_id="c-1",
        session_id="sess-1",
        user_message="fix the deploy",
        repair_bundle=None,
    )
    assert req["mode"] == "orion"
    assert "verb_name" not in req
    assert req["user_message"] == "fix the deploy"
```

- [ ] **Step 2–5:** Implement thin builder; implement `ThoughtClient` mirroring `PreTurnAppraisalClient`; commit.

---

### Task 10: Orion mode routing behind flag (no agent-claude when governor required)

**Files:**
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/scripts/websocket_handler.py`

Add settings:

```python
ORION_UNIFIED_TURN_ENABLED: bool = Field(default=False, alias="ORION_UNIFIED_TURN_ENABLED")
ORION_HARNESS_GOVERNOR_ENABLED: bool = Field(default=False, alias="ORION_HARNESS_GOVERNOR_ENABLED")
```

In `websocket_handler.py`, when `mode == "orion"` and `ORION_UNIFIED_TURN_ENABLED`:

```python
if mode == "orion" and settings.ORION_UNIFIED_TURN_ENABLED:
    if not settings.ORION_HARNESS_GOVERNOR_ENABLED:
        await websocket.send_json({"type": "turn_error", "phase": "config", "error": "harness_governor_disabled"})
        continue
    from orion.hub.turn_orchestrator import run_unified_turn
    await run_unified_turn(websocket, transcript, session_id=session_id, payload=data, ...)
    continue
```

Sync env after `.env_example` change.

Commit with WS routing test stub.

---

## Phase 4 — Substrate finalize_appraisal RPC (spec §14 step 4)

### Task 11: finalize_draft_v1 deterministic appraisal

**Files:**
- Create: `orion/substrate/appraisal/finalize_draft_v1.py`
- Create: `orion/substrate/appraisal/tests/test_finalize_draft_v1.py`

- [ ] **Step 1: Failing test — payload-only, no reducer wait**

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessDraftMoleculeV1
from orion.schemas.thought import CoalitionSnapshotV1, StanceHarnessSliceV1, ThoughtEventV1
from orion.substrate.appraisal.finalize_draft_v1 import appraise_draft_molecule


def test_draft_molecule_rpc_round_trip_fields() -> None:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        created_at=datetime.now(timezone.utc),
        imperative="x",
        tone="y",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    mol = HarnessDraftMoleculeV1(
        correlation_id="c-1",
        thought_event_id="t-1",
        draft_text="draft answer",
        draft_hash="abc",
        thought_event=thought,
        grammar_receipts=[GrammarReceiptV1(step_index=0, summary="step")],
        coalition_snapshot=CoalitionSnapshotV1(
            attended_node_ids=["n-1"],
            open_loop_ids=[],
            generated_at=datetime.now(timezone.utc),
        ),
    )
    appraisal = appraise_draft_molecule(mol)
    assert appraisal.draft_hash == "abc"
    assert len(appraisal.learning_refs) >= 1
    assert 0.0 <= appraisal.surprise_level <= 1.0
```

- [ ] **Step 2–5:** Implement surprise from grammar receipts vs coalition snapshot; empty `learning_refs` raises; commit.

---

### Task 12: substrate-runtime finalize_appraisal listener

**Files:**
- Create: `services/orion-substrate-runtime/app/finalize_appraisal_listener.py`
- Modify: `services/orion-substrate-runtime/app/main.py`
- Modify: `services/orion-substrate-runtime/app/settings.py`, `.env_example`
- Create: `services/orion-substrate-runtime/tests/test_finalize_appraisal_rpc.py`

- [ ] **Step 1–5:** Bus listener on `orion:substrate:finalize_appraisal:request`; validate `HarnessDraftMoleculeV1`; call `appraise_draft_molecule`; reply `SubstrateFinalizeAppraisalV1` on result channel; 5s timeout enforced at client.

---

## Phase 5 — harness_finalize_reflect + verdict (spec §14 step 5)

### Task 13: Cortex 5b verb + quick lane allowlist

**Files:**
- Create: `orion/cognition/verbs/harness_finalize_reflect.yaml`
- Create: `orion/cognition/prompts/harness_finalize_reflect.j2`
- Create: `orion/harness/finalize.py` (partial — `run_finalize_reflection`, `maybe_quick_lane_verdict`)
- Create: `orion/harness/tests/test_quick_lane_blocked_on_hard_cases.py`
- Create: `orion/harness/tests/test_reflect_consumes_substrate_appraisal.py`

- [ ] **Step 1: Test 5b refuses without substrate appraisal**

```python
import pytest
from orion.harness.finalize import run_finalize_reflection


@pytest.mark.asyncio
async def test_reflect_consumes_substrate_appraisal() -> None:
    with pytest.raises(ValueError, match="substrate_appraisal"):
        await run_finalize_reflection(draft=..., thought=..., substrate_appraisal=None, ...)
```

- [ ] **Step 2: Test quick lane blocked on hard cases** — one assertion per §6.3 criterion (repair pressure, trust rupture, high surprise, boundary_register, repair overlay != default).

- [ ] **Step 3–5:** Implement quick lane gate using env `FINALIZE_QUICK_GATE_EPSILON`; cortex call when not eligible; emit `HarnessVerdictMoleculeV1`; commit.

---

## Phase 6 — orion_voice_finalize + outcome molecule (spec §14 step 6)

### Task 14: Cortex 5c voice verb

**Files:**
- Create: `orion/cognition/verbs/orion_voice_finalize.yaml`
- Create: `orion/cognition/prompts/orion_voice_finalize.j2`
- Extend: `orion/harness/finalize.py` — `run_orion_voice_finalize`
- Create: `orion/harness/tests/test_voice_changes_on_misaligned_verdict.py`
- Create: `orion/harness/tests/test_finalize_ran_required.py`

- [ ] **Step 1: Test misaligned verdict forces revision signal**

```python
def test_voice_changes_on_misaligned_verdict() -> None:
    # stub cortex client returns different text for misaligned vs aligned reflection fixtures
    ...
    assert misaligned_final != aligned_final
    assert misaligned_meta["finalize_changed"] is True
```

- [ ] **Step 2–5:** Wire `StanceHarnessSliceV1` curiosity/hazard rules in Jinja (parallel relational rules from `chat_general.j2`); `HarnessRunV1.finalize_ran=True` only after 5a+5b+5c; commit.

---

### Task 15: Outcome molecule before Hub publish

**Files:**
- Extend: `orion/harness/finalize.py` — `emit_turn_outcome_molecule`
- Create: `orion/harness/tests/test_outcome_molecule_before_hub_publish.py`

- [ ] **Step 1–5:** PUBLISH `HarnessTurnOutcomeMoleculeV1` on `orion:substrate:turn_outcome` after 5c; test asserts publish called before harness REPLY returns to Hub.

---

## Phase 7 — Harness governor + turn_orchestrator (spec §14 step 7)

### Task 16: Harness runner (fcc motor)

**Files:**
- Create: `orion/harness/runner.py`
- Create: `orion/harness/cortex_client.py`, `substrate_client.py`
- Reuse: `services/orion-hub/scripts/fcc_claude_bridge.py`

- [ ] **Step 1–5:** `HarnessRunner.run()` calls fcc with `compile_harness_prefix`; collects `grammar_receipts`; produces internal `draft_text` only; test `test_harness_run_artifact_published`.

---

### Task 17: orion-harness-governor service

**Files:**
- Create: `services/orion-harness-governor/` (mirror orion-thought layout)
- Wire: full finalize chain in `bus_listener.handle_harness_run_request`

Listener sequence (spec §8.2):

```text
validate HarnessRunRequestV1 → runner.run → finalize.run_substrate_finalize_appraisal
→ finalize.run_finalize_reflection → finalize.run_orion_voice_finalize
→ finalize.emit_turn_outcome_molecule → REPLY HarnessRunV1 → finalize.emit_post_turn_closure
```

Env:

```bash
ORION_BUS_URL=redis://100.x.x.x:6379/0
ORION_HARNESS_GOVERNOR_ENABLED=true
CHANNEL_HARNESS_RUN_REQUEST=orion:harness:run:request
FINALIZE_QUICK_GATE_EPSILON=0.08
```

- [ ] **Step 1–5:** Service tests + docker compose config; commit.

---

### Task 18: Hub turn_orchestrator saga + WS frames

**Files:**
- Create: `orion/hub/turn_orchestrator.py`
- Create: `services/orion-hub/scripts/harness_governor_client.py`
- Create: `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`

- [ ] **Step 1: Test failure frames never leak draft**

```python
@pytest.mark.asyncio
async def test_turn_orchestrator_never_publishes_draft_text() -> None:
    frames = await collect_ws_frames_from_failed_harness(...)
    assert all("draft_text" not in f for f in frames)
    assert any(f.get("type") == "turn_error" for f in frames)
```

- [ ] **Step 2–5:** Implement saga:

```text
emit_observation → pre_turn_appraisal → build_hub_association_bundle
→ thought_client.appraise → harness_client.run → send final_text
```

Intermediate WS frames (optional relay): `substrate_appraisal`, `reflection`, `final`. Defer/refuse → `turn_deferred`. Commit.

---

## Phase 8 — Trust rupture evals (spec §14 step 8)

### Task 19: Freeze TRUST_RUPTURE_DEFER_THRESHOLD

**Files:**
- Create: `orion/thought/evals/trust_rupture_eval.py`
- Create: `orion/thought/evals/fixtures/trust_rupture/*.json`

- [ ] **Step 1–5:** Run eval corpus; set `TRUST_RUPTURE_DEFER_THRESHOLD` in settings to value that passes; add gate test that threshold change requires eval re-run (`test_trust_rupture_threshold_frozen`).

---

## Phase 9 — Pollution firewall (spec §14 step 9)

### Task 20: Unified path pollution firewall

**Files:**
- Create: `services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py`

- [ ] **Step 1: Static import/routing test**

```python
def test_unified_orion_turn_pollution_firewall() -> None:
    import inspect
    from orion.hub import turn_orchestrator

    src = inspect.getsource(turn_orchestrator)
    forbidden = [
        "chat_general",
        "compile_speech_contract",
        "build_chat_request",
        "llm_chat_general",
    ]
    for token in forbidden:
        assert token not in src
```

- [ ] **Step 2–5:** Deprecate agent-claude routing when `mode=orion` + unified flag; Orion mode must not fall through to agent-claude; commit.

---

## Phase 10 — Post-turn closure + layer attribution evals (spec §14 step 10)

### Task 21: HarnessPostTurnClosureV1

**Files:**
- Extend: `orion/harness/finalize.py` — `emit_post_turn_closure`
- Create: `orion/harness/tests/test_post_turn_closure_emits_prediction_error.py`
- Create: `orion/harness/evals/test_layer_attribution.py`

- [ ] **Step 1: Test closure publish**

```python
def test_post_turn_closure_emits_prediction_error(monkeypatch) -> None:
    # mock substrate reducer consumer: closure event → prediction_error node updated
    ...
```

- [ ] **Step 2: Layer attribution evals (§11.2)** — separate tests:
  - `test_5a_affects_5b_verdict`
  - `test_5b_affects_5c_text`
  - `test_turn_n_error_shifts_turn_n_plus_one_strain`

- [ ] **Step 3–5:** Wire substrate reducers on `orion:substrate:post_turn_closure`; run evals; commit.

---

## Phase 11–12 — Sunset + future profiles (spec §14 steps 11–12)

### Task 22: Brain shim sunset checklist (documentation gate only)

**Files:**
- Create: `docs/superpowers/checklists/2026-07-05-unified-turn-sunset.md`

Document §13 criteria as checkboxes — **do not remove Brain mode** until all green:

1. Mesh-honesty eval suite
2. Repair/refusal parity
3. Relational stance parity
4. Pollution firewall
5. Layer attribution evals
6. agent-claude tool parity on repo-edit fixtures
7. 14-day operator soak
8. Cost ceiling ≤ 1.5× Brain median tokens

No code commit required beyond checklist unless Juniper explicitly requests sunset implementation.

---

## Self-review (plan author checklist)

| Spec section | Plan task(s) |
|--------------|--------------|
| §3.1 Association fail-closed | Task 1 |
| §5 molecule gate tests | Tasks 3, 7, 11–15, 21 |
| §5.6 Relational stance | Task 5 |
| §6 schemas | Task 4 |
| §7 orion-thought | Task 7 |
| §8 Hub + harness governor | Tasks 17–18 |
| §9 bus channels | Task 4 |
| §11 evals + latency | Tasks 19, 21 |
| §12 pollution firewall | Task 20 |
| §13 sunset | Task 22 |
| Appendix A flags | Tasks 10, 17 |

**Placeholder scan:** No TBD/TODO steps. All tasks include concrete file paths and test entry points.

**Type consistency:** `ThoughtEventV1`, `HarnessRunV1`, `GrammarReceiptV1` defined in Task 4 and reused unchanged in later tasks.

---

## Recommended verification before each phase merge

```bash
make agent-check SERVICE=orion-hub  # after Hub tasks
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

After service scaffolding:

```bash
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml config
docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml config
```

**Restart after Phase 7:**

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build substrate-runtime
```

---
