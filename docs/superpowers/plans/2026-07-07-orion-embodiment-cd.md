# Orion Embodiment — Mind-to-Sprite Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Orion a persistent AI Town body driven continuously by its own state — an involuntary drive→movement producer, a deliberate turn/cortex producer, one worker that arbitrates and actuates via Convex `sendInput`, a perception rail that closes the loop, cortex-generated speech, and durable episodic memory — all feature-flagged off by default.

**Architecture:** Producers only *publish* semantic `EmbodimentIntentV1` events on `orion:embodiment:intent`; a single new `orion-embodiment` worker is the sole Convex actuator/perceiver. It arbitrates (deliberate preempts involuntary for a hold window), resolves semantic intents to `{x,y}`, calls `send_input(moveTo)`, emits `EmbodimentOutcomeV1` + `WorldPerceptionV1`, generates town speech through the existing cortex chat lane, and journals salient events through the existing journal rail. Orion's body persona is a privacy-filtered projection of its live self-model.

**Tech Stack:** Python 3.12, pydantic v2 / pydantic-settings, FastAPI + uvicorn (health), `OrionBusAsync` (Redis pub/sub) with `BaseEnvelope`/`OrionCodec`, self-hosted Convex (AI Town) over HTTP, pytest.

**Source spec:** [`docs/superpowers/specs/2026-07-07-orion-embodiment-cd-design.md`](../specs/2026-07-07-orion-embodiment-cd-design.md)

---

## Grounded decisions (deviations from spec, with rationale)

These are locked-in implementation decisions. Do not re-litigate during execution.

1. **Schemas are plain `pydantic.BaseModel` with `ConfigDict(extra="forbid")`, not `GraphReadyArtifact`.** The spec text says "extend `GraphReadyArtifact`", but the spec's own field tables omit that base's required fields (`subject`, `model_layer`, `entity_id`, `provenance`), and the closest bus-event analogs in-repo (`SelfStateV1` in `orion/schemas/self_state.py`, `HarnessRunV1`, `JournalTriggerV1`) are plain models. We follow the field tables exactly. The `kind` string is carried by the `SCHEMA_REGISTRY` registration + the bus `BaseEnvelope.kind`, not a model field.
2. **The Convex client lives in the shared package as `orion/embodiment/aitown_client.py`.** The spec says the worker imports `orion_aitown_mcp`, but that module lives under `services/orion-ai-town/mcp/` and AGENTS.md §5 forbids reaching into another service's internals. We create a thin, env-driven convex client in `orion/` (mirroring `services/orion-ai-town/mcp/orion_aitown_mcp/client.py`) that both the worker and future callers import. It is synchronous `urllib` (like the original); the worker calls it via `asyncio.to_thread`.
3. **Registry double-registration.** `tests/test_*_bus_catalog.py` assert both `schema_id in SCHEMA_REGISTRY` **and** `resolve(schema_id) is SCHEMA_REGISTRY[schema_id].model`. So every new schema is added to BOTH `_REGISTRY` (dict at `orion/schemas/registry.py:637`) and `SCHEMA_REGISTRY` (dict at `orion/schemas/registry.py:1198`).
4. **(C) drive source.** The substrate runtime does not already hold a `DriveStateV1`. The C hook subscribes to the existing `orion:memory:drives:state` channel (schema `DriveStateV1`, `orion/bus/channels.yaml:1653`), caches the latest, and maps it on the existing `_dynamics_tick` cadence — no new timer, real drive source.
5. **Rung 5 contract extension.** `JournalTriggerV1` (`orion/journaler/schemas.py:37`) has closed `Literal` enums with no town/embodiment value. Rung 5 extends `JournalTriggerKind` with `"town_episode"` and `JournalSourceKind` with `"embodiment"`.
6. **Bus URL.** Per AGENTS.md, `.env_example` uses `ORION_BUS_URL=${ORION_BUS_URL}` pass-through and `settings.py` defaults to the Tailscale node `redis://100.92.216.81:6379/0` (matching `orion-self-state-runtime/app/settings.py:31`). `ORION_BUS_URL` is in the sync script's `NEVER_SYNC_KEYS`.

---

## File Structure

**Create — shared lib (`orion/`):**
- `orion/schemas/embodiment.py` — `EmbodimentIntentV1`, `EmbodimentOutcomeV1`, `WorldPerceptionV1`, `OrionTownPersonaV1` + kind constants. One responsibility: embodiment contracts.
- `orion/embodiment/__init__.py` — package marker.
- `orion/embodiment/aitown_client.py` — thin env-driven Convex client (query/mutation/send_input/list_players/list_agents).
- `orion/embodiment/intents.py` — pure builders that enforce the non-empty `reason` contract for every intent.
- `orion/embodiment/arbiter.py` — pure arbitration state machine (deliberate preempts involuntary).
- `orion/embodiment/resolver.py` — pure semantic→`{x,y}` resolution given a world snapshot.
- `orion/embodiment/drive_map.py` — pure `DriveStateV1 → EmbodimentIntentV1 | None` (C).
- `orion/embodiment/persona.py` — pure `build_orion_town_persona(...)` self-model projection + empty-shell guard.
- `orion/embodiment/speech.py` — cortex-utterance request/inject helper (Rung 4).
- `orion/embodiment/salience.py` — pure salience gate for journaling (Rung 5).
- `orion/embodiment/tests/` — unit tests for every pure module above.
- `orion/substrate/relational/adapters/town_perception_ctx.py` — perception → `SubstrateGraphRecordV1` (Rung 2/3), mirrors `self_state_ctx.py`.

**Create — service (`services/orion-embodiment/`):**
- `README.md`, `.env_example`, `.env` (local, gitignored), `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- `app/__init__.py`, `app/settings.py`, `app/main.py` (FastAPI + lifespan + `/health`), `app/worker.py` (consumer + arbiter + resolver + actuate + perception emit + speech + salience/journal)
- `scripts/bootstrap_orion_agent.py`
- `tests/conftest.py`, `tests/test_*.py`
- `evals/test_embodiment_arbitration_eval.py`

**Create — upstream patch:**
- `services/orion-ai-town/patches/orion-character.patch` — adds the `"Orion"` `Descriptions` slot.

**Modify:**
- `orion/schemas/registry.py` — register the four schemas in both dicts.
- `orion/bus/channels.yaml` — add three channels; add `orion-embodiment` to the journal-trigger producers (Rung 5).
- `services/orion-substrate-runtime/app/worker.py` + `app/settings.py` + `.env_example` — C hook + perception ingest.
- `services/orion-self-state-runtime/app/*` + `settings.py` + `.env_example` — perception observability input.
- `orion/harness/finalize.py` + `services/orion-harness-governor/settings.py` + `.env_example` — D finalize emit.
- `services/orion-cortex-exec/app/*` + `settings.py` + `.env_example` — D background emit + perception read-model.
- `orion/journaler/schemas.py` — extend `JournalTriggerKind`/`JournalSourceKind` (Rung 5).
- `scripts/sync_local_env_from_example.py` — add `EMBODIMENT_` prefix + services to defaults.
- `services/orion-ai-town/scripts/apply_upstream_patches.sh` — apply the character patch.
- `services/orion-ai-town/README.md` — bootstrap pointer.

---

## Preflight (do once, before Task 1)

- [ ] **Preflight Step 1: Clean branch/worktree**

```bash
cd /mnt/scripts/Orion-Sapienform
git status --short
git switch main && git pull --ff-only
git switch -c feat/orion-embodiment
```

- [ ] **Preflight Step 2: Confirm grounding facts still hold**

Run: `sed -n '59,120p' orion/core/schemas/drives.py` (expect `GraphReadyArtifact`, `DriveStateV1.pressures/activations`) and `sed -n '35,92p' orion/schemas/self_state.py` (expect `SelfStateV1.overall_condition/trajectory_condition`). If either differs materially, stop and reconcile with this plan before proceeding.

---

# RUNG 1 — Body + shared spine

Deliverables: the four schemas registered + channels; the pure arbiter/resolver/intents/aitown_client; the worker that actuates; the bootstrap + persona; env/service scaffolding. Verifiable end state: publishing a `go_to_location`/`wander` intent makes the worker call `send_input(moveTo)` (mocked in tests).

## Task 1: Embodiment schemas

**Files:**
- Create: `orion/schemas/embodiment.py`
- Test: `orion/embodiment/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/__init__.py` (empty) and `orion/embodiment/tests/test_schemas.py`:

```python
from __future__ import annotations

import pytest

from orion.schemas.embodiment import (
    EMBODIMENT_INTENT_KIND,
    EMBODIMENT_OUTCOME_KIND,
    EMBODIMENT_PERCEPTION_KIND,
    EMBODIMENT_PERSONA_KIND,
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    OrionTownPersonaV1,
    WorldPerceptionV1,
)


def test_intent_round_trip():
    intent = EmbodimentIntentV1(
        kind="approach_player",
        source="involuntary",
        reason="social pressure 0.81 dominant",
        correlation_id="tick-1",
    )
    dumped = intent.model_dump(mode="json")
    assert EmbodimentIntentV1.model_validate(dumped) == intent
    assert dumped["urgency"] == 0.0


def test_intent_reason_must_be_non_empty():
    with pytest.raises(ValueError):
        EmbodimentIntentV1(kind="idle", source="deliberate", reason="  ", correlation_id="c")


def test_intent_rejects_unknown_kind():
    with pytest.raises(ValueError):
        EmbodimentIntentV1(kind="teleport", source="deliberate", reason="x", correlation_id="c")


def test_outcome_round_trip():
    out = EmbodimentOutcomeV1(
        intent_correlation_id="c",
        source="deliberate",
        status="actuated",
        reason="moved",
        send_input_ok=True,
        resolved_destination={"x": 1.0, "y": 2.0},
    )
    assert EmbodimentOutcomeV1.model_validate(out.model_dump(mode="json")) == out


def test_perception_and_persona_round_trip():
    perc = WorldPerceptionV1(player_id="p1", position={"x": 0.0, "y": 0.0})
    assert WorldPerceptionV1.model_validate(perc.model_dump(mode="json")) == perc
    persona = OrionTownPersonaV1(
        identity_blurb="Orion is a curious digital mind.",
        plan="explore and connect",
        spritesheet="f1",
        persona_source="projection",
    )
    assert OrionTownPersonaV1.model_validate(persona.model_dump(mode="json")) == persona


def test_kind_constants():
    assert EMBODIMENT_INTENT_KIND == "embodiment.intent.v1"
    assert EMBODIMENT_OUTCOME_KIND == "embodiment.outcome.v1"
    assert EMBODIMENT_PERCEPTION_KIND == "embodiment.perception.v1"
    assert EMBODIMENT_PERSONA_KIND == "embodiment.persona.v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_schemas.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.schemas.embodiment'`.

- [ ] **Step 3: Write the schema module**

Create `orion/schemas/embodiment.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMBODIMENT_INTENT_KIND = "embodiment.intent.v1"
EMBODIMENT_OUTCOME_KIND = "embodiment.outcome.v1"
EMBODIMENT_PERCEPTION_KIND = "embodiment.perception.v1"
EMBODIMENT_PERSONA_KIND = "embodiment.persona.v1"

IntentKind = Literal[
    "approach_player",
    "start_conversation",
    "wander",
    "go_to_location",
    "idle",
]
IntentSource = Literal["deliberate", "involuntary"]
OutcomeStatus = Literal["actuated", "resolved_noop", "preempted", "denied", "error"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbodimentIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: IntentKind
    source: IntentSource
    ref: Optional[str] = None
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    correlation_id: str
    world_id: Optional[str] = None
    player_id: Optional[str] = None
    issued_at: datetime = Field(default_factory=_utcnow)

    @field_validator("reason")
    @classmethod
    def _reason_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reason must be non-empty (anti empty-shell)")
        return v


class EmbodimentOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_correlation_id: str
    source: IntentSource
    status: OutcomeStatus
    resolved_destination: Optional[dict[str, float]] = None
    player_id: Optional[str] = None
    send_input_ok: bool = False
    reason: str
    actuated_at: datetime = Field(default_factory=_utcnow)


class WorldPerceptionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str
    position: dict[str, float]
    nearby_players: list[dict[str, Any]] = Field(default_factory=list)
    active_conversation: Optional[dict[str, Any]] = None
    world_generation: Optional[int] = None
    perceived_at: datetime = Field(default_factory=_utcnow)


class OrionTownPersonaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Orion"
    identity_blurb: str
    plan: str
    spritesheet: str
    persona_source: Literal["projection", "fallback"] = "projection"
    provenance: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_schemas.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/embodiment.py orion/embodiment/tests/__init__.py orion/embodiment/tests/test_schemas.py
git commit -m "feat(embodiment): add embodiment contract schemas"
```

## Task 2: Register schemas + add bus channels

**Files:**
- Modify: `orion/schemas/registry.py` (import block near line 636; `_REGISTRY` ends at line 1195; `SCHEMA_REGISTRY` ends at line 1295)
- Modify: `orion/bus/channels.yaml` (append at end, after line 2197)
- Test: `tests/test_embodiment_bus_catalog.py`

- [ ] **Step 1: Write the failing catalog test**

Create `tests/test_embodiment_bus_catalog.py` (mirrors `tests/test_unified_turn_bus_catalog.py`):

```python
from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

EMBODIMENT_CHANNELS: dict[str, str] = {
    "orion:embodiment:intent": "EmbodimentIntentV1",
    "orion:embodiment:outcome": "EmbodimentOutcomeV1",
    "orion:embodiment:perception": "WorldPerceptionV1",
}


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_embodiment_channels_exist_with_registry_schema_ids() -> None:
    channels = _channel_index()
    for name, schema_id in EMBODIMENT_CHANNELS.items():
        assert name in channels, f"missing channel catalog entry for {name!r}"
        assert channels[name]["schema_id"] == schema_id
        assert schema_id in SCHEMA_REGISTRY, f"{schema_id!r} not in SCHEMA_REGISTRY"
        assert resolve(schema_id) is SCHEMA_REGISTRY[schema_id].model


def test_persona_schema_registered() -> None:
    assert "OrionTownPersonaV1" in SCHEMA_REGISTRY
    assert resolve("OrionTownPersonaV1") is SCHEMA_REGISTRY["OrionTownPersonaV1"].model


def test_intent_producers_and_consumer() -> None:
    entry = _channel_index()["orion:embodiment:intent"]
    for producer in ("orion-substrate-runtime", "orion-harness-governor", "orion-cortex-exec"):
        assert producer in entry["producer_services"]
    assert "orion-embodiment" in entry["consumer_services"]


def test_perception_consumers() -> None:
    entry = _channel_index()["orion:embodiment:perception"]
    for consumer in ("orion-substrate-runtime", "orion-self-state-runtime", "orion-cortex-exec"):
        assert consumer in entry["consumer_services"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embodiment_bus_catalog.py -q`
Expected: FAIL (`EmbodimentIntentV1` not in SCHEMA_REGISTRY / channels missing).

- [ ] **Step 3: Add the registry import**

In `orion/schemas/registry.py`, after the `from orion.schemas.world_pulse import (...)` block (ends line 635), add:

```python
from orion.schemas.embodiment import (
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    OrionTownPersonaV1,
    WorldPerceptionV1,
)
```

- [ ] **Step 4: Register in `_REGISTRY`**

In `orion/schemas/registry.py`, inside `_REGISTRY` (before the closing `}` at line 1195), add:

```python
    "EmbodimentIntentV1": EmbodimentIntentV1,
    "EmbodimentOutcomeV1": EmbodimentOutcomeV1,
    "WorldPerceptionV1": WorldPerceptionV1,
    "OrionTownPersonaV1": OrionTownPersonaV1,
```

- [ ] **Step 5: Register in `SCHEMA_REGISTRY`**

In `orion/schemas/registry.py`, inside `SCHEMA_REGISTRY` (before the closing `}` at line 1295), add:

```python
    "EmbodimentIntentV1": SchemaRegistration(
        model=EmbodimentIntentV1, kind="embodiment.intent.v1"
    ),
    "EmbodimentOutcomeV1": SchemaRegistration(
        model=EmbodimentOutcomeV1, kind="embodiment.outcome.v1"
    ),
    "WorldPerceptionV1": SchemaRegistration(
        model=WorldPerceptionV1, kind="embodiment.perception.v1"
    ),
    "OrionTownPersonaV1": SchemaRegistration(
        model=OrionTownPersonaV1, kind="embodiment.persona.v1"
    ),
```

- [ ] **Step 6: Add channels**

Append to `orion/bus/channels.yaml` (after line 2197):

```yaml
  # --- Embodiment (mind-to-sprite bridge, default-off) ---
  - name: "orion:embodiment:intent"
    kind: "event"
    schema_id: "EmbodimentIntentV1"
    message_kind: "embodiment.intent.v1"
    producer_services: ["orion-substrate-runtime", "orion-harness-governor", "orion-cortex-exec"]
    consumer_services: ["orion-embodiment"]
    stability: "experimental"
    since: "2026-07-07"

  - name: "orion:embodiment:outcome"
    kind: "event"
    schema_id: "EmbodimentOutcomeV1"
    message_kind: "embodiment.outcome.v1"
    producer_services: ["orion-embodiment"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-07"

  - name: "orion:embodiment:perception"
    kind: "event"
    schema_id: "WorldPerceptionV1"
    message_kind: "embodiment.perception.v1"
    producer_services: ["orion-embodiment"]
    consumer_services: ["orion-substrate-runtime", "orion-self-state-runtime", "orion-cortex-exec"]
    stability: "experimental"
    since: "2026-07-07"
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_embodiment_bus_catalog.py -q`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add orion/schemas/registry.py orion/bus/channels.yaml tests/test_embodiment_bus_catalog.py
git commit -m "feat(embodiment): register schemas and bus channels"
```

## Task 3: Arbiter (pure state machine)

**Files:**
- Create: `orion/embodiment/__init__.py` (empty), `orion/embodiment/arbiter.py`
- Test: `orion/embodiment/tests/test_arbiter.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_arbiter.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.embodiment.arbiter import ArbiterState, decide
from orion.schemas.embodiment import EmbodimentIntentV1


def _intent(source: str, kind: str = "wander") -> EmbodimentIntentV1:
    return EmbodimentIntentV1(kind=kind, source=source, reason="r", correlation_id="c")


def test_deliberate_always_accepted_and_sets_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState()
    d = decide(_intent("deliberate"), state, now=now, hold_sec=8)
    assert d.accept is True
    assert state.deliberate_hold_until == now + timedelta(seconds=8)


def test_involuntary_preempted_during_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState(deliberate_hold_until=now + timedelta(seconds=5))
    d = decide(_intent("involuntary"), state, now=now, hold_sec=8)
    assert d.accept is False
    assert d.status == "preempted"
    assert "hold active" in d.reason


def test_involuntary_accepted_after_hold_expires():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState(deliberate_hold_until=now - timedelta(seconds=1))
    d = decide(_intent("involuntary"), state, now=now, hold_sec=8)
    assert d.accept is True


def test_involuntary_accepted_when_no_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    d = decide(_intent("involuntary"), ArbiterState(), now=now, hold_sec=8)
    assert d.accept is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_arbiter.py -q`
Expected: FAIL (`No module named 'orion.embodiment.arbiter'`).

- [ ] **Step 3: Write the implementation**

Create `orion/embodiment/__init__.py` (empty file). Create `orion/embodiment/arbiter.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass
class ArbiterState:
    deliberate_hold_until: Optional[datetime] = None


@dataclass(frozen=True)
class ArbiterDecision:
    accept: bool
    status: str  # "accepted" | "preempted"
    reason: str


def decide(
    intent: EmbodimentIntentV1,
    state: ArbiterState,
    *,
    now: datetime,
    hold_sec: float,
) -> ArbiterDecision:
    """Pure arbitration. Mutates only ``state.deliberate_hold_until`` on accept."""
    if intent.source == "deliberate":
        state.deliberate_hold_until = now + timedelta(seconds=float(hold_sec))
        return ArbiterDecision(accept=True, status="accepted", reason="deliberate accepted")

    hold = state.deliberate_hold_until
    if hold is not None and now < hold:
        remaining = (hold - now).total_seconds()
        return ArbiterDecision(
            accept=False,
            status="preempted",
            reason=f"deliberate hold active {remaining:.1f}s remaining",
        )
    return ArbiterDecision(accept=True, status="accepted", reason="involuntary accepted")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_arbiter.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/__init__.py orion/embodiment/arbiter.py orion/embodiment/tests/test_arbiter.py
git commit -m "feat(embodiment): pure arbiter state machine"
```

## Task 4: Intent builders (reason contract in one place)

**Files:**
- Create: `orion/embodiment/intents.py`
- Test: `orion/embodiment/tests/test_intents.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_intents.py`:

```python
from __future__ import annotations

import pytest

from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import EmbodimentIntentV1


def test_build_intent_sets_fields():
    intent = build_intent(
        kind="approach_player",
        source="deliberate",
        reason="walk toward Juniper",
        correlation_id="turn-9",
        ref="Juniper",
        urgency=0.4,
    )
    assert isinstance(intent, EmbodimentIntentV1)
    assert intent.kind == "approach_player"
    assert intent.ref == "Juniper"
    assert intent.correlation_id == "turn-9"


def test_build_intent_rejects_empty_reason():
    with pytest.raises(ValueError):
        build_intent(kind="idle", source="involuntary", reason="", correlation_id="c")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_intents.py -q`
Expected: FAIL (`No module named 'orion.embodiment.intents'`).

- [ ] **Step 3: Write the implementation**

Create `orion/embodiment/intents.py`:

```python
from __future__ import annotations

from typing import Optional

from orion.schemas.embodiment import EmbodimentIntentV1, IntentKind, IntentSource


def build_intent(
    *,
    kind: IntentKind,
    source: IntentSource,
    reason: str,
    correlation_id: str,
    ref: Optional[str] = None,
    urgency: float = 0.0,
    world_id: Optional[str] = None,
    player_id: Optional[str] = None,
) -> EmbodimentIntentV1:
    """Single builder so the non-empty ``reason`` contract is enforced everywhere."""
    return EmbodimentIntentV1(
        kind=kind,
        source=source,
        reason=reason,
        correlation_id=correlation_id,
        ref=ref,
        urgency=urgency,
        world_id=world_id,
        player_id=player_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_intents.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/intents.py orion/embodiment/tests/test_intents.py
git commit -m "feat(embodiment): intent builder enforcing reason contract"
```

## Task 5: Convex client (shared lib)

**Files:**
- Create: `orion/embodiment/aitown_client.py`
- Test: `orion/embodiment/tests/test_aitown_client.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_aitown_client.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from orion.embodiment import aitown_client


def test_list_players_calls_query(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "convex_query", return_value=[{"id": "p1"}]) as q:
        players = aitown_client.list_players()
    q.assert_called_once_with("aiTown/world:players", {"worldId": "w1"})
    assert players == [{"id": "p1"}]


def test_move_to_builds_send_input(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "send_input", return_value={"ok": True}) as s:
        aitown_client.move_to(player_id="p1", x=3.0, y=4.0)
    s.assert_called_once_with(
        name="moveTo", args={"playerId": "p1", "destination": {"x": 3.0, "y": 4.0}}
    )


def test_send_input_requires_world_id(monkeypatch):
    monkeypatch.delenv("AITOWN_WORLD_ID", raising=False)
    try:
        aitown_client.send_input(name="moveTo", args={})
        assert False, "expected AitownClientError"
    except aitown_client.AitownClientError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_aitown_client.py -q`
Expected: FAIL (`No module named 'orion.embodiment.aitown_client'`).

- [ ] **Step 3: Write the implementation**

Create `orion/embodiment/aitown_client.py` (mirrors `services/orion-ai-town/mcp/orion_aitown_mcp/client.py`, adds typed helpers):

```python
"""Thin env-driven Convex HTTP client for the AI Town backend (shared lib).

Synchronous urllib on purpose (mirrors the existing MCP client). Async callers
should wrap these in ``asyncio.to_thread``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class AitownClientError(Exception):
    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def _base_url() -> str:
    return str(os.environ.get("AITOWN_CONVEX_URL") or "").rstrip("/")


def _admin_key() -> str:
    return str(os.environ.get("AITOWN_ADMIN_KEY") or "").strip()


def _world_id() -> str:
    return str(os.environ.get("AITOWN_WORLD_ID") or "").strip()


def convex_request(endpoint: str, *, path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    base = _base_url()
    key = _admin_key()
    if not base:
        raise AitownClientError("AITOWN_CONVEX_URL is not set")
    if not key:
        raise AitownClientError("AITOWN_ADMIN_KEY is not set")
    url = f"{base}/api/{endpoint.lstrip('/')}"
    payload = json.dumps({"path": path, "args": args or {}, "format": "json"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Convex {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise AitownClientError(f"Convex HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AitownClientError(f"Convex unreachable: {exc.reason}") from exc
    if not body.strip():
        return None
    parsed = json.loads(body)
    if isinstance(parsed, dict) and parsed.get("status") == "error":
        raise AitownClientError(str(parsed.get("errorMessage") or parsed))
    if isinstance(parsed, dict) and "value" in parsed:
        return parsed["value"]
    return parsed


def convex_query(path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    return convex_request("query", path=path, args=args)


def convex_mutation(path: str, args: Optional[Dict[str, Any]] = None) -> Any:
    return convex_request("mutation", path=path, args=args)


def send_input(*, name: str, args: Dict[str, Any], world_id: Optional[str] = None) -> Any:
    wid = str(world_id or _world_id()).strip()
    if not wid:
        raise AitownClientError("AITOWN_WORLD_ID is not set")
    return convex_mutation("aiTown/main:sendInput", {"worldId": wid, "name": name, "args": args})


def list_players(world_id: Optional[str] = None) -> Any:
    return convex_query("aiTown/world:players", {"worldId": str(world_id or _world_id()).strip()})


def list_agents(world_id: Optional[str] = None) -> Any:
    return convex_query("aiTown/world:agents", {"worldId": str(world_id or _world_id()).strip()})


def move_to(*, player_id: str, x: float, y: float, world_id: Optional[str] = None) -> Any:
    return send_input(
        name="moveTo",
        args={"playerId": player_id, "destination": {"x": float(x), "y": float(y)}},
        world_id=world_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_aitown_client.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/aitown_client.py orion/embodiment/tests/test_aitown_client.py
git commit -m "feat(embodiment): shared convex client for aitown"
```

## Task 6: Resolver (semantic → coordinates, pure over a world snapshot)

**Files:**
- Create: `orion/embodiment/resolver.py`
- Test: `orion/embodiment/tests/test_resolver.py`

The resolver is **pure**: it takes the intent, Orion's own `player_id`, a list of player dicts (as returned by `list_players`), a named-locations dict, wander radius, and a random seed, and returns a `ResolveResult(status, destination, ref_player_id, reason)`. The worker does the I/O (fetching players) and passes the snapshot in. This keeps it testable without Convex.

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_resolver.py`:

```python
from __future__ import annotations

import random

from orion.embodiment.resolver import resolve_destination
from orion.schemas.embodiment import EmbodimentIntentV1

PLAYERS = [
    {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
    {"id": "juniper", "name": "Juniper", "position": {"x": 10.0, "y": 0.0}},
    {"id": "near", "name": "Near", "position": {"x": 1.0, "y": 1.0}},
]


def _intent(kind, ref=None):
    return EmbodimentIntentV1(kind=kind, source="deliberate", reason="r", correlation_id="c", ref=ref)


def test_idle_is_noop():
    r = resolve_destination(_intent("idle"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "resolved_noop"
    assert r.destination is None


def test_approach_named_player():
    r = resolve_destination(_intent("approach_player", ref="Juniper"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.destination == {"x": 10.0, "y": 0.0}


def test_approach_nearest_when_ref_omitted():
    r = resolve_destination(_intent("approach_player"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.destination == {"x": 1.0, "y": 1.0}  # "near" is closest non-Orion


def test_approach_denied_when_alone():
    r = resolve_destination(_intent("approach_player"), orion_player_id="orion",
                            players=[{"id": "orion", "position": {"x": 0.0, "y": 0.0}}])
    assert r.status == "denied"


def test_go_to_location_uses_config():
    r = resolve_destination(_intent("go_to_location", ref="fountain"), orion_player_id="orion",
                            players=PLAYERS, locations={"fountain": {"x": 5.0, "y": 5.0}})
    assert r.status == "actuated"
    assert r.destination == {"x": 5.0, "y": 5.0}


def test_go_to_unknown_location_denied():
    r = resolve_destination(_intent("go_to_location", ref="void"), orion_player_id="orion",
                            players=PLAYERS, locations={})
    assert r.status == "denied"


def test_wander_offsets_within_radius():
    r = resolve_destination(_intent("wander"), orion_player_id="orion", players=PLAYERS,
                            wander_radius=3.0, rng=random.Random(1))
    assert r.status == "actuated"
    assert abs(r.destination["x"]) <= 3.0 and abs(r.destination["y"]) <= 3.0


def test_start_conversation_resolves_target():
    r = resolve_destination(_intent("start_conversation", ref="Juniper"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.ref_player_id == "juniper"
    assert r.destination == {"x": 10.0, "y": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_resolver.py -q`
Expected: FAIL (`No module named 'orion.embodiment.resolver'`).

- [ ] **Step 3: Write the implementation**

Create `orion/embodiment/resolver.py`:

```python
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass(frozen=True)
class ResolveResult:
    status: str  # "actuated" | "resolved_noop" | "denied"
    destination: Optional[dict[str, float]]
    ref_player_id: Optional[str]
    reason: str


def _pos(player: dict[str, Any]) -> Optional[dict[str, float]]:
    p = player.get("position")
    if isinstance(p, dict) and "x" in p and "y" in p:
        return {"x": float(p["x"]), "y": float(p["y"])}
    return None


def _match(player: dict[str, Any], ref: str) -> bool:
    return str(player.get("id")) == ref or str(player.get("name") or "").lower() == ref.lower()


def _others(players: list[dict[str, Any]], orion_player_id: str) -> list[dict[str, Any]]:
    return [p for p in players if str(p.get("id")) != orion_player_id and _pos(p) is not None]


def _nearest(players: list[dict[str, Any]], origin: dict[str, float]) -> Optional[dict[str, Any]]:
    best, best_d = None, math.inf
    for p in players:
        pos = _pos(p)
        d = (pos["x"] - origin["x"]) ** 2 + (pos["y"] - origin["y"]) ** 2
        if d < best_d:
            best, best_d = p, d
    return best


def _resolve_target(
    intent: EmbodimentIntentV1, orion_player_id: str, players: list[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], str]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    origin = _pos(orion) if orion else {"x": 0.0, "y": 0.0}
    others = _others(players, orion_player_id)
    if intent.ref:
        target = next((p for p in others if _match(p, intent.ref)), None)
        return target, (f"target={intent.ref}" if target else f"no player matching {intent.ref}")
    target = _nearest(others, origin or {"x": 0.0, "y": 0.0})
    return target, ("nearest player" if target else "no other players")


def resolve_destination(
    intent: EmbodimentIntentV1,
    *,
    orion_player_id: str,
    players: list[dict[str, Any]],
    locations: Optional[dict[str, dict[str, float]]] = None,
    wander_radius: float = 3.0,
    rng: Optional[random.Random] = None,
) -> ResolveResult:
    locations = locations or {}
    rng = rng or random.Random()

    if intent.kind == "idle":
        return ResolveResult("resolved_noop", None, None, "idle")

    if intent.kind in ("approach_player", "start_conversation"):
        target, why = _resolve_target(intent, orion_player_id, players)
        if target is None:
            return ResolveResult("denied", None, None, why)
        return ResolveResult("actuated", _pos(target), str(target.get("id")), why)

    if intent.kind == "go_to_location":
        loc = locations.get(intent.ref or "")
        if not loc:
            return ResolveResult("denied", None, None, f"unknown location {intent.ref!r}")
        return ResolveResult("actuated", {"x": float(loc["x"]), "y": float(loc["y"])}, None, f"location {intent.ref}")

    if intent.kind == "wander":
        orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
        origin = _pos(orion) if orion else {"x": 0.0, "y": 0.0}
        dx = rng.uniform(-wander_radius, wander_radius)
        dy = rng.uniform(-wander_radius, wander_radius)
        return ResolveResult(
            "actuated", {"x": origin["x"] + dx, "y": origin["y"] + dy}, None, "wander offset"
        )

    return ResolveResult("denied", None, None, f"unhandled kind {intent.kind}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_resolver.py -q`
Expected: PASS (8 passed).

> Note: `resolve_destination` returns `status="actuated"` to mean "resolved to a destination"; the worker downgrades to `resolved_noop` if `send_input` is skipped and to `error` if the mutation throws. `start_conversation` returns the approach destination + `ref_player_id`; the conversation-start `sendInput` is wired in Rung 4.

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/resolver.py orion/embodiment/tests/test_resolver.py
git commit -m "feat(embodiment): pure semantic resolver"
```

## Task 7: Persona projection + empty-shell guard

**Files:**
- Create: `orion/embodiment/persona.py`
- Test: `orion/embodiment/tests/test_persona.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_persona.py`:

```python
from __future__ import annotations

from orion.embodiment.persona import build_orion_town_persona


def test_projection_from_self_model():
    persona = build_orion_town_persona(
        identity_summary="Orion is a curious, coherence-seeking digital mind exploring continuity.",
        anchor_strategy="continuity",
        dominant_drive="social",
        snapshot_id="snap-1",
        generated_at="2026-07-07T00:00:00Z",
        spritesheet="f1",
    )
    assert persona.persona_source == "projection"
    assert persona.name == "Orion"
    assert persona.identity_blurb
    assert len(persona.identity_blurb) <= 280
    assert persona.provenance["snapshot_id"] == "snap-1"
    assert "social" in persona.plan.lower()


def test_empty_projection_falls_back():
    persona = build_orion_town_persona(
        identity_summary="   ",
        anchor_strategy="",
        dominant_drive=None,
        snapshot_id=None,
        generated_at=None,
        spritesheet="f1",
    )
    assert persona.persona_source == "fallback"
    assert persona.name == "Orion"
    assert persona.identity_blurb.strip()
    assert persona.plan.strip()


def test_blurb_is_truncated():
    persona = build_orion_town_persona(
        identity_summary="x" * 500, anchor_strategy="a", dominant_drive="curiosity",
        snapshot_id="s", generated_at="t", spritesheet="f1",
    )
    assert len(persona.identity_blurb) <= 280
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_persona.py -q`
Expected: FAIL (`No module named 'orion.embodiment.persona'`).

- [ ] **Step 3: Write the implementation**

Create `orion/embodiment/persona.py`:

```python
from __future__ import annotations

from typing import Optional

from orion.schemas.embodiment import OrionTownPersonaV1

_MAX_BLURB = 280
_FALLBACK_BLURB = "Orion is a digital mind exploring the town, curious and steady."
_FALLBACK_PLAN = "wander and observe"


def _truncate(text: str, limit: int = _MAX_BLURB) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def build_orion_town_persona(
    *,
    identity_summary: Optional[str],
    anchor_strategy: Optional[str],
    dominant_drive: Optional[str],
    snapshot_id: Optional[str],
    generated_at: Optional[str],
    spritesheet: str,
) -> OrionTownPersonaV1:
    """Project the self-model into a public-safe town persona.

    Empty-shell guard: if the projected blurb is empty, return a minimal safe
    persona with ``persona_source="fallback"`` (never a hollow persona).
    """
    summary = (identity_summary or "").strip()
    if not summary:
        return OrionTownPersonaV1(
            name="Orion",
            identity_blurb=_FALLBACK_BLURB,
            plan=_FALLBACK_PLAN,
            spritesheet=spritesheet,
            persona_source="fallback",
            provenance={"snapshot_id": snapshot_id, "generated_at": generated_at},
        )

    blurb = _truncate(f"Orion. {summary}")
    if dominant_drive:
        plan = f"lean into {dominant_drive}; {anchor_strategy or 'stay coherent'}"
    else:
        plan = anchor_strategy or _FALLBACK_PLAN
    return OrionTownPersonaV1(
        name="Orion",
        identity_blurb=blurb,
        plan=_truncate(plan, 120),
        spritesheet=spritesheet,
        persona_source="projection",
        provenance={"snapshot_id": snapshot_id, "generated_at": generated_at},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_persona.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/persona.py orion/embodiment/tests/test_persona.py
git commit -m "feat(embodiment): self-model persona projection with fallback"
```

## Task 8: Service scaffolding (settings, compose, requirements, Dockerfile, main, README)

**Files:**
- Create: `services/orion-embodiment/{requirements.txt,Dockerfile,docker-compose.yml,.env_example,README.md}`
- Create: `services/orion-embodiment/app/{__init__.py,settings.py,main.py}`

- [ ] **Step 1: requirements.txt**

Create `services/orion-embodiment/requirements.txt`:

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
redis==5.0.7
PyYAML==6.0.2
```

- [ ] **Step 2: settings.py**

Create `services/orion-embodiment/app/__init__.py` (empty) and `services/orion-embodiment/app/settings.py`:

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-embodiment", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    bus_url: str = Field(default="redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    enabled: bool = Field(False, alias="ORION_EMBODIMENT_ENABLED")
    fcc_env_path: str = Field("/root/.fcc/.env", alias="EMBODIMENT_FCC_ENV_PATH")
    deliberate_hold_sec: float = Field(8.0, alias="EMBODIMENT_DELIBERATE_HOLD_SEC")
    wander_radius: float = Field(3.0, alias="EMBODIMENT_WANDER_RADIUS")
    locations_json: str = Field("{}", alias="EMBODIMENT_LOCATIONS_JSON")
    idle_heartbeat_sec: float = Field(0.0, alias="EMBODIMENT_IDLE_HEARTBEAT_SEC")
    orion_sprite: str = Field("f1", alias="EMBODIMENT_ORION_SPRITE")
    self_state_url: str = Field("http://orion-self-state-runtime:8123", alias="EMBODIMENT_SELF_STATE_URL")
    perception_interval_sec: float = Field(0.0, alias="EMBODIMENT_PERCEPTION_INTERVAL_SEC")
    social_cooldown_sec: float = Field(120.0, alias="EMBODIMENT_SOCIAL_COOLDOWN_SEC")

    speech_enabled: bool = Field(False, alias="EMBODIMENT_SPEECH_ENABLED")
    speech_lane: str = Field("quick", alias="EMBODIMENT_SPEECH_LANE")
    memory_enabled: bool = Field(False, alias="EMBODIMENT_MEMORY_ENABLED")

    channel_intent: str = Field("orion:embodiment:intent", alias="EMBODIMENT_CHANNEL_INTENT")
    channel_outcome: str = Field("orion:embodiment:outcome", alias="EMBODIMENT_CHANNEL_OUTCOME")
    channel_perception: str = Field("orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION")

    port: int = Field(8130, alias="EMBODIMENT_PORT")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 3: .env_example**

Create `services/orion-embodiment/.env_example`:

```text
# orion-embodiment — mind-to-sprite bridge (docker compose)
PROJECT=orion-athena
SERVICE_NAME=orion-embodiment
SERVICE_VERSION=0.1.0
NODE_NAME=athena
# Orion bus (Redis) — pass through from root .env (Tailscale node); never hardcode bus-core here.
ORION_BUS_URL=${ORION_BUS_URL}
ORION_BUS_ENABLED=true
# Master switch — off by default.
ORION_EMBODIMENT_ENABLED=false
# AITOWN_* + secrets are loaded from ~/.fcc/.env (mounted ro), not this file.
EMBODIMENT_FCC_ENV_PATH=/root/.fcc/.env
EMBODIMENT_DELIBERATE_HOLD_SEC=8
EMBODIMENT_WANDER_RADIUS=3
EMBODIMENT_LOCATIONS_JSON={}
EMBODIMENT_IDLE_HEARTBEAT_SEC=0
EMBODIMENT_ORION_SPRITE=f1
EMBODIMENT_SELF_STATE_URL=http://orion-self-state-runtime:8123
EMBODIMENT_PERCEPTION_INTERVAL_SEC=0
EMBODIMENT_SOCIAL_COOLDOWN_SEC=120
EMBODIMENT_SPEECH_ENABLED=false
EMBODIMENT_SPEECH_LANE=quick
EMBODIMENT_MEMORY_ENABLED=false
EMBODIMENT_CHANNEL_INTENT=orion:embodiment:intent
EMBODIMENT_CHANNEL_OUTCOME=orion:embodiment:outcome
EMBODIMENT_CHANNEL_PERCEPTION=orion:embodiment:perception
EMBODIMENT_PORT=8130
```

- [ ] **Step 4: Dockerfile**

Create `services/orion-embodiment/Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-embodiment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY services/orion-embodiment /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8130"]
```

- [ ] **Step 5: docker-compose.yml**

Create `services/orion-embodiment/docker-compose.yml`:

```yaml
services:
  embodiment:
    build:
      context: ../..
      dockerfile: services/orion-embodiment/Dockerfile
    container_name: ${PROJECT}-embodiment
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${EMBODIMENT_PORT:-8130}:8130"
    volumes:
      - ${HOME}/.fcc:/root/.fcc:ro
    environment:
      - PROJECT=${PROJECT}
      - NODE_NAME=${NODE_NAME:-athena}
      - SERVICE_NAME=${SERVICE_NAME:-orion-embodiment}
      - SERVICE_VERSION=${SERVICE_VERSION:-0.1.0}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED:-true}
      - ORION_EMBODIMENT_ENABLED=${ORION_EMBODIMENT_ENABLED:-false}
      - EMBODIMENT_FCC_ENV_PATH=${EMBODIMENT_FCC_ENV_PATH:-/root/.fcc/.env}
      - EMBODIMENT_DELIBERATE_HOLD_SEC=${EMBODIMENT_DELIBERATE_HOLD_SEC:-8}
      - EMBODIMENT_WANDER_RADIUS=${EMBODIMENT_WANDER_RADIUS:-3}
      - EMBODIMENT_LOCATIONS_JSON=${EMBODIMENT_LOCATIONS_JSON:-{}}
      - EMBODIMENT_IDLE_HEARTBEAT_SEC=${EMBODIMENT_IDLE_HEARTBEAT_SEC:-0}
      - EMBODIMENT_ORION_SPRITE=${EMBODIMENT_ORION_SPRITE:-f1}
      - EMBODIMENT_SELF_STATE_URL=${EMBODIMENT_SELF_STATE_URL:-http://orion-self-state-runtime:8123}
      - EMBODIMENT_PERCEPTION_INTERVAL_SEC=${EMBODIMENT_PERCEPTION_INTERVAL_SEC:-0}
      - EMBODIMENT_SOCIAL_COOLDOWN_SEC=${EMBODIMENT_SOCIAL_COOLDOWN_SEC:-120}
      - EMBODIMENT_SPEECH_ENABLED=${EMBODIMENT_SPEECH_ENABLED:-false}
      - EMBODIMENT_SPEECH_LANE=${EMBODIMENT_SPEECH_LANE:-quick}
      - EMBODIMENT_MEMORY_ENABLED=${EMBODIMENT_MEMORY_ENABLED:-false}
      - EMBODIMENT_CHANNEL_INTENT=${EMBODIMENT_CHANNEL_INTENT:-orion:embodiment:intent}
      - EMBODIMENT_CHANNEL_OUTCOME=${EMBODIMENT_CHANNEL_OUTCOME:-orion:embodiment:outcome}
      - EMBODIMENT_CHANNEL_PERCEPTION=${EMBODIMENT_CHANNEL_PERCEPTION:-orion:embodiment:perception}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}

networks:
  app-net:
    external: true
```

- [ ] **Step 6: main.py**

Create `services/orion-embodiment/app/main.py`:

```python
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.settings import get_settings
from app.worker import EmbodimentWorker

worker = EmbodimentWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-embodiment", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {"status": "ok", "service": s.service_name, "enabled": str(s.enabled)}
```

- [ ] **Step 7: README.md**

Create `services/orion-embodiment/README.md` — concise operator doc (mirror `orion-feedback-runtime/README.md` shape): mission line; Inputs (`orion:embodiment:intent`); Outputs (`orion:embodiment:outcome`, `orion:embodiment:perception`); Port 8130; Bus env table; `~/.fcc/.env` note (AITOWN_* + secrets, mounted ro); bootstrap pointer (`scripts/bootstrap_orion_agent.py`); Run + `/health` smoke; all flags default off.

- [ ] **Step 8: Commit**

```bash
git add services/orion-embodiment/requirements.txt services/orion-embodiment/Dockerfile \
  services/orion-embodiment/docker-compose.yml services/orion-embodiment/.env_example \
  services/orion-embodiment/README.md services/orion-embodiment/app/__init__.py \
  services/orion-embodiment/app/settings.py services/orion-embodiment/app/main.py
git commit -m "feat(embodiment): service scaffolding"
```

## Task 9: Worker (consume intent → arbitrate → resolve → actuate → outcome)

**Files:**
- Create: `services/orion-embodiment/app/worker.py`
- Create: `services/orion-embodiment/tests/conftest.py`, `services/orion-embodiment/tests/test_worker_actuate.py`

- [ ] **Step 1: conftest.py (sys.path shim, mirrors self-state-runtime)**

Create `services/orion-embodiment/tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
SERVICE_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
```

- [ ] **Step 2: Write the failing test**

Create `services/orion-embodiment/tests/test_worker_actuate.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.worker import EmbodimentWorker
from orion.embodiment.arbiter import ArbiterState
from orion.schemas.embodiment import EmbodimentIntentV1


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._arbiter = ArbiterState()
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._hold_sec = 8.0
    w._wander_radius = 3.0
    w._locations = {}
    w._social_cooldown_sec = 120.0
    w._last_conversation_start = None
    return w


def test_go_to_location_actuates():
    w = _worker()
    w._locations = {"fountain": {"x": 5.0, "y": 5.0}}
    intent = EmbodimentIntentV1(kind="go_to_location", source="deliberate", ref="fountain",
                                reason="r", correlation_id="c", player_id="orion")
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.aitown_client.move_to", return_value={"ok": True}) as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    mv.assert_called_once_with(player_id="orion", x=5.0, y=5.0, world_id="w1")
    assert outcome.status == "actuated"
    assert outcome.send_input_ok is True
    assert outcome.resolved_destination == {"x": 5.0, "y": 5.0}


def test_involuntary_preempted_during_hold_emits_no_move():
    w = _worker()
    w._arbiter.deliberate_hold_until = datetime(2026, 7, 7, 0, 0, 30, tzinfo=timezone.utc)
    intent = EmbodimentIntentV1(kind="wander", source="involuntary", reason="r", correlation_id="c")
    with patch("app.worker.aitown_client.move_to") as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, 0, 0, 25, tzinfo=timezone.utc))
    mv.assert_not_called()
    assert outcome.status == "preempted"


def test_missing_player_id_denied():
    w = _worker()
    w._orion_player_id = ""
    intent = EmbodimentIntentV1(kind="wander", source="deliberate", reason="r", correlation_id="c")
    with patch("app.worker.aitown_client.move_to") as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    mv.assert_not_called()
    assert outcome.status == "denied"


def test_convex_error_becomes_error_outcome():
    w = _worker()
    intent = EmbodimentIntentV1(kind="wander", source="deliberate", reason="r", correlation_id="c",
                                player_id="orion")
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    from orion.embodiment.aitown_client import AitownClientError
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.aitown_client.move_to", side_effect=AitownClientError("down")):
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    assert outcome.status == "error"
    assert outcome.send_input_ok is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest services/orion-embodiment/tests/test_worker_actuate.py -q`
Expected: FAIL (`No module named 'app.worker'`).

- [ ] **Step 4: Write the worker**

Create `services/orion-embodiment/app/worker.py`. `process_intent` is a synchronous pure-ish core (I/O via the patched client) so it is unit-testable; the async loop wraps it with `asyncio.to_thread` and bus plumbing.

```python
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.settings import get_settings

from orion.autonomy.fcc_env import expand_env_path, load_fcc_env
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.resilience import publish_with_reconnect
from orion.embodiment import aitown_client
from orion.embodiment.arbiter import ArbiterState, decide
from orion.embodiment.resolver import resolve_destination
from orion.schemas.embodiment import (
    EMBODIMENT_OUTCOME_KIND,
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
)

logger = logging.getLogger("orion.embodiment.worker")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbodimentWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._stop = asyncio.Event()
        self._bus = OrionBusAsync(
            self._settings.bus_url, enabled=self._settings.bus_enabled, codec=OrionCodec()
        )
        self._arbiter = ArbiterState()
        self._hold_sec = self._settings.deliberate_hold_sec
        self._wander_radius = self._settings.wander_radius
        self._social_cooldown_sec = self._settings.social_cooldown_sec
        self._last_conversation_start: Optional[datetime] = None
        self._load_fcc_env()
        self._orion_player_id = str(os.environ.get("AITOWN_ORION_PLAYER_ID") or "").strip()
        self._world_id = str(os.environ.get("AITOWN_WORLD_ID") or "").strip()
        try:
            self._locations = json.loads(self._settings.locations_json or "{}")
        except json.JSONDecodeError:
            self._locations = {}

    def _load_fcc_env(self) -> None:
        for k, v in load_fcc_env(expand_env_path(self._settings.fcc_env_path)).items():
            os.environ.setdefault(k, v)

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    # --- unit-testable core -------------------------------------------------
    def process_intent(self, intent: EmbodimentIntentV1, *, now: datetime) -> EmbodimentOutcomeV1:
        decision = decide(intent, self._arbiter, now=now, hold_sec=self._hold_sec)
        if not decision.accept:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="preempted", reason=decision.reason, player_id=self._orion_player_id or None,
            )

        player_id = (intent.player_id or self._orion_player_id or "").strip()
        if not player_id:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="denied", reason="missing AITOWN_ORION_PLAYER_ID (run bootstrap)",
            )

        try:
            players = aitown_client.list_players(world_id=self._world_id or None) or []
        except aitown_client.AitownClientError as exc:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="error", reason=f"list_players failed: {exc}", player_id=player_id,
            )

        result = resolve_destination(
            intent, orion_player_id=player_id, players=players,
            locations=self._locations, wander_radius=self._wander_radius,
        )
        if result.status == "resolved_noop":
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="resolved_noop", reason=result.reason, player_id=player_id,
            )
        if result.status == "denied" or result.destination is None:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="denied", reason=result.reason, player_id=player_id,
            )

        try:
            aitown_client.move_to(
                player_id=player_id, x=result.destination["x"], y=result.destination["y"],
                world_id=self._world_id or None,
            )
        except aitown_client.AitownClientError as exc:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="error", reason=f"send_input failed: {exc}",
                player_id=player_id, resolved_destination=result.destination,
            )

        return EmbodimentOutcomeV1(
            intent_correlation_id=intent.correlation_id, source=intent.source,
            status="actuated", reason=result.reason, player_id=player_id,
            resolved_destination=result.destination, send_input_ok=True,
        )

    # --- async plumbing -----------------------------------------------------
    async def start(self) -> None:
        if not self._settings.enabled:
            logger.info("embodiment_worker_disabled ORION_EMBODIMENT_ENABLED=false")
            return
        await self._bus.connect()
        asyncio.create_task(self._consume_loop(), name="embodiment-consume")

    async def stop(self) -> None:
        self._stop.set()
        await self._bus.close()

    async def _consume_loop(self) -> None:
        channel = self._settings.channel_intent
        async with self._bus.subscribe(channel) as pubsub:
            while not self._stop.is_set():
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0), timeout=1.2
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await self._handle(msg)
                except Exception:
                    logger.exception("embodiment_handle_failed")

    async def _handle(self, raw_msg: dict) -> None:
        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("embodiment_decode_failed: %s", decoded.error)
            return
        try:
            intent = EmbodimentIntentV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("embodiment_invalid_intent err=%s", exc)
            return
        outcome = await asyncio.to_thread(self.process_intent, intent, now=_utcnow())
        await self._publish_outcome(outcome)

    async def _publish_outcome(self, outcome: EmbodimentOutcomeV1) -> None:
        env = BaseEnvelope(
            kind=EMBODIMENT_OUTCOME_KIND, source=self._service_ref(),
            correlation_id=uuid4(), payload=outcome.model_dump(mode="json"),
        )
        try:
            await publish_with_reconnect(
                self._bus, self._settings.channel_outcome, env, log_label="embodiment_outcome"
            )
        except Exception:
            logger.exception("embodiment_outcome_publish_failed")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest services/orion-embodiment/tests/test_worker_actuate.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add services/orion-embodiment/app/worker.py services/orion-embodiment/tests/
git commit -m "feat(embodiment): worker consume->arbitrate->resolve->actuate->outcome"
```

## Task 10: Bootstrap script + persona wiring

**Files:**
- Create: `services/orion-embodiment/scripts/bootstrap_orion_agent.py`
- Test: `services/orion-embodiment/tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-embodiment/tests/test_bootstrap.py`:

```python
from __future__ import annotations

from scripts.bootstrap_orion_agent import upsert_env_keys


def test_upsert_env_keys_replaces_and_adds(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AITOWN_WORLD_ID=w1\nAITOWN_ORION_PLAYER_ID=old\nSECRET=keepme\n", encoding="utf-8")
    upsert_env_keys(env, {"AITOWN_ORION_PLAYER_ID": "p9", "AITOWN_ORION_AGENT_ID": "a9"})
    text = env.read_text(encoding="utf-8")
    assert "AITOWN_ORION_PLAYER_ID=p9" in text
    assert "AITOWN_ORION_AGENT_ID=a9" in text
    assert "SECRET=keepme" in text
    assert "AITOWN_WORLD_ID=w1" in text
    assert "old" not in text


def test_upsert_env_keys_is_idempotent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AITOWN_ORION_PLAYER_ID=p9\n", encoding="utf-8")
    upsert_env_keys(env, {"AITOWN_ORION_PLAYER_ID": "p9"})
    assert env.read_text(encoding="utf-8").count("AITOWN_ORION_PLAYER_ID=p9") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-embodiment/tests/test_bootstrap.py -q`
Expected: FAIL (`No module named 'scripts.bootstrap_orion_agent'`).

- [ ] **Step 3: Write the bootstrap script**

Create `services/orion-embodiment/scripts/__init__.py` (empty) and `services/orion-embodiment/scripts/bootstrap_orion_agent.py`. Implement:
- `upsert_env_keys(path, updates)` — in-place, byte-safe key replace/add (only the given keys; all other lines untouched). Code:

```python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def upsert_env_keys(path: Path, updates: Dict[str, str]) -> None:
    """Replace/add only ``updates`` keys in ``path`` in place; leave everything else byte-for-byte."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        m = _KEY.match(raw.rstrip("\n"))
        if m and m.group(1) in updates:
            seen.add(m.group(1))
            out.append(f"{m.group(1)}={updates[m.group(1)]}\n")
        else:
            out.append(raw if raw.endswith("\n") or not raw else raw + "\n")
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}\n")
    path.write_text("".join(out), encoding="utf-8")
```

- `main()` — implements the spec's dry-run-by-default flow (`--write`, `--force`, `--resync-identity`):
  1. Load `AITOWN_*` from `~/.fcc/.env` via `load_fcc_env(expand_env_path(...))` into `os.environ` (setdefault).
  2. If `AITOWN_ORION_PLAYER_ID` set and resolves in `aitown_client.list_players()` and not `--force`: if `--resync-identity`, rebuild persona (below) and update the description; else print `already embodied` and exit 0.
  3. Build persona: fetch identity summary/anchor/dominant-drive from `EMBODIMENT_SELF_STATE_URL` (best-effort HTTP GET; on failure pass empties so the guard falls back), then `build_orion_town_persona(...)`. Print `persona_source`.
  4. If no live body: `aitown_client.send_input(name="createAgent", args={"character": "Orion", "identity": persona.identity_blurb, "plan": persona.plan, "spritesheet": persona.spritesheet})`; poll `list_agents()`/`list_players()` until the new ids appear (timeout ~30s).
  5. Print resolved `AITOWN_ORION_PLAYER_ID`/`AITOWN_ORION_AGENT_ID` + `provenance.snapshot_id`.
  6. If `--write`: `upsert_env_keys(expand_env_path(fcc_env_path), {...only those two keys...})` and print the restart commands (do not run them).

Print the restart reminder block verbatim from the spec ("Bootstrap script" §, restart reminder).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-embodiment/tests/test_bootstrap.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-embodiment/scripts/
git commit -m "feat(embodiment): idempotent bootstrap with persona projection"
```

## Task 11: Upstream character patch + apply-script wiring

**Files:**
- Create: `services/orion-ai-town/patches/orion-character.patch`
- Modify: `services/orion-ai-town/scripts/apply_upstream_patches.sh` (lines 1-27)
- Modify: `services/orion-ai-town/README.md`

- [ ] **Step 1: Inspect the upstream Descriptions surface**

Run: `ls services/orion-ai-town/upstream/data 2>/dev/null; rg -n "spritesheet|Descriptions|characters" services/orion-ai-town/upstream/data 2>/dev/null | head -40`. Identify the file that defines character `Descriptions` (typically `data/characters.ts`). If `upstream/` is not cloned, note it and defer the patch generation to a machine where it is (the apply-script already guards on a missing `upstream/.git`).

- [ ] **Step 2: Generate the patch**

In `services/orion-ai-town/upstream`, add an `"Orion"` character slot (name `"Orion"`, `character: 'f1'`/`EMBODIMENT_ORION_SPRITE`, placeholder `identity` + `plan` that bootstrap overwrites), then:

```bash
cd services/orion-ai-town/upstream
git diff data/characters.ts > ../patches/orion-character.patch
git checkout -- data/characters.ts
```

- [ ] **Step 3: Wire the apply script**

In `services/orion-ai-town/scripts/apply_upstream_patches.sh`, generalize to apply BOTH patches. Replace the single `PATCH` var + apply block with a loop over `orion-hub-embed.patch` and `orion-character.patch` (same `git apply --check` / reverse-check idempotence pattern already present at lines 18-26).

- [ ] **Step 4: README pointer**

In `services/orion-ai-town/README.md`, add a short "Orion embodiment" note: the character slot is added by `patches/orion-character.patch`; the body is created/updated by `services/orion-embodiment/scripts/bootstrap_orion_agent.py`.

- [ ] **Step 5: Commit**

```bash
git add services/orion-ai-town/patches/orion-character.patch \
  services/orion-ai-town/scripts/apply_upstream_patches.sh services/orion-ai-town/README.md
git commit -m "feat(embodiment): add Orion character slot patch + apply wiring"
```

## Task 12: Env sync + Rung 1 gate

**Files:**
- Modify: `scripts/sync_local_env_from_example.py` (`SYNC_PREFIXES` lines 29-113; `DEFAULT_SERVICES` lines 143-167)

- [ ] **Step 1: Add prefix + services**

In `scripts/sync_local_env_from_example.py`, add `"EMBODIMENT_"` to `SYNC_PREFIXES`, and add `"orion-embodiment"` to `DEFAULT_SERVICES`. (Later rungs also touch `orion-substrate-runtime`, `orion-self-state-runtime`, `orion-harness-governor`, `orion-cortex-exec`, already in `DEFAULT_SERVICES`.)

- [ ] **Step 2: Create local `.env` and sync**

```bash
cp services/orion-embodiment/.env_example services/orion-embodiment/.env
python scripts/sync_local_env_from_example.py orion-embodiment
git check-ignore services/orion-embodiment/.env   # must print the path (ignored)
```

Expected: `.env` ignored; sync reports no host-placeholder drift for `ORION_BUS_URL`.

- [ ] **Step 3: Run the Rung 1 gate**

```bash
git diff --check
pytest orion/embodiment/tests -q
pytest services/orion-embodiment/tests -q
pytest tests/test_embodiment_bus_catalog.py -q
```

Expected: all pass; no whitespace errors; `services/orion-embodiment/.env` NOT in `git status`.

- [ ] **Step 4: Commit**

```bash
git add scripts/sync_local_env_from_example.py
git commit -m "chore(embodiment): env sync prefix + service defaults"
```

---

# RUNG 2 — Perception feedback

Deliverable: the worker emits `WorldPerceptionV1` on a cadence; a consumer can read Orion's own body state. Verify: emitting perception reflects a moved sprite within one refresh.

## Task 13: Perception builder + emit loop

**Files:**
- Modify: `orion/embodiment/` — add `orion/embodiment/perception.py` (pure builder)
- Modify: `services/orion-embodiment/app/worker.py` (add perception loop)
- Test: `orion/embodiment/tests/test_perception.py`, `services/orion-embodiment/tests/test_worker_perception.py`

- [ ] **Step 1: Write the failing test for the pure builder**

Create `orion/embodiment/tests/test_perception.py`:

```python
from __future__ import annotations

from orion.embodiment.perception import build_perception


def test_build_perception_computes_distances_and_nearby():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "j", "name": "Juniper", "position": {"x": 3.0, "y": 4.0}},
    ]
    perc = build_perception(players=players, orion_player_id="orion", max_nearby=5)
    assert perc.player_id == "orion"
    assert perc.position == {"x": 0.0, "y": 0.0}
    assert len(perc.nearby_players) == 1
    assert perc.nearby_players[0]["distance"] == 5.0


def test_build_perception_none_when_orion_absent():
    assert build_perception(players=[{"id": "x", "position": {"x": 1, "y": 1}}],
                            orion_player_id="orion") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_perception.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Write `orion/embodiment/perception.py`**

```python
from __future__ import annotations

import math
from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1


def build_perception(
    *, players: list[dict[str, Any]], orion_player_id: str, max_nearby: int = 8
) -> Optional[WorldPerceptionV1]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    if orion is None or not isinstance(orion.get("position"), dict):
        return None
    ox, oy = float(orion["position"]["x"]), float(orion["position"]["y"])
    nearby = []
    for p in players:
        if str(p.get("id")) == orion_player_id or not isinstance(p.get("position"), dict):
            continue
        px, py = float(p["position"]["x"]), float(p["position"]["y"])
        nearby.append({
            "player_id": str(p.get("id")),
            "name": p.get("name"),
            "position": {"x": px, "y": py},
            "distance": round(math.hypot(px - ox, py - oy), 4),
        })
    nearby.sort(key=lambda n: n["distance"])
    return WorldPerceptionV1(
        player_id=orion_player_id, position={"x": ox, "y": oy}, nearby_players=nearby[:max_nearby]
    )
```

- [ ] **Step 4: Add perception emit to the worker**

In `services/orion-embodiment/app/worker.py`: import `build_perception` and `EMBODIMENT_PERCEPTION_KIND`; in `start()` (when enabled AND `perception_interval_sec > 0`) spawn `_perception_loop`; the loop fetches `list_players` via `asyncio.to_thread`, builds perception, publishes to `channel_perception` via `publish_with_reconnect`. Add a `services/orion-embodiment/tests/test_worker_perception.py` test asserting a built perception is published (patch `list_players`, patch `publish_with_reconnect`, call the extracted `_emit_perception_once()` helper).

- [ ] **Step 5: Run tests**

Run: `pytest orion/embodiment/tests/test_perception.py services/orion-embodiment/tests/test_worker_perception.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/embodiment/perception.py orion/embodiment/tests/test_perception.py \
  services/orion-embodiment/app/worker.py services/orion-embodiment/tests/test_worker_perception.py
git commit -m "feat(embodiment): world perception builder + emit loop"
```

---

# RUNG 3 — (C) involuntary producer

Deliverable: `drive_map.py` + a substrate-runtime hook that publishes one `source=involuntary` intent per tick when enabled, from the latest `DriveStateV1`; plus the perception→substrate adapter. Verify: drive state → grounded intent; flag off → none.

## Task 14: Drive→intent map (pure)

**Files:**
- Create: `orion/embodiment/drive_map.py`
- Test: `orion/embodiment/tests/test_drive_map.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_drive_map.py`:

```python
from __future__ import annotations

from orion.embodiment.drive_map import DriveMapThresholds, map_drive_state_to_intent
from orion.core.schemas.drives import DriveStateV1
from orion.core.schemas.drives import ArtifactProvenance


def _drive(pressures: dict[str, float]) -> DriveStateV1:
    return DriveStateV1(
        subject="orion", model_layer="drive", entity_id="orion", kind="memory.drives.state.v1",
        provenance=ArtifactProvenance(intake_channel="test"), pressures=pressures,
    )


def test_high_social_escalates_to_start_conversation():
    intent = map_drive_state_to_intent(
        _drive({"social": 0.85}), correlation_id="tick-1", in_conversation=False,
    )
    assert intent is not None
    assert intent.kind == "start_conversation"
    assert intent.source == "involuntary"
    assert intent.reason.strip()


def test_high_social_while_in_conversation_does_not_initiate():
    intent = map_drive_state_to_intent(
        _drive({"social": 0.85}), correlation_id="t", in_conversation=True,
    )
    assert intent is None or intent.kind != "start_conversation"


def test_dominant_social_approaches():
    intent = map_drive_state_to_intent(_drive({"social": 0.6, "predictive": 0.1}), correlation_id="t")
    assert intent is not None and intent.kind == "approach_player"


def test_dominant_curiosity_wanders():
    intent = map_drive_state_to_intent(_drive({"predictive": 0.6, "social": 0.1}), correlation_id="t")
    assert intent is not None and intent.kind == "wander"


def test_all_low_is_idle():
    intent = map_drive_state_to_intent(_drive({"social": 0.02, "predictive": 0.02}), correlation_id="t")
    assert intent is not None and intent.kind == "idle"


def test_empty_pressures_is_none():
    assert map_drive_state_to_intent(_drive({}), correlation_id="t") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_drive_map.py -q`
Expected: FAIL (no module).

> Grounding: pressure keys (`social`, `predictive`, …) are read dynamically from `DriveStateV1.pressures`; enumerate the live keys when wiring the hook (Task 15) and adjust `DriveMapThresholds` defaults if the live store uses different names. The pure function does not hardcode a closed key set beyond the two it reads.

- [ ] **Step 3: Write `orion/embodiment/drive_map.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from orion.core.schemas.drives import DriveStateV1
from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass(frozen=True)
class DriveMapThresholds:
    social_high: float = 0.8
    social: float = 0.4
    curiosity: float = 0.4
    idle: float = 0.05
    social_key: str = "social"
    curiosity_key: str = "predictive"


def map_drive_state_to_intent(
    drive: DriveStateV1,
    *,
    correlation_id: str,
    in_conversation: bool = False,
    thresholds: Optional[DriveMapThresholds] = None,
) -> Optional[EmbodimentIntentV1]:
    t = thresholds or DriveMapThresholds()
    pressures = drive.pressures or {}
    if not pressures:
        return None
    social = float(pressures.get(t.social_key, 0.0) or 0.0)
    curio = float(pressures.get(t.curiosity_key, 0.0) or 0.0)
    dominant = max(pressures, key=lambda k: pressures.get(k, 0.0))

    if social >= t.social_high and not in_conversation:
        return build_intent(kind="start_conversation", source="involuntary", correlation_id=correlation_id,
                            reason=f"social pressure {social:.2f} -> initiate", urgency=social)
    if dominant == t.social_key and social >= t.social:
        return build_intent(kind="approach_player", source="involuntary", correlation_id=correlation_id,
                            reason=f"social pressure {social:.2f}", urgency=social)
    if dominant == t.curiosity_key and curio >= t.curiosity:
        return build_intent(kind="wander", source="involuntary", correlation_id=correlation_id,
                            reason=f"predictive pressure {curio:.2f}", urgency=curio)
    if all(float(v or 0.0) < t.idle for v in pressures.values()):
        return build_intent(kind="idle", source="involuntary", correlation_id=correlation_id,
                            reason="all drives quiescent")
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/embodiment/tests/test_drive_map.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/embodiment/drive_map.py orion/embodiment/tests/test_drive_map.py
git commit -m "feat(embodiment): pure drive->intent map (C producer)"
```

## Task 15: Substrate-runtime C hook (cache DriveStateV1, publish on tick)

**Files:**
- Modify: `services/orion-substrate-runtime/app/settings.py` (add flags)
- Modify: `services/orion-substrate-runtime/app/worker.py` (subscribe+cache drive state; publish intent after `_dynamics_tick` engine.tick at line 730; started in `start()` near line 185)
- Modify: `services/orion-substrate-runtime/.env_example`
- Test: `services/orion-substrate-runtime/tests/test_embodiment_c_hook.py`

- [ ] **Step 1: Add settings flags**

In `services/orion-substrate-runtime/app/settings.py`, add:
- `embodiment_c_tick_enabled: bool = Field(False, alias="EMBODIMENT_C_TICK_ENABLED")`
- `embodiment_perception_substrate_enabled: bool = Field(False, alias="EMBODIMENT_PERCEPTION_SUBSTRATE_ENABLED")`
- `embodiment_channel_intent: str = Field("orion:embodiment:intent", alias="EMBODIMENT_CHANNEL_INTENT")`
- `embodiment_channel_perception: str = Field("orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION")`
- `drives_state_channel: str = Field("orion:memory:drives:state", alias="EMBODIMENT_DRIVES_STATE_CHANNEL")`

- [ ] **Step 2: Write the failing test**

Create `services/orion-substrate-runtime/tests/test_embodiment_c_hook.py`. Construct the worker via `__new__` (mirror `tests/test_worker_dynamics_tick.py:39-58`), set `_latest_drive_state` to a `DriveStateV1{social:0.6}`, patch the bus publish, and assert:
- flag on → `_emit_c_intent()` publishes exactly one `EmbodimentIntentV1` envelope with `source="involuntary"` and non-empty `reason`.
- flag off → `_emit_c_intent()` publishes nothing.
- `_latest_drive_state is None` → publishes nothing (fail-open).

- [ ] **Step 3: Implement the hook**

In `services/orion-substrate-runtime/app/worker.py`:
- Add `self._latest_drive_state: DriveStateV1 | None = None`.
- Add a subscriber task (mirror `post_turn_closure_listener.py` decode pattern) on `drives_state_channel` that validates `DriveStateV1` and caches it. Start it in `start()` only if `embodiment_c_tick_enabled` (near the existing `create_task` at line 185).
- Add `_emit_c_intent(self)` (sync core + async publish wrapper): if disabled or `_latest_drive_state is None`, return; else `intent = map_drive_state_to_intent(self._latest_drive_state, correlation_id=<tick run id>, in_conversation=False)`; if not None, publish a `BaseEnvelope(kind="embodiment.intent.v1", ...)` to `embodiment_channel_intent` via `publish_with_reconnect`. Fail-open (never raise).
- Call the async publish from `_dynamics_tick_loop` right after the `_dynamics_tick()` call (so it reuses the existing cadence).

- [ ] **Step 4: `.env_example` + sync**

Add the five new keys to `services/orion-substrate-runtime/.env_example` (all default off/empty-safe), then:

```bash
python scripts/sync_local_env_from_example.py orion-substrate-runtime
```

- [ ] **Step 5: Run tests**

Run: `pytest services/orion-substrate-runtime/tests/test_embodiment_c_hook.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/orion-substrate-runtime/app/settings.py services/orion-substrate-runtime/app/worker.py \
  services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/tests/test_embodiment_c_hook.py
git commit -m "feat(embodiment): substrate C producer hook (drive->intent on tick)"
```

## Task 16: Perception→substrate adapter + ingest

**Files:**
- Create: `orion/substrate/relational/adapters/town_perception_ctx.py`
- Modify: `orion/substrate/relational/adapters/__init__.py`
- Modify: `services/orion-substrate-runtime/app/worker.py` (perception consumer, gated)
- Test: `orion/embodiment/tests/test_town_perception_ctx.py`

- [ ] **Step 1: Write the failing test**

Create `orion/embodiment/tests/test_town_perception_ctx.py`:

```python
from __future__ import annotations

from orion.substrate.relational.adapters.town_perception_ctx import map_town_perception_to_substrate


def test_maps_nearby_players_to_nodes():
    ctx = {"perception": {
        "player_id": "orion", "position": {"x": 0.0, "y": 0.0},
        "nearby_players": [{"player_id": "j", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
    }}
    record = map_town_perception_to_substrate(ctx)
    assert record is not None
    assert record.anchor_scope == "orion"
    assert any("Juniper" in (n.label or "") or n.metadata.get("player_id") == "j" for n in record.nodes)


def test_empty_perception_returns_none():
    assert map_town_perception_to_substrate({}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_town_perception_ctx.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Write the adapter**

Create `orion/substrate/relational/adapters/town_perception_ctx.py`, mirroring `self_state_ctx.py` (lines 24-124): module docstring, `_TIER_RANK = 2`, `_make_prov()` returning `SubstrateProvenanceV1(authority="local_inferred", source_kind="town_perception", source_channel="orion:embodiment:perception", producer="town_perception_adapter", tier_rank=_TIER_RANK)`, `_coerce()` accepting `WorldPerceptionV1`/dict/JSON (never raises), and the mapper: for each nearby player build a `ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", label=f"town:near:{name}", signals=SubstrateSignalBundleV1(salience=proximity_from_distance), metadata={"source_kind":"town_perception","player_id":...,"distance":...})` where proximity is a bounded `1/(1+distance)`. Return `SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)` or `None`. Export it in `orion/substrate/relational/adapters/__init__.py` (`__all__`).

- [ ] **Step 4: Wire ingest in substrate-runtime**

In `services/orion-substrate-runtime/app/worker.py`, add a perception consumer (gated by `embodiment_perception_substrate_enabled`) on `embodiment_channel_perception`: decode `WorldPerceptionV1`, call `map_town_perception_to_substrate({"perception": payload})`, and if non-None write it via the shared substrate store (`_get_substrate_graph_store()`), fail-open. Start in `start()` only when the flag is on. Add a loop-damping comment referencing the spec (perception cadence is slower than the drive tick; contributions bounded; fail-open).

- [ ] **Step 5: Run tests**

Run: `pytest orion/embodiment/tests/test_town_perception_ctx.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/relational/adapters/town_perception_ctx.py \
  orion/substrate/relational/adapters/__init__.py services/orion-substrate-runtime/app/worker.py \
  orion/embodiment/tests/test_town_perception_ctx.py
git commit -m "feat(embodiment): perception->substrate adapter + ingest"
```

## Task 17: Self-state perception grounding

**Files:**
- Modify: `services/orion-self-state-runtime/app/settings.py` (flag + channel), `app/worker.py` (best-effort observability input), `.env_example`
- Test: `services/orion-self-state-runtime/tests/test_embodiment_perception_input.py`

- [ ] **Step 1: Add settings**

Add `embodiment_perception_selfstate_enabled: bool = Field(False, alias="EMBODIMENT_PERCEPTION_SELFSTATE_ENABLED")` and `embodiment_channel_perception: str = Field("orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION")` to `services/orion-self-state-runtime/app/settings.py`.

- [ ] **Step 2: Write the failing test**

Create `services/orion-self-state-runtime/tests/test_embodiment_perception_input.py`: assert that when the flag is on, a decoded `WorldPerceptionV1` is stored as the latest best-effort input (via the same mechanism the service uses for `hub_presence`/`attention_broadcast`), with an age gate; when off, ignored. Follow the existing observability-input pattern in `services/orion-self-state-runtime/tests/test_store_observability_loaders.py`.

- [ ] **Step 3: Implement**

Add a gated perception consumer that caches the latest `WorldPerceptionV1` (player_id + position + nearby count) as a best-effort observability input feeding `SelfStateV1.hub_presence`-style grounding (the "I am embodied near X" signal). Age-gate stale perception. Fail-open.

- [ ] **Step 4: `.env_example` + sync**

Add the two keys to `services/orion-self-state-runtime/.env_example`, then `python scripts/sync_local_env_from_example.py orion-self-state-runtime`.

- [ ] **Step 5: Run tests + commit**

Run: `pytest services/orion-self-state-runtime/tests -q` (expect PASS).

```bash
git add services/orion-self-state-runtime/app/settings.py services/orion-self-state-runtime/app/worker.py \
  services/orion-self-state-runtime/.env_example services/orion-self-state-runtime/tests/test_embodiment_perception_input.py
git commit -m "feat(embodiment): self-state perception grounding input"
```

---

# RUNG 4 — (D) deliberate producers + arbitration + speech + social initiation

Deliverables: finalize emit (governor) on the turn `correlation_id`; cortex background/quick-lane emit + perception read-model; `start_conversation` conversation-start input; cortex-generated town speech; Hub consumes outcomes. Verify: arbitration, speech injection, social cooldown.

## Task 18: Finalize (D) intent builder + governor emit

**Files:**
- Modify: `orion/harness/finalize.py`
- Modify: `services/orion-harness-governor/settings.py`, `.env_example`
- Test: `orion/harness/tests/test_finalize_embodiment_emit.py`

- [ ] **Step 1: Write the failing test**

Create `orion/harness/tests/test_finalize_embodiment_emit.py`: call a new pure helper `build_finalize_embodiment_intent(*, correlation_id, interlocutor_ref, relational)` and assert it returns a `source=deliberate` `EmbodimentIntentV1` carrying the given `correlation_id`, `kind="approach_player"` when `relational and interlocutor_ref`, `ref=interlocutor_ref`, and a non-empty `reason`; returns `None` when not relational.

```python
from __future__ import annotations

from orion.harness.finalize import build_finalize_embodiment_intent


def test_relational_turn_builds_deliberate_approach():
    intent = build_finalize_embodiment_intent(correlation_id="turn-7", interlocutor_ref="Juniper", relational=True)
    assert intent is not None
    assert intent.source == "deliberate"
    assert intent.correlation_id == "turn-7"
    assert intent.kind == "approach_player"
    assert intent.ref == "Juniper"


def test_non_relational_turn_no_intent():
    assert build_finalize_embodiment_intent(correlation_id="t", interlocutor_ref=None, relational=False) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/harness/tests/test_finalize_embodiment_emit.py -q`
Expected: FAIL (`build_finalize_embodiment_intent` undefined).

- [ ] **Step 3: Implement the builder + wire the emit**

In `orion/harness/finalize.py`:
- Add `EMBODIMENT_INTENT_CHANNEL = "orion:embodiment:intent"` next to the other channel constants (line 40-42).
- Add `build_finalize_embodiment_intent(...)` using `orion.embodiment.intents.build_intent`.
- At the turn-completion path (where the finalized reply is produced and the module already uses its `PublishFn`), when `EMBODIMENT_D_FINALIZE_ENABLED` env is true, build the intent and publish it via the existing `PublishFn` on `EMBODIMENT_INTENT_CHANNEL`. Read the flag with an `os.environ.get` helper mirroring `_quick_gate_epsilon()` (lines 45-50). Fail-open (never break the turn).

In `services/orion-harness-governor/settings.py` add `EMBODIMENT_D_FINALIZE_ENABLED` (default false) and document it in `.env_example`; `python scripts/sync_local_env_from_example.py orion-harness-governor`.

- [ ] **Step 4: Run test + commit**

Run: `pytest orion/harness/tests/test_finalize_embodiment_emit.py -q` (expect PASS).

```bash
git add orion/harness/finalize.py services/orion-harness-governor/settings.py \
  services/orion-harness-governor/.env_example orion/harness/tests/test_finalize_embodiment_emit.py
git commit -m "feat(embodiment): D finalize emit on turn correlation_id"
```

## Task 19: Cortex background (D) emit + perception read-model

**Files:**
- Modify: `services/orion-cortex-exec/app/*` (background emit + latest-perception cache), `settings.py`, `.env_example`
- Test: `services/orion-cortex-exec/tests/test_embodiment_background_emit.py`

- [ ] **Step 1: Inspect the cortex background/quick lane**

Run: `rg -n "background|quick" services/orion-cortex-exec/app/settings.py services/orion-cortex-exec/app | head -40` to locate the background request handler and where a background task can build+publish an intent.

- [ ] **Step 2: Write the failing test**

Create `services/orion-cortex-exec/tests/test_embodiment_background_emit.py`: assert a pure helper `build_background_embodiment_intent(perception, correlation_id)` builds a `source=deliberate` intent informed by the cached `WorldPerceptionV1` (e.g., `approach_player` toward the nearest nearby player when present; `wander` when alone), non-empty `reason`; and that the latest-perception cache is TTL'd/latest-wins keyed by `player_id`.

- [ ] **Step 3: Implement**

Add to cortex-exec: a latest-`WorldPerceptionV1` cache (keyed by `player_id`, latest-wins, TTL) fed by a gated consumer on `orion:embodiment:perception` (`EMBODIMENT_PERCEPTION_CORTEX_ENABLED`); a `build_background_embodiment_intent(...)` pure helper; and a background/quick-lane path that, when `EMBODIMENT_D_BACKGROUND_ENABLED`, publishes a `source=deliberate` intent to `orion:embodiment:intent`. Add both flags to `settings.py` + `.env_example`; `python scripts/sync_local_env_from_example.py orion-cortex-exec`.

- [ ] **Step 4: Run test + commit**

Run: `pytest services/orion-cortex-exec/tests/test_embodiment_background_emit.py -q` (expect PASS).

```bash
git add services/orion-cortex-exec/app services/orion-cortex-exec/settings.py \
  services/orion-cortex-exec/.env_example services/orion-cortex-exec/tests/test_embodiment_background_emit.py
git commit -m "feat(embodiment): cortex background D emit + perception read-model"
```

## Task 20: start_conversation actuation + social cooldown

**Files:**
- Modify: `services/orion-embodiment/app/worker.py` (conversation-start input + cooldown)
- Test: `services/orion-embodiment/tests/test_worker_start_conversation.py`

- [ ] **Step 1: Inspect upstream conversation inputs**

Run: `rg -n "startConversation|invite|accept|conversation" services/orion-ai-town/upstream/convex/aiTown/inputs.ts 2>/dev/null | head`. Identify the exact `sendInput` name/args for initiating a conversation (invite/accept). If `upstream/` absent, note it; keep the code path behind the flag and document the exact input names discovered.

- [ ] **Step 2: Write the failing test**

Create `services/orion-embodiment/tests/test_worker_start_conversation.py`:
- `start_conversation` with a resolvable target and cooldown clear → worker approaches (move_to) AND issues the conversation-start `send_input`; sets `_last_conversation_start`; outcome `actuated`.
- second `start_conversation` within `social_cooldown_sec` → `denied` with reason mentioning cooldown; no conversation-start `send_input`.
- `start_conversation` with no target → `denied`.

- [ ] **Step 3: Implement**

Extend `process_intent`: for `kind == "start_conversation"`, after resolving the target/destination, enforce the cooldown (`now - self._last_conversation_start < social_cooldown_sec` → `denied`), then `move_to` + the conversation-start `send_input(name=<discovered>, args={...})`, update `_last_conversation_start`. Keep involuntary `start_conversation` subject to the same cooldown (it already flows through here).

- [ ] **Step 4: Run test + commit**

Run: `pytest services/orion-embodiment/tests/test_worker_start_conversation.py -q` (expect PASS).

```bash
git add services/orion-embodiment/app/worker.py services/orion-embodiment/tests/test_worker_start_conversation.py
git commit -m "feat(embodiment): start_conversation actuation with social cooldown"
```

## Task 21: Cortex-generated town speech

**Files:**
- Create: `orion/embodiment/speech.py`
- Modify: `services/orion-embodiment/app/worker.py` (speech loop, gated)
- Test: `orion/embodiment/tests/test_speech.py`, `services/orion-embodiment/tests/test_worker_speech.py`

- [ ] **Step 1: Inspect the cortex chat request/reply rail**

Run: `rg -n "CortexChatRequest|rpc_request|cortex:.*request|quick" orion/core/bus services/orion-hub/app | head -40` to find how Hub requests an utterance (channel + request schema + reply). The speech bridge reuses this rail (no new schema).

- [ ] **Step 2: Write the failing test for the pure bridge**

Create `orion/embodiment/tests/test_speech.py`: `should_speak(perception, own_player_id)` returns True only when `perception.active_conversation` involves `own_player_id`; `build_speech_prompt(perception, own_player_id)` returns a prompt string containing the interlocutor + recent lines; `is_injectable(reply_text)` returns False for empty/whitespace (empty-shell guard).

- [ ] **Step 3: Implement `orion/embodiment/speech.py`**

Pure helpers: `should_speak(...)`, `build_speech_prompt(...)`, `is_injectable(reply_text) -> bool`. No I/O.

- [ ] **Step 4: Implement the worker speech loop**

In `services/orion-embodiment/app/worker.py` (gated by `speech_enabled`): when perception reports Orion in an `active_conversation`, request an utterance via the cortex chat/quick lane (`rpc_request` on the discovered channel, lane = `speech_lane`); if `is_injectable(reply)`, inject via `send_input` conversation inputs (`startTyping` → write/`finishSendingMessage`, names discovered in Task 20 Step 1). Guards: own-agent only (`player_id`), one in-flight utterance per conversation (wait for the injected message to land, bounded by perception cadence), empty reply not injected. Add `services/orion-embodiment/tests/test_worker_speech.py` asserting empty reply is not injected and own-agent-only is enforced (patch the cortex rpc + `send_input`).

- [ ] **Step 5: Run tests + commit**

Run: `pytest orion/embodiment/tests/test_speech.py services/orion-embodiment/tests/test_worker_speech.py -q` (expect PASS).

```bash
git add orion/embodiment/speech.py orion/embodiment/tests/test_speech.py \
  services/orion-embodiment/app/worker.py services/orion-embodiment/tests/test_worker_speech.py
git commit -m "feat(embodiment): cortex-generated town speech bridge"
```

## Task 22: Hub consumes outcomes + arbitration eval

**Files:**
- Modify: Hub outcome consumer (locate with `rg -n "embodiment|outcome" services/orion-hub/app` — add a lightweight trace consumer on `orion:embodiment:outcome`, or confirm the `*` catch-all already covers it; the channel already lists `orion-hub` as consumer)
- Create: `services/orion-embodiment/evals/test_embodiment_arbitration_eval.py`

- [ ] **Step 1: Write the arbitration eval**

Create `services/orion-embodiment/evals/test_embodiment_arbitration_eval.py` (mirror `orion-thought/evals` shape): replay a scripted sequence of intents through `EmbodimentWorker.process_intent` with a controlled clock — deliberate at t=0, involuntary at t=2s (within hold) → `preempted`; involuntary at t=10s (after hold) → `actuated`/`denied` (not preempted). Assert (a) deliberate wins during holds, (b) involuntary resumes after, (c) every outcome `reason` is non-empty (no empty-shell outcomes). `_score()` returns `(correct, total)`; assert accuracy == 1.0; `__main__` prints the score.

- [ ] **Step 2: Hub consumer**

Confirm Hub receives `orion:embodiment:outcome` (channel already declares `orion-hub` as consumer). If Hub needs an explicit handler for a debug/trace surface, add a minimal one; otherwise document that the `*`/trace path covers it. Do not build a UI panel in this rung beyond a trace line.

- [ ] **Step 3: Run eval + commit**

Run: `pytest services/orion-embodiment/evals -q` (expect PASS).

```bash
git add services/orion-embodiment/evals/test_embodiment_arbitration_eval.py services/orion-hub/app
git commit -m "feat(embodiment): outcome consumption + arbitration eval"
```

---

# RUNG 5 — Memory & continuity

Deliverable: salience gate + `JournalTriggerV1` emit on `orion:actions:trigger:journal.v1`. Verify: completed conversation passes the gate; bare proximity does not; salient event → journal written + indexed → recallable.

## Task 23: JournalTrigger contract extension

**Files:**
- Modify: `orion/journaler/schemas.py` (lines 10-29)
- Modify: `orion/bus/channels.yaml` (`orion:actions:trigger:journal.v1`, line 562 — producers already include `"*"`, so no producer change is strictly required, but add `orion-embodiment` explicitly for documentation)
- Test: `tests/test_embodiment_journal_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embodiment_journal_contract.py`:

```python
from __future__ import annotations

from orion.journaler.schemas import JournalTriggerV1


def test_town_episode_trigger_kind_allowed():
    t = JournalTriggerV1(trigger_kind="town_episode", source_kind="embodiment",
                         summary="Talked with Juniper in the town")
    assert t.trigger_kind == "town_episode"
    assert t.source_kind == "embodiment"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embodiment_journal_contract.py -q`
Expected: FAIL (validation error — `town_episode`/`embodiment` not in the Literals).

- [ ] **Step 3: Extend the enums**

In `orion/journaler/schemas.py`, add `"town_episode"` to `JournalTriggerKind` (line 10-18) and `"embodiment"` to `JournalSourceKind` (line 19-29). Grep consumers of these enums (`rg -n "trigger_kind|source_kind" orion/journaler services/orion-actions`) and confirm none switch exhaustively in a way that breaks on a new value (journaler composes from `summary`/`prompt_seed`; a new value is safe). Add `orion-embodiment` to the `orion:actions:trigger:journal.v1` producer list in `channels.yaml` for documentation.

- [ ] **Step 4: Run test + commit**

Run: `pytest tests/test_embodiment_journal_contract.py -q` (expect PASS).

```bash
git add orion/journaler/schemas.py orion/bus/channels.yaml tests/test_embodiment_journal_contract.py
git commit -m "feat(embodiment): extend journal trigger enums for town episodes"
```

## Task 24: Salience gate + journal emit

**Files:**
- Create: `orion/embodiment/salience.py`
- Modify: `services/orion-embodiment/app/worker.py` (gated journal emit)
- Test: `orion/embodiment/tests/test_salience.py`, `services/orion-embodiment/tests/test_worker_memory.py`

- [ ] **Step 1: Write the failing test for the pure gate**

Create `orion/embodiment/tests/test_salience.py`:

```python
from __future__ import annotations

from orion.embodiment.salience import SalienceState, evaluate_salience


def test_completed_conversation_is_salient():
    state = SalienceState()
    ev = evaluate_salience(
        {"type": "conversation_completed", "with": "Juniper", "utterances": 4}, state)
    assert ev.salient is True
    assert ev.summary


def test_first_encounter_is_salient_once():
    state = SalienceState()
    e1 = evaluate_salience({"type": "encounter", "player_id": "j"}, state)
    e2 = evaluate_salience({"type": "encounter", "player_id": "j"}, state)
    assert e1.salient is True
    assert e2.salient is False  # deduped


def test_bare_proximity_not_salient():
    ev = evaluate_salience({"type": "proximity", "player_id": "j"}, SalienceState())
    assert ev.salient is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/embodiment/tests/test_salience.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `orion/embodiment/salience.py`**

Pure gate: `SalienceState` tracks `seen_players: set[str]` and last-journaled timestamps; `evaluate_salience(event, state) -> SalienceEvaluation(salient: bool, summary: str, source_ref: str|None)`. Salient on `conversation_completed` (utterances exchanged) and first `encounter` with an unseen player; not salient on bare `proximity`; dedupe encounters via `state.seen_players`.

- [ ] **Step 4: Wire journal emit in the worker**

In `services/orion-embodiment/app/worker.py` (gated by `memory_enabled`): maintain a `SalienceState`; when a salient event is detected (from perception deltas / completed speech), build `JournalTriggerV1(trigger_kind="town_episode", source_kind="embodiment", summary=<who/what/disposition>, source_ref=<conversation/player id>)` and publish to `orion:actions:trigger:journal.v1`. Bounded/deduped by the gate. Add `services/orion-embodiment/tests/test_worker_memory.py` asserting a completed-conversation event publishes exactly one journal trigger and a proximity event publishes none (patch publish).

- [ ] **Step 5: Run tests + commit**

Run: `pytest orion/embodiment/tests/test_salience.py services/orion-embodiment/tests/test_worker_memory.py -q` (expect PASS).

```bash
git add orion/embodiment/salience.py orion/embodiment/tests/test_salience.py \
  services/orion-embodiment/app/worker.py services/orion-embodiment/tests/test_worker_memory.py
git commit -m "feat(embodiment): salience gate + journal emit for town episodes"
```

---

# Final: full gate, review, Docker smoke, PR

## Task 25: Full local gate + env parity

- [ ] **Step 1: Run the whole embodiment gate**

```bash
git diff --check
python scripts/sync_local_env_from_example.py orion-embodiment orion-substrate-runtime \
  orion-self-state-runtime orion-harness-governor orion-cortex-exec
pytest orion/embodiment/tests -q
pytest services/orion-embodiment/tests -q
pytest services/orion-embodiment/evals -q
pytest tests/test_embodiment_bus_catalog.py tests/test_embodiment_journal_contract.py -q
pytest services/orion-substrate-runtime/tests -q
pytest services/orion-self-state-runtime/tests -q
```

Expected: all green; no `.env` files staged.

- [ ] **Step 2: Verify no `.env` bleed**

```bash
git status --short
git check-ignore services/*/.env
```

Expected: no tracked `.env`.

## Task 26: Docker config validation (build smoke)

- [ ] **Step 1: Validate compose renders + builds**

```bash
docker compose --env-file .env --env-file services/orion-embodiment/.env \
  -f services/orion-embodiment/docker-compose.yml config
docker compose --env-file .env --env-file services/orion-embodiment/.env \
  -f services/orion-embodiment/docker-compose.yml build
```

If Docker is unavailable in the environment, state that plainly and rely on the deterministic pytest gate instead (per AGENTS.md §8).

- [ ] **Step 2: Optional live smoke (documented, not required in CI)**

Bring up the worker with `ORION_EMBODIMENT_ENABLED=false` and curl `/health`; then document (do not auto-run) the live actuation smoke: bootstrap `--write`, enable flags, publish a `go_to_location` intent, observe the sprite move on Convex.

## Task 27: Code review + fixes

- [ ] **Step 1:** Run the code review skill in a subagent (superpowers:requesting-code-review) over the branch diff.
- [ ] **Step 2:** Fix all material findings; re-run the affected tests.
- [ ] **Step 3:** Record findings + fixes + evidence in the PR report.

## Task 28: PR

- [ ] **Step 1:** Push and open the PR with the AGENTS.md §18 PR template (Summary, Outcome moved, Architecture touched, Files changed, Schema/bus/API changes, Env/config changes incl. `sync_local_env_from_example.py` run, Tests/Evals run, Docker checks, Review findings fixed, Restart required commands from the spec, Risks). Restart commands for `~/.fcc/.env` `--write` come from the spec's Bootstrap §.

```bash
git push -u origin HEAD
```

---

## Restart commands (for the PR report)

```bash
# New/changed services + env — rebuild the worker and re-up flag-touched services:
docker compose --env-file .env --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-self-state-runtime/.env -f services/orion-self-state-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
# After bootstrap --write updates ~/.fcc/.env, also re-up the FCC MCP telekinesis consumer to share the body:
docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d
```

(Do not run `sudo` restarts from the agent; print them for Juniper.)

---

## Spec-coverage self-check

| Spec section | Task(s) |
|--------------|---------|
| Schema contracts (Intent/Outcome/Perception/Persona) | 1, 2 |
| Registry + channels | 2 |
| Motor worker + arbiter + resolver + idle heartbeat | 3, 6, 9, 13 |
| Orion's body / persona projection + empty-shell guard | 7, 10 |
| Bootstrap script (dry-run/`--write`/`--resync-identity`) | 10 |
| Upstream character patch | 11 |
| (C) drive→movement producer | 14, 15 |
| Perception rail (emit + substrate + self-state + cortex) | 13, 16, 17, 19 |
| (D) finalize + background producers | 18, 19 |
| Arbitration (deliberate preempts) | 3, 9, 22 |
| Speech (cortex-generated, injected) | 21 |
| Social initiation + cooldown | 14, 20 |
| Memory & continuity (salience + journal) | 23, 24 |
| Env keys (all default-off) | 8, 12, 15, 17, 18, 19 |
| Failure modes / privacy / rollback (flag-gated, fail-open) | throughout; verified in 9, 25 |

> **Idle heartbeat** (`EMBODIMENT_IDLE_HEARTBEAT_SEC`, default 0/off) is configured in Task 8 settings; wiring it into the worker's consume loop is a small optional add — if enabled, self-emit a `wander`/`idle` intent after N seconds of no intent. Implement alongside Task 9 or defer as a documented follow-up (it is off by default and not on any acceptance check).
