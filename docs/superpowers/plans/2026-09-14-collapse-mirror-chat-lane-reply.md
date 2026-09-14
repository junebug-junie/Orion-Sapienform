# Collapse Mirror → live chat-lane reply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Juniper submits a Collapse Mirror, inject it as a You bubble in the live Hub chat session and generate Orion’s reply on the chat lane via `execute_unified_turn`, retiring the metacog+Notify toast path.

**Architecture:** Actions keeps the Juniper gate + `event_id` dedupe, then publishes a thin bus envelope to Hub instead of `cortex.orch.request`. Hub resolves the newest connected websocket `session_id` (fail closed if none), idempotently writes/pushes the You bubble, runs chat-lane unified turn (`fcc_model_label` unset, `source=collapse_mirror_reply`), and delivers Orion’s reply with the existing outreach socket+history rails. Old verb `actions.respond_to_juniper_collapse_mirror.v1` is deleted in the same patch.

**Tech Stack:** Python 3, Pydantic v2 schemas, Orion bus (`BaseEnvelope` + `channels.yaml` + registry), Hub FastAPI/websockets, `execute_unified_turn`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-collapse-mirror-chat-reply` — rename branch `docs/collapse-mirror-chat-reply` → `feat/collapse-mirror-chat-reply` at Task 1’s first code commit.

## Global Constraints

- Generation lane: **chat lane only** — `fcc_model_label` must be unset / absent on the unified-turn payload.
- Session: **live Hub websocket session only** — newest connected entry with a non-empty `session_id`. No `ACTIONS_SESSION_ID` / `collapse_mirror` / outreach fallback session.
- No live session → skip generation; audit/log `skipped:no_live_session`; no You bubble; optional quiet notify only.
- You bubble = mirror markdown from `collapse_to_markdown(entry)` as a user turn in that live session (UI + chat history).
- Do **not** re-run light `collapse_mirror.v1` recall or the `[INTROSPECT]`/`[MESSAGE]` metacog prompt for Juniper replies.
- No feature / kill-switch env flags. Ships live on merge. Rollback = revert the PR.
- No keyword / phrase detectors on mirror content.
- Hub idempotent on `event_id` — bus redelivery must not double You bubbles.
- Unified turn deferred / degraded / empty / timeout / error → no fabricated Orion reply; leave You if already written; audit failure.
- Prefer **delete** the old reply verb over leaving it dead.
- Observer ≠ juniper and Actions `event_id` dedupe behavior stay unchanged.
- Env parity: if any `.env_example` key is added/removed/renamed, run `python scripts/sync_local_env_from_example.py` from repo root in the same task.
- Follow TDD: failing test → minimal impl → pass → commit per task.
- Work from this worktree only (not the shared primary checkout).

## File map

| File | Responsibility |
|------|----------------|
| `orion/schemas/collapse_mirror_chat_reply.py` | Thin Actions→Hub request schema + channel/kind constants |
| `orion/bus/channels.yaml` | Register `orion:hub:collapse_mirror:chat_reply` |
| `orion/schemas/registry.py` | Register `CollapseMirrorChatReplyRequestV1` |
| `services/orion-actions/app/logic.py` | Replace cortex envelope builder with Hub envelope builder |
| `services/orion-actions/app/main.py` | `_handle_collapse` publishes Hub event |
| `services/orion-actions/app/settings.py` + `.env_example` + README | Drop reply-verb wiring; add Hub channel key |
| `services/orion-hub/scripts/collapse_mirror_chat_reply.py` | Consume event; You inject; chat-lane turn; deliver; idempotency |
| `services/orion-hub/scripts/endogenous_outreach.py` | Expose fail-closed live session + reusable push/history helpers |
| `services/orion-hub/scripts/main.py` | Construct + start/stop the Hub consumer |
| `services/orion-hub/static/js/app.js` | Render You bubble for `collapse_mirror_you` |
| `services/orion-hub/app/settings.py` + `.env_example` | Subscribe channel for the new envelope |
| Retire: verb adapter, verb YAML, orch direct trigger, self-study consumer refs | Remove metacog+Notify reply brain |
| Tests under actions + hub + schema/channel gates | Acceptance from the spec |

### Locked names (do not bikeshed mid-implementation)

| Concept | Value |
|---------|-------|
| Schema class | `CollapseMirrorChatReplyRequestV1` |
| Envelope `kind` / `message_kind` | `collapse.mirror.chat_reply.request.v1` |
| Bus channel | `orion:hub:collapse_mirror:chat_reply` |
| Python channel constant | `COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL` |
| Unified-turn `source` | `collapse_mirror_reply` |
| History / WS tag | `collapse_mirror_reply` |
| WS You kind | `collapse_mirror_you` |
| Orion socket kind | reuse `orion_outreach` via existing `_deliver` / `_push_to_sockets` |
| Actions audit `action_name` | keep `respond_to_juniper_collapse_mirror.v1` |
| Actions publish status | `dispatched` with `channel=orion:hub:collapse_mirror:chat_reply` |

---

### Task 1: Contract — schema + channel + registry

**Files:**
- Create: `orion/schemas/collapse_mirror_chat_reply.py`
- Modify: `orion/bus/channels.yaml` (add channel near other `orion:hub:*` / collapse entries)
- Modify: `orion/schemas/registry.py` (import + `"CollapseMirrorChatReplyRequestV1": CollapseMirrorChatReplyRequestV1`)
- Create: `orion/schemas/tests/test_collapse_mirror_chat_reply.py`

**Interfaces:**
- Produces:
  - `COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL: str = "orion:hub:collapse_mirror:chat_reply"`
  - `COLLAPSE_MIRROR_CHAT_REPLY_KIND: str = "collapse.mirror.chat_reply.request.v1"`
  - `class CollapseMirrorChatReplyRequestV1(BaseModel)` with fields below
- Consumes: `CollapseMirrorEntryV2` (nested full entry)

- [ ] **Step 1: Rename branch for code work**

```bash
cd /mnt/scripts/Orion-Sapienform-collapse-mirror-chat-reply
git branch -m feat/collapse-mirror-chat-reply
```

- [ ] **Step 2: Write the failing schema test**

Create `orion/schemas/tests/test_collapse_mirror_chat_reply.py`:

```python
from __future__ import annotations

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.collapse_mirror_chat_reply import (
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
from orion.schemas.registry import resolve


def test_collapse_mirror_chat_reply_request_round_trip() -> None:
    entry = CollapseMirrorEntryV2(
        event_id="evt-1",
        observer="juniper",
        trigger="t",
        observer_state=["tired"],
        type="reflect",
        emergent_entity="x",
        summary="felt a shift",
        mantra="stay with it",
    )
    req = CollapseMirrorChatReplyRequestV1(
        event_id="evt-1",
        observer="juniper",
        mirror_text="### Collapse Mirror\n",
        entry=entry,
    )
    dumped = req.model_dump(mode="json")
    again = CollapseMirrorChatReplyRequestV1.model_validate(dumped)
    assert again.event_id == "evt-1"
    assert again.observer == "juniper"
    assert again.entry.summary == "felt a shift"
    assert COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL == "orion:hub:collapse_mirror:chat_reply"
    assert COLLAPSE_MIRROR_CHAT_REPLY_KIND == "collapse.mirror.chat_reply.request.v1"


def test_collapse_mirror_chat_reply_registered() -> None:
    assert resolve("CollapseMirrorChatReplyRequestV1") is CollapseMirrorChatReplyRequestV1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest orion/schemas/tests/test_collapse_mirror_chat_reply.py -q`

Expected: FAIL with `ModuleNotFoundError` / import error for `collapse_mirror_chat_reply`.

- [ ] **Step 4: Write minimal schema + channel + registry**

Create `orion/schemas/collapse_mirror_chat_reply.py`:

```python
"""Actions → Hub: reply to a Juniper Collapse Mirror on the live chat lane.

Design: docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2

COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL = "orion:hub:collapse_mirror:chat_reply"
COLLAPSE_MIRROR_CHAT_REPLY_KIND = "collapse.mirror.chat_reply.request.v1"


class CollapseMirrorChatReplyRequestV1(BaseModel):
    """Thin Hub-bound request: mirror text for the You bubble + full entry for audit."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    observer: str = Field(min_length=1)
    mirror_text: str = Field(min_length=1, description="User-turn text (markdown) for the live session.")
    entry: CollapseMirrorEntryV2
```

Add to `orion/bus/channels.yaml` (place with other hub/collapse channels; exact YAML block):

```yaml
  - name: "orion:hub:collapse_mirror:chat_reply"
    kind: "event"
    schema_id: "CollapseMirrorChatReplyRequestV1"
    message_kind: "collapse.mirror.chat_reply.request.v1"
    producer_services: ["orion-actions"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-09-14"
```

In `orion/schemas/registry.py`:
1. Add import: `from orion.schemas.collapse_mirror_chat_reply import CollapseMirrorChatReplyRequestV1`
2. Add dict entry next to the other collapse schemas: `"CollapseMirrorChatReplyRequestV1": CollapseMirrorChatReplyRequestV1,`

- [ ] **Step 5: Run schema test + contract gates**

```bash
pytest orion/schemas/tests/test_collapse_mirror_chat_reply.py -q
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/collapse_mirror_chat_reply.py \
  orion/schemas/tests/test_collapse_mirror_chat_reply.py \
  orion/schemas/registry.py \
  orion/bus/channels.yaml
git commit -m "$(cat <<'EOF'
feat(collapse-mirror): add Actions→Hub chat-reply bus contract

EOF
)"
```

---

### Task 2: Actions — publish Hub envelope instead of cortex verb

**Files:**
- Modify: `services/orion-actions/app/logic.py`
- Modify: `services/orion-actions/app/main.py` (`_handle_collapse` ~L1527–1596)
- Modify: `services/orion-actions/app/settings.py`
- Modify: `services/orion-actions/.env_example`
- Modify: `services/orion-actions/README.md` (reply path description only)
- Modify: `services/orion-actions/tests/test_actions_v1.py`
- Run: `python scripts/sync_local_env_from_example.py` after `.env_example` change

**Interfaces:**
- Consumes: `CollapseMirrorChatReplyRequestV1`, `COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL`, `COLLAPSE_MIRROR_CHAT_REPLY_KIND`, existing `should_trigger` / `dedupe_key_for` / `collapse_to_markdown` / `ActionDedupe`
- Produces:
  - `build_collapse_mirror_chat_reply_envelope(parent, *, source, entry) -> BaseEnvelope`
  - `async def publish_collapse_mirror_chat_reply(*, bus, channel, envelope) -> None`
- Removes live use of `build_cortex_orch_envelope` from `_handle_collapse` (function may remain only if another caller needs it — after this task, grep; if only collapse used it, delete it in this task)

- [ ] **Step 1: Rewrite failing Actions tests**

Replace the body of `services/orion-actions/tests/test_actions_v1.py` so the primary dispatch test targets the Hub envelope (keep filter + dedupe tests):

```python
import asyncio
import os
import sys
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import (  # noqa: E402
    ActionDedupe,
    build_collapse_mirror_chat_reply_envelope,
    publish_collapse_mirror_chat_reply,
    should_trigger,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402
from orion.schemas.collapse_mirror_chat_reply import (  # noqa: E402
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
)


class _FakeBus:
    def __init__(self) -> None:
        self.published = []
        self.rpc_calls = 0

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))

    async def rpc_request(self, *args, **kwargs):
        self.rpc_calls += 1
        raise AssertionError("rpc_request should not be used for collapse chat reply")


def _entry(observer: str = "juniper", event_id: str = "evt-1") -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        event_id=event_id,
        observer=observer,
        trigger="t",
        observer_state=["a"],
        type="reflect",
        emergent_entity="x",
        summary="s",
        mantra="m",
    )


def _env() -> BaseEnvelope:
    return BaseEnvelope(
        kind="collapse.mirror.entry",
        source=ServiceRef(name="test"),
        correlation_id=str(uuid4()),
        payload={},
    )


def test_actions_publishes_hub_chat_reply_not_cortex_verb():
    bus = _FakeBus()
    parent = _env()
    env = build_collapse_mirror_chat_reply_envelope(
        parent,
        source=ServiceRef(name="orion-actions"),
        entry=_entry(observer="Juniper"),
    )

    asyncio.run(
        publish_collapse_mirror_chat_reply(
            bus=bus,
            channel=COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
            envelope=env,
        )
    )

    assert bus.rpc_calls == 0
    assert len(bus.published) == 1
    channel, published = bus.published[0]
    assert channel == COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL
    assert published.kind == COLLAPSE_MIRROR_CHAT_REPLY_KIND
    assert published.payload["event_id"] == "evt-1"
    assert published.payload["observer"].lower() == "juniper"
    assert "Collapse Mirror" in published.payload["mirror_text"]
    assert published.payload["entry"]["summary"] == "s"
    assert "verb" not in published.payload


def test_actions_filters_juniper_casefold():
    assert should_trigger(_entry(observer="juniper")) is True
    assert should_trigger(_entry(observer="Juniper")) is True
    assert should_trigger(_entry(observer="JUNIPER")) is True
    assert should_trigger(_entry(observer="orion")) is False


def test_actions_dedupe_prevents_double_dispatch():
    d = ActionDedupe(ttl_seconds=60)
    key = "collapse_123"
    assert d.try_acquire(key) is True
    assert d.try_acquire(key) is False
    d.mark_done(key)
    assert d.try_acquire(key) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-actions/tests/test_actions_v1.py -q`

Expected: FAIL — `build_collapse_mirror_chat_reply_envelope` not defined.

- [ ] **Step 3: Implement builder + publisher in `logic.py`**

Add imports at top of `logic.py`:

```python
from orion.schemas.collapse_mirror_chat_reply import (
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
```

Update `ACTION_CATALOG` description string to:

```python
description="Signal Hub to reply in the live chat session when Juniper writes a Collapse Mirror.",
```

Add (and delete `build_cortex_orch_envelope` if grep shows no remaining callers outside tests):

```python
def build_collapse_mirror_chat_reply_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    entry: CollapseMirrorEntryV2,
) -> BaseEnvelope:
    event_id = str(entry.event_id or entry.id or parent.correlation_id)
    req = CollapseMirrorChatReplyRequestV1(
        event_id=event_id,
        observer=str(entry.observer or ""),
        mirror_text=collapse_to_markdown(entry),
        entry=entry,
    )
    return parent.derive_child(
        kind=COLLAPSE_MIRROR_CHAT_REPLY_KIND,
        source=source,
        payload=req,
        reply_to=None,
    )


async def publish_collapse_mirror_chat_reply(
    *, bus: Any, channel: str, envelope: BaseEnvelope
) -> None:
    await bus.publish(channel, envelope)
```

Keep `ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB` **only** if Task 5 still needs the string for deletion greps; otherwise delete the constant in Task 5. For this task, stop importing it from tests.

- [ ] **Step 4: Swap `_handle_collapse` in `main.py`**

Change imports: drop `build_cortex_orch_envelope` / `dispatch_cortex_request` for this path; import `build_collapse_mirror_chat_reply_envelope`, `publish_collapse_mirror_chat_reply`.

Replace the try-body of `_handle_collapse` (the part that builds cortex env + dispatches) with:

```python
            req_env = build_collapse_mirror_chat_reply_envelope(
                env,
                source=src,
                entry=entry,
            )
            await publish_collapse_mirror_chat_reply(
                bus=hunter.bus,
                channel=settings.collapse_mirror_chat_reply_channel,
                envelope=req_env,
            )

            dt_ms = int((time.monotonic() - t0) * 1000)
            deduper.mark_done(event_id)
            await _audit(
                env,
                status="dispatched",
                event_id=event_id,
                action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
                extra={
                    "duration_ms": dt_ms,
                    "channel": settings.collapse_mirror_chat_reply_channel,
                    "kind": req_env.kind,
                },
            )
            logger.info(
                "dispatched collapse mirror chat reply event_id=%s corr=%s channel=%s",
                event_id,
                env.correlation_id,
                settings.collapse_mirror_chat_reply_channel,
            )
```

Skip/dedupe/fail audit branches stay as they are (same `action_name`).

- [ ] **Step 5: Settings + `.env_example` + README**

In `settings.py`:
- Add:
  ```python
  collapse_mirror_chat_reply_channel: str = Field(
      "orion:hub:collapse_mirror:chat_reply",
      alias="ACTIONS_COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL",
  )
  ```
- Remove (or stop documenting as the Juniper reply path): `actions_verb` / `ACTIONS_VERB`. If journal or another path still reads `ACTIONS_SESSION_ID` / `ACTIONS_RECALL_PROFILE`, leave those keys; only remove keys whose sole consumer was the collapse cortex dispatch. Grep before deleting:
  ```bash
  rg "actions_verb|ACTIONS_VERB|actions_session_id|ACTIONS_SESSION_ID|actions_recall_profile" services/orion-actions
  ```
  - If `ACTIONS_VERB` is only used by `_handle_collapse`, delete the field from settings + `.env_example`.
  - If `ACTIONS_SESSION_ID` / `ACTIONS_RECALL_PROFILE` are still used by journal or other handlers, keep them and update README to say they are **not** used for Juniper collapse chat reply.

Update `.env_example` to match. Sync local env:

```bash
python scripts/sync_local_env_from_example.py
```

Update `services/orion-actions/README.md` collapse section: Actions publishes `orion:hub:collapse_mirror:chat_reply`; Hub owns generation.

- [ ] **Step 6: Run Actions tests**

```bash
pytest services/orion-actions/tests/test_actions_v1.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add services/orion-actions/app/logic.py \
  services/orion-actions/app/main.py \
  services/orion-actions/app/settings.py \
  services/orion-actions/.env_example \
  services/orion-actions/README.md \
  services/orion-actions/tests/test_actions_v1.py
git commit -m "$(cat <<'EOF'
feat(actions): signal Hub for Juniper collapse chat replies

EOF
)"
```

---

### Task 3: Hub — live session fail-closed + You inject + chat-lane turn

**Files:**
- Modify: `services/orion-hub/scripts/endogenous_outreach.py` (add `live_session_id()`; optionally thin public wrappers)
- Create: `services/orion-hub/scripts/collapse_mirror_chat_reply.py`
- Create: `services/orion-hub/tests/test_collapse_mirror_chat_reply.py`
- Modify: `services/orion-hub/app/settings.py` + `.env_example` (subscribe channel)
- Run: `python scripts/sync_local_env_from_example.py`

**Interfaces:**
- Consumes: `EndogenousOutreach.register_connection` / `note_session` / `_deliver` / `_push_to_sockets` / history helpers; `execute_unified_turn` from `orion.hub.turn_orchestrator` (same import path outreach uses); `CollapseMirrorChatReplyRequestV1`
- Produces:
  - `EndogenousOutreach.live_session_id() -> str | None` (no fallback)
  - `class CollapseMirrorChatReplyHandler` with:
    - `async def start(self, bus) -> None`
    - `async def stop(self) -> None`
    - `async def handle(self, env: BaseEnvelope) -> dict` (returns status dict for tests/audit)
  - Constants: `YOU_KIND = "collapse_mirror_you"`, `SOURCE_TAG = "collapse_mirror_reply"`

- [ ] **Step 1: Write failing Hub tests**

Create `services/orion-hub/tests/test_collapse_mirror_chat_reply.py`:

```python
from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(SERVICE_DIR, "scripts")
for path in (SERVICE_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402
from orion.schemas.collapse_mirror_chat_reply import (  # noqa: E402
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
from scripts.collapse_mirror_chat_reply import CollapseMirrorChatReplyHandler  # noqa: E402
from scripts.endogenous_outreach import EndogenousOutreach  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.enabled = True
        self.published = []

    async def publish(self, channel: str, envelope) -> None:
        self.published.append((channel, envelope))

    def subscribe(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("tests call handle() directly")


def _entry(event_id: str = "evt-1") -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        event_id=event_id,
        observer="juniper",
        trigger="t",
        observer_state=["a"],
        type="reflect",
        emergent_entity="x",
        summary="mirror summary",
        mantra="m",
    )


def _request_env(event_id: str = "evt-1") -> BaseEnvelope:
    entry = _entry(event_id=event_id)
    req = CollapseMirrorChatReplyRequestV1(
        event_id=event_id,
        observer="juniper",
        mirror_text="### Collapse Mirror\n\n**Summary:** mirror summary\n",
        entry=entry,
    )
    return BaseEnvelope(
        kind=COLLAPSE_MIRROR_CHAT_REPLY_KIND,
        source=ServiceRef(name="orion-actions"),
        correlation_id=str(uuid4()),
        payload=req.model_dump(mode="json"),
    )


def _outreach_with_session(session_id: str = "live-sess") -> EndogenousOutreach:
    outreach = EndogenousOutreach(
        enabled=True,
        fallback_session_id="MUST_NOT_USE",
        notify_channel="orion:notify:in_app",
    )
    q: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q, active_turn={})
    outreach.note_session("c1", session_id)
    return outreach


@pytest.mark.asyncio
async def test_live_session_id_fail_closed_no_fallback() -> None:
    outreach = EndogenousOutreach(
        enabled=True,
        fallback_session_id="fallback-sess",
        notify_channel="orion:notify:in_app",
    )
    assert outreach.live_session_id() is None
    q: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q, active_turn={})
    # connected but no session yet
    assert outreach.live_session_id() is None
    outreach.note_session("c1", "sess-9")
    assert outreach.live_session_id() == "sess-9"


@pytest.mark.asyncio
async def test_no_live_session_skips_without_you_or_generation(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = EndogenousOutreach(
        enabled=True,
        fallback_session_id="fallback-sess",
        notify_channel="orion:notify:in_app",
    )
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    called = {"turn": 0}

    async def _boom(*args, **kwargs):
        called["turn"] += 1
        raise AssertionError("must not generate without live session")

    monkeypatch.setattr(
        "scripts.collapse_mirror_chat_reply.execute_unified_turn",
        _boom,
    )
    result = await handler.handle(_request_env())
    assert result["status"] == "skipped"
    assert result["reason"] == "no_live_session"
    assert called["turn"] == 0
    # no user history publish
    assert bus.published == []


@pytest.mark.asyncio
async def test_live_session_injects_you_runs_chat_lane_and_delivers(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    turn_calls = []

    async def _fake_turn(**kwargs):
        turn_calls.append(kwargs)
        return [
            {
                "llm_response": "I hear that shift.",
                "fcc_model_label": None,
            }
        ]

    monkeypatch.setattr(
        "scripts.collapse_mirror_chat_reply.execute_unified_turn",
        _fake_turn,
    )

    history = []

    async def _fake_publish_history(bus_arg, envelopes):
        history.extend(envelopes)

    monkeypatch.setattr(
        "scripts.chat_history.publish_chat_history",
        _fake_publish_history,
    )

    result = await handler.handle(_request_env("evt-live"))
    assert result["status"] == "delivered"
    assert result["session_id"] == "live-sess"
    assert len(turn_calls) == 1
    payload = turn_calls[0]["payload"]
    assert payload.get("source") == "collapse_mirror_reply"
    assert "fcc_model_label" not in payload or payload.get("fcc_model_label") in (None, "")
    assert turn_calls[0]["user_message"].startswith("### Collapse Mirror")
    # You + Orion history (You first)
    roles = [getattr(e.payload, "role", None) for e in history]
    assert roles[0] == "user"
    assert "assistant" in roles
    # socket got You then Orion
    q = outreach._connections["c1"]["queue"]
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert any(f.get("kind") == "orion_outreach" for f in frames)


@pytest.mark.asyncio
async def test_idempotent_on_event_id(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    turn_calls = {"n": 0}

    async def _fake_turn(**kwargs):
        turn_calls["n"] += 1
        return [{"llm_response": "ok", "fcc_model_label": None}]

    monkeypatch.setattr(
        "scripts.collapse_mirror_chat_reply.execute_unified_turn",
        _fake_turn,
    )
    monkeypatch.setattr(
        "scripts.chat_history.publish_chat_history",
        lambda *a, **k: asyncio.sleep(0),
    )

    env = _request_env("evt-dup")
    first = await handler.handle(env)
    second = await handler.handle(env)
    assert first["status"] == "delivered"
    assert second["status"] == "skipped"
    assert second["reason"] == "deduped"
    assert turn_calls["n"] == 1
```

Note: `EndogenousOutreach.__init__` kwargs may differ slightly — open the constructor and pass the same required args tests in `test_endogenous_outreach.py` use. Adjust the test helpers to match the real signature; do not invent new constructor params.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform-collapse-mirror-chat-reply
pytest services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q
```

Expected: FAIL — module / `live_session_id` / handler missing.

- [ ] **Step 3: Add `live_session_id()` on `EndogenousOutreach`**

In `endogenous_outreach.py`, next to `_active_session_id`:

```python
    def live_session_id(self) -> str | None:
        """Newest connected socket that has a session_id. None if none — no fallback."""
        for entry in reversed(list(self._connections.values())):
            sid = entry.get("session_id")
            if sid:
                return str(sid)
        return None
```

Do **not** change `_active_session_id` behavior (outreach still uses fallback).

- [ ] **Step 4: Implement `collapse_mirror_chat_reply.py`**

Create `services/orion-hub/scripts/collapse_mirror_chat_reply.py` with this shape (adapt imports to match outreach’s real `execute_unified_turn` import path — copy from `_attempt_unified_turn`):

```python
"""Hub: Juniper Collapse Mirror → live chat You bubble + chat-lane reply.

Design: docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror_chat_reply import (
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
from orion.hub.turn_orchestrator import execute_unified_turn

logger = logging.getLogger("orion-hub.collapse_mirror_chat_reply")

YOU_KIND = "collapse_mirror_you"
SOURCE_TAG = "collapse_mirror_reply"
DEFAULT_TURN_TIMEOUT_SEC = 120.0


class CollapseMirrorChatReplyHandler:
    def __init__(
        self,
        *,
        outreach: Any,
        bus: Any = None,
        channel: str = COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
        dedupe_ttl_seconds: int = 86400,
        turn_timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> None:
        self._outreach = outreach
        self._bus = bus
        self._channel = channel
        self._dedupe_ttl = int(dedupe_ttl_seconds)
        self._turn_timeout_sec = float(turn_timeout_sec)
        self._done_expiry: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def _prune(self, now: float) -> None:
        expired = [k for k, exp in self._done_expiry.items() if exp <= now]
        for k in expired:
            self._done_expiry.pop(k, None)

    def _try_claim(self, event_id: str) -> bool:
        now = time.time()
        self._prune(now)
        if event_id in self._done_expiry:
            return False
        self._done_expiry[event_id] = now + self._dedupe_ttl
        return True

    async def start(self, bus: Any) -> None:
        self._bus = bus
        self._stopping = False
        self._task = asyncio.create_task(self._consume_loop(), name="collapse_mirror_chat_reply")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _consume_loop(self) -> None:
        assert self._bus is not None
        while not self._stopping:
            try:
                async with self._bus.subscribe(self._channel) as pubsub:
                    async for msg in self._bus.iter_messages(pubsub):
                        if self._stopping:
                            break
                        try:
                            decoded = self._bus.codec.decode(msg.get("data"))
                            env = BaseEnvelope.model_validate(decoded) if not isinstance(decoded, BaseEnvelope) else decoded
                            await self.handle(env)
                        except Exception:  # noqa: BLE001
                            logger.exception("collapse_mirror_chat_reply_handle_failed")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("collapse_mirror_chat_reply_subscribe_failed")
                await asyncio.sleep(1.0)

    async def handle(self, env: BaseEnvelope) -> dict:
        try:
            req = CollapseMirrorChatReplyRequestV1.model_validate(env.payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("collapse_mirror_chat_reply_invalid_payload err=%s", exc)
            return {"status": "skipped", "reason": "invalid_payload"}

        event_id = req.event_id
        if not self._try_claim(event_id):
            logger.info("collapse_mirror_chat_reply_deduped event_id=%s", event_id)
            return {"status": "skipped", "reason": "deduped", "event_id": event_id}

        session_id = self._outreach.live_session_id()
        if not session_id:
            # release claim? No — redelivery should stay skipped for this event_id
            # until TTL; optional quiet notify only.
            await self._optional_held_back_notify(event_id=event_id, correlation_id=str(env.correlation_id))
            logger.info("collapse_mirror_chat_reply_skipped reason=no_live_session event_id=%s", event_id)
            return {"status": "skipped", "reason": "no_live_session", "event_id": event_id}

        correlation_id = str(env.correlation_id)
        await self._inject_you(
            text=req.mirror_text,
            session_id=session_id,
            correlation_id=correlation_id,
        )

        try:
            text, debug = await self._run_chat_lane_turn(
                user_message=req.mirror_text,
                session_id=session_id,
                correlation_id=correlation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "collapse_mirror_chat_reply_turn_failed event_id=%s corr=%s",
                event_id,
                correlation_id,
            )
            return {
                "status": "failed",
                "reason": str(exc),
                "event_id": event_id,
                "session_id": session_id,
                "you_written": True,
            }

        if not text or not str(text).strip():
            logger.info(
                "collapse_mirror_chat_reply_empty_reply event_id=%s corr=%s",
                event_id,
                correlation_id,
            )
            return {
                "status": "failed",
                "reason": "empty_reply",
                "event_id": event_id,
                "session_id": session_id,
                "you_written": True,
            }

        await self._outreach._deliver(
            text=str(text).strip(),
            session_id=session_id,
            correlation_id=correlation_id,
            model=debug.get("fcc_model_label") if isinstance(debug, dict) else None,
            source_tag=SOURCE_TAG,
        )
        return {
            "status": "delivered",
            "event_id": event_id,
            "session_id": session_id,
            "you_written": True,
        }

    async def _inject_you(self, *, text: str, session_id: str, correlation_id: str) -> None:
        message_id = str(uuid4())
        payload = {
            "kind": YOU_KIND,
            "text": text,
            "correlation_id": correlation_id,
            "message_id": message_id,
            "session_id": session_id,
        }
        for connection_id, entry in list(self._outreach._connections.items()):
            queue = entry.get("queue")
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "collapse_mirror_you_push_failed connection=%s err=%s",
                    connection_id,
                    exc,
                )
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            env = build_chat_history_envelope(
                content=text,
                role="user",
                session_id=session_id,
                correlation_id=correlation_id,
                speaker="Juniper",
                tags=[SOURCE_TAG],
                message_id=message_id,
                client_meta={"collapse_mirror_you": True},
            )
            await publish_chat_history(self._bus or self._outreach._bus, [env])
        except Exception as exc:  # noqa: BLE001
            logger.warning("collapse_mirror_you_history_failed corr=%s err=%s", correlation_id, exc)

    async def _run_chat_lane_turn(
        self,
        *,
        user_message: str,
        session_id: str,
        correlation_id: str,
    ) -> tuple[str, dict]:
        request_payload: dict[str, Any] = {
            "no_write": True,
            "source": SOURCE_TAG,
        }
        # Intentionally omit fcc_model_label → chat lane.
        frames = await asyncio.wait_for(
            execute_unified_turn(
                bus=self._bus or self._outreach._bus,
                correlation_id=correlation_id,
                session_id=session_id,
                user_message=user_message,
                payload=request_payload,
                continuity_messages=None,
            ),
            timeout=self._turn_timeout_sec,
        )
        final = frames[-1] if frames else {}
        text = str((final or {}).get("llm_response") or "").strip()
        debug = {
            "fcc_model_label": (final or {}).get("fcc_model_label"),
        }
        return text, debug

    async def _optional_held_back_notify(self, *, event_id: str, correlation_id: str) -> None:
        """Quiet note that a reply was held back — never the reply body itself."""
        # Reuse outreach notify channel if present; swallow all errors.
        try:
            from datetime import datetime, timezone
            from uuid import UUID

            from orion.schemas.notify import HubNotificationEvent  # adjust import if path differs

            bus = self._bus or getattr(self._outreach, "_bus", None)
            channel = getattr(self._outreach, "notify_channel", None)
            if not bus or not channel:
                return
            notification = HubNotificationEvent(
                notification_id=uuid4(),
                created_at=datetime.now(timezone.utc),
                severity="info",
                event_kind="orion.chat.message",
                source_service="orion-hub",
                title="Collapse Mirror reply held back",
                body_text="No live Hub chat session was connected, so Orion did not reply in chat.",
                tags=[SOURCE_TAG, "held_back"],
                correlation_id=correlation_id,
                notification_type="collapse_mirror_held_back",
            )
            env = BaseEnvelope(
                kind="notify.in_app.v1",
                source=ServiceRef(name="orion-hub"),
                correlation_id=correlation_id,
                payload=notification.model_dump(mode="json"),
            )
            await bus.publish(channel, env)
        except Exception:  # noqa: BLE001
            logger.debug("collapse_mirror_held_back_notify_failed event_id=%s", event_id, exc_info=True)
```

**Implementer notes (do not skip):**
1. Match `execute_unified_turn`’s real keyword signature from `endogenous_outreach._attempt_unified_turn` (copy kwargs exactly; drop ones you do not have).
2. Match `HubNotificationEvent` constructor fields from outreach’s `_publish_notification`.
3. `EndogenousOutreach` constructor in tests must match production — copy from `test_endogenous_outreach.py` helpers.
4. Calling `outreach._deliver` is intentional reuse for this thin seam; do not duplicate socket+history+notify.

- [ ] **Step 5: Hub settings for subscribe channel**

In `services/orion-hub/app/settings.py` add:

```python
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL: str = Field(
        "orion:hub:collapse_mirror:chat_reply",
        alias="COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL",
    )
```

Mirror in `services/orion-hub/.env_example`. Sync:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 6: Run Hub tests**

```bash
pytest services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q
```

Expected: PASS. Fix constructor/import mismatches until green — do not weaken assertions about chat lane / no fallback / idempotency.

- [ ] **Step 7: Commit**

```bash
git add services/orion-hub/scripts/collapse_mirror_chat_reply.py \
  services/orion-hub/scripts/endogenous_outreach.py \
  services/orion-hub/tests/test_collapse_mirror_chat_reply.py \
  services/orion-hub/app/settings.py \
  services/orion-hub/.env_example
git commit -m "$(cat <<'EOF'
feat(hub): collapse mirror live-session chat-lane reply handler

EOF
)"
```

---

### Task 4: Wire Hub lifespan + frontend You bubble

**Files:**
- Modify: `services/orion-hub/scripts/main.py` (construct/start/stop handler beside `endogenous_outreach`)
- Modify: `services/orion-hub/static/js/app.js` (WS kind handler before generic assistant branch)
- Optional: tiny JS-free Python test already covers push payload; no separate JS test harness required unless one already exists for outreach kinds

**Interfaces:**
- Consumes: module-global `endogenous_outreach`, Hub `bus`, `settings.COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL`
- Produces: running consumer; client renders You via `appendMessage('You', …)`

- [ ] **Step 1: Wire lifespan in `main.py`**

Near other globals (`endogenous_outreach = None`):

```python
collapse_mirror_chat_reply_handler = None
```

After `endogenous_outreach` is constructed and `await endogenous_outreach.start(...)`:

```python
            from scripts.collapse_mirror_chat_reply import CollapseMirrorChatReplyHandler

            collapse_mirror_chat_reply_handler = CollapseMirrorChatReplyHandler(
                outreach=endogenous_outreach,
                bus=bus,
                channel=settings.COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
            )
            await collapse_mirror_chat_reply_handler.start(bus)
```

On shutdown, before/after outreach stop:

```python
    if collapse_mirror_chat_reply_handler is not None:
        try:
            await collapse_mirror_chat_reply_handler.stop()
        except Exception:  # noqa: BLE001
            logger.exception("collapse_mirror_chat_reply_stop_failed")
        collapse_mirror_chat_reply_handler = None
```

Ensure `global` declarations include the new name where outreach’s do.

- [ ] **Step 2: Frontend You handler**

In `services/orion-hub/static/js/app.js`, immediately **before** the `if (d.kind === 'orion_outreach')` block (~L11454), insert:

```javascript
          if (d.kind === 'collapse_mirror_you') {
            const youText = String(d.text || '').trim();
            if (youText) {
              appendMessage('You', youText, 'text-white', {
                correlationId: d.correlation_id,
                messageId: d.message_id || null,
                turnId: d.correlation_id,
                unsolicited: false,
                collapseMirror: true,
              });
            }
            return;
          }
```

Also, in the toast/notification path that skips toast for `endogenous_outreach` (~L6567), skip toast for `notification_type === 'collapse_mirror_held_back'` **or** show a quiet/info toast — either is fine; do not show the held-back note as if it were Orion’s chat reply.

- [ ] **Step 3: Smoke-level import check**

```bash
python -c "from scripts.collapse_mirror_chat_reply import CollapseMirrorChatReplyHandler; from scripts.main import *"
```

If `scripts.main` import is too heavy, instead:

```bash
pytest services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q
rg -n "collapse_mirror_you|CollapseMirrorChatReplyHandler" services/orion-hub/scripts/main.py services/orion-hub/static/js/app.js
```

Expected: handler wired; JS kind present.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/scripts/main.py services/orion-hub/static/js/app.js
git commit -m "$(cat <<'EOF'
feat(hub): wire collapse mirror chat reply consumer and You bubble

EOF
)"
```

---

### Task 5: Retire metacog+Notify reply verb

**Files:**
- Delete: `orion/cognition/verbs/actions.respond_to_juniper_collapse_mirror.v1.yaml`
- Modify: `services/orion-cortex-exec/app/verb_adapters.py` — remove `RespondToJuniperCollapseMirrorVerb` class and `JuniperCollapseActionOutput` if unused
- Modify: `services/orion-cortex-exec/app/main.py` — drop import
- Modify: `services/orion-cortex-orch/app/orchestrator.py` — remove from `_DIRECT_VERB_TRIGGERS`
- Modify: `services/orion-cortex-exec/app/self_study_policy.py` — remove `elif consumer_name == "actions.respond_to_juniper_collapse_mirror.v1"` branch
- Modify: `services/orion-cortex-exec/app/self_study_harness.py` — remove scenario that targets that consumer
- Modify/Delete: `services/orion-cortex-exec/tests/test_actions_verb.py` — delete file if solely for this verb; otherwise strip tests
- Modify: `services/orion-cortex-exec/tests/test_self_study_harness.py` / `test_self_study_policy.py` — stop asserting that consumer
- Modify: `docs/superpowers/specs/2026-09-03-self-study-wiring-map.md` — note consumer retired (one-line)
- Grep purge leftovers

**Interfaces:**
- Produces: no callable `actions.respond_to_juniper_collapse_mirror.v1`
- Consumes: nothing — deletion task

- [ ] **Step 1: Grep for all live references**

```bash
rg -n "respond_to_juniper_collapse_mirror|RespondToJuniperCollapseMirror|ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB|JuniperCollapseActionOutput" \
  --glob '!docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md' \
  --glob '!docs/superpowers/plans/2026-09-14-collapse-mirror-chat-lane-reply.md'
```

Treat every hit outside the design/plan docs as a deletion or update target.

- [ ] **Step 2: Write a failing “verb gone” guard test**

Create `services/orion-cortex-exec/tests/test_collapse_mirror_reply_verb_retired.py`:

```python
from __future__ import annotations

import os
from pathlib import Path


def test_respond_to_juniper_collapse_mirror_verb_yaml_removed() -> None:
    root = Path(__file__).resolve().parents[3]
    verb_path = root / "orion" / "cognition" / "verbs" / "actions.respond_to_juniper_collapse_mirror.v1.yaml"
    assert not verb_path.exists()


def test_verb_adapter_symbol_removed() -> None:
    import app.verb_adapters as adapters

    assert not hasattr(adapters, "RespondToJuniperCollapseMirrorVerb")
```

- [ ] **Step 3: Run to verify fail**

```bash
pytest services/orion-cortex-exec/tests/test_collapse_mirror_reply_verb_retired.py -q
```

Expected: FAIL (yaml still exists / symbol still present).

- [ ] **Step 4: Delete verb + update callers**

1. Delete the verb YAML.
2. Remove the class and any now-unused helpers only used by it in `verb_adapters.py` (keep shared helpers if journal/other verbs use them — grep first).
3. Drop import from `main.py`.
4. Remove orch direct trigger string.
5. Update self_study policy/harness + their tests so they no longer require that consumer.
6. Delete or gut `test_actions_verb.py`.

- [ ] **Step 5: Run focused tests**

```bash
pytest services/orion-cortex-exec/tests/test_collapse_mirror_reply_verb_retired.py \
  services/orion-cortex-exec/tests/test_self_study_harness.py \
  services/orion-cortex-exec/tests/test_self_study_policy.py -q
# If test_actions_verb.py still exists, it must pass or be deleted:
pytest services/orion-cortex-exec/tests/test_actions_verb.py -q 2>/dev/null || true
rg -n "RespondToJuniperCollapseMirror|actions.respond_to_juniper_collapse_mirror.v1" \
  services/orion-cortex-exec services/orion-cortex-orch orion/cognition \
  --glob '!**/2026-09-14-collapse-mirror-chat-lane-reply*'
```

Expected: tests PASS; grep clean outside historical docs.

- [ ] **Step 6: Commit**

```bash
git add -u orion/cognition/verbs \
  services/orion-cortex-exec \
  services/orion-cortex-orch \
  docs/superpowers/specs/2026-09-03-self-study-wiring-map.md
git commit -m "$(cat <<'EOF'
chore: retire metacog Notify collapse mirror reply verb

EOF
)"
```

---

### Task 6: Final gates, docs pointer, deploy notes

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md` — set Status to `implemented` only after code is green (or leave status and add “see plan” — prefer Status → `implemented` when Task 6 finishes)
- No new feature-flag docs

- [ ] **Step 1: Run acceptance gate suite**

```bash
cd /mnt/scripts/Orion-Sapienform-collapse-mirror-chat-reply
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
python scripts/check_env_template_parity.py
pytest orion/schemas/tests/test_collapse_mirror_chat_reply.py \
  services/orion-actions/tests/test_actions_v1.py \
  services/orion-hub/tests/test_collapse_mirror_chat_reply.py \
  services/orion-cortex-exec/tests/test_collapse_mirror_reply_verb_retired.py -q
```

Expected: all PASS.

- [ ] **Step 2: Confirm no silent second brain**

```bash
rg -n "metacog_background|orion.chat.message|respond_to_juniper_collapse" \
  services/orion-actions/app services/orion-cortex-exec/app/verb_adapters.py \
  services/orion-hub/scripts/collapse_mirror_chat_reply.py
```

Expected: Actions/Hub reply path does not call metacog_background; Notify `orion.chat.message` only appears as optional held-back note (if implemented), not as the reply brain.

- [ ] **Step 3: Update spec status line**

In the design spec header: `**Status:** implemented` (keep the design body as historical truth).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md
git commit -m "$(cat <<'EOF'
docs: mark collapse mirror chat-lane reply design implemented

EOF
)"
```

- [ ] **Step 5: Manual smoke checklist (operator, post-deploy)**

Restart required (print for Juniper; do not sudo from the agent):

```bash
# From the feat worktree, via safe_docker_build.sh:
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Manual:

1. Open Hub chat (ensure a `session_id` is established — send any hello/message if needed).
2. Submit a Juniper Collapse Mirror.
3. Expect **You** (mirror markdown) then **Orion** in that session — not a toast-as-reply.
4. Confirm turn evidence is chat-lane / stance / harness-gov path (Hub/cortex logs), not `metacog_background`.
5. Close all Hub tabs; submit another mirror → skipped/`no_live_session`; no fake Orion text.

---

## Self-review (plan author)

**Spec coverage**

| Spec item | Task |
|-----------|------|
| Actions→Hub thin envelope + channel + registry | Task 1 |
| Actions gate/dedupe kept; stop cortex verb dispatch | Task 2 |
| Hub live session only; fail closed | Task 3 |
| You bubble + chat-lane `execute_unified_turn` + outreach deliver | Task 3–4 |
| Idempotent on `event_id` | Task 3 |
| Optional quiet held-back notify | Task 3 |
| Frontend shows You | Task 4 |
| Retire old verb / no second brain | Task 5 |
| No feature flag; env is wiring only | Tasks 2–3 |
| Gate tests listed in acceptance | Tasks 1–3, 5–6 |
| Manual smoke | Task 6 |
| Privacy: live session only | Task 3 (no multi-session fanout beyond existing connection push) |

**Placeholder scan:** none intentional — implementer must still align `EndogenousOutreach` / `execute_unified_turn` / `HubNotificationEvent` constructors with live signatures (called out explicitly).

**Type consistency:** `CollapseMirrorChatReplyRequestV1`, channel/kind constants, `SOURCE_TAG=collapse_mirror_reply`, `YOU_KIND=collapse_mirror_you` are locked across tasks.
