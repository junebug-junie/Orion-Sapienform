# AC Sensing Fix (Plan 1 of 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the portable-AC reading impossible to fake: a silent Shelly plug or dead Z-Wave link must show up as `stale` with no wattage, everywhere (bus sample, Postgres, Hub panel), within 2 minutes.

**Architecture:** `orion-zwave` tracks when it last *actually* heard from the plug (a successful `node.poll_value` or a pushed `value updated` event for Electric_W). Each published sample carries `provenance.sample_age_sec`; past `COOLING_STALE_AFTER_SEC` the sample omits every cached reading and sets `state.stale=true`. The poll loop reconnects a dead websocket instead of publishing cache forever. sql-writer persists `stale` + `sample_age_sec`; Hub reads them from `payload_json` and paints the Cabinet panel red.

**Tech Stack:** Python 3, pydantic v2, websockets, SQLAlchemy, FastAPI, vanilla JS. Tests: pytest (+ pytest-asyncio).

**Spec:** `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 5). Plans 2–5 (pool urgent priority; seeded runs + report + run mechanics; hardware-watch) are written after this one merges.

## Global Constraints

- `COOLING_STALE_AFTER_SEC=120` default.
- Stale ⇒ omit `cooling_watts`, `cooling_volts`, `cooling_amps`, `switch_on`; never republish cache as live; never zero-fill.
- "Fresh" = successful `node.poll_value` for Electric_W (live check 2026-09-28: 3/3 polls succeeded, 0.04–0.44 s, 885.7 W) or a pushed Electric_W `value updated` event.
- `ORION_BUS_URL` stays `redis://100.92.216.81:6379/0`.
- Test interpreter: `/mnt/scripts/Orion-Sapienform/.venv/bin/python` (system python has no pytest). Below, `PY` means that path.
- Baselines before any change (2026-09-28): `orion-zwave` 12 passed; Hub `test_cabinet_cooling_routes.py` 11 passed; sql-writer `test_home_cooling_sample_sql_shape.py` 5 passed.
- Work in worktree `/mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch` on a new branch `fix/ac-sensing-stale` cut from `origin/main` (not the docs branch).

## File Structure

| File | Responsibility |
|---|---|
| `orion/schemas/telemetry/home_cooling.py` | add `CoolingObservedStateV1.stale` |
| `services/orion-zwave/app/zwave_client.py` | freshness tracking, `connected`, loud poll failures |
| `services/orion-zwave/app/main.py` | stale-aware `build_cooling_sample`, `poll_once`, reconnecting loop, heartbeat details |
| `services/orion-zwave/app/settings.py`, `.env_example`, `README.md` | `COOLING_STALE_AFTER_SEC` |
| `services/orion-zwave/tests/test_freshness.py` (new) | client + sample + loop tests |
| `services/orion-sql-writer/app/models/home_cooling_sample.py`, `app/worker.py`, `app/main.py` | `stale`, `sample_age_sec` columns + boot DDL + mapping |
| `services/orion-hub/scripts/cabinet_cooling_routes.py` | `sensor_stale`, `sample_age_sec`, `last_fresh_at` in `/latest` |
| `services/orion-hub/static/js/cabinet-sensors.js` | red STALE render |

---

### Task 0: Branch

- [ ] **Step 1: Cut the branch**

```bash
cd /mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch
git fetch -q origin main
git switch -c fix/ac-sensing-stale origin/main
git checkout docs/urgent-curiosity-hardware-watch -- docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-1-ac-sensing.md
git commit -m "docs: urgent curiosity + hardware watch spec and AC sensing plan"
```

---

### Task 1: Schema field `stale`

**Files:**
- Modify: `orion/schemas/telemetry/home_cooling.py` (`CoolingObservedStateV1`)
- Test: `services/orion-zwave/tests/test_freshness.py` (create)

**Interfaces:**
- Produces: `CoolingObservedStateV1.stale: Optional[bool]` (None = producer predates the field; True/False = explicit).

- [ ] **Step 1: Write the failing test**

Create `services/orion-zwave/tests/test_freshness.py`:

```python
from __future__ import annotations

from orion.schemas.telemetry.home_cooling import CoolingObservedStateV1


def test_state_carries_explicit_stale_flag():
    assert CoolingObservedStateV1().stale is None
    assert CoolingObservedStateV1(stale=True).stale is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests/test_freshness.py -q`
Expected: FAIL — `extra_forbidden` validation error for `stale`.

- [ ] **Step 3: Implement**

In `orion/schemas/telemetry/home_cooling.py` replace `CoolingObservedStateV1` with:

```python
class CoolingObservedStateV1(BaseModel):
    """Observational only — not an actuator affordance.

    ``stale=True`` means no fresh reading from the plug within the producer's
    stale window: every cached reading is omitted from the sample.
    """

    model_config = ConfigDict(extra="forbid")

    switch_on: Optional[bool] = None
    stale: Optional[bool] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests/test_freshness.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/telemetry/home_cooling.py services/orion-zwave/tests/test_freshness.py
git commit -m "feat(schema): explicit stale flag on home cooling samples"
```

---

### Task 2: Client freshness tracking

**Files:**
- Modify: `services/orion-zwave/app/zwave_client.py`
- Test: `services/orion-zwave/tests/test_freshness.py`

**Interfaces:**
- Produces on `ZWaveJSClient`:
  - `__init__(ws_url: str, node_id: int, now_fn: Callable[[], datetime] = _utcnow)`
  - `connected -> bool` (property)
  - `last_fresh_at(node_id: Optional[int] = None) -> Optional[datetime]`
  - `consecutive_poll_failures: int`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_freshness.py`)

```python
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.zwave_client import METER_CC, METER_W_PROPERTY_KEY, ZWaveJSClient

T0 = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
W_KEY = f"{METER_CC}-0-value-{METER_W_PROPERTY_KEY}"


def _client() -> ZWaveJSClient:
    return ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)


@pytest.mark.asyncio
async def test_successful_poll_marks_fresh_and_resets_failures():
    client = _client()
    client.consecutive_poll_failures = 4
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    assert await client.refresh_meter_watts(2) == 885.7
    assert client.last_fresh_at(2) == T0
    assert client.consecutive_poll_failures == 0


@pytest.mark.asyncio
async def test_failed_poll_is_counted_and_never_marks_fresh(caplog):
    client = _client()
    client._values_by_node[2] = {W_KEY: {"commandClass": METER_CC, "property": "value",
                                         "propertyKey": METER_W_PROPERTY_KEY, "value": 850.0}}
    client._request = AsyncMock(side_effect=TimeoutError("no answer"))
    with caplog.at_level("WARNING"):
        assert await client.refresh_meter_watts(2) is None
    assert client.last_fresh_at(2) is None
    assert client.consecutive_poll_failures == 1
    assert "poll_value" in caplog.text
    assert client.get_values(2)[W_KEY]["value"] == 850.0  # cache untouched, but not fresh


@pytest.mark.asyncio
async def test_unsuccessful_result_is_a_failure():
    client = _client()
    client._request = AsyncMock(return_value={"success": False, "errorCode": "node_dead"})
    assert await client.refresh_meter_watts(2) is None
    assert client.last_fresh_at(2) is None
    assert client.consecutive_poll_failures == 1


def test_pushed_watts_event_marks_fresh():
    client = _client()
    client._handle_message({"type": "event", "event": {
        "source": "node", "event": "value updated", "nodeId": 2,
        "args": {"commandClass": METER_CC, "endpoint": 0, "property": "value",
                 "propertyKey": METER_W_PROPERTY_KEY, "newValue": 870.0}}})
    assert client.last_fresh_at(2) == T0


def test_pushed_non_watts_event_does_not_mark_fresh():
    client = _client()
    client._handle_message({"type": "event", "event": {
        "source": "node", "event": "value updated", "nodeId": 2,
        "args": {"commandClass": 37, "endpoint": 0, "property": "currentValue", "newValue": True}}})
    assert client.last_fresh_at(2) is None


def test_not_connected_without_socket():
    assert _client().connected is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests/test_freshness.py -q`
Expected: FAIL — `unexpected keyword argument 'now_fn'`.

- [ ] **Step 3: Implement**

In `services/orion-zwave/app/zwave_client.py`:

Replace the imports block at the top with:

```python
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websockets
```

Add after `normalize_value_entry`'s definition block (module level, before `_value_map_key`):

```python
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
```

Replace `ZWaveJSClient.__init__` with:

```python
    def __init__(self, ws_url: str, node_id: int, now_fn: Callable[[], datetime] = _utcnow) -> None:
        self.ws_url = ws_url
        self.node_id = node_id
        self._now = now_fn
        self._ws: Any = None
        self._message_id = 0
        self._inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._values_by_node: dict[int, dict[str, dict[str, Any]]] = {}
        self._controller_ready = False
        self._device_online: dict[int, bool] = {}
        self._product_by_node: dict[int, str] = {}
        self._last_fresh_at: dict[int, datetime] = {}
        self.consecutive_poll_failures = 0
        self._listener_task: Optional[asyncio.Task[None]] = None
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._pending_results: dict[str, asyncio.Future[dict[str, Any]]] = {}

    @property
    def connected(self) -> bool:
        """True while the websocket is open and its listener is still reading."""
        task = self._listener_task
        return self._ws is not None and task is not None and not task.done()

    def last_fresh_at(self, node_id: Optional[int] = None) -> Optional[datetime]:
        """When the plug last actually answered (poll) or pushed an Electric_W value."""
        target = self.node_id if node_id is None else node_id
        return self._last_fresh_at.get(target)

    def _mark_fresh(self, node_id: int) -> None:
        self._last_fresh_at[node_id] = self._now()

    def _poll_failed(self, node_id: int, reason: str) -> None:
        self.consecutive_poll_failures += 1
        logger.warning(
            "node.poll_value Electric_W failed node=%s consecutive_failures=%s reason=%s",
            node_id,
            self.consecutive_poll_failures,
            reason,
        )
```

Replace the body of `refresh_meter_watts` with:

```python
    async def refresh_meter_watts(self, node_id: Optional[int] = None) -> Optional[float]:
        """Ask the stick for a fresh Electric_W reading. Success updates the cache and marks
        the node fresh; any failure is counted and logged, and never marks it fresh."""
        target = self.node_id if node_id is None else node_id
        try:
            result = await self._request(
                "node.poll_value",
                nodeId=target,
                valueId={
                    "commandClass": METER_CC,
                    "endpoint": 0,
                    "property": "value",
                    "propertyKey": METER_W_PROPERTY_KEY,
                },
            )
        except Exception as exc:  # noqa: BLE001 -- includes a closed socket (assert in _request)
            self._poll_failed(target, f"{type(exc).__name__}: {exc}")
            return None
        if not result.get("success", True):
            self._poll_failed(target, str(result.get("errorCode") or "unsuccessful"))
            return None
        body = result.get("result") if isinstance(result.get("result"), dict) else {}
        watts = _numeric_value((body or {}).get("value"))
        if watts is None:
            self._poll_failed(target, "non_numeric_value")
            return None
        bucket = self._values_by_node.setdefault(target, {})
        key = f"{METER_CC}-0-value-{METER_W_PROPERTY_KEY}"
        bucket[key] = {
            "commandClass": METER_CC,
            "endpoint": 0,
            "property": "value",
            "propertyKey": METER_W_PROPERTY_KEY,
            "propertyKeyName": "Electric_W_Consumed",
            "value": watts,
            "nodeId": target,
        }
        self.consecutive_poll_failures = 0
        self._mark_fresh(target)
        return watts
```

In `_handle_message`, in the `value added / value updated` branch, replace the last three lines

```python
            bucket = self._values_by_node.setdefault(node_id, {})
            bucket[_value_map_key(normalized)] = normalized
            return
```

with:

```python
            bucket = self._values_by_node.setdefault(node_id, {})
            bucket[_value_map_key(normalized)] = normalized
            if (
                normalized.get("commandClass") == METER_CC
                and normalized.get("property") == "value"
                and _is_electric_watts(normalized)
                and _entry_numeric(normalized) is not None
            ):
                self._mark_fresh(node_id)
            return
```

- [ ] **Step 4: Run tests**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests -q`
Expected: all pass (12 baseline + 7 new).

- [ ] **Step 5: Commit**

```bash
git add services/orion-zwave/app/zwave_client.py services/orion-zwave/tests/test_freshness.py
git commit -m "fix(zwave): track real plug freshness; count and log failed polls"
```

---

### Task 3: Stale-aware sample builder

**Files:**
- Modify: `services/orion-zwave/app/main.py` (`build_cooling_sample`)
- Modify: `services/orion-zwave/tests/test_build_sample.py`
- Test: `services/orion-zwave/tests/test_freshness.py`

**Interfaces:**
- Produces: `build_cooling_sample(*, node_id, controller_ready, device_online, watts, volts, amps, switch_on, device_path, product, now, last_fresh_at: Optional[datetime], stale_after_sec: float, device_id=..., device_name=..., instance_id=...) -> HomeCoolingSampleV1`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_freshness.py`)

```python
from datetime import timedelta

from app.main import build_cooling_sample


def _sample(last_fresh_at, now=T0 + timedelta(seconds=30)):
    return build_cooling_sample(
        node_id=2, controller_ready=True, device_online=True,
        watts=850.0, volts=121.0, amps=7.0, switch_on=True,
        device_path="/dev/zwave", product="Shelly Wave Plug",
        now=now, last_fresh_at=last_fresh_at, stale_after_sec=120.0,
    )


def test_fresh_sample_keeps_readings_and_reports_age():
    s = _sample(T0)
    assert s.state.stale is False
    assert s.measurements.cooling_watts == 850.0
    assert s.state.switch_on is True
    assert s.provenance.sample_age_sec == 30.0


def test_stale_sample_omits_every_cached_reading():
    s = _sample(T0, now=T0 + timedelta(seconds=121))
    assert s.state.stale is True
    assert s.measurements.cooling_watts is None
    assert s.measurements.cooling_volts is None
    assert s.measurements.cooling_amps is None
    assert s.state.switch_on is None
    assert s.provenance.sample_age_sec == 121.0


def test_never_fresh_is_stale_with_unknown_age():
    s = _sample(None)
    assert s.state.stale is True
    assert s.measurements.cooling_watts is None
    assert s.provenance.sample_age_sec is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests/test_freshness.py -q`
Expected: FAIL — `unexpected keyword argument 'last_fresh_at'`.

- [ ] **Step 3: Implement**

Replace `build_cooling_sample` in `services/orion-zwave/app/main.py` with:

```python
def build_cooling_sample(
    *,
    node_id: int,
    controller_ready: bool,
    device_online: bool,
    watts: Optional[float],
    volts: Optional[float],
    amps: Optional[float],
    switch_on: Optional[bool],
    device_path: Optional[str],
    product: Optional[str],
    now: datetime,
    last_fresh_at: Optional[datetime],
    stale_after_sec: float,
    device_id: str = "shelly-wave-plug-ac",
    device_name: str = "portable_ac",
    instance_id: str = "athena",
) -> HomeCoolingSampleV1:
    age = max(0.0, (now - last_fresh_at).total_seconds()) if last_fresh_at is not None else None
    stale = age is None or age > stale_after_sec
    if stale:
        measurements = CoolingMeasurementsV1()
        state = CoolingObservedStateV1(stale=True)
    else:
        measurements = CoolingMeasurementsV1(
            cooling_watts=watts,
            cooling_volts=volts,
            cooling_amps=amps,
        )
        state = CoolingObservedStateV1(switch_on=switch_on, stale=False)

    return HomeCoolingSampleV1(
        ts=now,
        node=instance_id,
        controller=CoolingControllerV1(
            ready=controller_ready,
            device_path=device_path,
        ),
        device=CoolingDeviceV1(
            id=device_id,
            name=device_name,
            product=product,
            online=device_online,
        ),
        measurements=measurements,
        state=state,
        provenance=CoolingProvenanceV1(zwave_node_id=node_id, sample_age_sec=age),
    )
```

Replace `services/orion-zwave/tests/test_build_sample.py` body with:

```python
from datetime import datetime, timezone

from app.main import build_cooling_sample
from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1


def test_build_sample_omits_zero_fill_when_watts_missing():
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    sample = build_cooling_sample(
        node_id=2,
        controller_ready=True,
        device_online=True,
        watts=None,
        volts=None,
        amps=None,
        switch_on=None,
        device_path="/dev/zwave",
        product="Shelly Wave Plug",
        now=now,
        last_fresh_at=now,
        stale_after_sec=120.0,
    )
    assert isinstance(sample, HomeCoolingSampleV1)
    assert sample.measurements.cooling_watts is None
    assert sample.state.stale is False
    assert sample.role == "cabinet_cooling"
```

- [ ] **Step 4: Run tests**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/orion-zwave/app/main.py services/orion-zwave/tests/test_build_sample.py services/orion-zwave/tests/test_freshness.py
git commit -m "fix(zwave): stale samples carry no readings, only their age"
```

---

### Task 4: Reconnecting poll loop, setting, heartbeat details

**Files:**
- Modify: `services/orion-zwave/app/main.py` (`poll_cooling_loop`, `build_heartbeat_chassis`, new `poll_once`, `_COOLING_STATUS`)
- Modify: `services/orion-zwave/app/settings.py`, `services/orion-zwave/.env_example`, `services/orion-zwave/README.md`
- Test: `services/orion-zwave/tests/test_freshness.py`

**Interfaces:**
- Consumes: Task 2 client API, Task 3 builder.
- Produces: `async def poll_once(client: ZWaveJSClient, settings: Settings, now_fn: Callable[[], datetime] = _utcnow) -> HomeCoolingSampleV1`; `cooling_status() -> dict` (heartbeat details: `cooling_sensor` = `fresh|stale|unknown`, `cooling_sample_age_sec`, `zwave_connected`, `consecutive_poll_failures`); setting `COOLING_STALE_AFTER_SEC: float = 120.0`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_freshness.py`)

```python
import app.main as zwave_main
from app.settings import Settings


@pytest.mark.asyncio
async def test_poll_once_after_silence_publishes_stale_not_cache():
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)
    client._values_by_node[2] = {W_KEY: {"commandClass": METER_CC, "property": "value",
                                         "propertyKey": METER_W_PROPERTY_KEY, "value": 850.0}}
    client._last_fresh_at[2] = T0
    client._request = AsyncMock(side_effect=TimeoutError("plug gone"))
    settings = Settings(ZWAVE_NODE_ID=2, COOLING_STALE_AFTER_SEC=120.0)
    sample = await zwave_main.poll_once(client, settings, now_fn=lambda: T0 + timedelta(seconds=300))
    assert sample.state.stale is True
    assert sample.measurements.cooling_watts is None
    assert zwave_main.cooling_status()["cooling_sensor"] == "stale"
    assert zwave_main.cooling_status()["consecutive_poll_failures"] == 1


@pytest.mark.asyncio
async def test_poll_once_fresh_poll_publishes_reading():
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    settings = Settings(ZWAVE_NODE_ID=2, COOLING_STALE_AFTER_SEC=120.0)
    sample = await zwave_main.poll_once(client, settings, now_fn=lambda: T0 + timedelta(seconds=1))
    assert sample.state.stale is False
    assert sample.measurements.cooling_watts == 885.7
    assert zwave_main.cooling_status()["cooling_sensor"] == "fresh"


def test_heartbeat_carries_cooling_status():
    chassis = zwave_main.build_heartbeat_chassis(Settings())
    assert chassis._heartbeat_details is not None
    assert "cooling_sensor" in chassis._heartbeat_details()


def test_stale_window_setting_default():
    assert Settings().COOLING_STALE_AFTER_SEC == 120.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests/test_freshness.py -q`
Expected: FAIL — `module 'app.main' has no attribute 'poll_once'`.

- [ ] **Step 3: Implement settings + env + README**

In `services/orion-zwave/app/settings.py` add after `COOLING_POLL_INTERVAL_SEC`:

```python
    # No fresh reading from the plug for this long => samples carry no readings, stale=true.
    COOLING_STALE_AFTER_SEC: float = Field(default=120.0)
```

In `services/orion-zwave/.env_example` add after `COOLING_POLL_INTERVAL_SEC=5`:

```text
COOLING_STALE_AFTER_SEC=120
```

In `services/orion-zwave/README.md` env table, add after the `COOLING_POLL_INTERVAL_SEC` row:

```markdown
| `COOLING_STALE_AFTER_SEC` | `120` | No fresh plug reading (successful poll or pushed Electric_W) for this long ⇒ sample omits all readings and sets `state.stale=true`; heartbeat details report `cooling_sensor=stale` |
```

- [ ] **Step 4: Implement `poll_once`, status, loop, heartbeat**

In `services/orion-zwave/app/main.py`:

Change `from typing import Optional` to `from typing import Any, Callable, Optional`.

Add after `setup_logging`:

```python
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_COOLING_STATUS: dict[str, Any] = {
    "cooling_sensor": "unknown",
    "cooling_sample_age_sec": None,
    "zwave_connected": False,
    "consecutive_poll_failures": 0,
}


def cooling_status() -> dict[str, Any]:
    """Heartbeat details: is the AC reading real right now?"""
    return dict(_COOLING_STATUS)


async def poll_once(
    client: ZWaveJSClient,
    settings: Settings,
    now_fn: Callable[[], datetime] = _utcnow,
) -> HomeCoolingSampleV1:
    node_id = settings.ZWAVE_NODE_ID
    await client.refresh_meter_watts(node_id)
    values = client.get_values(node_id)
    sample = build_cooling_sample(
        node_id=node_id,
        controller_ready=client.controller_ready,
        device_online=client.device_online(node_id),
        watts=extract_meter_watts(values),
        volts=extract_meter_volts(values),
        amps=extract_meter_amps(values),
        switch_on=extract_switch_on(values),
        device_path="/dev/zwave",
        product=client.product_name(node_id),
        now=now_fn(),
        last_fresh_at=client.last_fresh_at(node_id),
        stale_after_sec=settings.COOLING_STALE_AFTER_SEC,
        device_id=settings.ZWAVE_DEVICE_ID,
        device_name=settings.ZWAVE_DEVICE_NAME,
        instance_id=settings.INSTANCE_ID,
    )
    _COOLING_STATUS.update(
        cooling_sensor="stale" if sample.state.stale else "fresh",
        cooling_sample_age_sec=sample.provenance.sample_age_sec,
        zwave_connected=client.connected,
        consecutive_poll_failures=client.consecutive_poll_failures,
    )
    if sample.state.stale:
        logger.warning(
            "cooling_sample_stale age_sec=%s consecutive_poll_failures=%s connected=%s",
            sample.provenance.sample_age_sec,
            client.consecutive_poll_failures,
            client.connected,
        )
    return sample
```

Replace everything in `poll_cooling_loop` from `client = ZWaveJSClient(...)` to the end of the function with:

```python
    client = ZWaveJSClient(settings.ZWAVE_JS_WS_URL, settings.ZWAVE_NODE_ID)
    try:
        while True:
            if not client.connected:
                # A dead socket must not leave us publishing cache forever: reconnect, and
                # keep publishing (stale) samples while it is down so silence is visible.
                try:
                    await client.close()
                    await client.connect()
                    logger.info("zwave_js_connected ws=%s", settings.ZWAVE_JS_WS_URL)
                except Exception:
                    logger.exception("Z-Wave connect attempt failed ws=%s", settings.ZWAVE_JS_WS_URL)
            try:
                sample = await poll_once(client, settings)
                await _publish_sample(bus, settings, sample)
            except Exception:
                logger.exception("Cooling poll cycle failed; will retry")

            await asyncio.sleep(settings.COOLING_POLL_INTERVAL_SEC)
    finally:
        await client.close()
```

Replace `build_heartbeat_chassis` with:

```python
def build_heartbeat_chassis(settings: Optional[Settings] = None) -> HeartbeatOnly:
    s = settings if settings is not None else get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.SERVICE_NAME,
            service_version=s.SERVICE_VERSION,
            node_name=s.INSTANCE_ID,
            bus_url=s.ORION_BUS_URL,
            bus_enabled=s.ORION_BUS_ENABLED,
            heartbeat_interval_sec=s.HEARTBEAT_INTERVAL_SEC,
            health_channel=s.ORION_HEALTH_CHANNEL,
        ),
        heartbeat_details=cooling_status,
    )
```

- [ ] **Step 5: Run all zwave tests**

Run: `cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6: Sync env and commit**

```bash
cd /mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch
python3 scripts/sync_local_env_from_example.py orion-zwave
git check-ignore services/orion-zwave/.env
git add services/orion-zwave/app/main.py services/orion-zwave/app/settings.py services/orion-zwave/.env_example services/orion-zwave/README.md services/orion-zwave/tests/test_freshness.py
git commit -m "fix(zwave): reconnect dead socket, publish stale not cache, report freshness in heartbeat"
```

---

### Task 5: Persist `stale` and `sample_age_sec`

**Files:**
- Modify: `services/orion-sql-writer/app/models/home_cooling_sample.py`
- Modify: `services/orion-sql-writer/app/worker.py` (`_normalize_home_cooling_sample_payload`)
- Modify: `services/orion-sql-writer/app/main.py` (boot DDL block)
- Test: `services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py`

**Interfaces:**
- Produces: columns `home_cooling_sample.stale BOOLEAN NULL`, `home_cooling_sample.sample_age_sec DOUBLE PRECISION NULL` (consumed by Plan 5's hardware-watch).

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
def test_stale_and_age_columns_exist_and_are_mapped() -> None:
    cols = _columns()
    assert "stale" in cols and "sample_age_sec" in cols
    now = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    payload = HomeCoolingSampleV1(
        ts=now,
        controller={"ready": True},
        device={"id": "node-2", "name": "Cabinet AC", "online": True},
        measurements={},
        state={"stale": True},
        provenance={"zwave_node_id": 2, "sample_age_sec": 301.5},
    ).model_dump(mode="json")
    mapped = _normalize_home_cooling_sample_payload(payload)
    assert mapped["stale"] is True
    assert mapped["sample_age_sec"] == 301.5
    assert "cooling_watts" not in mapped or mapped["cooling_watts"] is None


def test_boot_ddl_adds_the_new_columns() -> None:
    src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    assert "ALTER TABLE IF EXISTS home_cooling_sample ADD COLUMN IF NOT EXISTS stale BOOLEAN" in src
    assert (
        "ALTER TABLE IF EXISTS home_cooling_sample ADD COLUMN IF NOT EXISTS sample_age_sec DOUBLE PRECISION"
        in src
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/orion-sql-writer && PYTHONPATH=../..:. $PY -m pytest tests/test_home_cooling_sample_sql_shape.py -q`
Expected: FAIL — `assert 'stale' in cols`.

- [ ] **Step 3: Implement**

`services/orion-sql-writer/app/models/home_cooling_sample.py`: add after `switch_on`:

```python
    stale = Column(Boolean, nullable=True)
    sample_age_sec = Column(Float, nullable=True)
```

`services/orion-sql-writer/app/worker.py`, in `_normalize_home_cooling_sample_payload`, replace

```python
    state = out.pop("state", None)
    if isinstance(state, dict) and "switch_on" in state:
        out["switch_on"] = state.get("switch_on")

    provenance = out.pop("provenance", None)
    if isinstance(provenance, dict) and provenance.get("zwave_node_id") is not None:
        out["zwave_node_id"] = int(provenance["zwave_node_id"])
```

with:

```python
    state = out.pop("state", None)
    if isinstance(state, dict) and "switch_on" in state:
        out["switch_on"] = state.get("switch_on")
    if isinstance(state, dict) and state.get("stale") is not None:
        out["stale"] = bool(state["stale"])

    provenance = out.pop("provenance", None)
    if isinstance(provenance, dict) and provenance.get("zwave_node_id") is not None:
        out["zwave_node_id"] = int(provenance["zwave_node_id"])
    if isinstance(provenance, dict) and isinstance(provenance.get("sample_age_sec"), (int, float)):
        out["sample_age_sec"] = float(provenance["sample_age_sec"])
```

`services/orion-sql-writer/app/main.py`: inside the same `with engine.begin() as conn:` boot-DDL block, directly after the `orion_biometrics_summary_node_ts_idx` statement, add:

```python
            # Same hazard as `measurements` above: the mapper declares these columns, so they
            # enter every home_cooling_sample INSERT and must exist before the writer serves.
            conn.exec_driver_sql(
                "ALTER TABLE IF EXISTS home_cooling_sample ADD COLUMN IF NOT EXISTS stale BOOLEAN;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE IF EXISTS home_cooling_sample ADD COLUMN IF NOT EXISTS sample_age_sec DOUBLE PRECISION;"
            )
```

- [ ] **Step 4: Run tests**

Run: `cd services/orion-sql-writer && PYTHONPATH=../..:. $PY -m pytest tests/test_home_cooling_sample_sql_shape.py tests/test_biometrics_summary_sql_shape.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer/app/models/home_cooling_sample.py services/orion-sql-writer/app/worker.py services/orion-sql-writer/app/main.py services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py
git commit -m "feat(sql-writer): persist cooling stale flag and sample age"
```

---

### Task 6: Hub shows STALE in red

**Files:**
- Modify: `services/orion-hub/scripts/cabinet_cooling_routes.py` (`_load_latest`, new `_payload_freshness`)
- Modify: `services/orion-hub/static/js/cabinet-sensors.js` (`renderCoolingLatest`)
- Test: `services/orion-hub/tests/test_cabinet_cooling_routes.py`, `services/orion-hub/tests/test_cabinet_sensors_panel.py`

**Interfaces:**
- Produces: `/api/cabinet/cooling/latest` adds `sensor_stale: bool` always when a row exists, plus `sample_age_sec: float` and `last_fresh_at: ISO str` when the producer reported an age. `ok` is false when `sensor_stale`. No-row shape unchanged. Reads from `payload_json` only (no dependency on Task 5's columns existing yet).

- [ ] **Step 1: Write the failing tests**

Append to `services/orion-hub/tests/test_cabinet_cooling_routes.py`:

```python
def test_latest_sensor_stale_row_is_not_ok_and_reports_last_fresh(client, monkeypatch):
    async def latest(*, node: str):
        return _row(
            cooling_watts=None,
            cooling_volts=None,
            switch_on=None,
            payload_json={
                "device": {"product": "Shelly Wave Plug", "online": True},
                "state": {"stale": True},
                "provenance": {"zwave_node_id": 2, "sample_age_sec": 600.0},
            },
        )

    monkeypatch.setattr(cabinet_cooling_routes, "_latest_query", latest)
    body = client.get("/api/cabinet/cooling/latest").json()
    assert body["ok"] is False
    assert body["sensor_stale"] is True
    assert body["sample_age_sec"] == 600.0
    assert body["last_fresh_at"] == "2026-09-25T14:50:00Z"
    assert "cooling_watts" not in body["sample"]


def test_latest_fresh_row_reports_not_sensor_stale(client, monkeypatch):
    async def latest(*, node: str):
        return _row(payload_json={"state": {"stale": False}, "provenance": {"zwave_node_id": 2, "sample_age_sec": 3.0}})

    monkeypatch.setattr(cabinet_cooling_routes, "_latest_query", latest)
    body = client.get("/api/cabinet/cooling/latest").json()
    assert body["ok"] is True
    assert body["sensor_stale"] is False
    assert body["sample_age_sec"] == 3.0
```

Append to `services/orion-hub/tests/test_cabinet_sensors_panel.py`:

```python
def test_cabinet_sensors_js_renders_sensor_stale_in_red() -> None:
    assert "payload.sensor_stale" in CABINET_SENSORS_JS
    assert '"AC reading STALE since "' in CABINET_SENSORS_JS
    assert "text-red-400" in CABINET_SENSORS_JS
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/orion-hub && $PY -m pytest tests/test_cabinet_cooling_routes.py tests/test_cabinet_sensors_panel.py -q`
Expected: FAIL — `KeyError: 'sensor_stale'` and the JS assertion.

- [ ] **Step 3: Implement the route**

In `services/orion-hub/scripts/cabinet_cooling_routes.py` replace `_load_latest` with:

```python
def _payload_freshness(row: Mapping[str, Any]) -> tuple[bool, Optional[float]]:
    """The producer's own verdict: was this reading real, and how old was it?"""
    payload_json = row.get("payload_json")
    if not isinstance(payload_json, dict):
        return False, None
    state = payload_json.get("state") if isinstance(payload_json.get("state"), dict) else {}
    provenance = payload_json.get("provenance") if isinstance(payload_json.get("provenance"), dict) else {}
    age = provenance.get("sample_age_sec")
    return state.get("stale") is True, float(age) if isinstance(age, (int, float)) else None


def _load_latest(row: Optional[Mapping[str, Any]], *, stale_after_sec: float, now: datetime) -> dict[str, Any]:
    if row is None:
        return {"ok": False, "age_sec": None, "sample": None}

    received_at = _parse_db_timestamp(row["ts"])
    age_sec = (now.astimezone(timezone.utc) - received_at).total_seconds()
    sensor_stale, sample_age_sec = _payload_freshness(row)
    out: dict[str, Any] = {
        "ok": age_sec <= stale_after_sec and not sensor_stale,
        "age_sec": age_sec,
        "sample": row_to_sample(row),
        "sensor_stale": sensor_stale,
    }
    if sample_age_sec is not None:
        out["sample_age_sec"] = sample_age_sec
        out["last_fresh_at"] = _iso_utc(received_at - timedelta(seconds=sample_age_sec))
    return out
```

- [ ] **Step 4: Implement the JS**

In `services/orion-hub/static/js/cabinet-sensors.js`, in `renderCoolingLatest`, directly after `if (!sample) return;` insert:

```js
    if (payload.sensor_stale) {
      var since = payload.last_fresh_at
        ? new Date(payload.last_fresh_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : "unknown";
      if (els.coolingWatts) els.coolingWatts.textContent = "STALE";
      if (els.coolingVolts) els.coolingVolts.textContent = "absent";
      if (els.coolingSwitch) els.coolingSwitch.textContent = "absent";
      if (els.coolingAge) els.coolingAge.textContent = age(payload.age_sec);
      if (els.coolingLiveStatus) {
        els.coolingLiveStatus.textContent = "AC reading STALE since " + since;
        els.coolingLiveStatus.className = "mt-1 font-mono text-sm text-red-400";
      }
      return;
    }
```

- [ ] **Step 5: Run tests**

Run: `cd services/orion-hub && $PY -m pytest tests/test_cabinet_cooling_routes.py tests/test_cabinet_sensors_panel.py tests/test_biometrics_view_ui.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/cabinet_cooling_routes.py services/orion-hub/static/js/cabinet-sensors.js services/orion-hub/tests/test_cabinet_cooling_routes.py services/orion-hub/tests/test_cabinet_sensors_panel.py
git commit -m "feat(hub): cabinet panel shows AC reading STALE in red"
```

---

### Task 7: Spec corrections, gates, review, deploy, live proof, PR

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md`

- [ ] **Step 1: Correct the spec**

In the AC gate section, replace the sentence `One anomaly: the first 1 h 38 m after pairing read **exactly 33.1 W for 1,171 samples**.` with:

```markdown
One anomaly: the first 1 h 38 m after pairing read **exactly 33.1 W for 1,171 samples** — the
since-fixed bug that read the kWh energy counter (propertyKey 65537) as watts
(`zwave_client.py` comment on `METER_W_PROPERTY_KEY`). It is a real example of a lying reading
(the frozen and low-power rules both catch it), not evidence that fan-only draws ~33 W.
```

In theory anchor (3), replace `fan-only draws ~33 W; off draws ~0.` with `fan-only draw is unmeasured; off draws ~0.`

In Part 5, replace `Poll failures: warning log + counter; \`SystemHealthV1\` reports \`degraded\` while stale.` with:

```markdown
- Poll failures: warning log + counter. Heartbeat `details` carry `cooling_sensor`
  (`fresh|stale|unknown`), `cooling_sample_age_sec`, `zwave_connected`,
  `consecutive_poll_failures` (the chassis status field stays `ok` = process alive).
- A dead websocket is reconnected by the poll loop; stale samples keep publishing meanwhile, so
  an outage is visible as `stale=true` rows rather than silence.
- Live check 2026-09-28: `node.poll_value` Electric_W succeeded 3/3 in 0.04–0.44 s (885.7 W), so
  "fresh = successful poll" will not false-alarm.
```

- [ ] **Step 2: Run the gates**

```bash
cd /mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch
git diff --check origin/main...HEAD
(cd services/orion-zwave && PYTHONPATH=../..:. $PY -m pytest tests -q)
(cd services/orion-sql-writer && PYTHONPATH=../..:. $PY -m pytest tests -q)
(cd services/orion-hub && $PY -m pytest tests/test_cabinet_cooling_routes.py tests/test_cabinet_sensors_panel.py tests/test_biometrics_view_ui.py -q)
python3 scripts/check_env_template_parity.py
python3 scripts/check_schema_registry.py
python3 scripts/check_bus_channels.py
```

Expected: all pass. (If a checker script does not exist, record that in the PR.)

- [ ] **Step 3: Code review in a subagent**

Dispatch the code-review skill (`/home/athena/.codex/skills/.system/review-agent/SKILL.md`) against `origin/main...HEAD`. Fix every material finding, re-run Step 2, commit fixes.

- [ ] **Step 4: Commit spec + push**

```bash
git add docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md
git commit -m "docs: correct AC gate evidence and Part 5 heartbeat wording"
git push -u origin fix/ac-sensing-stale
```

- [ ] **Step 5: Deploy (sql-writer first so the columns exist before new payloads matter)**

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-zwave up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

- [ ] **Step 6: Live proof**

```bash
URI=$(rg -N "^POSTGRES_URI=" /mnt/scripts/Orion-Sapienform/services/orion-hub/.env | cut -d= -f2-)
psql "$URI" -c "SELECT ts, cooling_watts, stale, sample_age_sec FROM home_cooling_sample ORDER BY ts DESC LIMIT 5;"
curl -fsS http://localhost:8080/api/cabinet/cooling/latest
```

Expected: newest rows have `stale=false`, `sample_age_sec` < 10, real watts; `/latest` shows `"sensor_stale": false` and a `sample_age_sec`. (Pre-change baseline 2026-09-28: `/latest` returned `{"ok":true,"age_sec":3.29,"sample":{..."cooling_watts":885.7...}}` with no freshness fields.)

Stale-path live proof needs a real outage (stopping `athena-zwave-js-ui` for ~3 minutes). That is an operator action: ask Juniper before doing it; otherwise mark the stale path live-`UNVERIFIED` (unit-tested only) in the PR.

- [ ] **Step 7: PR**

```bash
gh pr create --title "fix(zwave): AC reading can no longer fake a healthy cooler" --body-file /tmp/ac-sensing-pr.md
```

Write `/tmp/ac-sensing-pr.md` in the AGENTS.md §18 template shape, including: root cause (`refresh_meter_watts` swallowed failures at debug; cache republished with `ts=now`; dead listener never reconnected), the live poll check, test counts, review findings fixed, restart commands above, and the live-proof output. Then check CI (`gh pr checks`) and merge conflicts; fix until green.

- [ ] **Step 8: After merge, sync the shared checkout's env**

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && python3 scripts/sync_local_env_from_example.py orion-zwave
```
