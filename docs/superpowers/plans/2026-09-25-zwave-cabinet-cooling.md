# Z-Wave Cabinet Cooling Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish read-only Shelly Wave portable-AC watts from Athena’s SONOFF stick onto the Orion bus and show them inside Hub Biometrics → Cabinet beside Nano sensors.

**Architecture:** Host Z-Wave JS UI owns the USB stick and pairing. Thin `orion-zwave` is a websocket client that publishes `home.cooling.sample.v1` on `orion:home:cooling:sample`. Hub Cabinet polls `/api/cabinet/cooling/*`. sql-writer persists samples for history charts. Never mount the stick into Orion containers; never fold watts into chassis `peak_pressure`.

**Tech Stack:** Python 3.12, pydantic v2, Orion bus (`OrionBusAsync` + `BaseEnvelope`), FastAPI Hub routes, Docker Compose, `zwavejs/zwave-js-ui`, Postgres via sql-writer, vanilla Hub JS.

**Spec:** `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md`

## Global Constraints

- Read-only: no on/off control channel, no Hub toggle buttons, no power-intent.
- Pairing only in Z-Wave JS UI (no Hub Z-Wave management tab).
- Cooling UI lives **inside** `#cabinet` (Biometrics modal), not a sibling subtab.
- USB: only Z-Wave JS compose may map `/dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_a6e1256f698cf011906026b9d9065118-if00-port0`.
- Absent-is-not-zero: omit unknown meter fields; UI shows `absent` / stale, never fake `0.0` watts.
- Do not add cooling watts to `chassis_watts`, `pdu_watts`, `peak_pressure`, strain, or cluster role weights.
- `ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0` — never `bus-core` / bare `redis`.
- Register channel + schema in the same changeset (no unregistered power-guard-style gap).
- Work only in worktree `/mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling` on branch `docs/zwave-cabinet-cooling` (rename/cut to `feat/zwave-cabinet-cooling` when implementation starts).
- After any `.env_example` change: `python scripts/sync_local_env_from_example.py`.

---

## File structure (locked)

| Path | Responsibility |
|---|---|
| `/home/athena/zwave-js/docker-compose.yml` | Host Z-Wave JS UI + stick device + store volume (ops, not Orion cognition) |
| `/home/athena/zwave-js/.env` | `SESSION_SECRET` only; never commit |
| `orion/schemas/telemetry/home_cooling.py` | `HomeCoolingSampleV1` contract |
| `orion/schemas/registry.py` | Dual registration (`_REGISTRY` + `SCHEMA_REGISTRY`) |
| `orion/bus/channels.yaml` | `orion:home:cooling:sample` |
| `tests/test_home_cooling_bus_catalog.py` | Channel + registry gate |
| `services/orion-zwave/` | Poller service (WS client → bus + heartbeat) |
| `services/orion-sql-writer/app/models/home_cooling_sample.py` | Postgres table |
| `services/orion-sql-writer/app/settings.py` | `DEFAULT_ROUTE_MAP` entry |
| `services/orion-sql-writer/app/worker.py` | `MODEL_MAP` entry |
| `services/orion-sql-writer/.env_example` | Subscribe channel |
| `services/orion-hub/scripts/cabinet_cooling_routes.py` | `/api/cabinet/cooling/latest` + `/history` |
| `services/orion-hub/scripts/api_routes.py` | `include_router` |
| `services/orion-hub/templates/index.html` | Cooling block inside `#cabinet` |
| `services/orion-hub/static/js/cabinet-sensors.js` | Poll + render Cooling strip |

---

### Task 1: Host Z-Wave JS UI on Athena (ops)

**Files:**
- Create: `/home/athena/zwave-js/docker-compose.yml`
- Create: `/home/athena/zwave-js/.env` (local only)
- Modify: `services/orion-zwave/README.md` (add “Host Z-Wave JS” section once service scaffold exists — if README not yet created, write a short `/home/athena/zwave-js/README.md` now and move the ops section into the service README in Task 3)

**Interfaces:**
- Consumes: SONOFF by-id path on Athena
- Produces: UI on `http://athena:8091`, zwave-js-server WS on `127.0.0.1:3000`

- [ ] **Step 1: Create host directory and compose**

```bash
mkdir -p /home/athena/zwave-js
```

Write `/home/athena/zwave-js/docker-compose.yml`:

```yaml
services:
  zwave-js-ui:
    image: zwavejs/zwave-js-ui:9.29.0
    container_name: athena-zwave-js-ui
    restart: unless-stopped
    devices:
      - /dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_a6e1256f698cf011906026b9d9065118-if00-port0:/dev/zwave
    volumes:
      - zwave-store:/usr/src/app/store
    ports:
      - "8091:8091"
      - "127.0.0.1:3000:3000"
    environment:
      SESSION_SECRET: ${SESSION_SECRET}
    # Stick ownership: this container only.
```

Write `/home/athena/zwave-js/.env`:

```bash
SESSION_SECRET=$(openssl rand -hex 32)
# paste into .env as SESSION_SECRET=...
```

- [ ] **Step 2: Bring up and verify stick**

```bash
cd /home/athena/zwave-js
docker compose up -d
docker logs --tail 50 athena-zwave-js-ui
ls -l /dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_*
```

Expected: container healthy; logs show driver start; no other container holds `ttyUSB0`.

- [ ] **Step 3: Operator pair (manual — Juniper)**

1. Open `http://192.168.1.43:8091` (or Athena Tailscale IP).
2. Settings → Z-Wave → serial port `/dev/zwave` → start driver.
3. Settings → Home Assistant → enable WS Server on port 3000.
4. Control panel → Inclusion → include Shelly Wave plug (button on plug per Shelly docs).
5. Note the Z-Wave **node id** for the plug (needed as `ZWAVE_NODE_ID` later).

- [ ] **Step 4: Commit ops note only if under repo**

Do **not** commit `/home/athena/zwave-js/.env` or store volume. If documenting in-repo, only README paths under `services/orion-zwave/` (Task 3). No git commit required for host ops alone.

---

### Task 2: Schema + channel contract (TDD)

**Files:**
- Create: `orion/schemas/telemetry/home_cooling.py`
- Create: `tests/test_home_cooling_bus_catalog.py`
- Modify: `orion/schemas/registry.py`
- Modify: `orion/bus/channels.yaml`

**Interfaces:**
- Produces: `HomeCoolingSampleV1`, kind `home.cooling.sample.v1`, channel `orion:home:cooling:sample`

- [ ] **Step 1: Write the failing catalog test**

Create `tests/test_home_cooling_bus_catalog.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1

CHANNEL = "orion:home:cooling:sample"
ROOT = Path(__file__).resolve().parents[1]


def _channels() -> dict:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    return {c["name"]: c for c in raw["channels"]}


def test_home_cooling_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "HomeCoolingSampleV1"
    assert entry["message_kind"] == "home.cooling.sample.v1"
    assert "orion-zwave" in entry["producer_services"]
    assert "orion-sql-writer" in entry["consumer_services"]


def test_home_cooling_schema_registry_aligns_with_resolve() -> None:
    reg = SCHEMA_REGISTRY["HomeCoolingSampleV1"]
    assert reg.kind == "home.cooling.sample.v1"
    assert resolve("HomeCoolingSampleV1") is HomeCoolingSampleV1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling
pytest tests/test_home_cooling_bus_catalog.py -v
```

Expected: FAIL (import or missing channel).

- [ ] **Step 3: Implement schema**

Create `orion/schemas/telemetry/home_cooling.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoolingControllerV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    driver: str = "zwave-js"
    device_path: Optional[str] = None


class CoolingDeviceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    product: Optional[str] = None
    online: bool


class CoolingMeasurementsV1(BaseModel):
    """Only include fields that were actually read. Omit unknowns — never zero-fill."""

    model_config = ConfigDict(extra="forbid")

    cooling_watts: Optional[float] = Field(default=None, ge=0.0)
    cooling_volts: Optional[float] = Field(default=None, ge=0.0)
    cooling_amps: Optional[float] = Field(default=None, ge=0.0)
    cooling_power_factor: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CoolingObservedStateV1(BaseModel):
    """Observational only — not an actuator affordance."""

    model_config = ConfigDict(extra="forbid")

    switch_on: Optional[bool] = None


class CoolingProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zwave_node_id: int = Field(ge=1)
    source: str = "zwave-js"
    sample_age_sec: Optional[float] = Field(default=None, ge=0.0)


class HomeCoolingSampleV1(BaseModel):
    """Read-only cabinet cooling sample (portable AC on Shelly Wave)."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(default="home.cooling.sample.v1", alias="schema")
    ts: datetime
    node: str = "athena"
    role: str = "cabinet_cooling"
    controller: CoolingControllerV1
    device: CoolingDeviceV1
    measurements: CoolingMeasurementsV1
    state: CoolingObservedStateV1 = Field(default_factory=CoolingObservedStateV1)
    provenance: CoolingProvenanceV1

    @field_validator("ts")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
```

- [ ] **Step 4: Register schema + channel**

In `orion/schemas/registry.py`:
1. `from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1`
2. Add `"HomeCoolingSampleV1": HomeCoolingSampleV1` to the string `_REGISTRY` dict (near other telemetry).
3. Add to `SCHEMA_REGISTRY`:

```python
    "HomeCoolingSampleV1": SchemaRegistration(
        model=HomeCoolingSampleV1,
        kind="home.cooling.sample.v1",
    ),
```

In `orion/bus/channels.yaml` (near cabinet ambient spike):

```yaml
  - name: "orion:home:cooling:sample"
    kind: "telemetry"
    schema_id: "HomeCoolingSampleV1"
    message_kind: "home.cooling.sample.v1"
    producer_services: ["orion-zwave"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-25"
    description: "Read-only portable AC / cabinet cooling watts from Athena Z-Wave Shelly Wave plug."
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_home_cooling_bus_catalog.py -v
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/telemetry/home_cooling.py orion/schemas/registry.py orion/bus/channels.yaml tests/test_home_cooling_bus_catalog.py
git commit -m "feat(contract): home.cooling.sample.v1 channel + schema"
```

---

### Task 3: `orion-zwave` service (read-only poller)

**Files:**
- Create: `services/orion-zwave/app/__init__.py`
- Create: `services/orion-zwave/app/settings.py`
- Create: `services/orion-zwave/app/zwave_client.py`
- Create: `services/orion-zwave/app/main.py`
- Create: `services/orion-zwave/Dockerfile`
- Create: `services/orion-zwave/docker-compose.yml`
- Create: `services/orion-zwave/requirements.txt`
- Create: `services/orion-zwave/.env_example`
- Create: `services/orion-zwave/README.md`
- Create: `services/orion-zwave/tests/test_heartbeat_chassis.py`
- Create: `services/orion-zwave/tests/test_build_sample.py`
- Create: `services/orion-zwave/tests/test_zwave_client_parse.py`

**Interfaces:**
- Consumes: `ZWAVE_JS_WS_URL` (zwave-js-server on host), `ZWAVE_NODE_ID`
- Produces: bus publishes of `HomeCoolingSampleV1` on `COOLING_SAMPLE_CHANNEL`; `HeartbeatOnly` on `orion:system:health`

- [ ] **Step 1: Write failing unit tests for sample builder + meter parse**

`services/orion-zwave/tests/test_zwave_client_parse.py`:

```python
from app.zwave_client import extract_meter_watts, extract_switch_on


def test_extract_meter_watts_from_value_id():
    values = {
        "50-0-value-65537": {"commandClass": 50, "property": "value", "propertyKey": 65537, "value": 412.5},
    }
    assert extract_meter_watts(values) == 412.5


def test_extract_meter_watts_absent():
    assert extract_meter_watts({}) is None


def test_extract_switch_on():
    values = {
        "37-0-currentValue": {"commandClass": 37, "property": "currentValue", "value": True},
    }
    assert extract_switch_on(values) is True
```

`services/orion-zwave/tests/test_build_sample.py`:

```python
from datetime import datetime, timezone

from app.main import build_cooling_sample
from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1


def test_build_sample_omits_zero_fill_when_watts_missing():
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
        now=datetime(2026, 9, 25, tzinfo=timezone.utc),
    )
    assert isinstance(sample, HomeCoolingSampleV1)
    assert sample.measurements.cooling_watts is None
    assert sample.role == "cabinet_cooling"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest services/orion-zwave/tests -q
```

- [ ] **Step 3: Implement settings, client, main**

`settings.py` (mirror power-guard):

```python
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-zwave")
    INSTANCE_ID: str = Field(default="athena")
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_ZWAVE_ENABLED: bool = Field(default=False)
    ZWAVE_JS_WS_URL: str = Field(default="ws://host.docker.internal:3000")
    ZWAVE_NODE_ID: int = Field(default=2)
    ZWAVE_DEVICE_ID: str = Field(default="shelly-wave-plug-ac")
    ZWAVE_DEVICE_NAME: str = Field(default="portable_ac")
    COOLING_SAMPLE_CHANNEL: str = Field(default="orion:home:cooling:sample")
    COOLING_POLL_INTERVAL_SEC: float = Field(default=5.0)
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`zwave_client.py` — minimal zwave-js-server client:

- Connect to `ZWAVE_JS_WS_URL` with `websockets`.
- On connect: read version hello; send `{"messageId":"1","command":"set_api_schema","schemaVersion":33}` (use schema version from hello if present).
- Send `start_listening`; keep last `nodes` / value map for `ZWAVE_NODE_ID`.
- Helpers `extract_meter_watts(values)` / `extract_switch_on(values)`:
  - Meter CC **50**: prefer Electric W consumed (propertyKey often `65537` / label containing `W`); return `None` if no numeric value.
  - Binary switch CC **37**: `currentValue` bool → `switch_on`.
- Never invent `0.0` when the value map lacks a meter entry.

`main.py`:

- `build_cooling_sample(...)` → `HomeCoolingSampleV1` (used by tests).
- If `ORION_ZWAVE_ENABLED=false`: sleep loop + heartbeat only (no WS).
- Else: connect client, poll every `COOLING_POLL_INTERVAL_SEC`, publish:

```python
envelope = BaseEnvelope(
    kind="home.cooling.sample.v1",
    source=f"service=orion-zwave;instance={settings.INSTANCE_ID}",
    payload=sample.model_dump(mode="json", by_alias=True),
)
await bus.publish(settings.COOLING_SAMPLE_CHANNEL, envelope)
```

- Wire `HeartbeatOnly` exactly like `services/orion-power-guard/app/main.py` (`build_heartbeat_chassis`).

**Dockerfile / compose / requirements:**

- Dockerfile: copy `services/orion-zwave` + `orion` package (same as power-guard).
- `requirements.txt`: `pydantic==2.9.2`, `pydantic-settings==2.7.1`, `redis==5.0.8`, `orjson==3.10.7`, `pyyaml==6.0.3`, `websockets==13.1`
- `docker-compose.yml`: **no** `devices:` block; `extra_hosts: host.docker.internal:host-gateway`; `env_file: [.env]`; network `app-net` like power-guard.

`.env_example`:

```bash
SERVICE_NAME=orion-zwave
INSTANCE_ID=athena
ORION_BUS_URL=redis://100.92.216.81:6379/0
ORION_BUS_ENABLED=true
ORION_ZWAVE_ENABLED=false
ZWAVE_JS_WS_URL=ws://host.docker.internal:3000
ZWAVE_NODE_ID=2
ZWAVE_DEVICE_ID=shelly-wave-plug-ac
ZWAVE_DEVICE_NAME=portable_ac
COOLING_SAMPLE_CHANNEL=orion:home:cooling:sample
COOLING_POLL_INTERVAL_SEC=5
HEARTBEAT_INTERVAL_SEC=10
ORION_HEALTH_CHANNEL=orion:system:health
```

README must document: host Z-Wave JS path, pair steps, set `ZWAVE_NODE_ID`, `ORION_ZWAVE_ENABLED=true`, never mount stick here.

- [ ] **Step 4: Heartbeat test**

Copy pattern from `services/orion-power-guard/tests/test_heartbeat_chassis.py` into `services/orion-zwave/tests/test_heartbeat_chassis.py` (assert chassis builds with service name `orion-zwave`).

- [ ] **Step 5: Run tests + sync env**

```bash
pytest services/orion-zwave/tests -q
python scripts/sync_local_env_from_example.py orion-zwave
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-zwave
git commit -m "feat(orion-zwave): read-only cooling sample publisher"
```

---

### Task 4: sql-writer history consumer

**Files:**
- Create: `services/orion-sql-writer/app/models/home_cooling_sample.py`
- Create: `services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py`
- Modify: `services/orion-sql-writer/app/models/__init__.py`
- Modify: `services/orion-sql-writer/app/settings.py` (`DEFAULT_ROUTE_MAP`)
- Modify: `services/orion-sql-writer/app/worker.py` (`MODEL_MAP`)
- Modify: `services/orion-sql-writer/.env_example` (subscribe list)

**Interfaces:**
- Consumes: bus kind `home.cooling.sample.v1`
- Produces: table `home_cooling_sample` rows for Hub history

- [ ] **Step 1: Failing shape test**

Mirror `services/orion-sql-writer/tests/test_cabinet_ambient_spike_sql_shape.py`: assert route map, MODEL_MAP, subscribe channel string, column names ↔ payload fields (`ts`, `node`, `role`, `cooling_watts`, `cooling_volts`, `cooling_amps`, `switch_on`, `zwave_node_id`, `controller_ready`, `device_online`).

- [ ] **Step 2: Implement model**

```python
class HomeCoolingSampleSQL(Base):
    __tablename__ = "home_cooling_sample"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    node = Column(String, nullable=False)
    role = Column(String, nullable=False)
    cooling_watts = Column(Float, nullable=True)
    cooling_volts = Column(Float, nullable=True)
    cooling_amps = Column(Float, nullable=True)
    switch_on = Column(Boolean, nullable=True)
    zwave_node_id = Column(Integer, nullable=False)
    controller_ready = Column(Boolean, nullable=False)
    device_online = Column(Boolean, nullable=False)
    payload_json = Column(JSON, nullable=True)
```

Wire mapper from `HomeCoolingSampleV1` → SQL row (flatten measurements; store full payload in `payload_json`).

- [ ] **Step 3: Route + subscribe**

```python
# settings DEFAULT_ROUTE_MAP
"home.cooling.sample.v1": "HomeCoolingSampleSQL",
```

```python
# worker MODEL_MAP
"HomeCoolingSampleSQL": (HomeCoolingSampleSQL, HomeCoolingSampleV1),
```

Add `orion:home:cooling:sample` to `SQL_WRITER_SUBSCRIBE_CHANNELS` in `.env_example`. Sync local `.env`.

- [ ] **Step 4: Tests**

```bash
pytest services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py -q
python scripts/sync_local_env_from_example.py orion-sql-writer
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer
git commit -m "feat(sql-writer): persist home.cooling.sample.v1"
```

---

### Task 5: Hub Cabinet Cooling strip

**Files:**
- Create: `services/orion-hub/scripts/cabinet_cooling_routes.py`
- Create: `services/orion-hub/tests/test_cabinet_cooling_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/templates/index.html` (inside `#cabinet`, after ambient block / before or beside sensor grid)
- Modify: `services/orion-hub/static/js/cabinet-sensors.js`
- Modify: `services/orion-hub/.env_example` if DB DSN keys needed (reuse existing Postgres settings ambient uses)

**Interfaces:**
- Consumes: Postgres `home_cooling_sample` (history); latest = newest row or in-process bus cache if Hub already has a pattern — prefer **latest row from Postgres** + optional soft-fail empty (same as ambient history soft-fail). If a bus cache helper exists and is cheap, use it for latest; otherwise `ORDER BY ts DESC LIMIT 1`.
- Produces: JSON for UI; **no** control endpoints.

- [ ] **Step 1: Failing route tests**

Assert router prefix `/api/cabinet/cooling`, `latest` returns `ok` / `sample` keys with absent-safe nulls, `history` accepts `window=24h`.

- [ ] **Step 2: Implement routes**

```python
router = APIRouter(prefix="/api/cabinet/cooling", tags=["cabinet-cooling"])

@router.get("/latest")
async def api_cabinet_cooling_latest() -> dict[str, Any]:
    ...

@router.get("/history")
async def api_cabinet_cooling_history(window: str = Query("24h")) -> dict[str, Any]:
    ...
```

Response `latest` sketch:

```json
{
  "ok": true,
  "age_sec": 1.2,
  "sample": {
    "ts": "...",
    "cooling_watts": 412.5,
    "switch_on": true,
    "controller_ready": true,
    "device_online": true,
    "product": "Shelly Wave Plug"
  }
}
```

When no rows: `ok=false`, `sample=null` (not zeros).

Wire in `api_routes.py` next to ambient/sensors routers.

- [ ] **Step 3: UI markup in `#cabinet`**

Insert a Cooling card (title: “Cooling — portable AC (Shelly Wave)”) with tiles: watts, volts (optional), reported state (text only), age, status; plus `#cabinetCoolingWattsChart` and window buttons `24h/3d/7d` mirroring ambient.

**Do not** add a primary-nav Z-Wave tab. **Do not** add an on/off button.

- [ ] **Step 4: JS poll in `cabinet-sensors.js`**

- URLs: `/api/cabinet/cooling/latest`, `/api/cabinet/cooling/history?window=`
- Poll only while Cabinet subview visible (same activate/deactivate lifecycle).
- Render `—` / `absent` when `cooling_watts` is null.
- Chart watts series from history when present.

- [ ] **Step 5: Tests**

```bash
pytest services/orion-hub/tests/test_cabinet_cooling_routes.py -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub
git commit -m "feat(hub): Cabinet cooling strip for Shelly Wave watts"
```

---

### Task 6: Live smoke + docs + PR gate

**Files:**
- Modify: `services/orion-zwave/README.md` (pair + enable checklist)
- Modify: `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md` (Status → Implemented / awaiting merge)
- Create: `docs/superpowers/pr-reports/2026-09-25-zwave-cabinet-cooling-pr.md` (at PR time)

- [ ] **Step 1: Enable on Athena**

```bash
# After Shelly paired and ZWAVE_NODE_ID known:
# set ORION_ZWAVE_ENABLED=true in services/orion-zwave/.env
cd /mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling
scripts/safe_docker_build.sh orion-zwave up -d --build
# recreate sql-writer + hub with synced env so subscribe + routes load
```

- [ ] **Step 2: Verify bus sample**

```bash
# redis-cli or existing bus peek tool on COOLING_SAMPLE_CHANNEL
# Expect home.cooling.sample.v1 with cooling_watts when AC running
```

- [ ] **Step 3: Verify Hub**

Open Hub → Biometrics → Cabinet → Cooling strip shows watts beside Environment temp. Confirm no on/off control. Confirm Athena biometrics `peak_pressure` unchanged when AC watts move.

- [ ] **Step 4: Failure smoke**

Stop `athena-zwave-js-ui` briefly → Cooling shows stale/absent, not `0.0`.

- [ ] **Step 5: Agent checks**

```bash
git diff --check
pytest tests/test_home_cooling_bus_catalog.py services/orion-zwave/tests services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py services/orion-hub/tests/test_cabinet_cooling_routes.py -q
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

- [ ] **Step 6: Commit docs + open PR**

```bash
git add docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md docs/superpowers/pr-reports/
git commit -m "docs: zwave cabinet cooling acceptance notes"
git push -u origin HEAD
gh pr create ...
```

---

## Spec coverage self-review

| Spec requirement | Task |
|---|---|
| Host Z-Wave JS owns USB + pairing | Task 1 |
| Thin `orion-zwave` WS client | Task 3 |
| `home.cooling.sample.v1` + registered channel | Task 2 |
| sql-writer history | Task 4 |
| Hub Cabinet Cooling strip (inside `#cabinet`) | Task 5 |
| Read-only / no Hub Z-Wave tab / no peak_pressure fold-in | Global + Tasks 3/5 |
| Absent-is-not-zero | Tasks 2–5 |
| Live smoke + README | Task 6 |

## Placeholder scan

No TBD/TODO/“implement later” steps remain. Device node id is operator-filled after Task 1 pairing (`ZWAVE_NODE_ID`).

## Type consistency

- Schema id: `HomeCoolingSampleV1`
- Message kind: `home.cooling.sample.v1`
- Channel: `orion:home:cooling:sample`
- SQL class: `HomeCoolingSampleSQL` / table `home_cooling_sample`
- Hub prefix: `/api/cabinet/cooling`
