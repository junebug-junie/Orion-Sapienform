from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import websockets

logger = logging.getLogger("orion-zwave.zwave_client")

METER_CC = 50
BINARY_SWITCH_CC = 37
DEFAULT_API_SCHEMA_VERSION = 33

# zwave-js Meter propertyKey / propertyKeyName for Shelly Wave Plug (and peers).
# 65537 is Electric_kWh_Consumed (energy) — NOT watts. Treating it as W pinned
# Hub at ~33 "watts" while the AC drew ~700 W on Electric_W_Consumed (66049).
METER_W_PROPERTY_KEY = 66049
METER_V_PROPERTY_KEY = 66561
METER_A_PROPERTY_KEY = 66817
METER_KWH_PROPERTY_KEY = 65537


def _numeric_value(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _entry_numeric(entry: dict[str, Any]) -> Optional[float]:
    """Prefer ``value``, then zwave-js event fields ``newValue`` / ``prevValue``."""
    for key in ("value", "newValue"):
        parsed = _numeric_value(entry.get(key))
        if parsed is not None:
            return parsed
    return None


def _value_label(entry: dict[str, Any]) -> str:
    return str(
        entry.get("propertyKeyName")
        or entry.get("label")
        or entry.get("propertyLabel")
        or ""
    )


def _meter_entries(values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in values.values()
        if isinstance(entry, dict)
        and entry.get("commandClass") == METER_CC
        and entry.get("property") == "value"
    ]


def _is_electric_watts(entry: dict[str, Any]) -> bool:
    if entry.get("propertyKey") == METER_W_PROPERTY_KEY:
        return True
    name = _value_label(entry).upper()
    if "KWH" in name or "KW/H" in name:
        return False
    if entry.get("propertyKey") == METER_KWH_PROPERTY_KEY:
        return False
    # Electric_W_Consumed, "Electric W", "Power (W)", etc.
    if "ELECTRIC_W" in name or name.endswith("_W_CONSUMED"):
        return True
    if "WATT" in name:
        return True
    if name == "W" or name.endswith(" W") or " (W)" in name:
        return True
    return False


def extract_meter_watts(values: dict[str, dict[str, Any]]) -> Optional[float]:
    """Return Electric W consumed from Meter CC 50, or None if absent.

    Never treats kWh / energy scales as watts.
    """
    for entry in _meter_entries(values):
        if not _is_electric_watts(entry):
            continue
        watts = _entry_numeric(entry)
        if watts is not None:
            return watts
    return None


def extract_meter_volts(values: dict[str, dict[str, Any]]) -> Optional[float]:
    for entry in _meter_entries(values):
        name = _value_label(entry).upper()
        if entry.get("propertyKey") == METER_V_PROPERTY_KEY or "ELECTRIC_V" in name:
            volts = _entry_numeric(entry)
            if volts is not None:
                return volts
    return None


def extract_meter_amps(values: dict[str, dict[str, Any]]) -> Optional[float]:
    for entry in _meter_entries(values):
        name = _value_label(entry).upper()
        if entry.get("propertyKey") == METER_A_PROPERTY_KEY or "ELECTRIC_A" in name:
            amps = _entry_numeric(entry)
            if amps is not None:
                return amps
    return None


def extract_switch_on(values: dict[str, dict[str, Any]]) -> Optional[bool]:
    """Return binary switch currentValue from CC 37, or None if absent."""
    for entry in values.values():
        if entry.get("commandClass") != BINARY_SWITCH_CC:
            continue
        if entry.get("property") != "currentValue":
            continue
        raw = entry.get("value")
        if raw is None and "newValue" in entry:
            raw = entry.get("newValue")
        if isinstance(raw, bool):
            return raw
    return None


def normalize_value_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Copy a zwave-js value / event args dict into a cacheable shape with ``value``."""
    out = dict(entry)
    if "value" not in out or out.get("value") is None:
        if "newValue" in out:
            out["value"] = out["newValue"]
    return out


def _value_map_key(entry: dict[str, Any]) -> str:
    command_class = entry.get("commandClass")
    endpoint = entry.get("endpoint", 0)
    prop = entry.get("property")
    property_key = entry.get("propertyKey")
    if property_key is not None:
        return f"{command_class}-{endpoint}-{prop}-{property_key}"
    return f"{command_class}-{endpoint}-{prop}"


def _schema_version_from_message(message: dict[str, Any]) -> Optional[int]:
    for key in ("apiSchemaVersion", "schemaVersion"):
        raw = message.get(key)
        if isinstance(raw, int):
            return raw
    driver = message.get("driver")
    if isinstance(driver, dict):
        for key in ("apiSchemaVersion", "schemaVersion"):
            raw = driver.get(key)
            if isinstance(raw, int):
                return raw
    return None


def ingest_node_values(
    values_by_node: dict[int, dict[str, dict[str, Any]]],
    node_id: int,
    values: Any,
) -> None:
    """Store Meter/Switch value entries from a start_listening node snapshot."""
    if isinstance(values, dict):
        entries = list(values.values())
    elif isinstance(values, list):
        entries = values
    else:
        return
    bucket = values_by_node.setdefault(node_id, {})
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized = normalize_value_entry(entry)
        bucket[_value_map_key(normalized)] = normalized


class ZWaveJSClient:
    """Minimal read-only zwave-js-server websocket client."""

    def __init__(self, ws_url: str, node_id: int) -> None:
        self.ws_url = ws_url
        self.node_id = node_id
        self._ws: Any = None
        self._message_id = 0
        self._inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._values_by_node: dict[int, dict[str, dict[str, Any]]] = {}
        self._controller_ready = False
        self._device_online: dict[int, bool] = {}
        self._product_by_node: dict[int, str] = {}
        self._listener_task: Optional[asyncio.Task[None]] = None
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._pending_results: dict[str, asyncio.Future[dict[str, Any]]] = {}

    @property
    def controller_ready(self) -> bool:
        return self._controller_ready

    def device_online(self, node_id: Optional[int] = None) -> bool:
        target = self.node_id if node_id is None else node_id
        return self._device_online.get(target, False)

    def product_name(self, node_id: Optional[int] = None) -> Optional[str]:
        target = self.node_id if node_id is None else node_id
        return self._product_by_node.get(target)

    def get_values(self, node_id: Optional[int] = None) -> dict[str, dict[str, Any]]:
        target = self.node_id if node_id is None else node_id
        return dict(self._values_by_node.get(target, {}))

    async def refresh_meter_watts(self, node_id: Optional[int] = None) -> Optional[float]:
        """Ask the stick for a fresh Electric_W reading; update local cache on success."""
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
        except Exception:
            logger.debug("node.poll_value Electric_W failed node=%s", target, exc_info=True)
            return None
        if not result.get("success", True):
            return None
        body = result.get("result") if isinstance(result.get("result"), dict) else {}
        watts = _numeric_value((body or {}).get("value"))
        if watts is None:
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
        return watts

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(self.ws_url, open_timeout=10)
            self._listener_task = asyncio.create_task(self._listen())
            await self._bootstrap()
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        for fut in list(self._pending_results.values()):
            if not fut.done():
                fut.cancel()
        self._pending_results.clear()
        for task_name in ("_dispatch_task", "_listener_task"):
            task = getattr(self, task_name)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_name, None)
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _bootstrap(self) -> None:
        hello = await self._read_until(lambda msg: msg.get("type") in {"version", "api", "event", "result"})
        schema_version = _schema_version_from_message(hello) or DEFAULT_API_SCHEMA_VERSION
        await self._request("set_api_schema", schemaVersion=schema_version)
        listening = await self._request("start_listening")
        self._ingest_listening_result(listening)
        # Optional metadata refresh; ignore failures.
        try:
            await self._request("get_all_nodes_metadata")
        except Exception:
            logger.debug("get_all_nodes_metadata skipped", exc_info=True)

    async def _listen(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Ignoring non-JSON zwave-js message: %r", raw)
                    continue
                await self._inbound.put(message)
                # Resolve pending RPC waits from the same stream (bootstrap path).
                if message.get("type") == "result":
                    mid = message.get("messageId")
                    fut = self._pending_results.get(str(mid)) if mid is not None else None
                    if fut is not None and not fut.done():
                        fut.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Z-Wave websocket listener stopped with error")

    async def _dispatch_loop(self) -> None:
        while True:
            message = await self._inbound.get()
            # Skip messages already consumed as RPC replies during bootstrap;
            # still handle events/results for live updates.
            self._handle_message(message)

    async def _read_until(self, predicate) -> dict[str, Any]:
        while True:
            message = await asyncio.wait_for(self._inbound.get(), timeout=10)
            if predicate(message):
                return message

    def _next_message_id(self) -> str:
        self._message_id += 1
        return str(self._message_id)

    async def _request(self, command: str, **params: Any) -> dict[str, Any]:
        assert self._ws is not None
        message_id = self._next_message_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_results[message_id] = fut
        payload = {"messageId": message_id, "command": command, **params}
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=30)
        finally:
            self._pending_results.pop(message_id, None)

    def _ingest_listening_result(self, message: dict[str, Any]) -> None:
        if not message.get("success", True):
            return
        result = message.get("result") or {}
        state = result.get("state") if isinstance(result, dict) else None
        if not isinstance(state, dict):
            return
        self._controller_ready = True
        nodes = state.get("nodes") or []
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("nodeId")
            if not isinstance(node_id, int):
                continue
            self._device_online[node_id] = bool(node.get("ready", True))
            label = node.get("label") or node.get("name")
            if isinstance(label, str) and label.strip():
                self._product_by_node[node_id] = label.strip()
            ingest_node_values(self._values_by_node, node_id, node.get("values"))

    def _handle_message(self, message: dict[str, Any]) -> None:
        if message.get("type") == "version":
            return

        if message.get("type") == "result":
            # Late/duplicate start_listening-shaped payloads (defensive).
            result = message.get("result")
            if isinstance(result, dict) and isinstance(result.get("state"), dict):
                self._ingest_listening_result(message)
            return

        event = message.get("event")
        if not isinstance(event, dict):
            return

        source = event.get("source")
        name = event.get("event")

        if source == "driver" and name in {"driver ready", "driver failed"}:
            self._controller_ready = name == "driver ready"
            return

        if source == "node" and name in {"ready", "sleep", "dead"}:
            node_id = event.get("nodeId")
            if isinstance(node_id, int):
                self._device_online[node_id] = name == "ready"
            return

        if source == "node" and name in {"value added", "value updated"}:
            args = event.get("args")
            entry: Any = None
            if isinstance(args, list) and args:
                entry = args[0]
            elif isinstance(args, dict):
                entry = args
            if not isinstance(entry, dict):
                return
            node_id = entry.get("nodeId") or event.get("nodeId")
            if not isinstance(node_id, int):
                return
            # Events carry newValue/prevValue; cache must expose ``value`` for readers.
            normalized = normalize_value_entry(entry)
            if "nodeId" not in normalized:
                normalized["nodeId"] = node_id
            bucket = self._values_by_node.setdefault(node_id, {})
            bucket[_value_map_key(normalized)] = normalized
            return

        if source == "node" and name == "metadata updated":
            node_id = event.get("nodeId")
            args = event.get("args") or []
            if isinstance(node_id, int) and args:
                meta = args[0]
                if isinstance(meta, dict):
                    product = meta.get("productLabel") or meta.get("productDescription")
                    if isinstance(product, str) and product.strip():
                        self._product_by_node[node_id] = product.strip()
