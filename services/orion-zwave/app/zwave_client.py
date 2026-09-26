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


def _numeric_value(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _value_label(entry: dict[str, Any]) -> str:
    return str(entry.get("label") or entry.get("propertyLabel") or "")


def extract_meter_watts(values: dict[str, dict[str, Any]]) -> Optional[float]:
    """Return Electric W consumed from Meter CC 50, or None if absent."""
    for entry in values.values():
        if entry.get("commandClass") != METER_CC:
            continue
        if entry.get("propertyKey") == 65537:
            watts = _numeric_value(entry.get("value"))
            if watts is not None:
                return watts

    for entry in values.values():
        if entry.get("commandClass") != METER_CC:
            continue
        label = _value_label(entry).upper()
        if " W" in label or label.endswith("W") or "WATT" in label:
            watts = _numeric_value(entry.get("value"))
            if watts is not None:
                return watts

    for entry in values.values():
        if entry.get("commandClass") != METER_CC:
            continue
        if entry.get("property") == "value":
            watts = _numeric_value(entry.get("value"))
            if watts is not None:
                return watts

    return None


def extract_switch_on(values: dict[str, dict[str, Any]]) -> Optional[bool]:
    """Return binary switch currentValue from CC 37, or None if absent."""
    for entry in values.values():
        if entry.get("commandClass") != BINARY_SWITCH_CC:
            continue
        if entry.get("property") != "currentValue":
            continue
        raw = entry.get("value")
        if isinstance(raw, bool):
            return raw
    return None


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
        await self._send_command("set_api_schema", schemaVersion=schema_version)
        await self._send_command("start_listening")
        await self._send_command("get_all_nodes_metadata")

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
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Z-Wave websocket listener stopped with error")

    async def _dispatch_loop(self) -> None:
        while True:
            message = await self._inbound.get()
            self._handle_message(message)

    async def _read_until(self, predicate) -> dict[str, Any]:
        while True:
            message = await asyncio.wait_for(self._inbound.get(), timeout=10)
            if predicate(message):
                return message

    def _next_message_id(self) -> str:
        self._message_id += 1
        return str(self._message_id)

    async def _send_command(self, command: str, **params: Any) -> None:
        assert self._ws is not None
        payload = {"messageId": self._next_message_id(), "command": command, **params}
        await self._ws.send(json.dumps(payload))

    def _handle_message(self, message: dict[str, Any]) -> None:
        if message.get("type") == "version":
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
            args = event.get("args") or []
            if not args:
                return
            entry = args[0]
            if not isinstance(entry, dict):
                return
            node_id = entry.get("nodeId") or event.get("nodeId")
            if not isinstance(node_id, int):
                return
            bucket = self._values_by_node.setdefault(node_id, {})
            bucket[_value_map_key(entry)] = entry
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
