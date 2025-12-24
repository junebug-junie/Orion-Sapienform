from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Dict, Optional

from loguru import logger


class BusWorker:
    """
    Runs a blocking Redis PubSub listener in a background thread, using the existing
    orion.core.bus.service.OrionBus (sync redis client).

    It forwards decoded payload dicts into the FastAPI asyncio loop via
    asyncio.run_coroutine_threadsafe().
    """

    def __init__(
        self,
        *,
        bus,
        channel: str,
        loop: asyncio.AbstractEventLoop,
        on_payload: Callable[[Dict[str, Any], str], "asyncio.Future"],
    ):
        self.bus = bus
        self.channel = channel
        self.loop = loop
        self.on_payload = on_payload

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pubsub = None

    def start(self) -> None:
        if self._thread is not None:
            return

        if not getattr(self.bus, "enabled", False) or not getattr(self.bus, "client", None):
            raise RuntimeError("BusWorker cannot start: OrionBus is disabled or not connected")

        self._thread = threading.Thread(target=self._run, name="visionhost-bus-worker", daemon=True)
        self._thread.start()
        logger.info(f"[BUS_WORKER] started channel={self.channel}")

    def stop(self) -> None:
        self._stop.set()
        if self._pubsub is not None:
            try:
                self._pubsub.close()
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info("[BUS_WORKER] stopped")

    def _run(self) -> None:
        try:
            self._pubsub = self.bus.client.pubsub()
            self._pubsub.subscribe(self.channel)
            logger.info(f"[BUS_WORKER] subscribed channel={self.channel}")

            while not self._stop.is_set():
                msg = None
                try:
                    # timeout keeps thread responsive to stop()
                    msg = self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                except Exception as e:
                    logger.error(f"[BUS_WORKER] get_message failed err={e}")
                    continue

                if not msg:
                    continue

                if msg.get("type") != "message":
                    continue

                chan = msg.get("channel")
                raw = msg.get("data")
                if raw is None:
                    continue

                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                except Exception as e:
                    logger.error(f"[BUS_WORKER] JSON parse failed channel={chan} err={e} raw={raw!r}")
                    continue

                # schedule onto asyncio loop
                fut = asyncio.run_coroutine_threadsafe(self.on_payload(payload, str(chan)), self.loop)
                fut.add_done_callback(self._done_callback)

        except Exception as e:
            logger.error(f"[BUS_WORKER] crashed err={e}")
        finally:
            try:
                if self._pubsub is not None:
                    self._pubsub.close()
            except Exception:
                pass
            self._pubsub = None

    @staticmethod
    def _done_callback(fut) -> None:
        try:
            fut.result()
        except Exception as e:
            logger.error(f"[BUS_WORKER] handler error err={e}")
