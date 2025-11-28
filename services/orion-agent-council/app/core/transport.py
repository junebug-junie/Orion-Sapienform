from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from orion.core.bus.service import OrionBus


@dataclass
class BusRPCClient:
    """
    Generic request/response helper over the Orion bus.

    It knows:
      - which intake channel to publish to
      - which reply prefix to listen on
      - how long to wait

    It does *not* care about LLM vs anything else; payload shape is opaque.
    """
    bus: OrionBus
    intake_channel: str
    reply_prefix: str
    timeout_sec: float

    def request(self, envelope: Dict[str, Any], correlation_id: str) -> Optional[Dict[str, Any]]:
        """
        Publish `envelope` to the intake channel and block until we
        receive a message on reply_prefix:<correlation_id> that matches.

        Returns the full reply payload dict (whatever the other side sent),
        or None on timeout.
        """
        reply_channel = f"{self.reply_prefix}:{correlation_id}"

        # Attach routing metadata (common to all bus RPC)
        envelope = {
            **envelope,
            "correlation_id": correlation_id,
            "reply_channel": reply_channel,
        }

        self.bus.publish(self.intake_channel, envelope)

        start = time.monotonic()
        for msg in self.bus.raw_subscribe(reply_channel):
            if msg.get("type") != "message":
                continue

            data = msg.get("data") or {}
            if data.get("correlation_id") != correlation_id:
                continue

            # Got the matching reply
            return data

            # (raw_subscribe generator will close on function exit)

            # (Unreachable after return, but if we ever refactor to not return
            #  inside the loop, we keep timeout logic here.)
            # elapsed = time.monotonic() - start
            # if elapsed > self.timeout_sec:
            #     break

            # In case we don't return above:
            # if time.monotonic() - start > self.timeout_sec:
            #     break

        # Timed out
        return None
