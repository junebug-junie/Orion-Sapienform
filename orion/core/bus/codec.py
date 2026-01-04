# orion/core/bus/codec.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from uuid import UUID

from pydantic import BaseModel, ValidationError

from .bus_schemas import BaseEnvelope


@dataclass(frozen=True)
class DecodeResult:
    envelope: BaseEnvelope
    raw: Dict[str, Any]
    ok: bool
    error: Optional[str] = None


class OrionCodec:
    """
    Bulletproof encode/decode layer.

    Goals:
    - Never leak raw JSON dicts into business logic.
    - Centralize backwards-compat handling.
    - Provide helpful errors for telemetry.
    """

    def __init__(self, *, default_envelope_cls: Type[BaseEnvelope] = BaseEnvelope):
        self.default_envelope_cls = default_envelope_cls

    def encode(self, obj: BaseModel | Dict[str, Any]) -> bytes:
        if isinstance(obj, BaseModel):
            # Always serialize using aliases so our on-the-wire
            # envelope field names remain stable (e.g. schema_id -> schema).
            return obj.model_dump_json(by_alias=True).encode("utf-8")
        # dict payload: best-effort JSON
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def decode(self, data: bytes | str) -> DecodeResult:
        try:
            if isinstance(data, (bytes, bytearray)):
                s = data.decode("utf-8", "ignore")
            else:
                s = data
            raw = json.loads(s)
        except Exception as e:
            return DecodeResult(
                envelope=self.default_envelope_cls(
                    kind="system.error",
                    source={"name": "codec"},
                    payload={"error": f"invalid_json: {e}"},
                ),
                raw={},
                ok=False,
                error=f"invalid_json: {e}",
            )

        # Backward compatibility: if it isn't an envelope, wrap it.
        # Many legacy producers publish a raw dict (no schema) that contains:
        #   - reply_channel (string) instead of reply_to
        #   - correlation_id (string UUID)
        #   - no kind at all
        if not isinstance(raw, dict) or raw.get("schema") != "orion.envelope":
            legacy: Dict[str, Any] = raw if isinstance(raw, dict) else {"value": raw}

            # Best-effort promotion of transport fields.
            kind = legacy.get("kind") or "legacy.message"
            reply_to = legacy.get("reply_to") or legacy.get("reply_channel")

            corr_val = legacy.get("correlation_id")
            correlation_id: Optional[str] = None
            if isinstance(corr_val, (str, UUID)):
                correlation_id = str(corr_val)

            src = legacy.get("source")
            if not isinstance(src, dict):
                src = {"name": "legacy"}

            wrapped: Dict[str, Any] = {
                "schema": "orion.envelope",
                "schema_version": "2.0.0",
                "kind": kind,
                "source": src,
                "reply_to": reply_to,
                # keep the legacy message (including reply_channel/correlation_id) as payload
                "payload": legacy,
            }
            if correlation_id is not None:
                wrapped["correlation_id"] = correlation_id
            raw = wrapped

        try:
            env = self.default_envelope_cls.model_validate(raw)
            return DecodeResult(envelope=env, raw=raw, ok=True)
        except ValidationError as e:
            # Keep raw for error telemetry
            return DecodeResult(
                envelope=self.default_envelope_cls(
                    kind="system.error",
                    source={"name": "codec"},
                    payload={"error": "envelope_validation_failed", "details": e.errors()},
                ),
                raw=raw,
                ok=False,
                error="envelope_validation_failed",
            )
