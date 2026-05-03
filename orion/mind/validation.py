"""Pure validators for Mind contracts (no HTTP / service imports beyond schemas)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from orion.schemas.chat_stance import ChatStanceBrief


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def hash_snapshot_inputs(snapshot_inputs: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(snapshot_inputs)).hexdigest()


def validate_merged_stance_brief(data: dict[str, Any]) -> ChatStanceBrief:
    return ChatStanceBrief.model_validate(data)


def validate_merged_stance_brief_optional(data: dict[str, Any]) -> tuple[ChatStanceBrief | None, str | None]:
    try:
        return validate_merged_stance_brief(data), None
    except ValidationError as exc:
        return None, str(exc)
