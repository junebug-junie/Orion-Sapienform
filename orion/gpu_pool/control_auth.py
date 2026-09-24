"""Operator control signing: the pool's operator secret never travels on the bus.

``orion:gpu_pool:control:request`` is visible to every ``orion:*`` subscriber (bus-mirror,
bus-tap). So a control message carries an HMAC-SHA256 over its own canonical content, an
``issued_at`` and a one-time ``nonce``, instead of the secret. The pool recomputes the HMAC with its
``GPU_POOL_OPERATOR_TOKEN``, refuses anything older than ``MAX_SKEW_SEC`` and any nonce it has
already accepted, so a message copied off the bus can be neither forged nor replayed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

MAX_SKEW_SEC = 60.0


def _canonical(fields: dict[str, Any]) -> bytes:
    return json.dumps({k: v for k, v in fields.items() if k != "signature"}, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def sign(fields: dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), _canonical(fields), hashlib.sha256).hexdigest()


def signed_control(secret: str, *, now: datetime | None = None, **fields: Any):
    """Build a signed GpuPoolControlV1 (the caller never handles the signature itself)."""
    from orion.schemas.gpu_pool import GpuPoolControlV1

    unsigned = GpuPoolControlV1(issued_at=now or datetime.now(timezone.utc), nonce=uuid.uuid4().hex,
                                signature="0" * 64, **fields)
    body = unsigned.model_dump(mode="json")
    return unsigned.model_copy(update={"signature": sign(body, secret)})


class NonceLedger:
    """Nonces accepted within the skew window; older ones cannot pass the time check anyway."""

    def __init__(self) -> None:
        self._seen: dict[str, datetime] = {}

    def check_and_record(self, nonce: str, now: datetime) -> bool:
        horizon = now - timedelta(seconds=2 * MAX_SKEW_SEC)
        self._seen = {n: t for n, t in self._seen.items() if t > horizon}
        if nonce in self._seen:
            return False
        self._seen[nonce] = now
        return True


def verify(ctl: Any, secret: str, now: datetime, ledger: NonceLedger) -> str | None:
    """None if the control message is authentic, fresh and new; otherwise the refusal reason."""
    if not secret:
        return "operator_secret_not_configured"
    if abs((now - ctl.issued_at).total_seconds()) > MAX_SKEW_SEC:
        return "operator_signature_stale"
    expected = sign(ctl.model_dump(mode="json"), secret)
    if not hmac.compare_digest(expected, ctl.signature):
        return "operator_signature_rejected"
    if not ledger.check_and_record(ctl.nonce, now):
        return "operator_nonce_replayed"
    return None
