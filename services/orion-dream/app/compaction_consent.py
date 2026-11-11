"""Dream compaction asks the want market's consent gate (attention patch 3d, first consumer).

Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md section 3. The gate:
orion/physiology/consent.py, rules in config/physiology/consent.yaml.

``market_consent()`` returns the applier's consent callable (``app.compaction_applier.CompactionConsent``).
For one delta and plan it bids ``POST /physiology/bid`` (``orion.physiology.client.bid``) for family
``dream.compaction`` with a ``ConsentRequestV1``:

* verb ``compact`` when the apply only downscales and writes gist cards (``downscale_only``, or a delta with
  nothing to prune), ``reversibility: snapshot_restorable`` (the applier snapshots first and rolls back);
* verb ``prune`` when it would delete episodic rows, ``reversibility: irreversible`` (always refused);
* attests: ``snapshot_dir_writable`` (checked here, on the real directory) and ``downscale_only``;
* ``payload_digest``: the delta's sha256, so the verdict is bound to this exact delta.

It approves only when ALL hold: the market answered; its decision carries a consent verdict ``allow`` for
this want and this digest; and the shared binding rule (``orion.physiology.binding.bind_verdict``) admits it
with the hold ending now. The applier is one-shot (no hold loop), so a price or hunger deferral does not
hold it; only consent decides. A consent block is never admitted, even at the hold's end. No answer
(market down, timeout, an old market that rejects the request) is a no: the applier fails closed.

Every ask is one log line ``compaction_consent delta=... verdict=... rule=... approved=...`` and one
want_ledger row (producer ``PRODUCER``).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from orion.physiology.binding import bind_verdict
from orion.schemas.compaction import MemoryCompactionDeltaV1
from orion.schemas.physiology_want import ConsentRequestV1, WantDecisionV1, WantV1

from app.compaction_applier import CompactionApplyReceiptV1, CompactionConsent, ConsentAnswer, apply_compaction_delta

logger = logging.getLogger("orion-dream.compaction_consent")

FAMILY = "dream.compaction"
TIER = 2
PRODUCER = "orion-dream.compaction_applier.consent"
VERB_COMPACT = "compact"
VERB_PRUNE = "prune"
# A REM pass waits on this once per staged delta (a cold market history read can take seconds).
CONSENT_TIMEOUT_SEC = 5.0

Bid = Callable[..., Optional[WantDecisionV1]]


def delta_digest(delta: MemoryCompactionDeltaV1) -> str:
    """sha256 of the delta's canonical JSON: the verdict is for exactly this payload."""
    body = json.dumps(delta.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def snapshot_dir_writable(path: Optional[str] = None) -> bool:
    """The applier's §14 snapshot directory exists (or can be made) and is writable right now."""
    if path is None:
        from app.settings import settings

        path = settings.DREAM_COMPACTION_SNAPSHOT_DIR
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def plan(delta: MemoryCompactionDeltaV1, downscale_only: bool) -> tuple[str, str, bool]:
    """(verb, reversibility, downscale_only attest) for this apply. Nothing to prune is downscale only."""
    only = bool(downscale_only) or not delta.prune
    if only:
        return VERB_COMPACT, "snapshot_restorable", True
    return VERB_PRUNE, "irreversible", False


def consent_want(delta: MemoryCompactionDeltaV1, downscale_only: bool, *, writable: bool) -> WantV1:
    verb, rev, only = plan(delta, downscale_only)
    want_id = f"{FAMILY}:{delta.delta_id}"[:200]
    attests = (["snapshot_dir_writable"] if writable else []) + (["downscale_only"] if only else [])
    created = delta.created_at if delta.created_at.tzinfo else delta.created_at.replace(tzinfo=timezone.utc)
    return WantV1(want_id=want_id, family=FAMILY, producer=PRODUCER, verb=verb, object_ref=delta.delta_id[:500],
                  tier=TIER, opened_at=created, correlation_id=want_id,
                  consent_request=ConsentRequestV1(want_id=want_id, family=FAMILY, verb=verb,
                                                   object_ref=delta.delta_id[:500],
                                                   payload_digest=delta_digest(delta), reversibility=rev,
                                                   attests=attests))


def judge(want: WantV1, decision: Optional[WantDecisionV1], now: datetime) -> ConsentAnswer:
    """Pure: the market's decision -> the applier's answer. Fails closed on anything but a clean allow."""
    if decision is None:
        return ConsentAnswer(False, reason="no_answer: the market did not answer (down, timeout, or rejected the request)")
    v = decision.consent
    did = decision.decision_id
    if v is None:
        return ConsentAnswer(False, reason="no_verdict: the decision carries no consent verdict", decision_id=did)
    sent = want.consent_request
    got = decision.want.consent_request
    if decision.want_id != want.want_id or got is None or sent is None or got.payload_digest != sent.payload_digest:
        return ConsentAnswer(False, v.verdict, v.rule_id, "verdict_mismatch: not for this delta", did)
    if v.verdict != "allow":
        return ConsentAnswer(False, v.verdict, v.rule_id, v.reason, did)
    bv = bind_verdict(decision, now, now)
    if not bv.admit:
        return ConsentAnswer(False, v.verdict, v.rule_id, f"not_admitted:{bv.why}", did)
    return ConsentAnswer(True, v.verdict, v.rule_id, v.reason, did)


def market_consent(*, bid: Optional[Bid] = None, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                   writable_fn: Callable[[], bool] = snapshot_dir_writable,
                   timeout_sec: float = CONSENT_TIMEOUT_SEC) -> CompactionConsent:
    """The applier's consent callable, backed by the want market. Never raises (the applier also guards)."""
    if bid is None:
        from orion.physiology.client import bid as client_bid

        bid = client_bid

    def ask(delta: MemoryCompactionDeltaV1, downscale_only: bool) -> ConsentAnswer:
        try:
            writable = bool(writable_fn())
        except Exception:  # noqa: BLE001 -- unknown is not writable
            writable = False
        want = consent_want(delta, downscale_only, writable=writable)
        try:
            decision = bid(want, timeout_sec=timeout_sec)
        except Exception:  # noqa: BLE001
            logger.warning("compaction_consent_bid_failed delta=%s", delta.delta_id, exc_info=True)
            decision = None
        ans = judge(want, decision, now_fn())
        logger.info("compaction_consent delta=%s verb=%s verdict=%s rule=%s approved=%s decision=%s reason=%s",
                    delta.delta_id, want.verb, ans.verdict, ans.rule_id, ans.approved, ans.decision_id, ans.reason)
        return ans

    return ask


async def ask_consent_without_store(delta: MemoryCompactionDeltaV1, *,
                                    consent: Optional[CompactionConsent] = None) -> CompactionApplyReceiptV1:
    """The REM pass's consent ask: the applier with the market's consent and NO memory store (``store=None``
    is fixed here, never a parameter), so the verdict is logged and stored (want_ledger) and nothing can be
    written. The bid is blocking HTTP, so it runs in a worker thread."""
    receipt = await asyncio.to_thread(apply_compaction_delta, delta, consent or market_consent(), store=None)
    logger.info("rem compaction consent asked id=%s status=%s verdict=%s rule=%s",
                delta.delta_id, receipt.status, receipt.consent_verdict, receipt.consent_rule)
    return receipt


__all__ = ["ask_consent_without_store", "CONSENT_TIMEOUT_SEC", "FAMILY", "PRODUCER", "consent_want", "delta_digest", "judge", "market_consent",
           "plan", "snapshot_dir_writable"]
