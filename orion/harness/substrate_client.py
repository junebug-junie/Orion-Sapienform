from __future__ import annotations

import logging
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.harness_finalize import HarnessDraftMoleculeV1, SubstrateFinalizeAppraisalV1

logger = logging.getLogger("orion.harness.substrate")

_REQUEST_KIND = "harness.draft.molecule.v1"
_RESULT_KIND = "substrate.finalize.appraisal.v1"


class HarnessSubstrateClient:
    """RPC client: HarnessDraftMoleculeV1 → SubstrateFinalizeAppraisalV1 (5a)."""

    def __init__(
        self,
        bus: OrionBusAsync,
        *,
        request_channel: str,
        result_prefix: str,
        source_name: str = "orion-harness-governor",
        timeout_sec: float = 5.0,
    ) -> None:
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix
        self.source_name = source_name
        self.timeout_sec = timeout_sec

    async def finalize_appraisal(
        self,
        molecule: HarnessDraftMoleculeV1,
        *,
        correlation_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> SubstrateFinalizeAppraisalV1:
        corr = correlation_id or molecule.correlation_id
        reply_channel = f"{self.result_prefix}{corr}"
        source = ServiceRef(name=self.source_name)
        env = BaseEnvelope(
            kind=_REQUEST_KIND,
            source=source,
            correlation_id=corr,
            reply_to=reply_channel,
            payload=molecule.model_dump(mode="json"),
        )
        logger.info(
            "harness substrate RPC -> %s corr=%s draft_hash=%s",
            self.request_channel,
            corr,
            molecule.draft_hash,
        )
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec if timeout_sec is not None else self.timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Substrate finalize RPC failed: {decoded.error}")

        payload = decoded.envelope.payload
        if decoded.envelope.kind == "system.error":
            detail = payload.get("details") if isinstance(payload, dict) else payload
            raise RuntimeError(f"Substrate finalize appraisal error: {detail}")

        if isinstance(payload, dict):
            return SubstrateFinalizeAppraisalV1.model_validate(payload)
        raise RuntimeError("Substrate finalize RPC returned non-dict payload")
