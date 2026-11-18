"""Ask Hub to take one curiosity self-inquiry turn now.

The ``self_review`` workflow ("reflect on yourself", "run a self review") used to run the
self-study reflection loop. That loop was retired on 2026-10-30
(docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md). The phrase now
routes here instead: cortex-orch publishes one ``CuriositySelfInquiryRequestV1`` on
``orion:curiosity:self_inquiry:request`` and Hub calls
``CuriosityInvestigation.tick_self_inquiry(force=True)``, the same path as the operator's
``POST /api/self-inquiry/run-now``, but narrower: the chat phrase may skip only the line's
cooldown and daily cap. Heat, strain, rest, the time window, ``enabled`` and every health gate
still apply.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SELF_INQUIRY_REQUEST_CHANNEL = "orion:curiosity:self_inquiry:request"
SELF_INQUIRY_REQUEST_KIND = "curiosity.self_inquiry.request.v1"


class CuriositySelfInquiryRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.self_inquiry.request.v1"] = "curiosity.self_inquiry.request.v1"
    # Who asked: "workflow:self_review" for the chat phrase. Free text, logged only.
    requested_by: str = Field(min_length=1, max_length=64)
    # The workflow invocation's correlation id, so the Hub log line joins the chat turn.
    correlation_id: str = Field(default="", max_length=128)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
