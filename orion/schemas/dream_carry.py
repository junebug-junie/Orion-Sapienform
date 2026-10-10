"""The `dream.carry` admitted durable workflow: one dream carried through words and pictures.

Design: docs/superpowers/specs/2026-10-10-dream-carry-through-design.md.

    text(0) -> image(1) -> text(2) -> image(3) -> text(4) -> image(5) -> finish

orion-durable-runs drives the run. A **text hop** is one `DreamCarryStepRequestV1` (step="text")
answered by orion-dream under the run's LLM hold (`options.gpu_lease` at the gateway). An **image
hop** is a child `reverie.visual` run (mode dream hop: `ReverieVisualRunBriefV1.dream_hop`) that
paints the previous text hop's `image_prompt` and captions what it painted; the carry waits for its
terminal detail without holding anything. **finish** (step="finish") hands every hop to
orion-dream, which publishes one `dream.result.v1` (hops in `fragments`) to the `dreams` table.

A carry past its deadline still finishes, with the hops it made and `stopped_reason`: partial,
never empty-and-failed.
"""
from __future__ import annotations

from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.telemetry.dream import DreamSleepDigestV1

DREAM_CARRY_WORKFLOW = "dream.carry"
DREAM_CARRY_STEP_CHANNEL = "orion:dream:carry:step:request"
DREAM_CARRY_STEP_REPLY_PREFIX = "orion:dream:carry:step:reply"
DREAM_CARRY_STEP_REQUEST_KIND = "dream.carry.step.request.v1"
DREAM_CARRY_STEP_RESULT_KIND = "dream.carry.step.result.v1"
DREAM_CARRY_LLM_ROUTE = "metacog_background"
# The diffusion model's CLIP encoder silently drops everything past 77 tokens
# (orion-thought visual_chain.select_context_slot); 60 words stays inside it.
IMAGE_PROMPT_MAX_WORDS = 60

HopKind = Literal["text", "image"]
DreamCarryStep = Literal["text", "finish"]
DreamCarryStepStatus = Literal["done", "retry", "terminal"]


def dream_carry_run_id(trigger_id: str) -> str:
    """Deterministic: a resubmitted trigger dedupes at the durable store."""
    return "dream-carry-" + uuid5(NAMESPACE_URL, f"dream.carry:{trigger_id}").hex


def dream_hop_dispatch_id(run_id: str, hop_index: int) -> str:
    """The child reverie.visual run's dispatch id (and so its attempt row) for one image hop."""
    return f"dream-carry:{run_id}:{hop_index}"


def clip_image_prompt(text: str) -> str:
    return " ".join(text.split()[:IMAGE_PROMPT_MAX_WORDS])


class DreamCarryHopV1(BaseModel):
    """One hop. Text: passage + image_prompt. Image: what was painted (sha256) and seen (caption)."""
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    kind: HopKind
    passage: str | None = None
    image_prompt: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    caption: str | None = None
    child_run_id: str | None = None
    elapsed_sec: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def hop_shape(self):
        if self.kind == "text" and not ((self.passage or "").strip() and (self.image_prompt or "").strip()):
            raise ValueError("a text hop needs a passage and an image_prompt")
        if self.kind == "image" and not (self.sha256 and (self.caption or "").strip()):
            raise ValueError("an image hop needs the painted sha256 and what was seen")
        return self


class DreamCarryBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_id: str = Field(min_length=1)
    # The sleep this carry ends (#2565 digest); None for a hand-started carry.
    sleep: DreamSleepDigestV1 | None = None
    # T, I, T, I, ... ending on an image: even, so what Orion saw is the last word.
    hops: int = Field(default=6, ge=2, le=8)
    llm_route: str = DREAM_CARRY_LLM_ROUTE
    # Per text-hop RPC budget (AdmissionRuntime.execute reads brief.timeout_sec).
    timeout_sec: float = Field(default=180.0, gt=0)
    session_id: str = "dream-carry"

    @model_validator(mode="after")
    def ends_on_an_image(self):
        if self.hops % 2:
            raise ValueError("hops must be even: a carry alternates text/image and ends on an image")
        return self


class DreamCarryStepRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dream.carry.step.request.v1"] = DREAM_CARRY_STEP_REQUEST_KIND
    run_id: str
    correlation_id: str
    step: DreamCarryStep
    brief: DreamCarryBriefV1
    # Every hop made so far, in order. A text hop continues from the last two.
    hops: list[DreamCarryHopV1] = Field(default_factory=list, max_length=8)
    # text: the index this hop will have. Required for text.
    hop_index: int | None = Field(default=None, ge=0)
    # text only: the run's LLM hold, attached at the gateway (options.gpu_lease).
    gpu_lease: GpuLeaseRefV1 | None = None
    # finish only: why the carry stopped short (None when every hop was made).
    stopped_reason: str | None = None

    @model_validator(mode="after")
    def step_shape(self):
        if self.step == "text":
            if self.hop_index is None or self.hop_index % 2:
                raise ValueError("a text step needs an even hop_index")
            if self.gpu_lease is None:
                raise ValueError("a text step runs under the run's LLM hold")
        elif self.gpu_lease is not None:
            raise ValueError("only a text step carries the hold")
        return self


class DreamCarryStepResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dream.carry.step.result.v1"] = DREAM_CARRY_STEP_RESULT_KIND
    run_id: str
    correlation_id: str
    step: DreamCarryStep
    status: DreamCarryStepStatus
    reason: str | None = None
    # text done: the hop made.
    hop: DreamCarryHopV1 | None = None
    # finish done: the dreams.result dream_id published.
    dream_id: str | None = None
    elapsed_sec: float | None = Field(default=None, ge=0)
    retry_after_sec: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def result_shape(self):
        if self.status == "done" and self.step == "text" and (self.hop is None or self.hop.kind != "text"):
            raise ValueError("a done text step returns its text hop")
        if self.status == "done" and self.step == "finish" and not self.dream_id:
            raise ValueError("a done finish step names the dream it published")
        return self
