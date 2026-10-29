"""Token counts for ONE LLM call (the per-call token ledger, 2026-10-29).

Why: "cost per 1,000 tokens" and "joules per token" need tokens per call, joinable to the
turn or act that made the call. The gateway's grammar windows (services/orion-llm-gateway/
app/grammar_emit.py) only carry per-window token TOTALS and, by design, no correlation ids.

Producers (``recorder``):

* ``gateway`` -- orion-llm-gateway, every call that reached a model: bus RPC
  (``llm.chat.request``) and the HTTP passthroughs (``/v1/chat/completions``,
  ``/v1/messages``). This is every GPU call the gateway places.
* ``harness_relay`` -- the harness governor's warm FCC relay (orion/harness/fcc_warm_relay.py),
  ONLY for a turn with no GPU lease. That turn goes relay -> FCC proxy -> gateway, and the FCC
  proxy drops our headers, so the gateway sees the same call with no correlation id.

Join rules (no double count under either):

* Totals over time (tokens per hour, joules per token): ``recorder = 'gateway'`` only.
* Tokens for one turn or act: every row with that ``correlation_id``. The FCC-path call the
  relay recorded is also a gateway row, but that gateway row has ``correlation_id`` NULL.

Known attribution gaps (the call is still in the totals, but no row carries the turn id):

* A per-turn FCC spawn (not the warm pool) with NO GPU lease goes straight to the FCC proxy; the
  proxy drops ``X-Orion-Correlation-Id`` and the relay is not in that path. Rare: lease-less
  harness turns were ~8 calls in 48 h on 2026-10-29, warm and spawned together.
* HTTP callers that send no ``X-Orion-Correlation-Id`` (AI Town NPCs, other OpenAI clients).

Token fields, normalized across the two wire formats llama.cpp speaks:

* ``prompt_tokens`` -- ALL prompt tokens, including the ones reused from the KV cache.
  OpenAI ``usage.prompt_tokens`` already counts them; Anthropic ``usage.input_tokens`` does NOT,
  so it is ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``.
* ``cached_prompt_tokens`` -- of those, the ones reused from the cache (not recomputed).
* ``completion_tokens`` -- generated tokens.
* ``token_source`` -- ``usage`` (the reply's ``usage`` block), ``timings`` (llama.cpp's own
  ``timings.prompt_n/cache_n/predicted_n``, e.g. an OpenAI stream with no usage chunk), or
  ``none`` (nothing reported; the call still ran).

Bounded: no prompt text, no response text.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

LLM_CALL_USAGE_KIND = "llm.call.usage.v1"
LLM_CALL_USAGE_CHANNEL = "orion:llm:call:usage"

Recorder = Literal["gateway", "harness_relay"]
CallPath = Literal["bus_rpc", "http_openai", "http_anthropic", "relay_anthropic"]
TokenSource = Literal["usage", "timings", "none"]


class LlmCallUsageV1(BaseModel):
    """One LLM call: who asked, where it ran, how many tokens, how long."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["llm.call.usage.v1"] = LLM_CALL_USAGE_KIND
    call_id: str = Field(min_length=1, max_length=64)  # uuid4 hex; the table's primary key
    recorded_at: AwareDatetime  # when the reply (or the stream's end) came back
    recorder: Recorder
    path: CallPath
    correlation_id: Optional[str] = Field(default=None, max_length=256)
    route: Optional[str] = Field(default=None, max_length=128)  # gateway route key (lane)
    model: Optional[str] = Field(default=None, max_length=256)  # the model the reply named, else requested
    served_by: Optional[str] = Field(default=None, max_length=256)  # worker label (circe-worker-chat)
    pool_role: Optional[str] = Field(default=None, max_length=64)  # GPU-pool role that was granted
    holder: Optional[str] = Field(default=None, max_length=256)  # who asked (bus source, http:anthropic)
    verb: Optional[str] = Field(default=None, max_length=128)  # bus path: options.verb
    node: Optional[str] = Field(default=None, max_length=128)  # recorder's own node name
    outcome: str = Field(min_length=1, max_length=64)  # served / upstream_timeout / client_gone ...
    stream: bool = False
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    cached_prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    token_source: TokenSource = "none"
    # grant -> reply (or stream end): the time the model worked on this call, queue excluded.
    duration_ms: Optional[int] = Field(default=None, ge=0)
    # lease request -> grant: the time spent in the GPU pool's line.
    wait_ms: Optional[int] = Field(default=None, ge=0)
    # llama.cpp's own timings.predicted_per_second; never derived from wall time.
    decode_tps: Optional[float] = Field(default=None, gt=0)


__all__ = [
    "LLM_CALL_USAGE_CHANNEL",
    "LLM_CALL_USAGE_KIND",
    "LlmCallUsageV1",
]
