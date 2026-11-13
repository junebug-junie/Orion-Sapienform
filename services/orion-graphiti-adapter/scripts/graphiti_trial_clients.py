"""Clients for the Graphiti full-engine trial (spec 2026-11-13 core memory pages, Patch 0).

Three pieces Graphiti needs that Orion does not ship in the OpenAI shape:

- an LLM client that sends Graphiti's structured-output prompts to the local agent lane through
  llm-gateway's OpenAI passthrough, with the pool hold header, a real ``json_schema`` grammar and
  call accounting;
- an embedder that calls orion-vector-host's ``/embedding`` (EmbeddingGenerateV1), and RAISES on a
  bad answer (the live adapter's search embedder returns [] on failure; a write path must not
  store an empty vector);
- a cross-encoder stub (add_episode never reranks; Graphiti only needs the object).

The pure helpers (schema, think-stripping, request bodies) carry no graphiti import so the gate
tests run without graphiti-core installed. The graphiti subclasses are built by factories that
import graphiti lazily. Nothing here is imported by the live adapter (app/).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# One LLM call's JSON answer. Qwen-family servers can leak a <think> block into content when
# reasoning is not split out by the server; strip it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_reasoning(content: str) -> str:
    text = _THINK_RE.sub("", content or "")
    if "</think>" in text:  # unclosed opening tag swallowed by the server: keep what follows
        text = text.split("</think>", 1)[1]
    m = _FENCE_RE.match(text)
    return (m.group(1) if m else text).strip()


def parse_json_content(content: str) -> dict[str, Any]:
    text = strip_reasoning(content)
    if not text:
        raise ValueError("empty_llm_content")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"llm_json_not_object:{type(value).__name__}")
    return value


def response_format_for(schema: dict[str, Any] | None, name: str = "graphiti_response") -> dict[str, Any]:
    """``json_schema`` (llama.cpp turns it into a grammar) when Graphiti gave a response model;
    plain ``json_object`` otherwise."""
    if not schema:
        return {"type": "json_object"}
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "graphiti_response"
    return {"type": "json_schema", "json_schema": {"name": safe, "schema": schema, "strict": True}}


@dataclass
class CallStats:
    """Every LLM call the trial made, for the cost section of the report."""
    calls: int = 0
    failures: int = 0
    retries: int = 0              # Graphiti's own invalid-answer retries (generate_response loop)
    schema_fallbacks: int = 0     # json_schema refused by the server -> json_object
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls, "failures": self.failures, "retries": self.retries,
                "schema_fallbacks": self.schema_fallbacks, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.llm_seconds, 3)}

    def diff(self, before: dict[str, Any]) -> dict[str, Any]:
        now = self.snapshot()
        return {k: round(now[k] - before.get(k, 0), 3) if isinstance(now[k], float) else now[k] - before.get(k, 0)
                for k in now}


class HeaderBox:
    """Mutable per-call headers: the pool hold ref changes when a hold is recalled and re-taken."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def set_hold(self, encoded_ref: str | None) -> None:
        from orion.llm.resource_lease import GPU_LEASE_HEADER  # noqa: PLC0415
        if encoded_ref:
            self.headers[GPU_LEASE_HEADER] = encoded_ref
        else:
            self.headers.pop(GPU_LEASE_HEADER, None)


def embedding_request(text: str, doc_id: str = "graphiti_trial") -> dict[str, Any]:
    from orion.schemas.vector.schemas import EmbeddingGenerateV1  # noqa: PLC0415
    return EmbeddingGenerateV1(doc_id=doc_id, text=text, embedding_profile="default").model_dump(mode="json")


def embedding_from_response(data: Any, expected_dim: int) -> list[float]:
    from orion.schemas.vector.schemas import EmbeddingResultV1  # noqa: PLC0415
    if not isinstance(data, dict):
        raise ValueError("embedding_response_not_object")
    vector = EmbeddingResultV1.model_validate(data).embedding or []
    if len(vector) != expected_dim:
        raise ValueError(f"embedding_dim_mismatch:{len(vector)}!={expected_dim}")
    return [float(v) for v in vector]


def build_llm_client(*, base_url: str, model: str, header_box: HeaderBox, stats: CallStats,
                     max_tokens: int = 8192, temperature: float = 0.0, timeout_sec: float = 600.0):
    """OpenAIGenericClient that sends ``json_schema`` from Graphiti's response model."""
    import openai  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415
    from graphiti_core.llm_client.config import LLMConfig, ModelSize  # noqa: PLC0415
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: PLC0415
    from graphiti_core.llm_client.errors import RateLimitError  # noqa: PLC0415

    class OrionLocalLLMClient(OpenAIGenericClient):
        async def _generate_response(self, messages, response_model=None, max_tokens: int = max_tokens,
                                     model_size=ModelSize.medium) -> dict[str, Any]:
            msgs = []
            for m in messages:
                m.content = self._clean_input(m.content)
                if m.role in ("user", "system"):
                    msgs.append({"role": m.role, "content": m.content})
            schema = response_model.model_json_schema() if response_model is not None else None
            fmt = response_format_for(schema, getattr(response_model, "__name__", "graphiti_response"))
            started = time.monotonic()
            stats.calls += 1
            try:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model, messages=msgs, temperature=self.temperature,
                        max_tokens=max_tokens or self.max_tokens, response_format=fmt,
                        extra_headers=dict(header_box.headers))
                except openai.BadRequestError as exc:
                    if fmt.get("type") != "json_schema":
                        raise
                    stats.schema_fallbacks += 1
                    stats.errors.append(f"json_schema_refused:{str(exc)[:200]}")
                    resp = await self.client.chat.completions.create(
                        model=self.model, messages=msgs, temperature=self.temperature,
                        max_tokens=max_tokens or self.max_tokens, response_format={"type": "json_object"},
                        extra_headers=dict(header_box.headers))
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    stats.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    stats.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                choice = resp.choices[0]
                if getattr(choice, "finish_reason", None) == "length":
                    raise ValueError("llm_output_truncated_at_max_tokens")
                return parse_json_content(choice.message.content or "")
            except openai.RateLimitError as exc:
                stats.failures += 1
                stats.errors.append(f"rate_limit:{str(exc)[:200]}")
                raise RateLimitError from exc
            except Exception as exc:
                stats.failures += 1
                stats.errors.append(f"{type(exc).__name__}:{str(exc)[:200]}")
                raise
            finally:
                stats.llm_seconds += time.monotonic() - started

        async def generate_response(self, messages, response_model=None, max_tokens=None,
                                    model_size=ModelSize.medium):
            # Count Graphiti's own retry loop: every failed parse appends one "previous response
            # attempt was invalid" user message before the next try.
            before = len(messages)
            try:
                return await super().generate_response(messages, response_model, max_tokens, model_size)
            finally:
                stats.retries += sum(1 for m in messages[before:]
                                     if "previous response attempt was invalid" in (m.content or ""))

    config = LLMConfig(api_key="orion-local", model=model, small_model=model, base_url=base_url,
                       temperature=temperature, max_tokens=max_tokens)
    raw = AsyncOpenAI(api_key="orion-local", base_url=base_url, timeout=timeout_sec, max_retries=0)
    return OrionLocalLLMClient(config=config, client=raw)


def build_embedder(embed_url: str, *, expected_dim: int = 1024, timeout_sec: float = 30.0):
    import httpx  # noqa: PLC0415
    from graphiti_core.embedder.client import EmbedderClient  # noqa: PLC0415

    class OrionVectorHostEmbedder(EmbedderClient):
        def __init__(self) -> None:
            self._http = httpx.AsyncClient(timeout=timeout_sec)

        async def create(self, input_data) -> list[float]:  # type: ignore[override]
            text = input_data[0] if isinstance(input_data, list) and input_data else input_data
            resp = await self._http.post(embed_url.rstrip("/"), json=embedding_request(str(text)))
            resp.raise_for_status()
            return embedding_from_response(resp.json(), expected_dim)

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:  # type: ignore[override]
            return [await self.create(item) for item in input_data_list]

    return OrionVectorHostEmbedder()


def build_cross_encoder():
    from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: PLC0415

    class KeepOrderCrossEncoder(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            n = len(passages)
            return [(p, float(n - i) / max(n, 1)) for i, p in enumerate(passages)]

    return KeepOrderCrossEncoder()
