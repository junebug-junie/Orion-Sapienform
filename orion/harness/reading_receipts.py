"""Deterministic grounding for FCC reading recommendation receipts.

The model sees tool results while composing its draft, but that does not make
its persistence claims authoritative. This module follows the standard Claude
tool_use/tool_result IDs, validates the server-owned receipt, and supplies the
last gate before draft/final text leaves the harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

from orion.schemas.reading import (
    DurableReadingReceiptV1,
    ReadingRecommendationOutcomeV1,
    ReadingToolBindingV1,
    ReadingToolResultV1,
)
from orion.world_pulse_read.tools import deterministic_reading_request_id
from orion.world_pulse_read.urls import normalize_source_url

_RECOMMEND_TOOL = "mcp__orion-reading__recommend_reading"
_CONTEXT_FETCH_TOOL = "mcp__plugin_context-mode_context-mode__ctx_fetch_and_index"
_FETCH_TOOLS = {
    "WebFetch",
    "mcp__firecrawl__scrape",
    "mcp__firecrawl__firecrawl_scrape",
    # Context Mode's URL-bound fetch was observed on the live Unified Chat
    # rail. Its successful result is also direct evidence that this turn
    # contacted the source rather than merely seeing a URL in the prompt.
    _CONTEXT_FETCH_TOOL,
}


@dataclass
class _Recommendation:
    key: str
    url: str
    expected_request_id: UUID | None
    tool_use_ids: list[str] = field(default_factory=list)
    failure_kind: str = "missing_result"
    receipt: DurableReadingReceiptV1 | None = None


@dataclass
class _PendingCall:
    kind: str
    key: str
    url: str
    tool_name: str = ""


def _content_blocks(step: dict[str, Any]) -> list[dict[str, Any]]:
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return []
    message = raw.get("message") if isinstance(raw.get("message"), dict) else raw
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _body_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return "\n".join(
            str(block.get("text"))
            for block in body
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def _failure_kind(text: str) -> str:
    lowered = text.lower()
    return "rpc_timeout" if "timeout" in lowered or "timed out" in lowered else "tool_error"


def _fetch_url(name: str, arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return None
    if name not in _FETCH_TOOLS:
        return None
    raw = arguments.get("url")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else None


def _url_identity(raw: str) -> tuple[str, str, str, str] | None:
    try:
        parts = urlsplit(raw)
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    path = parts.path.rstrip("/") or "/"
    return parts.scheme.lower(), f"{parts.hostname.lower()}{port}", path, parts.query


def _same_source(left: str, right: str) -> bool:
    return _url_identity(left) == _url_identity(right)


def _has_source_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_source_content(item) for item in value)
    if isinstance(value, dict):
        return any(
            _has_source_content(value.get(key))
            for key in ("markdown", "content", "html", "text", "data")
        )
    return False


def _usable_fetch_result(tool_name: str, body: str) -> bool:
    if not body.strip():
        return False
    if tool_name == _CONTEXT_FETCH_TOOL:
        # A cache hit explicitly tells the model to use ctx_search and does
        # not contact or reveal the source in this turn.
        return not body.lstrip().lower().startswith("cached:")
    if not tool_name.startswith("mcp__firecrawl__"):
        return True
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, dict):
        return False
    if "success" in payload and payload["success"] is not True:
        return False
    return _has_source_content(payload)


class ReadingReceiptTracker:
    """Collect and aggregate recommendation attempts from raw FCC steps."""

    def __init__(self, binding: ReadingToolBindingV1 | None) -> None:
        self.binding = binding
        self._pending: dict[str, _PendingCall] = {}
        self._recommendations: dict[str, _Recommendation] = {}
        self._successful_fetch_urls: list[str] = []
        self._anonymous_id = 0

    def observe(self, step: dict[str, Any]) -> None:
        for block in _content_blocks(step):
            block_type = str(block.get("type") or "")
            if block_type == "tool_use":
                self._observe_tool_use(block)
            elif block_type == "tool_result":
                self._observe_tool_result(block)

    def _tool_use_id(self, block: dict[str, Any]) -> str:
        raw = block.get("id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        self._anonymous_id += 1
        return f"missing-tool-use-id-{self._anonymous_id}"

    def _observe_tool_use(self, block: dict[str, Any]) -> None:
        name = str(block.get("name") or "")
        arguments = block.get("input")
        tool_use_id = self._tool_use_id(block)
        if name == _RECOMMEND_TOOL:
            args = arguments if isinstance(arguments, dict) else {}
            raw_url = str(args.get("url") or "").strip()
            why_now = str(args.get("why_now") or "")
            try:
                url = normalize_source_url(raw_url)
            except ValueError:
                url = raw_url
            expected: UUID | None = None
            if self.binding is not None and url and why_now:
                expected = deterministic_reading_request_id(
                    self.binding, url=url, why_now=why_now
                )
            key = str(expected) if expected is not None else f"{url}\n{why_now}"
            recommendation = self._recommendations.setdefault(
                key,
                _Recommendation(key=key, url=url, expected_request_id=expected),
            )
            recommendation.tool_use_ids.append(tool_use_id)
            self._pending[tool_use_id] = _PendingCall("recommend", key, url)
            return
        fetch_url = _fetch_url(name, arguments)
        if fetch_url:
            self._pending[tool_use_id] = _PendingCall(
                "fetch", tool_use_id, fetch_url, tool_name=name
            )

    def _observe_tool_result(self, block: dict[str, Any]) -> None:
        raw_id = block.get("tool_use_id")
        if not isinstance(raw_id, str):
            return
        pending = self._pending.pop(raw_id, None)
        if pending is None:
            return
        body = _body_text(block.get("content"))
        if pending.kind == "fetch":
            if not block.get("is_error") and _usable_fetch_result(
                pending.tool_name, body
            ):
                self._successful_fetch_urls.append(pending.url)
            return

        recommendation = self._recommendations[pending.key]
        if block.get("is_error"):
            recommendation.failure_kind = _failure_kind(body)
            return
        try:
            payload = json.loads(body)
            response = ReadingToolResultV1.model_validate(payload)
            if not response.ok:
                recommendation.failure_kind = _failure_kind(response.error or body)
                return
            if response.error is not None:
                raise ValueError("successful receipt cannot also contain an error")
            receipt = DurableReadingReceiptV1.model_validate(response.result)
            if (
                recommendation.expected_request_id is None
                or receipt.request_id != recommendation.expected_request_id
            ):
                raise ValueError("receipt request ID does not match the bound call")
        except (ValueError, TypeError, json.JSONDecodeError):
            recommendation.failure_kind = "malformed_receipt"
            return
        recommendation.receipt = receipt

    def outcomes(self) -> list[ReadingRecommendationOutcomeV1]:
        outcomes: list[ReadingRecommendationOutcomeV1] = []
        for recommendation in self._recommendations.values():
            source_read = any(
                _same_source(recommendation.url, fetched)
                for fetched in self._successful_fetch_urls
            )
            if recommendation.receipt is not None:
                outcomes.append(
                    ReadingRecommendationOutcomeV1(
                        tool_use_ids=recommendation.tool_use_ids,
                        attempt_count=len(recommendation.tool_use_ids),
                        url=recommendation.url,
                        acceptance="accepted",
                        request_id=recommendation.receipt.request_id,
                        status=recommendation.receipt.status,
                        source_read=source_read,
                    )
                )
            else:
                outcomes.append(
                    ReadingRecommendationOutcomeV1(
                        tool_use_ids=recommendation.tool_use_ids,
                        attempt_count=len(recommendation.tool_use_ids),
                        url=recommendation.url,
                        acceptance="unknown",
                        source_read=source_read,
                        failure_kind=recommendation.failure_kind,
                    )
                )
        return outcomes


def enforce_reading_receipt_grounding(
    text: str, outcomes: list[ReadingRecommendationOutcomeV1]
) -> str:
    """Return text whose persistence claims cannot exceed durable receipts.

    Any unresolved recommendation causes a deterministic receipt-only response.
    This makes semantically novel false-success wording harmless without trying
    to maintain an ever-growing phrase blacklist. All-success turns retain the
    model response but receive a canonical, inspectable receipt footer.
    """

    if not outcomes:
        return text
    unknown = [item for item in outcomes if item.acceptance == "unknown"]
    receipts = [item for item in outcomes if item.acceptance == "accepted"]
    receipt_lines = [
        "Reading recommendation accepted with durable request_id "
        f"`{item.request_id}`; current status: `{item.status}`."
        for item in receipts
    ]
    if not unknown:
        missing_lines = [line for line in receipt_lines if line not in text]
        footer = "\n".join(missing_lines)
        return f"{text.strip()}\n\n{footer}".strip() if footer else text.strip()

    failure_lines: list[str] = []
    for item in unknown:
        attempts = f" after {item.attempt_count} attempts" if item.attempt_count > 1 else ""
        failure_lines.append(
            f"The reading recommendation for `{item.url}` was not confirmed by a "
            f"durable receipt{attempts}; acceptance is unknown."
        )
        if not item.source_read:
            failure_lines.append("I did not read that source during this turn.")
    return "\n".join([*receipt_lines, *failure_lines])
