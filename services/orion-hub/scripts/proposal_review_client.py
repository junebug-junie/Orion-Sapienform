"""Read-only Hub client for the context-exec proposal review API."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from scripts.settings import settings

logger = logging.getLogger("orion-hub.proposal-review")

_ALLOWED_GET_PATHS = frozenset(
    {
        "/health",
        "/proposals",
    }
)


class ProposalReviewClientError(Exception):
    """Controlled proposal review client failure."""


class ProposalReviewUnavailable(ProposalReviewClientError):
    """Upstream proposal review API is unreachable or unhealthy."""


def _base_url() -> str:
    return str(settings.HUB_PROPOSAL_REVIEW_API_URL or "").strip().rstrip("/")


def _timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=float(settings.HUB_PROPOSAL_REVIEW_TIMEOUT_SEC))


def is_enabled() -> bool:
    return bool(settings.HUB_PROPOSAL_REVIEW_ENABLED)


def _assert_get_path(path: str) -> None:
    if path in _ALLOWED_GET_PATHS:
        return
    if path.startswith("/proposals/") and path.endswith("/eligibility"):
        return
    if path.startswith("/proposals/") and path.count("/") == 2:
        return
    raise ProposalReviewClientError(f"forbidden proposal review path: {path}")


async def _get_json(path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
    _assert_get_path(path)
    base = _base_url()
    if not base:
        raise ProposalReviewUnavailable("HUB_PROPOSAL_REVIEW_API_URL is not configured")

    url = f"{base}{path}"
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.get(url, params=params) as response:
                text = await response.text()
                try:
                    body: dict[str, Any] = {} if not text else json.loads(text)
                except ValueError:
                    body = {"raw": text}
                if response.status >= 500:
                    raise ProposalReviewUnavailable(
                        f"proposal review API error {response.status}: {body.get('detail', text)}"
                    )
                if response.status >= 400:
                    raise ProposalReviewClientError(
                        f"proposal review API error {response.status}: {body.get('detail', text)}"
                    )
                if not isinstance(body, dict):
                    return {"data": body}
                return body
    except aiohttp.ClientError as exc:
        logger.warning("proposal_review_client_transport_error path=%s err=%s", path, exc)
        raise ProposalReviewUnavailable("proposal review API unavailable") from exc


async def fetch_health() -> dict[str, Any]:
    return await _get_json("/health")


async def list_proposals(*, status: str | None = "pending_review") -> dict[str, Any]:
    params = {"status": status} if status else None
    return await _get_json("/proposals", params=params)


async def get_proposal(proposal_id: str) -> dict[str, Any]:
    return await _get_json(f"/proposals/{proposal_id}")


async def get_eligibility(proposal_id: str) -> dict[str, Any]:
    return await _get_json(f"/proposals/{proposal_id}/eligibility")
