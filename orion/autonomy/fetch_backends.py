from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

_FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"


async def firecrawl_search_backend(query: str, *, max_articles: int, api_key: str) -> dict:
    """Tier B readonly search via Firecrawl REST API."""

    def _call() -> dict:
        payload = json.dumps({"query": query, "limit": max(1, int(max_articles))}).encode("utf-8")
        req = urllib.request.Request(
            _FIRECRAWL_SEARCH_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            return {"success": False, "error": f"http_{exc.code}", "detail": detail}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}

        if not isinstance(parsed, dict):
            return {"success": False, "error": "invalid_response"}

        data = parsed.get("data") or []
        urls: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    url = item.get("url")
                    if url:
                        urls.append(str(url))
        success = bool(parsed.get("success")) and bool(urls)
        return {"success": success, "urls": urls}

    return await asyncio.to_thread(_call)
