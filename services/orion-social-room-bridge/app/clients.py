from __future__ import annotations

from typing import Any, Dict

import httpx

from orion.schemas.social_bridge import ExternalRoomPostRequestV1


def _callsyne_bridge_post_body(request: ExternalRoomPostRequestV1) -> Dict[str, Any]:
    """Shape for POST /api/bridge/messages: required room_id + text; optional fields omitted when empty."""
    meta = dict(request.metadata or {})
    media_hint = meta.pop("media_hint", None)
    body: Dict[str, Any] = {
        "room_id": request.room_id,
        "text": request.text,
    }
    if request.reply_to_message_id and str(request.reply_to_message_id).isdigit():
        body["reply_to_message_id"] = int(str(request.reply_to_message_id))
    if request.thread_id:
        body["thread_id"] = request.thread_id
    if media_hint:
        body["media_hint"] = media_hint
    if request.correlation_id:
        meta.setdefault("correlation_id", request.correlation_id)
    if meta:
        body["metadata"] = meta
    return body


class HubClient:
    def __init__(self, *, base_url: str, chat_path: str, timeout_sec: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_path = chat_path
        self._timeout = timeout_sec

    async def chat(self, *, payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}{self._chat_path}",
                json=payload,
                headers={"X-Orion-Session-Id": session_id},
            )
            response.raise_for_status()
            return response.json()


class CallSyneClient:
    def __init__(self, *, base_url: str, api_token: str, timeout_sec: float, post_path_template: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._timeout = timeout_sec
        self._post_path_template = post_path_template

    async def post_message(self, request: ExternalRoomPostRequestV1) -> Dict[str, Any]:
        path = self._post_path_template.format(room_id=request.room_id, thread_id=request.thread_id or "")
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        payload = _callsyne_bridge_post_body(request)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json() if response.content else {}
        if "message_id" not in data:
            data["message_id"] = data.get("id") or ""
        return data

    async def fetch_recent_messages(
        self,
        *,
        path: str,
        room_id: str,
        limit: int,
        since_message_id: str | None = None,
    ) -> Dict[str, Any] | list[Dict[str, Any]]:
        url = f"{self._base_url}{path}"
        headers: Dict[str, str] = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        params: Dict[str, Any] = {
            "room_id": room_id,
            "limit": max(int(limit), 1),
        }
        if since_message_id:
            params["since_message_id"] = since_message_id
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json() if response.content else {}


class SocialMemoryClient:
    def __init__(self, *, base_url: str, timeout_sec: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_sec

    async def get_summary(self, *, platform: str, room_id: str, participant_id: str | None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/summary",
                params={
                    "platform": platform,
                    "room_id": room_id,
                    "participant_id": participant_id,
                },
            )
            response.raise_for_status()
            return response.json()
