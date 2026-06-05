"""Fuseki/TDB transient HTTP failure detection and client-side retry."""

from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

import requests

FUSEKI_LOCK_MARKERS = ("maximum lock count exceeded",)


def fuseki_http_error_body(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        return response.text or ""
    except Exception:
        return ""


def is_fuseki_lock_exhaustion(status_code: int | None, body: str) -> bool:
    if status_code not in (500, 503):
        return False
    lower = (body or "").lower()
    return any(marker in lower for marker in FUSEKI_LOCK_MARKERS)


def is_fuseki_retryable_http_error(status_code: int | None, body: str) -> bool:
    if is_fuseki_lock_exhaustion(status_code, body):
        return True
    return status_code in (502, 503, 504)


def fuseki_http_retry_attempts() -> int:
    raw = os.getenv("FUSEKI_HTTP_RETRY_ATTEMPTS", "4").strip()
    try:
        return max(1, min(int(raw), 8))
    except ValueError:
        return 4


def fuseki_http_retry_base_delay_sec() -> float:
    raw = os.getenv("FUSEKI_HTTP_RETRY_BASE_DELAY_SEC", "0.5").strip()
    try:
        return max(0.05, float(raw))
    except ValueError:
        return 0.5


T = TypeVar("T")


def call_with_fuseki_retry(fn: Callable[[], T]) -> T:
    """Retry Fuseki graph-store / SPARQL HTTP calls on lock exhaustion and gateway errors."""
    attempts = fuseki_http_retry_attempts()
    delay = fuseki_http_retry_base_delay_sec()
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except requests.HTTPError as exc:
            last_exc = exc
            resp = exc.response
            code = resp.status_code if resp is not None else None
            body = fuseki_http_error_body(resp)
            if attempt + 1 >= attempts or not is_fuseki_retryable_http_error(code, body):
                raise
            time.sleep(min(8.0, delay * (2**attempt)))
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(8.0, delay * (2**attempt)))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("fuseki retry loop exhausted")
