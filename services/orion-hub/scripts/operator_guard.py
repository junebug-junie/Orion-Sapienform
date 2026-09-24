"""Hub's operator check for mutating routes, shared by every panel that has controls.

The browser proves it is the operator with the HttpOnly ``orion_operator_token`` cookie (set by the
index route from ``SUBSTRATE_MUTATION_OPERATOR_TOKEN``) or an ``X-Orion-Operator-Token`` header.
Kept dependency-free so a small router (e.g. gpu_pool_routes) need not import api_routes.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Request


def _require_mutation_operator_guard(token: str | None) -> None:
    expected = str(os.getenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "")).strip()
    if not expected:
        raise HTTPException(status_code=503, detail="mutation_operator_token_not_configured")
    if not token or token.strip() != expected:
        raise HTTPException(status_code=403, detail="operator_guard_rejected")


def _resolve_operator_token(request: Request | None, token: str | None) -> str | None:
    header_token = str(token or "").strip()
    if header_token:
        return header_token
    cookie_token = str((request.cookies.get("orion_operator_token") if request is not None else "") or "").strip()
    return cookie_token or None
