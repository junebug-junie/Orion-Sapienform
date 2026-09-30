from __future__ import annotations

import json

EMPTY_COMPLETION_ERROR = "compactor_digest_empty_completion"


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    body = text[3:]
    newline = body.find("\n")
    body = body[newline + 1 :] if newline >= 0 else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_compactor_digest_json(raw: str, model_cls):
    """Parse an LLM digest JSON payload into the given compactor digest model.

    Shared by chat_history_compactor and github_compactor so both fail with the
    same tokens:
    - ``compactor_digest_empty_completion`` for an empty/whitespace completion
      (live 2026-09: ~7 github_compactor failures were `Expecting value: line 1
      column 1`, i.e. nothing came back) -- callers treat it as retryable;
    - ``compactor_digest_not_object`` for a non-object payload.

    ``strict=False`` accepts raw control characters (literal newlines/tabs)
    inside JSON strings: a long markdown journal_body often carries them, and
    `Invalid control character` failed two otherwise-complete digests live.
    """
    text = str(raw or "").strip()
    if not text:
        raise ValueError(EMPTY_COMPLETION_ERROR)
    text = _strip_code_fence(text)
    if not text:
        raise ValueError(EMPTY_COMPLETION_ERROR)
    payload = json.loads(text, strict=False)
    if not isinstance(payload, dict):
        raise ValueError("compactor_digest_not_object")
    return model_cls.model_validate(payload)
