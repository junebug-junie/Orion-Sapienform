from __future__ import annotations

import json


def parse_compactor_digest_json(raw: str, model_cls):
    """Parse an LLM digest JSON payload into the given compactor digest model.

    Shared by chat_history_compactor and github_compactor so both fail with the
    same ``compactor_digest_not_object`` token on non-object payloads.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("compactor_digest_not_object")
    return model_cls.model_validate(payload)
