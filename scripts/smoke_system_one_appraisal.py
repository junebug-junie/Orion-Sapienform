#!/usr/bin/env python3
"""Read-only smoke for the shadow System One appraisal projection."""

from __future__ import annotations

import json
import os
import sys
from urllib.request import urlopen


def main() -> int:
    base = os.getenv("SUBSTRATE_RUNTIME_URL", "http://127.0.0.1:8115").rstrip("/")
    with urlopen(f"{base}/projections/system_one_appraisal", timeout=5) as response:
        payload = json.load(response)

    if not payload.get("ok"):
        print(json.dumps(payload, indent=2))
        return 2

    frame = payload.get("projection") or {}
    if frame.get("schema_version") != "system_one.appraisal.frame.v1":
        raise SystemExit("wrong system_one appraisal schema_version")

    expected = {
        "reverie_fit",
        "curiosity_pull",
        "deliberation_need",
        "attention_interrupt",
    }
    answers = frame.get("answers") or {}
    if set(answers) != expected:
        raise SystemExit(f"unexpected answer keys: {sorted(answers)}")

    for key in sorted(expected):
        answer = answers[key]
        if answer.get("type") != "score":
            raise SystemExit(f"{key}: expected score answer")
        if answer.get("score") is None:
            raise SystemExit(f"{key}: missing score")

    print(
        json.dumps(
            {
                "ok": True,
                "frame_id": frame.get("frame_id"),
                "provider": frame.get("provider"),
                "model_id": frame.get("model_id"),
                "generated_at": frame.get("generated_at"),
                "expires_at": frame.get("expires_at"),
                "answers": {
                    key: {
                        "score": answers[key].get("score"),
                        "confidence": answers[key].get("confidence"),
                    }
                    for key in sorted(expected)
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
