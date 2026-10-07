"""Live eval: what camera perception adds to a Hub chat prompt, on real data.

Builds Hub's Situation block twice through the real shared builder
(`hub_settings_to_runtime_namespace` -> `build_situation_for_ctx`), once with
ORION_SITUATION_PERCEPTION_ENABLED off and once on, against the live
`vision_events` table, and reports the Room/Street lines, the provider status,
and the added characters versus the prompt budget.

Read-only. The identity-ask claim is stubbed to False so running this never
spends Orion's real once-per-cooldown "is that you?" ask.

Usage (from repo root; POSTGRES_URI must point at the live database):

    POSTGRES_URI=... PYTHONPATH=. python services/orion-hub/evals/run_situation_perception_eval.py

Exit 1 when the perception-on fragment exceeds the budget or a non-live
perception status still carries scene text (a stale scene leaking).
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from orion.situational import context as situation_mod

_OTHERS_OFF = dict(
    ORION_SITUATION_WEATHER_ENABLED=False,
    ORION_SITUATION_AFFECT_ENABLED=False,
    ORION_SITUATION_CURIOSITY_ENABLED=False,
    ORION_SITUATION_REVERIE_ENABLED=False,
    ORION_SITUATION_CABINET_ENABLED=False,
)


async def _no_ask(*_a, **_k) -> bool:
    return False


async def _build(perception_on: bool) -> tuple[dict, str, int]:
    ns = situation_mod.hub_settings_to_runtime_namespace(
        SimpleNamespace(**_OTHERS_OFF, ORION_SITUATION_PERCEPTION_ENABLED=perception_on)
    )
    ns.orion_situation_runtime_enabled = False
    situation_mod._SITUATION_CACHE.clear()
    brief, frag = await situation_mod.build_situation_for_ctx(
        {"session_id": f"perception-eval-{perception_on}"}, ns
    )
    return brief, str(frag.get("compact_text") or ""), int(frag.get("max_chars_applied") or 0)


def main() -> int:
    situation_mod.try_claim_identity_ask = _no_ask  # never spend the real ask
    _off_brief, off_text, _ = asyncio.run(_build(False))
    brief, on_text, cap = asyncio.run(_build(True))
    perception = brief.get("perception") or {}
    status = {
        k: v
        for k, v in (brief.get("diagnostics") or {}).get("provider_status", {}).items()
        if k.startswith("perception")
    }
    lines = [l for l in on_text.splitlines() if l.startswith(("- Room", "- Street"))]
    report = {
        "perception_source": perception.get("source"),
        "observation_age_seconds": perception.get("observation_age_seconds"),
        "provider_status": status,
        "lines": lines,
        "off_chars": len(off_text),
        "on_chars": len(on_text),
        "added_chars": len(on_text) - len(off_text),
        "budget": cap,
    }
    print(json.dumps(report, indent=2, default=str))
    failures = []
    if cap and len(on_text) > cap:
        failures.append("fragment exceeds budget")
    if perception.get("source") != "live" and perception.get("scene_summary"):
        failures.append("non-live perception carries scene text")
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
