"""Gate: NPC LLM calls carry the open town session's id as X-Orion-Correlation-Id (2026-10-30).

orion-embodiment's session controller sets the Convex env var ORION_TOWN_CORRELATION_ID when a
session opens and removes it when the session ends (services/orion-embodiment/app/town_sessions.py
CORRELATION_ENV). The patched llm.ts sends it on every gateway call, so llm_call_usage rows of AI
Town dialogue carry the session id instead of no correlation at all (~1,100 unattributed calls/day
before this).
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_REPO = _SERVICE.parents[1]
_PATCH = _SERVICE / "patches" / "orion-town-correlation-id.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def _apply_order() -> list[str]:
    return [line.strip().strip('",') for line in _APPLY.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith('"orion-')]


def test_patch_is_registered_after_the_hub_embed_patch_that_also_touches_llm_ts():
    order = _apply_order()
    assert "orion-town-correlation-id.patch" in order
    assert order.index("orion-hub-embed.patch") < order.index("orion-town-correlation-id.patch")


def test_patch_reads_the_same_env_var_the_controller_writes():
    patch = _PATCH.read_text(encoding="utf-8")
    controller = (_REPO / "services/orion-embodiment/app/town_sessions.py").read_text(encoding="utf-8")
    assert 'CORRELATION_ENV = "ORION_TOWN_CORRELATION_ID"' in controller
    assert "+  const id = (process.env.ORION_TOWN_CORRELATION_ID ?? '').trim();" in patch
    assert "'X-Orion-Correlation-Id': id" in patch
    # Every gateway call builds its headers from AuthHeaders(): the header rides on all of them.
    assert "+  ...OrionCorrelationHeaders()," in patch
