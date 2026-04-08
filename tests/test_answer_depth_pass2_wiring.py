"""Pass 2 wiring tests (root testpaths)."""

from __future__ import annotations

from pathlib import Path

import yaml
import orion
from orion.cognition.runtime_pack_merge import ensure_delivery_pack_in_packs
from orion.cognition.packs_loader import PackManager


def test_delivery_pack_for_discord_deploy_prompt():
    packs = ensure_delivery_pack_in_packs(
        ["executive_pack", "memory_pack"],
        output_mode=None,
        user_text="Please provide instructions on how to deploy you onto Discord.",
    )
    assert "delivery_pack" in packs


def test_resolved_llm_tools_include_delivery_verbs():
    base = Path(orion.__file__).resolve().parent / "cognition"

    def _svc(v: str) -> list:
        data = yaml.safe_load((base / "verbs" / f"{v}.yaml").read_text())
        return list(data.get("services") or [])

    packs = ensure_delivery_pack_in_packs(
        ["executive_pack"],
        output_mode="implementation_guide",
        user_text="",
    )
    pm = PackManager(base)
    pm.load_packs()
    ids = set()
    for pname in packs:
        for v in pm.get_pack(pname).verbs:
            if _svc(v) == ["LLMGatewayService"]:
                ids.add(v)
    assert {"write_guide", "finalize_response", "plan_action"} <= ids
