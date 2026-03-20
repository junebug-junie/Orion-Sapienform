"""Pass 2: merged packs include delivery verbs (YAML-level, no jinja)."""

from __future__ import annotations

from pathlib import Path

import yaml
import orion
from orion.cognition.packs_loader import PackManager
from orion.cognition.runtime_pack_merge import ensure_delivery_pack_in_packs


def _verb_services(base: Path, verb_name: str) -> list:
    p = base / "verbs" / f"{verb_name}.yaml"
    data = yaml.safe_load(p.read_text())
    return list(data.get("services") or [])


def test_implementation_guide_packs_include_write_guide_and_finalize():
    base = Path(orion.__file__).resolve().parent / "cognition"
    packs = ensure_delivery_pack_in_packs(
        ["executive_pack"],
        output_mode="implementation_guide",
        user_text="",
    )
    pm = PackManager(base)
    pm.load_packs()
    llm_only = set()
    for pname in packs:
        for v in pm.get_pack(pname).verbs:
            if _verb_services(base, v) == ["LLMGatewayService"]:
                llm_only.add(v)
    assert "write_guide" in llm_only
    assert "finalize_response" in llm_only
    assert "answer_direct" in llm_only
