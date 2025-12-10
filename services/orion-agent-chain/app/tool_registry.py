# services/orion-agent-chain/app/tool_registry.py

from __future__ import annotations

from pathlib import Path
from typing import List

from orion_cognition.packs_loader import PackManager  # you already have this
from orion_cognition.planner.loader import VerbRegistry
from .api import ToolDef   # the Pydantic model you already defined


class ToolRegistry:
    """
    Bridge between Orion's semantic verbs/packs and Agent Chain ToolDef objects.
    """

    def __init__(self, base_dir: Path):
        # base_dir is the root of the orion-cognition repo inside this container
        self.base_dir = base_dir
        self._pack_manager = PackManager(base_dir)
        self._verb_registry = VerbRegistry(base_dir / "verbs")
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._pack_manager.load_packs()
        self._verb_registry.load()
        self._loaded = True

    def tools_for_packs(self, pack_names: List[str]) -> List[ToolDef]:
        """
        Given a list of pack names (e.g. ["memory_pack", "executive_pack"]),
        return a list of ToolDef objects derived from the verbs in those packs.
        """
        self._ensure_loaded()

        tools: List[ToolDef] = []

        for pack_name in pack_names:
            pack = self._pack_manager.get_pack(pack_name)
            for verb_name in pack.verbs:
                verb_cfg = self._verb_registry.get(verb_name)

                # Basic/generic schemas for now; we can specialize per-verb later.
                input_schema = {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "context": {"type": "object"},
                    },
                    "required": ["text"],
                }
                output_schema = {
                    "type": "object",
                    "properties": {
                        "llm_output": {"type": "string"},
                        "structured": {"type": "object"},
                    },
                }

                tools.append(
                    ToolDef(
                        tool_id=verb_cfg.name,          # match Planner & Cortex
                        description=verb_cfg.description or f"Orion verb: {verb_cfg.name}",
                        input_schema=input_schema,
                        output_schema=output_schema,
                    )
                )

        return tools
