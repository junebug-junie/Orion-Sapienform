from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    raw_node: str | None
    role: str
    expected_online: bool
    capabilities: dict[str, bool] = field(default_factory=dict)
    known: bool = True


class NodeCatalog:
    def __init__(
        self,
        profiles: dict[str, NodeProfile],
        aliases: dict[str, str],
        defaults: dict[str, Any],
    ) -> None:
        self.profiles = profiles
        self.aliases = aliases
        self.defaults = defaults

    @classmethod
    def load(cls, path: str | Path) -> "NodeCatalog":
        data = yaml.safe_load(Path(path).read_text()) or {}
        defaults = data.get("defaults") or {}
        profiles: dict[str, NodeProfile] = {}
        aliases: dict[str, str] = {}

        for node_id, spec in (data.get("nodes") or {}).items():
            canonical = str(node_id).strip().lower()
            role = str(spec.get("role") or defaults.get("role") or "unknown")
            expected_online = bool(
                spec.get("expected_online", defaults.get("expected_online", True))
            )
            capabilities = {
                str(k): bool(v) for k, v in (spec.get("capabilities") or {}).items()
            }

            profiles[canonical] = NodeProfile(
                node_id=canonical,
                raw_node=None,
                role=role,
                expected_online=expected_online,
                capabilities=capabilities,
                known=True,
            )

            aliases[canonical] = canonical
            for alias in spec.get("aliases") or []:
                aliases[str(alias).strip().lower()] = canonical

        return cls(profiles=profiles, aliases=aliases, defaults=defaults)

    def resolve(self, raw_node: str | None) -> NodeProfile:
        raw = str(raw_node or "").strip()
        key = raw.lower()
        canonical = self.aliases.get(key)

        if canonical and canonical in self.profiles:
            base = self.profiles[canonical]
            return NodeProfile(
                node_id=base.node_id,
                raw_node=raw or None,
                role=base.role,
                expected_online=base.expected_online,
                capabilities=dict(base.capabilities),
                known=True,
            )

        fallback_id = key or "unknown"
        return NodeProfile(
            node_id=fallback_id,
            raw_node=raw or None,
            role=str(self.defaults.get("role") or "unknown"),
            expected_online=bool(self.defaults.get("expected_online", True)),
            capabilities={
                str(k): bool(v)
                for k, v in (self.defaults.get("capabilities") or {}).items()
            },
            known=False,
        )
