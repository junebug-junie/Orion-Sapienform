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
    # True for a node listed under the catalog's `retired_nodes:` section: a host that
    # no longer exists. It still resolves (so historical trace ids and aliases keep a
    # name), always with expected_online=False, but it is not part of the live roster
    # (`NodeCatalog.profiles`) and reducers drop it from live projections.
    retired: bool = False


class NodeCatalog:
    def __init__(
        self,
        profiles: dict[str, NodeProfile],
        aliases: dict[str, str],
        defaults: dict[str, Any],
        retired: dict[str, NodeProfile] | None = None,
    ) -> None:
        self.profiles = profiles
        self.aliases = aliases
        self.defaults = defaults
        # Retired hosts, keyed by canonical id. Kept apart from `profiles` so any
        # consumer that enumerates the live fleet never sees them.
        self.retired = dict(retired or {})

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

        retired: dict[str, NodeProfile] = {}
        for node_id, spec in (data.get("retired_nodes") or {}).items():
            spec = spec or {}
            canonical = str(node_id).strip().lower()
            if canonical in profiles:
                raise ValueError(
                    f"node {canonical!r} is listed under both nodes: and retired_nodes:"
                )
            retired[canonical] = NodeProfile(
                node_id=canonical,
                raw_node=None,
                role=str(spec.get("role") or "retired"),
                expected_online=False,
                capabilities={},
                known=True,
                retired=True,
            )
            aliases[canonical] = canonical
            for alias in spec.get("aliases") or []:
                aliases[str(alias).strip().lower()] = canonical

        return cls(profiles=profiles, aliases=aliases, defaults=defaults, retired=retired)

    def is_retired(self, raw_node: str | None) -> bool:
        """True when `raw_node` (canonical id or any alias) names a retired host."""
        key = str(raw_node or "").strip().lower()
        return self.aliases.get(key, key) in self.retired

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

        if canonical and canonical in self.retired:
            base = self.retired[canonical]
            return NodeProfile(
                node_id=base.node_id,
                raw_node=raw or None,
                role=base.role,
                expected_online=False,
                capabilities={},
                known=True,
                retired=True,
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
