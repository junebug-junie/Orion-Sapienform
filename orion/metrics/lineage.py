"""Resolver: joins the four metric-bearing registries into one URN space.

URN form:

    metric://<surface>/<producer>/<name>[#<field>]

Examples::

    metric://field_channel/orion-field-digester/cpu_pressure
    metric://inner_state/orion-self-state-runtime/self_state.v1#reasoning_pressure
    metric://organ_signal/biometrics/gpu_load#level
    metric://bus_channel/orion-substrate-runtime/orion:substrate:brain_frame

Nothing here is hand-authored. Each resolver reads an existing registry and
projects it. A registry entry that disappears takes its URNs with it, which is
the point -- a static mirror of these lists is exactly what goes stale.

Liveness is deliberately absent from this module. Verdicts are computed from
real sampled history (orion/field/channel_glossary.py::classify_channel_series),
never declared in a file. See the field glossary YAML header for the incident
that established that rule.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

GLOSSARY_PATH = REPO_ROOT / "config" / "field" / "field_channel_glossary.v1.yaml"
CHANNELS_PATH = REPO_ROOT / "orion" / "bus" / "channels.yaml"

FIELD_DIGESTER = "orion-field-digester"

# Canonical dimension keys carried by OrionSignalV1.dimensions. These are the
# "#field" half of a URN, never the scannable leaf token -- searching the repo
# for `level` or `confidence` would match everything and mean nothing.
_DIMENSION_FIELDS = frozenset(
    {
        "level",
        "trend",
        "volatility",
        "valence",
        "confidence",
        "arousal",
        "coherence",
        "novelty",
        "salience",
    }
)


@dataclass(frozen=True)
class MetricNode:
    """One addressable metric, resolved from an existing registry entry."""

    urn: str
    surface: str  # field_channel | inner_state | organ_signal | bus_channel
    producer_service: str
    name: str
    registry_source: str  # the file this was projected from
    metric_field: str | None = None
    upstream: tuple[str, ...] = ()
    declared_consumers: tuple[str, ...] = ()
    schema_id: str | None = None
    meaning: str | None = None
    notes: str | None = None

    @property
    def scan_token(self) -> str:
        """The string literal a consumer would use to read this metric.

        Metrics are string dict keys in this codebase, not symbols, so the
        token -- not the URN -- is what downstream discovery searches for.
        """
        return self.name


def _urn(surface: str, producer: str, name: str, metric_field: str | None = None) -> str:
    base = f"metric://{surface}/{producer}/{name}"
    return f"{base}#{metric_field}" if metric_field else base


# --------------------------------------------------------------------------
# field channels
# --------------------------------------------------------------------------


def resolve_field_channels(path: Path | None = None) -> list[MetricNode]:
    """Project config/field/field_channel_glossary.v1.yaml."""
    target = path or GLOSSARY_PATH
    if not target.exists():
        return []
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    source = str(target.relative_to(REPO_ROOT))

    nodes: list[MetricNode] = []
    for entry in raw.get("channels", []):
        name = entry["channel"]
        # self_state_dimension / evidence_dimension point at what this channel
        # FEEDS, so they are recorded here and inverted into `upstream` on the
        # inner-state side by build_graph().
        feeds = tuple(
            v
            for v in (entry.get("self_state_dimension"), entry.get("evidence_dimension"))
            if v
        )
        nodes.append(
            MetricNode(
                urn=_urn("field_channel", FIELD_DIGESTER, name),
                surface="field_channel",
                producer_service=FIELD_DIGESTER,
                name=name,
                registry_source=source,
                meaning=entry.get("meaning"),
                notes=f"category={entry.get('category')} level={','.join(entry.get('level', []))}",
                declared_consumers=feeds,
            )
        )
    return nodes


# --------------------------------------------------------------------------
# inner-state signals
# --------------------------------------------------------------------------


def _float_fields(model: Any) -> list[str]:
    """Scalar float fields on a pydantic model -- the addressable metrics.

    dict[str, float] fields (e.g. SelfStateV1-style dimension bags) are not
    statically enumerable and are skipped here; their keys surface through the
    field-channel and organ surfaces instead.
    """
    fields = getattr(model, "model_fields", None)
    if not fields:
        return []

    out: list[str] = []
    for fname, finfo in fields.items():
        ann = getattr(finfo, "annotation", None)
        if _is_float_like(ann):
            out.append(fname)
    return out


def _is_float_like(ann: Any) -> bool:
    if ann is float:
        return True
    origin = typing.get_origin(ann)
    if origin is typing.Union or (origin is not None and str(origin) == "types.UnionType"):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        return len(args) == 1 and args[0] is float
    return False


def resolve_inner_state() -> list[MetricNode]:
    """Project orion/inner_state_registry.py::REGISTRY."""
    # Import errors deliberately propagate: a registry that no longer imports
    # is rot, and returning [] here would hide it behind a plausible-looking
    # zero. Same reasoning as check_inner_state_registry.py's rot check.
    from orion.inner_state_registry import REGISTRY

    source = "orion/inner_state_registry.py"
    nodes: list[MetricNode] = []
    for sig in REGISTRY:
        producer = sig.producer_service
        schema = sig.schema
        schema_id = schema.__name__ if schema is not None else None
        consumers = tuple(sig.cognition_consumers)

        # The signal itself is addressable even when it has no enumerable
        # float fields (schema=None entries are real registry entries).
        nodes.append(
            MetricNode(
                urn=_urn("inner_state", producer, sig.signal_id),
                surface="inner_state",
                producer_service=producer,
                name=sig.signal_id,
                registry_source=source,
                schema_id=schema_id,
                declared_consumers=consumers,
                notes=f"cadence={sig.cadence.value} composition={sig.composition_status.value}",
            )
        )
        for fname in _float_fields(schema):
            nodes.append(
                MetricNode(
                    urn=_urn("inner_state", producer, sig.signal_id, fname),
                    surface="inner_state",
                    producer_service=producer,
                    name=fname,
                    metric_field=fname,
                    registry_source=source,
                    schema_id=schema_id,
                    declared_consumers=consumers,
                    upstream=(_urn("inner_state", producer, sig.signal_id),),
                    notes=f"scalar field on {schema_id}",
                )
            )
    return nodes


# --------------------------------------------------------------------------
# organ signals
# --------------------------------------------------------------------------


def resolve_organ_signals() -> list[MetricNode]:
    """Project orion/signals/registry.py::ORGAN_REGISTRY."""
    # Import errors propagate -- see resolve_inner_state().
    from orion.signals.registry import ORGAN_REGISTRY

    source = "orion/signals/registry.py"
    nodes: list[MetricNode] = []
    for organ_id, entry in ORGAN_REGISTRY.items():
        parents = tuple(
            _urn("organ_signal", p, p) for p in entry.causal_parent_organs
        )
        for kind in entry.signal_kinds:
            for dim in entry.canonical_dimensions:
                nodes.append(
                    MetricNode(
                        urn=_urn("organ_signal", organ_id, kind, dim),
                        surface="organ_signal",
                        producer_service=entry.service,
                        name=kind,
                        metric_field=dim,
                        registry_source=source,
                        upstream=parents,
                        notes=f"organ={organ_id} class={entry.organ_class.value}",
                    )
                )
    return nodes


# --------------------------------------------------------------------------
# bus channels
# --------------------------------------------------------------------------


def resolve_bus_channels(path: Path | None = None) -> list[MetricNode]:
    """Project orion/bus/channels.yaml."""
    target = path or CHANNELS_PATH
    if not target.exists():
        return []
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    source = str(target.relative_to(REPO_ROOT))

    nodes: list[MetricNode] = []
    for entry in raw.get("channels", []):
        name = entry["name"]
        producers = entry.get("producer_services") or ["unknown"]
        nodes.append(
            MetricNode(
                urn=_urn("bus_channel", producers[0], name),
                surface="bus_channel",
                producer_service=producers[0],
                name=name,
                registry_source=source,
                schema_id=entry.get("schema_id"),
                declared_consumers=tuple(entry.get("consumer_services") or ()),
                notes=f"kind={entry.get('kind')} stability={entry.get('stability')}",
            )
        )
    return nodes


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


@dataclass
class MetricGraph:
    nodes: dict[str, MetricNode] = field(default_factory=dict)

    def by_surface(self, surface: str) -> list[MetricNode]:
        return [n for n in self.nodes.values() if n.surface == surface]

    def by_token(self, token: str) -> list[MetricNode]:
        return [n for n in self.nodes.values() if n.scan_token == token]

    def scan_tokens(self) -> dict[str, list[MetricNode]]:
        """token -> nodes. The search space for downstream discovery.

        Pure canonical dimension names are excluded: they are the `#field`
        half of a URN and would match nearly every dict access in the repo.
        """
        out: dict[str, list[MetricNode]] = {}
        for node in self.nodes.values():
            token = node.scan_token
            if token in _DIMENSION_FIELDS:
                continue
            out.setdefault(token, []).append(node)
        return out

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.surface] = counts.get(node.surface, 0) + 1
        return counts


def build_graph() -> MetricGraph:
    """Resolve every registry into one URN space.

    Later resolvers do not overwrite earlier ones; a duplicate URN keeps the
    first-registered node so the join is deterministic regardless of dict
    ordering.
    """
    graph = MetricGraph()
    for resolver in (
        resolve_field_channels,
        resolve_inner_state,
        resolve_organ_signals,
        resolve_bus_channels,
    ):
        for node in resolver():
            graph.nodes.setdefault(node.urn, node)
    return graph


def to_dict(node: MetricNode) -> dict[str, Any]:
    return {
        "urn": node.urn,
        "surface": node.surface,
        "producer_service": node.producer_service,
        "name": node.name,
        "field": node.metric_field,
        "registry_source": node.registry_source,
        "schema_id": node.schema_id,
        "meaning": node.meaning,
        "upstream": list(node.upstream),
        "declared_consumers": list(node.declared_consumers),
        "notes": node.notes,
    }
