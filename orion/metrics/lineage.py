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

import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

GLOSSARY_PATH = REPO_ROOT / "config" / "field" / "field_channel_glossary.v1.yaml"
CHANNELS_PATH = REPO_ROOT / "orion" / "bus" / "channels.yaml"

FIELD_DIGESTER = "orion-field-digester"

# NOTE: an earlier version excluded canonical dimension names (`level`,
# `confidence`, `valence`, ...) from the scan-token set by NAME, to stop them
# matching every dict access in the repo. That was both unnecessary and
# actively wrong:
#
#   - unnecessary, because scan_token is `node.name` (a field channel, a
#     signal_kind, or a scalar field name) and never `node.metric_field`, so
#     a bare dimension name is only ever a token when some registry really
#     declares a metric by that name;
#   - wrong, because it then deleted those real metrics. The glossary channel
#     `confidence` and five real inner-state scalars (autonomy_state_v2
#     #confidence, mood_arc_corpus.v1 #valence/#coherence/#novelty,
#     drive_state.v1 #confidence) got no blast radius AND never appeared in
#     the orphan list -- invisible in both outputs.
#
# Role, not name, decides: the `#field` half of a URN is already excluded
# structurally by scan_token's definition.


@dataclass(frozen=True)
class MetricNode:
    """One addressable metric, resolved from an existing registry entry."""

    urn: str
    surface: str  # field_channel | inner_state | organ_signal | bus_channel
    producer_service: str
    name: str
    registry_source: str  # the file this was projected from
    metric_field: str | None = None
    # Parent URNs. Every entry MUST resolve to a real node in the graph --
    # see test_no_dangling_upstream_urns.
    upstream: tuple[str, ...] = ()
    # Organ-level causal parents (organ_ids, NOT URNs). Kept separate because
    # an organ is not itself an addressable metric; synthesising URNs for them
    # produced 14 permanently-dangling parents.
    upstream_organs: tuple[str, ...] = ()
    # Declared DOWNSTREAM consumer *services / call sites*, as recorded by the
    # source registry. Service-shaped only.
    declared_consumers: tuple[str, ...] = ()
    # Field channels only: the self-state/evidence dimension this channel
    # FEEDS. Deliberately not folded into declared_consumers -- these are
    # dimension names, not services, and printing both under one label gave a
    # reader two incompatible meanings from one field.
    feeds_dimensions: tuple[str, ...] = ()
    # All producer services, when a channel has more than one.
    all_producers: tuple[str, ...] = ()
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
        channel = entry["channel"]
        # Version 2 (2026-09-03): a `node:` qualifier disambiguates entries
        # that share a channel name but mean different things per node (e.g.
        # bus_synaptic vs. vision prediction_error). Fold it into `name` so
        # each qualified entry gets its own URN -- without this, two
        # qualified entries sharing `channel` would build the identical
        # `metric://field_channel/orion-field-digester/<channel>` URN and
        # silently overwrite each other in build_graph()'s dict[urn, node].
        # The bare (unqualified) entry's name/URN is unchanged.
        node_qualifier = entry.get("node")
        name = f"{node_qualifier}.{channel}" if node_qualifier else channel
        # self_state_dimension / evidence_dimension name the dimension this
        # channel FEEDS. They are recorded as feeds_dimensions and are NOT
        # inverted into upstream anywhere -- an earlier comment here claimed
        # build_graph() performed that inversion; it never did.
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
                feeds_dimensions=feeds,
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
    """float, Optional[float], or `float | None`.

    The PEP-604 (`float | None`) branch must compare against types.UnionType
    itself -- str(types.UnionType) is "<class 'types.UnionType'>", never
    "types.UnionType", so a string comparison silently never fires. That
    dropped every `X | None` metric from the URN space, including
    FieldStateV1.recent_perturbation_zscore, and orion/schemas/ uses the
    `X | None` style predominantly.
    """
    if ann is float:
        return True
    origin = typing.get_origin(ann)
    if origin is typing.Union or origin is types.UnionType:
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
        # Organ ids, not synthesised URNs: metric://organ_signal/<p>/<p> could
        # never match a real node (whose form is /<organ>/<kind>#<dim>), so it
        # left 14 permanently-dangling upstream edges across 252 organ nodes.
        parents = tuple(entry.causal_parent_organs)
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
                        upstream_organs=parents,
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
        # 41 channels have multiple producers. Sorted, so the URN -- an
        # addressing identity -- does not silently change when someone
        # reorders a YAML list cosmetically. The full set is kept on
        # all_producers rather than dropped.
        producers = sorted(entry.get("producer_services") or ["unknown"])
        nodes.append(
            MetricNode(
                urn=_urn("bus_channel", producers[0], name),
                surface="bus_channel",
                producer_service=producers[0],
                all_producers=tuple(producers),
                name=name,
                registry_source=source,
                schema_id=entry.get("schema_id"),
                declared_consumers=tuple(entry.get("consumer_services") or ()),
                notes=f"kind={entry.get('kind')} stability={entry.get('stability')}",
            )
        )
    return nodes


# --------------------------------------------------------------------------
# brain regions (Self tab's Self-Observability EKG -- orion/schemas/brain_frame.py)
# --------------------------------------------------------------------------

SUBSTRATE_RUNTIME = "orion-substrate-runtime"

# BrainRegionV1.dimension's full Literal set (orion/schemas/brain_frame.py) --
# a closed, small enumeration, so a static table here is complete, not an
# approximation of something larger. Producer is the service that actually
# COMPUTES the region's value inside assemble_brain_frame()
# (services/orion-substrate-runtime/app/brain_frame_producer.py), which for
# 5 of the 6 dimensions is the same service that owns the whole brain-frame
# tick. `field_anomaly` is the one exception: its true producer is
# orion-field-digester's mood-arc encoder (app/anomaly_scorer.py), relayed
# onto the bus and merely cached/assembled by substrate-runtime -- see its
# `upstream` link to the real bus_channel node in resolve_brain_regions()
# below.
_BRAIN_REGION_PRODUCERS: dict[str, str] = {
    "node_kind": SUBSTRATE_RUNTIME,
    "lane": SUBSTRATE_RUNTIME,
    "self_state": SUBSTRATE_RUNTIME,
    "lattice_layer": SUBSTRATE_RUNTIME,
    "honesty_metrics": SUBSTRATE_RUNTIME,
    "field_anomaly": FIELD_DIGESTER,
}


def resolve_brain_regions() -> list[MetricNode]:
    """Project BrainRegionV1.dimension.

    Added 2026-09-04 to close a real gap: this URN space already covered
    field channels, inner-state signals, organ signals, and bus channels,
    but nothing named which service backs an EKG *region* -- the Self tab's
    Self-Observability panel had no way to answer "what produced this bar"
    for any of its 6 dimensions.

    `declared_consumers` names `orion-hub` (resolves to `services/orion-hub/`
    via the bare-service-name form `_resolve_consumer_path()` already
    supports) rather than a dotted Python callable: the real consumer is
    `static/js/self-brain.js`, and this repo's AST-based consumer scan
    (`orion.metrics.consumers`) covers Python only -- the bare-service form
    is the honest way to say "the Hub renders this" without fabricating a
    module path that doesn't exist.
    """
    source = "orion/schemas/brain_frame.py"
    nodes: list[MetricNode] = []
    for dimension, producer in _BRAIN_REGION_PRODUCERS.items():
        upstream: tuple[str, ...] = ()
        if dimension == "field_anomaly":
            upstream = (
                _urn("bus_channel", FIELD_DIGESTER, "orion:field_channel:anomaly_score"),
            )
        nodes.append(
            MetricNode(
                urn=_urn("brain_region", producer, dimension),
                surface="brain_region",
                producer_service=producer,
                name=dimension,
                registry_source=source,
                declared_consumers=("orion-hub",),
                upstream=upstream,
                notes="Self tab / Self-Observability EKG region dimension",
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

        No name-based exclusions: scan_token is structurally the `name` half
        of a URN, never the `#field` half, so dimension names only appear here
        when a registry genuinely declares a metric by that name.
        """
        out: dict[str, list[MetricNode]] = {}
        for node in self.nodes.values():
            out.setdefault(node.scan_token, []).append(node)
        return out

    def registry_sources_for(self, token: str) -> set[str]:
        """Files `token` is declared in -- never counted as its consumers."""
        return {n.registry_source for n in self.nodes.values() if n.scan_token == token}

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
        resolve_brain_regions,
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
        "upstream_organs": list(node.upstream_organs),
        "declared_consumers": list(node.declared_consumers),
        "feeds_dimensions": list(node.feeds_dimensions),
        "all_producers": list(node.all_producers),
        "notes": node.notes,
    }
