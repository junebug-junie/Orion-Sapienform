"""Recall bundle adapter — snapshot_ephemeral tier.

Maps ``ctx["recall_bundle"]`` fragments into substrate nodes.  No network call;
always-fresh from ctx.  pull_on_cold=False (registered with TTL=0).

Fragment source tags:
  journal / metacog → ConceptNodeV1  (label = snippet)
  tension            → TensionNodeV1
  dream              → EventNodeV1
  sql_timeline       → EventNodeV1 (conservative episodic mapping)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EventNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    TensionNodeV1,
)

from orion.substrate.adapters._common import make_temporal

_TIER_RANK = 4  # snapshot_ephemeral
_ANCHOR_DEFAULT = "orion"


def _make_prov(*, source_tag: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind=f"recall.{source_tag}",
        source_channel="ctx.recall_bundle",
        producer="recall_adapter",
        tier_rank=_TIER_RANK,
    )


def _tag_list(frag: dict[str, Any]) -> list[str]:
    raw = frag.get("tags")
    if isinstance(raw, list):
        return [str(t).lower() for t in raw if t]
    return []


def _meta_dict(frag: dict[str, Any]) -> dict[str, Any]:
    for key in ("metadata", "meta"):
        val = frag.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _anchor_hint(*parts: str) -> str:
    joined = " ".join(p for p in parts if p).lower()
    if "relationship" in joined:
        return "relationship"
    if "juniper" in joined or joined.strip() in {"user", "human"}:
        return "juniper"
    return _ANCHOR_DEFAULT


def _anchor_for_fragment(frag: dict[str, Any]) -> str:
    tags = _tag_list(frag)
    meta = _meta_dict(frag)
    return _anchor_hint(
        str(frag.get("subject") or ""),
        str(frag.get("anchor") or ""),
        str(meta.get("subject") or ""),
        str(meta.get("anchor") or ""),
        " ".join(tags),
        str(frag.get("source_ref") or ""),
    )


def _fragment_metadata(frag: dict[str, Any], *, source: str, snippet: str) -> dict[str, Any]:
    meta = _meta_dict(frag)
    out: dict[str, Any] = {
        "recall_source": source,
        "original_source": source,
        "snippet": snippet,
    }
    if frag.get("source_ref"):
        out["source_ref"] = frag.get("source_ref")
    if tags := _tag_list(frag):
        out["tags"] = tags
    if frag.get("session_id"):
        out["session_id"] = frag.get("session_id")
    if frag.get("node_id"):
        out["node_id"] = frag.get("node_id")
    if meta.get("turn_effect_delta") is not None:
        out["turn_effect_delta"] = meta.get("turn_effect_delta")
    elif frag.get("turn_effect_delta") is not None:
        out["turn_effect_delta"] = frag.get("turn_effect_delta")
    return out


def _map_sql_timeline_fragment(
    frag: dict[str, Any],
    *,
    snippet: str,
    temporal: Any,
) -> tuple[Any | None, str]:
    """Map sql_timeline fragment → conservative substrate node."""
    source = "sql_timeline"
    tags = _tag_list(frag)
    source_ref = str(frag.get("source_ref") or "").lower()
    anchor = _anchor_for_fragment(frag)
    metadata = _fragment_metadata(frag, source=source, snippet=snippet)

    tension_explicit = any(t in tags for t in ("tension", "pressure")) or "tension" in source_ref
    if tension_explicit and any(t in tags for t in ("tension", "pressure", "turn_effect")):
        return (
            TensionNodeV1(
                anchor_scope=anchor,
                tension_kind="recall_timeline",
                intensity=0.5,
                temporal=temporal,
                provenance=_make_prov(source_tag="sql_timeline.tension"),
                signals=SubstrateSignalBundleV1(confidence=0.55, salience=0.45),
                metadata={**metadata, "label": snippet},
            ),
            "tension",
        )

    if "chat_timeline" in tags or source_ref in ("chat_history_log", "chat_history"):
        return (
            EventNodeV1(
                anchor_scope=anchor,
                event_type="chat_timeline",
                summary=snippet,
                temporal=temporal,
                provenance=_make_prov(source_tag="sql_timeline.chat"),
                signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.32),
                metadata=metadata,
            ),
            "event.chat_timeline",
        )

    if source_ref == "collapse_mirror" or "collapse_mirror" in tags:
        return (
            EventNodeV1(
                anchor_scope=anchor,
                event_type="collapse_mirror",
                summary=snippet,
                temporal=temporal,
                provenance=_make_prov(source_tag="sql_timeline.collapse_mirror"),
                signals=SubstrateSignalBundleV1(confidence=0.55, salience=0.4),
                metadata=metadata,
            ),
            "event.collapse_mirror",
        )

    return (
        EventNodeV1(
            anchor_scope=anchor,
            event_type="timeline_memory",
            summary=snippet,
            temporal=temporal,
            provenance=_make_prov(source_tag="sql_timeline"),
            signals=SubstrateSignalBundleV1(confidence=0.48, salience=0.3),
            metadata=metadata,
        ),
        "event.timeline_memory",
    )


def map_recall_bundle_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ctx recall_bundle fragments → substrate nodes (snapshot_ephemeral)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    recall_bundle = ctx.get("recall_bundle") or {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle, dict) else []

    if not isinstance(fragments, list) or not fragments:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    nodes: list[Any] = []
    dropped_counts: dict[str, int] = {}
    mapped_counts_by_source: dict[str, int] = {}
    mapped_counts_by_node_type: dict[str, int] = {}
    original_sources_seen: set[str] = set()
    recall_fragments_seen = 0
    recall_fragments_mapped = 0

    for frag in fragments[:24]:
        recall_fragments_seen += 1
        if not isinstance(frag, dict):
            dropped_counts["non_dict_fragment"] = dropped_counts.get("non_dict_fragment", 0) + 1
            continue
        source = str(frag.get("source") or "").lower()
        original_sources_seen.add(source or "unknown")
        snippet = str(frag.get("snippet") or frag.get("text") or "").strip()[:200]
        if not snippet:
            dropped_counts["missing_snippet"] = dropped_counts.get("missing_snippet", 0) + 1
            continue

        anchor = _anchor_for_fragment(frag)
        mapped_node: Any | None = None
        node_type_key = ""

        if "journal" in source or "metacog" in source:
            mapped_node = ConceptNodeV1(
                anchor_scope=anchor,
                label=snippet,
                temporal=temporal,
                provenance=_make_prov(source_tag="journal" if "journal" in source else "metacog"),
                signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.4),
                metadata=_fragment_metadata(frag, source=source, snippet=snippet),
            )
            node_type_key = "concept"
        elif "tension" in source and "sql_timeline" not in source:
            mapped_node = TensionNodeV1(
                anchor_scope=anchor,
                tension_kind="recall",
                intensity=0.5,
                temporal=temporal,
                provenance=_make_prov(source_tag="tension"),
                signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.5),
                metadata={**_fragment_metadata(frag, source=source, snippet=snippet), "label": snippet},
            )
            node_type_key = "tension"
        elif "dream" in source:
            mapped_node = EventNodeV1(
                anchor_scope=anchor,
                event_type="dream",
                summary=snippet,
                temporal=temporal,
                provenance=_make_prov(source_tag="dream"),
                signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.3),
                metadata=_fragment_metadata(frag, source=source, snippet=snippet),
            )
            node_type_key = "event.dream"
        elif source == "sql_timeline" or "sql_timeline" in source:
            mapped_node, node_type_key = _map_sql_timeline_fragment(frag, snippet=snippet, temporal=temporal)
        else:
            dropped_counts["unsupported_source"] = dropped_counts.get("unsupported_source", 0) + 1
            continue

        if mapped_node is None:
            dropped_counts["map_failed"] = dropped_counts.get("map_failed", 0) + 1
            continue

        nodes.append(mapped_node)
        recall_fragments_mapped += 1
        mapped_counts_by_source[source or "unknown"] = mapped_counts_by_source.get(source or "unknown", 0) + 1
        mapped_counts_by_node_type[node_type_key] = mapped_counts_by_node_type.get(node_type_key, 0) + 1

    if not nodes:
        return None

    adapter_diag = {
        "recall_fragments_seen": recall_fragments_seen,
        "recall_fragments_mapped": recall_fragments_mapped,
        "recall_fragments_dropped": recall_fragments_seen - recall_fragments_mapped,
        "dropped_counts_by_reason": dropped_counts,
        "mapped_counts_by_source": mapped_counts_by_source,
        "mapped_counts_by_node_type": mapped_counts_by_node_type,
        "original_sources_seen": sorted(original_sources_seen),
    }
    if nodes and hasattr(nodes[0], "metadata"):
        first_meta = dict(getattr(nodes[0], "metadata", None) or {})
        first_meta["recall_adapter"] = adapter_diag
        nodes[0] = nodes[0].model_copy(update={"metadata": first_meta})

    return SubstrateGraphRecordV1(anchor_scope=_ANCHOR_DEFAULT, nodes=nodes)
