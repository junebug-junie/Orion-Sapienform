"""Bounded evidence pack builder for Mind LLM synthesis."""

from __future__ import annotations

import json
from typing import Any

from orion.mind.synthesis_v1 import MindEvidenceItemV1, MindEvidencePackV1

_SOURCE_TAG_LABELS = frozenset(
    {
        "identity_yaml",
        "social_bridge",
        "autonomy",
        "projection",
        "snapshot_source",
        "cognitive_projection",
        "substrate",
        "concept_induction",
        "recall_digest",
        "metacog",
        "tool_outcomes",
        "misc",
        "producer",
        "source",
    }
)


def _ref(prefix: str, idx: int) -> str:
    return f"{prefix}:{idx}"


def _append_item(
    items: list[MindEvidenceItemV1],
    *,
    source_kind: str,
    text: str,
    label: str | None = None,
    source_ref: str | None = None,
    item_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    trust_hint: str | None = None,
    freshness_hint: str | None = None,
) -> None:
    text_clean = (text or "").strip()
    if not text_clean:
        return
    items.append(
        MindEvidenceItemV1(
            evidence_ref=_ref(source_kind, len(items)),
            source_kind=source_kind,
            text=text_clean[:2000],
            label=(label or text_clean[:120]).strip()[:160] or None,
            source_ref=source_ref,
            item_id=item_id,
            metadata=dict(metadata or {}),
            trust_hint=trust_hint,
            freshness_hint=freshness_hint,
        )
    )


def _projection_items(projection: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    anchors = projection.get("anchors") if isinstance(projection.get("anchors"), dict) else {}
    out: list[dict[str, Any]] = []
    for anchor, payload in anchors.items():
        if not isinstance(payload, dict):
            continue
        for item in payload.get("items") or []:
            if isinstance(item, dict):
                enriched = dict(item)
                enriched.setdefault("anchor", anchor)
                out.append(enriched)
    out.sort(
        key=lambda item: (
            float(item.get("salience") or 0.0),
            float(item.get("confidence") or 0.0),
            str(item.get("label") or ""),
        ),
        reverse=True,
    )
    return out[:limit]


def _compact_autonomy(autonomy: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "attention_items",
        "candidate_impulses",
        "inhibited_impulses",
        "last_action_outcomes",
        "unknowns",
        "evidence_refs",
        "goal_headlines",
    )
    compact: dict[str, Any] = {}
    for key in keys:
        value = autonomy.get(key)
        if value:
            compact[key] = value
    return compact


def _compact_social(ctx: dict[str, Any]) -> dict[str, Any]:
    social_keys = (
        "social_inspection_snapshot",
        "social_stance_snapshot",
        "social_turn_policy",
        "social_peer_style_hint",
        "social_context_window",
    )
    compact: dict[str, Any] = {}
    for key in social_keys:
        value = ctx.get(key)
        if value is not None:
            compact[key] = value
    return compact


def build_evidence_pack(
    snapshot: dict[str, Any],
    *,
    max_messages: int = 8,
    max_recall_fragments: int = 8,
    max_projection_items: int = 16,
    max_total_chars: int = 12_000,
) -> MindEvidencePackV1:
    items: list[MindEvidenceItemV1] = []
    user_text = str(snapshot.get("user_text") or "").strip()
    if not user_text and isinstance(snapshot.get("messages_tail"), list) and snapshot["messages_tail"]:
        last = snapshot["messages_tail"][-1]
        if isinstance(last, dict):
            user_text = str(last.get("content") or last.get("text") or "").strip()
    if user_text:
        _append_item(
            items,
            source_kind="current_turn",
            text=user_text,
            label="current user turn",
            metadata={"role": "user"},
        )

    messages_tail = snapshot.get("messages_tail") if isinstance(snapshot.get("messages_tail"), list) else []
    for msg in messages_tail[-max_messages:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if not content:
            continue
        _append_item(
            items,
            source_kind="message_history",
            text=content,
            label=f"{role} message",
            metadata={"role": role},
        )

    facets = snapshot.get("facets") if isinstance(snapshot.get("facets"), dict) else {}
    recall = facets.get("recall_bundle") if isinstance(facets.get("recall_bundle"), dict) else {}
    if not recall and isinstance(snapshot.get("recall_bundle"), dict):
        recall = snapshot["recall_bundle"]
    fragments = recall.get("fragments") if isinstance(recall.get("fragments"), list) else []
    for frag in fragments[:max_recall_fragments]:
        if not isinstance(frag, dict):
            continue
        snippet = str(frag.get("snippet") or frag.get("text") or frag.get("summary") or "").strip()
        if not snippet:
            continue
        _append_item(
            items,
            source_kind="recall_fragment",
            text=snippet,
            label=str(frag.get("source") or "recall")[:80],
            source_ref=str(frag.get("doc_id") or frag.get("id") or "") or None,
            metadata={"recall_source": frag.get("source")},
        )

    projection = facets.get("cognitive_projection")
    if not isinstance(projection, dict):
        projection = facets.get("cognitive_projection_degraded")
    projection_count = 0
    if isinstance(projection, dict):
        for item in _projection_items(projection, limit=max_projection_items):
            label = str(item.get("label") or item.get("summary") or item.get("node_id") or "").strip()
            summary = str(item.get("summary") or label).strip()
            if not summary:
                continue
            projection_count += 1
            _append_item(
                items,
                source_kind="cognitive_projection",
                text=summary,
                label=label[:160] if label else None,
                source_ref=str(item.get("node_id") or item.get("item_id") or "") or None,
                item_id=str(item.get("item_id") or "") or None,
                metadata={
                    "anchor": item.get("anchor"),
                    "bucket": item.get("bucket"),
                    "salience": item.get("salience"),
                },
            )

    autonomy = facets.get("autonomy_compact")
    autonomy_fields = 0
    if isinstance(autonomy, dict):
        compact = _compact_autonomy(autonomy)
        autonomy_fields = len(compact)
        if compact:
            _append_item(
                items,
                source_kind="autonomy_compact",
                text=json.dumps(compact, default=str)[:1800],
                label="autonomy state compact",
                metadata={"background": False},
            )

    social = facets.get("social_compact")
    social_fields = 0
    if isinstance(social, dict) and social:
        social_fields = len(social)
        _append_item(
            items,
            source_kind="social_compact",
            text=json.dumps(social, default=str)[:1200],
            label="social context compact",
        )

    situation = facets.get("situation_compact")
    if isinstance(situation, dict) and situation:
        _append_item(
            items,
            source_kind="situation_compact",
            text=json.dumps(situation, default=str)[:1200],
            label="situation context compact",
        )

    identity = facets.get("identity_background")
    if isinstance(identity, dict) and identity:
        _append_item(
            items,
            source_kind="background_identity",
            text=json.dumps(identity, default=str)[:1200],
            label="identity kernel (background only)",
            metadata={"background_identity": True},
            trust_hint="low",
        )

    total_chars = sum(len(it.text) for it in items)
    while total_chars > max_total_chars and len(items) > 1:
        removed = items.pop()
        total_chars -= len(removed.text)

    return MindEvidencePackV1(
        current_user_text=user_text,
        items=items,
        diagnostics={
            "message_count": len(messages_tail),
            "recall_fragment_count": len(fragments),
            "projection_item_count": projection_count,
            "autonomy_fields_seen": autonomy_fields,
            "social_fields_seen": social_fields,
            "total_chars": total_chars,
            "truncated_for_budget": total_chars >= max_total_chars,
        },
    )


def evidence_refs_in_pack(pack: MindEvidencePackV1) -> set[str]:
    return {item.evidence_ref for item in pack.items}


_EVIDENCE_REF_KIND_ALIASES: dict[str, str] = {
    "projection": "cognitive_projection",
    "cognitive_projection_item": "cognitive_projection",
    "recall": "recall_fragment",
    "message": "message_history",
    "turn": "current_turn",
    "identity": "background_identity",
    "autonomy": "autonomy_compact",
    "social": "social_compact",
    "situation": "situation_compact",
}


def _pack_ref_indexes(pack: MindEvidencePackV1) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    dict[str, str],
]:
    by_kind: dict[str, list[str]] = {}
    by_source_ref: dict[str, str] = {}
    by_item_id: dict[str, str] = {}
    for item in pack.items:
        by_kind.setdefault(item.source_kind, []).append(item.evidence_ref)
        if item.source_ref:
            by_source_ref[str(item.source_ref).strip()] = item.evidence_ref
        if item.item_id:
            by_item_id[str(item.item_id).strip()] = item.evidence_ref
    return by_kind, by_source_ref, by_item_id


def resolve_evidence_ref_for_pack(
    ref: str,
    pack: MindEvidencePackV1,
    *,
    by_kind: dict[str, list[str]] | None = None,
    by_source_ref: dict[str, str] | None = None,
    by_item_id: dict[str, str] | None = None,
    valid: set[str] | None = None,
) -> str | None:
    """Resolve one ref to a pack evidence_ref, or None if unresolvable."""
    cleaned = str(ref or "").strip()
    if not cleaned:
        return None
    valid_refs = valid if valid is not None else evidence_refs_in_pack(pack)
    if cleaned in valid_refs:
        return cleaned
    if by_kind is None or by_source_ref is None or by_item_id is None:
        by_kind, by_source_ref, by_item_id = _pack_ref_indexes(pack)
    if cleaned in by_source_ref:
        return by_source_ref[cleaned]
    if cleaned in by_item_id:
        return by_item_id[cleaned]
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        mapped = _EVIDENCE_REF_KIND_ALIASES.get(prefix, prefix)
        candidate = f"{mapped}:{suffix}"
        if candidate in valid_refs:
            return candidate
        kind_refs = by_kind.get(mapped) or by_kind.get(prefix)
        if kind_refs:
            try:
                idx = int(suffix)
            except ValueError:
                idx = -1
            if 0 <= idx < len(kind_refs):
                return kind_refs[idx]
            if idx == 0:
                return kind_refs[0]
    if cleaned in by_kind and len(by_kind[cleaned]) == 1:
        return by_kind[cleaned][0]
    return None


def normalize_evidence_refs_for_pack(
    refs: list[str],
    pack: MindEvidencePackV1,
) -> list[str]:
    """Map common LLM ref mistakes to evidence_pack refs without weakening guardrails."""
    if not refs:
        return []
    valid = evidence_refs_in_pack(pack)
    by_kind, by_source_ref, by_item_id = _pack_ref_indexes(pack)
    out: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        resolved = resolve_evidence_ref_for_pack(
            str(raw or ""),
            pack,
            by_kind=by_kind,
            by_source_ref=by_source_ref,
            by_item_id=by_item_id,
            valid=valid,
        )
        if resolved and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def is_source_tag_label(label: str) -> bool:
    normalized = (label or "").strip().lower().replace(" ", "_")
    if not normalized:
        return True
    if normalized in _SOURCE_TAG_LABELS:
        return True
    for tag in _SOURCE_TAG_LABELS:
        if normalized == tag or normalized.endswith(f"_{tag}") or normalized.startswith(f"{tag}_"):
            return True
    return False
