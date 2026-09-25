"""Fold one gateway window trace into per-serving-node states."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.llm_inference_projection import (
    ROLE_NODE_WINDOW,
    ROLE_WINDOW_COMPLETED,
    LlmInferenceNodeStateV1,
)

from .constants import KNOWN_FIELD_NODES, LLM_INFERENCE_SOURCE_SERVICE, LLM_INFERENCE_TRACE_PREFIX

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def parse_llm_inference_trace_id(trace_id: str) -> tuple[str, str] | None:
    """``llm_gateway.inference:<gateway_node>:<window_id>`` -> (gateway_node, window_id)."""
    if not trace_id or not trace_id.startswith(LLM_INFERENCE_TRACE_PREFIX):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2].strip()


def known_field_node(node_hint: str | None) -> str | None:
    """The gateway's node hint ("circe") if it names a real field-topology node, else None.

    The field digester creates whatever node id it is handed, so an unknown hint
    must stay unattributed rather than mint a phantom field node."""
    if not node_hint or not str(node_hint).strip():
        return None
    node = str(node_hint).strip().lower()
    return node if node in KNOWN_FIELD_NODES else None


def _parse_kv(summary: str) -> dict[str, str]:
    return {k.lower(): v.strip() for k, v in _KV_RE.findall(summary or "")}


def _int(kv: dict[str, str], key: str) -> int:
    try:
        return max(0, int(kv.get(key, "0") or 0))
    except ValueError:
        return 0


def _opt_int(kv: dict[str, str], key: str) -> int | None:
    raw = kv.get(key)
    if raw in (None, "", "none", "None"):
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _parse_classes(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (raw or "").split("|"):
        name, _, count = part.partition(":")
        name = name.strip()
        if not name or name == "none":
            continue
        try:
            out[name] = out.get(name, 0) + max(0, int(count or 0))
        except ValueError:
            continue
    return out


def _parse_labels(raw: str | None) -> list[str]:
    return [p.strip() for p in (raw or "").split("|") if p.strip() and p.strip() != "none"]


def inference_failure_pressure(*, served: int, upstream_failed: int) -> float | None:
    """Share of the calls the gateway actually sent upstream that came back without
    an answer. None when nothing went upstream: absence of traffic is not calm."""
    attempted = served + upstream_failed
    if attempted <= 0:
        return None
    return min(1.0, upstream_failed / attempted)


def extract_llm_inference_states_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
) -> tuple[dict[str, LlmInferenceNodeStateV1], int]:
    """Returns (states keyed by target_id, unattributed_calls)."""
    clock = _utc_now(now)
    if not events:
        raise ValueError("events must not be empty")
    trace_id = events[0].trace_id or ""
    parsed = parse_llm_inference_trace_id(trace_id)
    if not parsed:
        raise ValueError(f"invalid llm_inference trace_id: {trace_id}")
    gateway_node, window_id = parsed

    window_sec = 0.0
    node_rows: list[tuple[str, dict[str, str]]] = []
    for event in events:
        if event.provenance.source_service != LLM_INFERENCE_SOURCE_SERVICE:
            continue
        atom = event.atom
        if not atom:
            continue
        role = (atom.semantic_role or "").strip()
        kv = _parse_kv(atom.summary or "")
        if role == ROLE_WINDOW_COMPLETED:
            try:
                window_sec = float(kv.get("window_sec", "0") or 0.0)
            except ValueError:
                window_sec = 0.0
        elif role == ROLE_NODE_WINDOW:
            node_rows.append((event.event_id, kv))

    states: dict[str, LlmInferenceNodeStateV1] = {}
    unattributed = 0
    for event_id, kv in node_rows:
        node = known_field_node(kv.get("node"))
        if node is None:
            unattributed += _int(kv, "calls")
            continue
        target_id = f"llm_node:{node}"
        if target_id in states:
            # One atom per node per window is the wire contract; a duplicate is
            # a producer bug, and summing would double-count a replayed event.
            continue
        served = _int(kv, "served")
        failed = _int(kv, "upstream_failed")
        states[target_id] = LlmInferenceNodeStateV1(
            target_id=target_id,
            node_id=node,
            gateway_node=gateway_node,
            sample_window_id=window_id,
            source_trace_id=trace_id,
            calls=_int(kv, "calls"),
            served=served,
            upstream_failed=failed,
            refused=_int(kv, "refused"),
            request_invalid=_int(kv, "request_invalid"),
            prompt_tokens=_int(kv, "prompt_tokens"),
            completion_tokens=_int(kv, "completion_tokens"),
            latency_p50_ms=_opt_int(kv, "p50_ms"),
            latency_p95_ms=_opt_int(kv, "p95_ms"),
            served_by_labels=_parse_labels(kv.get("workers")),
            outcome_classes=_parse_classes(kv.get("classes")),
            inference_failure_pressure=inference_failure_pressure(served=served, upstream_failed=failed),
            evidence_event_ids=[event_id],
            observed_at=clock,
        )
    for state in states.values():
        state.window_sec = window_sec
    return states, unattributed
