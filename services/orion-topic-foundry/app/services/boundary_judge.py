from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.models import WindowingSpec
from app.settings import settings
from app.storage.repository import fetch_boundary_cache, insert_boundary_cache, utc_now
from app.services.llm_client import get_llm_client
from app.services.types import BoundaryContext, RowBlock


logger = logging.getLogger("topic-foundry.boundary-judge")


def judge_boundaries(
    *,
    blocks: List[RowBlock],
    candidate_indices: List[int],
    spec: WindowingSpec,
    context: BoundaryContext,
) -> Dict[int, Dict[str, Any]]:
    decisions: Dict[int, Dict[str, Any]] = {}
    for boundary_index in candidate_indices:
        prompt = _build_prompt(blocks, boundary_index, spec)
        context_hash = _hash_text(prompt)
        cache_key = _cache_key(spec, context, boundary_index, context_hash)

        cached = fetch_boundary_cache(cache_key)
        if cached and cached.get("decision"):
            decisions[boundary_index] = cached["decision"]
            _write_artifact(context.run_dir, cached["decision"], boundary_index, cached=True)
            continue

        if not settings.topic_foundry_llm_enable:
            decisions[boundary_index] = {
                "split": False,
                "confidence": 0.0,
                "reason": "llm disabled",
                "topic_left": "",
                "topic_right": "",
            }
            continue

        decision = _call_llm(prompt)
        if decision is None:
            continue

        decisions[boundary_index] = decision
        insert_boundary_cache(
            cache_key=cache_key,
            run_id=context.run_id,
            spec_hash=context.spec_hash,
            dataset_id=context.dataset_id,
            model_id=context.model_id,
            boundary_index=boundary_index,
            context_hash=context_hash,
            decision=decision,
        )
        _write_artifact(context.run_dir, decision, boundary_index, cached=False)

    return decisions


def _build_prompt(blocks: List[RowBlock], boundary_index: int, spec: WindowingSpec) -> str:
    start = max(0, boundary_index - spec.llm_boundary_context_blocks + 1)
    end = min(len(blocks), boundary_index + spec.llm_boundary_context_blocks + 1)
    selected = blocks[start:end]

    payload = {
        "boundary_index": boundary_index,
        "segments": [
            {
                "index": idx + start,
                "text": block.text,
                "timestamps": block.timestamps,
            }
            for idx, block in enumerate(selected)
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) > spec.llm_boundary_max_chars:
        serialized = serialized[: spec.llm_boundary_max_chars]

    return (
        "Decide if there is a topic boundary between the two blocks surrounding boundary_index.\n"
        "Return STRICT JSON only with keys: split, confidence, reason, topic_left, topic_right.\n"
        f"Input: {serialized}"
    )


def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        return get_llm_client().request_json(
            system_prompt="Return STRICT JSON only.",
            user_prompt=prompt,
            temperature=0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Boundary LLM judge failed: %s", exc)
        return None


def _cache_key(spec: WindowingSpec, context: BoundaryContext, boundary_index: int, context_hash: str) -> str:
    raw = f"{context.spec_hash}:{boundary_index}:{context_hash}:{settings.topic_foundry_llm_bus_route}:{spec.segmentation_mode}"
    return _hash_text(raw)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_artifact(run_dir: Optional[str], decision: Dict[str, Any], boundary_index: int, *, cached: bool) -> None:
    if not run_dir:
        return
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    path = run_path / "boundary_judgements.jsonl"
    payload = {
        "boundary_index": boundary_index,
        "decision": decision,
        "cached": cached,
        "created_at": utc_now().isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
