"""doc_semantic_drift producer: cold-start-sha polling loop (same pattern as
git_delta_loop) over this repo's own real ``*.md`` changes, scoring each
changed doc file's drift via a diff-scoped embedding-diff -- confirmed by real
replay (docs/superpowers/pr-reports/2026-08-11-doc-semantic-drift-diff-scoped-
embedding.md) to separate trivial doc edits from real ones.

Gets its embeddings from orion-vector-host's real, already-live bus contract
(``EmbeddingGenerateV1``/``EmbeddingResultV1`` on ``orion:embedding:generate``)
rather than the offline calibration script's direct-container access, which
was only ever a debug/calibration shortcut, not a production-safe path.
Confirmed live (code read of ``services/orion-vector-host/app/main.py``'s
``_handle_embedding_request()``): every real embedding request over this
contract unconditionally persists the embedded text as a vector-store
document -- no opt-out exists today. Rather than treat that as a blocker,
Juniper made an explicit call (2026-08-11, via AskUserQuestion): use the real
contract as-is, scoped to this producer's own ``doc_semantic_drift``
collection so these hunk-diff embeddings aren't commingled with chat/social
memory. This mirrors every other existing caller of this same contract
(orion-chat-memory, orion-hub, orion-cortex-exec), which already accept the
same persistence side effect.

RPC request/reply pattern mirrors
``services/orion-chat-memory/app/main.py::_request_embedding_bus()`` --
forked dedicated RPC bus client (``fork_rpc_client``) so replies aren't stolen
by any other subscribe loop on the shared producer bus connection.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.rpc_fork import fork_rpc_client
from orion.schemas.doc_semantic_drift import DocSemanticDriftV1
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1
from orion.structural_mass.doc_semantic_drift import DocHunkChange, doc_semantic_drift_changes

logger = logging.getLogger("orion.cocreation_signals.doc_semantic_drift")


def _current_head_sha(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


async def _request_embedding(
    rpc_bus: OrionBusAsync,
    *,
    request_channel: str,
    result_channel_prefix: str,
    source: ServiceRef,
    doc_id: str,
    text: str,
    collection: str,
    timeout_sec: float,
) -> EmbeddingResultV1 | None:
    """Real RPC call over orion-vector-host's already-live bus contract --
    mirrors orion-chat-memory's own ``_request_embedding_bus()``. Returns
    None on any failure (timeout, decode failure, missing embedding) so the
    caller can publish a real ``diff_scoped_embedding_diff=None`` rather than
    fabricate a 0.0 (CLAUDE.md's "no empty-shell cognition" gate)."""
    reply_channel = f"{result_channel_prefix}:{uuid4()}"
    envelope = BaseEnvelope(
        kind="embedding.generate.v1",
        source=source,
        reply_to=reply_channel,
        payload=EmbeddingGenerateV1(
            doc_id=doc_id,
            text=text,
            collection=collection,
        ).model_dump(mode="json"),
    )
    try:
        msg = await rpc_bus.rpc_request(
            request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec
        )
        decoded = rpc_bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            logger.warning("cocreation_doc_semantic_drift_embed_decode_failed doc_id=%s error=%s", doc_id, decoded.error)
            return None
        payload = decoded.envelope.payload
        payload_dict = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        if not isinstance(payload_dict, dict):
            return None
        payload_dict.pop("latent", None)
        return EmbeddingResultV1.model_validate(payload_dict)
    except Exception:
        logger.warning("cocreation_doc_semantic_drift_embed_request_failed doc_id=%s", doc_id, exc_info=True)
        return None


async def _score_change(
    rpc_bus: OrionBusAsync,
    *,
    request_channel: str,
    result_channel_prefix: str,
    source: ServiceRef,
    change: DocHunkChange,
    collection: str,
    embed_timeout_sec: float,
    truncation_char_threshold: int,
) -> DocSemanticDriftV1:
    removed_result, added_result = await asyncio.gather(
        _request_embedding(
            rpc_bus,
            request_channel=request_channel,
            result_channel_prefix=result_channel_prefix,
            source=source,
            doc_id=f"doc_semantic_drift:{change.sha}:{change.path}:removed",
            text=change.hunk_removed,
            collection=collection,
            timeout_sec=embed_timeout_sec,
        ),
        _request_embedding(
            rpc_bus,
            request_channel=request_channel,
            result_channel_prefix=result_channel_prefix,
            source=source,
            doc_id=f"doc_semantic_drift:{change.sha}:{change.path}:added",
            text=change.hunk_added,
            collection=collection,
            timeout_sec=embed_timeout_sec,
        ),
    )

    diff: float | None = None
    if removed_result is not None and added_result is not None:
        similarity = _cosine_similarity(removed_result.embedding, added_result.embedding)
        if similarity is not None:
            diff = 1.0 - similarity

    return DocSemanticDriftV1(
        observed_at=datetime.now(timezone.utc),
        sha=change.sha,
        path=change.path,
        commit_prefix=change.commit_prefix,
        diff_scoped_embedding_diff=diff,
        possibly_truncated=(
            len(change.hunk_removed) > truncation_char_threshold
            or len(change.hunk_added) > truncation_char_threshold
        ),
        hunk_removed_len_chars=len(change.hunk_removed),
        hunk_added_len_chars=len(change.hunk_added),
    )


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, event: DocSemanticDriftV1) -> bool:
    """Same True/False publish contract as git_delta_loop's own
    ``_publish()`` -- False only on a real transient publish failure, so the
    caller knows not to advance ``last_sha`` past this change."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_doc_semantic_drift_publish_skipped_bus_disabled sha=%s path=%s", event.sha, event.path)
        return True
    envelope = BaseEnvelope(
        kind="substrate.doc_semantic_drift.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_doc_semantic_drift_published sha=%s path=%s diff=%s possibly_truncated=%s",
            event.sha, event.path, event.diff_scoped_embedding_diff, event.possibly_truncated,
        )
        return True
    except Exception:
        logger.exception("cocreation_doc_semantic_drift_publish_failed sha=%s path=%s", event.sha, event.path)
        return False


async def doc_semantic_drift_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    repo_path: str,
    embed_request_channel: str,
    embed_collection: str,
    embed_timeout_sec: float,
    truncation_char_threshold: int,
    poll_interval_sec: float,
    stop: asyncio.Event,
) -> None:
    """Cold-start-sha pattern (same as git_delta_loop): no persisted state, a
    restart just re-seeds the baseline at the current HEAD instead of scoring
    whatever doc changes happened while the container was down. Same accepted
    simplification as git_delta_loop's own docstring -- the very next real
    doc edit still gets scored, just against a fresh baseline.

    A dedicated forked RPC bus client is used for the embedding requests so
    replies never race with anything else subscribed on the shared producer
    ``bus`` connection (same reasoning as orion-chat-memory's own
    ``embed_rpc_bus``). Unlike chat-memory's module-level global (which lives
    for the whole process and is closed explicitly in a shutdown() hook),
    this fork is scoped to the loop's own lifetime and closed in the
    ``finally`` below -- code review 2026-08-11 caught the first draft
    leaking this connection on every task cancellation/redeploy (no
    teardown anywhere in this file)."""
    rpc_bus = await fork_rpc_client(bus)
    result_channel_prefix = f"orion:embedding:result:doc_semantic_drift:{source.node}"

    try:
        last_sha: str | None = None
        while not stop.is_set():
            try:
                head_sha = await asyncio.to_thread(_current_head_sha, repo_path)
                if last_sha is None:
                    last_sha = head_sha
                    logger.info("cocreation_doc_semantic_drift_cold_start head_sha=%s", head_sha)
                elif head_sha != last_sha:
                    changes = await asyncio.to_thread(
                        doc_semantic_drift_changes, last_sha, head_sha, repo_path
                    )
                    all_published = True
                    for change in changes:
                        event = await _score_change(
                            rpc_bus,
                            request_channel=embed_request_channel,
                            result_channel_prefix=result_channel_prefix,
                            source=source,
                            change=change,
                            collection=embed_collection,
                            embed_timeout_sec=embed_timeout_sec,
                            truncation_char_threshold=truncation_char_threshold,
                        )
                        published = await _publish(bus, channel, source, event)
                        all_published = all_published and published
                    if all_published:
                        last_sha = head_sha
                    # else: leave last_sha unchanged -- the next tick's diff will
                    # naturally cover this failed range too (same reasoning as
                    # git_delta_loop's own _publish() contract).
            except Exception:
                logger.exception("cocreation_doc_semantic_drift_tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    finally:
        try:
            await rpc_bus.close()
        except Exception:
            logger.exception("cocreation_doc_semantic_drift_rpc_bus_close_failed")
