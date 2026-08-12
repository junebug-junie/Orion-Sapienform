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


async def _load_last_sha(bus: OrionBusAsync, state_key: str) -> str | None:
    """Real, durable baseline read -- confirmed live 2026-08-12 (Juniper):
    an in-process-only ``last_sha`` gets wiped by every redeploy, and this
    service redeploys far more often than a real doc commit lands. A
    redeploy landing right after a doc merge (the common real sequence,
    not an edge case) silently swallows that doc forever under the old
    pure-in-memory design -- this Redis-backed key is the fix. Returns
    ``None`` on any real failure (bus disabled, not yet connected, no key
    set yet) so the caller falls back to the same cold-start behavior as
    before, never a fabricated SHA."""
    try:
        redis = bus.redis
    except Exception:
        return None
    try:
        raw = await redis.get(state_key)
    except Exception:
        logger.warning("cocreation_doc_semantic_drift_state_load_failed key=%s", state_key, exc_info=True)
        return None
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)


async def _save_last_sha(bus: OrionBusAsync, state_key: str, sha: str) -> None:
    """Best-effort durable write -- a failure here must not crash the tick
    that already published real data; the next tick's read just falls back
    to cold-starting at that point, same as before this fix existed."""
    try:
        redis = bus.redis
    except Exception:
        return
    try:
        await redis.set(state_key, sha)
    except Exception:
        logger.warning("cocreation_doc_semantic_drift_state_save_failed key=%s sha=%s", state_key, sha, exc_info=True)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split a hunk into <=``max_chars`` windows on line boundaries.

    Exists because the real embedding model silently clips rather than
    erroring: confirmed live 2026-08-12 that orion-vector-host runs
    BAAI/bge-large-en-v1.5 with a hard 512-token ceiling (read from the
    running container's own ``tokenizer_config.json`` ``model_max_length``
    and ``config.json`` ``max_position_embeddings``). Before this, a 20KB
    PR-report diff and its first 2KB scored identically -- the tail was
    never measured, and the resulting score was reported with a
    ``possibly_truncated`` flag that was True on every real event.

    Line-boundary splitting keeps each window a coherent run of real diff
    lines rather than slicing mid-sentence. A single line longer than
    ``max_chars`` is hard-split, since there's no boundary to respect.
    Returns ``[]`` for empty text -- a real state (a newly-added file has
    no removed side), not an error."""
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        while len(line) > max_chars:
            # No line boundary to respect -- emit what's buffered, then
            # hard-split the oversized line itself.
            if current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        # +1 for the newline this line would rejoin with.
        addition = len(line) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [line], len(line)
        else:
            current.append(line)
            current_len += addition
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c]


def _mean_pool(vectors: list[list[float]]) -> list[float] | None:
    """L2-normalize each chunk vector, average, then renormalize.

    Normalizing before averaging keeps one long chunk from dominating the
    pooled direction purely through magnitude -- the standard way to pool
    sentence embeddings across windows. Returns ``None`` if there's nothing
    real to pool (no vectors, mismatched dims, or an all-zero average),
    matching ``_cosine_similarity``'s own "return None rather than
    fabricate" contract."""
    usable = [v for v in vectors if v]
    if not usable:
        return None
    dim = len(usable[0])
    if any(len(v) != dim for v in usable):
        return None
    summed = [0.0] * dim
    contributing = 0
    for vec in usable:
        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0.0:
            continue
        for i, x in enumerate(vec):
            summed[i] += x / norm
        contributing += 1
    if contributing == 0:
        return None
    pooled = [x / contributing for x in summed]
    pooled_norm = sum(x * x for x in pooled) ** 0.5
    if pooled_norm == 0.0:
        return None
    return [x / pooled_norm for x in pooled]


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
    chunk_char_size: int,
) -> DocSemanticDriftV1:
    """One event per changed doc file. Both hunk sides are chunked to fit
    the embedding model's real 512-token window, embedded chunk-by-chunk,
    and mean-pooled before the cosine -- see ``_chunk_text()`` for why."""
    removed_chunks = _chunk_text(change.hunk_removed, chunk_char_size)
    added_chunks = _chunk_text(change.hunk_added, chunk_char_size)

    async def _embed_side(chunks: list[str], side: str) -> list[float] | None:
        if not chunks:
            return None
        results = await asyncio.gather(
            *[
                _request_embedding(
                    rpc_bus,
                    request_channel=request_channel,
                    result_channel_prefix=result_channel_prefix,
                    source=source,
                    # Chunk index keeps each window a distinct vector-store
                    # doc_id -- without it, chunk N would overwrite chunk 0
                    # in the collection every request.
                    doc_id=f"doc_semantic_drift:{change.sha}:{change.path}:{side}:{index}",
                    text=chunk,
                    collection=collection,
                    timeout_sec=embed_timeout_sec,
                )
                for index, chunk in enumerate(chunks)
            ]
        )
        # A single failed chunk must not silently shrink the pooled vector
        # into a partial measurement presented as a whole one.
        if any(result is None for result in results):
            return None
        return _mean_pool([result.embedding for result in results if result is not None])

    removed_vector, added_vector = await asyncio.gather(
        _embed_side(removed_chunks, "removed"),
        _embed_side(added_chunks, "added"),
    )

    diff: float | None = None
    if removed_vector is not None and added_vector is not None:
        similarity = _cosine_similarity(removed_vector, added_vector)
        if similarity is not None:
            diff = 1.0 - similarity

    return DocSemanticDriftV1(
        observed_at=datetime.now(timezone.utc),
        sha=change.sha,
        path=change.path,
        commit_prefix=change.commit_prefix,
        change_kind=change.change_kind,
        diff_scoped_embedding_diff=diff,
        chunk_count_removed=len(removed_chunks),
        chunk_count_added=len(added_chunks),
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
            "cocreation_doc_semantic_drift_published sha=%s path=%s kind=%s diff=%s chunks=%s/%s",
            event.sha, event.path, event.change_kind, event.diff_scoped_embedding_diff,
            event.chunk_count_removed, event.chunk_count_added,
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
    chunk_char_size: int,
    poll_interval_sec: float,
    state_key: str,
    stop: asyncio.Event,
) -> None:
    """Durable-baseline-sha pattern -- ``last_sha`` is persisted to Redis
    (``state_key``) after every real advance, not just held in process
    memory. Fixed live 2026-08-12: the original cold-start-sha design
    (matching ``git_delta_loop``'s own in-memory-only pattern) meant every
    redeploy re-seeded the baseline at whatever HEAD was current *at that
    moment* -- and since this service gets redeployed far more often than a
    real doc commit lands, a redeploy landing right after a doc merge (the
    common real sequence for this repo, not a rare edge case) silently
    swallowed that doc forever, confirmed live against two real PRs
    (#1571, #1577) before this fix. On startup, the baseline resumes from
    the last real value this producer ever advanced to, not from whatever
    HEAD happens to be right now -- a restart with no persisted state yet
    (first-ever boot, or the Redis key genuinely doesn't exist) still cold
    starts exactly as before.

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
        last_sha: str | None = await _load_last_sha(bus, state_key)
        if last_sha is not None:
            logger.info("cocreation_doc_semantic_drift_resumed_from_durable_state last_sha=%s", last_sha)
        while not stop.is_set():
            try:
                head_sha = await asyncio.to_thread(_current_head_sha, repo_path)
                if last_sha is None:
                    last_sha = head_sha
                    logger.info("cocreation_doc_semantic_drift_cold_start head_sha=%s", head_sha)
                    await _save_last_sha(bus, state_key, last_sha)
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
                            chunk_char_size=chunk_char_size,
                        )
                        published = await _publish(bus, channel, source, event)
                        all_published = all_published and published
                    if all_published:
                        last_sha = head_sha
                        await _save_last_sha(bus, state_key, last_sha)
                    # else: leave last_sha unchanged -- the next tick's diff will
                    # naturally cover this failed range too (same reasoning as
                    # git_delta_loop's own _publish() contract). Not persisting
                    # here is deliberate: persisting a not-yet-fully-published
                    # sha would durably lose the failed range across a restart
                    # too, not just within this process's lifetime.
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
