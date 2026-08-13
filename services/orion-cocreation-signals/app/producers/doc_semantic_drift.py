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

# Max embedding requests in flight per scored file.
#
# orion-vector-host consumes these serially -- its Hunter is built without
# concurrent_handlers (default False, orion/core/bus/bus_service_chassis.py)
# and the live deployment runs the `hf` backend on CPU at ~0.6s per
# 1200-char embed (measured on the idle live host 2026-08-12). It is also
# the shared embedding host for live chat memory (3748 real
# embedding.generate requests in the 24h before this was written), so an
# unbounded fan-out over a large doc's chunks does not just make this
# producer slow -- it monopolizes a service other cognition depends on, and
# the later chunks time out anyway while queued behind their own siblings.
#
# Deliberately small: this producer is a background signal on a 300s poll,
# with nothing time-critical about any single file's score.
_EMBED_CONCURRENCY = 4


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


def _is_unscoreable(change: DocHunkChange) -> bool:
    """True when this change has no measurable drift, so no event is worth
    publishing at all.

    Drift is defined between a "before" and an "after". A brand-new file has
    no before; a pure-append edit has no removed lines under ``--unified=0``;
    a pure-deletion edit has no added lines. In every case one side is empty,
    cosine is undefined, and the only honest score is ``None``.

    Publishing those anyway is what this repo's first real live batch
    actually did: 6 of 8 events (2026-08-13) carried ``diff=None`` with an
    empty side, all of them new PR reports and specs. That ratio is
    structural, not a small sample -- this repo's doc volume is dominated by
    files that are created once and never revised, so waiting for more data
    would only produce more nulls. A channel whose traffic is mostly "we
    could not measure this" is the empty-shell pattern CLAUDE.md 0A rules
    out, even when each individual null is honestly labelled.

    Deliberately keyed on the real hunk text rather than on
    ``change_kind == "added"``: a *modified* file can be just as unscoreable
    (34.8% of real modified-doc changes over 300 commits are pure appends),
    and the text is what actually determines whether a cosine exists."""
    return not change.hunk_removed or not change.hunk_added


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


def _max_pair_drift(
    removed_vectors: list[list[float]], added_vectors: list[list[float]]
) -> float | None:
    """Drift of the *most-changed* chunk: for each chunk, how unmatched it is
    against the best-matching chunk on the other side; the score is the worst
    such mismatch in either direction.

    Deliberately NOT a mean-pooled cosine. Measured against the real model
    2026-08-12: pooling dilutes drift as roughly 1/N^2, so an identical
    unrelated rewrite of one section scored 0.2392 in a 1-chunk doc and
    0.0044 in an 8-chunk doc -- 54x lower for the same real edit. The
    calibration lineage this signal rests on (docs/superpowers/pr-reports/
    2026-08-11-doc-semantic-drift-diff-scoped-embedding.md) proposed a 0.3996
    threshold from single-window samples, so under pooling every long doc
    would read as a trivial edit -- the inversion of what the signal claims
    to measure, and a direct failure of CLAUDE.md's metric quality gate.

    For a hunk that fits in one window this reduces exactly to
    ``1 - cos(a, b)`` -- the pre-chunking formula -- so single-window scores
    and the 0.3996 threshold keep their meaning. That exact reduction needs
    BOTH sides to be a single chunk, not just one of them.

    It is NOT length-invariant, and an earlier version of this docstring
    wrongly claimed it was. Being a ``max`` over ``2*(N+M)`` non-negative
    terms makes it an extreme-value statistic: more chunks means more draws
    means a higher expected maximum, independent of whether more real change
    occurred. Measured over 30 real hunk pairs from this repo's history:

        N=1     n=8   median 0.1684
        N=2-3   n=18  median 0.2237
        N=4+    n=4   median 0.2861

    Monotone, ~70% median inflation from 1 to 4+ chunks (small buckets at
    the top -- treat the coefficient as directional, the ordering as real).
    So a multi-chunk event is biased HIGH relative to the single-window
    0.3996 threshold, and a consumer applying that cutoff across chunk
    counts will over-flag long docs. A multi-chunk threshold has to be
    derived separately.

    The bias is a tendency, not a guarantee: an extra chunk adds a term to
    the max (pushing up) but also adds a match candidate for every chunk on
    the other side (pushing down), so the net on any single event is
    content-dependent. A second changed section resembling the first can
    give that chunk a better match than its true counterpart and lower the
    score. Both directions are pinned by tests. Still far smaller than the
    ~54x collapse that disqualified mean-pooling, and it errs toward
    noticing change rather than hiding it.

    Symmetric (max over both directions) rather than only removed->added,
    which is a deliberate deviation from the directed form: a large block of
    *added* text with no counterpart on the removed side is real drift, and
    a directed removed->added scan cannot see it. For the single-chunk case
    both directions are identical, so this changes nothing about threshold
    compatibility.

    Honest limitation, not hidden: an extra chunk on the opposing side is
    one more chance for a changed chunk to find a match. Symmetry contains
    most of that -- an unmatched chunk contributes its own high term in the
    other direction, so it cannot quietly drag the score down -- but it does
    not eliminate it. A second changed section present on both sides whose
    content resembles the first section's changed text will give that chunk
    a better match than its true counterpart and lower the reported drift.
    Far weaker than pooling's ~1/N^2 collapse, which fired on every long doc
    regardless of content, but not strict length-invariance. Covered by
    ``test_partially_matching_neighbours_do_lower_the_score`` and
    ``test_unmatched_chunk_alone_does_not_lower_the_score``.

    Returns ``None`` when either side has no chunks (a newly-added file has
    no "before" text -- cosine is genuinely undefined, not a failure) or when
    any pair is unmeasurable, rather than fabricating a number."""
    if not removed_vectors or not added_vectors:
        return None
    worst: float | None = None
    for vectors, others in ((removed_vectors, added_vectors), (added_vectors, removed_vectors)):
        for vector in vectors:
            similarities = [_cosine_similarity(vector, other) for other in others]
            usable = [s for s in similarities if s is not None]
            if not usable:
                return None
            drift = 1.0 - max(usable)
            worst = drift if worst is None else max(worst, drift)
    return worst


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
        # orion-vector-host does NOT signal failure by withholding a reply --
        # confirmed by code read 2026-08-12 (services/orion-vector-host/
        # app/main.py:540-571): on a missing-text or embedder exception it
        # publishes a *well-formed* EmbeddingResultV1 with embedding=[] and
        # an extra `error` key. EmbeddingResultV1 is extra="ignore"
        # (orion/schemas/vector/schemas.py:73), so validating it silently
        # drops that key and hands back a result that looks successful.
        # Both guards are load-bearing: without them a failed chunk gets
        # dropped downstream and the remaining chunks are aggregated into a
        # score presented as a whole measurement.
        if payload_dict.get("error"):
            logger.warning(
                "cocreation_doc_semantic_drift_embed_host_error doc_id=%s error=%s",
                doc_id, payload_dict.get("error"),
            )
            return None
        result = EmbeddingResultV1.model_validate(payload_dict)
        if not result.embedding:
            logger.warning("cocreation_doc_semantic_drift_embed_empty_vector doc_id=%s", doc_id)
            return None
        return result
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
    and scored by the most-changed chunk pair -- see ``_chunk_text()`` for
    why chunking, and ``_max_pair_drift()`` for why max-pair rather than a
    pooled centroid.

    Embedding requests are capped at ``_EMBED_CONCURRENCY`` in flight. The
    embedding host handles requests serially on CPU (~0.6s each, measured on
    the idle live host 2026-08-12) and is shared with live chat memory --
    firing one unbounded ``gather`` over every chunk of a large doc both
    starves that shared host and guarantees the later chunks blow their own
    30s timeout."""
    removed_chunks = _chunk_text(change.hunk_removed, chunk_char_size)
    added_chunks = _chunk_text(change.hunk_added, chunk_char_size)
    # Shared across both sides -- the cap is about the embedding host's real
    # serial throughput, so counting each side separately would double it.
    semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def _embed_side(chunks: list[str], side: str) -> list[list[float]] | None:
        if not chunks:
            return None

        async def _one(index: int, chunk: str):
            async with semaphore:
                return await _request_embedding(
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

        results = await asyncio.gather(
            *[_one(index, chunk) for index, chunk in enumerate(chunks)]
        )
        # A single failed chunk must not silently drop out of the comparison
        # and leave the rest presented as a whole measurement. `_request_
        # embedding` now returns None for the embedding host's own
        # error-shaped replies too, which is what makes this guard real.
        if any(result is None for result in results):
            return None
        return [result.embedding for result in results if result is not None]

    removed_vectors, added_vectors = await asyncio.gather(
        _embed_side(removed_chunks, "removed"),
        _embed_side(added_chunks, "added"),
    )

    diff: float | None = None
    if removed_vectors is not None and added_vectors is not None:
        # O(N*M) cosines over 1024-dim vectors in pure Python. Off the event
        # loop because run_producers shares one loop with the heartbeat
        # chassis and five other producers -- a synchronous ~0.5s+ block
        # here stalls all of them.
        diff = await asyncio.to_thread(_max_pair_drift, removed_vectors, added_vectors)

    return DocSemanticDriftV1(
        observed_at=datetime.now(timezone.utc),
        sha=change.sha,
        path=change.path,
        commit_prefix=change.commit_prefix,
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
            "cocreation_doc_semantic_drift_published sha=%s path=%s diff=%s chunks=%s/%s",
            event.sha, event.path, event.diff_scoped_embedding_diff,
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
                        if _is_unscoreable(change):
                            # Not a failure -- drift between "before" and
                            # "after" is undefined when one side has no
                            # text. Skipped BEFORE embedding: a 15-chunk new
                            # doc otherwise burned 15 real embedding
                            # requests (plus 15 vector-store writes and 15
                            # tissue feeds) to publish a guaranteed None.
                            logger.info(
                                "cocreation_doc_semantic_drift_skipped_unscoreable "
                                "sha=%s path=%s kind=%s removed_chars=%d added_chars=%d",
                                change.sha, change.path, change.change_kind,
                                len(change.hunk_removed), len(change.hunk_added),
                            )
                            continue
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
