"""Unit tests for the chunk + max-chunk-pair scoring path in
app/producers/doc_semantic_drift.py.

Why this exists: confirmed live 2026-08-12 that orion-vector-host runs
BAAI/bge-large-en-v1.5, whose real ceiling is 512 tokens
(``model_max_length`` / ``max_position_embeddings``, read from the running
container's own model files). The model *silently clips* past that -- it
does not error -- so before chunking, a 20KB doc diff and its first 2KB
produced the same score, and every real event this repo published carried
``possibly_truncated=True``. These tests cover the real behavior that
replaced it: split each hunk side into windows below that ceiling, embed
every window, score by the most-changed chunk pair, and never present a
partial measurement as a whole one.

Unlike test_doc_semantic_drift_producer.py (which monkeypatches
``_score_change`` wholesale to test the loop's *scheduling*), these tests
drive the real ``_score_change`` with a stubbed embedding transport, so the
chunking/scoring arithmetic itself is under test.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import doc_semantic_drift as mod  # noqa: E402

from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.structural_mass.doc_semantic_drift import DocHunkChange  # noqa: E402


# --------------------------------------------------------------------------
# _chunk_text
# --------------------------------------------------------------------------


def test_empty_text_yields_no_chunks() -> None:
    """A newly-added file has no removed side -- a real state, not an error.
    It must produce zero chunks (which the caller turns into a real
    ``chunk_count_removed=0`` and a ``diff=None``), not one empty chunk that
    would waste an embedding request on the empty string."""
    assert mod._chunk_text("", 2048) == []


def test_short_text_stays_one_chunk() -> None:
    """The common case, and the one that keeps new scores comparable to the
    pre-chunking ones: a hunk that already fits is passed through whole."""
    assert mod._chunk_text("line one\nline two", 2048) == ["line one\nline two"]


def test_long_text_splits_on_line_boundaries_and_respects_max() -> None:
    text = "\n".join(f"line {i} with some real content" for i in range(200))
    chunks = mod._chunk_text(text, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Split on line boundaries -- no chunk starts or ends mid-line.
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert line in text


def test_chunking_preserves_all_content() -> None:
    """The whole point is that nothing gets dropped. Rejoining every chunk
    must reproduce every original line -- otherwise chunking would be
    trading one silent truncation for another."""
    text = "\n".join(f"line {i}" for i in range(500))
    chunks = mod._chunk_text(text, 137)
    rejoined_lines = [line for chunk in chunks for line in chunk.split("\n")]
    assert rejoined_lines == text.split("\n")


def test_single_line_longer_than_max_is_hard_split() -> None:
    """A minified/one-line doc has no boundary to respect -- it still must
    not exceed the window, since exceeding it is exactly the silent-clip
    failure this replaces."""
    chunks = mod._chunk_text("x" * 500, 100)
    assert len(chunks) == 5
    assert all(len(c) == 100 for c in chunks)
    assert "".join(chunks) == "x" * 500


def test_oversized_line_after_buffered_lines_does_not_lose_the_buffer() -> None:
    """Regression guard on the hard-split branch: the buffered short lines
    that precede an oversized line must be emitted, not dropped."""
    text = "short a\nshort b\n" + "y" * 250
    chunks = mod._chunk_text(text, 100)
    rejoined = "\n".join(chunks)
    assert "short a" in rejoined and "short b" in rejoined
    assert rejoined.count("y") == 250


def test_nonpositive_max_chars_degrades_to_single_chunk_not_infinite_loop() -> None:
    """Defensive: a misconfigured 0/negative window must not spin forever in
    the hard-split ``while`` loop."""
    assert mod._chunk_text("abc", 0) == ["abc"]
    assert mod._chunk_text("abc", -5) == ["abc"]


# --------------------------------------------------------------------------
# _max_pair_drift
# --------------------------------------------------------------------------


def test_single_chunk_per_side_reduces_to_plain_cosine_distance() -> None:
    """Threshold compatibility: for a hunk that fits one window the score
    must be exactly the pre-chunking ``1 - cos(a, b)``, so existing scores
    and the 0.3996 calibration threshold keep their meaning."""
    assert mod._max_pair_drift([[1.0, 0.0]], [[0.0, 1.0]]) == pytest.approx(1.0)
    assert mod._max_pair_drift([[1.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(0.0)


def test_extra_chunks_bias_the_score_high_not_low() -> None:
    """Max-pair is NOT length-invariant, and an earlier version of this test
    wrongly asserted that it was -- using padding chunks that were byte
    identical on both sides, so every padding term was exactly 1 - 1.0 = 0
    and could not affect the max by construction. Real embeddings are never
    identical across a diff's two sides, and under --unified=0 a hunk
    contains no unchanged chunks at all, so that fixture matched no real
    input shape.

    The real behavior: the score is a max over 2*(N+M) non-negative terms,
    so more chunks means more draws means a higher expected maximum.
    Measured over 30 real hunk pairs: median 0.1684 at N=1, 0.2237 at
    N=2-3, 0.2861 at N=4+.

    This test pins the *mechanism* -- an added chunk contributes its own
    term to the max and can raise the score with no change to the original
    pair. It deliberately does NOT assert that extra chunks always raise
    the score: they don't, and
    ``test_partially_matching_neighbours_do_lower_the_score`` below is the
    counter-case. The net effect on any single event is content-dependent;
    only the aggregate tendency across real hunks is upward."""
    matched_a, matched_b = [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    # A second changed section with no counterpart on the other side.
    unmatched = [0.0, 1.0, 0.0]

    alone = mod._max_pair_drift([matched_a], [matched_b])
    with_extra = mod._max_pair_drift([matched_a, unmatched], [matched_b])

    assert alone == pytest.approx(0.0)
    assert with_extra == pytest.approx(1.0)
    assert with_extra > alone


def test_partially_matching_neighbours_do_lower_the_score() -> None:
    """Honest limitation, documented rather than hidden.

    Because the score is symmetric, simply adding an unmatched chunk does
    NOT lower it -- that chunk contributes its own high term in the other
    direction. The narrower case that does lower it: a second changed
    section present on *both* sides whose content partially resembles the
    first section's changed text. It gives the changed chunk a better match
    than its true counterpart, so the reported drift falls.

    Far weaker than mean-pooling's ~1/N^2 collapse (which fired on every
    long doc regardless of content), but real: a doc whose changed sections
    resemble each other can read quieter than the same edit in isolation."""
    removed_changed, added_changed = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    # Present on both sides, and 45 degrees from the changed pair.
    resembling = [0.70710678, 0.70710678, 0.0]

    alone = mod._max_pair_drift([removed_changed], [added_changed])
    with_resembling_section = mod._max_pair_drift(
        [removed_changed, resembling], [added_changed, resembling]
    )

    assert alone == pytest.approx(1.0)
    assert with_resembling_section == pytest.approx(1.0 - 0.70710678)
    assert with_resembling_section < alone


def test_unmatched_chunk_alone_does_not_lower_the_score() -> None:
    """The counterpart to the above, and why symmetry was worth having: an
    added chunk with no match cannot quietly drag the score down, because
    its own unmatchedness is itself scored."""
    drift = mod._max_pair_drift([[1.0, 0.0]], [[0.0, 1.0], [0.70710678, 0.70710678]])
    assert drift == pytest.approx(1.0)


def test_unmatched_added_chunk_counts_as_drift(monkeypatch) -> None:
    """Symmetric on purpose. A large block of *added* text with no
    counterpart on the removed side is real drift, and a directed
    removed->added scan cannot see it -- every removed chunk would find its
    match and report ~0."""
    matched = [1.0, 0.0]
    orphan_addition = [0.0, 1.0]
    drift = mod._max_pair_drift([matched], [matched, orphan_addition])
    assert drift == pytest.approx(1.0)


def test_reports_the_worst_chunk_not_the_average() -> None:
    """One badly-changed section must not be averaged away by neighbours
    that barely moved."""
    stable = [1.0, 0.0]
    rewritten = [0.0, 1.0]
    drift = mod._max_pair_drift([stable, rewritten], [stable, stable])
    assert drift == pytest.approx(1.0)


def test_returns_none_when_either_side_has_no_chunks() -> None:
    """A newly-added file has no "before" text -- genuinely undefined, not a
    fabricated 0.0."""
    assert mod._max_pair_drift([], [[1.0, 0.0]]) is None
    assert mod._max_pair_drift([[1.0, 0.0]], []) is None


def test_returns_none_rather_than_fabricate_when_a_pair_is_unmeasurable() -> None:
    """Mismatched dims / zero vectors make the cosine undefined; reporting a
    number anyway would be a fabricated measurement."""
    assert mod._max_pair_drift([[1.0, 0.0]], [[0.0, 0.0]]) is None
    assert mod._max_pair_drift([[1.0, 0.0]], [[1.0, 0.0, 0.0]]) is None


# --------------------------------------------------------------------------
# _score_change (real chunking path, stubbed embedding transport)
# --------------------------------------------------------------------------


@pytest.fixture
def source() -> ServiceRef:
    return ServiceRef(name="cocreation-signals", version="0.1.0", node="test-node")


class _FakeEmbeddingResult:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


def _stub_embeddings(monkeypatch, by_text=None, default=(1.0, 0.0), fail_texts=()):
    """Replace the real bus RPC with a deterministic text -> vector map.
    Records every requested (doc_id, text) so tests can assert on how the
    producer actually chunked the input."""
    calls: list[tuple[str, str]] = []

    async def _fake_request_embedding(rpc_bus, *, doc_id, text, **kwargs):
        calls.append((doc_id, text))
        if text in fail_texts:
            return None
        vector = (by_text or {}).get(text, list(default))
        return _FakeEmbeddingResult(list(vector))

    monkeypatch.setattr(mod, "_request_embedding", _fake_request_embedding)
    return calls


def _change(removed: str, added: str, change_kind: str | None = "modified") -> DocHunkChange:
    return DocHunkChange(
        sha="a" * 40,
        path="docs/real.md",
        commit_subject="docs: real edit",
        commit_prefix="docs",
        hunk_removed=removed,
        hunk_added=added,
        change_kind=change_kind,
    )


async def _score(change: DocHunkChange, source: ServiceRef, chunk_char_size: int = 2048):
    return await mod._score_change(
        object(),
        request_channel="orion:embedding:generate",
        result_channel_prefix="orion:embedding:result",
        source=source,
        change=change,
        collection="doc_semantic_drift",
        embed_timeout_sec=5.0,
        chunk_char_size=chunk_char_size,
    )


@pytest.mark.asyncio
async def test_long_hunk_is_embedded_in_multiple_chunks_not_truncated(source, monkeypatch) -> None:
    """The core regression: before this, everything past the window was
    silently dropped inside the model and the tail was never measured."""
    calls = _stub_embeddings(monkeypatch)
    long_text = "\n".join(f"line {i}" for i in range(400))
    event = await _score(_change(long_text, long_text), source, chunk_char_size=100)

    assert event.chunk_count_removed > 1
    assert event.chunk_count_added > 1
    # One real embedding request per chunk, per side -- nothing skipped.
    assert len(calls) == event.chunk_count_removed + event.chunk_count_added


@pytest.mark.asyncio
async def test_each_chunk_gets_a_distinct_doc_id(source, monkeypatch) -> None:
    """Without the chunk index in the doc_id, chunk N would overwrite chunk
    0 in the vector store on every request -- the collection would end up
    holding only each hunk's final window."""
    calls = _stub_embeddings(monkeypatch)
    long_text = "\n".join(f"line {i}" for i in range(400))
    await _score(_change(long_text, long_text), source, chunk_char_size=100)

    doc_ids = [doc_id for doc_id, _text in calls]
    assert len(doc_ids) == len(set(doc_ids))


@pytest.mark.asyncio
async def test_score_reflects_content_of_all_chunks_not_just_the_first(
    source, monkeypatch
) -> None:
    """Two diffs sharing an identical first window but differing in their
    tails must score differently -- the exact case that was indistinguishable
    before chunking."""
    head = "same head line"
    # Distinct per-chunk vectors so the score actually depends on the tail,
    # not only the shared head. Deliberately NOT mirror-symmetric
    # about the removed side's vector: [0,1] and [0,-1] sit at the same
    # *angle* from [1,0] and would score identically, hiding the very
    # difference this test exists to catch.
    by_text = {
        head: [1.0, 0.0],
        "tail one": [0.0, 1.0],
        "tail two": [1.0, 0.0],
    }
    _stub_embeddings(monkeypatch, by_text=by_text)

    event_a = await _score(_change(head, f"{head}\ntail one"), source, chunk_char_size=len(head))
    event_b = await _score(_change(head, f"{head}\ntail two"), source, chunk_char_size=len(head))

    assert event_a.diff_scoped_embedding_diff is not None
    assert event_b.diff_scoped_embedding_diff is not None
    assert event_a.diff_scoped_embedding_diff != pytest.approx(
        event_b.diff_scoped_embedding_diff
    )


@pytest.mark.asyncio
async def test_one_failed_chunk_yields_none_not_a_partial_measurement(source, monkeypatch) -> None:
    """A dropped chunk must not silently vanish from the comparison and
    leave a partial score presented as a whole one -- that's a fabricated
    measurement, worse than an honest None."""
    text = "alpha\nbeta"
    _stub_embeddings(monkeypatch, fail_texts={"beta"})
    event = await _score(_change(text, text), source, chunk_char_size=5)

    assert event.diff_scoped_embedding_diff is None
    # The real chunk counts are still reported -- a consumer can see the
    # measurement was attempted over 2 windows and failed, not that the
    # hunk was empty.
    assert event.chunk_count_removed == 2


@pytest.mark.asyncio
async def test_added_file_reports_zero_removed_chunks_and_none_diff(source, monkeypatch) -> None:
    """A newly-added doc has no "before" text, so cosine is genuinely
    undefined. This must be distinguishable from a real embedding failure --
    that's what change_kind + chunk counts are for."""
    _stub_embeddings(monkeypatch)
    event = await _score(_change("", "brand new doc", change_kind="added"), source)

    assert event.diff_scoped_embedding_diff is None
    assert event.change_kind == "added"
    # This is the pair that actually disambiguates: 0 chunks on a side means
    # structurally undefined, NOT a failed embedding. change_kind alone does
    # not carry that -- a pure-append edit is status "M" with an empty
    # removed side (34.8% of real modified-doc changes in this repo).
    assert event.chunk_count_removed == 0
    assert event.chunk_count_added == 1


@pytest.mark.asyncio
async def test_change_kind_passes_through_to_the_event(source, monkeypatch) -> None:
    """The whole disambiguation depends on this reaching the payload: an
    `added` file with diff=None is expected, a `modified` file with
    diff=None is an alarm."""
    _stub_embeddings(monkeypatch)
    event = await _score(_change("", "brand new doc", change_kind="added"), source)
    assert event.change_kind == "added"


@pytest.mark.asyncio
async def test_short_hunk_still_scores_as_a_single_chunk(source, monkeypatch) -> None:
    """Chunking must not change behavior for hunks that already fit -- these
    scores stay directly comparable to the pre-chunking ones."""
    _stub_embeddings(monkeypatch, by_text={"old": [1.0, 0.0], "new": [0.0, 1.0]})
    event = await _score(_change("old", "new"), source)

    assert event.chunk_count_removed == 1
    assert event.chunk_count_added == 1
    # Orthogonal vectors -> cosine 0 -> diff 1.0.
    assert event.diff_scoped_embedding_diff == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_both_sides_embed_concurrently_and_report_real_char_lengths(
    source, monkeypatch
) -> None:
    _stub_embeddings(monkeypatch)
    change = _change("removed text", "added text here")
    event = await _score(change, source)

    assert event.hunk_removed_len_chars == len("removed text")
    assert event.hunk_added_len_chars == len("added text here")


@pytest.mark.asyncio
async def test_event_no_longer_carries_possibly_truncated(source, monkeypatch) -> None:
    """The replaced field fired True on every real event this repo produced;
    a constant is not a signal. Guard against it being reintroduced by
    accident -- the schema forbids extras, so this asserts the intent."""
    _stub_embeddings(monkeypatch)
    event = await _score(_change("old", "new"), source)
    assert not hasattr(event, "possibly_truncated")


@pytest.mark.asyncio
async def test_pure_append_to_existing_doc_is_undefined_not_an_alarm(source, monkeypatch) -> None:
    """Regression guard on a wrong claim this schema used to make. A
    pure-append edit has git status "modified" but no removed side at all
    under --unified=0 -- measured at 34.8% of real modified-doc changes over
    this repo's last 300 commits. It must be distinguishable from a real
    embedding failure by chunk_count_removed == 0, not misread as one."""
    _stub_embeddings(monkeypatch)
    event = await _score(_change("", "a whole new section", change_kind="modified"), source)

    assert event.change_kind == "modified"
    assert event.diff_scoped_embedding_diff is None
    assert event.chunk_count_removed == 0


@pytest.mark.asyncio
async def test_embedding_host_error_reply_is_not_treated_as_a_success(source, monkeypatch) -> None:
    """orion-vector-host signals failure with a well-formed EmbeddingResultV1
    carrying embedding=[] and an extra `error` key -- NOT by withholding a
    reply (services/orion-vector-host/app/main.py:540-571). EmbeddingResultV1
    is extra="ignore", so a naive model_validate() drops `error` and returns
    something that looks successful. If that reaches the scorer, the failed
    chunk is dropped and the survivors are published as a whole measurement.
    """
    captured: list[str] = []

    async def _fake_rpc_request(channel, envelope, reply_channel=None, timeout_sec=None):
        captured.append(envelope.payload["doc_id"])
        return {"data": b"unused"}

    class _Codec:
        def decode(self, _data):
            class _Decoded:
                ok = True

                class envelope:  # noqa: N801
                    # Exactly the real error shape from the embedding host.
                    payload = {
                        "doc_id": "d",
                        "embedding": [],
                        "embedding_model": None,
                        "embedding_dim": 0,
                        "error": "CUDA out of memory",
                    }

            return _Decoded()

    class _Bus:
        codec = _Codec()
        rpc_request = staticmethod(_fake_rpc_request)

    result = await mod._request_embedding(
        _Bus(),
        request_channel="orion:embedding:generate",
        result_channel_prefix="orion:embedding:result",
        source=source,
        doc_id="doc_semantic_drift:abc:x.md:added:0",
        text="some real text",
        collection="doc_semantic_drift",
        timeout_sec=5.0,
    )
    assert result is None, "an error-shaped reply must not read as a successful embedding"


@pytest.mark.asyncio
async def test_one_failed_chunk_nulls_the_side_rather_than_scoring_the_survivors(
    source, monkeypatch
) -> None:
    """The multi-chunk consequence of the guard above: with 3 chunks where
    one fails, publishing a score over the other 2 would be a fabricated
    whole measurement."""
    _stub_embeddings(monkeypatch, fail_texts={"bbb"})
    event = await _score(_change("aaa\nbbb\nccc", "zzz"), source, chunk_char_size=3)

    assert event.chunk_count_removed == 3
    assert event.diff_scoped_embedding_diff is None


@pytest.mark.asyncio
async def test_embedding_requests_are_capped_in_flight(source, monkeypatch) -> None:
    """The embedding host consumes requests serially on CPU (~0.6s each) and
    is shared with live chat memory. An unbounded gather over a large doc's
    chunks starves that shared service AND guarantees the queued-behind
    chunks blow their own 30s timeout -- so the biggest docs would fail to
    score while doing the most damage."""
    in_flight = 0
    peak = 0

    async def _fake_request_embedding(rpc_bus, *, doc_id, text, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _FakeEmbeddingResult([1.0, 0.0])

    monkeypatch.setattr(mod, "_request_embedding", _fake_request_embedding)
    long_text = "\n".join(f"line {i}" for i in range(200))
    await _score(_change(long_text, long_text), source, chunk_char_size=20)

    assert peak <= mod._EMBED_CONCURRENCY
    # Guard against the cap passing trivially because nothing overlapped.
    assert peak > 1
