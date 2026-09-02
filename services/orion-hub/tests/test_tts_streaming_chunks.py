"""Streamed sentence-chunk TTS (2026-09-02).

Guards the behavior that makes Orion's voice start fast: the reply is
synthesized in sentence-aligned chunks and each is queued to the browser as
soon as it exists, rather than one clip rendered after the whole reply.
"""
import asyncio
import sys

import pytest

from scripts.utils import split_sentences
from scripts.websocket_handler import chunk_text_for_speech, run_tts_remote

# conftest's _ensure_hub_paths() deletes every cached `scripts.*` module so the
# corrected sys.path wins. A plain `import scripts.websocket_handler` inside a
# test therefore hands back a FRESH module object, not the one `run_tts_remote`
# above was bound to -- monkeypatching that copy silently does nothing and the
# real TTS client gets called. Always patch the module the function under test
# actually lives in.
_WH = sys.modules[run_tts_remote.__module__]


# --- chunker ------------------------------------------------------------

def test_chunks_are_sentence_aligned_and_drop_no_sentence():
    """Every chunk boundary lands on a sentence end; no sentence is lost or reordered.

    The guarantee is over SENTENCES, not bytes. The pre-existing
    `split_sentences` helper this builds on collapses newlines to spaces and
    strips each sentence, so chunking is not byte-identical to the input:
    "A.\nB." comes back as "A. B.". That is why the assertion below compares
    against split_sentences' own output rather than the raw string -- an
    earlier version of this test asserted `" ".join(chunks) == text` and
    passed only because its fixture happened to be single-spaced with no
    newlines. Audibly this is a non-issue (XTTS splits on sentences itself),
    but the weaker claim is the true one.
    """
    text = (
        "One sentence here. Two sentences here now. Three of them by now. "
        "Four is where we are. Five and still going on. Six ends the run."
    )
    chunks = chunk_text_for_speech(text, first_chunk_chars=40, chunk_chars=80)
    assert len(chunks) > 1, "this fixture must actually split"
    for c in chunks:
        assert c.endswith((".", "!", "?")), c
    # Every sentence survives, in order, exactly once.
    assert " ".join(chunks).split() == " ".join(split_sentences(text)).split()


def test_whitespace_is_normalized_not_preserved():
    """Pins the known, inherited whitespace behavior so it cannot change silently."""
    chunks = chunk_text_for_speech(
        "First line here.\nSecond line here.", first_chunk_chars=10, chunk_chars=20
    )
    assert chunks == ["First line here.", "Second line here."]


def test_first_chunk_is_smaller_than_later_chunks():
    """The first chunk sets time-to-first-sound, so it must be the short one."""
    text = " ".join(f"This is sentence number {i} of the reply." for i in range(20))
    chunks = chunk_text_for_speech(text, first_chunk_chars=80, chunk_chars=280)
    assert len(chunks) >= 3
    assert len(chunks[0]) < len(chunks[1])
    # Hand-computed against the doubling ramp. Each sentence here is 39 chars
    # ("This is sentence number 0 of the reply."), and buf_len adds len+1 per
    # sentence, so the accumulator hits a target exactly on a sentence count.
    # Targets ramp 80 -> 160 -> 280 (capped):
    #   chunk 0: target 80  -> 2 sentences, 39*2 + 1 space  =  79
    #   chunk 1: target 160 -> 4 sentences, 39*4 + 3 spaces = 159
    # Exact values, not bounds: a chunker that silently shifted a sentence
    # between chunks would still satisfy an inequality here.
    assert len(chunks[0]) == 79, chunks[0]
    assert len(chunks[1]) == 159, chunks[1]
    assert chunks[0].startswith("This is sentence number 0")


def test_targets_double_so_the_first_chunk_can_be_short_without_starving():
    """The ramp is load-bearing, not cosmetic.

    Measured 2026-09-02: a flat pair (first=80, rest=280) starts the voice at
    1.95s and then runs DRY for 1.77s, because a ~5s first clip cannot cover
    the ~7s synthesis of a 280-char second chunk. Doubling closes that gap
    while keeping the same fast start. A regression to flat targets would
    reintroduce audible silence mid-reply, which no other test here catches.
    """
    text = " ".join(f"This is sentence number {i} of the reply." for i in range(24))
    chunks = chunk_text_for_speech(text, first_chunk_chars=80, chunk_chars=280)
    assert len(chunks) >= 4
    # Second chunk must be ~2x the first, NOT the full 280-char cap.
    assert len(chunks[1]) < 280
    assert 1.5 <= len(chunks[1]) / len(chunks[0]) <= 2.5


def test_no_chunk_exceeds_three_times_its_predecessor():
    """The starvation invariant.

    Synthesis runs at a measured real-time factor of ~0.33, so chunk k+1
    renders in ~0.33x its own playback time. It must finish before chunk k
    stops playing, which holds while a chunk is under ~3x its predecessor.
    A regression that lets chunks grow faster than that would make the
    voice run dry mid-reply -- the exact failure this design avoids.
    """
    text = " ".join(f"Sentence {i} of a fairly long spoken reply." for i in range(40))
    chunks = chunk_text_for_speech(text, first_chunk_chars=80, chunk_chars=280)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert len(nxt) <= 3 * len(prev), f"{len(nxt)} > 3x{len(prev)}"


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_yields_no_chunks(text):
    assert chunk_text_for_speech(text, first_chunk_chars=80, chunk_chars=280) == []


def test_single_short_sentence_is_one_chunk():
    """Short replies must not regress into extra round trips."""
    assert chunk_text_for_speech(
        "Yes, that is done.", first_chunk_chars=80, chunk_chars=280
    ) == ["Yes, that is done."]


# --- streaming loop -----------------------------------------------------

class _FakeSettings:
    HUB_TTS_TIMEOUT_SEC = 180.0
    HUB_TTS_STREAM_ENABLED = True
    HUB_TTS_STREAM_FIRST_CHUNK_CHARS = 80
    HUB_TTS_STREAM_CHUNK_CHARS = 280


LONG_REPLY = " ".join(f"This is sentence number {i} of the reply." for i in range(20))


def _patch(monkeypatch, *, results, settings_obj=None):
    """Stub synthesize_tts_reply with a scripted result per chunk."""
    seen = []

    async def fake_synth(text, client, **kw):
        seen.append({"text": text, "chunk_index": kw.get("chunk_index"),
                     "chunk_total": kw.get("chunk_total")})
        return results[len(seen) - 1]

    monkeypatch.setattr(_WH, "synthesize_tts_reply", fake_synth)
    monkeypatch.setattr(_WH, "settings", settings_obj or _FakeSettings())
    return seen


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_each_chunk_is_queued_separately_and_in_order(monkeypatch):
    """The whole point: N audio messages, ordered, not one blob at the end."""
    n_chunks = len(chunk_text_for_speech(LONG_REPLY, first_chunk_chars=80, chunk_chars=280))
    seen = _patch(monkeypatch, results=[
        {"audio_response": f"audio{i}", "tts_source_text": "x"} for i in range(n_chunks)
    ])
    q = asyncio.Queue()
    asyncio.run(run_tts_remote(LONG_REPLY, object(), q))
    msgs = _drain(q)
    assert len(msgs) == n_chunks > 1
    assert [m["audio_response"] for m in msgs] == [f"audio{i}" for i in range(n_chunks)]
    assert all(m["state"] == "speaking" for m in msgs)
    # Chunk metadata is threaded through for the voice.tts.* trace.
    assert [s["chunk_index"] for s in seen] == list(range(n_chunks))
    assert {s["chunk_total"] for s in seen} == {n_chunks}


def test_audio_is_queued_between_syntheses_not_batched_at_the_end(monkeypatch):
    """Time-to-first-sound is the metric; this pins it.

    An implementation that synthesized every chunk and only then queued
    them all would satisfy the ordering test above while delivering the
    first sound just as late as the old single-shot path. So assert the
    interleaving directly: each chunk must reach the queue BEFORE the next
    one is synthesized.
    """
    monkeypatch.setattr(_WH, "settings", _FakeSettings())
    timeline = []

    async def fake_synth(text, client, **kw):
        timeline.append(f"synth{kw['chunk_index']}")
        return {"audio_response": f"a{kw['chunk_index']}"}

    monkeypatch.setattr(_WH, "synthesize_tts_reply", fake_synth)

    class RecordingQueue(asyncio.Queue):
        async def put(self, item):
            timeline.append("queued" + item["audio_response"][1:])
            await super().put(item)

    n = len(chunk_text_for_speech(LONG_REPLY, first_chunk_chars=80, chunk_chars=280))
    assert n >= 3, "fixture must produce several chunks"
    asyncio.run(run_tts_remote(LONG_REPLY, object(), RecordingQueue()))

    expected = []
    for i in range(n):
        expected += [f"synth{i}", f"queued{i}"]
    assert timeline == expected


def test_stream_stops_on_mid_reply_failure(monkeypatch):
    """A hole in the middle of speech is worse than a short reply."""
    _patch(monkeypatch, results=[
        {"audio_response": "a0"},
        {"tts_error": "boom"},
        {"audio_response": "a2"},
    ])
    q = asyncio.Queue()
    asyncio.run(run_tts_remote(LONG_REPLY, object(), q))
    msgs = _drain(q)
    assert len(msgs) == 2
    assert msgs[0]["audio_response"] == "a0"
    assert msgs[1]["tts_error"] == "boom"
    assert msgs[1]["state"] == "idle"
    assert "a2" not in [m.get("audio_response") for m in msgs]


def test_disabled_flag_restores_single_shot(monkeypatch):
    """The kill switch must synthesize the WHOLE reply in one call."""
    class Off(_FakeSettings):
        HUB_TTS_STREAM_ENABLED = False

    seen = _patch(monkeypatch, results=[{"audio_response": "whole"}], settings_obj=Off())
    q = asyncio.Queue()
    asyncio.run(run_tts_remote(LONG_REPLY, object(), q))
    assert len(_drain(q)) == 1
    assert seen[0]["text"] == LONG_REPLY
    assert seen[0]["chunk_total"] == 1
