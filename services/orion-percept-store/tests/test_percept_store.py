"""Tests for the percept blob store.

The interesting assertions here are the ones about what the store REFUSES.
This holds camera frames of a private home, so "returns the right bytes" is
table stakes and "cannot be enumerated, cannot be traversed, does not outlive
its retention" is the actual contract.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.storage import SHA256_RE, PerceptStore, sniff_mime  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64
WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"fmt " + b"\x00" * 60
WAV_HEADER_ONLY_NO_FMT = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18" + b"ftyp" + b"isom" + b"\x00" * 64


@pytest.fixture()
def store(tmp_path) -> PerceptStore:
    return PerceptStore(tmp_path / "percepts")


# -- round trip -------------------------------------------------------------


def test_put_returns_the_content_hash_and_get_returns_the_bytes(store) -> None:
    sha = store.put(JPEG, mime="image/jpeg")
    assert sha == hashlib.sha256(JPEG).hexdigest()
    data, mime = store.get(sha)
    assert data == JPEG and mime == "image/jpeg"


def test_storing_the_same_frame_twice_is_idempotent(store) -> None:
    """Content-addressing means duplicate uploads are the common case here.

    The same camera pointed at the same still room produces identical frames
    constantly; two of them must collide on one blob, not two.
    """
    a = store.put(JPEG, mime="image/jpeg")
    b = store.put(JPEG, mime="image/jpeg")
    assert a == b
    assert len(list(store._iter_blobs())) == 1


# -- what it refuses --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "not-a-hash",
        "",
        "A" * 64,                    # uppercase is not our alphabet
        "0" * 63,                    # too short
        "0" * 65,                    # too long
        "0" * 63 + "g",              # not hex
        "../" + "0" * 61,
    ],
)
def test_get_refuses_anything_that_is_not_a_bare_hex_digest(store, bad) -> None:
    """The only caller-supplied component of a read is the digest.

    There is no directory to traverse and no filename to guess, which is what
    makes this store non-enumerable rather than merely obscure.
    """
    with pytest.raises(ValueError):
        store.get(bad)


def test_get_refuses_content_that_does_not_match_its_own_address(store) -> None:
    """A content-addressed store returning the wrong content is worse than one
    returning nothing. Simulates on-disk corruption or a swapped file."""
    sha = store.put(JPEG, mime="image/jpeg")
    store._blob(sha).write_bytes(PNG)          # same address, different bytes
    with pytest.raises(ValueError, match="hash mismatch"):
        store.get(sha)


def test_missing_blob_raises_keyerror_not_something_that_leaks(store) -> None:
    with pytest.raises(KeyError):
        store.get("0" * 64)


# -- mime sniffing ----------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected",
    [
        (JPEG, "image/jpeg"),
        (PNG, "image/png"),
        (WEBP, "image/webp"),
        (WAV, "audio/wav"),
        (MP4, "video/mp4"),
    ],
)
def test_sniffs_real_percept_types(data, expected) -> None:
    assert sniff_mime(data) == expected


def test_wav_and_webp_share_a_riff_container_but_are_distinguished(store) -> None:
    """Regression guard: both formats pass the same `RIFF` prefix check: only
    the fourcc at bytes 8-12 tells them apart. This used to be asserted
    backwards -- a WAV signature was in the "refuses to identify" list before
    audio/wav sniffing was added (2026-08-22)."""
    assert sniff_mime(WEBP) == "image/webp"
    assert sniff_mime(WAV) == "audio/wav"
    sha_wav = store.put(WAV, mime="audio/wav")
    data, mime = store.get(sha_wav)
    assert data == WAV and mime == "audio/wav"


def test_wav_header_with_no_fmt_subchunk_is_refused() -> None:
    """A bare RIFF/WAVE fourcc with no real audio data is not a usable WAV
    file, and used to sniff as one (review finding, 2026-08-22) -- it would
    have been stored and handed straight to orion-affectgpt-worker's audio
    decode. Not a full parser (still just a cheap sniff), but this specific
    degenerate case is now caught."""
    assert sniff_mime(WAV_HEADER_ONLY_NO_FMT) is None


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"GIF89a" + b"\x00" * 32,               # a real image type we do not accept
        b"#!/bin/sh\nrm -rf /\n",
        b"<?php echo 1; ?>",
        b"RIFF" + b"\x00" * 4 + b"AVI ",         # RIFF container, but neither WEBP nor WAVE
    ],
)
def test_refuses_to_identify_non_percepts(data) -> None:
    """The declared content-type is never trusted -- the sniffed value is what
    reaches a model prompt downstream."""
    assert sniff_mime(data) is None


# -- retention --------------------------------------------------------------


def test_sweep_removes_expired_blobs_and_their_mime_sidecars(store) -> None:
    sha = store.put(JPEG, mime="image/jpeg")
    assert store._blob(sha).exists() and store._mime(sha).exists()

    removed = store.sweep(max_age_seconds=3600, now=time.time() + 7200)
    assert removed == 1
    assert not store._blob(sha).exists()
    assert not store._mime(sha).exists(), "orphaned .mime sidecar left behind"
    with pytest.raises(KeyError):
        store.get(sha)


def test_sweep_keeps_blobs_inside_the_window(store) -> None:
    sha = store.put(JPEG, mime="image/jpeg")
    assert store.sweep(max_age_seconds=3600, now=time.time() + 60) == 0
    assert store.get(sha)[0] == JPEG


def test_restoring_a_frame_refreshes_its_clock(store) -> None:
    """Age means "not seen recently", not "first seen long ago".

    A frame still being requested must not be swept out from under an in-flight
    interpretation.
    """
    sha = store.put(JPEG, mime="image/jpeg")
    old = store._blob(sha).stat().st_mtime
    time.sleep(0.02)
    store.put(JPEG, mime="image/jpeg")
    assert store._blob(sha).stat().st_mtime > old


def test_sweep_ignores_foreign_files_in_the_root(store, tmp_path) -> None:
    """Never delete something this store did not write."""
    stray = store.root / "notes.txt"
    stray.write_text("not mine")
    store.put(JPEG, mime="image/jpeg")
    store.sweep(max_age_seconds=0, now=time.time() + 99999)
    assert stray.exists(), "swept a file outside the content-addressed namespace"


# -- stats ------------------------------------------------------------------


def test_stats_reports_size_without_enumerating_hashes(store) -> None:
    """/stats must be safe to expose; it must not become a listing endpoint."""
    store.put(JPEG, mime="image/jpeg")
    store.put(PNG, mime="image/png")
    s = store.stats()
    assert s["count"] == 2 and s["bytes"] > 0
    blob = "".join(str(v) for v in s.values())
    assert hashlib.sha256(JPEG).hexdigest() not in blob
