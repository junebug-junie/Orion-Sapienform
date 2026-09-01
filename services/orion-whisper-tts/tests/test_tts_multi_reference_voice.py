"""Multi-reference speaker resolution: a directory of wavs, not just one file.

Why this exists. XTTS truncates EACH reference independently to `max_ref_len`
(30s in the shipped checkpoint's config.json), so a single long recording is
silently cut to its first 30 seconds -- 27.8s of a 100.8s clean take was all
Orion's voice had ever used. `get_conditioning_latents` accepts a list and
means the per-file speaker embeddings, which both lifts that cap and averages
out per-clip codec artifacts that one clip would bake into the clone.

Before this change `_resolve_speaker_wav_path` returned a single Path and the
plan passed a single string, so there was no way to hand XTTS more than one
reference from any config surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.tts import (  # noqa: E402
    _resolve_speaker_wav_refs,
    _speaker_wav_refs_exist,
    resolve_synthesis_plan,
)


def _settings(**overrides):
    class S:
        tts_backend = "coqui"
        tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts_default_language = "en"
        tts_default_speaker = None
        tts_default_speaker_wav = None
        tts_split_sentences = True
        tts_voice_profile_dir = "/models/voices"

    for k, v in overrides.items():
        setattr(S, k, v)
    return S()


def _chunks(root: Path, names: list[str]) -> list[Path]:
    d = root / "orion_v3_chunks"
    d.mkdir()
    made = []
    for n in names:
        p = d / n
        p.write_bytes(b"RIFF")
        made.append(p)
    return made


def test_default_speaker_wav_directory_yields_every_wav(tmp_path) -> None:
    """The deployed shape: TTS_DEFAULT_SPEAKER_WAV points at a chunk dir."""
    _chunks(tmp_path, [f"chunk_{i}.wav" for i in range(1, 8)])

    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_v3_chunks",
        ),
        voice_id=None,
        language="en",
        options=None,
    )

    got = plan.kwargs["speaker_wav"]
    assert isinstance(got, list), got
    assert len(got) == 7
    assert [Path(p).name for p in got] == [f"chunk_{i}.wav" for i in range(1, 8)]
    assert plan.metadata["speaker_wav_count"] == 7
    assert plan.metadata["speaker_wav_used"] is True


def test_single_file_still_passes_a_bare_string(tmp_path) -> None:
    """Regression guard: the single-reference path must not change shape.

    Every existing deployment configures one file. If this started handing XTTS
    a one-element list the change would be invisible in tests but different in
    the live call.
    """
    wav = tmp_path / "orion_reference_v2.wav"
    wav.write_bytes(b"RIFF")

    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_reference_v2.wav",
        ),
        voice_id=None,
        language="en",
        options=None,
    )

    assert plan.kwargs["speaker_wav"] == str(wav.resolve())
    assert not isinstance(plan.kwargs["speaker_wav"], list)
    assert plan.metadata["speaker_wav_count"] == 1


def test_request_speaker_wav_directory_also_works(tmp_path) -> None:
    _chunks(tmp_path, ["a.wav", "b.wav"])

    plan = resolve_synthesis_plan(
        _settings(tts_voice_profile_dir=str(tmp_path)),
        voice_id=None,
        language="en",
        options={"speaker_wav": "orion_v3_chunks"},
    )

    assert len(plan.kwargs["speaker_wav"]) == 2
    assert plan.metadata["speaker_wav_count"] == 2


def test_voice_id_directory_also_works(tmp_path) -> None:
    _chunks(tmp_path, ["a.wav", "b.wav", "c.wav"])

    plan = resolve_synthesis_plan(
        _settings(tts_voice_profile_dir=str(tmp_path)),
        voice_id="orion_v3_chunks",
        language="en",
        options=None,
    )

    assert len(plan.kwargs["speaker_wav"]) == 3


def test_order_is_sorted_and_stable(tmp_path) -> None:
    """gpt_cond_len caps the concatenation, so which files come first matters.

    Directory iteration order is filesystem-dependent; without the sort the
    prosody latent would silently vary between hosts and between runs.
    """
    _chunks(tmp_path, ["chunk_3.wav", "chunk_1.wav", "chunk_2.wav"])

    first = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))
    second = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))

    assert [p.name for p in first] == ["chunk_1.wav", "chunk_2.wav", "chunk_3.wav"]
    assert first == second


def test_non_wav_files_are_ignored(tmp_path) -> None:
    """The dir sits under a :ro mount that also holds provenance text."""
    d = tmp_path / "orion_v3_chunks"
    d.mkdir()
    (d / "chunk_1.wav").write_bytes(b"RIFF")
    (d / "chunk_2.wav").write_bytes(b"RIFF")
    (d / "provenance.txt").write_text("built from ...", encoding="utf-8")
    (d / "source.mp3").write_bytes(b"ID3")

    refs = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))

    assert [p.name for p in refs] == ["chunk_1.wav", "chunk_2.wav"]


def test_empty_directory_is_an_error_not_a_silent_fallback(tmp_path) -> None:
    """An empty dir must not resolve to "no reference" and fall through.

    Falling back would hand XTTS a built-in speaker and change Orion's voice
    with nothing in the logs to say why.
    """
    (tmp_path / "orion_v3_chunks").mkdir()

    with pytest.raises(FileNotFoundError, match="holds no .wav files"):
        _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))


def test_directory_outside_the_profile_dir_is_refused(tmp_path) -> None:
    """Containment still applies -- a directory must not widen the escape."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.wav").write_bytes(b"RIFF")
    profile = tmp_path / "voices"
    profile.mkdir()

    with pytest.raises(ValueError, match="must be under"):
        _resolve_speaker_wav_refs(str(outside), str(profile))


def test_traversal_out_of_the_profile_dir_is_refused(tmp_path) -> None:
    profile = tmp_path / "voices"
    profile.mkdir()
    (tmp_path / "escape.wav").write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="must be under"):
        _resolve_speaker_wav_refs("../escape.wav", str(profile))


def test_missing_path_still_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_speaker_wav_refs("nope.wav", str(tmp_path))


def test_symlink_inside_the_directory_cannot_escape(tmp_path) -> None:
    """Containment regression guard for the directory branch.

    `glob` does not resolve symlinks and `is_file()` follows them, so the
    candidate-level `relative_to(root)` check does not cover children. Without
    a per-file check this hands an arbitrary out-of-root file to the audio
    decoder, and the decoder's error text reaches the bus as a system.error.
    """
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "private.wav").write_bytes(b"RIFF")
    root = tmp_path / "voices"
    root.mkdir()
    d = root / "chunks"
    d.mkdir()
    (d / "a.wav").write_bytes(b"RIFF")
    (d / "zz_escape.wav").symlink_to(secret / "private.wav")

    with pytest.raises(ValueError, match="escapes"):
        _resolve_speaker_wav_refs("chunks", str(root))


def test_symlinked_directory_itself_is_refused(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.wav").write_bytes(b"RIFF")
    root = tmp_path / "voices"
    root.mkdir()
    (root / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="must be under"):
        _resolve_speaker_wav_refs("link", str(root))


def test_profile_root_itself_is_refused(tmp_path) -> None:
    """`options.speaker_wav: "."` would blend every voice on the host.

    `relative_to` succeeds when resolved == root, so before this guard a bus
    request could average every reference in the profile dir into one
    composite speaker embedding -- a remotely triggerable voice change with
    nothing in the metadata to flag it.
    """
    (tmp_path / "orion_reference.wav").write_bytes(b"RIFF")
    (tmp_path / "orion_reference_v2.wav").write_bytes(b"RIFF")

    for candidate in (".", "sub/..", str(tmp_path)):
        with pytest.raises(ValueError, match="not the profile root"):
            _resolve_speaker_wav_refs(candidate, str(tmp_path))


def test_a_directory_named_like_a_wav_is_skipped(tmp_path) -> None:
    """`glob("*.wav")` matches directories too; only files are references."""
    d = tmp_path / "orion_v3_chunks"
    d.mkdir()
    (d / "chunk_1.wav").write_bytes(b"RIFF")
    (d / "fake.wav").mkdir()

    refs = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))

    assert [p.name for p in refs] == ["chunk_1.wav"]


def test_natural_sort_orders_chunk_2_before_chunk_10(tmp_path) -> None:
    """Lexicographic order silently reshuffles a set of ten or more.

    Order decides which files reach the prosody latent under gpt_cond_len, so
    plain sorted() would put chunk_10 second and change the voice with nothing
    to notice.
    """
    _chunks(tmp_path, [f"chunk_{i}.wav" for i in (1, 2, 3, 10, 11, 12)])

    refs = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))

    assert [p.name for p in refs] == [
        "chunk_1.wav", "chunk_2.wav", "chunk_3.wav",
        "chunk_10.wav", "chunk_11.wav", "chunk_12.wav",
    ]


def test_zero_padded_names_also_sort_correctly(tmp_path) -> None:
    _chunks(tmp_path, ["chunk_01.wav", "chunk_02.wav", "chunk_10.wav"])

    refs = _resolve_speaker_wav_refs("orion_v3_chunks", str(tmp_path))

    assert [p.name for p in refs] == [
        "chunk_01.wav", "chunk_02.wav", "chunk_10.wav",
    ]


def test_metadata_names_the_directory_not_the_first_chunk(tmp_path) -> None:
    """A 7-file voice must not read as an ordinary single-file one."""
    _chunks(tmp_path, [f"chunk_{i}.wav" for i in range(1, 8)])

    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_v3_chunks",
        ),
        voice_id=None,
        language="en",
        options=None,
    )

    assert plan.metadata["speaker_wav_basename"] == "orion_v3_chunks"
    assert plan.metadata["speaker_wav_count"] == 7
    assert plan.metadata["speaker_wav_basenames"][0] == "chunk_1.wav"
    assert plan.metadata["speaker_wav_source"] == "orion_v3_chunks"


def test_single_reference_metadata_is_unchanged(tmp_path) -> None:
    wav = tmp_path / "orion_reference_v2.wav"
    wav.write_bytes(b"RIFF")

    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_reference_v2.wav",
        ),
        voice_id=None,
        language="en",
        options=None,
    )

    assert plan.metadata["speaker_wav_basename"] == "orion_reference_v2.wav"
    assert plan.metadata["speaker_wav_basenames"] is None


def test_boot_validation_accepts_a_directory(tmp_path) -> None:
    """_validate_xtts_defaults runs on lazy engine construction, i.e. on
    Orion's first real turn after a deploy -- not at boot. If it rejected a
    directory the failure would surface as a dead first utterance."""
    from app.tts import _validate_xtts_defaults

    _chunks(tmp_path, ["chunk_1.wav", "chunk_2.wav"])
    _validate_xtts_defaults(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_v3_chunks",
        )
    )


def test_exists_check_is_true_for_a_real_multi_reference_list(tmp_path) -> None:
    """The diagnostic an operator reads first must not be permanently False.

    `Path(str(<list>))` is never a real path, so the previous inline expression
    logged speaker_wav_exists=False on every multi-reference synthesis while
    the files were sitting right there.
    """
    refs = _chunks(tmp_path, [f"chunk_{i}.wav" for i in range(1, 8)])

    assert _speaker_wav_refs_exist([str(p) for p in refs]) is True
    # and it must still be able to report a genuinely missing file
    assert _speaker_wav_refs_exist([str(refs[0]), str(tmp_path / "gone.wav")]) is False


def test_exists_check_survives_a_reference_list_past_name_max(tmp_path) -> None:
    """Above ~85 refs the stringified list exceeds NAME_MAX.

    `is_file()` RAISES ENAMETOOLONG rather than returning False -- ENAMETOOLONG
    is not in pathlib's ignored errnos. At the failure-logging site that would
    raise inside the `except`, replacing the real synthesis error.
    """
    many = [f"/models/voices/orion_v3_chunks/chunk_{i}.wav" for i in range(1, 121)]
    assert len(str(many)) > 255

    assert _speaker_wav_refs_exist(many) is False  # must not raise


def test_exists_check_handles_the_single_string_and_none(tmp_path) -> None:
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF")

    assert _speaker_wav_refs_exist(str(wav)) is True
    assert _speaker_wav_refs_exist(str(tmp_path / "nope.wav")) is False
    assert _speaker_wav_refs_exist(None) is False


def test_boot_validation_rejects_an_empty_directory(tmp_path) -> None:
    from app.tts import _validate_xtts_defaults

    (tmp_path / "orion_v3_chunks").mkdir()
    with pytest.raises(FileNotFoundError):
        _validate_xtts_defaults(
            _settings(
                tts_voice_profile_dir=str(tmp_path),
                tts_default_speaker_wav="orion_v3_chunks",
            )
        )
