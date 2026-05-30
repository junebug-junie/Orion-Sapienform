# Orion Whisper-TTS XTTS-v2 Voice-Aware Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `services/orion-whisper-tts` from single-speaker VITS to a production-grade, voice-aware TTS service using Coqui XTTS-v2 as the default local model, wiring `voice_id`, `language`, and `options` from bus payload through to synthesis with rich metadata on replies.

**Architecture:** Keep a thin `TTSEngine` facade in `app/tts.py` that selects a backend by `TTS_BACKEND` (Coqui first; HTTP backends later). Coqui backend loads the model once, resolves speaker/language/speaker_wav from request + settings, builds model-specific kwargs, calls `tts_to_file`, and returns a `TTSOutput` dataclass. `tts_worker.py` stays the only bus seam—no dict swamp. Legacy `{text, response_channel, trace_id}` callers keep working; typed envelopes get full `TTSResultPayload.metadata`.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic Settings, Coqui `TTS==0.22.0`, XTTS-v2, Redis bus (`OrionBusAsync`), Docker Compose + NVIDIA GPU runtime, pytest.

---

## File structure

| File | Responsibility |
|------|----------------|
| `services/orion-whisper-tts/app/settings.py` | New TTS env fields; XTTS-v2 defaults |
| `services/orion-whisper-tts/app/tts.py` | `TTSOutput`, voice/kwarg resolution, `CoquiBackend`, `TTSEngine` facade |
| `services/orion-whisper-tts/app/tts_worker.py` | Pass voice/language/options; publish metadata on legacy + typed replies |
| `services/orion-whisper-tts/app/main.py` | Log backend/model/gpu at startup (after engine init path is clear) |
| `services/orion-whisper-tts/.env_example` | XTTS defaults + mode comments |
| `services/orion-whisper-tts/.env` | Sync from `.env_example` (local only, gitignored) |
| `services/orion-whisper-tts/docker-compose.yml` | New env passthrough + `/models/voices` volume |
| `services/orion-whisper-tts/requirements.txt` | Pin/comment if XTTS deps need bumps (stay on `TTS==0.22.0` unless import fails) |
| `services/orion-whisper-tts/README.md` | Runbook, speakers, smoke, bus payload examples |
| `services/orion-whisper-tts/scripts/smoke_xtts.py` | Container-local synthesis smoke test |
| `services/orion-whisper-tts/tests/test_tts_voice_resolution.py` | Pure unit tests for kwarg resolution (no GPU) |
| `services/orion-whisper-tts/tests/test_tts_engine_settings.py` | Update defaults + static guards |
| `services/orion-whisper-tts/tests/test_tts_worker_replies.py` | Worker reply shape with mocked engine |
| `orion/schemas/tts.py` | **No change required** — already has `voice_id`, `language`, `options` |
| `orion/schemas/registry.py` | **No change required** |
| `orion/bus/channels.yaml` | **No change required** — `TTSRequestPayload` / `TTSResultPayload` already registered |

**Out of scope (explicit):** Fish-Speech, F5-TTS, Hub code changes (Hub `TTSClient.speak` already accepts full `TTSRequestPayload`).

---

### Task 0: Git worktree + feature branch

**Files:** (git only)

- [ ] **Step 1: Create worktree and branch**

Run from repo root:

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add ../Orion-Sapienform-xtts-v2 -b feat/whisper-tts-xtts-v2 origin/main
cd ../Orion-Sapienform-xtts-v2
```

Expected: new directory `../Orion-Sapienform-xtts-v2` on branch `feat/whisper-tts-xtts-v2`.

- [ ] **Step 2: Confirm isolation**

```bash
git status -sb
git branch --show-current
```

Expected: `feat/whisper-tts-xtts-v2`, clean or only intentional edits.

**Constraint:** Do not copy unrelated local changes into the worktree. Only `.env` may be synced locally (not committed).

---

### Task 1: Settings + env files

**Files:**
- Modify: `services/orion-whisper-tts/app/settings.py`
- Modify: `services/orion-whisper-tts/.env_example`
- Modify: `services/orion-whisper-tts/.env` (local sync, not committed)

- [ ] **Step 1: Write failing settings test**

Add to `services/orion-whisper-tts/tests/test_tts_engine_settings.py`:

```python
def test_tts_settings_xtts_defaults() -> None:
    mod = _load_settings_module()
    s = mod.Settings()
    assert s.tts_backend == "coqui"
    assert s.tts_model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert s.tts_default_language == "en"
    assert s.tts_split_sentences is True
    assert s.tts_voice_profile_dir == "/models/voices"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/orion-whisper-tts
python -m pytest tests/test_tts_engine_settings.py::test_tts_settings_xtts_defaults -v
```

Expected: `AssertionError` on `tts_model_name` (still `ljspeech/vits`).

- [ ] **Step 3: Implement settings**

Replace TTS block in `services/orion-whisper-tts/app/settings.py`:

```python
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ... existing identity/bus/channel fields unchanged ...

    # TTS config
    tts_backend: str = Field("coqui", env="TTS_BACKEND")
    tts_model_name: str = Field(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        env="TTS_MODEL_NAME",
    )
    tts_use_gpu: bool = Field(True, env="TTS_USE_GPU")
    tts_default_language: str = Field("en", env="TTS_DEFAULT_LANGUAGE")
    tts_default_speaker: Optional[str] = Field(None, env="TTS_DEFAULT_SPEAKER")
    tts_default_speaker_wav: Optional[str] = Field(None, env="TTS_DEFAULT_SPEAKER_WAV")
    tts_split_sentences: bool = Field(True, env="TTS_SPLIT_SENTENCES")
    tts_voice_profile_dir: str = Field("/models/voices", env="TTS_VOICE_PROFILE_DIR")

    whisper_tts_stt_timeout_sec: float = Field(90.0, env="WHISPER_TTS_STT_TIMEOUT_SEC")
    whisper_tts_synth_timeout_sec: float = Field(120.0, env="WHISPER_TTS_SYNTH_TIMEOUT_SEC")
```

- [ ] **Step 4: Update `.env_example`**

Replace TTS section with:

```bash
# TTS backend (coqui today; fish_speech_http / f5_tts_http later)
TTS_BACKEND=coqui
TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2
TTS_USE_GPU=true
TTS_DEFAULT_LANGUAGE=en
# Mode 1 — built-in XTTS speaker name (list via smoke script / Coqui API)
TTS_DEFAULT_SPEAKER=
# Mode 2 — reference voice cloning (.wav under TTS_VOICE_PROFILE_DIR)
TTS_DEFAULT_SPEAKER_WAV=/models/voices/orion_reference.wav
TTS_SPLIT_SENTENCES=true
TTS_VOICE_PROFILE_DIR=/models/voices
```

- [ ] **Step 5: Sync local `.env`**

```bash
cp services/orion-whisper-tts/.env_example services/orion-whisper-tts/.env
# Preserve any machine-specific ORION_BUS_URL / GPU_INDEX from existing .env if needed
```

- [ ] **Step 6: Run test — expect PASS**

```bash
python -m pytest tests/test_tts_engine_settings.py::test_tts_settings_xtts_defaults -v
```

- [ ] **Step 7: Commit**

```bash
git add services/orion-whisper-tts/app/settings.py \
        services/orion-whisper-tts/.env_example \
        services/orion-whisper-tts/tests/test_tts_engine_settings.py
git commit -m "feat(whisper-tts): add XTTS-oriented TTS settings"
```

---

### Task 2: Voice resolution + `TTSOutput` (pure logic, TDD)

**Files:**
- Modify: `services/orion-whisper-tts/app/tts.py`
- Create: `services/orion-whisper-tts/tests/test_tts_voice_resolution.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-whisper-tts/tests/test_tts_voice_resolution.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.tts import resolve_synthesis_plan  # noqa: E402


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


def test_options_speaker_wav_wins() -> None:
    plan = resolve_synthesis_plan(
        _settings(),
        voice_id="ignored",
        language=None,
        options={"speaker_wav": "/tmp/ref.wav", "language": "fr"},
    )
    assert plan.kwargs["speaker_wav"] == "/tmp/ref.wav"
    assert plan.kwargs["language"] == "fr"
    assert plan.metadata["speaker_wav_used"] is True


def test_voice_id_resolves_under_profile_dir(tmp_path) -> None:
    wav = tmp_path / "orion_reference.wav"
    wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(tts_voice_profile_dir=str(tmp_path)),
        voice_id="orion_reference.wav",
        language="en",
        options=None,
    )
    assert plan.kwargs["speaker_wav"] == str(wav)
    assert plan.metadata["speaker_wav_used"] is True


def test_voice_id_passed_as_coqui_speaker_when_not_a_file() -> None:
    plan = resolve_synthesis_plan(
        _settings(tts_default_speaker="Claribel Dervla"),
        voice_id="Ana Florence",
        language="en",
        options=None,
    )
    assert plan.kwargs.get("speaker") == "Ana Florence"
    assert "speaker_wav" not in plan.kwargs
    assert plan.metadata["speaker_wav_used"] is False


def test_default_speaker_wav_from_settings() -> None:
    plan = resolve_synthesis_plan(
        _settings(tts_default_speaker_wav="/models/voices/orion_reference.wav"),
        voice_id=None,
        language=None,
        options=None,
    )
    assert plan.kwargs["speaker_wav"] == "/models/voices/orion_reference.wav"


def test_missing_speaker_wav_raises_loud(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="speaker_wav"):
        resolve_synthesis_plan(
            _settings(tts_voice_profile_dir=str(tmp_path)),
            voice_id="missing.wav",
            language="en",
            options=None,
        )
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd services/orion-whisper-tts
python -m pytest tests/test_tts_voice_resolution.py -v
```

Expected: `ImportError: cannot import name 'resolve_synthesis_plan'`.

- [ ] **Step 3: Implement resolution + dataclasses in `tts.py`**

Replace `services/orion-whisper-tts/app/tts.py` with (core structure):

```python
from __future__ import annotations

import base64
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from TTS.api import TTS

from .settings import Settings, settings

logger = logging.getLogger(__name__)


@dataclass
class TTSOutput:
    audio_b64: str
    content_type: str = "audio/wav"
    metadata: dict = field(default_factory=dict)
    duration_sec: Optional[float] = None


@dataclass
class SynthesisPlan:
    kwargs: dict[str, Any]
    metadata: dict[str, Any]


def _is_xtts_model(model_name: str) -> bool:
    return "xtts" in model_name.lower()


def _resolve_speaker_wav_path(
    candidate: str,
    profile_dir: str,
) -> Path:
    p = Path(candidate)
    if not p.is_absolute():
        p = Path(profile_dir) / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"TTS speaker_wav not found: {p} (profile_dir={profile_dir})"
        )
    return p


def resolve_synthesis_plan(
    cfg: Settings,
    *,
    voice_id: Optional[str],
    language: Optional[str],
    options: Optional[dict],
) -> SynthesisPlan:
    opts = dict(options or {})
    lang = (
        language
        or opts.pop("language", None)
        or cfg.tts_default_language
    )
    split = bool(opts.pop("split_sentences", cfg.tts_split_sentences))

    speaker_wav: Optional[str] = opts.pop("speaker_wav", None)
    speaker: Optional[str] = opts.pop("speaker", None)
    speaker_wav_used = False

    if speaker_wav:
        _resolve_speaker_wav_path(speaker_wav, cfg.tts_voice_profile_dir)
        speaker_wav_used = True
    elif cfg.tts_default_speaker_wav:
        speaker_wav = cfg.tts_default_speaker_wav
        _resolve_speaker_wav_path(speaker_wav, cfg.tts_voice_profile_dir)
        speaker_wav_used = True
    elif voice_id:
        looks_like_file = (
            voice_id.startswith("/")
            or voice_id.endswith(".wav")
            or Path(voice_id).suffix.lower() == ".wav"
        )
        profile_candidate = Path(cfg.tts_voice_profile_dir) / voice_id
        if looks_like_file or profile_candidate.exists():
            resolved = _resolve_speaker_wav_path(voice_id, cfg.tts_voice_profile_dir)
            speaker_wav = str(resolved)
            speaker_wav_used = True
        else:
            speaker = voice_id
    elif cfg.tts_default_speaker:
        speaker = cfg.tts_default_speaker

    kwargs: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "backend": cfg.tts_backend,
        "model_name": cfg.tts_model_name,
        "language": lang,
        "voice_id": voice_id,
        "speaker": speaker,
        "speaker_wav_used": speaker_wav_used,
        "split_sentences": split,
    }

    if _is_xtts_model(cfg.tts_model_name):
        kwargs["language"] = lang
        kwargs["split_sentences"] = split
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker:
            kwargs["speaker"] = speaker
        elif not speaker_wav:
            raise ValueError(
                "XTTS requires a speaker: set TTS_DEFAULT_SPEAKER, "
                "TTS_DEFAULT_SPEAKER_WAV, voice_id, or options.speaker_wav"
            )
    else:
        # Legacy single-speaker models (only if explicitly configured)
        if speaker:
            kwargs["speaker"] = speaker

    # Pass through only known-safe extras for coqui (ignore unknown keys in opts with warning)
    for key in list(opts.keys()):
        logger.warning("[TTS] Ignoring unsupported option key=%s", key)

    return SynthesisPlan(kwargs=kwargs, metadata=meta)


class CoquiBackend:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        logger.info(
            "[TTS] Loading coqui model=%s gpu=%s",
            cfg.tts_model_name,
            cfg.tts_use_gpu,
        )
        self.tts = TTS(cfg.tts_model_name, gpu=cfg.tts_use_gpu)
        logger.info("[TTS] Coqui model loaded.")

    def synthesize(
        self,
        text: str,
        *,
        voice_id: Optional[str],
        language: Optional[str],
        options: Optional[dict],
    ) -> TTSOutput:
        if not text:
            return TTSOutput(audio_b64="", metadata={"empty_text": True})

        plan = resolve_synthesis_plan(
            self.cfg,
            voice_id=voice_id,
            language=language,
            options=options,
        )
        started = time.perf_counter()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            call_kwargs = {"text": text, "file_path": f.name, **plan.kwargs}
            try:
                self.tts.tts_to_file(**call_kwargs)
            except TypeError as exc:
                logger.error(
                    "[TTS] Coqui tts_to_file rejected kwargs=%s model=%s error=%s",
                    sorted(plan.kwargs.keys()),
                    self.cfg.tts_model_name,
                    exc,
                )
                raise RuntimeError(
                    f"Coqui synthesis failed for model={self.cfg.tts_model_name}: {exc}. "
                    f"kwargs={sorted(plan.kwargs.keys())}"
                ) from exc
            f.seek(0)
            audio_bytes = f.read()

        duration_sec: Optional[float] = None
        try:
            with wave.open(f.name, "rb") as wf:
                duration_sec = wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

        synthesis_ms = int((time.perf_counter() - started) * 1000)
        metadata = {
            **plan.metadata,
            "synthesis_ms": synthesis_ms,
            "gpu_enabled": self.cfg.tts_use_gpu,
        }
        return TTSOutput(
            audio_b64=base64.b64encode(audio_bytes).decode("utf-8"),
            content_type="audio/wav",
            metadata=metadata,
            duration_sec=duration_sec,
        )


class TTSEngine:
    """Facade: select backend by TTS_BACKEND."""

    def __init__(self):
        backend = (settings.tts_backend or "coqui").strip().lower()
        if backend == "coqui":
            self._backend = CoquiBackend(settings)
            self.backend_name = "coqui"
        else:
            raise RuntimeError(
                f"Unsupported TTS_BACKEND={settings.tts_backend!r}. "
                "Supported: coqui"
            )
        logger.info(
            "[TTS] Engine ready backend=%s model=%s gpu=%s",
            self.backend_name,
            settings.tts_model_name,
            settings.tts_use_gpu,
        )

    def synthesize_to_b64(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> TTSOutput:
        return self._backend.synthesize(
            text,
            voice_id=voice_id,
            language=language,
            options=options,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_tts_voice_resolution.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-whisper-tts/app/tts.py \
        services/orion-whisper-tts/tests/test_tts_voice_resolution.py
git commit -m "feat(whisper-tts): voice-aware Coqui synthesis plan and TTSOutput"
```

---

### Task 3: Wire `tts_worker.py` + legacy `options`

**Files:**
- Modify: `services/orion-whisper-tts/app/tts_worker.py`

- [ ] **Step 1: Write failing worker test**

Create `services/orion-whisper-tts/tests/test_tts_worker_replies.py`:

```python
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tts import TTSOutput


@pytest.mark.asyncio
async def test_typed_reply_includes_metadata():
    from app.tts_worker import process_tts_request
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.tts import TTSRequestPayload

    bus = AsyncMock()
    envelope = BaseEnvelope(
        kind="tts.synthesize.request",
        source=ServiceRef(name="hub", version="0.1.0"),
        correlation_id="cid-1",
        reply_to="orion:tts:result:test",
        payload=TTSRequestPayload(
            text="hi",
            voice_id="Ana Florence",
            language="en",
        ).model_dump(),
    )
    fake = TTSOutput(
        audio_b64="YWJj",
        metadata={"backend": "coqui", "model_name": "xtts", "language": "en"},
        duration_sec=1.0,
    )

    with patch("app.tts_worker.get_tts_engine") as get_engine:
        get_engine.return_value.synthesize_to_b64.return_value = fake
        with patch("app.tts_worker.settings") as st:
            st.whisper_tts_synth_timeout_sec = 30.0
            st.service_name = "whisper-tts"
            st.service_version = "0.1.0"
            await process_tts_request(bus, envelope, {})

    bus.publish.assert_awaited_once()
    published = bus.publish.await_args[0][1]
    assert published.payload["audio_b64"] == "YWJj"
    assert published.payload["metadata"]["backend"] == "coqui"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
python -m pytest tests/test_tts_worker_replies.py -v
```

- [ ] **Step 3: Patch worker**

In `process_tts_request`:

1. Legacy parse — add `options=payload_src.get("options")` to `TTSRequestPayload(...)`.
2. Replace synthesis block:

```python
        def _synthesize():
            engine = get_tts_engine()
            return engine.synthesize_to_b64(
                request_data.text,
                voice_id=request_data.voice_id,
                language=request_data.language,
                options=request_data.options,
            )

        result: TTSOutput = await asyncio.wait_for(
            loop.run_in_executor(None, _synthesize),
            timeout=float(settings.whisper_tts_synth_timeout_sec),
        )
        audio_b64 = result.audio_b64
        metadata = result.metadata or {}
```

3. Legacy reply:

```python
            legacy_reply = {
                "trace_id": final_trace_id,
                "audio_b64": audio_b64,
                "mime_type": result.content_type or "audio/wav",
                "metadata": metadata,
            }
```

4. Typed reply:

```python
            result_payload = TTSResultPayload(
                audio_b64=audio_b64,
                content_type=result.content_type,
                duration_sec=result.duration_sec,
                metadata=metadata,
            )
```

Add import: `from .tts import TTSEngine, TTSOutput` (or `TTSOutput` only if engine type not needed).

- [ ] **Step 4: Run test — expect PASS**

```bash
python -m pytest tests/test_tts_worker_replies.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-whisper-tts/app/tts_worker.py \
        services/orion-whisper-tts/tests/test_tts_worker_replies.py
git commit -m "feat(whisper-tts): wire voice-aware synthesis through TTS worker"
```

---

### Task 4: Docker Compose + README + smoke script

**Files:**
- Modify: `services/orion-whisper-tts/docker-compose.yml`
- Modify: `services/orion-whisper-tts/README.md`
- Create: `services/orion-whisper-tts/scripts/smoke_xtts.py`
- Modify: `services/orion-whisper-tts/tests/test_tts_engine_settings.py` (compose default assertions)

- [ ] **Step 1: Update `docker-compose.yml`**

Under `environment`, replace TTS lines with:

```yaml
      - TTS_BACKEND=${TTS_BACKEND:-coqui}
      - TTS_MODEL_NAME=${TTS_MODEL_NAME:-tts_models/multilingual/multi-dataset/xtts_v2}
      - TTS_USE_GPU=${TTS_USE_GPU:-true}
      - TTS_DEFAULT_LANGUAGE=${TTS_DEFAULT_LANGUAGE:-en}
      - TTS_DEFAULT_SPEAKER=${TTS_DEFAULT_SPEAKER:-}
      - TTS_DEFAULT_SPEAKER_WAV=${TTS_DEFAULT_SPEAKER_WAV:-}
      - TTS_SPLIT_SENTENCES=${TTS_SPLIT_SENTENCES:-true}
      - TTS_VOICE_PROFILE_DIR=${TTS_VOICE_PROFILE_DIR:-/models/voices}
```

Under `volumes`, add (adjust host path to your machine):

```yaml
      - ${TTS_VOICE_PROFILE_HOST_DIR:-/mnt/telemetry/models/coqui/voices}:/models/voices:ro
```

Keep existing Coqui model cache mount and GPU blocks unchanged.

- [ ] **Step 2: Create smoke script**

`services/orion-whisper-tts/scripts/smoke_xtts.py`:

```python
#!/usr/bin/env python3
"""Synthesize a short XTTS smoke WAV inside the container."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.parent))  # /app parent when copied flat
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import Settings
from app.tts import TTSEngine
import base64


def main() -> int:
    out = Path(os.getenv("SMOKE_OUTPUT", "/tmp/orion_xtts_smoke.wav"))
    text = os.getenv(
        "SMOKE_TEXT",
        "Hello Juniper. This is Orion testing the upgraded voice engine.",
    )
    cfg = Settings()
    engine = TTSEngine()
    result = engine.synthesize_to_b64(
        text,
        voice_id=os.getenv("TTS_DEFAULT_SPEAKER") or None,
        language=os.getenv("TTS_DEFAULT_LANGUAGE", cfg.tts_default_language),
        options={
            "speaker_wav": os.getenv("TTS_DEFAULT_SPEAKER_WAV"),
            "split_sentences": cfg.tts_split_sentences,
        }
        if os.getenv("TTS_DEFAULT_SPEAKER_WAV")
        else None,
    )
    out.write_bytes(base64.b64decode(result.audio_b64))
    print(f"Wrote {out} metadata={result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Expand README**

Document:
- `docker compose -f services/orion-whisper-tts/docker-compose.yml up -d --build`
- Startup logs: backend, model, gpu
- List speakers: `python -c "from TTS.api import TTS; t=TTS('tts_models/multilingual/multi-dataset/xtts_v2'); print(t.speakers)"` (inside container)
- Smoke: `python scripts/smoke_xtts.py`
- Bus examples (text-only, voice_id, options.speaker_wav, language) — use user-provided JSON verbatim

- [ ] **Step 4: Fix static test expectations**

Update `test_tts_engine_settings.py`:
- `test_whisper_tts_timeout_settings_defaults`: expect `xtts_v2` model
- `test_tts_engine_reads_settings_not_hardcoded_gpu`: assert `xtts_v2` in settings/compose, not `ljspeech/vits`

- [ ] **Step 5: Run unit tests**

```bash
cd services/orion-whisper-tts
python -m pytest tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-whisper-tts/docker-compose.yml \
        services/orion-whisper-tts/README.md \
        services/orion-whisper-tts/scripts/smoke_xtts.py \
        services/orion-whisper-tts/tests/test_tts_engine_settings.py
git commit -m "chore(whisper-tts): docker voice mount, smoke script, and docs"
```

---

### Task 5: Acceptance verification (GPU host)

**Files:** none (commands only)

- [ ] **Step 1: compileall**

```bash
python -m compileall services/orion-whisper-tts/app orion/schemas
```

Expected: no syntax errors.

- [ ] **Step 2: Build container**

```bash
cd services/orion-whisper-tts
docker compose build whisper-tts
```

Expected: image builds (may take several minutes; downloads XTTS weights on first run).

- [ ] **Step 3: Start service and check logs**

```bash
docker compose up -d whisper-tts
docker compose logs whisper-tts | tail -n 80
```

Expected log lines include `backend=coqui`, `model=tts_models/multilingual/multi-dataset/xtts_v2`, `gpu=True`.

- [ ] **Step 4: Smoke synthesis**

```bash
docker compose exec whisper-tts python3 scripts/smoke_xtts.py
docker compose exec whisper-tts ls -la /tmp/orion_xtts_smoke.wav
```

Expected: non-zero WAV file.

- [ ] **Step 5: Bus legacy RPC test**

From a host with Redis + running hub/bus CLI (adjust URL):

```bash
TRACE=$(uuidgen)
REPLY="orion:tts:result:manual-${TRACE}"
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE "$REPLY" &
sleep 1
redis-cli -u "$ORION_BUS_URL" PUBLISH orion:tts:intake '{"kind":"legacy.message","payload":{"text":"Hello legacy","response_channel":"'"$REPLY"'","trace_id":"'"$TRACE"'"}}'
```

Expected JSON with `audio_b64`, `mime_type`, `metadata.backend`.

- [ ] **Step 6: Bus typed RPC test**

Use Hub `TTSClient` or publish `tts.synthesize.request` envelope with full payload (see README example with `voice_id` + `options.speaker_wav`).

Expected: `tts.synthesize.result` with `metadata.speaker_wav_used=true` when wav provided.

---

### Task 6: Code review subagent + PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-29-whisper-tts-xtts-v2-pr.md`

- [ ] **Step 1: Run code-reviewer subagent**

Use `superpowers:requesting-code-review` / `code-reviewer` subagent on the full diff vs `main`. Fix all blocking issues.

- [ ] **Step 2: Final commit if fixes applied**

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/whisper-tts-xtts-v2
gh pr create --title "feat(whisper-tts): XTTS-v2 voice-aware TTS" --body "$(cat <<'EOF'
## Summary
- Default Coqui backend to XTTS-v2 with voice_id / language / options wired end-to-end.
- Reference voice via speaker_wav; structured metadata on legacy and typed bus replies.
- Docker voice profile mount, smoke script, and README bus examples.

## Test plan
- [ ] `python -m pytest services/orion-whisper-tts/tests/ -v`
- [ ] `python -m compileall services/orion-whisper-tts/app orion/schemas`
- [ ] `docker compose -f services/orion-whisper-tts/docker-compose.yml build`
- [ ] Service logs show backend/model/gpu
- [ ] `docker compose exec whisper-tts python3 scripts/smoke_xtts.py`
- [ ] Legacy + typed bus requests return audio with metadata

EOF
)"
```

---

## Post-implementation deliverables (for PR description report)

Include in `docs/superpowers/pr-reports/2026-05-29-whisper-tts-xtts-v2-pr.md`:

| Item | Value |
|------|-------|
| Rebuild | `cd services/orion-whisper-tts && docker compose build whisper-tts && docker compose up -d whisper-tts` |
| Smoke | `docker compose exec whisper-tts python3 scripts/smoke_xtts.py` |
| Unit tests | `cd services/orion-whisper-tts && python -m pytest tests/ -v` |
| Known risks | First XTTS load is slow/large; requires GPU RAM; XTTS needs speaker or speaker_wav; torch/CUDA mismatch fails loudly at startup |

---

## Self-review

### Spec coverage

| Requirement | Task |
|-------------|------|
| TTS backend settings | Task 1 |
| Coqui + XTTS default | Tasks 1–2 |
| voice_id/language/options wired | Tasks 2–3 |
| speaker_wav reference voice | Task 2 |
| synthesis metadata | Tasks 2–3 |
| Legacy `{text, response_channel, trace_id}` | Task 3 |
| No silent ljspeech fallback | Task 1 defaults |
| Backend seam for future HTTP backends | Task 2 `TTSEngine` |
| Docker `/models/voices` mount | Task 4 |
| Smoke script | Task 4 |
| README bus examples | Task 4 |
| Acceptance tests | Task 5 |
| channels/registry | No change (documented) |

### Placeholder scan

No TBD/TODO steps. All code blocks are complete starter implementations.

### Type consistency

- `synthesize_to_b64` returns `TTSOutput` everywhere.
- Worker uses `result.content_type`, `result.duration_sec`, `result.metadata`.
- `resolve_synthesis_plan` is the single voice resolution entry point.
