"""Warm AffectGPT model wrapper.

Ports the exact working sequence proven live 2026-08-22 (see this service's
README "Provenance" section) from AffectGPT's own `inference_sample.py`
CLI script into a class that loads the 7B LLM + adapters ONCE at service
startup instead of once per request (~30s of checkpoint-shard loading each
time otherwise -- this was an open question in the original benchmark,
resolved here: yes, warm-load).

This module imports directly from the vendored AffectGPT source (cloned at
Docker build time, see Dockerfile -- pinned to commit
fffca794c6792023234565370e3f69f0488aede9 of
https://github.com/zeroQiaoba/AffectGPT.git). That source is NOT part of this
repo; treat AFFECTGPT_SRC_ROOT as an external dependency, same as any pip
package -- do not hand-edit files under it in this repo's checkout (they
don't exist here; they exist only inside the built image).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

from .face_extract import FaceExtractionResult
from .settings import Settings
from .transcribe import load_whisper_model, transcribe_audio

# NOT stdlib logging.getLogger -- see transcribe.py's own comment on this
# import for why (confirmed live 2026-08-22: stdlib log calls here were
# silently invisible in this service's actual container output).


@dataclass
class AssessmentResult:
    ok: bool
    raw_response: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    model_ckpt: Optional[str] = None
    face_or_frame_mode: Optional[str] = None
    face_detection: Optional[dict] = None
    timings: dict = field(default_factory=dict)
    subtitle_source: Optional[str] = None
    transcript: Optional[str] = None


class AffectGptRuntime:
    """Loads once in `load()`; `assess()` is safe to call repeatedly."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._loaded = False
        self._chat = None
        self._dataset_cls = None
        self._face_or_frame = settings.AFFECTGPT_FACE_OR_FRAME
        self._ckpt_path: Optional[str] = None
        self._whisper_model = None

    def load(self) -> None:
        s = self.settings
        src_root = s.AFFECTGPT_SRC_ROOT
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        # Imports deferred until src_root is on sys.path -- these packages
        # live in the vendored AffectGPT clone, not this repo. `import X`
        # (not `from X import *`) deliberately: upstream's own
        # inference_sample.py uses star-imports at true module level to
        # trigger each package's registry.register_*() decorator side
        # effects, but `from X import *` is a SyntaxError inside a function
        # body (confirmed live) -- a plain module import still runs the same
        # top-level registration code without that restriction.
        import config as affectgpt_config  # noqa: F401  (side effect: registers PATH_TO_* dicts consumed downstream)
        import my_affectgpt.tasks  # noqa: F401
        import my_affectgpt.models  # noqa: F401
        import my_affectgpt.runners  # noqa: F401
        import my_affectgpt.processors  # noqa: F401
        import my_affectgpt.datasets.builders  # noqa: F401
        import my_affectgpt.datasets.builders.image_text_pair_builder  # noqa: F401
        from my_affectgpt.common.config import Config
        from my_affectgpt.common.registry import registry
        from my_affectgpt.conversation.conversation_video import Chat
        from my_affectgpt.datasets.datasets.mer2025ov_dataset import MER2025OV_Dataset
        from my_affectgpt.processors.base_processor import BaseProcessor

        import torch

        # Determinism (partial -- see settings.py comment on the SDPA ceiling).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        cfg_path = os.path.join(src_root, s.AFFECTGPT_CFG_PATH)
        args = argparse.Namespace(
            cfg_path=cfg_path,
            options=[
                f"inference.ckpt_root={s.AFFECTGPT_CKPT_ROOT}",
                f"inference.gpu={s.AFFECTGPT_DEVICE.split(':')[-1]}",
            ],
        )
        old_cwd = os.getcwd()
        os.chdir(src_root)  # config.py's PATH_TO_* are relative to cwd
        try:
            cfg = Config(args)
            model_cfg = cfg.model_cfg
            datasets_cfg = cfg.datasets_cfg
            inference_cfg = cfg.inference_cfg

            ckpt_candidates = sorted(
                glob.glob("%s/*%06d*.pth" % (s.AFFECTGPT_CKPT_ROOT, s.AFFECTGPT_CKPT_EPOCH))
            )
            if len(ckpt_candidates) != 1:
                raise RuntimeError(
                    f"expected exactly 1 checkpoint for epoch {s.AFFECTGPT_CKPT_EPOCH} "
                    f"under {s.AFFECTGPT_CKPT_ROOT}, found {len(ckpt_candidates)}: {ckpt_candidates}"
                )
            self._ckpt_path = ckpt_candidates[0]
            model_cfg.ckpt_3 = self._ckpt_path

            model_cls = registry.get_model_class(model_cfg.arch)
            model = model_cls.from_config(model_cfg)
            model = model.to(s.AFFECTGPT_DEVICE).eval()
            self._chat = Chat(model, model_cfg, device=s.AFFECTGPT_DEVICE)

            dataset_cls = MER2025OV_Dataset()
            dataset_cls.needed_data = dataset_cls.get_needed_data(self._face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor")
            img_processor_cfg = inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(
                    vis_processor_cfg.train.name
                ).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(
                    img_processor_cfg.train.name
                ).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms
            self._dataset_cls = dataset_cls
        finally:
            os.chdir(old_cwd)

        self._loaded = True

        # Whisper is advisory, not core -- a load failure here must never
        # block readiness for the actual AffectGPT model (self._loaded is
        # already True above). assess() below checks self._whisper_model
        # is not None before ever using it, falling back to today's
        # subtitle="" behavior on any failure.
        if s.AFFECTGPT_TRANSCRIBE_ENABLED:
            try:
                self._whisper_model = load_whisper_model(
                    s.AFFECTGPT_WHISPER_MODEL, s.AFFECTGPT_DEVICE
                )
            except Exception as exc:  # noqa: BLE001
                self._whisper_model = None
                logger.warning(
                    f"[WARM] whisper_load_failed error={exc} -- subtitle transcription disabled for this session"
                )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def ckpt_path(self) -> Optional[str]:
        return self._ckpt_path

    def assess(
        self,
        *,
        face_result: FaceExtractionResult,
        audio_path: str,
        subtitle: str,
        user_message: Optional[str] = None,
    ) -> AssessmentResult:
        if not self._loaded:
            return AssessmentResult(ok=False, error="model not loaded", error_code="not_loaded")

        s = self.settings
        user_message = user_message or s.AFFECTGPT_DEFAULT_USER_MESSAGE
        timings: dict[str, Any] = {}
        t0 = time.monotonic()

        # Populate subtitle from Whisper ONLY when the caller supplied none
        # -- an explicit caller-provided subtitle always wins, this never
        # overwrites real text. Both Hub's ambient loop and its manual
        # "Check now" route currently never send one, so this is the
        # effective default path today, not an edge case.
        subtitle_source = "caller" if subtitle else "none"
        transcript: Optional[str] = None
        if not subtitle and self._whisper_model is not None:
            t_stt = time.monotonic()
            stt_result = transcribe_audio(
                self._whisper_model,
                audio_path,
                peak_threshold=s.AFFECTGPT_TRANSCRIBE_NEAR_SILENT_PEAK_INT16,
            )
            timings["transcribe_s"] = round(time.monotonic() - t_stt, 3)
            if stt_result.text:
                subtitle = stt_result.text
                transcript = stt_result.text
                subtitle_source = "transcribed"

        # AffectGPT's load_face() reads from a .npy path (np.load), not an
        # in-memory array -- round-trip through a scratch file for byte-exact
        # fidelity with the proven-working path rather than reimplementing
        # its uniform-sampling logic ourselves.
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
            np.save(tf, face_result.faces)
            face_npy_path = tf.name

        try:
            sample_data = self._dataset_cls.read_frame_face_audio_text(
                None, face_npy_path, audio_path, None
            )
            timings["data_load_s"] = round(time.monotonic() - t0, 3)

            t1 = time.monotonic()
            audio_hiddens, audio_llms = self._chat.postprocess_audio(sample_data)
            _, frame_llms = self._chat.postprocess_frame(sample_data)
            face_hiddens, face_llms = self._chat.postprocess_face(sample_data)
            _, image_llms = self._chat.postprocess_image(sample_data)
            multi_llms = None
            if self._face_or_frame.startswith("multiface"):
                _, multi_llms = self._chat.postprocess_multi(face_hiddens, audio_hiddens)
            elif self._face_or_frame.startswith("multiframe"):
                _, multi_llms = self._chat.postprocess_multi(frame_llms, audio_hiddens)

            img_list = {
                "audio": audio_llms,
                "frame": frame_llms,
                "face": face_llms,
                "image": image_llms,
                "multi": multi_llms,
            }
            prompt = self._dataset_cls.get_prompt_for_multimodal(
                self._face_or_frame, subtitle, user_message
            )
            timings["encode_s"] = round(time.monotonic() - t1, 3)

            t2 = time.monotonic()
            response = self._chat.answer_sample(
                prompt=prompt,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=s.AFFECTGPT_MAX_NEW_TOKENS,
                max_length=s.AFFECTGPT_MAX_LENGTH,
            )
            timings["generate_s"] = round(time.monotonic() - t2, 3)
            timings["total_s"] = round(time.monotonic() - t0, 3)

            return AssessmentResult(
                ok=True,
                raw_response=response,
                model_ckpt=os.path.basename(self._ckpt_path) if self._ckpt_path else None,
                face_or_frame_mode=self._face_or_frame,
                face_detection=face_result.as_meta(),
                timings=timings,
                subtitle_source=subtitle_source,
                transcript=transcript,
            )
        except Exception as exc:  # noqa: BLE001 -- boundary: never crash the service on a bad clip
            return AssessmentResult(
                ok=False,
                error=str(exc),
                error_code="inference_error",
                face_or_frame_mode=self._face_or_frame,
                face_detection=face_result.as_meta(),
                timings=timings,
                subtitle_source=subtitle_source,
                transcript=transcript,
            )
        finally:
            try:
                os.unlink(face_npy_path)
            except OSError:
                pass
