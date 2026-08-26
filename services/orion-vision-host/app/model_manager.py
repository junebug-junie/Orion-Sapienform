from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loguru import logger

import torch


@dataclass(frozen=True)
class ModelKey:
    profile: str
    device: str


class ModelManager:
    """
    Lazy per-(profile,device) loader for torch/transformers models.

    - Avoids duplicate loads under concurrency.
    - Keeps models resident once loaded.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._locks: Dict[ModelKey, threading.Lock] = {}
        self._models: Dict[ModelKey, Any] = {}
        self._processors: Dict[ModelKey, Any] = {}

    def _key_lock(self, key: ModelKey) -> threading.Lock:
        with self._lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, key: ModelKey) -> Tuple[Optional[Any], Optional[Any]]:
        return self._models.get(key), self._processors.get(key)

    def set(self, key: ModelKey, model: Any, processor: Any) -> None:
        self._models[key] = model
        self._processors[key] = processor

    @staticmethod
    def _torch_dtype(dtype: str, device: str) -> torch.dtype:
        # "auto" -> fp16 on CUDA, including for Qwen2-VL/Qwen2.5-VL -- review
        # finding, 2026-08-25: Qwen's own docs recommend bf16. Not applied
        # here: this fleet's GPUs (P100/V100, Pascal/Volta) predate Ampere,
        # the first generation with real bf16 tensor-core support -- fp16 is
        # the correct choice for this hardware, not an oversight. Live-
        # verified, not assumed: Qwen2-VL-2B-Instruct in fp16 on circe's
        # P100 (`auto` -> this branch) produced detailed, accurate, non-
        # degenerate captions/VQA answers the same day this was written
        # (see services/orion-vision-council/README.md's Foveal probe
        # section). Revisit only if this profile ever targets an Ampere+
        # card (VISION_DTYPE=bf16 already available as an explicit override
        # for that case -- no code change needed).
        dtype = (dtype or "auto").lower()
        if device.startswith("cuda"):
            if dtype in ("fp16", "float16"):
                return torch.float16
            if dtype in ("bf16", "bfloat16"):
                return torch.bfloat16
            if dtype in ("fp32", "float32"):
                return torch.float32
            return torch.float16  # auto default for CUDA
        return torch.float32

    def load_siglip_image_embedder(
        self,
        *,
        profile_name: str,
        device: str,
        dtype: str,
        model_id: str,
        fallback_model_id: str,
    ):
        """
        Loads SigLIP2 if possible; falls back to SigLIP.
        """
        from transformers import AutoProcessor, AutoModel

        key = ModelKey(profile=profile_name, device=device)
        lock = self._key_lock(key)

        with lock:
            m, p = self.get(key)
            if m is not None and p is not None:
                return m, p

            torch_dtype = self._torch_dtype(dtype, device)
            logger.info(f"[MODEL] loading embedder profile={profile_name} device={device} dtype={torch_dtype} id={model_id}")

            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype)
            except Exception as e:
                logger.warning(f"[MODEL] embedder load failed id={model_id} err={e}; falling back id={fallback_model_id}")
                processor = AutoProcessor.from_pretrained(fallback_model_id)
                model = AutoModel.from_pretrained(fallback_model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor

    def load_grounding_dino(
        self,
        *,
        profile_name: str,
        device: str,
        dtype: str,
        model_id: str,
    ):
        """
        Loads GroundingDINO open-vocab detector.
        """
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

        key = ModelKey(profile=profile_name, device=device)
        lock = self._key_lock(key)

        with lock:
            m, p = self.get(key)
            if m is not None and p is not None:
                return m, p

            torch_dtype = torch.float32
            if not device.startswith("cuda"):
                torch_dtype = self._torch_dtype(dtype, device)
            logger.info(f"[MODEL] loading grounding-dino profile={profile_name} device={device} dtype={torch_dtype} id={model_id}")

            processor = GroundingDinoProcessor.from_pretrained(model_id)
            model = GroundingDinoForObjectDetection.from_pretrained(model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor

    def load_face_identity_models(
        self,
        *,
        profile_name: str,
        device: str,
        torch_home: Optional[str] = None,
    ):
        """
        Loads the face-detection + embedding pair for identity_face
        (docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-
        situated-observation-design.md section 4): MTCNN (detect + align,
        facenet-pytorch) and InceptionResnetV1 pretrained on VGGFace2
        (512-dim embeddings). Small models, no language -- matches the
        design doc's "runs on the P4, small, no language" framing.

        Returns (resnet, mtcnn) -- reuses this manager's existing
        (model, processor) cache slot rather than adding a third dict for
        this one two-model case; MTCNN plays the "processor" role here
        (detect + align feeding the embedder), same shape every other
        loader in this file already uses.

        ``torch_home``, when given, is set as the ``TORCH_HOME`` env var
        before import -- facenet-pytorch's weights download via
        ``torch.hub``, which defaults to ``~/.cache/torch/hub`` and does
        NOT respect this service's own ``MODEL_CACHE_DIR``/``HF_HOME``
        conventions. Without this, the container would silently re-download
        ~107MB on every restart instead of using the persistent volume
        mount every other model loader in this file already gets for free.
        """
        import os

        if torch_home:
            os.environ.setdefault("TORCH_HOME", torch_home)

        from facenet_pytorch import MTCNN, InceptionResnetV1

        key = ModelKey(profile=profile_name, device=device)
        lock = self._key_lock(key)

        with lock:
            m, p = self.get(key)
            if m is not None and p is not None:
                return m, p

            logger.info(f"[MODEL] loading face-identity profile={profile_name} device={device}")

            mtcnn_device = device if device.startswith("cuda") else "cpu"
            mtcnn = MTCNN(keep_all=True, device=mtcnn_device)
            resnet = InceptionResnetV1(pretrained="vggface2").eval()
            if device.startswith("cuda"):
                resnet.to(device)

            self.set(key, resnet, mtcnn)
            return resnet, mtcnn

    def load_vlm_captioner(
        self,
        *,
        profile_name: str,
        device: str,
        dtype: str,
        model_id: str,
        qwen_min_pixels: Optional[int] = None,
        qwen_max_pixels: Optional[int] = None,
    ):
        """
        Loads a VLM for captioning (BLIP/BLIP2, Qwen2-VL/Qwen2.5-VL, or a
        generic AutoModelForVision2Seq fallback for anything else). Family
        is selected from ``model_id`` via ``vlm_family.py`` -- runner.py's
        prompt-building/decode path reads the same module so the two never
        drift on which model_ids count as which family.

        ``qwen_min_pixels``/``qwen_max_pixels`` only apply to the Qwen2-VL/
        Qwen2.5-VL branches (BLIP/BLIP2's processors don't accept them) --
        bounds the resolution-scaled visual-token count/VRAM those families'
        "naive dynamic resolution" processor would otherwise apply to
        whatever size image this service was handed, uncapped. Caller
        (runner.py) is expected to pass real values from settings; None
        falls back to the processor's own checkpoint-shipped default.
        """
        from transformers import AutoProcessor, AutoModelForVision2Seq

        from .vlm_family import is_qwen2_5_vl_model, is_qwen2_vl_model

        key = ModelKey(profile=profile_name, device=device)
        lock = self._key_lock(key)

        with lock:
            m, p = self.get(key)
            if m is not None and p is not None:
                return m, p

            torch_dtype = self._torch_dtype(dtype, device)
            logger.info(f"[MODEL] loading vlm profile={profile_name} device={device} dtype={torch_dtype} id={model_id}")

            mid = model_id.lower()
            if "blip2" in mid:
                from transformers import Blip2ForConditionalGeneration, Blip2Processor

                processor = Blip2Processor.from_pretrained(model_id)
                model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
            elif "blip" in mid:
                from transformers import BlipForConditionalGeneration, BlipProcessor

                processor = BlipProcessor.from_pretrained(model_id)
                model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
            elif is_qwen2_5_vl_model(model_id) or is_qwen2_vl_model(model_id):
                if is_qwen2_5_vl_model(model_id):
                    from transformers import Qwen2_5_VLForConditionalGeneration as _QwenModelClass
                else:
                    from transformers import Qwen2VLForConditionalGeneration as _QwenModelClass

                # Review finding: Qwen2VLImageProcessor.__init__ takes plain
                # `int` params, not Optional -- passing min_pixels=None/
                # max_pixels=None explicitly OVERRIDES the checkpoint's own
                # default with a literal None, which later blows up as
                # `TypeError: '>' not supported between 'int' and 'NoneType'`
                # inside transformers' smart_resize() the first time this
                # profile actually runs inference. Only pass the kwarg when
                # the caller gave a real value, so an explicit None caller
                # (this function's own documented contract) genuinely falls
                # through to the processor's checkpoint default instead of a
                # bound this loader silently poisoned to None.
                pixel_kwargs: dict[str, int] = {}
                if qwen_min_pixels is not None:
                    pixel_kwargs["min_pixels"] = qwen_min_pixels
                if qwen_max_pixels is not None:
                    pixel_kwargs["max_pixels"] = qwen_max_pixels

                processor = AutoProcessor.from_pretrained(model_id, **pixel_kwargs)
                model = _QwenModelClass.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor
