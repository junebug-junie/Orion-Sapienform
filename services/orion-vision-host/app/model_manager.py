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
    ):
        """
        Loads a VLM for captioning (e.g. IDEFICS, BLIP-2, Git, etc).
        Assumes standard transformers AutoProcessor/AutoModelForVision2Seq usage.
        """
        from transformers import AutoProcessor, AutoModelForVision2Seq

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
            else:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor
