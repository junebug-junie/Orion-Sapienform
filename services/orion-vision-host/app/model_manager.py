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

            torch_dtype = self._torch_dtype(dtype, device)
            logger.info(f"[MODEL] loading grounding-dino profile={profile_name} device={device} dtype={torch_dtype} id={model_id}")

            processor = GroundingDinoProcessor.from_pretrained(model_id)
            model = GroundingDinoForObjectDetection.from_pretrained(model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor

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

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)

            model.eval()
            if device.startswith("cuda"):
                model.to(device)

            self.set(key, model, processor)
            return model, processor
