"""OSNet shadow body vector for person crops (2026-10-16, Juniper approved).

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "A better body
vector". The live body vector is SigLIP (a picture-and-caption model). OSNet is
a person re-identification model. For about a week both vectors are stored on
every person crop so ``services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py``
can score them against each other. SHADOW ONLY: nothing reads the OSNet vector
to assign a crop to an individual.

Weights: OSNet-AIN x1.0, multi-source (MSMT17 + DukeMTMC-reID + CUHK03) from the
Torchreid model zoo, as converted (classifier head stripped, keys unchanged) in
huggingface.co/LibreYOLO/LibreReID-osnet at revision ``WEIGHTS_REVISION``. The
Dockerfile downloads the file at build time and checks ``WEIGHTS_SHA256``; this
module checks it again before loading and loads with ``weights_only=True`` (no
pickled code runs). Not committed to git.

Failure is contained: if the weights are missing, the wrong file, or do not fit
the network, ``OsnetShadow`` records the error, logs it at ERROR once per retry
window, and returns no vectors. Detection and the SigLIP vector never depend on
it. ``status()`` is what /health shows.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger
from PIL import Image

ARCH = "osnet_ain_x1_0"
WEIGHTS_REPO = "LibreYOLO/LibreReID-osnet"
WEIGHTS_REVISION = "5c7c20e54ccf80c9889a64020748f148ad5f7634"
WEIGHTS_FILE = "osnet_ain_x1_0.pt"
WEIGHTS_SHA256 = "34c24e98b6b70c8b62480f846fd0d581aa2fd1535bc0276aecf1f10430b731d1"
DEFAULT_WEIGHTS_PATH = "/app/models/osnet/osnet_ain_x1_0.pt"
# What a stored vector says it is: architecture, training sources, weights hash.
MODEL_ID = f"{ARCH}:ms_d_c:{WEIGHTS_SHA256[:12]}"
FEATURE_DIM = 512

# Torchreid's own test transform: resize to 256 x 128 (height x width), ImageNet mean/std.
INPUT_HW = (256, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RETRY_AFTER_SEC = 600.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(crops: List[Image.Image]):
    """Crops -> float32 (N, 3, 256, 128) tensor, Torchreid's test transform."""
    import torch

    h, w = INPUT_HW
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arrs = []
    for crop in crops:
        img = crop.convert("RGB").resize((w, h), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arrs.append((a - mean) / std)
    return torch.from_numpy(np.stack(arrs, axis=0))


def load_osnet(weights_path: Path, device: str):
    """Build OSNet-AIN x1.0 and load the pinned weights. Raises on any mismatch."""
    import torch

    from .osnet_ain import osnet_ain_x1_0

    if not weights_path.is_file():
        raise FileNotFoundError(f"OSNet weights not found at {weights_path}")
    digest = sha256_file(weights_path)
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"OSNet weights sha256 {digest} != pinned {WEIGHTS_SHA256}")
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model = osnet_ain_x1_0()
    model.load_state_dict(state, strict=True)  # every layer, nothing extra
    model.eval()
    # fp32 on purpose: the network is small (2.2 M parameters) and its
    # InstanceNorm layers are where fp16 drifts.
    model.to(device)
    return model


class OsnetShadow:
    """Lazy, per-device OSNet with a failure state that never raises to the caller."""

    def __init__(
        self,
        *,
        enabled: bool,
        weights_path: str,
        loader: Optional[Callable[[Path, str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS_PATH)
        self._loader = loader or load_osnet
        self._clock = clock
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self._failed_at: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.vectors = 0
        self.failures = 0
        self.vram_mb: Dict[str, float] = {}
        if not self.enabled:
            logger.info("[OSNET] shadow body vector disabled (VISION_OSNET_SHADOW_ENABLED=false)")

    def _model(self, device: str):
        with self._lock:
            m = self._models.get(device)
            if m is not None:
                return m
            failed = self._failed_at.get(device)
            if failed is not None and self._clock() - failed < RETRY_AFTER_SEC:
                return None
            before = _cuda_allocated_mb(device)
            try:
                m = self._loader(self.weights_path, device)
            except Exception as exc:
                self._failed_at[device] = self._clock()
                self.failures += 1
                self.last_error = f"load: {exc}"[:300]
                logger.error(
                    f"[OSNET] load FAILED device={device} path={self.weights_path} err={exc}; "
                    f"person crops keep their SigLIP vector and get no OSNet vector "
                    f"(retry in {RETRY_AFTER_SEC:.0f}s)")
                return None
            self._models[device] = m
            self._failed_at.pop(device, None)
            self.last_error = None
            after = _cuda_allocated_mb(device)
            if before is not None and after is not None:
                self.vram_mb[device] = round(after - before, 1)
            logger.info(f"[OSNET] loaded {MODEL_ID} device={device} vram_mb={self.vram_mb.get(device)}")
            return m

    def warm(self, device: str) -> bool:
        return self.enabled and self._model(device) is not None

    def embed_fn(self, device: str) -> Optional[Callable[[List[Image.Image]], np.ndarray]]:
        """The ``osnet_fn`` for attach_crop_embeddings, or None (disabled / not loadable now)."""
        if not self.enabled:
            return None
        model = self._model(device)
        if model is None:
            return None

        def _embed(crops: List[Image.Image]) -> np.ndarray:
            import torch

            try:
                batch = preprocess(crops).to(device)
                with torch.inference_mode():
                    feats = model(batch).detach().float().cpu().numpy()
            except Exception as exc:
                self.failures += 1
                self.last_error = f"embed: {exc}"[:300]
                raise
            self.vectors += len(crops)
            return feats

        return _embed

    def status(self) -> Dict[str, Any]:
        if not self.enabled:
            state = "disabled"
        elif self._models:
            state = "loaded"
        elif self._failed_at:
            state = "failed"
        else:
            state = "not_loaded"
        return {
            "enabled": self.enabled,
            "state": state,
            "model_id": MODEL_ID,
            "weights_path": str(self.weights_path),
            "devices": sorted(self._models),
            "vram_mb": dict(self.vram_mb),
            "vectors": self.vectors,
            "failures": self.failures,
            "last_error": self.last_error,
        }


def _cuda_allocated_mb(device: str) -> Optional[float]:
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        return torch.cuda.memory_allocated(torch.device(device)) / (1024 * 1024)
    except Exception:
        return None
