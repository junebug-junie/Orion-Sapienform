from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from PIL import Image

import torch

from .model_manager import ModelManager
from .models import VisionResult, VisionTask
from .profiles import PipelineDef, ProfileDef, VisionProfiles


def _safe_when(expr: str, request: Dict[str, Any]) -> bool:
    if not expr:
        return True
    expr = expr.replace("true", "True").replace("false", "False")
    ns = SimpleNamespace(**request)
    try:
        return bool(eval(expr, {"__builtins__": {}}, {"request": ns}))
    except Exception as e:
        logger.warning(f"[PIPE] when eval failed expr={expr} err={e}")
        return False


def _load_image_from_request(request: Dict[str, Any]) -> Image.Image:
    """
    We do NOT ship frames over Redis. We take a pointer.
    Required:
      request.image_path (preferred)
    Optional aliases:
      request.frame_path
    """
    path = request.get("image_path") or request.get("frame_path")
    if not path:
        raise ValueError("request.image_path is required (do not send raw frames over bus)")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image_path not found: {path}")
    img = Image.open(p).convert("RGB")
    return img


class VisionRunner:
    """
    Executes profiles/pipelines.

    What I implemented (real inference):
      - kind=embedding via SigLIP2 (fallback SigLIP)
      - kind=detect_open_vocab via GroundingDINO
    """

    DEFAULT_EMBED_MODEL = "google/siglip2-so400m-patch14-384"
    DEFAULT_EMBED_FALLBACK = "google/siglip-so400m-patch14-384"
    DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"

    def __init__(self, profiles: VisionProfiles, enabled_names: List[str], cache_dir: str):
        self.profiles = profiles
        self.enabled = set(enabled_names)

        self.cache_root = Path(cache_dir)
        self.artifacts_dir = self.cache_root / "artifacts" / "vision-host"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.models = ModelManager()

    def _is_enabled(self, name: str) -> bool:
        return name in self.enabled

    def warm_profiles(self) -> List[str]:
        """
        Warm only profiles that have warm_on_start=true AND are enabled.
        We keep this conservative: warm does not force-load placeholders.
        """
        warmed = []
        for name, p in self.profiles.profiles.items():
            if not self._is_enabled(name) or not p.enabled or not p.warm_on_start:
                continue
            # Only warm the two implemented backends; others remain lazy.
            if p.kind in ("embedding", "detect_open_vocab"):
                warmed.append(name)
        return warmed

    def execute(self, task: VisionTask, device: str) -> VisionResult:
        t0 = time.time()
        warnings: List[str] = []

        target = self.profiles.resolve_target(task.task_type)

        try:
            if self.profiles.is_pipeline(target):
                if not self._is_enabled(target):
                    return VisionResult(
                        corr_id=task.corr_id,
                        ok=False,
                        task_type=task.task_type,
                        device=device,
                        error=f"pipeline disabled: {target}",
                    )
                artifacts = self._run_pipeline(self.profiles.get_pipeline(target), task.request, device, warnings)
            else:
                if not self._is_enabled(target):
                    return VisionResult(
                        corr_id=task.corr_id,
                        ok=False,
                        task_type=task.task_type,
                        device=device,
                        error=f"profile disabled: {target}",
                    )
                artifacts = self._run_profile(self.profiles.get_profile(target), task.request, device, warnings)

        except KeyError:
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=f"unknown task/profile: {target}",
            )
        except Exception as e:
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=str(e),
                warnings=warnings,
            )

        dt = time.time() - t0
        meta = dict(task.meta or {})
        meta.update({"latency_s": round(dt, 3)})

        return VisionResult(
            corr_id=task.corr_id,
            ok=True,
            task_type=task.task_type,
            device=device,
            artifacts=artifacts,
            warnings=warnings,
            meta=meta,
        )

    def _run_pipeline(
        self,
        pipe: PipelineDef,
        request: Dict[str, Any],
        device: str,
        warnings: List[str],
    ) -> Dict[str, Any]:
        if not pipe.enabled:
            raise RuntimeError(f"pipeline not enabled: {pipe.name}")

        out: Dict[str, Any] = {"pipeline": pipe.name, "steps": [], "artifacts": {}}

        for step in pipe.steps:
            if step.when and not _safe_when(step.when, request):
                continue

            if not self._is_enabled(step.use):
                warnings.append(f"step profile disabled: {step.use}")
                continue

            p = self.profiles.get_profile(step.use)
            if not p.enabled:
                warnings.append(f"step profile not enabled in config: {step.use}")
                continue

            artifacts = self._run_profile(p, request, device, warnings)
            out["steps"].append({"use": step.use, "kind": p.kind})
            out["artifacts"][step.use] = artifacts

        return out

    def _run_profile(
        self,
        p: ProfileDef,
        request: Dict[str, Any],
        device: str,
        warnings: List[str],
    ) -> Dict[str, Any]:
        if p.kind == "embedding":
            return self._run_embedding_siglip(p, request, device)

        if p.kind == "detect_open_vocab":
            return self._run_detect_grounding_dino(p, request, device)

        # Everything else remains contract-only for now (no fake inference).
        warnings.append(f"kind not implemented yet: {p.kind}")
        return {
            "configured": True,
            "implemented": False,
            "kind": p.kind,
            "backend": p.backend,
            "model_id": p.model_id,
            "device": device,
            "params": p.params,
        }

    # ------------------------
    # Real embedding (SigLIP2)
    # ------------------------
    def _run_embedding_siglip(self, p: ProfileDef, request: Dict[str, Any], device: str) -> Dict[str, Any]:
        img = _load_image_from_request(request)

        model_id = p.model_id if p.model_id and not p.model_id.startswith("REPLACE_ME") else self.DEFAULT_EMBED_MODEL
        dtype = p.dtype or "auto"

        model, processor = self.models.load_siglip_image_embedder(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
            fallback_model_id=self.DEFAULT_EMBED_FALLBACK,
        )

        inputs = processor(images=img, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs)
            else:
                out = model(**inputs)
                feats = getattr(out, "pooler_output", None)
                if feats is None:
                    # fallback: CLS token
                    feats = out.last_hidden_state[:, 0, :]

        vec = feats.detach().float().cpu().numpy()[0]

        if bool(p.params.get("normalize", True)):
            n = np.linalg.norm(vec) + 1e-12
            vec = vec / n

        # Store as .npy
        seed = f"{request.get('image_path') or request.get('frame_path')}|{model_id}"
        h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        out_path = self.artifacts_dir / "embeddings"
        out_path.mkdir(parents=True, exist_ok=True)

        npy_path = out_path / f"{p.name}_{h}.npy"
        np.save(str(npy_path), vec)

        ref = f"emb:{p.name}:{h}"
        return {
            "configured": True,
            "implemented": True,
            "kind": "embedding",
            "model_id": model_id,
            "device": device,
            "embedding_ref": ref,
            "path": str(npy_path),
            "dim": int(vec.shape[0]),
        }

    # -----------------------------------
    # Real open-vocab detect (GroundingDINO)
    # -----------------------------------
    def _run_detect_grounding_dino(self, p: ProfileDef, request: Dict[str, Any], device: str) -> Dict[str, Any]:
        img = _load_image_from_request(request)

        model_id = p.model_id if p.model_id and not p.model_id.startswith("REPLACE_ME") else self.DEFAULT_GDINO_MODEL
        dtype = p.dtype or "auto"

        # Prompts:
        prompts = request.get("prompts")
        if not prompts:
            prompts = p.params.get("default_prompts") or []
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(x).strip() for x in (prompts or []) if str(x).strip()]
        if not prompts:
            prompts = ["person", "face", "phone", "screen", "door", "package"]

        # GroundingDINO wants a caption-like string; dot-separated works well.
        text = " . ".join(prompts)
        if not text.endswith("."):
            text = text + " ."

        box_th = float(p.params.get("box_threshold", 0.25))
        text_th = float(p.params.get("text_threshold", 0.25))
        max_det = int(p.params.get("max_detections", 30))

        model, processor = self.models.load_grounding_dino(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
        )

        inputs = processor(images=img, text=text, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        # target size: (h, w)
        target_sizes = torch.tensor([[img.height, img.width]], device=device)

        # Robust call (transformers signature differences)
        try:
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.get("input_ids"),
                box_threshold=box_th,
                text_threshold=text_th,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                box_threshold=box_th,
                text_threshold=text_th,
                target_sizes=target_sizes,
            )

        r0 = results[0] if isinstance(results, list) and results else (results or {})

        boxes = r0.get("boxes")
        scores = r0.get("scores")
        labels = r0.get("text_labels") or r0.get("labels") or []

        if boxes is None or scores is None:
            return {
                "configured": True,
                "implemented": True,
                "kind": "detect_open_vocab",
                "model_id": model_id,
                "device": device,
                "objects": [],
                "note": "no boxes/scores returned",
            }

        boxes = boxes.detach().float().cpu().numpy()
        scores = scores.detach().float().cpu().numpy()

        # labels might be list[str] OR list[int]
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()

        objects = []
        for i in range(min(len(scores), len(boxes))):
            if len(objects) >= max_det:
                break
            lab = labels[i] if i < len(labels) else "object"
            if isinstance(lab, (int, float)):
                lab = str(lab)
            obj = {
                "label": str(lab),
                "score": float(scores[i]),
                "box_xyxy": [float(x) for x in boxes[i].tolist()],
            }
            objects.append(obj)

        # Store as JSON artifact
        seed = f"{request.get('image_path') or request.get('frame_path')}|{model_id}|{text}"
        h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

        out_path = self.artifacts_dir / "detections"
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"{p.name}_{h}.json"

        payload = {
            "profile": p.name,
            "model_id": model_id,
            "prompts": prompts,
            "box_threshold": box_th,
            "text_threshold": text_th,
            "objects": objects,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "configured": True,
            "implemented": True,
            "kind": "detect_open_vocab",
            "model_id": model_id,
            "device": device,
            "prompts": prompts,
            "objects": objects,
            "path": str(json_path),
        }
