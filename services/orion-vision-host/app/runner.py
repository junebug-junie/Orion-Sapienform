from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from PIL import Image

import torch

from orion.vision.caption_echo import strip_echoed_prompt_prefix

from .artifacts import merge_result_inputs
from .caption_sanitize import CAPTION_PROMPT, sanitize_answer, sanitize_caption
from .detections import cap_by_score, nms
from .model_manager import ModelManager
from .models import VisionResult, VisionTask
from .profiles import PipelineDef, ProfileDef, VisionProfiles
from .settings import Settings
from .when_guard import safe_when

settings = Settings()

_safe_when = safe_when


def _resolve_latest_frame_path() -> Path:
    """The "on-demand capture" primitive P1's design doc calls for --
    docs/superpowers/specs/2026-08-12-perception-frontier-design.md names
    this explicitly as separate, larger work from the passive window/
    council pipeline ("bypassing window/council... a direct vision-host
    RPC"). Doesn't trigger a NEW capture (vision-edge already captures
    continuously regardless of any downstream consumer, confirmed live at
    ~5s cadence) -- resolves to whatever it captured most recently, which
    for a slow-moving room is the practical equivalent of "look now" without
    inventing a second capture path. Raises (not a fabricated/empty result)
    if the frames directory is empty or unreadable -- an honest "nothing to
    look at" is the caller's problem to handle, not this function's to hide.
    """
    frames_dir = Path(settings.VISION_FRAMES_DIR)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")
    candidates = sorted(frames_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no frames found in: {frames_dir}")
    return candidates[0]


_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def _load_image_from_percept_store(sha256: str) -> Image.Image:
    """Fetch a content-addressed frame from orion-percept-store.

    This is the leg that lets a second machine feed this pipeline at all --
    before it, capture had to share a filesystem with this host.

    The digest is regex-validated before it becomes part of a URL: it is the
    only caller-supplied component of that URL, and keeping it to 64 hex chars
    is what stops it being a path or an authority. Same reasoning as the LLM
    gateway's own attachment resolver.

    urllib rather than a new dependency (AGENTS.md section 10); this runs on a
    worker thread already.
    """
    import io
    import urllib.error
    import urllib.request

    if not _SHA256_RE.match(sha256):
        raise ValueError(f"percept_sha256 must be 64 lowercase hex chars, got {sha256[:16]!r}")
    base = str(getattr(settings, "VISION_PERCEPT_STORE_URL", "") or "").strip().rstrip("/")
    if not base:
        raise ValueError(
            "percept_sha256 supplied but VISION_PERCEPT_STORE_URL is unset; "
            "this host cannot resolve content-addressed frames"
        )
    url = f"{base}/{sha256}"
    token = str(getattr(settings, "VISION_PERCEPT_STORE_TOKEN", "") or "")
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(
            req, timeout=float(getattr(settings, "VISION_PERCEPT_TIMEOUT_SEC", 10.0))
        ) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise FileNotFoundError(f"percept {sha256[:12]} not retrievable from {base}: {exc}") from exc
    if not data:
        raise FileNotFoundError(f"percept {sha256[:12]} came back empty")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        # percept-store is one unpartitioned sha256 keyspace, and
        # PERCEPT_ALLOWED_MIMES now also accepts audio/wav and video/mp4
        # (2026-08-22, AffectGPT) -- this path has never routed a non-image
        # sha to this function (vision-frame-router only ever forwards JPEG-
        # frame refs), but nothing enforced that structurally. A clear error
        # here beats an unhandled PIL.UnidentifiedImageError crash if that
        # ever stops being true (review finding, 2026-08-22).
        raise ValueError(
            f"percept {sha256[:12]} did not decode as an image "
            f"(percept-store now also stores non-image content -- audio/video "
            f"clips): {exc}"
        ) from exc


def _load_image_from_request(request: Dict[str, Any]) -> Image.Image:
    """
    We do NOT ship frames over Redis. We take a pointer.
    Required (one of):
      request.image_path (preferred)
      request.frame_path (alias)
      request.use_latest_frame: true -- resolves to the most recently
        captured frame instead of a caller-supplied path (see
        _resolve_latest_frame_path). Opt-in, not a silent fallback when
        image_path is merely absent -- every existing caller that relies on
        the prior "image_path is required" error for a genuinely missing
        pointer keeps that exact behavior unless it explicitly asks for the
        latest-frame resolution instead.
    """
    path = request.get("image_path") or request.get("frame_path")
    if not path:
        sha = str(request.get("percept_sha256") or "").strip()
        if sha:
            # A frame captured on a node that shares no filesystem with us.
            # Fetched, never spooled: the bytes go straight into PIL and the
            # percept store expires its own copy on its own schedule.
            return _load_image_from_percept_store(sha)
        if request.get("use_latest_frame"):
            p = _resolve_latest_frame_path()
            img = Image.open(p).convert("RGB")
            return img
        raise ValueError(
            "request needs image_path or percept_sha256 "
            "(do not send raw frames over bus)"
        )
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
      - kind=caption_frame via VLM
      - kind=vlm (VQA -- a caller-supplied question, not a fixed caption
        prompt, against the same VLM family; see _run_vlm_vqa)
    """

    DEFAULT_EMBED_MODEL = "google/siglip2-so400m-patch14-384"
    DEFAULT_EMBED_FALLBACK = "google/siglip-so400m-patch14-384"
    DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"
    # Default VLM handled by env/settings

    def __init__(self, profiles: VisionProfiles, enabled_names: List[str], cache_dir: str):
        self.profiles = profiles
        self.enabled = set(enabled_names)

        self.cache_root = Path(cache_dir)
        self.artifacts_dir = self.cache_root / "artifacts" / "vision-host"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.models = ModelManager()
        self.warm_errors: Dict[str, str] = {}

    def _is_enabled(self, name: str) -> bool:
        return name in self.enabled

    def _resolve_dtype(self, p: ProfileDef) -> str:
        profile_dtype = (p.dtype or "").strip().lower()
        if profile_dtype and profile_dtype != "auto":
            return profile_dtype
        return (settings.VISION_DTYPE or "auto").strip().lower()

    def warm_profiles(self) -> List[str]:
        """
        Load weights for profiles with warm_on_start=true that are enabled and implemented.
        Failures are recorded in ``warm_errors`` (profile name -> message) without stopping other profiles.
        """
        self.warm_errors.clear()
        warmed: List[str] = []
        devices = settings.devices
        device = devices[0] if devices else settings.VISION_DEFAULT_DEVICE

        if not device.startswith("cuda"):
            logger.warning("[WARM] skipping CUDA model warmup (primary device is not CUDA)")
            return warmed

        for name, p in self.profiles.profiles.items():
            if not self._is_enabled(name) or not p.enabled or not p.warm_on_start:
                continue
            if p.kind not in ("embedding", "detect_open_vocab", "caption_frame", "vlm"):
                continue
            try:
                self._warm_profile_backend(p, device)
                warmed.append(name)
            except Exception as e:
                msg = str(e)
                logger.warning(f"[WARM] profile={name} kind={p.kind} err={msg}")
                self.warm_errors[name] = msg
        return warmed

    def _warm_profile_backend(self, p: ProfileDef, device: str) -> None:
        dtype = self._resolve_dtype(p)
        if p.kind == "embedding":
            model_id = (
                p.model_id if p.model_id and not p.model_id.startswith("REPLACE_ME") else self.DEFAULT_EMBED_MODEL
            )
            self.models.load_siglip_image_embedder(
                profile_name=p.name,
                device=device,
                dtype=dtype,
                model_id=model_id,
                fallback_model_id=self.DEFAULT_EMBED_FALLBACK,
            )
        elif p.kind == "detect_open_vocab":
            model_id = (
                p.model_id if p.model_id and not p.model_id.startswith("REPLACE_ME") else self.DEFAULT_GDINO_MODEL
            )
            self.models.load_grounding_dino(
                profile_name=p.name,
                device=device,
                dtype=dtype,
                model_id=model_id,
            )
        elif p.kind == "caption_frame":
            model_id = settings.VISION_VLM_MODEL_ID
            if p.model_id and not p.model_id.startswith("REPLACE_ME"):
                model_id = p.model_id
            self.models.load_vlm_captioner(
                profile_name=p.name,
                device=device,
                dtype=dtype,
                model_id=model_id,
            )
        elif p.kind == "vlm":
            # Same loader as caption_frame (both are "a VLM, generic
            # transformers Vision2Seq/BLIP prompting" -- see
            # _run_vlm_vqa/_run_caption_frame) -- keyed by this profile's own
            # `p.name` ("vlm_vqa"), same per-profile caching convention every
            # other kind above uses, so it loads as its own resident model
            # rather than silently sharing vlm_caption's. `vlm_vqa` ships
            # with `warm_on_start: false`, so this branch is currently dead
            # code -- reachable only if BOTH gates in `warm_profiles()`'s own
            # loop agree: `p.warm_on_start` (this one) AND `p.kind` being in
            # that loop's own separate kind-allowlist tuple (review finding,
            # 2026-08-21: "vlm" was missing from that tuple too, so flipping
            # warm_on_start alone would silently still not warm this profile
            # -- fixed there in the same patch as this branch, not a second
            # thing left for later). Real requests already lazy-load via
            # `_run_vlm_vqa`'s own `load_vlm_captioner` call regardless of
            # whether this warm path ever runs.
            model_id = settings.VISION_VLM_MODEL_ID
            if p.model_id and not p.model_id.startswith("REPLACE_ME"):
                model_id = p.model_id
            self.models.load_vlm_captioner(
                profile_name=p.name,
                device=device,
                dtype=dtype,
                model_id=model_id,
            )

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
                        inputs=merge_result_inputs(task.request, task.meta),
                        meta={"error_code": "pipeline_disabled"},
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
                        inputs=merge_result_inputs(task.request, task.meta),
                        meta={"error_code": "profile_disabled"},
                    )
                artifacts = self._run_profile(self.profiles.get_profile(target), task.request, device, warnings)

        except KeyError:
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=f"unknown task/profile: {target}",
                inputs=merge_result_inputs(task.request, task.meta),
                meta={"error_code": "unknown_task"},
            )
        except FileNotFoundError as e:
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=str(e),
                warnings=warnings,
                inputs=merge_result_inputs(task.request, task.meta),
                meta={"error_code": "image_not_found"},
            )
        except ValueError as e:
            msg = str(e)
            code = "missing_image_path" if "image_path" in msg else "request_validation"
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=msg,
                warnings=warnings,
                inputs=merge_result_inputs(task.request, task.meta),
                meta={"error_code": code},
            )
        except Exception as e:
            msg = str(e)
            lower = msg.lower()
            code = "cuda_oom" if "out of memory" in lower else "inference_error"
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=device,
                error=msg,
                warnings=warnings,
                inputs=merge_result_inputs(task.request, task.meta),
                meta={"error_code": code},
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
            inputs=merge_result_inputs(task.request, task.meta),
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
        merged_artifacts = {}
        fingerprints: Dict[str, str] = {}

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

            # Merge fields for the consolidated artifact
            if isinstance(artifacts, dict):
                merged_artifacts.update(artifacts)
                mid = artifacts.get("model_id")
                if mid:
                    fingerprints[step.use] = str(mid)

        if fingerprints:
            merged_artifacts["_fingerprints"] = fingerprints

        # For retina_fast or other pipelines, we return the merged result as the top-level artifact content
        # preserving key metadata
        return merged_artifacts

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

        if p.kind == "caption_frame":
            return self._run_caption_frame(p, request, device, warnings)

        if p.kind == "vlm":
            return self._run_vlm_vqa(p, request, device, warnings)

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
        dtype = self._resolve_dtype(p)

        model, processor = self.models.load_siglip_image_embedder(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
            fallback_model_id=self.DEFAULT_EMBED_FALLBACK,
        )

        inputs = processor(images=img, return_tensors="pt")
        if device.startswith("cuda"):
            model_dtype = next(model.parameters()).dtype
            inputs = {
                k: v.to(device=device, dtype=model_dtype if torch.is_floating_point(v) else v.dtype)
                for k, v in inputs.items()
            }

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
            "embedding": {
                "ref": ref,
                "path": str(npy_path),
                "dim": int(vec.shape[0]),
                # 2026-08-19 (P2 wire-contract patch, orion/schemas/vision.py
                # VisionEmbedding.vector): inlined so a bus consumer can score
                # this vector without a filesystem seam into this service's
                # own model-cache volume. `vec` is already L2-normalized above
                # when normalize=true (the profile default) -- consumers doing
                # cosine similarity get a real unit vector, not a raw one.
                "vector": [float(x) for x in vec.tolist()],
            }
        }

    # -----------------------------------
    # Real open-vocab detect (GroundingDINO)
    # -----------------------------------
    def _run_detect_grounding_dino(self, p: ProfileDef, request: Dict[str, Any], device: str) -> Dict[str, Any]:
        img = _load_image_from_request(request)

        model_id = p.model_id if p.model_id and not p.model_id.startswith("REPLACE_ME") else self.DEFAULT_GDINO_MODEL
        dtype = self._resolve_dtype(p)

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

        # `score_threshold` is the historical spelling in vision_profiles.yaml and
        # was silently ignored; honour it as an alias so the documented knob works.
        box_th = float(
            p.params.get("box_threshold", p.params.get("score_threshold", 0.25))
        )
        text_th = float(p.params.get("text_threshold", 0.25))
        max_det = int(p.params.get("max_detections", 30))
        nms_iou = float(p.params.get("nms_iou", 0.6))

        model, processor = self.models.load_grounding_dino(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
        )

        inputs = processor(images=img, text=text, return_tensors="pt")
        if device.startswith("cuda"):
            model_dtype = next(model.parameters()).dtype
            inputs = {
                k: v.to(device=device, dtype=model_dtype if torch.is_floating_point(v) else v.dtype)
                for k, v in inputs.items()
            }

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
            lab = labels[i] if i < len(labels) else "object"
            if isinstance(lab, (int, float)):
                lab = str(lab)
            obj = {
                "label": str(lab),
                "score": float(scores[i]),
                "box_xyxy": [float(x) for x in boxes[i].tolist()],
            }
            objects.append(obj)

        # Suppress duplicate boxes before capping, so the cap spends its budget
        # on distinct objects rather than on six views of the same desk.
        raw_count = len(objects)
        objects = nms(objects, nms_iou)
        objects = cap_by_score(objects, max_det)
        if raw_count != len(objects):
            logger.debug(
                f"[DETECT] profile={p.name} raw={raw_count} after_nms_and_cap={len(objects)} "
                f"nms_iou={nms_iou} max_det={max_det}"
            )

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
            "nms_iou": nms_iou,
            "max_detections": max_det,
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
            "artifact_path": str(json_path),
        }

    # ------------------------
    # Real Captioning (VLM)
    # ------------------------
    def _run_caption_frame(
        self,
        p: ProfileDef,
        request: Dict[str, Any],
        device: str,
        warnings: List[str],
    ) -> Dict[str, Any]:
        img = _load_image_from_request(request)

        # Use env defaults if not specified in profile
        model_id = settings.VISION_VLM_MODEL_ID
        if p.model_id and not p.model_id.startswith("REPLACE_ME"):
            model_id = p.model_id

        dtype = self._resolve_dtype(p)

        model, processor = self.models.load_vlm_captioner(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
        )

        # Simple prompt generation (task agnostic usually)
        # Some models require specific prompting formats.
        # For simplicity, we assume standard image-to-text here.

        # prompt = request.get("prompt", "Describe this image.") # Some VLMs need text
        # But many like BLIP/IDEFICS can just take image or image+prompt.
        # We'll use a standard prompt if supported by the processor.

        # Note: API differences between VLMs are significant.
        # Using a generic approach for "IDEFICS2" or similar.

        text_prompt = CAPTION_PROMPT
        inputs = processor(images=img, text=text_prompt, return_tensors="pt")

        if device.startswith("cuda"):
            model_dtype = next(model.parameters()).dtype
            inputs = {
                k: v.to(device=device, dtype=model_dtype if torch.is_floating_point(v) else v.dtype)
                for k, v in inputs.items()
            }

        max_tokens = settings.VISION_VLM_MAX_TOKENS
        temperature = settings.VISION_VLM_TEMPERATURE

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Case-insensitive prefix strip -- see strip_echoed_prompt_prefix's
        # own docstring for why the plain str.replace() this used to be
        # silently failed to strip a lowercased echo of a mixed-case prompt.
        cleaned = strip_echoed_prompt_prefix(generated_text, prompt=text_prompt)
        caption_text, ok, reason = sanitize_caption(cleaned)
        if not ok:
            warnings.append(f"caption_rejected:{reason}")
            caption_text = ""

        return {
            "configured": True,
            "implemented": True,
            "kind": "caption_frame",
            "model_id": model_id,
            "device": device,
            "caption": {
                "text": caption_text,
                "confidence": 1.0 # Placeholder
            }
        }

    # ------------------------
    # Real VQA (VLM, caller-supplied question)
    # ------------------------
    def _run_vlm_vqa(
        self,
        p: ProfileDef,
        request: Dict[str, Any],
        device: str,
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Movement/`look()`'s "active vision" primitive from
        docs/superpowers/specs/2026-08-12-perception-frontier-design.md --
        ``request.question`` is the caller's real question ("is that door
        open?"), not the fixed caption prompt. Same VLM family and
        prompt->generate->decode mechanics as ``_run_caption_frame`` (BLIP's
        "conditional generation" is exactly text-conditioned image
        description, which is what VQA is), deliberately not a separate
        code path -- the only real difference is which text goes in.

        Loads its own model under this profile's own name (``vlm_vqa``, kept
        separate from ``vlm_caption``'s), matching every other kind's
        per-profile caching convention above rather than special-casing a
        cross-profile share -- confirmed live before shipping that there is
        real VRAM headroom for a second small VLM instance (~4.2GB free of
        7.68GB on the P4 serving this host at the time of writing), not
        assumed.
        """
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("request.question is required for VQA (task_type=vqa)")

        img = _load_image_from_request(request)

        model_id = settings.VISION_VLM_MODEL_ID
        if p.model_id and not p.model_id.startswith("REPLACE_ME"):
            model_id = p.model_id

        dtype = self._resolve_dtype(p)

        model, processor = self.models.load_vlm_captioner(
            profile_name=p.name,
            device=device,
            dtype=dtype,
            model_id=model_id,
        )

        inputs = processor(images=img, text=question, return_tensors="pt")

        if device.startswith("cuda"):
            model_dtype = next(model.parameters()).dtype
            inputs = {
                k: v.to(device=device, dtype=model_dtype if torch.is_floating_point(v) else v.dtype)
                for k, v in inputs.items()
            }

        # This profile's own declared params, not the caption profile's
        # global settings.VISION_VLM_MAX_TOKENS/TEMPERATURE -- vlm_vqa's
        # config already declares its own max_tokens/temperature (see
        # config/vision_profiles.yaml) and answers have different length
        # needs than captions; falls back to the caption settings only if a
        # profile is somehow missing its own params.
        max_tokens = int(p.params.get("max_tokens", settings.VISION_VLM_MAX_TOKENS))
        temperature = float(p.params.get("temperature", settings.VISION_VLM_TEMPERATURE))

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        cleaned = strip_echoed_prompt_prefix(generated_text, prompt=question)
        answer_text, ok, reason = sanitize_answer(cleaned, question)
        if not ok:
            warnings.append(f"answer_rejected:{reason}")
            answer_text = ""

        return {
            "configured": True,
            "implemented": True,
            "kind": "vlm",
            "model_id": model_id,
            "device": device,
            "vqa": {
                "question": question,
                "answer": answer_text,
                "confidence": 1.0,  # Placeholder -- same convention _run_caption_frame uses.
            },
        }
