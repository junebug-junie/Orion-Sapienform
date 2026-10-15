"""
One GroundingDINO forward pass over one or more images -> per-image objects.

Shared by runner.py (live detect) and evals/walkway_tiled_detection_eval.py so
the eval measures the exact inference path the host runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch


def detect_images(
    model: Any,
    processor: Any,
    images: List[Any],
    text: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
    cast_inputs: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[List[Dict[str, Any]]]:
    """Run the model once on ``images`` (a batch) and return, per image,
    ``[{label, score, box_xyxy}]`` in that image's own pixel space. Raw:
    no NMS, no cap."""
    if not images:
        return []
    inputs = processor(images=list(images), text=[text] * len(images), return_tensors="pt")
    if cast_inputs is not None:
        inputs = cast_inputs(inputs)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[im.height, im.width] for im in images], device=device)
    try:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    if not isinstance(results, list):
        results = [results or {}]

    per_image: List[List[Dict[str, Any]]] = []
    for r in results:
        boxes = r.get("boxes")
        scores = r.get("scores")
        labels = r.get("text_labels") or r.get("labels") or []
        if boxes is None or scores is None:
            per_image.append([])
            continue
        boxes = boxes.detach().float().cpu().numpy()
        scores = scores.detach().float().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        objs = []
        for i in range(min(len(scores), len(boxes))):
            lab = labels[i] if i < len(labels) else "object"
            objs.append({
                "label": str(lab),
                "score": float(scores[i]),
                "box_xyxy": [float(x) for x in boxes[i].tolist()],
            })
        per_image.append(objs)
    return per_image
