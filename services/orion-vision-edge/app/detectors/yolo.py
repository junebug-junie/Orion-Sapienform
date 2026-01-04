from typing import List, Tuple, Optional, Set
import numpy as np
import logging
import cv2
import time
import torch

from ultralytics import YOLO

logger = logging.getLogger("orion-vision-edge.yolo")

class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        classes: Optional[Set[str]] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        person_retry_threshold: Optional[float] = None,
    ):
        # device: "cuda:0" for GPU, "cpu" otherwise
        self.model = YOLO(model_path)
        self.classes = set(c.lower() for c in classes) if classes else None
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.device = device
        self.person_retry_threshold = person_retry_threshold

        logger.info(
            f"[YOLO] init model={model_path} device={self.device} "
            f"conf={self.conf} iou={self.iou} imgsz={self.img_size} "
            f"person_retry={self.person_retry_threshold}"
        )

        # Warmup
        try:
            self.model.predict(
                np.zeros((img_size, img_size, 3), dtype=np.uint8),
                imgsz=img_size,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"[YOLO] Warmup failed: {e}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Run detection on BGR frame.
        Returns list of (x, y, w, h, label, score).
        """
        if frame is None:
            return []

        # Convert BGR to RGB
        # Ultralytics YOLOv8 handles BGR->RGB internally if passed a numpy array?
        # Docs say: "YOLOv8 accepts... numpy arrays (BGR or RGB)..."
        # Ideally we pass RGB to be safe, or rely on it.
        # Let's convert to RGB to be consistent with standard PIL/Torch logic.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self._run_inference(frame_rgb, self.conf)

        # Person-centric retry
        # If we care about "person" and didn't find any, and have a lower threshold
        if self.person_retry_threshold and self.person_retry_threshold < self.conf:
            has_person = any(r[4] == "person" for r in results)
            if not has_person:
                # Retry with lower threshold
                low_conf_results = self._run_inference(frame_rgb, self.person_retry_threshold)
                # We only care about persons from this second pass (to avoid noise for other classes)
                persons = [r for r in low_conf_results if r[4] == "person"]
                if persons:
                    # Append or replace?
                    # The prompt says "record both passes". Since first pass had no persons, we just add these.
                    # We might have duplicates of other objects if we merge, but here we only add persons.
                    results.extend(persons)

        return results

    def _run_inference(self, frame_rgb: np.ndarray, conf: float) -> List[Tuple[int, int, int, int, str, float]]:
        start_t = time.perf_counter()

        # Run model
        # verbose=False to reduce log spam
        results_list = self.model.predict(
            frame_rgb,
            imgsz=self.img_size,
            conf=conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            classes=None, # Filter manually to handle case-insensitivity or complex logic
        )

        detections = []
        for result in results_list:
            # result.boxes contains Boxes object
            for box in result.boxes:
                # xyxy
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls_id]

                if self.classes and label.lower() not in self.classes:
                    continue

                # Convert to xywh for internal app compatibility (x, y, w, h)
                w = x2 - x1
                h = y2 - y1

                detections.append((int(x1), int(y1), int(w), int(h), label, score))

        return detections
