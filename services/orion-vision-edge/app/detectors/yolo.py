from typing import List, Tuple, Optional, Set
import numpy as np
import logging
import torch

from ultralytics import YOLO

logger = logging.getLogger("orion-vision-edge.yolo")

class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        classes: Optional[Set[str]] = None,
        conf: float = 0.25,
        device: Optional[str] = None,
    ):
        # device: "cuda:0" for GPU, "cpu" otherwise
        self.model = YOLO(model_path)
        self.classes = set(c.lower() for c in classes) if classes else None
        self.conf = conf
        self.device = device

        logger.info(
            f"[YOLO] init model={model_path} device={self.device} "
            f"cuda_available={torch.cuda.is_available()} "
            f"cuda_count={torch.cuda.device_count()}"
        )

        # Warmup (helps allocate on the right device)
        self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            imgsz=640,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
