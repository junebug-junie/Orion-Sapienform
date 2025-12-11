
import cv2
from typing import List, Tuple
import os

class FaceDetector:
    def __init__(self, cascade_path: str, scale_factor: float=1.1, min_neighbors: int=5, min_size: int=30):
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Missing cascade file: {cascade_path}")
        self.face = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = (min_size, min_size)

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        boxes = [(int(x), int(y), int(w), int(h)) for (x,y,w,h) in faces]
        return boxes
