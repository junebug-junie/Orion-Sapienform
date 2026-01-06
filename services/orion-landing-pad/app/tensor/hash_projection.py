from __future__ import annotations

import base64
import hashlib
from typing import List

from orion.schemas.pad import PadEventV1, TensorBlobV1

from .interface import Tensorizer


class HashProjectionTensorizer(Tensorizer):
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, events: List[PadEventV1]) -> TensorBlobV1:
        seed = "|".join(sorted([f"{e.type}:{e.salience:.3f}:{e.subject or ''}" for e in events]))
        digest = hashlib.sha256(seed.encode("utf-8")).digest()

        # Repeat digest to fill dimension
        raw_vec = (digest * ((self.dim * 4 // len(digest)) + 1))[: self.dim * 4]
        vector_b64 = base64.b64encode(raw_vec).decode("utf-8")
        features = {"event_count": len(events), "seed": seed}
        return TensorBlobV1(dim=self.dim, vector_b64=vector_b64, features=features)
