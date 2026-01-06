from __future__ import annotations

from orion.schemas.pad import PadEventV1, TensorBlobV1


class Tensorizer:
    def encode(self, events: list[PadEventV1]) -> TensorBlobV1:  # pragma: no cover - interface
        raise NotImplementedError
