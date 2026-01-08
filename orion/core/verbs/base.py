from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, List, Tuple, TypeVar

from pydantic import BaseModel

from .models import VerbEffectV1


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class VerbContext:
    request_id: str | None = None
    caller: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseVerb(ABC, Generic[InputT, OutputT]):
    """Base class for runtime verbs."""

    input_model: type[InputT]
    output_model: type[OutputT]

    @abstractmethod
    def execute(self, ctx: VerbContext, payload: InputT) -> Tuple[OutputT, List[VerbEffectV1]]:
        """Execute the verb with a validated payload."""
        raise NotImplementedError
