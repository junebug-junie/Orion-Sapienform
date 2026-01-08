from __future__ import annotations

from typing import List

from pydantic import BaseModel, ValidationError

from .base import VerbContext
from .models import VerbEffectV1, VerbRequestV1, VerbResultV1
from .registry import VerbRegistry, registry


class VerbRuntime:
    def __init__(self, verb_registry: VerbRegistry | None = None) -> None:
        self.registry = verb_registry or registry

    def handle_request(self, request: VerbRequestV1) -> VerbResultV1:
        verb_cls = self.registry.get(request.verb)
        if verb_cls is None:
            return VerbResultV1(
                verb=request.verb,
                ok=False,
                error=f"verb_not_registered:{request.verb}",
                request_id=request.request_id,
            )

        if not hasattr(verb_cls, "input_model"):
            return VerbResultV1(
                verb=request.verb,
                ok=False,
                error=f"verb_missing_input_model:{request.verb}",
                request_id=request.request_id,
            )

        try:
            payload = verb_cls.input_model.model_validate(request.payload)
        except ValidationError as exc:
            return VerbResultV1(
                verb=request.verb,
                ok=False,
                error=f"invalid_payload:{exc}",
                request_id=request.request_id,
            )

        verb = verb_cls()
        ctx = VerbContext(request_id=request.request_id, caller=request.caller, meta=dict(request.meta))
        output, effects = verb.execute(ctx, payload)

        output_payload = output.model_dump() if isinstance(output, BaseModel) else output
        effect_models: List[VerbEffectV1] = []
        for effect in effects or []:
            if isinstance(effect, VerbEffectV1):
                effect_models.append(effect)
            else:
                effect_models.append(VerbEffectV1.model_validate(effect))

        return VerbResultV1(
            verb=request.verb,
            ok=True,
            output=output_payload,
            effects=effect_models,
            request_id=request.request_id,
        )
