from __future__ import annotations

from typing import List, Any
import inspect
import logging
import os

from pydantic import BaseModel, ValidationError

from .base import VerbContext
from .models import VerbEffectV1, VerbRequestV1, VerbResultV1
from .registry import VerbRegistry, registry


class VerbRuntime:
    def __init__(
        self,
        service_name: str | None = None,
        instance_id: str | None = None,
        bus: Any | None = None,
        logger: logging.Logger | None = None,
        *,
        verb_registry: VerbRegistry | None = None,
        allow_backdoor: bool | None = None,
    ) -> None:
        if allow_backdoor is None:
            allow_backdoor = os.getenv("ORION_VERB_BACKDOOR_ENABLED", "false").lower() == "true"
        if not allow_backdoor and service_name not in {"cortex-exec", "orion-cortex-exec"}:
            raise RuntimeError("VerbRuntime direct usage disabled outside cortex-exec.")

        self.registry = verb_registry or registry
        self.service_name = service_name
        self.instance_id = instance_id
        self.bus = bus
        self.logger = logger or logging.getLogger("orion.verbs.runtime")

    async def handle_request(
        self,
        request: VerbRequestV1,
        *,
        extra_meta: dict[str, Any] | None = None,
    ) -> VerbResultV1:
        verb_cls = self.registry.get(request.trigger)
        if verb_cls is None:
            return VerbResultV1(
                verb=request.trigger,
                ok=False,
                error=f"verb_not_registered:{request.trigger}",
                request_id=request.request_id,
            )

        if not hasattr(verb_cls, "input_model"):
            return VerbResultV1(
                verb=request.trigger,
                ok=False,
                error=f"verb_missing_input_model:{request.trigger}",
                request_id=request.request_id,
            )

        try:
            payload = verb_cls.input_model.model_validate(request.payload)
        except ValidationError as exc:
            return VerbResultV1(
                verb=request.trigger,
                ok=False,
                error=f"invalid_payload:{exc}",
                request_id=request.request_id,
            )

        verb = verb_cls()
        ctx_meta = dict(request.meta)
        if extra_meta:
            ctx_meta.update(extra_meta)
        ctx = VerbContext(request_id=request.request_id, caller=request.caller, meta=ctx_meta)
        result = verb.execute(ctx, payload)
        if inspect.isawaitable(result):
            result = await result
        output, effects = result

        output_payload = output.model_dump() if isinstance(output, BaseModel) else output
        effect_models: List[VerbEffectV1] = []
        for effect in effects or []:
            if isinstance(effect, VerbEffectV1):
                effect_models.append(effect)
            else:
                effect_models.append(VerbEffectV1.model_validate(effect))

        return VerbResultV1(
            verb=request.trigger,
            ok=True,
            output=output_payload,
            effects=effect_models,
            request_id=request.request_id,
        )
