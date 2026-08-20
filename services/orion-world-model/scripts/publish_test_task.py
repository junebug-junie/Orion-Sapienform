"""Publish a real WorldModelTaskRequestPayload over the bus and print the
reply. Mirrors services/orion-vision-host/scripts/publish_test_task.py.

This is the real producer for `orion:exec:request:WorldModelService` in this
patch -- no other service issues world-model requests yet (see README.md
"No real downstream consumer yet").

Usage:
    python scripts/publish_test_task.py --url redis://<host>:6379/0 \
        --dim-biometrics 32 --dim-affect 16 --dim-execution-context 16 \
        --dim-memory-pointers 32 --dim-temporal 8 --dim-vision-embedding 512 \
        --window 4
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.world_model import (
    WorldModelFeatureGroupV1,
    WorldModelPredictionPayload,
    WorldModelTaskRequestPayload,
    WorldModelTrajectoryStepV1,
)


def _zero_group(dim: int) -> WorldModelFeatureGroupV1:
    return WorldModelFeatureGroupV1(dim=dim, vector=[0.0] * dim)


def build_trajectory(
    window: int,
    *,
    dim_biometrics: int,
    dim_affect: int,
    dim_execution_context: int,
    dim_memory_pointers: int,
    dim_temporal: int,
    dim_vision_embedding: int,
) -> list[WorldModelTrajectoryStepV1]:
    return [
        WorldModelTrajectoryStepV1(
            ts=float(i),
            biometrics=_zero_group(dim_biometrics),
            affect=_zero_group(dim_affect),
            execution_context=_zero_group(dim_execution_context),
            memory_pointers=_zero_group(dim_memory_pointers),
            temporal=_zero_group(dim_temporal),
            vision_embedding=_zero_group(dim_vision_embedding),
        )
        for i in range(window)
    ]


def build_request_envelope(
    trajectory: list[WorldModelTrajectoryStepV1], *, correlation_id: str, reply_channel: str
) -> BaseEnvelope:
    """Extracted so `tests/test_publish_test_task.py` can construct a real
    envelope without a live bus -- this exact construction call is what was
    broken (twice) before being caught by hand-testing during code review;
    a test now pins it against silent regression."""
    payload = WorldModelTaskRequestPayload(
        task_type="predict_next_state",
        trajectory=trajectory,
        meta={"source": "cli_test"},
    )
    return BaseEnvelope(
        # NOTE: do NOT pass schema_id= here. BaseEnvelope.schema_id
        # (bus_schemas.py) is a fixed `Literal["orion.envelope"]` field
        # (aliased to the wire key "schema") -- it is the envelope-format
        # marker, not the message-kind field. Passing a message-kind string
        # for it raises a pydantic ValidationError at construction time
        # (verified live while building this script; the analogous line in
        # vision-host's own publish_test_task.py has the identical bug,
        # out of scope to fix here). `kind=` below is the real message-kind
        # field.
        kind="world_model.task.request",
        # BaseEnvelope.source is typed ServiceRef, not a bare string --
        # a plain str here raises a pydantic ValidationError at construction
        # time (verified live; the analogous line in vision-host's own
        # publish_test_task.py has the same latent bug, out of scope to fix
        # here -- not replicated in this new script).
        source=ServiceRef(name="cli-tester", version="0.1.0"),
        correlation_id=correlation_id,
        reply_to=reply_channel,
        # BaseEnvelope.payload is typed Dict[str, Any] (bus_schemas.py) -- it
        # does NOT accept a pydantic model directly outside of
        # derive_child()'s explicit conversion. Passing the model itself here
        # raises a pydantic ValidationError at construction time (verified
        # live while building this script).
        payload=payload.model_dump(mode="json"),
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--window", type=int, default=4, help="Trajectory window length")
    parser.add_argument("--dim-biometrics", type=int, default=32)
    parser.add_argument("--dim-affect", type=int, default=16)
    parser.add_argument("--dim-execution-context", type=int, default=16)
    parser.add_argument("--dim-memory-pointers", type=int, default=32)
    parser.add_argument("--dim-temporal", type=int, default=8)
    parser.add_argument("--dim-vision-embedding", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.url)
    await bus.connect()

    request_channel = "orion:exec:request:WorldModelService"
    reply_channel = f"orion:worldmodel:reply:{uuid.uuid4()}"
    correlation_id = str(uuid.uuid4())

    trajectory = build_trajectory(
        args.window,
        dim_biometrics=args.dim_biometrics,
        dim_affect=args.dim_affect,
        dim_execution_context=args.dim_execution_context,
        dim_memory_pointers=args.dim_memory_pointers,
        dim_temporal=args.dim_temporal,
        dim_vision_embedding=args.dim_vision_embedding,
    )
    envelope = build_request_envelope(trajectory, correlation_id=correlation_id, reply_channel=reply_channel)

    print(f"Sending request to {request_channel} (window={args.window})...")
    print(f"Reply channel: {reply_channel}")

    try:
        async with bus.subscribe(reply_channel) as pubsub:
            await bus.publish(request_channel, envelope)
            print("Waiting for reply...")
            async for msg in bus.iter_messages(pubsub):
                data = msg.get("data")
                decoded = bus.codec.decode(data)
                if decoded.ok:
                    res_env = decoded.envelope
                    print("\n--- RECEIVED REPLY ---")
                    print(f"Kind: {res_env.kind}")
                    print(f"CorrID: {res_env.correlation_id}")
                    if isinstance(res_env.payload, dict):
                        try:
                            res_payload = WorldModelPredictionPayload(**res_env.payload)
                            print(
                                f"ok={res_payload.ok} device={res_payload.device} "
                                f"model_untrained={res_payload.model_untrained} "
                                f"state_dim={res_payload.state_dim} "
                                f"param_count={res_payload.model_param_count}"
                            )
                        except Exception as exc:
                            print(f"Payload (dict, validation error {exc}): {res_env.payload}")
                    else:
                        print(f"Payload: {res_env.payload}")
                    break
                else:
                    print(f"Decode error: {decoded.error}")
                    break
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
