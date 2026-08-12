"""`SubstrateReadService` contract — a verb asking the substrate about the real
host it runs on.

WHY THIS EXISTS (2026-08-12)

Design doc: `docs/superpowers/specs/2026-08-12-substrate-action-perception-design.md`
(Option B(2)).

Orion's substrate actions could not perceive anything. `substrate.inspect` runs
~15,000 times a day as a single LLM call with `max_recursion_depth: 0`, no
tools, and the only real data it receives is `motivating_dimensions` -- the same
field-pressure numbers that caused the proposal to exist. It cannot learn
anything that was not already in its prompt.

Three ways to fix that were traced and costed. This is B(2), the one selected:
a bus request/reply service, following the `RecallService` template
(`orion/bus/channels.yaml`) exactly. The rejected alternatives, recorded so they
are not re-proposed:

  B(1)  add a skill step to the verb YAML -- DOES NOT WORK. `planner.py:194`
        stamps `verb_name` from the PLAN onto every step, so a step inside
        `substrate.inspect` always carries `verb_name == "substrate.inspect"`,
        and `executor.py:2639`'s local-adapter branch requires
        `startswith("skills.")`. It would never fire.
  B(3)  change that dispatch keying -- REJECTED. `call_step_services` is the
        single step loop for all 97 verbs; the largest blast radius in the repo
        for the smallest capability.
  C     have the dispatcher pre-fetch into the prompt -- specified but unbuilt.
        Its predecessor (the 2026-07-28 grounding fix, same seam) stopped real
        fabrication and moved zero outcome signals.

B(2) is additive by construction: a service name no existing verb declares, so
the other 96 verbs are untouched.

WHY A READ AND NOT JUST A PROMPT

The first consumer is not `substrate.inspect`. It is autonomous Docker
build-cache pruning -- the first action available in this arc with a real,
continuous, per-action, externally-verifiable outcome (disk % before and after).
That action must *read real host state and act on the answer*, which a
pre-fetched prompt payload cannot serve: "how much is reclaimable right now"
has to be asked at decision time.

SCOPE CORRECTION (2026-08-12, same day as the original)

The first version of this contract carried a a Docker build-cache payload --
build-cache totals, reclaimable bytes, stale-entry counts -- because the
motivating consumer is autonomous Docker build-cache pruning.

That was wrong, and the trace that should have preceded it came after:
`orion-biometrics`, the ONLY registered consumer of this channel, has no
`/var/run/docker.sock` mount. It could never have populated that field. A
contract promising a payload its responder cannot produce is worse than one
that omits it -- the omission is visible, the promise is not.

`orion-cortex-exec` DOES have the socket, which is precisely why
`skills.docker.ps_status.v1` and
`skills.runtime.docker_prune_stopped_containers.v1` already work there as
LOCAL verb adapters. And `CortexRouteTemplateV1.cortex_verb` is an
unconstrained `str`, so an execution-dispatch route can name a `skills.*` verb
directly, which makes `executor.py:2639`'s local-adapter branch fire.

**So the prune action needs no bus service at all.** It is a local skill in the
one service that already holds the capability. This contract is therefore
narrowed to what `orion-biometrics` genuinely owns and can answer without new
privileges: host filesystem usage, via `shutil.disk_usage`, which needs no
socket and which it already computes for `disk_pressure`/
`disk_capacity_pressure`.

Extending a hardware-telemetry service with root-equivalent Docker socket
access, in order to answer a question a service that already has it can answer
locally, would have been a real privilege expansion bought for nothing.

DESIGN NOTES

`read_kind` is a closed Literal, not a free string. One kind ships here.
Extending means adding a member and a typed payload -- deliberately more
friction than passing an arbitrary dict, because an untyped `payload: dict` is
how a contract becomes a place to smuggle whatever the caller happened to have
(`CLAUDE.md` §0A, no keyword cathedrals: a new name needs a schema, a producer,
a consumer, and a test in the same patch).

Unavailability is a first-class reply, never an exception and never a zero.
`available: false` + `reason` copies the shape
`services/orion-cortex-exec/app/verb_adapters.py::BiometricsSnapshotVerb`
already uses live (`{"available": False, "reason": str(exc)}`). A read that
fails must not be indistinguishable from a read that found nothing -- this repo
has shipped decayed-to-zero-looks-like-calm before, and the whole point of an
action with a measurable outcome is that the measurement is trustworthy.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# One kind today. See the module docstring for why this is closed rather than a
# free-form string.
SubstrateReadKind = Literal["host_storage"]


class SubstrateReadQueryV1(BaseModel):
    """Request published to ``orion:exec:request:SubstrateReadService``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate_read.query.v1"] = "substrate_read.query.v1"
    read_kind: SubstrateReadKind

    # Echoed back on the reply so a caller correlating several reads cannot
    # mis-attribute one to another. The bus reply channel already carries a
    # uuid; this is for the caller's own bookkeeping, not for routing.
    request_id: str | None = None

    # Which mount to report on. Defaults to the Docker data-root, which is the
    # one under pressure and the one the prune action acts on. Kept explicit
    # rather than implied so a reply is never ambiguous about what it measured
    # -- this host has eight independent filesystems (see
    # scripts/disk_threshold_watchdog.py), and usage on one says nothing about
    # the others.
    mount_path: str = "/mnt/docker"


class HostStorageReadV1(BaseModel):
    """Payload for ``read_kind="host_storage"``."""

    model_config = ConfigDict(extra="forbid")

    mount_path: str
    total_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)
    used_pct: float = Field(ge=0.0, le=100.0)


class SubstrateReadReplyV1(BaseModel):
    """Reply published to ``orion:exec:result:SubstrateReadService:<uuid>``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate_read.reply.v1"] = "substrate_read.reply.v1"
    read_kind: SubstrateReadKind
    request_id: str | None = None

    observed_at: datetime

    # False means the read could not be performed. `payload` is then None and
    # `reason` says why. Never a fabricated zero -- an unavailable read and an
    # empty one are different facts.
    available: bool
    reason: str | None = None

    payload: HostStorageReadV1 | None = None
