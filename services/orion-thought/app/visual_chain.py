"""Reverie VISUAL chain — Patch 2 of docs/superpowers/specs/
2026-08-20-reverie-visual-chain-design.md.

A second, parallel reverie chain alongside the text chain (`chain.py`): on a
slow, capacity-gated cadence, generate an image about the last reverie's
description via `orion-diffusion-host`, run it back through
`orion-vision-host`'s existing `caption_frame` task to get a fresh
description, persist both, and carry that description forward as the next
run's `prior_description` — a real generate -> observe -> interpret loop
(design doc §1).

Patch 2 scope (design doc §8 non-goals, shipped 2026-08-25): the mechanical
loop and the `prior_description` continuity wiring only -- `build_visual_prompt`
was deliberately the thinnest honest placeholder (prior_description, or a
fixed seed prompt for the very first run), not a fabricated stand-in for the
context-seeding that patch did not own.

Patch 3 (shipped 2026-08-26): real context-seeding. `build_visual_prompt` now
also takes `context_text` -- the text reverie chain's own most recent, real
(non-hollow) narration (`store.load_latest_reverie_interpretation`), a
deliberately narrow first slice of the design doc's full "recent activity /
chat / dream" list (§1): already-summarized content that already reaches the
same Hub Reverie tab this feeds, so no new privacy surface (see that store
function's own docstring and reverie_routes.py's privacy note). Widening to
raw chat/dream sources is a separate, later change that must redo that
privacy check.

Patch 4 (this changeset, design doc §15): live-caught 2026-08-27 --
Juniper reported "still doing the same images of Roman aqueducts, no
change" a few hours after Patch 3 shipped. Real: `prior_description`
continuity had locked onto one visual attractor across 10+ runs / 100+
minutes, predating Patch 3 and unmoved by it -- `context_text` is real,
correctly varying content (confirmed live in Postgres), but a short
abstract clause ("Orion is currently thinking: the coalition is fixated on
...") has nowhere near the prompt weight of a long, concrete continuity
description, and abstract cognitive-state narration isn't strongly
visualizable content regardless of prompt order. `resolve_visual_chain_continuity`
below is the actual fix: a deterministic, testable reset -- after
`settings.visual_chain_continuity_max_runs` CONSECUTIVE runs carrying
`prior_description` forward, the next run forces continuity to drop for
that one prompt (re-seeding from `context_text`, or the fixed seed if
neither exists), then continuity resumes normally. Not a prompt-reweighting
guess; a mechanical guarantee the loop cannot run unbounded.

One run = one step (`step_index=0` always). The design doc's "chain" here is
the *sequence of runs over time* (each with its own `chain_id`, linked by
`prior_description`), not a multi-step loop within one run like the text
chain's ladder — there is no coalition/pressure signal to climb here.

Single-flight, no backlog (design doc §4): achieved structurally by
`run_visual_chain_worker`'s own sequential loop (run, then sleep, then run
again -- the same shape `chain.py`'s `run_reverie_chain_worker` and
`reverie.py`'s `run_reverie_worker` already use), not by a persisted
`running_since` marker. A trigger literally cannot exist while a run is in
flight, since there is no separate scheduler event outside this loop. On top
of that structural guarantee, `run_visual_chain_once` also holds a
process-local `asyncio.Lock` and no-ops (returns None) if already held --
defense-in-depth matching `orion-diffusion-host`'s own `_generation_lock`
pattern, and what actually makes the design doc's "busy -> no-op" acceptance
check (§9) true as a real, testable runtime property rather than an artifact
of caller discipline.

Every hop degrades honestly rather than fabricating:
  - diffusion-host call fails -> chain row with terminal_reason=
    "generation_failed", no artifact row (nothing was generated), no rows
    depend on it existing.
  - image sniffing fails (diffusion-host returned bytes that aren't a
    supported image format) -> same as generation_failed.
  - re-observation (percept upload or vision-host RPC) fails -> the image was
    still real and gets stored + its artifact row written with
    description=None (never a fabricated caption) -- and the *previous*
    run's prior_description is carried forward unchanged rather than
    propagating a null description and losing continuity for one bad step.

Default-off: `run_visual_chain_worker` is a no-op unless
ORION_VISUAL_CHAIN_ENABLED.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.reverie.visual_storage import StoredVisualArtifact, store_visual_artifact
from orion.schemas.reverie_visual import ReverieVisualArtifactV1, ReverieVisualChainV1
from orion.schemas.vision import VisionTaskRequestPayload, VisionTaskResultPayload

from .settings import settings
from .store import (
    load_latest_reverie_interpretation,
    load_latest_visual_chain_continuity_streak,
    load_latest_visual_chain_prior_description,
    persist_reverie_visual_artifact,
    persist_reverie_visual_chain,
)

logger = logging.getLogger("orion-thought.visual_chain")

# Last-resort fallback: no prior_description AND no real reverie-thought
# interpretation yet exists (a fresh install, before the text chain has
# written anything). Deliberately small and neutral -- this only needs to
# produce *something* real to generate and observe.
DEFAULT_SEED_PROMPT = "a calm orion, soft abstract light, dreaming"

# Defense-in-depth single-flight guard (module docstring). Not the mechanism
# that makes overlap structurally impossible -- the worker loop's own
# sequencing is -- but a real, testable property independent of caller
# discipline (mirrors orion-diffusion-host's _generation_lock).
_visual_chain_lock = asyncio.Lock()


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_visual_prompt(prior_description: str | None, context_text: str | None = None) -> str:
    """The diffusion prompt for one run. See module docstring for scope.

    `prior_description` (visual continuity -- the previous run's own
    re-observed caption) and `context_text` (Patch 3's context-seed --
    Orion's own most recent real reverie-thought interpretation) are
    independent inputs: continuity keeps the image chain visually coherent
    frame-to-frame; context-seeding keeps it grounded in what Orion is
    actually narrating instead of drifting into purely self-referential
    imagery with nothing anchoring it to a real cognitive state. Falls back
    to DEFAULT_SEED_PROMPT only when both are empty.
    """
    prior = (prior_description or "").strip()
    context = (context_text or "").strip()
    if prior and context:
        return (
            f"{prior}. Orion is currently thinking: {context}. "
            "Continue this train of imagination, soft dreamlike style."
        )
    if prior:
        return f"{prior}. Continue this train of imagination, soft dreamlike style."
    if context:
        return f"Orion is currently thinking: {context}. Soft abstract dreamlike style."
    return DEFAULT_SEED_PROMPT


def resolve_visual_chain_continuity(
    prior_description: str | None, streak: int, max_runs: int
) -> tuple[str | None, int, bool]:
    """Patch 4 (module docstring): decide whether THIS run may use
    `prior_description` continuity in its prompt, or must force a reset.

    Pure function -- `streak` is `store.load_latest_visual_chain_continuity_
    streak()`'s return (how many consecutive prior runs used continuity),
    `max_runs` is `settings.visual_chain_continuity_max_runs`.

    Returns `(effective_prior_description, next_streak, was_reset)`:
      - `effective_prior_description`: what to actually pass to
        `build_visual_prompt` for this run's prompt -- `prior_description`
        unchanged, or `None` on a forced reset.
      - `next_streak`: the value to record in this run's `chain_json.
        continuity_streak`, for the *next* run's decision.
      - `was_reset`: whether this run forced a reset (for logging/
        `chain_json.continuity_reset` -- inspectable evidence, not just an
        inferred side effect).

    No prior_description at all (cold start, or continuity already broken
    by a prior reset/failed re-observation) needs no reset and starts the
    streak fresh at 0 -- there is nothing to cap yet.
    """
    if not (prior_description or "").strip():
        return prior_description, 0, False
    if streak >= max_runs:
        return None, 0, True
    return prior_description, streak + 1, False


class DiffusionGenerationError(RuntimeError):
    """orion-diffusion-host's /generate call failed or returned non-2xx."""


def call_diffusion_generate(prompt: str, *, base_url: str, timeout_sec: float) -> bytes:
    """POST {base_url}/generate, return raw image bytes. Blocking (urllib) --
    callers must run this via asyncio.to_thread. stdlib only (AGENTS.md
    section 10 -- no dependency for one POST call), same choice
    foveal_probe.py and this service's own cortex_client make elsewhere.
    """
    url = str(base_url).rstrip("/") + "/generate"
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        # HTTPError is a URLError subclass -- this branch MUST come first, or
        # the except below (review finding) silently swallows every non-2xx
        # response (including orion-diffusion-host's documented 429
        # busy-reject) under the generic "failed" message, losing the status
        # code/reason a caller needs to tell "busy" from "broken".
        raise DiffusionGenerationError(
            f"diffusion-host /generate returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DiffusionGenerationError(f"diffusion-host /generate failed: {exc}") from exc
    if not data:
        raise DiffusionGenerationError("diffusion-host /generate returned empty body")
    return data


class PerceptUploadError(RuntimeError):
    """The generated image could not be handed to orion-percept-store."""


def upload_to_percept_store(
    data: bytes, *, base_url: str, token: str | None, timeout_sec: float
) -> str:
    """POST image bytes to orion-percept-store, return the content hash.

    Hash-verified on response, mirrors
    services/orion-vision-council/app/foveal_probe.py::upload_frame_bytes
    exactly (not imported cross-service -- CLAUDE.md section 5: each service
    owns its own small client).
    """
    local_sha = hashlib.sha256(data).hexdigest()
    url = str(base_url).rstrip("/")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        # A syntactically-valid but non-object JSON response (e.g. `[]`,
        # `null`) parses fine above but has no .get -- checked inside this
        # try (review finding) so it raises the documented PerceptUploadError
        # instead of an uncaught AttributeError.
        if not isinstance(body, dict):
            raise ValueError(f"percept store response was not a JSON object: {body!r}")
        sha256 = str(body.get("sha256") or "")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PerceptUploadError(f"percept upload to {url} failed: {exc}") from exc

    if sha256 != local_sha:
        raise PerceptUploadError(
            f"percept store returned {sha256[:12]!r} for content hashing to {local_sha[:12]!r}"
        )
    return sha256


def _extract_caption(payload: dict[str, Any]) -> str | None:
    """Pull the caption text out of a decoded VisionTaskResultPayload dict.
    None on anything short of a real, non-empty caption -- never a partial
    or placeholder string."""
    result = VisionTaskResultPayload.model_validate(payload)
    if not result.ok or result.artifact is None:
        return None
    caption = result.artifact.outputs.caption if result.artifact.outputs else None
    if caption is None:
        return None
    text = (caption.text or "").strip()
    return text or None


async def request_caption(
    bus: OrionBusAsync, percept_sha256: str, *, timeout_sec: float
) -> str | None:
    """RPC orion-vision-host's existing caption_frame task over its shared
    intake channel. Best-effort: any failure (timeout, decode error, ok=false,
    empty caption) returns None rather than raising or fabricating a caption
    -- see module docstring on why a failed re-observation must not break the
    run or corrupt prior_description.
    """
    corr_id = uuid4()
    reply_to = f"{settings.channel_vision_reply_prefix}:{corr_id}"
    task = VisionTaskRequestPayload(
        task_type="caption_frame", request={"percept_sha256": percept_sha256}
    )
    envelope = BaseEnvelope(
        kind="vision.task.request",
        source=_source(),
        correlation_id=corr_id,
        reply_to=reply_to,
        payload=task.model_dump(mode="json"),
    )
    try:
        msg = await bus.rpc_request(
            settings.channel_vision_host_request,
            envelope,
            reply_channel=reply_to,
            timeout_sec=timeout_sec,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.warning("visual chain caption reply decode failed: %s", decoded.error)
            return None
        return _extract_caption(decoded.envelope.payload)
    except Exception as exc:
        logger.warning("visual chain caption RPC failed sha=%s err=%s", percept_sha256[:12], exc)
        return None


async def run_visual_chain_once(
    bus: OrionBusAsync, *, now_fn: Any = _now
) -> ReverieVisualChainV1 | None:
    """One generate -> store -> observe -> persist run. Returns the chain
    readout, or None if a run was already in flight (single-flight no-op,
    not an error). Never raises.
    """
    if _visual_chain_lock.locked():
        logger.info("visual chain skipped: run already in flight")
        return None

    async def _generation_failed(chain_id: str, error: BaseException, prompt: str,
                                  prior_description: str | None, context_text: str | None,
                                  continuity_streak: int, continuity_reset: bool
                                  ) -> ReverieVisualChainV1:
        logger.warning("visual chain generation failed chain=%s err=%s", chain_id, error)
        chain = ReverieVisualChainV1(
            chain_id=chain_id,
            created_at=now_fn(),
            terminal_reason="generation_failed",
            prior_description=prior_description,
            chain_json={
                "prompt": prompt,
                "context_text": context_text,
                "continuity_streak": continuity_streak,
                "continuity_reset": continuity_reset,
                "error": str(error),
            },
        )
        with suppress(Exception):
            await asyncio.to_thread(persist_reverie_visual_chain, chain)
        return chain

    async with _visual_chain_lock:
        chain_id = str(uuid4())
        # Three independent reads (different tables/columns, no data
        # dependency) -- concurrent so the cost is max() of the round trips,
        # not sum() (review finding: this function already makes exactly
        # this argument a few lines below for store_visual_artifact/
        # upload_to_percept_store; the same reasoning applies here).
        prior_description, context_text, continuity_streak = await asyncio.gather(
            asyncio.to_thread(load_latest_visual_chain_prior_description),
            asyncio.to_thread(load_latest_reverie_interpretation),
            asyncio.to_thread(load_latest_visual_chain_continuity_streak),
        )
        # Patch 4 (module docstring): cap how many consecutive runs may
        # carry prior_description continuity before forcing one reset --
        # computed here (before generation) so a failed run still records
        # the correct streak for whichever run picks continuity back up.
        effective_prior, continuity_streak, continuity_reset = resolve_visual_chain_continuity(
            prior_description, continuity_streak, settings.visual_chain_continuity_max_runs
        )
        if continuity_reset:
            logger.info(
                "visual chain continuity reset chain=%s -- forcing a fresh seed after %s runs",
                chain_id,
                settings.visual_chain_continuity_max_runs,
            )
        prompt = build_visual_prompt(effective_prior, context_text)

        try:
            png_bytes = await asyncio.to_thread(
                call_diffusion_generate,
                prompt,
                base_url=settings.diffusion_host_base_url,
                timeout_sec=settings.visual_chain_diffusion_timeout_sec,
            )
        except Exception as exc:
            return await _generation_failed(
                chain_id, exc, prompt, prior_description, context_text,
                continuity_streak, continuity_reset,
            )

        # store_visual_artifact (disk write) and upload_to_percept_store (a
        # network round trip) both operate on the same immutable png_bytes
        # with no data dependency between them -- run concurrently so the
        # wall-clock cost is max(), not sum() (review finding). A store
        # failure is a real generation failure (nothing to persist); an
        # upload failure only degrades the re-observation step below (module
        # docstring) -- handled separately despite running concurrently.
        store_result, upload_result = await asyncio.gather(
            asyncio.to_thread(
                store_visual_artifact, png_bytes, base_dir=settings.visual_chain_storage_dir
            ),
            asyncio.to_thread(
                upload_to_percept_store,
                png_bytes,
                base_url=settings.visual_chain_percept_store_url,
                token=settings.visual_chain_percept_store_token,
                timeout_sec=settings.visual_chain_percept_upload_timeout_sec,
            ),
            return_exceptions=True,
        )

        if isinstance(store_result, BaseException):
            return await _generation_failed(
                chain_id, store_result, prompt, prior_description, context_text,
                continuity_streak, continuity_reset,
            )
        stored: StoredVisualArtifact = store_result

        description: str | None = None
        if isinstance(upload_result, BaseException):
            logger.warning(
                "visual chain re-observation failed chain=%s sha=%s err=%s -- "
                "image stored without a caption",
                chain_id,
                stored.sha256[:12],
                upload_result,
            )
        else:
            description = await request_caption(
                bus, upload_result, timeout_sec=settings.visual_chain_caption_timeout_sec
            )

        # Only advance continuity on a real, non-empty description -- a failed
        # re-observation forwards the *previous* prior_description unchanged
        # rather than propagating None and losing continuity for one step.
        next_prior_description = description or prior_description

        chain = ReverieVisualChainV1(
            chain_id=chain_id,
            created_at=now_fn(),
            terminal_reason="max_steps",
            prior_description=next_prior_description,
            chain_json={
                "prompt": prompt,
                "context_text": context_text,
                "continuity_streak": continuity_streak,
                "continuity_reset": continuity_reset,
                "artifact_sha256": stored.sha256,
                "description": description,
            },
        )
        # Chain row before artifact row: reverie_visual_artifact.chain_id is a
        # real FK (manual_migration_reverie_visual_chain.sql). The artifact
        # insert is skipped (not just attempted-and-swallowed) when the chain
        # row itself failed to persist -- review finding: persisting the
        # artifact unconditionally meant a transient chain-row failure still
        # attempted the artifact insert, which would then also fail its own
        # FK check, burying the real cause behind a second, confusing warning.
        chain_persisted = await asyncio.to_thread(persist_reverie_visual_chain, chain)
        if not chain_persisted:
            logger.warning(
                "visual chain artifact skipped chain=%s: chain row failed to persist", chain_id
            )
            return chain

        artifact = ReverieVisualArtifactV1(
            sha256=stored.sha256,
            chain_id=chain_id,
            step_index=0,
            mime=stored.mime,
            bytes=stored.bytes,
            width=stored.width,
            height=stored.height,
            path=stored.path,
            description=description,
        )
        await asyncio.to_thread(persist_reverie_visual_artifact, artifact)

        logger.info(
            "visual chain complete chain=%s sha=%s described=%s",
            chain_id,
            stored.sha256[:12],
            bool(description),
        )
        return chain


async def run_visual_chain_worker(stop_event: asyncio.Event | None = None) -> None:
    """Self-driven visual-chain loop. Default-off; no-op unless
    ORION_VISUAL_CHAIN_ENABLED. Sequential run-then-sleep shape (module
    docstring) is what gives the single-flight/no-backlog guarantee its
    structural teeth.
    """
    if not settings.visual_chain_enabled:
        logger.info("visual chain disabled; worker not started")
        return
    if not settings.orion_bus_enabled:
        logger.info("bus disabled; visual chain worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()

    logger.info(
        "visual chain worker started interval=%ss diffusion_host=%s",
        settings.visual_chain_interval_sec,
        settings.diffusion_host_base_url,
    )
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await run_visual_chain_once(bus)
            except Exception:
                logger.exception("unhandled visual chain error")
            try:
                if stop_event is not None:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=settings.visual_chain_interval_sec
                    )
                    break
                await asyncio.sleep(settings.visual_chain_interval_sec)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()
