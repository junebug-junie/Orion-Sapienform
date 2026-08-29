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

Patch 4 (shipped 2026-08-27): live-caught -- Juniper reported "still doing
the same images of Roman aqueducts, no change" a few hours after Patch 3
shipped. Real: `prior_description` continuity had locked onto one visual
attractor across 10+ runs / 100+ minutes, predating Patch 3 and unmoved by
it -- `context_text` is real, correctly varying content (confirmed live in
Postgres), but a short abstract clause ("Orion is currently thinking: the
coalition is fixated on ...") has nowhere near the prompt weight of a long,
concrete continuity description, and abstract cognitive-state narration
isn't strongly visualizable content regardless of prompt order.
`resolve_visual_chain_continuity` below is the actual fix: a deterministic,
testable reset -- after `settings.visual_chain_continuity_max_runs`
CONSECUTIVE runs carrying `prior_description` forward, the next run forces
continuity to drop for that one prompt, then continuity resumes normally.
Not a prompt-reweighting guess; a mechanical guarantee the loop cannot run
unbounded.

Patch 5 (this changeset, design doc §16): a second, richer context-seed --
`build_visual_prompt` now also takes `self_study_text`
(`store.load_latest_self_study_reflection`), the self-study analysis
system's real quantified self-observation (window-contrast prose:
"vision events dropped 0.36x vs baseline, a status category disappeared"),
not a bare narration sentence. Live-caught 2026-08-27, same session as
Patch 4: Juniper directly asked for "actual memory or a recent chat" as a
context-seed; live-checking `memory_crystallizations` (the actual-memory
candidate) found its `summary`/`subject` columns hold VERBATIM personal
chat content -- including a real, sensitive example naming a family
member's medical history -- with no safe column or `kind` filter available
on that table as it stands. Declined outright, not wired in. Self-study
analysis was the one candidate that live-verified safe: its four
deterministic producers (concept induction, vision events, affective
state, co-creation signals) render pure numeric window-contrasts, no chat
quotes, confirmed by reading real bodies before writing any code.
`store._SAFE_SELF_STUDY_SOURCE_PREFIXES` is an explicit allowlist of only
those four producers -- `source_kind='self_study'` also covers a sibling
free-form "Curiosity" reflection confirmed live to quote sensitive personal
content, which this allowlist deliberately excludes.

Patch 6 (design doc §17): a third context-seed, `memory_text` --
`store.load_latest_memory_crystallization`, real shared-life content from
the Recall system's `memory_crystallizations` table. Reverses Patch 5's
declined call on this same table, on new evidence rather than a policy
change: the only consumer of this reader's output is this service's own
`reverie_visual_chain` row (surfaced via `reverie_routes.py`, which has no
external port mapping and no auth/multi-user surface), so there is no
second audience for that content beyond the one person who is also its
original source. See `store.load_latest_memory_crystallization`'s own
docstring for the full writeup.

Patch 7 (design doc §18): the REAL root cause of "the memory got washed
out and Orion just kept generating stars" (live report, 2026-08-28, same
day Patch 6 shipped) -- Patches 3/5/6 concatenated ALL THREE context-seeds
into one prompt string, but `orion-diffusion-host`'s SDXL-turbo model
truncates its CLIP text encoder input at 77 tokens, SILENTLY (diffusers'
own default, no exception, no response-visible signal). Verified live
with the real tokenizer against an actual generated prompt: 191 real
tokens, encoder only sees the first 77 -- cutting off mid self-study
clause and NEVER reaching `memory_text` or even the trailing style
suffix. Every context-seed added after Patch 3 had, in practice, almost
never actually reached the model, regardless of how correctly it was
computed, stored, and displayed in the Hub tab -- a real "no empty-shell
cognition" (CLAUDE.md §0A) violation once understood: the UI honestly
showed content the image could not possibly have reflected.

Fix, two parts:
  1. `select_context_slot` below: stop concatenating all three -- round-
     robin ONE per run (after `prior_description`/continuity, which keeps
     its own separate reset mechanism from Patch 4). Reduces the realistic
     worst case from 4 competing clauses to 2 (continuity + one selected
     slot), and each individual slot's own char cap was independently
     re-derived against the REAL tokenizer (see `store.py`'s
     `MAX_SELF_STUDY_CONTEXT_CHARS`/`MAX_MEMORY_CRYSTALLIZATION_CONTEXT_
     CHARS` -- both cut substantially from their Patch 5/6 values, which
     were never checked against a real token budget at all).
  2. `orion-diffusion-host`'s `_run_generation` now logs a WARNING with
     real token counts whenever a prompt exceeds either of SDXL's two
     text-encoder budgets (CLIP-L and OpenCLIP-bigG) -- this exact failure
     mode was previously invisible in every log this system produces; it
     took a forensic re-tokenization of an already-stored prompt to find
     it. Visibility only (does not change what gets generated) -- the
     actual behavioral fix is #1.

A genuinely long-context text encoder (T5-XXL, used by FLUX.1/SD3.5)
would remove this ceiling entirely, but that is a real model-swap
decision (different VRAM footprint, different generation parameters, a
fresh Circe GPU/VRAM check) -- out of scope for this patch, which fixes
the model actually running today.

Patch 8 (design doc §22): FLUX.1-schnell shipped (§19) and Patch 7's rotation
fixed which clause reaches the model, but Juniper's next report was the real
remaining gap -- "how does this translate into fluffy cloud??" pointed at a
`memory_text` clause about moving a server between Ethernet ports. Every
context-seed above is composed VERBATIM into the prompt (`build_visual_prompt`
is pure string concatenation, no semantic step at all); the diffusion model
then pattern-matches whatever concrete nouns happen to be in that raw text,
and generic abstract prose has none -- so it falls back to its own generic
"soft dreamlike style" priors (clouds, nebulas, aqueducts), regardless of how
correctly the text was selected, capped, and displayed.

`interpret_context_for_visual` below closes this: one metacog-routed
cortex-exec call (`visual_context_interpret` verb) that asks the LLM to
invent a concrete visual metaphor for the selected slot's actual meaning,
run between `select_context_slot` and `build_visual_prompt`. Deliberately
the SAME reuse the text-reverie chain already established for its own
metacog-routed narration call (`reverie.py`'s `_metacog_route()`) -- not a
new mechanism, and per Juniper's stated priority ("diffusion trumps text if
we are too tight"), this call always stays on plain `metacog` (never
`metacog_background`), so it is never the one waiting under lane contention.
Fails open on any timeout/error/empty response straight back to the raw
selected-slot text -- `build_visual_prompt`'s own fallback behavior is
completely unchanged, so a metacog outage degrades this run's imagery
quality, never breaks the chain. Default ON
(`ORION_VISUAL_CHAIN_INTERPRETATION_ENABLED`): this is the actual fix for
the reported bug, not an experiment to observe first -- unlike Patch 8's
sibling `ORION_REVERIE_METACOG_BACKGROUND_ENABLED` (a routing-priority
change Juniper explicitly wanted to watch fail before activating), there is
no failure mode here worth observing unmitigated first.

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

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.reverie.visual_storage import StoredVisualArtifact, store_visual_artifact
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.reverie_visual import ReverieVisualArtifactV1, ReverieVisualChainV1
from orion.schemas.vision import VisionTaskRequestPayload, VisionTaskResultPayload

from .bus_listener import extract_stance_react_payload
from .cortex_client import CortexExecClient
from .settings import settings
from .store import (
    load_latest_memory_crystallization,
    load_latest_reverie_interpretation,
    load_latest_self_study_reflection,
    load_latest_visual_chain_continuity_state,
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


# Framing prefix per context-seed source, keyed by the slot name
# `select_context_slot` returns -- moved out of `build_visual_prompt`
# itself (Patch 7) since only ONE slot's text is ever composed into a
# prompt now, not all three.
_CONTEXT_SLOT_LABELS: dict[str, str] = {
    "context": "Orion is currently thinking",
    "self_study": "Orion recently noticed",
    "memory": "Orion remembers",
}


def select_context_slot(
    context_text: str | None,
    self_study_text: str | None,
    memory_text: str | None,
    rotation_index: int,
) -> tuple[str | None, str | None, int]:
    """Patch 7 (module docstring): pick ONE of the three context-seeds to
    actually put in the prompt this run, round-robin among whichever
    currently have real content.

    Why one, not all three: SDXL-turbo's CLIP text encoder truncates at 77
    tokens, silently. Concatenating all three (Patches 3/5/6's original
    design) meant real prompts routinely hit 150-200+ tokens -- verified
    live 2026-08-28 against an actual generated prompt (191 tokens) -- so
    every clause after the first was, in practice, invisible to the model
    regardless of how correct its content was. Rotation guarantees whichever
    ONE slot wins this run gets its full, real token budget instead of
    fighting two other clauses for the same 77-token window every single
    time.

    Pure function -- `rotation_index` is `store.load_latest_visual_chain_
    continuity_state()`'s third return value (a monotonically increasing
    counter this service writes every run). `idx = rotation_index %
    len(available)` re-indexes against whichever slots currently have
    content, so a currently-empty slot never gets skipped-but-still-counted
    the way a fixed `rotation_index % 3` would (a self-study body absent
    this tick costs nothing -- rotation just cycles the other two).

    Review finding, not fixed by design: this re-indexing is NOT guaranteed
    fair in the long run when the available set itself fluctuates tick to
    tick (verified by direct simulation) -- a slot that happens to be
    present on more ticks than another can end up visited disproportionately
    more often, since `rotation_index % len(available)` maps the SAME
    counter value to a different position whenever `len(available)`
    changes. This does not get anything stuck (progress and eventual
    coverage of every currently-available slot are still guaranteed each
    tick) -- it just is not a perfect long-run-fair scheduler under
    fluctuating availability, and this function does not attempt to be one.
    The actual regression this fixes (concatenating all three, guaranteeing
    2 of 3 are silently truncated away every single tick) does not require
    long-run fairness to be solved -- "sometimes one slot" beats "always
    the same one or two truncated."

    Returns `(slot_name, slot_text, next_rotation_index)`:
      - `slot_name`/`slot_text`: `None`/`None` when no context-seed has
        real content this run (`build_visual_prompt` then falls back to
        `prior_description`/`DEFAULT_SEED_PROMPT`, unchanged from before
        this patch).
      - `next_rotation_index`: the value to record in this run's
        `chain_json.context_slot_rotation`, for the next run's decision.
        Left unchanged when nothing was available this run -- no reason to
        advance a counter that picked nothing.
    """
    available = [
        (name, text)
        for name, text in (
            ("context", context_text),
            ("self_study", self_study_text),
            ("memory", memory_text),
        )
        if (text or "").strip()
    ]
    if not available:
        return None, None, rotation_index
    idx = rotation_index % len(available)
    slot_name, slot_text = available[idx]
    return slot_name, slot_text, rotation_index + 1


def build_visual_interpretation_plan_request(
    *,
    source_label: str,
    source_text: str,
    prior_description: str | None,
    correlation_id: str,
) -> PlanExecutionRequest:
    """Plan request for the `visual_context_interpret` verb (Patch 8, module
    docstring). Always plain `metacog`, never `metacog_background` -- see
    Patch 8's writeup on why this call is the one that must never wait.
    """
    plan = build_plan_for_verb("visual_context_interpret", mode="metacog")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source=settings.service_name,
            extra={
                "llm_profile": "metacog",
                "mode": "metacog",
                "llm_route": "metacog",
                "execution_lane": "background",
            },
        ),
        context={
            "source_label": source_label,
            "source_text": source_text,
            "prior_description": (prior_description or "").strip() or None,
            "options": {"llm_lane": "background", "allow_chat_fallback": False},
        },
    )


async def interpret_context_for_visual(
    bus: OrionBusAsync,
    *,
    cortex_client: CortexExecClient | None,
    slot_name: str,
    slot_text: str,
    prior_description: str | None,
    correlation_id: str,
    timeout_sec: float,
) -> str | None:
    """Turn the selected context-seed clause into a concrete visual metaphor
    via one metacog-routed cortex-exec call (Patch 8, module docstring).

    Fails open to `None` on ANY failure -- timeout, RPC error, malformed or
    empty response -- never raises. The caller falls back to `slot_text`
    unchanged, exactly today's behavior, so a metacog outage degrades this
    run's imagery, never breaks the chain.
    """
    client = cortex_client or CortexExecClient(
        bus, request_channel=settings.channel_cortex_exec_request
    )
    label = _CONTEXT_SLOT_LABELS.get(slot_name, "Orion notices")
    plan_request = build_visual_interpretation_plan_request(
        source_label=label,
        source_text=slot_text,
        prior_description=prior_description,
        correlation_id=correlation_id,
    )
    try:
        exec_result = await client.execute_plan(
            source=_source(),
            req=plan_request,
            correlation_id=correlation_id,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        logger.warning(
            "visual chain interpretation call failed slot=%s corr=%s err=%s",
            slot_name, correlation_id, exc,
        )
        return None
    try:
        raw = extract_stance_react_payload(exec_result)
    except Exception as exc:
        logger.warning(
            "visual chain interpretation payload missing slot=%s corr=%s err=%s",
            slot_name, correlation_id, exc,
        )
        return None
    text = raw.strip() if isinstance(raw, str) else ""
    return text or None


def build_visual_prompt(
    prior_description: str | None,
    context_slot_name: str | None = None,
    context_slot_text: str | None = None,
) -> str:
    """The diffusion prompt for one run. See module docstring for scope.

    Two independent inputs, each optional: `prior_description` (visual
    continuity -- the previous run's own re-observed caption) and ONE
    selected context-seed (`context_slot_name`/`context_slot_text` --
    Patch 7's `select_context_slot`, the round-robin winner among
    {context_text, self_study_text, memory_text} for THIS run only).
    Continuity keeps the image chain visually coherent frame-to-frame; the
    selected context-seed keeps it grounded in what Orion is actually
    narrating/observing/remembering instead of drifting into purely
    self-referential imagery with nothing anchoring it to a real cognitive
    state. Falls back to DEFAULT_SEED_PROMPT only when both are empty.

    Was a four-way list-join over all three context-seeds at once
    (Patches 3/5/6) until Patch 7 found that design silently discarded
    everything past the diffusion model's real 77-token budget -- see
    `select_context_slot`'s own docstring and the module docstring's
    Patch 7 entry for the live evidence. This function's own wording for
    each labeled clause is unchanged; only how many clauses it is ever
    asked to compose changed (always at most one selected slot now, never
    up to three).
    """
    prior = (prior_description or "").strip()
    slot_text = (context_slot_text or "").strip()
    clauses = []
    if prior:
        clauses.append(prior)
    if slot_text and context_slot_name in _CONTEXT_SLOT_LABELS:
        clauses.append(f"{_CONTEXT_SLOT_LABELS[context_slot_name]}: {slot_text}")
    if not clauses:
        return DEFAULT_SEED_PROMPT
    style = (
        "Continue this train of imagination, soft dreamlike style."
        if prior
        else "Soft abstract dreamlike style."
    )
    return ". ".join(clauses) + ". " + style


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
    bus: OrionBusAsync, *, now_fn: Any = _now, cortex_client: CortexExecClient | None = None
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
                                  self_study_text: str | None, memory_text: str | None,
                                  context_slot_used: str | None, context_slot_rotation: int,
                                  context_slot_interpreted: str | None,
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
                "self_study_text": self_study_text,
                "memory_text": memory_text,
                "context_slot_used": context_slot_used,
                "context_slot_rotation": context_slot_rotation,
                "context_slot_interpreted": context_slot_interpreted,
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
        # Four independent reads (different tables, no data dependency) --
        # concurrent so the cost is max() of the round trips, not sum()
        # (review finding: this function already makes exactly this
        # argument a few lines below for store_visual_artifact/
        # upload_to_percept_store; the same reasoning applies here).
        # prior_description, continuity_streak, AND context_slot_rotation
        # come from the SAME row of the SAME table, so they're one combined
        # read (review finding on the original 2-value version: two
        # separate round trips to the same row wasted a query and left a
        # theoretical read-your-own-write race), not three gathered reads.
        (
            (prior_description, continuity_streak, context_slot_rotation),
            context_text,
            self_study_text,
            memory_text,
        ) = await asyncio.gather(
            asyncio.to_thread(load_latest_visual_chain_continuity_state),
            asyncio.to_thread(
                load_latest_reverie_interpretation,
                char_limit=settings.reverie_context_char_limit,
                max_age_sec=settings.reverie_context_max_age_sec,
            ),
            asyncio.to_thread(
                load_latest_self_study_reflection,
                char_limit=settings.self_study_context_char_limit,
                max_age_sec=settings.self_study_context_max_age_sec,
            ),
            asyncio.to_thread(
                load_latest_memory_crystallization,
                char_limit=settings.memory_crystallization_context_char_limit,
                max_age_sec=settings.memory_crystallization_context_max_age_sec,
            ),
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
        # Review finding: on a reset run, the ORIGINAL (stale, pre-reset)
        # prior_description must never come back as this row's own
        # prior_description -- that would silently resurrect the exact
        # attractor the reset just broke out of the moment generation,
        # storage, or captioning fails on this run (a real, previously-
        # untested failure mode: the next tick would read streak=0 against
        # the SAME stale text and grind through another full max_runs cycle
        # before resetting again). A non-reset run keeps the pre-Patch-4
        # behavior unchanged: carry the old value forward on any failure.
        continuity_fallback = None if continuity_reset else prior_description
        # Patch 7 (module docstring): pick ONE context-seed for this run's
        # prompt instead of concatenating all three -- see
        # select_context_slot's own docstring for why. Computed here
        # (before generation) so a failed run still records which slot/
        # rotation value it used, same discipline as continuity_streak.
        context_slot_used, context_slot_text, context_slot_rotation = select_context_slot(
            context_text, self_study_text, memory_text, context_slot_rotation
        )
        # Patch 8 (module docstring): turn the raw selected clause into a concrete visual
        # metaphor before composing the prompt. Fails open to None -- build_visual_prompt then
        # gets context_slot_text unchanged, exactly Patch 7's behavior, so a metacog outage
        # degrades this run's imagery, never breaks the chain.
        context_slot_interpreted: str | None = None
        if context_slot_used and context_slot_text and settings.visual_chain_interpretation_enabled:
            context_slot_interpreted = await interpret_context_for_visual(
                bus,
                cortex_client=cortex_client,
                slot_name=context_slot_used,
                slot_text=context_slot_text,
                prior_description=effective_prior,
                correlation_id=chain_id,
                timeout_sec=settings.visual_chain_interpretation_timeout_sec,
            )
        prompt = build_visual_prompt(
            effective_prior, context_slot_used, context_slot_interpreted or context_slot_text
        )

        try:
            png_bytes = await asyncio.to_thread(
                call_diffusion_generate,
                prompt,
                base_url=settings.diffusion_host_base_url,
                timeout_sec=settings.visual_chain_diffusion_timeout_sec,
            )
        except Exception as exc:
            return await _generation_failed(
                chain_id, exc, prompt, continuity_fallback, context_text, self_study_text,
                memory_text, context_slot_used, context_slot_rotation, context_slot_interpreted,
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
                chain_id, store_result, prompt, continuity_fallback, context_text, self_study_text,
                memory_text, context_slot_used, context_slot_rotation, context_slot_interpreted,
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
        # re-observation forwards `continuity_fallback` (the previous
        # prior_description unchanged on a normal run, or None on a reset run
        # -- see continuity_fallback's own comment above) rather than
        # propagating a stale value on a reset run or losing continuity
        # entirely on a normal one.
        next_prior_description = description or continuity_fallback

        chain = ReverieVisualChainV1(
            chain_id=chain_id,
            created_at=now_fn(),
            terminal_reason="max_steps",
            prior_description=next_prior_description,
            chain_json={
                "prompt": prompt,
                "context_text": context_text,
                "self_study_text": self_study_text,
                "memory_text": memory_text,
                "context_slot_used": context_slot_used,
                "context_slot_rotation": context_slot_rotation,
                "context_slot_interpreted": context_slot_interpreted,
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
