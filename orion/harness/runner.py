from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable

from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
from orion.fcc.context_budget import (
    apply_context_overflow_hint,
    is_context_overflow_text,
    measure_step_payload_chars,
)
from orion.harness.fcc_motor import (
    _ROUTE_PROBE_BACKENDS as _POOL_BACKENDS,
    DEFAULT_FCC_MODEL_LABEL,
    _extract_tool_name,
    _extract_tool_result_errors,
    classify_step_tool_kind,
    expand_env_path,
    extract_result_output_tokens,
    load_fcc_env,
    probe_current_served_model,
    resolve_auth_token,
    resolve_fcc_backend,
    run_fcc_turn,
    is_progress_frame,
    summarize_harness_step,
)
from orion.harness.grammar_emit import (
    HarnessGrammarCollector,
    build_harness_grammar_events,
    publish_harness_lifecycle_grammar,
    short_error_kind,
)
from orion.gpu_pool.placement import (
    ServingPlacement,
    discovered_role,
    fetch_pool_state,
    placement_from_lease,
    placement_from_route_default,
)
from orion.harness.grammar_publish import publish_harness_step_grammar
from orion.harness.cut_short import TurnFindings, build_cut_short_draft, is_cut_short_code
from orion.harness.last_tool_fetch_cache import publish_last_tool_fetch, read_last_tool_fetch
from orion.harness.attachment_staging import describe_for_prompt
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.reading_receipts import (
    ReadingReceiptTracker,
    enforce_reading_receipt_grounding,
)
from orion.harness.repair import map_repair_pressure_contract
from orion.harness.step_stream import publish_harness_run_step
from orion.harness.tool_provenance_audit import detect_tool_provenance_mismatch, fetch_shaped_tool_names
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import (
    GrammarReceiptV1,
    HarnessDraftMoleculeV1,
    HarnessRepairOverlayV1,
    HarnessRunRequestV1,
)
from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1
from orion.schemas.reading import ReadingRecommendationOutcomeV1, SourceFetchEvidenceV1
from orion.schemas.thought import CoalitionSnapshotV1, ThoughtEventV1

logger = logging.getLogger("orion.harness.runner")

FccRunner = Callable[..., AsyncIterator[dict[str, Any]]]


@dataclass
class HarnessMotorResult:
    draft_text: str
    grammar_receipts: list[GrammarReceiptV1] = field(default_factory=list)
    step_count: int = 0
    exit_code: int | None = None
    compliance_verdict: str = "completed"
    grounding_status: str = "grounded"
    # Set to the motor error code when the motor stopped a still-working turn
    # (orion/harness/cut_short.py::CUT_SHORT_CODES) and the draft was built from
    # the turn's own findings. The governor keeps the cut-short marker on the
    # final text so finalize/repair cannot present it as a finished answer.
    cut_short_reason: str | None = None
    draft_molecule: HarnessDraftMoleculeV1 | None = None
    grammar_collector: HarnessGrammarCollector | None = None
    # Wall time for the FCC leg ALONE -- the motor loop, not the turn. This is
    # the quantity `HARNESS_FCC_TIMEOUT_SEC` (7200s) actually compares against,
    # and the one that decides `grounding_status == "fcc_timeout"`.
    #
    # Nothing measured it before. Hub could only time the WHOLE unified turn
    # (stance <=400s + governor queue + this + finalize <=1025s), so up to
    # ~1425s of what it recorded was not the motor -- and for a timed-out run
    # this leg is pinned at 7200s by construction, meaning every bit of
    # variance Hub could see was overhead. With this, a grounded run's
    # distance from 7200s is real headroom and "the budget is too small"
    # stops being a guess.
    fcc_elapsed_sec: float | None = None
    # Where the FCC leg ran, for the RPC-health hop key `fcc:<role>`
    # (services/orion-harness-governor/app/bus_listener.py::fcc_hop_key). A held
    # turn (request.gpu_lease, a durable run's hold) runs every call on the
    # hold's role, so `serving_role` is that granted role. Without a hold each
    # call is placed on its own and the harness never sees the grants, so only
    # the requested gateway route (`fcc_route`) is known. Both are known before
    # the subprocess starts, so an early timeout keys the same as a success.
    serving_role: str | None = None
    fcc_route: str | None = None
    # A non-pool backend (e.g. MODEL_HAIKU -> nvidia_nim): its own latency population, keyed
    # `fcc:backend:<backend>`, never folded into `fcc:unknown`.
    fcc_backend: str | None = None
    # Verbosity/stuck-loop signals (see runner.py's step loop for how these accumulate).
    # Carried on the result object -- not just recorded into grammar_collector -- because
    # services/orion-harness-governor/app/bus_listener.py's _emit_finalize_lifecycle_grammar
    # re-calls record_result_assembled() on this SAME collector after the motor's own
    # lifecycle publish; that second call must re-pass these or it silently overwrites them
    # back to 0/"unknown" in the event grammar_extract.py actually sees (record_result_assembled
    # is idempotent-per-atom_id, so the second call replaces the first's atom in the collector).
    step_char_sum: int = 0
    step_char_max: int = 0
    tool_failure_streak_max: int = 0
    # Real output_tokens from the harness CLI's own result-event usage object, and a
    # deterministic tool-name step-kind split (see fcc_motor.py's
    # extract_result_output_tokens/classify_step_tool_kind) -- same carry-through
    # reason as step_char_sum above (bus_listener.py re-invokes record_result_assembled
    # on this same collector).
    reasoning_output_tokens: int = 0
    context_gathering_step_count: int = 0
    execution_step_count: int = 0
    # Distinct from grounding_status: that field is an overloaded error/
    # overflow code already surfaced as a user-visible error by downstream
    # consumers (orion/hub/turn_orchestrator.py, orion-harness-governor).
    # This is a soft audit signal, not a motor failure -- never repurpose
    # grounding_status for it.
    tool_provenance_audit: str | None = None
    # Real backend model the CLI's stream-json "assistant" events echoed back
    # (see fcc_motor.py's _served_model_from_assistant), distinct from the
    # ~/.fcc/.env route alias in HarnessRunRequestV1.fcc_model_label -- that
    # label alone can't distinguish MODEL_SONNET from MODEL_OPUS when both
    # point at the same route. None when discovery never fired (e.g. a
    # fast-fail before any assistant turn).
    fcc_served_model: str | None = None
    reading_receipts: list[ReadingRecommendationOutcomeV1] = field(default_factory=list)
    # Fetch tool calls that returned usable source content (tool-trace evidence,
    # see SourceFetchEvidenceV1). Empty on every path that never ran the motor.
    source_fetches: list[SourceFetchEvidenceV1] = field(default_factory=list)


def _default_harness_node_name() -> str:
    return os.environ.get("HARNESS_NODE_NAME", "athena")


def _served_model_from_metadata(meta: object) -> str | None:
    """Read fcc_motor.py's fcc_served_model out of a "final"/"error" event's
    metadata dict, if present. Shared by both branches below so the
    extraction rule (and its stripping) can't drift between a clean turn and
    a degraded/partial one.
    """
    if not isinstance(meta, dict):
        return None
    raw = meta.get("fcc_served_model")
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    return raw or None


def _record_recall_gate_from_debug(
    collector: HarnessGrammarCollector,
    recall_debug: dict[str, Any] | None,
) -> None:
    if recall_debug is None:
        return
    collector.record_recall_gate_observed(
        run_recall=True,
        profile=recall_debug.get("profile"),
        reason=str(recall_debug.get("source") or "recall_observed"),
    )


def build_coalition_snapshot(thought: ThoughtEventV1) -> CoalitionSnapshotV1:
    attended = list(dict.fromkeys([*thought.strain_refs, *thought.evidence_refs]))
    return CoalitionSnapshotV1(
        attended_node_ids=attended,
        selected_open_loop_id=None,
        open_loop_ids=[],
        generated_at=datetime.now(timezone.utc),
        broadcast_stale=False,
    )


def is_chat_reply_request(request: Any) -> bool:
    """True only for a Hub chat reply: a turn Juniper started by typing.

    Curiosity, self-inquiry and urgent runs carry utterance_origin="orion";
    outreach, collapse-mirror and reading turns carry none. Missing field (an
    older Hub build) reads as not-a-chat-reply, which is the pre-L7 behavior.
    """
    return str(getattr(request, "utterance_origin", None) or "").strip().lower() == "juniper"


def _draft_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def build_harness_prompt(
    *,
    thought: ThoughtEventV1,
    user_message: str,
    repair_overlay: HarnessRepairOverlayV1,
    answer_contract: AnswerContract | None = None,
    workspace: str | None = None,
    prior_tool_fetch_names: list[str] | None = None,
    attachments: list[Any] | None = None,
    serving_placement: ServingPlacement | None = None,
    recent_turns: list[TurnWindowMessageV1] | None = None,
    situation_prompt_fragment: str | None = None,
    reading_binding: Any = None,
    reading_only: bool = False,
) -> str:
    prefix = compile_harness_prefix(
        thought,
        repair_overlay=repair_overlay,
        user_message=user_message,
        answer_contract=answer_contract,
        workspace=workspace or os.environ.get("HARNESS_FCC_WORKSPACE"),
        prior_tool_fetch_names=prior_tool_fetch_names,
        serving_placement=serving_placement,
        recent_turns=recent_turns,
        situation_prompt_fragment=situation_prompt_fragment,
        reading_binding=reading_binding,
        reading_only=reading_only,
    )
    instruction = harness_motor_instruction(
        thought=thought,
        answer_contract=answer_contract,
    )
    # Appended last so it sits closest to the instruction, and only when there
    # is something to say -- a text-only turn's prompt is unchanged byte for
    # byte, which is what keeps this safe for every existing turn.
    attachment_block = describe_for_prompt(attachments or [])
    if user_message.strip():
        return f"{prefix}\n\n{instruction}{attachment_block}"
    return f"{prefix}{attachment_block}"


def build_draft_molecule(
    *,
    correlation_id: str,
    thought: ThoughtEventV1,
    draft_text: str,
    grammar_receipts: list[GrammarReceiptV1],
    coalition_snapshot: CoalitionSnapshotV1,
    repair_overlay: HarnessRepairOverlayV1,
    tool_provenance_audit: str | None = None,
) -> HarnessDraftMoleculeV1:
    return HarnessDraftMoleculeV1(
        correlation_id=correlation_id,
        thought_event_id=thought.event_id,
        draft_text=draft_text,
        draft_hash=_draft_hash(draft_text),
        thought_event=thought,
        grammar_receipts=list(grammar_receipts),
        coalition_snapshot=coalition_snapshot,
        repair_overlay_mode=repair_overlay.mode if repair_overlay.mode != "default" else None,
        tool_provenance_audit=tool_provenance_audit,
    )


async def default_fcc_runner(
    *,
    prompt: str,
    correlation_id: str,
    fcc_model_label: str | None = None,
    timeout_sec: float = 120.0,
    reading_binding: Any = None,
    reading_only: bool = False,
    gpu_lease: dict[str, Any] | None = None,
    pool_state: dict[str, Any] | None = None,
    chat_reply: bool = False,
    **_: Any,
) -> AsyncIterator[dict[str, Any]]:
    env_path = expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env"))
    env = load_fcc_env(env_path)
    token = resolve_auth_token(env, override=os.environ.get("HARNESS_FCC_AUTH_TOKEN", ""))
    async for event in run_fcc_turn(
        chat_reply=chat_reply,
        gpu_lease=gpu_lease,
        pool_state=pool_state,
        reading_binding=reading_binding,
        reading_only=reading_only,
        prompt=prompt,
        correlation_id=correlation_id,
        fcc_model_label=fcc_model_label,
        workspace=os.environ.get("HARNESS_FCC_WORKSPACE", os.getcwd()),
        fcc_server_url=os.environ.get(
            "HARNESS_FCC_SERVER_URL",
            os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080"),
        ),
        auth_token=token,
        claude_bin=os.environ.get("HARNESS_FCC_CLAUDE_BIN", "claude"),
        timeout_sec=timeout_sec,
    ):
        yield event


class HarnessRunner:
    """FCC motor loop: harness prefix → fcc steps → grammar receipts → draft_text."""

    def __init__(
        self,
        bus: Any,
        *,
        grammar_channel: str = "orion:grammar:event",
        step_channel: str = "orion:harness:run:step",
        fcc_runner: FccRunner | None = None,
        fcc_timeout_sec: float = 120.0,
        node_name: str | None = None,
        served_model_probe: Callable[..., Awaitable[str | None]] | None = None,
        pool_state_probe: Callable[[], Awaitable[dict[str, Any] | None]] | None = None,
    ) -> None:
        self.bus = bus
        self.grammar_channel = grammar_channel
        self.step_channel = step_channel
        self.fcc_runner = fcc_runner or default_fcc_runner
        self.fcc_timeout_sec = fcc_timeout_sec
        self.served_model_probe = served_model_probe or probe_current_served_model
        # The GPU pool's live state (discovered role -> profile/model/ctx, plus the pool's config so
        # a route can be mapped to the role it lands on). Read at most once per turn, for a held
        # turn or a pool-backed route, and shared by the prompt's self-context line and the motor's
        # window probe (GPU pool stage 6.3: both used to read the gateway's GET /routes).
        # Injectable for tests.
        self.pool_state_probe = pool_state_probe or self._read_pool_state
        self.node_name = node_name or _default_harness_node_name()

    async def _read_pool_state(self) -> dict[str, Any] | None:
        return await fetch_pool_state(self.bus, source="orion-harness-governor", include_config=True)

    async def run(
        self,
        request: HarnessRunRequestV1,
        *,
        repair_overlay: HarnessRepairOverlayV1 | None = None,
        coalition_snapshot: CoalitionSnapshotV1 | None = None,
        publish_grammar_fn: Callable[..., Awaitable[None]] | None = None,
        recall_debug: dict[str, Any] | None = None,
    ) -> HarnessMotorResult:
        """Orion capability: FCC motor loop for the unified turn.

        Drives one fcc-claude turn from the compiled harness prefix and
        converts streamed steps into grammar receipts, producing a
        HarnessMotorResult with a draft molecule. The draft is not the
        user-visible answer — substrate appraisal, reflection, and voice
        finalization happen downstream in the finalize chain.

        Runtime evidence: per-step grammar receipts on the grammar channel,
        step frames relayed to Hub, and HarnessDraftMoleculeV1. Start here
        when the motor produced no draft, wrong receipts, or a failed or
        timed-out exit.
        """
        # The FCC leg's own clock. Started before ANY work in this method --
        # the served-model probe and the concurrent bus reads below are part of
        # the leg the 2400s deadline governs, so excluding them would understate
        # it in exactly the direction that hides a budget problem.
        fcc_started = time.monotonic()
        thought = request.thought_event
        overlay = repair_overlay or map_repair_pressure_contract(request.repair_pressure_contract)
        coalition = coalition_snapshot or build_coalition_snapshot(thought)

        gpu_lease = getattr(request, "gpu_lease", None)
        serving_role = gpu_lease.role if gpu_lease is not None else None
        # Same default the motor applies (run_fcc_turn), so the hop key and the
        # route-default probe name the route the subprocess actually asks for.
        fcc_label = request.fcc_model_label or DEFAULT_FCC_MODEL_LABEL
        fcc_route: str | None = None
        fcc_backend: str | None = None
        try:
            parsed = resolve_fcc_backend(fcc_label)
        except Exception:
            logger.warning("fcc_route_resolve failed corr=%s", request.correlation_id, exc_info=True)
            parsed = None
        if parsed is not None:
            if parsed[0] in _POOL_BACKENDS:
                fcc_route = parsed[1]
            else:
                fcc_backend = parsed[0]

        turn_pool_state: dict[str, Any] | None = None

        async def _probe_serving_placement() -> ServingPlacement | None:
            # Best-effort self-context: which real backend serves this turn.
            # A held turn: the hold's role (a fact -- every call under a hold
            # runs on it) and the pool's discovered profile for that role.
            # Otherwise: the route's default model, stated as a default, since
            # the pool places each unheld call on its own and may spill it
            # (spec 2026-09-24-gpu-pool-design.md, reader impacts item 5).
            # Both read GPU pool state, once per turn; the same state then sizes
            # the motor's window (stage 6.3: no GET /routes read). Never the
            # reason a turn doesn't start.
            nonlocal turn_pool_state
            try:
                if serving_role or fcc_route:
                    turn_pool_state = await self.pool_state_probe()
                if serving_role:
                    return placement_from_lease(serving_role, discovered_role(turn_pool_state, serving_role))
                model = await self.served_model_probe(fcc_label, pool_state=turn_pool_state)
                return placement_from_route_default(fcc_route, model) if model else None
            except Exception:
                logger.warning(
                    "serving_placement_probe failed corr=%s", request.correlation_id, exc_info=True
                )
                return placement_from_lease(serving_role, None) if serving_role else None

        # Independent reads (bus lookup, pool/gateway probe) -- run concurrently
        # rather than serially so a slow/unreachable probe (its own timeout,
        # default 2s) doesn't add to the bus round-trip on top of its own latency.
        prior_tool_fetch, serving_placement = await asyncio.gather(
            read_last_tool_fetch(self.bus, session_id=thought.session_id),
            _probe_serving_placement(),
        )
        prior_tool_fetch_names = (prior_tool_fetch or {}).get("tool_names")

        prompt = build_harness_prompt(
            thought=thought,
            user_message=request.user_message,
            repair_overlay=overlay,
            answer_contract=request.answer_contract,
            workspace=os.environ.get("HARNESS_FCC_WORKSPACE"),
            prior_tool_fetch_names=prior_tool_fetch_names,
            attachments=list(getattr(request, "attachments", None) or []),
            serving_placement=serving_placement,
            recent_turns=list(getattr(request, "recent_turns", None) or []),
            situation_prompt_fragment=getattr(request, "situation_prompt_fragment", None),
            reading_binding=getattr(request, "reading_binding", None),
            reading_only=getattr(request, "reading_only", False),
        )

        try:
            await publish_harness_run_step(
                self.bus,
                correlation_id=request.correlation_id,
                step_index=-1,
                step={
                    "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
                    "prompt": prompt,
                    "prompt_char_len": len(prompt),
                },
                channel=self.step_channel,
            )
        except Exception:
            logger.warning(
                "harness motor_boot cockpit step publish failed corr=%s",
                request.correlation_id,
                exc_info=True,
            )

        collector = HarnessGrammarCollector(
            node_name=self.node_name,
            correlation_id=request.correlation_id,
            observed_at=datetime.now(timezone.utc),
            # getattr default, not request.mode directly: HarnessRunRequestV1.mode
            # is a new, optional field (2026-09-03) -- a request built by a Hub
            # build that predates it must not crash this on a rolling deploy.
            mode=str(getattr(request, "mode", None) or "orion").strip().lower(),
        )
        collector.record_request_received()
        collector.record_plan_started(step_count=0)
        _record_recall_gate_from_debug(collector, recall_debug)

        receipts: list[GrammarReceiptV1] = []
        step_count = 0
        last_recorded_step_order = 0  # order of the last non-progress frame in the collector
        draft_text = ""
        exit_code: int | None = None
        fcc_served_model: str | None = None
        compliance_verdict = "completed"
        grounding_status = "grounded"
        motor_failed = False
        # Per-step verbosity (chars) and repeated-tool-failure streak (idea 2/3 of
        # docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md).
        # Only the max streak is kept, not a growing list of every failure -- this
        # pipeline only ever sees a single end-of-run grammar flush per turn, so
        # "current streak" and "max streak" are the same observation in practice.
        step_char_sum = 0
        step_char_max = 0
        tool_failure_streak = 0
        tool_failure_streak_max = 0
        reasoning_output_tokens = 0
        context_gathering_step_count = 0
        execution_step_count = 0
        # Only updated inside the is_error loop below -- a step with zero tool_result
        # errors doesn't touch this state at all, so a streak persists across an
        # intervening clean step (e.g. fail/success/fail on the same error kind still
        # counts as a streak of 2, not reset by the successful step in between). This
        # matches "is Orion stuck repeating the same failure" better than a strict
        # consecutive-step definition would.
        _last_tool_failure_kind: str | None = None
        # Distinct from `motor_failed`: that's only True on a *fully* failed
        # turn (no partial text). This tracks the "error" event branch
        # regardless of whether a partial draft was salvaged, so a
        # timed-out/errored turn that still produced some visible text
        # doesn't get treated as clean for cross-turn continuity purposes.
        error_path_taken = False
        reading_tracker = ReadingReceiptTracker(
            getattr(request, "reading_binding", None),
            # Only a reading turn's fetched text is retained (Hub's reading pipeline);
            # chat/outreach/curiosity turns never ship page text over the bus.
            retain_text=bool(getattr(request, "reading_only", False)),
        )
        turn_findings = TurnFindings()
        cut_short_reason: str | None = None

        async for event in self.fcc_runner(
            **({"gpu_lease": request.gpu_lease.model_dump(mode="json")}
               if getattr(request, "gpu_lease", None) is not None else {}),
            **({"pool_state": turn_pool_state} if isinstance(turn_pool_state, dict) else {}),
            **({"reading_binding": request.reading_binding} if getattr(request, "reading_binding", None) else {}),
            **({"reading_only": True} if getattr(request, "reading_only", False) else {}),
            # Only a Juniper-originated Hub chat reply. Passed only when true so
            # custom fcc_runner callables that predate it keep working.
            **({"chat_reply": True} if is_chat_reply_request(request) else {}),
            prompt=prompt,
            correlation_id=request.correlation_id,
            fcc_model_label=request.fcc_model_label,
            timeout_sec=request.inference_timeout_sec or self.fcc_timeout_sec,
        ):
            etype = str(event.get("type") or "")
            if etype == "step":
                step = event.get("step")
                if not isinstance(step, dict):
                    continue
                reading_tracker.observe(step)
                # Heartbeat frames stay in the live stream, receipts and step_count (unchanged)
                # but are not grammar-recorded work: kept out of the collector atoms AND out
                # of step_char_sum/max so avg_step_chars (= sum / completed atoms) keeps a
                # numerator and denominator over the same population.
                progress_frame = is_progress_frame(step)
                if not progress_frame:
                    turn_findings.observe(step)
                    step_chars = measure_step_payload_chars(step)
                    step_char_sum += step_chars
                    step_char_max = max(step_char_max, step_chars)
                for error_text in _extract_tool_result_errors(step):
                    kind = short_error_kind(error_text)
                    if kind == _last_tool_failure_kind:
                        tool_failure_streak += 1
                    else:
                        tool_failure_streak = 1
                        _last_tool_failure_kind = kind
                    tool_failure_streak_max = max(tool_failure_streak_max, tool_failure_streak)
                summary = summarize_harness_step(step, index=step_count)
                if not progress_frame:
                    last_recorded_step_order = step_count + 1
                    collector.record_step_started(order=step_count + 1, summary=summary)
                tool_name = _extract_tool_name(step)
                step_kind = classify_step_tool_kind(tool_name)
                if step_kind == "context_gathering":
                    context_gathering_step_count += 1
                elif step_kind == "execution":
                    execution_step_count += 1
                result_tokens = extract_result_output_tokens(step)
                if result_tokens is not None:
                    reasoning_output_tokens = max(reasoning_output_tokens, result_tokens)
                receipt = await publish_harness_step_grammar(
                    self.bus,
                    correlation_id=request.correlation_id,
                    channel=self.grammar_channel,
                    step_index=step_count,
                    tool_name=tool_name,
                    summary=summary,
                    publish_fn=publish_grammar_fn,
                )
                receipts.append(receipt)
                step_count += 1
                if not progress_frame:
                    collector.record_step_completed(order=step_count)
                try:
                    await publish_harness_run_step(
                        self.bus,
                        correlation_id=request.correlation_id,
                        step_index=step_count - 1,
                        step=step,
                        channel=self.step_channel,
                    )
                except Exception:
                    logger.warning(
                        "harness run step publish failed corr=%s index=%s",
                        request.correlation_id,
                        step_count - 1,
                        exc_info=True,
                    )
            elif etype == "final":
                draft_text = str(event.get("llm_response") or "").strip()
                if is_context_overflow_text(draft_text):
                    draft_text = apply_context_overflow_hint(draft_text)
                meta = event.get("metadata")
                if isinstance(meta, dict):
                    raw_exit = meta.get("exit_code")
                    if isinstance(raw_exit, int):
                        exit_code = raw_exit
                seen_served_model = _served_model_from_metadata(meta)
                if seen_served_model:
                    fcc_served_model = seen_served_model
            elif etype == "error":
                error_path_taken = True
                partial = str(event.get("llm_response") or "").strip()
                error_code = str(event.get("error_code") or "").strip()
                error_msg = str(event.get("error") or "").strip()
                err_meta = event.get("metadata")
                seen_served_model = _served_model_from_metadata(err_meta)
                if seen_served_model:
                    fcc_served_model = seen_served_model
                # fcc_nonzero_exit carries the real exit code (negative = killed by a
                # signal, e.g. a Hub cancel's SIGKILL); keep it instead of None.
                if isinstance(err_meta, dict) and isinstance(err_meta.get("exit_code"), int):
                    exit_code = err_meta["exit_code"]
                # Not cut short when the CLI already reported the turn complete
                # (the process only hung on exit -- `partial` IS the answer), nor
                # on reading-only machine turns: their consumer needs JSON and
                # retries on a clean `turn_error:<code>`, which a findings draft
                # would turn into a finalize parse failure.
                result_seen = isinstance(err_meta, dict) and bool(err_meta.get("fcc_result_seen"))
                use_cut_short = (
                    is_cut_short_code(error_code)
                    and not result_seen
                    and not getattr(request, "reading_only", False)
                )
                cut_short_draft = (
                    build_cut_short_draft(
                        error_code=error_code,
                        step_count=step_count,
                        findings=turn_findings,
                        last_text=partial,
                    )
                    if use_cut_short
                    else ""
                )
                if cut_short_draft:
                    # The motor stopped a still-working turn. Its `llm_response` is
                    # only the LAST text fragment -- on a long investigation, a
                    # lead-in like "Let me check X:". Build the draft from what the
                    # turn actually recorded instead, plainly marked as cut short.
                    draft_text = cut_short_draft
                    compliance_verdict = "partial"
                    grounding_status = error_code
                    cut_short_reason = error_code
                elif use_cut_short:
                    # Cut short with nothing recorded: no findings to stand behind
                    # a draft, so the motor failed -- never a lone lead-in line.
                    compliance_verdict = "failed"
                    grounding_status = error_code
                    motor_failed = True
                elif partial:
                    draft_text = apply_context_overflow_hint(partial)
                    compliance_verdict = "partial"
                    grounding_status = error_code or "partial"
                else:
                    compliance_verdict = "failed"
                    # Prefer structured error_code (e.g. fcc_timeout) over the
                    # human message so Hub/tests can key off stable codes.
                    hinted = apply_context_overflow_hint(error_msg) if error_msg else ""
                    grounding_status = error_code or hinted or error_msg or "failed"
                    motor_failed = True
                if last_recorded_step_order > 0:
                    # Not step_count: trailing progress frames have no started atom to
                    # link the failure to.
                    collector.record_step_failed(
                        order=last_recorded_step_order,
                        error_kind=short_error_kind(error_code or error_msg),
                    )
                logger.warning(
                    "fcc motor error corr=%s code=%s err=%s",
                    request.correlation_id,
                    event.get("error_code"),
                    event.get("error"),
                )
                break

        reading_receipts = reading_tracker.outcomes()
        grounded_draft = enforce_reading_receipt_grounding(draft_text, reading_receipts)
        if grounded_draft != draft_text:
            logger.warning(
                "reading_receipt_grounding_applied corr=%s recommendations=%s "
                "accepted=%s unknown=%s",
                request.correlation_id,
                len(reading_receipts),
                sum(item.acceptance == "accepted" for item in reading_receipts),
                sum(item.acceptance == "unknown" for item in reading_receipts),
            )
            draft_text = grounded_draft

        # Post-hoc audit, not prevention: the fcc subprocess has already run
        # to completion by this point, so this can only flag a mismatch
        # between what the draft claims and this turn's own tool trace, not
        # stop it before generation. Computed once here (before either
        # _publish_motor_lifecycle call) so both the empty-draft and success
        # paths' grammar events pick it up via build_harness_grammar_events.
        tool_provenance_audit = detect_tool_provenance_mismatch(draft_text, receipts)
        if tool_provenance_audit:
            collector.record_tool_provenance_mismatch(mismatch=tool_provenance_audit)
            logger.warning(
                "harness_tool_provenance_mismatch corr=%s detail=%s",
                request.correlation_id,
                tool_provenance_audit,
            )

        async def _publish_motor_lifecycle(
            *, status: str, final_text_present: bool, compliance_verdict: str
        ) -> None:
            collector.record_result_assembled(
                status=status,
                final_text_present=final_text_present,
                step_count=step_count,
                grammar_receipt_count=len(receipts),
                reflection_ran=False,
                quick_lane_skipped_5b=True,
                compliance_verdict=compliance_verdict,
                step_char_sum=step_char_sum,
                step_char_max=step_char_max,
                tool_failure_streak_max=tool_failure_streak_max,
                reasoning_output_tokens=reasoning_output_tokens,
                context_gathering_step_count=context_gathering_step_count,
                execution_step_count=execution_step_count,
            )
            events = build_harness_grammar_events(collector)
            logger.info(
                "harness_motor_lifecycle_publish_begin corr=%s trace_id=%s events=%s",
                request.correlation_id,
                collector.trace_id,
                len(events),
            )
            try:
                await publish_harness_lifecycle_grammar(
                    self.bus,
                    channel=self.grammar_channel,
                    events=events,
                )
            except Exception:
                logger.warning(
                    "harness_motor_lifecycle_grammar_publish_failed corr=%s",
                    request.correlation_id,
                    exc_info=True,
                )

        if not draft_text:
            await _publish_motor_lifecycle(
                status="failed",
                final_text_present=False,
                # Same normalization as the return value below: an empty draft is
                # always a real failure even if the loop never took the "error"
                # branch (e.g. the fcc subprocess just produced nothing).
                compliance_verdict=(
                    compliance_verdict if compliance_verdict != "completed" else "failed"
                ),
            )
            logger.info(
                "harness_motor_complete corr=%s steps=%s grammar_receipts=%s verdict=%s grounding=%s draft_len=0",
                request.correlation_id,
                step_count,
                len(receipts),
                compliance_verdict if compliance_verdict != "completed" else "failed",
                grounding_status if grounding_status != "grounded" else "empty_draft",
            )
            return HarnessMotorResult(
                draft_text="",
                fcc_elapsed_sec=round(time.monotonic() - fcc_started, 3),
                grammar_receipts=receipts,
                step_count=step_count,
                exit_code=exit_code,
                compliance_verdict=compliance_verdict if compliance_verdict != "completed" else "failed",
                grounding_status=grounding_status if grounding_status != "grounded" else "empty_draft",
                grammar_collector=collector,
                tool_provenance_audit=tool_provenance_audit,
                step_char_sum=step_char_sum,
                step_char_max=step_char_max,
                tool_failure_streak_max=tool_failure_streak_max,
                reasoning_output_tokens=reasoning_output_tokens,
                context_gathering_step_count=context_gathering_step_count,
                execution_step_count=execution_step_count,
                fcc_served_model=fcc_served_model,
                serving_role=serving_role,
                fcc_route=fcc_route,
                fcc_backend=fcc_backend,
                reading_receipts=reading_receipts,
                source_fetches=reading_tracker.source_fetches(),
            )

        await _publish_motor_lifecycle(
            status="success", final_text_present=False, compliance_verdict=compliance_verdict
        )

        # Cross-turn continuity within this same session: only on a turn that
        # completed cleanly via the "final" event, not one that crashed/timed
        # out after using a fetch tool (error_path_taken) even if a partial
        # draft was salvaged -- a fetch tied to a turn the user may not have
        # fully seen isn't worth surfacing to the next one. (The `if not
        # draft_text` branch above already excludes the fully-empty-draft
        # case; error_path_taken additionally excludes the partial-draft
        # error case, which reaches this point with non-empty draft_text.)
        fetch_tool_names = fetch_shaped_tool_names(receipts) if not error_path_taken else []
        if fetch_tool_names:
            await publish_last_tool_fetch(
                self.bus,
                session_id=thought.session_id,
                correlation_id=request.correlation_id,
                tool_names=fetch_tool_names,
            )

        molecule = build_draft_molecule(
            correlation_id=request.correlation_id,
            thought=thought,
            draft_text=draft_text,
            grammar_receipts=receipts,
            coalition_snapshot=coalition,
            repair_overlay=overlay,
            tool_provenance_audit=tool_provenance_audit,
        )
        logger.info(
            "harness_motor_complete corr=%s steps=%s grammar_receipts=%s verdict=%s grounding=%s draft_len=%s",
            request.correlation_id,
            step_count,
            len(receipts),
            compliance_verdict,
            grounding_status,
            len(draft_text),
        )
        return HarnessMotorResult(
            draft_text=draft_text,
            fcc_elapsed_sec=round(time.monotonic() - fcc_started, 3),
            grammar_receipts=receipts,
            step_count=step_count,
            exit_code=exit_code,
            compliance_verdict=compliance_verdict,
            grounding_status=grounding_status,
            cut_short_reason=cut_short_reason,
            draft_molecule=molecule,
            grammar_collector=collector,
            tool_provenance_audit=tool_provenance_audit,
            step_char_sum=step_char_sum,
            step_char_max=step_char_max,
            tool_failure_streak_max=tool_failure_streak_max,
            reasoning_output_tokens=reasoning_output_tokens,
            context_gathering_step_count=context_gathering_step_count,
            execution_step_count=execution_step_count,
            fcc_served_model=fcc_served_model,
            serving_role=serving_role,
            fcc_route=fcc_route,
            fcc_backend=fcc_backend,
            reading_receipts=reading_receipts,
            source_fetches=reading_tracker.source_fetches(),
        )
