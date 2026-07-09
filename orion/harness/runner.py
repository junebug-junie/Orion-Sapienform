from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable

from orion.harness.fcc_motor import (
    _extract_tool_name,
    expand_env_path,
    load_fcc_env,
    resolve_auth_token,
    run_fcc_turn,
    summarize_harness_step,
)
from orion.harness.grammar_emit import (
    HarnessGrammarCollector,
    build_harness_grammar_events,
    publish_harness_lifecycle_grammar,
    short_error_kind,
)
from orion.harness.grammar_publish import publish_harness_step_grammar
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.repair import map_repair_pressure_contract
from orion.harness.step_stream import publish_harness_run_step
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import (
    GrammarReceiptV1,
    HarnessDraftMoleculeV1,
    HarnessRepairOverlayV1,
    HarnessRunRequestV1,
)
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
    draft_molecule: HarnessDraftMoleculeV1 | None = None
    grammar_collector: HarnessGrammarCollector | None = None


def _default_harness_node_name() -> str:
    return os.environ.get("HARNESS_NODE_NAME", "athena")


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


def _draft_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def build_harness_prompt(
    *,
    thought: ThoughtEventV1,
    user_message: str,
    repair_overlay: HarnessRepairOverlayV1,
    answer_contract: AnswerContract | None = None,
    workspace: str | None = None,
) -> str:
    prefix = compile_harness_prefix(
        thought,
        repair_overlay=repair_overlay,
        user_message=user_message,
        answer_contract=answer_contract,
        workspace=workspace or os.environ.get("HARNESS_FCC_WORKSPACE"),
    )
    instruction = harness_motor_instruction(
        thought=thought,
        answer_contract=answer_contract,
    )
    if user_message.strip():
        return f"{prefix}\n\n{instruction}"
    return prefix


def build_draft_molecule(
    *,
    correlation_id: str,
    thought: ThoughtEventV1,
    draft_text: str,
    grammar_receipts: list[GrammarReceiptV1],
    coalition_snapshot: CoalitionSnapshotV1,
    repair_overlay: HarnessRepairOverlayV1,
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
    )


async def default_fcc_runner(
    *,
    prompt: str,
    correlation_id: str,
    fcc_model_label: str | None = None,
    timeout_sec: float = 120.0,
    **_: Any,
) -> AsyncIterator[dict[str, Any]]:
    env_path = expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env"))
    env = load_fcc_env(env_path)
    token = resolve_auth_token(env, override=os.environ.get("HARNESS_FCC_AUTH_TOKEN", ""))
    async for event in run_fcc_turn(
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
    ) -> None:
        self.bus = bus
        self.grammar_channel = grammar_channel
        self.step_channel = step_channel
        self.fcc_runner = fcc_runner or default_fcc_runner
        self.fcc_timeout_sec = fcc_timeout_sec
        self.node_name = node_name or _default_harness_node_name()

    async def run(
        self,
        request: HarnessRunRequestV1,
        *,
        repair_overlay: HarnessRepairOverlayV1 | None = None,
        coalition_snapshot: CoalitionSnapshotV1 | None = None,
        publish_grammar_fn: Callable[..., Awaitable[None]] | None = None,
        recall_debug: dict[str, Any] | None = None,
    ) -> HarnessMotorResult:
        thought = request.thought_event
        overlay = repair_overlay or map_repair_pressure_contract(request.repair_pressure_contract)
        coalition = coalition_snapshot or build_coalition_snapshot(thought)
        prompt = build_harness_prompt(
            thought=thought,
            user_message=request.user_message,
            repair_overlay=overlay,
            answer_contract=request.answer_contract,
            workspace=os.environ.get("HARNESS_FCC_WORKSPACE"),
        )

        collector = HarnessGrammarCollector(
            node_name=self.node_name,
            correlation_id=request.correlation_id,
            observed_at=datetime.now(timezone.utc),
        )
        collector.record_request_received()
        collector.record_plan_started(step_count=0)
        _record_recall_gate_from_debug(collector, recall_debug)

        receipts: list[GrammarReceiptV1] = []
        step_count = 0
        draft_text = ""
        exit_code: int | None = None
        compliance_verdict = "completed"
        grounding_status = "grounded"
        motor_failed = False

        async for event in self.fcc_runner(
            prompt=prompt,
            correlation_id=request.correlation_id,
            fcc_model_label=request.fcc_model_label,
            timeout_sec=self.fcc_timeout_sec,
        ):
            etype = str(event.get("type") or "")
            if etype == "step":
                step = event.get("step")
                if not isinstance(step, dict):
                    continue
                summary = summarize_harness_step(step, index=step_count)
                collector.record_step_started(order=step_count + 1, summary=summary)
                tool_name = _extract_tool_name(step)
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
                meta = event.get("metadata")
                if isinstance(meta, dict):
                    raw_exit = meta.get("exit_code")
                    if isinstance(raw_exit, int):
                        exit_code = raw_exit
            elif etype == "error":
                partial = str(event.get("llm_response") or "").strip()
                error_code = str(event.get("error_code") or "").strip()
                error_msg = str(event.get("error") or "").strip()
                if partial:
                    draft_text = partial
                    compliance_verdict = "partial"
                    grounding_status = error_code or "partial"
                else:
                    compliance_verdict = "failed"
                    grounding_status = error_code or error_msg or "failed"
                    motor_failed = True
                if step_count > 0:
                    collector.record_step_failed(
                        order=step_count,
                        error_kind=short_error_kind(error_code or error_msg),
                    )
                logger.warning(
                    "fcc motor error corr=%s code=%s err=%s",
                    request.correlation_id,
                    event.get("error_code"),
                    event.get("error"),
                )
                break

        async def _publish_motor_lifecycle(*, status: str, final_text_present: bool) -> None:
            collector.record_result_assembled(
                status=status,
                final_text_present=final_text_present,
                step_count=step_count,
                grammar_receipt_count=len(receipts),
                reflection_ran=False,
                quick_lane_skipped_5b=True,
            )
            try:
                await publish_harness_lifecycle_grammar(
                    self.bus,
                    channel=self.grammar_channel,
                    events=build_harness_grammar_events(collector),
                )
            except Exception:
                logger.warning(
                    "harness_motor_lifecycle_grammar_publish_failed corr=%s",
                    request.correlation_id,
                    exc_info=True,
                )

        if not draft_text:
            await _publish_motor_lifecycle(status="failed", final_text_present=False)
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
                grammar_receipts=receipts,
                step_count=step_count,
                exit_code=exit_code,
                compliance_verdict=compliance_verdict if compliance_verdict != "completed" else "failed",
                grounding_status=grounding_status if grounding_status != "grounded" else "empty_draft",
                grammar_collector=collector,
            )

        await _publish_motor_lifecycle(status="success", final_text_present=False)

        molecule = build_draft_molecule(
            correlation_id=request.correlation_id,
            thought=thought,
            draft_text=draft_text,
            grammar_receipts=receipts,
            coalition_snapshot=coalition,
            repair_overlay=overlay,
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
            grammar_receipts=receipts,
            step_count=step_count,
            exit_code=exit_code,
            compliance_verdict=compliance_verdict,
            grounding_status=grounding_status,
            draft_molecule=molecule,
            grammar_collector=collector,
        )
