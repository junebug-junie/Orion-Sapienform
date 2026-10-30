"""Caller-bound introspection tools over the existing internal Orion bus trust boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.transport import (
    CURIOSITY_REQUEST_CHANNEL,
    DREAM_REQUEST_CHANNEL,
    NODE_HISTORY_REQUEST_CHANNEL,
    ORION_DAY_REQUEST_CHANNEL,
    POWER_ATTRIBUTION_REQUEST_CHANNEL,
    QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
    RECORD_CORRECTION_REQUEST_CHANNEL,
    REQUEST_KIND,
    RESULT_KIND,
    RESULT_PREFIX,
)
from orion.schemas.introspect import (
    CuriosityArguments,
    DreamsArguments,
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    NodeHistoryArguments,
    OrionDayArguments,
    PowerAttributionArguments,
    QueueCuriosityFocusArguments,
    ReadingResultArguments,
    RecordCorrectionArguments,
)
from orion.schemas.reading import ReadingToolRequestV1, ReadingToolResultV1
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.urls import normalize_reading_source

RPC_TIMEOUT_SEC = 15.0

READING_RESULTS_DESCRIPTION = (
    "Search what you actually learned from sources read through your reading pipeline. "
    "Pass query=<what you want to recall, in plain words> to find readings by meaning; each "
    "item carries similarity (0-1) and only matches above a relevance floor come back. Or pass "
    "url or request_id for one reading, or nothing for your most recent finished reads. "
    "since=<ISO timestamp with timezone> narrows query or recent mode; limit up to 5. Each "
    "item's text is the learned summary; learned=false means no output exists yet, so report "
    "its reading_status instead. Results are source-attributed candidates, not settled beliefs. "
    "items=[] means nothing matched; a tool error means the answer is unknown, never that "
    "nothing happened."
)

DREAMS_DESCRIPTION = (
    "Read back your own dreams. Two kinds come back, labeled: dream_narrative (the nightly "
    "dream story: tldr, themes, narrative) and dream_hypothesis (a speculative link a sleep "
    "cycle proposed between two memories, shown to you once during curiosity). Pass "
    "query=<what you want to recall, in plain words> to find dreams by meaning (items carry "
    "similarity 0-1; weak matches are dropped). Every record here is already a dream, so put "
    "only the topic in query -- 'pull requests', not 'a dream about pull requests'. Or pass "
    "dream_id=<id> for one dream in full, or "
    "nothing for your most recent (mostly hypotheses, which are more frequent). "
    "kind=narrative returns only the nightly dream narratives; kind=hypothesis only the "
    "sleep-cycle hypotheses already offered to you. since=<ISO timestamp "
    "with timezone> narrows query or recent mode; limit up to 5. Dreams are experiences you "
    "had, not facts about the world (epistemic_status=unsettled). items=[] means no dream "
    "matched; a tool error means the answer is unknown, never that you did not dream."
)

CURIOSITY_DESCRIPTION = (
    "Read back your own curiosity runs: the sittings where you chose something to look into "
    "(line investigate = world question, self_inquiry = self question, self_sense_eval = "
    "self-sense check). A run with a write-up comes back unsettled: what you concluded then, "
    "not established fact; lists show its Answer section or opening, run_id=<id> returns the "
    "write-up in full (up to ~9000 characters). A run that failed, was cancelled, or wrote "
    "nothing comes back as a short record of what happened -- failures are part of your "
    "history. extra carries status, error, hops, findings, revisions, the prior touched, "
    "reach_out, and belief-move counts (n_tested, n_moved, n_formed) when they were recorded; "
    "graph_read=false means the hop counts are unknown, not zero. Pass query=<topic in plain "
    "words> to find runs by meaning (items carry similarity 0-1), run_id for one run, or "
    "nothing for your most recent finished runs; line and since=<ISO timestamp with timezone, "
    "within the last 90 days> narrow query or recent mode; limit up to 5. kind=self_question lists your open "
    "self-questions instead. kind=focus lists the focuses you queued with queue_curiosity_focus, "
    "newest first, each with its story: queued -> taken by run X -> disposition (explored, set "
    "aside and why, or unknown), or expired / replaced / cleared / failed; a run that took a "
    "focus carries extra.focus. items=[] means nothing matched; a tool error means the answer is "
    "unknown, never that no run happened."
)


ORION_DAY_DESCRIPTION = (
    "Reread your own Orion's Day letters (the nightly letter to Juniper) together with the "
    "records each was written from. The text is what you wrote then, not established fact "
    "(epistemic_status=unsettled). Parts are numbered the way Juniper's email numbers them: "
    "ids like '2026-10-09 ¶3' (a note paragraph) and '2026-10-09 carry 5' (a carry-forward "
    "item) are the same numbers Juniper sees. part=list (default) gives the outline: each "
    "paragraph and carry item by number with its opening words, and how many records each "
    "day section holds (limit does not apply to it). part=note or part=carry_forward with index=<N> returns that part's "
    "exact words; without index, the first parts up to limit (5). A note paragraph comes "
    "with extra.claim_check: each concrete token in it (numbers, timestamps, PR numbers, "
    "ids, quotes) with found_in = the records of that day containing it verbatim. This is "
    "string evidence only, not a verdict: an empty found_in means not in that day's records "
    "verbatim (it may be derived, from another day, or wrong), and a match is not proof the "
    "sentence around it is right. A carry item comes with extra.citations: each [ref] it "
    "cites resolved against that day's records with an excerpt, or resolved=false when that "
    "day holds no such record. part=section with section=curiosity|self_sense|readings|"
    "dreams|code_changes|conversations|world_news|reveries returns that section's records. "
    "letter_date=<YYYY-MM-DD> picks a letter; omitted means the most recent. query=<plain "
    "words> finds paragraphs and carry items by meaning across letters (items carry "
    "similarity 0-1; letter_date or part=note|carry_forward narrow it). A letter that exists "
    "with an empty section gives items=[]; a letter, paragraph or item that does not exist "
    "is a tool error saying not found; any other tool error means the answer is unknown, "
    "never that you wrote nothing."
)


RECORD_CORRECTION_DESCRIPTION = (
    "Record a correction to something you wrote in an Orion's Day letter, once the conversation "
    "with Juniper has established that the claim was wrong and you actually accept that. It "
    "edits nothing: not the letter, and not any belief. It saves the correction as evidence; a "
    "later curiosity run is shown it once, with the priors you name and their current "
    "confidence, and asked to re-test them. That run decides how far, if at all, a prior moves, "
    "and whether it re-tested each one is recorded on the correction. "
    "ref is the part's id exactly as orion_day prints it ('2026-10-09 ¶3' or '2026-10-09 carry 5') "
    "and must name a part that exists. claim: the words you are correcting, quoted from that "
    "part. correction: what is actually true, in plain words. evidence: what the correction "
    "rests on, one item each: record refs from orion_day's claim check or citations (e.g. "
    "'curiosity:e7d03d2ecdbb'), or 'Juniper said so in this conversation'. prior_ids: the exact "
    "prior_id of each :Prior in your graph that the claim rested on (zero or more; every one "
    "must exist; an id containing spaces or quotes cannot be named). Only from a conversation "
    "with Juniper. A tool error saying 'not recorded' means nothing was written (it says why); "
    "'answer unknown' means it may or may not have been (calling again with the same words is "
    "safe: it returns the same correction_id). Never report a correction as recorded without "
    "ok=true and a correction_id."
)

QUEUE_CURIOSITY_FOCUS_DESCRIPTION = (
    "Queue one question for your next regular curiosity run (the investigate line) to look at, "
    "when you have chosen what you want to focus on, or Juniper asked you to look into something "
    "next. question: the question, in plain words. why: why you want it now. origin: 'self' "
    "(your own choice, the default) or 'juniper' (Juniper named it). There is one slot: a new "
    "focus replaces the pending one, and the result names what it replaced; your own choice "
    "cannot replace a focus Juniper asked for (that is a tool error saying not queued). It "
    "starts nothing now: the run still waits for its normal schedule, cooldown and daily cap, "
    "and the result says when the next run can start. A pending focus expires after 48 hours. "
    "The run is shown the focus first and decides itself whether to explore it or set it "
    "aside; that disposition is recorded, and curiosity(kind=focus) reads it back. Only from a "
    "conversation with Juniper. A tool error saying 'not queued' means nothing was written "
    "(it says why); 'answer unknown' means it may or may not have been (calling again with the "
    "same words is safe: it returns the same focus_id). Never say a focus was queued without "
    "ok=true and a focus_id."
)

NODE_HISTORY_DESCRIPTION = (
    "Read the history of one of your substrate nodes' prediction error (how surprised that "
    "part of you was, reading by reading). The node in your graph holds only its current "
    "value; the series lives here, up to 7 days back. Pass node_id=<exact id, e.g. "
    "node:substrate.bus_synaptic> and hours=<window length, up to 168, default 24>; "
    "until=<ISO timestamp with timezone> ends the window there instead of now (the whole "
    "window must start within the last 168 hours). You get count, min, median, max, the "
    "first and last reading, a series of at most 48 time buckets (bucket_start, n, mean, "
    "min, max), the largest changes between consecutive readings inside the window with when "
    "they happened, and earliest_stored_at (nothing older exists). A node "
    "only records a reading when it is refreshed, so a bucket missing from the series means "
    "no new reading then, not zero. Pass no node_id to list which nodes have readings in the "
    "window. items=[] means that node has no reading in that window; a node_id with no stored "
    "reading at all is a tool error saying not found; any other tool error means the answer "
    "is unknown, never that the node was quiet."
)


POWER_ATTRIBUTION_DESCRIPTION = (
    "Heat archaeology: where your machines' power (and so the cabinet's heat) went, and since "
    "when. For each machine (circe, athena, hecate, and cooling = the cabinet AC plug) over the "
    "last days=<2-30, default 14> days (until=<ISO timestamp with timezone> ends the window "
    "earlier; node=<name> picks one): the steps in hourly wall watts (when, before/after W), "
    "each split into GPU, CPU and rest watts; a least-squares fit of hourly wall watts on "
    "GPU-pool workload busy-hours (harness = http:anthropic, ai-town = http:openai, cortex-exec, "
    "vision-council, ...) plus a CPU-load regime, with R2 and each term's watts per busy-hour "
    "and standard error; each step's watts attributed to the workloads that changed with it "
    "(unexplained = what the fit cannot place); CPU-load shifts, which GPU leases cannot see "
    "(point at host processes and container restarts), with the same split across each shift; "
    "and durable-run holds, reported as "
    "reservations, never load. Each item's text is the plain-English answer; extra holds the "
    "numbers and the evidence for each claim, and a last item holds the exact SQL to re-run. "
    "Everything is a correlation (epistemic_status=unsettled, verified=false): say UNVERIFIED "
    "until stopping the suspect drops the watts. items=[] means no machine had a wall-power "
    "reading in the window; a tool error means the answer is unknown, never that nothing changed."
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]


class IntrospectUnknownError(RuntimeError):
    """The owning service could not be asked, or did not answer coherently."""


class IntrospectTools:
    def __init__(self, bus, binding: IntrospectToolBindingV1):
        self.bus = bus
        self.binding = binding

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments),
            ToolSpec("dreams", DREAMS_DESCRIPTION, DreamsArguments),
            ToolSpec("curiosity", CURIOSITY_DESCRIPTION, CuriosityArguments),
            ToolSpec("orion_day", ORION_DAY_DESCRIPTION, OrionDayArguments),
            ToolSpec("record_correction", RECORD_CORRECTION_DESCRIPTION, RecordCorrectionArguments),
            ToolSpec("queue_curiosity_focus", QUEUE_CURIOSITY_FOCUS_DESCRIPTION, QueueCuriosityFocusArguments),
            ToolSpec("node_history", NODE_HISTORY_DESCRIPTION, NodeHistoryArguments),
            ToolSpec("power_attribution", POWER_ATTRIBUTION_DESCRIPTION, PowerAttributionArguments),
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Validate before transport: extra fields are rejected, never ignored.
        if name == "dreams":
            dream_args = DreamsArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                DREAM_REQUEST_CHANNEL, "dreams", dream_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name == "curiosity":
            curiosity_args = CuriosityArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                CURIOSITY_REQUEST_CHANNEL, "curiosity",
                curiosity_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name == "orion_day":
            day_args = OrionDayArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                ORION_DAY_REQUEST_CHANNEL, "orion_day", day_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name == "node_history":
            history_args = NodeHistoryArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                NODE_HISTORY_REQUEST_CHANNEL, "node_history",
                history_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name == "power_attribution":
            power_args = PowerAttributionArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                POWER_ATTRIBUTION_REQUEST_CHANNEL, "power_attribution",
                power_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name == "record_correction":
            correction_args = RecordCorrectionArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                RECORD_CORRECTION_REQUEST_CHANNEL, "record_correction",
                correction_args.model_dump(mode="json"),
                refused="not recorded",
            )
            return result.model_dump(mode="json")
        if name == "queue_curiosity_focus":
            focus_args = QueueCuriosityFocusArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL, "queue_curiosity_focus",
                # Never refs from a chat turn: nothing on this path opens them first (orion/refs.py rule).
                focus_args.model_dump(mode="json", exclude={"refs"}),
                refused="not queued",
            )
            return result.model_dump(mode="json")
        if name != "reading_results":
            raise ValueError(f"unknown introspect tool: {name}")
        args = ReadingResultArguments.model_validate(arguments)
        command = ReadingToolRequestV1(
            operation="reading_result",
            request_id=args.request_id,
            url=normalize_reading_source(args.url) if args.url is not None else None,
            query=args.query,
            limit=args.limit,
            since=args.since,
        )
        return (await self._reading_rpc(command)).model_dump(mode="json")

    async def _introspect_rpc(
        self, channel: str, operation: str, args: dict[str, Any], *, refused: str = "answer unknown",
    ) -> IntrospectResultV1:
        """``refused`` words an ok=false answer. A write tool's responder answers ok=false only
        before it writes, so that is "not recorded", not "unknown"; no reply stays unknown."""
        correlation_id = uuid4()
        reply = f"{RESULT_PREFIX}{correlation_id}"
        request = IntrospectRequestV1(operation=operation, binding=self.binding, args=args)
        try:
            raw = await self.bus.rpc_request(
                channel,
                BaseEnvelope(
                    kind=REQUEST_KIND, correlation_id=correlation_id, reply_to=reply,
                    source=ServiceRef(name="orion-harness-governor"),
                    payload=request.model_dump(mode="json"),
                ),
                reply_channel=reply, timeout_sec=RPC_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise IntrospectUnknownError(
                f"{operation}: answer unknown (no reply: {type(exc).__name__})"
            ) from exc
        decoded = self.bus.codec.decode(raw.get("data"))
        if (
            not decoded.ok
            or str(decoded.envelope.correlation_id) != str(correlation_id)
            or decoded.envelope.kind != RESULT_KIND
        ):
            raise IntrospectUnknownError(f"{operation}: answer unknown (malformed reply)")
        try:
            result = IntrospectResultV1.model_validate(decoded.envelope.payload)
        except ValueError as exc:
            raise IntrospectUnknownError(f"{operation}: answer unknown (malformed result)") from exc
        if not result.ok:
            label = "answer unknown" if "answer unknown" in (result.error or "") else refused
            raise IntrospectUnknownError(f"{operation}: {label} ({result.error})")
        if result.operation != operation:
            raise IntrospectUnknownError(f"{operation}: answer unknown (mismatched result)")
        return result

    async def _reading_rpc(self, command: ReadingToolRequestV1) -> IntrospectResultV1:
        correlation_id = uuid4()
        reply = f"{TOOL_RESULT_PREFIX}{correlation_id}"
        try:
            raw = await self.bus.rpc_request(
                TOOL_CHANNEL,
                BaseEnvelope(
                    kind="reading.tool.request.v1", correlation_id=correlation_id,
                    reply_to=reply, source=ServiceRef(name="orion-harness-governor"),
                    payload=command.model_dump(mode="json"),
                ),
                reply_channel=reply, timeout_sec=RPC_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise IntrospectUnknownError(
                f"reading_results: answer unknown (no reply: {type(exc).__name__})"
            ) from exc
        decoded = self.bus.codec.decode(raw.get("data"))
        if not decoded.ok or str(decoded.envelope.correlation_id) != str(correlation_id):
            raise IntrospectUnknownError("reading_results: answer unknown (malformed reply)")
        try:
            envelope_result = ReadingToolResultV1.model_validate(decoded.envelope.payload)
        except ValueError as exc:
            raise IntrospectUnknownError("reading_results: answer unknown (malformed reply)") from exc
        if not envelope_result.ok:
            raise IntrospectUnknownError(f"reading_results: answer unknown ({envelope_result.error})")
        try:
            result = IntrospectResultV1.model_validate(envelope_result.result)
        except ValueError as exc:
            raise IntrospectUnknownError("reading_results: answer unknown (malformed result)") from exc
        if not result.ok or result.operation != "reading_result":
            raise IntrospectUnknownError("reading_results: answer unknown (mismatched result)")
        return result
