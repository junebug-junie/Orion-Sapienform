"""Patch 1 of `docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`.

Read-only, on history. Reads every `Hop` Orion has ever written to
`orion_worldview`, asks an LLM to read each one's prose and produce a
`HopReadingV1` -- which live-or-closed `Prior` (if any) the hop was about,
what kind of step it was, and whether it moved the claim -- and reports a
deterministic `is_circling` verdict per prior from those readings.

NOTHING HERE WRITES TO `orion_worldview`, PUBLISHES ON THE BUS AS AN EVENT, OR
CHANGES WHAT ORION DOES NEXT. The single bus interaction is an RPC call to the
cortex brain lane to generate readings -- the same shape
`orion/memory_graph/suggest_runner.py` uses for a standalone structured-output
call -- and its result is handed back to the caller, not persisted by this
module. Per the spec's own "Recommended next patch": no interventions, no
live subscription, no `Hop -> Prior` write-side link.

`Hop` carries no timestamp (spec's own Missing Question 2), so "chronological"
below means the best the graph can give: each run ordered by its
`TurnOutcome.written_at` where one exists, hops within a run ordered by their
own `n`. A run with no `TurnOutcome` (died before writing one, or the read
failed) sorts last, same "unknown is not oldest" rule `worldview.build_recent_
runs` already applies -- not a claim that it actually happened last.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional, Sequence

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory_graph.json_extract import extract_first_json_object_text
from orion.curiosity.worldview import (
    HopRecord,
    Prior,
    RECENT_RUNS_CYPHER,
    RECENT_RUNS_LIMIT,
    WorldviewReader,
    WorldviewUnavailable,
    build_recent_runs,
    read_all_hops,
    read_all_priors,
)
from orion.schemas.curiosity_supervisor import HopReadingBatchV1, HopReadingV1
from orion.schemas.cortex.contracts import (
    CortexClientContext,
    CortexClientRequest,
    LLMMessage,
    RecallDirective,
)

logger = logging.getLogger("orion.curiosity.supervisor")

# ROOT-CAUSED (was mis-blamed on max_tokens in an earlier version of this
# file). "metacog" is Qwen3-8B on a **4096-token total context window** --
# confirmed live against the real server: `curl :8012/slots` returns
# `"n_ctx": 4096`, and that figure covers PROMPT + COMPLETION together, not
# completion alone. A multi-hop batch's prompt (priors list + several hop
# notes) can eat most of that 4096 before the model writes a single
# character of its answer, so raising `max_tokens` past what's left cannot
# help -- the model hits the hard context wall mid-string and stops, which
# is exactly the "Unterminated string..." truncation live verification hit
# on 3/40 runs. `chat` (Qwen3.6-35B-A3B, port 8011) runs at 131072 tokens of
# context -- confirmed live the same way -- and per
# reference_agent_lane_27b_vs_chat_lane_35b_speed.md is also the FASTER lane
# per token despite being nominally bigger (MoE: ~3B active params/token).
# Only 1 concurrent slot vs metacog's 4, which does not matter here --
# generate_all_readings awaits one run's RPC before starting the next, so
# nothing this module does is concurrent on the lane anyway.
DEFAULT_LLM_ROUTE = "chat"
DEFAULT_TIMEOUT_SEC = 180.0
DEFAULT_MAX_TOKENS = 6000
_VERB = "curiosity_hop_reading"


def group_hops_by_run(
    hops: Sequence[HopRecord], *, run_order: dict[str, int]
) -> list[tuple[str, list[HopRecord]]]:
    """(run_id, hops) pairs, most-recent-run-last -- so a per-prior "last N
    readings" read (see `is_circling`) reads as "most recent" in the flattened
    output without needing its own sort.

    `run_order` is a `run_id -> written_at` map (missing/`None` sorts as
    oldest -- see this module's docstring); hops within a run sort by `n`.
    """
    by_run: dict[str, list[HopRecord]] = {}
    for hop in hops:
        by_run.setdefault(hop.run_id, []).append(hop)
    for run_hops in by_run.values():
        run_hops.sort(key=lambda h: h.n)
    ordered_run_ids = sorted(
        by_run.keys(), key=lambda rid: (run_order.get(rid) is None, run_order.get(rid) or 0)
    )
    return [(run_id, by_run[run_id]) for run_id in ordered_run_ids]


def build_run_order(reader: WorldviewReader) -> dict[str, int]:
    """`run_id -> written_at` (ms), from the same read `kickoff_prompt` uses
    to show Orion its own recent thread. `{}` on failure -- callers treat a
    missing entry as "unknown", not as an error.

    Repurposes a query sized for "recent", not "all of history" --
    `RECENT_RUNS_CYPHER` caps at `RECENT_RUNS_LIMIT` rows. A run past that
    bound silently sorts as "unknown -> last" in `group_hops_by_run` rather
    than erroring, so the only place this can go wrong quietly is here: the
    warning below is what makes a future population big enough to hit the
    cap visible, the same discipline `read_all_hops`/`read_all_priors` apply
    to their own bounds. Caught in review.
    """
    try:
        rows = reader.query(RECENT_RUNS_CYPHER)
    except WorldviewUnavailable as exc:
        logger.warning("curiosity_supervisor_run_order_read_failed err=%s", exc)
        return {}
    if len(rows) >= RECENT_RUNS_LIMIT:
        logger.warning(
            "curiosity_supervisor_run_order_truncated limit=%s -- runs past "
            "this bound sort as unknown-timestamp (last) in group_hops_by_run, "
            "not as an error",
            RECENT_RUNS_LIMIT,
        )
    recent = build_recent_runs(rows, limit=len(rows) or 1)
    return {r.run_id: r.written_at for r in recent if r.written_at is not None}


def build_reading_prompt(priors: Sequence[Prior], hops: Sequence[HopRecord]) -> str:
    """The prompt for one run's worth of hops.

    Every prior is listed, live and closed both -- see `read_all_priors` for
    why closed ones must be included. Priors are listed with the SAME
    `prior_id` Orion's own kickoff prompt uses, so a reading's
    `about_prior_id` can be matched back to a real node without inventing a
    lookup the graph does not offer.
    """
    lines = [
        "You are reading Orion's own investigation notes -- NOT writing new "
        "ones. For each hop below, decide which prior (if any) it was about, "
        "what kind of step it was, and whether it moved that claim.",
        "",
        "Every prior Orion currently holds (live and closed):",
    ]
    if not priors:
        lines.append("  (none recorded)")
    for p in priors:
        conf = f"{p.confidence:.2f}" if p.confidence is not None else "none"
        lines.append(
            f"  - prior_id: {p.prior_id}\n"
            f"    claim: {p.claim}\n"
            f"    status: {p.status}, confidence: {conf}, times_tested: {p.times_tested}"
        )
    lines += ["", "Hops from this run, in order:"]
    for h in hops:
        lines.append(f"  - n={h.n}: {h.note}")
    lines += [
        "",
        "For EVERY hop listed above, return one reading. `about_prior_id` "
        "must be an exact prior_id from the list above, or null if the hop "
        "is not about any of them -- never invent one. `moved_the_claim` is "
        "true if the hop changed the claim's confidence or status, false if "
        "it tested the claim and left it where it was, or null if you "
        "cannot tell. `kind` is a short free-text label for the kind of "
        "step (e.g. 'test', 'revise', 'dead_end', 'bookkeeping') -- your own "
        "words, not a fixed list.",
    ]
    return "\n".join(lines)


def build_reading_options(
    *,
    llm_route: str = DEFAULT_LLM_ROUTE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Mirrors `orion/memory_graph/suggest_runner.py:
    build_memory_graph_suggest_options` -- same structured-output shape, a
    different schema and verb."""
    return {
        "llm_route": llm_route,
        "no_write": True,
        "skip_brain_reply_context": True,
        "skip_unified_beliefs": True,
        "skip_autonomy_context": True,
        "skip_chat_stance_inputs": True,
        "structured_output_schema_name": "HopReadingBatchV1",
        "structured_output_schema": HopReadingBatchV1.model_json_schema(),
        "structured_output_method": "json_object_schema",
        "structured_output_thinking_policy": "disabled_for_artifact",
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }


def _extract_cortex_result_text(payload: Any) -> str:
    """The model's own text out of a CortexClientResult-shaped payload.

    Same field-priority convention `orion/memory_graph/cortex_suggest_
    extract.py:extract_suggest_text_from_cortex_payload` uses for the same
    problem one domain over -- `final_text` is where brain-mode structured
    output actually lands. Falls back to the last step's own
    `LLMGatewayService` content when `final_text` is empty (a timed-out or
    partially-failed turn can still carry a usable draft in `steps`, the same
    "partial draft as an ordinary final frame" case `journal.py` documents
    for curiosity investigation runs).

    A DELIBERATELY THINNER COPY, not a reuse of that module's version.
    `extract_suggest_text_from_cortex_payload` is more thorough (checks a
    `detail` container, `reasoning_content`/`output` fallbacks, per-candidate
    JSON-extraction scoring) but is memory-graph-coupled (its first branch
    checks a `"draft"` key specific to that domain) and its real search logic
    is private to that module. Importing it wholesale for a narrower need
    seemed like the wrong trade; noted in review as a real, if minor,
    duplication rather than something worth re-deciding silently.
    """
    if not isinstance(payload, dict):
        return ""
    for field in ("final_text", "text", "content"):
        val = payload.get(field)
        if isinstance(val, str) and val.strip():
            return val
    for step in reversed(payload.get("steps") or []):
        result = (step or {}).get("result") or {}
        for block in result.values():
            content = (block or {}).get("content") if isinstance(block, dict) else None
            if isinstance(content, str) and content.strip():
                return content
    return ""


def parse_reading_batch(
    payload: Any, *, run_id: str, hop_ns: Sequence[int]
) -> list[HopReadingV1]:
    """Validate the LLM's response into `HopReadingV1` rows.

    Tolerant like the rest of this arc's readers: a row that fails to
    validate is dropped and logged rather than raised, so one malformed
    reading in a batch of six does not cost the other five.

    `hop_run_id` is ALWAYS overwritten with the caller's own `run_id`, never
    trusted from the model -- confirmed live: nothing in the prompt asks for
    it (the caller already knows it), yet the JSON schema requires the field,
    and the model filled it with a hallucinated placeholder (`"n=1"`) rather
    than leaving it out. `hop_n` is trusted (the model DOES need to say which
    hop it's reading) but checked against `hop_ns`: one the model returns
    that was NOT one of the hops we asked about is dropped as unusable, the
    same "cannot be presented back honestly" rule `build_prior` applies.
    """
    if isinstance(payload, str):
        import json

        try:
            payload = json.loads(payload)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "curiosity_supervisor_reading_unparseable run=%s err=%s", run_id, exc
            )
            return []
    if not isinstance(payload, dict):
        logger.warning(
            "curiosity_supervisor_reading_wrong_shape run=%s type=%s",
            run_id,
            type(payload).__name__,
        )
        return []
    raw_readings = payload.get("readings")
    if not isinstance(raw_readings, list):
        logger.warning("curiosity_supervisor_reading_no_list run=%s", run_id)
        return []

    known_ns = set(hop_ns)
    out: list[HopReadingV1] = []
    dropped = 0
    for raw in raw_readings:
        if not isinstance(raw, dict):
            dropped += 1
            continue
        raw = dict(raw)
        raw["hop_run_id"] = run_id  # always ours, never the model's -- see docstring
        try:
            reading = HopReadingV1.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 -- a schema-drifted row must not raise
            logger.warning(
                "curiosity_supervisor_reading_invalid run=%s raw=%r err=%s",
                run_id, raw, exc,
            )
            dropped += 1
            continue
        if reading.hop_n not in known_ns:
            logger.warning(
                "curiosity_supervisor_reading_unknown_hop run=%s hop_n=%s -- "
                "not one of the hops asked about; dropped",
                run_id, reading.hop_n,
            )
            dropped += 1
            continue
        out.append(reading)
    if dropped:
        logger.warning(
            "curiosity_supervisor_reading_dropped run=%s dropped=%s of %s",
            run_id, dropped, len(raw_readings),
        )
    # Sorted by hop_n, NOT left in whatever order the model's JSON array
    # came back in. `is_circling` reads `readings_for_one_prior[-min_hops:]`
    # as "the most recent N" -- that claim is only true if a run's readings
    # are in hop order, and nothing guarantees an LLM echoes a multi-hop
    # batch back in the order it read them. Caught in review.
    out.sort(key=lambda r: r.hop_n)
    return out


async def generate_readings_for_run(
    bus: Any,
    *,
    run_id: str,
    hops: Sequence[HopRecord],
    priors: Sequence[Prior],
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    llm_route: str = DEFAULT_LLM_ROUTE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_attempts: int = 3,
    retry_delay_sec: float = 3.0,
) -> list[HopReadingV1]:
    """One cortex RPC call, one run's worth of hops -- see this module's
    docstring for why batching per run beats one call per hop.

    Same envelope/RPC/decode shape as `orion/memory_graph/suggest_runner.py:
    suggest_once` and `scripts/dream_spine_smoke.py:_rpc_orch`: a
    `cortex.orch.request` envelope over `bus.rpc_request`, decoded with
    `bus.codec.decode`. `mode="brain"` DOES gate `verb` against
    `orion/cognition/verb_activation.py`'s discovered-and-active set --
    confirmed live the hard way (Patch 1's own live verification run) after
    an earlier version of this docstring claimed otherwise. A new verb (this
    module's `curiosity_hop_reading.yaml` + matching `.j2` prompt, mirroring
    `memory_graph_suggest`'s minimal single-step shape) had to be registered
    and `orion-cortex-orch` rebuilt before any call got past
    `_normalize_and_validate_verb` in that service's `app/main.py`.

    Retries up to `max_attempts` on ANY failure -- RPC/decode error, or a
    `CortexClientResult` with `ok: false` (verb-gate rejection, generation
    failure, anything). This is not optimism: the SAME request, byte-for-byte,
    was observed live to fail this gate roughly half the time and succeed on
    a bare retry seconds later, for reasons internal to `orion-cortex-orch`
    this patch does not touch or explain. It is the exact shape of failure the
    spec's own "Current architecture" section already documents from the bus
    transition log -- roughly 3 failed/resumed `harness_turn` transitions per
    curiosity run -- observed here first-hand from the caller's side instead
    of read off the transition log after the fact.
    """
    if not hops:
        return []
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    hop_ns = [h.n for h in hops]
    prompt = build_reading_prompt(priors, hops)
    last_err: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        trace_id = str(uuid.uuid4())
        reply_channel = f"{cortex_result_prefix}:{trace_id}"
        ctx = CortexClientContext(
            messages=[LLMMessage(role="user", content=prompt)],
            # Set explicitly, not left to be derived from `messages`: the
            # verb's own template (orion/cognition/prompts/curiosity_hop_
            # reading_prompt.j2) renders `{{ user_message }}` directly --
            # confirmed live against `memory_graph_suggest_prompt.j2`, the
            # same mechanism.
            user_message=prompt,
            trace_id=trace_id,
            metadata={"curiosity_supervisor_run_id": run_id},
        )
        cortex_req = CortexClientRequest(
            mode="brain",
            route_intent="none",
            verb=_VERB,
            packs=[],
            options=build_reading_options(llm_route=llm_route, max_tokens=max_tokens),
            recall=RecallDirective(enabled=False),
            context=ctx,
        )
        envelope = BaseEnvelope(
            kind="cortex.orch.request",
            source=source,
            correlation_id=trace_id,
            reply_to=reply_channel,
            payload=cortex_req.model_dump(mode="json"),
        )
        try:
            msg = await bus.rpc_request(
                cortex_request_channel,
                envelope,
                reply_channel=reply_channel,
                timeout_sec=timeout_sec,
            )
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                raise RuntimeError(f"cortex RPC decode failed: {decoded.error}")
            payload = decoded.envelope.payload
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump(mode="json")
            if isinstance(payload, dict) and payload.get("ok") is False:
                # A real CortexClientResult failure, NOT a malformed-JSON
                # reading -- surfaced distinctly so it retries (and logs as
                # what it is) instead of silently falling into
                # parse_reading_batch's "no readings list" path, which would
                # misreport a cortex-side failure as an empty response.
                raise RuntimeError(
                    f"cortex result not ok: {(payload.get('error') or {}).get('message')}"
                )
        except Exception as exc:  # noqa: BLE001 -- retried below, or re-raised on last attempt
            last_err = str(exc)
            if attempt >= max_attempts:
                raise
            logger.warning(
                "curiosity_supervisor_rpc_retry run=%s attempt=%s/%s err=%s",
                run_id, attempt, max_attempts, last_err,
            )
            await asyncio.sleep(retry_delay_sec)
            continue
        # `payload` is a CortexClientResult -- ok/mode/verb/status/final_text/
        # steps/... -- NOT the structured-output dict itself. Confirmed live
        # (Patch 1's own verification run): the model's JSON lands in
        # `final_text` as a STRING. Same field-priority convention
        # `orion/memory_graph/cortex_suggest_extract.py:
        # extract_suggest_text_from_cortex_payload` uses for the same shape.
        text = _extract_cortex_result_text(payload)
        json_blob = extract_first_json_object_text(text) or text
        return parse_reading_batch(json_blob, run_id=run_id, hop_ns=hop_ns)
    # Unreachable given the `max_attempts < 1` guard above -- every loop
    # iteration either returns on success or re-raises on the last attempt.
    # Kept as a defensive fallback rather than trusting that invariant to
    # hold forever silently. Caught in review (an earlier version had no
    # upfront guard, which made this line reachable and misleading: it
    # wrapped whatever exception type the last attempt actually raised in a
    # generic RuntimeError, hiding it from a caller matching on type).
    raise RuntimeError(f"curiosity_supervisor exhausted retries run={run_id}: {last_err}")


async def generate_all_readings(
    bus: Any,
    reader: WorldviewReader,
    *,
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    llm_route: str = DEFAULT_LLM_ROUTE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_run_done: Optional[Any] = None,
) -> list[HopReadingV1]:
    """Read every hop and every prior once, then one RPC call per run.

    A run's own LLM call failing (timeout, decode error) is caught and
    logged, not raised -- so one bad run out of twenty-three cannot blank out
    the readings for the other twenty-two. `on_run_done(run_id, readings)`,
    if given, fires after each run -- the CLI script uses it to print
    progress as it goes rather than going quiet for the whole sweep.
    """
    hops = read_all_hops(reader)
    priors = read_all_priors(reader)
    run_order = build_run_order(reader)
    grouped = group_hops_by_run(hops, run_order=run_order)

    all_readings: list[HopReadingV1] = []
    for run_id, run_hops in grouped:
        try:
            readings = await generate_readings_for_run(
                bus,
                run_id=run_id,
                hops=run_hops,
                priors=priors,
                cortex_request_channel=cortex_request_channel,
                cortex_result_prefix=cortex_result_prefix,
                source=source,
                llm_route=llm_route,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 -- one run's failure must not blank the sweep
            logger.warning(
                "curiosity_supervisor_run_failed run=%s hops=%s err=%s",
                run_id, len(run_hops), exc,
            )
            readings = []
        all_readings.extend(readings)
        if on_run_done is not None:
            on_run_done(run_id, readings)
    return all_readings


def group_readings_by_prior(
    readings: Sequence[HopReadingV1],
) -> dict[str, list[HopReadingV1]]:
    """Readings attributed to each prior, in the order `readings` arrived --
    which is run-chronological if `readings` came from `generate_all_readings`
    (see `group_hops_by_run`). Readings with `about_prior_id is None` are
    dropped; they are not evidence about any one prior's trajectory."""
    out: dict[str, list[HopReadingV1]] = {}
    for r in readings:
        if r.about_prior_id:
            out.setdefault(r.about_prior_id, []).append(r)
    return out


def is_circling(
    readings_for_one_prior: Sequence[HopReadingV1], *, min_hops: int = 3
) -> Optional[bool]:
    """`None` ("not enough evidence") below `min_hops` attributed readings.
    Otherwise `True` iff the most recent `min_hops` readings all read
    `moved_the_claim is False` -- tested repeatedly, moved nothing each time.

    Deliberately not `True` on a single `False`: one inconclusive hop is a
    real, ordinary outcome (`ask_claude_trigger.py`'s own comment: "Inconclusive
    is a real answer"). It takes a RUN of them to call it circling. A `None`
    reading (could not tell) breaks the run rather than counting toward it --
    an ambiguous hop is not evidence of stuck-ness, and treating it as one
    would flag a prior the supervisor is simply unsure about as circling.

    Pure function over already-emitted readings: no bus, no belief edit, no
    decision about what happens next -- the spec's Non-goals list draws that
    line and this stays on the read side of it.
    """
    if len(readings_for_one_prior) < min_hops:
        return None
    recent = readings_for_one_prior[-min_hops:]
    return all(r.moved_the_claim is False for r in recent)
