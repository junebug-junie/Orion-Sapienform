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

Missing Question 2 (does `Hop` land as the turn goes, or in one end-of-turn
burst) was answered live 2026-09-19: as the turn goes. `Hop` itself has
carried `written_at` (the graph's own clock) since that date; hops written
before then carry none. "Chronological" below means the best the graph can
give: each run ordered by its `TurnOutcome.written_at` where one exists, hops
within a run ordered by `worldview.hop_order_key` (untimestamped/legacy hops
first, then by `written_at`, then `n` -- NOT plain `n`, which a retried turn
under the same run_id restarts from 1 on top of the earlier attempt's hops).
A run with no `TurnOutcome` (died before writing one, or the read failed)
sorts last, same "unknown is not oldest" rule `worldview.build_recent_runs`
already applies -- not a claim that it actually happened last.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from typing import Any, Optional, Sequence

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.cursor_policy import assert_read_only_cli_argv, build_cursor_agent_argv
from orion.dev_economics.cursor_limit_events import decide_cursor_budget, observe_cursor_limit
from orion.memory_graph.json_extract import extract_first_json_object_text
from orion.curiosity.worldview import (
    HopRecord,
    Prior,
    ReviewRoleRecord,
    hop_order_key,
    latest_review_role_by_run,
    read_all_review_roles,
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

# Same default as orion-curiosity-peer's Cursor jobs (services/orion-curiosity-
# peer/app/worker.py:_default_cursor -> run_cursor_job's own default) -- one
# run's worth of hops is a comparable-sized read to one HelpRequest.
DEFAULT_CURSOR_TIMEOUT_SEC = 600.0


def group_hops_by_run(
    hops: Sequence[HopRecord], *, run_order: dict[str, int]
) -> list[tuple[str, list[HopRecord]]]:
    """(run_id, hops) pairs, most-recent-run-last -- so a per-prior "last N
    readings" read (see `is_circling`) reads as "most recent" in the flattened
    output without needing its own sort.

    `run_order` is a `run_id -> written_at` map (missing/`None` sorts as
    oldest -- see this module's docstring); hops within a run sort by
    `worldview.hop_order_key`, not plain `n` -- see that function.
    """
    by_run: dict[str, list[HopRecord]] = {}
    for hop in hops:
        by_run.setdefault(hop.run_id, []).append(hop)
    for run_hops in by_run.values():
        # Not `h.n`: a retried attempt's 1,2,3 sits on top of the first
        # attempt's 1,2,3 under one run_id, and `n` alone interleaves them.
        run_hops.sort(key=hop_order_key)
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


def build_grading_sealed_prompt(
    *, run_id: str, hops: Sequence[HopRecord], priors: Sequence[Prior], why: str = ""
) -> str:
    """Sealed read-only contractor prompt: grade Orion's own hops, not
    investigate a new claim. Orion chose this via `:ReviewRole
    {choice: "hire_cursor_review"}` -- see `kickoff_prompt._review_role_section`.

    Same "read-only contractor" framing convention as `services/orion-
    curiosity-peer/app/cursor_invoker.py:build_sealed_prompt`'s `self_inquiry`
    branch (that prompt already names "hop notes" as valid evidence), but
    targets `HopReadingBatchV1` JSON, not `PeerBriefV1` -- a free-form brief
    cannot carry "one reading per hop," which is exactly what grading needs.
    Content reuses `build_reading_prompt` verbatim so Cursor and the internal
    cortex grader are shown identical hop/prior material; only the framing
    and output-shape instructions differ.
    """
    lines = [
        "You are a read-only contractor hired by Orion to grade Orion's OWN "
        "past reasoning, not to investigate a new claim.",
        "Investigate with read/grep/glob/ls only. Do not edit, shell, delete, or mutate.",
        "Do not write :Prior, :Finding, :HopReading, :ReviewRole, or any belief graph node.",
        "",
        f"run_id: {run_id}",
    ]
    if why.strip():
        lines.append(f"Orion's stated reason for asking a contractor to grade this: {why.strip()}")
    lines += ["", build_reading_prompt(priors, hops), ""]
    lines += [
        "Respond with a single JSON object (no markdown fence required) shaped like:",
        '  {"readings": [<one object per hop above>]}',
        "Each reading object's schema (fields you do not know, like reading_id or "
        "timestamps, are filled in by the caller -- do not invent them):",
        json.dumps(_reading_schema_for_model()),
        'An honest "could not tell" (moved_the_claim: null) is a real answer -- '
        "never invent a verdict to fill the field.",
    ]
    return "\n".join(lines)


_MODEL_CANNOT_FILL_THESE = ("hop_written_at", "reading_id", "generated_at", "schema_version")


def _reading_schema_for_model() -> dict[str, Any]:
    """`HopReadingBatchV1`'s JSON schema, minus the caller-stamped fields.

    `parse_reading_batch` always overwrites every field in
    `_MODEL_CANNOT_FILL_THESE` from the caller's own state -- the graph's
    write clock, the row-identity uuid, the generation timestamp, the schema
    tag -- none of which the model can know or needs to think about. Unlike
    `hop_run_id`, which the model IS asked to think about (see that field's
    own docstring for why it stays in the schema despite the same
    overwrite), these cost generation tokens for a value that is discarded
    either way.
    """
    schema = HopReadingBatchV1.model_json_schema()
    reading_ref = schema.get("$defs", {}).get("HopReadingV1")
    if isinstance(reading_ref, dict):
        properties = reading_ref.get("properties", {})
        required = reading_ref.get("required")
        for field in _MODEL_CANNOT_FILL_THESE:
            properties.pop(field, None)
            if isinstance(required, list) and field in required:
                required.remove(field)
    return schema


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
        "structured_output_schema": _reading_schema_for_model(),
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
    payload: Any,
    *,
    run_id: str,
    hop_ns: Sequence[int],
    written_at_by_n: Optional[dict[int, Optional[int]]] = None,
) -> list[HopReadingV1]:
    """Validate the LLM's response into `HopReadingV1` rows.

    `hop_written_at` is stamped from `written_at_by_n` (the caller's own
    hops), never trusted from the model -- same rule as `hop_run_id` below.
    Absent map, or an `n` not in it, stamps None (a legacy hop).

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
        # Never the model's to fill: overwritten unconditionally, not just
        # defaulted, in case a stray key made it into the model's JSON
        # despite not being in the wire schema (_reading_schema_for_model
        # strips reading_id/generated_at/schema_version the same way it
        # strips hop_written_at).
        raw.pop("reading_id", None)
        raw.pop("generated_at", None)
        raw.pop("schema_version", None)
        raw["hop_run_id"] = run_id  # always ours, never the model's -- see docstring
        try:
            raw_n: Optional[int] = int(raw.get("hop_n"))
        except (TypeError, ValueError):
            raw_n = None  # validation below rejects the row for the same reason
        hop_written_at = (written_at_by_n or {}).get(raw_n)
        raw["hop_written_at"] = hop_written_at
        # Deterministic, not the model's random default, WHEN there is a real
        # graph clock to key on: re-running the report script (its own
        # "sample fifty of these" acceptance check invites exactly this) must
        # re-derive the SAME reading_id for the same hop, so the sql-writer's
        # existing insert-only duplicate-skip (a PK collision is caught and
        # logged, not inserted again -- see INSERT_ONLY_MODELS in
        # services/orion-sql-writer/app/worker.py) makes a re-read idempotent
        # instead of silently doubling every row on every re-run. Caught in
        # review: an earlier version left this on HopReadingV1's random
        # uuid4() default, which duplicated every historical reading on a
        # second `--publish` run.
        #
        # Legacy hops (hop_written_at is None) keep the random default --
        # `(run_id, n)` alone is exactly the ambiguous pair this arc's own
        # hop-identity patch found colliding on real data, so hashing on it
        # would silently drop one of two genuinely different readings behind
        # the sql-writer's PK-collision skip instead of storing both.
        if hop_written_at is not None:
            raw["reading_id"] = uuid.uuid5(
                uuid.NAMESPACE_URL, f"curiosity_hop_reading:{run_id}:{raw_n}:{hop_written_at}"
            ).hex
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


def _written_at_by_n(hops: Sequence[HopRecord]) -> dict[int, Optional[int]]:
    """`hop_n -> hop_written_at`, `None` on a genuine `n` collision rather
    than last-wins. Shared by both graders (`generate_readings_for_run` and
    `generate_readings_for_run_via_cursor`) -- one implementation of this
    rule, not a fork per caller.

    None, not last-wins, when an `n` is shared by more than one hop --
    legacy collision (both None, ambiguous either way) or, in principle, a
    fresh one (the resume preamble is advisory prose, not a write-time
    guard, so a model that ignores it can still collide with two real
    timestamps). Silently picking one via a plain dict comprehension would
    stamp a reading about the EARLIER hop with the LATER hop's clock.
    """
    grouped: dict[int, list[Optional[int]]] = {}
    for h in hops:
        grouped.setdefault(h.n, []).append(h.written_at)
    return {n: (vals[0] if len(vals) == 1 else None) for n, vals in grouped.items()}


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
    written_at_by_n = _written_at_by_n(hops)
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
        return parse_reading_batch(
            json_blob, run_id=run_id, hop_ns=hop_ns, written_at_by_n=written_at_by_n
        )
    # Unreachable given the `max_attempts < 1` guard above -- every loop
    # iteration either returns on success or re-raises on the last attempt.
    # Kept as a defensive fallback rather than trusting that invariant to
    # hold forever silently. Caught in review (an earlier version had no
    # upfront guard, which made this line reachable and misleading: it
    # wrapped whatever exception type the last attempt actually raised in a
    # generic RuntimeError, hiding it from a caller matching on type).
    raise RuntimeError(f"curiosity_supervisor exhausted retries run={run_id}: {last_err}")


async def generate_readings_for_run_via_cursor(
    *,
    run_id: str,
    hops: Sequence[HopRecord],
    priors: Sequence[Prior],
    why: str = "",
    agent_bin: str,
    cwd: str,
    model: Optional[str] = None,
    timeout_sec: float = DEFAULT_CURSOR_TIMEOUT_SEC,
    run: Optional[Any] = None,
) -> list[HopReadingV1]:
    """Grade one run's hops via a read-only Cursor contractor instead of the
    internal cortex LLM grader -- Orion's own choice, via `:ReviewRole
    {choice: "hire_cursor_review"}`.

    Same budget gate as the investigation-side hire path
    (`orion.dev_economics.cursor_limit_events.decide_cursor_budget`):
    fail-closed on anything but a fresh observed "clear" reading. Raises on
    refusal rather than returning `[]` -- the caller
    (`generate_readings_for_run_routed`) catches this and falls back to
    `self_review` for the run, so a refusal means "graded differently," not
    "graded nothing."

    Deliberately does NOT go through the `HelpRequest` -> bus ->
    `orion-curiosity-peer` -> `PeerBriefV1` pipeline the investigation hire
    path uses: `PeerBriefV1` is free-form prose (summary/evidence_pointers/
    open_questions), right for "investigate and report back," wrong for "one
    `HopReadingV1` per hop." This calls the Cursor Agent CLI directly, in the
    same read-only ask-mode argv (`orion.curiosity.cursor_policy.
    assert_read_only_cli_argv` -- the same safety check
    `orion-curiosity-peer` uses, not a fork of it), and parses the response
    through the SAME `parse_reading_batch` the internal grader uses, so a
    reading's shape is identical regardless of which grader produced it.
    `run` is an injectable `subprocess.run`-shaped callable for tests; never
    call a live Cursor agent from a test.
    """
    if not hops:
        return []
    limit = observe_cursor_limit()
    refusal = decide_cursor_budget(limit)
    if refusal is not None:
        raise RuntimeError(
            f"cursor_grading_budget_refused run={run_id} reason={refusal}"
        )

    hop_ns = [h.n for h in hops]
    written_at_by_n = _written_at_by_n(hops)
    prompt = build_grading_sealed_prompt(run_id=run_id, hops=hops, priors=priors, why=why)
    argv = build_cursor_agent_argv(agent_bin=agent_bin, prompt=prompt, workspace=cwd, model=model)
    assert_read_only_cli_argv(argv)

    runner = run or subprocess.run
    try:
        proc = await asyncio.to_thread(
            runner, argv, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"cursor agent binary not found: {agent_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"cursor grading timed out after {exc.timeout}s run={run_id}"
        ) from exc

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"cursor agent exited {proc.returncode} run={run_id}: {combined[:2000]}"
        )

    json_blob = extract_first_json_object_text(combined) or combined
    out = parse_reading_batch(
        json_blob, run_id=run_id, hop_ns=hop_ns, written_at_by_n=written_at_by_n
    )
    if not out:
        # `parse_reading_batch` is tolerant by design -- unparseable JSON,
        # prose instead of a JSON object, or a wrong-shape response all
        # return `[]` rather than raising (same contract the internal cortex
        # grader relies on for a single BAD reading in an otherwise-good
        # batch). Cursor returning exit 0 with prose instead of structured
        # output would silently pass through as "graded, zero readings" --
        # exactly the empty-shell success CLAUDE.md 0A bans, and exactly
        # what this function's own docstring promises never happens (raise,
        # not grade nothing, so the caller falls back to self_review).
        # Caught in review.
        raise RuntimeError(
            f"cursor_grading_produced_no_readings run={run_id} hops={len(hops)} "
            f"raw_output={combined[:500]!r}"
        )
    return out


async def generate_readings_for_run_routed(
    bus: Any,
    *,
    run_id: str,
    hops: Sequence[HopRecord],
    priors: Sequence[Prior],
    review_choice: Optional[ReviewRoleRecord],
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    llm_route: str = DEFAULT_LLM_ROUTE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cursor_agent_bin: Optional[str] = None,
    cursor_cwd: Optional[str] = None,
    cursor_model: Optional[str] = None,
    cursor_timeout_sec: float = DEFAULT_CURSOR_TIMEOUT_SEC,
) -> list[HopReadingV1]:
    """Dispatch one run's grading to self-review or Cursor, per Orion's
    `:ReviewRole` (missing node, or `cursor_agent_bin` unset, means
    `self_review` -- the same "missing node is no decision" default
    `:InvestigationRole` uses for hire choice).

    A Cursor failure (budget refusal, CLI error, timeout) falls back to
    `self_review` for THIS run rather than returning `[]` -- "graded
    differently," never "graded nothing." The fallback is logged so a soak
    can tell how often Orion's choice actually held.
    """
    choice = (review_choice.choice if review_choice else "self_review") or "self_review"
    if choice == "hire_cursor_review" and cursor_agent_bin:
        try:
            return await generate_readings_for_run_via_cursor(
                run_id=run_id,
                hops=hops,
                priors=priors,
                why=review_choice.why if review_choice else "",
                agent_bin=cursor_agent_bin,
                cwd=cursor_cwd or ".",
                model=cursor_model,
                timeout_sec=cursor_timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 -- fall back, do not grade nothing
            logger.warning(
                "curiosity_supervisor_cursor_grading_fell_back run=%s err=%s -- "
                "falling back to self_review for this run",
                run_id, exc,
            )
    return await generate_readings_for_run(
        bus,
        run_id=run_id,
        hops=hops,
        priors=priors,
        cortex_request_channel=cortex_request_channel,
        cortex_result_prefix=cortex_result_prefix,
        source=source,
        llm_route=llm_route,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
    )


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
    cursor_agent_bin: Optional[str] = None,
    cursor_cwd: Optional[str] = None,
    cursor_model: Optional[str] = None,
    cursor_timeout_sec: float = DEFAULT_CURSOR_TIMEOUT_SEC,
) -> list[HopReadingV1]:
    """Read every hop, every prior, and every `:ReviewRole` once, then one
    grading call per run -- self_review (cortex RPC) or hire_cursor_review
    (Cursor CLI), per Orion's own choice; see `generate_readings_for_run_routed`.

    A run's own grading call failing is caught and logged, not raised -- so
    one bad run out of twenty-three cannot blank out the readings for the
    other twenty-two. `cursor_agent_bin=None` (the default) disables the
    Cursor path entirely regardless of what any `:ReviewRole` says -- every
    run grades via self_review, same as before this patch existed, until a
    caller explicitly opts in by passing a real binary path.
    `on_run_done(run_id, readings)`, if given, fires after each run -- the
    CLI script uses it to print progress as it goes rather than going quiet
    for the whole sweep.
    """
    hops = read_all_hops(reader)
    priors = read_all_priors(reader)
    run_order = build_run_order(reader)
    grouped = group_hops_by_run(hops, run_order=run_order)
    review_choices = latest_review_role_by_run(read_all_review_roles(reader))

    all_readings: list[HopReadingV1] = []
    for run_id, run_hops in grouped:
        try:
            readings = await generate_readings_for_run_routed(
                bus,
                run_id=run_id,
                hops=run_hops,
                priors=priors,
                review_choice=review_choices.get(run_id),
                cortex_request_channel=cortex_request_channel,
                cortex_result_prefix=cortex_result_prefix,
                source=source,
                llm_route=llm_route,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
                cursor_agent_bin=cursor_agent_bin,
                cursor_cwd=cursor_cwd,
                cursor_model=cursor_model,
                cursor_timeout_sec=cursor_timeout_sec,
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
