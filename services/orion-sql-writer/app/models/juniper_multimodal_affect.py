from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base

# Repo convention (causal_geometry_snapshot.py, harness_turn_trace.py, and
# four others): real JSONB on Postgres, plain JSON everywhere else, so the
# SQLite-backed shape tests can still create this table. A bare JSONB()
# raises UnsupportedCompilationError under SQLite.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class JuniperMultimodalAffectSQL(Base):
    """Durable record of orion-juniper-affective-state's real webcam+mic
    AffectGPT reads, published on ``orion:affectgpt:assessment``
    (``JuniperMultimodalAffectV1``, see orion/schemas/affectgpt.py).

    First real consumer of that channel. Same dead-end shape found and
    fixed for the sibling text-only signal
    (``JuniperAffectiveStateSQL``/PR #1629) and for
    ``orion:substrate:doc_semantic_drift`` (PR #1730): ``OrionBusAsync.publish()``
    is Redis pub/sub, not a stream (``async_service.py:271``), so with no
    subscriber every event was dropped the instant it published. The one
    existing reader, ``orion/situational/juniper_affect_state.py`` (added
    PR #1865), reads a SEPARATE single-key Redis SETEX mirror with a 1h
    TTL -- once that key expires (or a redeploy clears Redis), the fact a
    capture ever happened is gone. This table is the actual durable
    record: a history to look back on, independent of that TTL.

    Deliberately NOT persisted: ``transcript``
    -----------------------------------------
    ``JuniperMultimodalAffectV1.transcript`` is Whisper's verbatim
    transcription of Juniper's own spoken words -- already a deliberate,
    Juniper-approved exception to this domain's "paths only, never raw
    content" rule for the bus wire (see that field's docstring), but a
    transient pub/sub broadcast and a durable, queryable Postgres table are
    materially different privacy postures. This model declares no
    ``transcript`` column; ``_write_row()``'s column-filter (matches
    incoming payload keys against declared SQLAlchemy columns, drops the
    rest) means it is silently never written here on the success path,
    without needing any bespoke redaction code. ``raw_response`` (the
    model's OWN generated read of Juniper's affect, already surfaced live
    in the Hub UI panel and already on the bus) is persisted in full --
    keeping it durably does not widen exposure beyond what already exists.

    Review finding, 2026-08-25: the column-filter only covers the success
    path. ``worker.py``'s ``_handle_envelope_body`` has one shared
    exception handler for every ``MODEL_MAP`` route that falls back to
    ``_write_fallback()`` with the RAW, unfiltered ``env.payload`` (schema
    drift raising in ``_coerce_payload``, a non-duplicate-key DB error,
    ...) -- ``_write_fallback`` does no field filtering of its own, only
    JSON-compatibility sanitization. Fixed by an explicit
    ``sql_model is JuniperMultimodalAffectSQL`` redaction scoped to that
    one except block, not a change to the shared handler.

    ``face_detection``/``timings``/``input_ref`` are also not persisted --
    debug/perf telemetry already inspectable via
    ``services/orion-juniper-affective-state/scripts/tap_assessments.py``
    and service logs; keeping this row lean to what answers "how has
    Juniper's affect trended" rather than duplicating a debug surface.

    ``event_id`` is NOT a field on ``JuniperMultimodalAffectV1`` (unlike
    the tiling-window text signal, this is a discrete per-capture event
    with no natural deterministic key) -- ``worker.py``'s
    ``_affectgpt_multimodal_event_id()`` synthesizes it from the
    envelope's ``correlation_id`` (the one id threading the
    retina-capture/worker-assess/this-event legs of a single tick, per the
    schema's own ``correlation_id`` docstring), falling back to the
    envelope id, then a fresh uuid4. Using ``correlation_id`` means a
    redelivery would merge onto the same row rather than duplicate it --
    UNVERIFIED whether real redelivery ever happens: ``OrionBusAsync`` is
    plain Redis pub/sub with no consumer-group/ack/replay mechanism, so a
    dropped event (subscriber disconnected) is lost, not redelivered.
    This only protects against an actual re-publish (e.g. a producer
    retry). The one real producer,
    ``orion-juniper-affective-state/app/main.py``'s ``_publish_event``,
    already explicitly keeps the envelope-level and payload-level
    ``correlation_id`` in sync; a hypothetical second producer that
    didn't would key its rows on an unrelated, non-reproducible
    envelope-generated id instead.
    """

    __tablename__ = "juniper_multimodal_affect_log"

    event_id = Column(String, primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # No SQLAlchemy-side default: JuniperMultimodalAffectV1.source is a
    # Literal["affectgpt", "vision"] with a default, so the wire payload
    # always already carries this key -- a column default here could never
    # actually fire and would misleadingly imply the value is optional.
    # (Widened from Literal["affectgpt"] on 2026-08-26; no DDL change needed
    # because this was always a plain String.)
    source = Column(String, nullable=False)
    # "manual" (POST /trigger or /capture_and_assess), "ambient" (Hub's
    # recurring toggle loop), or "chat_turn_pre"/"chat_turn_post" (Hub's
    # per-chat-turn bracket, 2026-08-26) -- see
    # JuniperMultimodalAffectV1.trigger. Plain String, not an enum/check
    # constraint, so widening the producer's Literal never needs a
    # coordinated DDL change.
    trigger = Column(String, nullable=True)
    subtitle_source = Column(String, nullable=True)

    ok = Column(Boolean, nullable=False)
    error = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    model_ckpt = Column(String, nullable=True)

    # The model's own generated affect read -- not Juniper's words. See
    # class docstring for why this is persisted in full while transcript
    # is not.
    raw_response = Column(Text, nullable=True)

    # Threads this row to the retina-capture/worker-assess legs of the
    # same tick (JuniperMultimodalAffectV1.correlation_id) -- indexed
    # since it is also this table's own event_id in the common case, but
    # kept as its own column for the fallback-generated-id path.
    correlation_id = Column(String, nullable=True, index=True)

    # The Orion-mode chat turn this capture belonged to, on the
    # chat_turn_pre/chat_turn_post pair only (NULL for manual/ambient).
    # A DIFFERENT join axis from correlation_id above: that one joins the
    # three RPC legs of a single capture attempt, this one joins a capture
    # to the conversation that caused it AND joins a turn's pre/post pair
    # to each other.
    #
    # Added 2026-08-26 in the same patch that introduced the field. Without
    # a declared column here, _write_row's column-filter would silently
    # drop the key -- the join would exist on the bus and in the 1h Redis
    # mirror but never durably, which defeats the point: the pair is only
    # worth capturing because you can go back and ask about it later.
    # Indexed because the only query this column exists to serve is
    # "give me both legs for turn X".
    chat_correlation_id = Column(String, nullable=True, index=True)

    # ── Added 2026-08-26 with the vision backend ────────────────────────
    #
    # These four exist because diagnosing the failure that motivated the
    # cutover required pointing Juniper's own webcam at her twice. The
    # producer had ALWAYS put face_detection and timings on the bus; this
    # table simply never declared columns for them, so _write_row's
    # column-filter dropped both on every insert and the durable record
    # could not answer "was there even a face in frame when the model said
    # that?". Six stored rows, and not one of them could be diagnosed.

    # "vision" | "affectgpt". Not just derivable from observed_at vs the
    # cutover date: the affectgpt path survives as a rollback, so the two
    # backends can interleave in time. An analysis excluding the three
    # known-bad reads needs provenance, not date arithmetic.
    backend = Column(String, nullable=True)

    # The structured AffectReadV1 (valence/arousal/primary_affect/cues/
    # confidence/cannot_tell). NULL for every affectgpt row -- that backend
    # only ever produced prose. A consumer must read NULL as "no structured
    # read", never as "affect was neutral".
    affect = Column(_JSONB, nullable=True)

    # frames_total / frames_detected / detection_rate / frames_sampled. The
    # quality gate's own input, so a stored read can be re-judged later
    # against a threshold different from the one live at write time --
    # query with (face_detection->>'detection_rate')::float.
    face_detection = Column(_JSONB, nullable=True)

    # Per-stage seconds (sample/upload/generate/total). Kept because the
    # whole point of the swap was that the replaced path cost ~28s of
    # camera-plus-inference to return a refusal; the claim that this one is
    # faster should be checkable from the record rather than from memory.
    timings = Column(_JSONB, nullable=True)

    frames_used = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
