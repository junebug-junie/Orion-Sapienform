from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.sql import func

from app.db import Base


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
    rest) means it is silently never written here, without needing any
    bespoke redaction code. ``raw_response`` (the model's OWN generated
    read of Juniper's affect, already surfaced live in the Hub UI panel
    and already on the bus) is persisted in full -- keeping it durably
    does not widen exposure beyond what already exists.

    ``face_detection``/``timings``/``input_ref`` are also not persisted --
    debug/perf telemetry already inspectable via
    ``services/orion-juniper-affective-state/scripts/tap_assessments.py``
    and service logs; keeping this row lean to what answers "how has
    Juniper's affect trended" rather than duplicating a debug surface.

    ``event_id`` is NOT a field on ``JuniperMultimodalAffectV1`` (unlike
    the tiling-window text signal, this is a discrete per-capture event
    with no natural deterministic key) -- ``worker.py`` synthesizes it
    from the envelope's ``correlation_id`` (the one id threading the
    retina-capture/worker-assess/this-event legs of a single tick, per the
    schema's own ``correlation_id`` docstring), falling back to the
    envelope id, then a fresh uuid4. Using ``correlation_id`` means a
    redelivered event upserts onto the same row instead of duplicating it.
    """

    __tablename__ = "juniper_multimodal_affect_log"

    event_id = Column(String, primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)

    source = Column(String, nullable=False, default="affectgpt")
    # "manual" (POST /trigger or /capture_and_assess) vs "ambient" (Hub's
    # recurring toggle loop) -- see JuniperMultimodalAffectV1.trigger.
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

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
