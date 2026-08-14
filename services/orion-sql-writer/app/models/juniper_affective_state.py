from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class JuniperAffectiveStateSQL(Base):
    """Durable record of orion-cocreation-signals' affective_state producer --
    one row per 15-minute window, published on
    ``orion:substrate:juniper_affective_state``.

    First real consumer of that channel, which was ``consumer_services: []``
    from 2026-08-11 until this patch. ``OrionBusAsync.publish()`` is Redis
    pub/sub (``async_service.py:271``), not a stream, so with no subscriber
    every event was dropped the instant it was published -- confirmed live,
    ``PUBSUB NUMSUB`` returned 0. Three days of real readings exist nowhere.

    Why the raw COUNTS are the point, not the rate
    ----------------------------------------------
    ``swear_frequency`` is published over a fixed 15-minute window, but real
    typing is bursty. Measured over the live 31-day corpus: 78% of windows
    are empty (``swear_frequency`` is NULL), and of the rest 69% read exactly
    0.0 -- only ~6.7% of all windows carry a non-zero value, and a short
    window turns a single swear into a huge spike (one real window reads
    0.077 off 13 words).

    Persisting ``swear_count`` and ``word_count`` rather than only the ratio
    means the window choice is no longer baked in: any consumer can re-derive
    the rate over an hour, a day, or a session by summing counts, which is
    exactly how ``aggregate_scores()`` weights correctly across messages of
    very different lengths. A stored ratio could not be re-aggregated without
    reintroducing the short-message bias.

    Deliberately no consumer behaviour, no threshold, and no Hub panel in
    this patch -- same discipline as ``DocSemanticDriftSQL``. Just stop
    throwing the data away.

    Note on comparability: readings from before 2026-08-13 are not comparable
    with later ones. The parser was counting ``/compact`` continuation
    summaries -- model-authored text -- as Juniper's prose, which was 30.7% of
    the denominator and 45.6% of the numerator (PR #1629). No such rows exist
    in this table, since nothing was ever persisted before this patch, so the
    series starts clean.
    """

    __tablename__ = "juniper_affective_state_log"

    event_id = Column(String, primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)

    # Half-open [window_since, window_until). Both indexed: the first query
    # anyone re-aggregating this will run is a range scan over window bounds.
    window_since = Column(DateTime(timezone=True), nullable=False, index=True)
    window_until = Column(DateTime(timezone=True), nullable=False, index=True)

    message_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    swear_count = Column(Integer, nullable=False)

    # NULL when word_count == 0 -- an empty window (Juniper typed nothing) is
    # a genuinely different fact from a calm one (messages existed, none were
    # swears), and 78% of real windows are the former. A NOT NULL constraint
    # with a 0.0 default would silently merge the two and make every
    # downstream average wrong.
    swear_frequency = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
