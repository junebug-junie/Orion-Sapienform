"""What a read changed in Orion: the seam the daily letter and reading outreach share.

A ``ReadingFindingV1`` is one recorded change a completed read caused, built only from stored
records (orion/world_pulse_read/findings.py), never from model prose:

* ``link_made`` -- Stage 2 proposed a relationship claim, ``reading_quote_rule_v1`` accepted it,
  the journal's latest decision still holds it (provisional/canonical) and, for ``projected``,
  the AssertionProjector applied it to the graph.
* ``belief_moved`` -- a curiosity run moved a prior's confidence (or formed it), and the text
  that run wrote on that prior (``why`` when the run last tested it, ``formed_from`` when it
  formed it) names this read by URL, seed id or request id. Readings never write priors
  themselves (PR #2199 boundary); this is the only path by which a read moves a belief.

One more kind is expected (``resonance``, design branch docs/reading-resonance-design): add the
value to ``ReadingFindingKind`` and a producer; consumers group by ``seed_id`` and render
``headline`` / ``evidence_text`` without knowing the kind.

Not a bus payload: producer and consumers all run in Hub. Register it and add a channel the day
a producer lives in another service.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReadingFindingKind = Literal["belief_moved", "link_made"]


class ReadingFindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)  # deterministic: "link:<assertion_id>" / "belief:<prior_id>@<run_id>"
    kind: ReadingFindingKind
    seed_id: str = Field(min_length=1)  # the read (world_pulse_read_seed.seed_id)
    observed_at: datetime  # when the change was recorded
    headline: str = Field(min_length=1)  # one deterministic line built from the records
    evidence_text: str = ""  # the record text that backs it: the verbatim quote, or Orion's own words on the prior
    evidence_refs: list[str] = Field(default_factory=list)
    # link_made
    assertion_id: str | None = None
    projected: bool | None = None  # None: projection state unknown (journal unreadable)
    # belief_moved
    prior_id: str | None = None
    run_id: str | None = None
    confidence_before: float | None = None  # None: the run formed the prior
    confidence_after: float | None = None


class ReadingOriginStepV1(BaseModel):
    """One read in the chain that led to a read (orion/world_pulse_read/origin.py)."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str = Field(min_length=1)
    kind: str  # world_pulse_read_seed.kind: digest_item | finding | reading
    url: str = ""
    title: str = ""
    requested_by: str = ""  # request_json.requested_by; empty on rows stored before requests existed
    invocation_context: str = ""
    why_now: str = ""  # the requester's note: Juniper's on an operator request, Orion's own otherwise
    parent_request_id: str = ""  # the read this one hopped from; empty at a chain's start


class ReadingFindingBundleV1(BaseModel):
    """Every unreported finding of one read: one outreach message per bundle."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str = Field(min_length=1)
    title: str = ""
    url: str = ""
    learned: str = ""  # Stage 2 summary (or Stage 1 what_i_learned), as stored: Orion's reading of it
    # What the piece itself says (Stage 2 ``article_summary``), for Juniper to learn from too.
    # A property of the read, not of one finding: a producer in another service (resonance)
    # leaves it to the consumer, which reads it from the seed row like every other read field.
    article_summary: str = ""
    findings: list[ReadingFindingV1] = Field(min_length=1)
    # How the read started, oldest first, ending with the read itself. Empty: not looked up.
    origin: list[ReadingOriginStepV1] = Field(default_factory=list)

    @property
    def finding_ids(self) -> list[str]:
        return sorted(f.finding_id for f in self.findings)
