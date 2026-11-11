"""Accepted reading evidence reopens at most two priors; only Orion's later test clears it."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from orion.schemas.reading_resonance import ReadingResonanceResultV1
from orion.world_pulse_read.resonance_retrieval import obj


@dataclass(frozen=True)
class StrainNote:
    prior_id: str
    strained_at: datetime
    resonance_id: str
    quote: str
    url: str

    def preview(self):
        return f'Reading challenges this belief: "{self.quote}" ({self.url}; resonance:{self.resonance_id})'


async def load_strains(conn, now):
    rows = await conn.fetch("SELECT result_json FROM reading_resonance WHERE completed_at >= $1 "
        "AND completed_at <= $2 AND result_json->'check'->>'accepted'='true' "
        "AND result_json->'judgment'->>'bears_on_prior'='strains' ORDER BY completed_at", now - timedelta(days=14), now)
    out = {}
    for r in rows:
        result = ReadingResonanceResultV1.model_validate(obj(r["result_json"]))
        j = result.judgment
        if result.check.accepted and j and j.decision != "none" and j.strained_prior_id:
            out[j.strained_prior_id] = StrainNote(j.strained_prior_id, result.completed_at, result.resonance_id,
                                                j.read_quote, result.packet.read_url)
    return out
