"""Write referent resolution into Postgres, inside the episode persist transaction (psycopg).

One advisory lock serializes resolution (two concurrent persists must not both mint a
node for the same key). Every write is idempotent: deterministic ids plus
ON CONFLICT DO NOTHING, and a descriptor's expiry is computed from the memory's own
time, so replaying a persist (or running the checkpoint backfill twice) writes the
same rows. Falkor is never touched here; the projector in orion-memory-consolidation
materializes what this writes (transactional outbox, #2497).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from orion.substrate.graph_journal import journal_row

from .aliases import TokenFrequency, first_token, normalize_alias, slug_text
from .cooccurrence import EndpointV1, cooccurrence_claims
from .resolve import (
    AliasIndex,
    AliasRow,
    ReferentPolicy,
    candidate_aliases,
    resolve_referent,
)

logger = logging.getLogger(__name__)

_LOCK = "SELECT pg_advisory_xact_lock(hashtext('referent_alias'))"
_LOAD = """
SELECT node_id, alias_norm, alias_text, alias_class, referent_kind, promotion_state, admitted_by, proposed_by,
       grounded_in, valid_until FROM referent_alias
"""
_INSERT_ALIAS = """
INSERT INTO referent_alias (node_id, alias_norm, alias_text, alias_class, referent_kind, promotion_state,
                            admitted_by, proposed_by, grounded_in, valid_until, created_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (node_id, alias_norm) DO NOTHING
"""
_REFRESH_ALIAS = """
UPDATE referent_alias SET valid_until = %s, updated_at = %s
WHERE node_id = %s AND alias_norm = %s AND (valid_until IS NULL OR valid_until < %s)
"""
_SET_NODE = """
UPDATE episode_memory_referent SET node_id = %s WHERE memory_id = %s AND referent_key = %s
  AND node_id IS DISTINCT FROM %s
"""
_INSERT_QUESTION = """
INSERT INTO memory_tension_shadow (question_id, text, kind, scope, answer_via, source_episode_id, source_refs,
                                   referent_keys, created_at)
VALUES (%s, %s, 'question', %s, %s, %s, %s, %s, %s) ON CONFLICT (question_id) DO NOTHING
"""
# Juniper's own words: the corpus whose word frequencies decide name vs descriptor.
_PROMPT_DF = """
SELECT count(*) FILTER (WHERE to_tsvector('simple', coalesce(prompt, '')) @@ plainto_tsquery('simple', %s)) AS df,
       count(*) AS n
FROM chat_history_log
"""
_DECIDED = """
SELECT DISTINCT target_id FROM substrate_graph_journal
WHERE event_kind = 'decision' AND proposal_kind = 'relationship_assertion'
"""
# Any unique violation (event id, or a second decision for the same revision) is a no-op:
# the journal already holds that fact.
_INSERT_JOURNAL = """
INSERT INTO substrate_graph_journal (event_id, event_kind, proposal_kind, proposal_id, decision_id, target_id,
                                     revision, outcome, actor, payload, recorded_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s) ON CONFLICT DO NOTHING
"""


@dataclass
class MemoryReferents:
    memory_id: str
    episode_id: str
    created_at: datetime                 # when Juniper last used these names (descriptor expiry base)
    referents: list[tuple[str, str]]     # (key, role)
    aliases_by_key: dict[str, list[str]] = field(default_factory=dict)
    # (source_id, quote) of the EPISODE's verified chat_prompt evidence: grounds aliases.
    prompt_quotes: list[tuple[str, str]] = field(default_factory=list)
    # This memory's own verified chat_prompt quotes: the only source of co-occurrence claims.
    own_prompt_quotes: list[str] = field(default_factory=list)


def _row(r: Any) -> AliasRow:
    return AliasRow(**{k: r[k] for k in ("node_id", "alias_norm", "alias_text", "alias_class", "referent_kind",
                                         "promotion_state", "admitted_by", "proposed_by", "grounded_in",
                                         "valid_until")})


async def _fetchall(conn: Any, sql: str, params: tuple = ()) -> list[Any]:
    return await (await conn.execute(sql, params)).fetchall()


async def _frequencies(conn: Any, tokens: set[str]) -> dict[str, TokenFrequency]:
    out: dict[str, TokenFrequency] = {}
    for token in sorted(tokens):
        row = (await _fetchall(conn, _PROMPT_DF, (token,)))[0]
        out[token] = TokenFrequency(documents=int(row["df"]), total=int(row["n"]))
    return out


async def persist_referents(
    conn: Any,
    memories: list[MemoryReferents],
    *,
    now: datetime,
    policy: ReferentPolicy,
    proposed_by: str = "episode_writer",
) -> dict[str, int]:
    """Resolve every referent, write aliases, node ids, questions and co-occurrence claims."""
    from psycopg.types.json import Jsonb

    await conn.execute(_LOCK)
    index = AliasIndex(_row(r) for r in await _fetchall(conn, _LOAD))
    tokens = {first_token(normalize_alias(text)) for m in memories for key, _ in m.referents
              for text in [slug_text(key), *m.aliases_by_key.get(key, [])]}
    frequency = await _frequencies(conn, {t for t in tokens if t})
    decided = {r["target_id"] for r in await _fetchall(conn, _DECIDED)}
    counts = {"resolved": 0, "minted": 0, "aliases": 0, "questions": 0, "proposals": 0, "decisions": 0}

    for m in memories:
        endpoints: dict[str, EndpointV1] = {}
        for key, _role in m.referents:
            cands = candidate_aliases(key, m.aliases_by_key.get(key, []), m.prompt_quotes, frequency)
            res = resolve_referent(key, cands, index, now=now, last_use=m.created_at, policy=policy,
                                   proposed_by=proposed_by)
            counts["resolved"] += 1
            counts["minted"] += int(res.minted)
            for row in res.new_rows:
                await conn.execute(_INSERT_ALIAS, (
                    row.node_id, row.alias_norm, row.alias_text, row.alias_class, row.referent_kind,
                    row.promotion_state, row.admitted_by, row.proposed_by, row.grounded_in, row.valid_until,
                    m.created_at, m.created_at))
                counts["aliases"] += int(row.alias_class != "key")
            for row in res.refreshed:
                await conn.execute(_REFRESH_ALIAS, (row.valid_until, now, row.node_id, row.alias_norm,
                                                    row.valid_until))
            for q in res.questions:
                await conn.execute(_INSERT_QUESTION, (
                    q.question_id, q.text, q.scope, q.answer_via, m.episode_id,
                    Jsonb([{"reason": q.reason, "node_ids": list(q.node_ids), "memory_id": m.memory_id}]),
                    list(q.referent_keys), m.created_at))
                counts["questions"] += 1
            await conn.execute(_SET_NODE, (res.node_id, m.memory_id, key, res.node_id))
            kind = index.node_kind(res.node_id) or ""
            endpoints[res.node_id] = EndpointV1(
                key=key, node_id=res.node_id, node_kind="concept" if kind == "concept" else "entity",
                state=index.node_state(res.node_id) or "proposed", names=tuple(index.live_name_norms(res.node_id, now)))
        claims = cooccurrence_claims(
            memory_id=m.memory_id, endpoints=list(endpoints.values()),
            prompt_quotes=m.own_prompt_quotes, decided_targets=decided,
            accept=policy.cooccurrence_auto_accept, recorded_at=m.created_at)
        for claim in claims:
            await conn.execute(_INSERT_JOURNAL, journal_row(claim.proposal))
            counts["proposals"] += 1
            if claim.decision is not None:
                await conn.execute(_INSERT_JOURNAL, journal_row(claim.decision))
                counts["decisions"] += 1
    logger.info("referents_persisted %s", " ".join(f"{k}={v}" for k, v in counts.items()))
    return counts
