"""Production acceptance: evidence, routing, strain and immutable admission contracts."""
import asyncio
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
import pytest
from pydantic import ValidationError
from orion.schemas.reading_resonance import (
    ResonancePacketV1, ResonanceCandidateV1, ReadingResonanceRunBriefV1, ReadingResonanceResultV1,
)
from orion.world_pulse_read.resonance import build_judge_prompt, check_judgment
from orion.world_pulse_read.resonance_findings import finding_from_result, resonance_findings
from orion.world_pulse_read.resonance_submit import request_for, bind_request
from orion.world_pulse_read.resonance_strain import StrainNote
from orion.substrate.resonance_seeds import resonance_seeds
from orion.curiosity.worldview import select_priors

NOW = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


def packet():
    return ResonancePacketV1(packet_id="packet-1", seed_id="read-1", root_request_id="root-1",
        read_title="GPU caches", read_url="https://example.org/cache", built_at=NOW,
        read_summary="The cache uses less memory on this GPU.", read_learned="The cache uses less memory on this GPU.",
        candidates=[ResonanceCandidateV1(ref_kind="self_prior", ref_id="p1", store="falkor:orion_worldview:Prior",
                     excerpt="My cache has no room to shrink.", score=0.7),
                    ResonanceCandidateV1(ref_kind="atlas_concept", ref_id="concept-cache", store="falkor:orion_substrate:Concept",
                     excerpt="GPU cache", score=0.5)])


def answer(decision="share"):
    return dict(article_summary="The article measures GPU cache size. It compares two formats. The smaller format saves memory.",
        read_quote="The cache uses less memory on this GPU", connection="This challenges what I believed about my cache.",
        citations=[dict(ref_kind="self_prior", ref_id="p1", how="This is my belief about cache size."),
                   dict(ref_kind="atlas_concept", ref_id="concept-cache", how="This names the cache.")],
        bears_on_prior="strains", strained_prior_id="p1", decision=decision,
        explore_next="Would this format change my cache use?", confidence=0.8)


def receipt(decision="share", raw=None):
    p = packet()
    raw = raw or json.dumps(answer(decision))
    j, check = check_judgment(p, raw, retained_texts=[], prompt=build_judge_prompt(p, []))
    return ReadingResonanceResultV1(resonance_id=p.packet_id, seed_id=p.seed_id, run_id="resonance-"+p.packet_id,
        packet=p, judgment=j, check=check, raw=raw, lane="agent", latency_ms=200, completed_at=NOW)


def test_background_request_has_fixed_identity_and_deadline():
    brief = ReadingResonanceRunBriefV1(packet=packet())
    a, b = request_for(brief), request_for(brief)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.workflow == "reading.resonance" and a.admission.priority == "background"
    assert a.admission.deadline_at == NOW + timedelta(hours=6)
    assert a.brief.packet == packet()
    from orion.schemas.durable_run import DurableRunRequestV1
    wrong = a.model_dump(mode="json")
    wrong["workflow"] = "reading.turn"
    with pytest.raises(ValidationError):
        DurableRunRequestV1.model_validate(wrong)


def test_bind_keeps_first_packet_after_config_or_evidence_changes():
    class Conn:
        stored = None
        async def execute(self, sql, seed, run, raw):
            if self.stored is None:
                self.stored = raw
        async def fetchval(self, *args):
            return self.stored
    async def run():
        conn = Conn()
        first = await bind_request(conn, ReadingResonanceRunBriefV1(packet=packet()))
        later = await bind_request(conn, ReadingResonanceRunBriefV1(
            packet=packet().model_copy(update={"built_at": NOW + timedelta(hours=1)}), max_tokens=1800))
        assert first == later
    asyncio.run(run())


@pytest.mark.parametrize("decision,expected", [("share", "share"), ("note", "note"), ("none", None)])
def test_decision_routes_only_checked_nonempty_connections(decision, expected):
    result = receipt(decision)
    assert result.check.accepted
    excerpts = {("self_prior", "p1"): "CURRENT prior text", ("atlas_concept", "concept-cache"): "CURRENT cache text"}
    f = finding_from_result(result, excerpts)
    if expected is None:
        assert f is None
    else:
        assert f.resonance_decision == expected
        assert "CURRENT prior text" in f.evidence_text
        assert "My cache has no room" not in f.evidence_text
        assert f.resonance_article_summary == answer()["article_summary"]


def test_missing_citation_suppresses_whole_connection():
    assert finding_from_result(receipt(), {("self_prior", "p1"): "Current text"}) is None


def test_bad_judgment_is_receipt_not_finding_or_seed():
    bad = answer()
    bad["citations"][0]["ref_id"] = "invented"
    result = receipt(raw=json.dumps(bad))
    assert not result.check.accepted
    assert finding_from_result(result, {}) is None
    assert resonance_seeds([{"result_json": result.model_dump(mode="json")}]) == []


def test_seed_is_unscored_bounded_deduplicated_and_requires_graph_anchor():
    r = receipt()
    rows = [{"result_json": r.model_dump(mode="json")}] * 3
    seeds = resonance_seeds(rows)
    assert len(seeds) == 1
    assert seeds[0].signal_strength == seeds[0].confidence == 0
    assert "strength:unscored_event" in seeds[0].notes
    assert seeds[0].focal_node_refs == ["concept-cache"]
    r.judgment.citations = [r.judgment.citations[0]]
    assert resonance_seeds([{"result_json": r.model_dump(mode="json")}]) == []
    assert resonance_seeds(rows, cap=0) == []


def test_strain_reopens_stale_prior_and_later_test_clears_it():
    rows = [dict(prior_id="p1", claim="Cache cannot shrink", confidence=0.9, status="supported", times_tested=9,
                 last_tested_at=(NOW-timedelta(days=1)).isoformat()),
            dict(prior_id="p2", claim="Another belief", confidence=0.5, status="open", times_tested=0)]
    note = StrainNote("p1", NOW, "resonance-1", "The cache uses less memory", "https://example.org")
    offered, stale, _ = select_priors(rows, sample=2, stale_after=3, strained={"p1": note})
    assert offered[0].prior_id == "p1" and not stale
    assert "resonance:resonance-1" in offered[0].preview()
    rows[0]["last_tested_at"] = (NOW+timedelta(seconds=1)).isoformat()
    offered, stale, _ = select_priors(rows, sample=2, stale_after=3, strained={"p1": note})
    assert offered[0].prior_id == "p2" and stale[0].prior_id == "p1"


def test_strain_cap_preserves_menu_size():
    rows = [dict(prior_id=f"p{i}", claim=f"Claim {i}", confidence=.9, status="open", times_tested=4) for i in range(6)]
    notes = {f"p{i}": StrainNote(f"p{i}", NOW, f"r{i}", "read quote", "https://example.org") for i in range(6)}
    offered, stale, _ = select_priors(rows, sample=4, stale_after=3, strained=notes)
    assert len(offered) == 2 and len(stale) == 4


def test_hydration_refills_after_own_thread_hits(monkeypatch):
    from orion.world_pulse_read import resonance_retrieval as retrieval
    from orion.world_pulse_read.resonance import PoolDoc, ScoredDoc
    docs = [PoolDoc("atlas_concept", str(i), "graph", f"Concept {i}") for i in range(14)]
    def hydrate(fk, node):
        return docs[int(node)], {"same-thread"} if int(node) < 12 else {"another-thread"}
    monkeypatch.setattr(retrieval, "atlas_hydrate", hydrate)
    scored = [ScoredDoc(d, .7, 1-int(d.ref_id)/100) for d in docs]
    got = asyncio.run(retrieval.hydrate_ranked(None, None, scored, {"same-thread"}))
    assert [x.doc.ref_id for x in got] == ["12", "13"]


def test_citation_novelty_hash_uses_a_set():
    r = receipt()
    excerpts = {(c.ref_kind, c.ref_id): "stored text" for c in r.judgment.citations}
    first = finding_from_result(r, excerpts)
    r.judgment.citations.append(r.judgment.citations[0])
    assert finding_from_result(r, excerpts).resonance_content_id == first.resonance_content_id


def test_landed_read_is_a_real_seed_anchor_even_with_only_memory_citations():
    r = receipt()
    r.judgment.citations = [r.judgment.citations[0]]
    r.packet.landed_concept_ids = ["stored-read-concept"]
    seed = resonance_seeds([{"result_json": r.model_dump(mode="json")}])[0]
    assert seed.focal_node_refs == ["stored-read-concept"]


def test_replica_reselection_keeps_strain_and_all_coverage_slots():
    from orion.curiosity.worldview import WorldviewSnapshot, with_head
    rows = [dict(prior_id=f"p{i}", claim=f"Claim {i}", confidence=.1+i*.05, status="open", times_tested=0) for i in range(15)]
    note = StrainNote("p1", NOW, "r1", "Evidence changed", "https://example.org")
    offered, stale, _ = select_priors(rows, sample=6, stale_after=3, coverage_slots=2, strained={"p1":note})
    view = WorldviewSnapshot(live_priors=offered, stale_priors=stale, prior_rows=tuple(rows), strain_notes={"p1":note})
    for seed in ["one", "two", "three"]:
        result = with_head(view, ["p7", "p9"], sample=6, stale_after=3, rotate_seed=seed, coverage_slots=2)
        assert result.live_priors[0].prior_id == "p1"
        assert len(result.live_priors) == 6
        assert sum(p.coverage for p in result.live_priors) == 2
        assert len({p.prior_id for p in result.live_priors}) == 6
