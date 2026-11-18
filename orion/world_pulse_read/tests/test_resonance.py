from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import get_args

import pytest

from orion.schemas.reading_resonance import (
    RELATION_REF_KINDS,
    ReadingResonanceJudgmentV1,
    ResonanceNeighborV1,
    ResonanceRefKind,
)
from orion.world_pulse_read.resonance import (
    JUDGE_INSTRUCTIONS,
    KIND_BUDGET,
    PoolDoc,
    ReadInput,
    ScoredDoc,
    build_judge_prompt,
    build_packet,
    check_judgment,
    fallback_article_summary,
    packet_id_for,
    strained_prior,
    with_recent_counts,
)
from orion.world_pulse_read.resonance_similarity import (
    CENTERED_OFFER_FLOOR,
    center,
    dot,
    hub_scores,
    mean_vector,
    rank_score,
)

AT = datetime(2026, 10, 13, tzinfo=timezone.utc)
READ = ReadInput(
    seed_id="seed-1", root_request_id="root-1", title="GPU Compute", url="https://example.test/gpu",
    summary="CoreWeave lists GB200 NVL72 racks at 130 kW per rack.",
    learned="CoreWeave rents bare-metal NVIDIA GPUs. The GB200 NVL72 rack is liquid cooled. It draws 130 kW.",
    retained_texts=[("a" * 64, "GB200 NVL72: 72 GPUs, 36 CPUs, liquid cooled, 130 kW per rack.")],
)


def sd(kind: str, ref: str, cos: float, rank: float | None = None, excerpt: str | None = None, **kw) -> ScoredDoc:
    return ScoredDoc(PoolDoc(kind, ref, f"store:{kind}", excerpt if excerpt is not None else f"text {ref}", **kw),
                     cos, cos if rank is None else rank)


def judged(**over) -> dict:
    base = {
        "article_summary": "CoreWeave rents GPUs. Its GB200 racks draw 130 kW. They are liquid cooled.",
        "read_quote": "liquid cooled, 130 kW per rack",
        "connection": "This is the industrial version of the V100 cards Juniper put in Circe.",
        "citations": [{"ref_kind": "crystallization", "ref_id": "c1", "how": "Juniper on my hardware"}],
        "bears_on_prior": None, "strained_prior_id": None, "decision": "share",
        "explore_next": None, "confidence": 0.7,
    }
    base.update(over)
    return base


# ---- packet builder ----

def test_packet_respects_budget_floor_dedup_and_exclusion():
    scored = [sd("crystallization", f"c{i}", 0.5 - i * 0.01) for i in range(8)]
    scored.append(sd("crystallization", "dup", 0.9, excerpt="text c0"))  # same excerpt as c0
    scored.append(sd("self_prior", "p-low", CENTERED_OFFER_FLOOR - 0.01))
    scored.append(sd("self_prior", "p-self", 0.8))
    scored.append(sd("open_thread", "t1", -0.2, floorless=True))  # open concern arc
    scored.append(sd("open_thread", "t2", -0.2))  # an earlier read's thread is floored
    p = build_packet(READ, scored, built_at=AT, exclude_ref_ids={"p-self"})
    kinds = [c.ref_kind for c in p.candidates]
    assert kinds.count("crystallization") == KIND_BUDGET["crystallization"]
    crys = [c.ref_id for c in p.candidates if c.ref_kind == "crystallization"]
    assert crys[0] == "dup" and "c0" not in crys  # dedup keeps the better-ranked copy
    assert "p-low" not in {c.ref_id for c in p.candidates}
    assert "p-self" not in {c.ref_id for c in p.candidates}
    assert "t1" in {c.ref_id for c in p.candidates} and "t2" not in {c.ref_id for c in p.candidates}


def test_packet_ranks_by_rank_not_cosine_and_is_deterministic():
    scored = [sd("dream", "hub", 0.6, rank=0.2), sd("dream", "real", 0.5, rank=0.45), sd("dream", "x", 0.4, rank=0.1)]
    p1 = build_packet(READ, scored, built_at=AT)
    p2 = build_packet(READ, list(reversed(scored)), built_at=AT)
    assert [c.ref_id for c in p1.candidates] == ["real", "hub"]
    assert p1.model_dump() == p2.model_dump()
    assert p1.packet_id == packet_id_for("seed-1")


def test_recent_counts_count_other_packets_only():
    a = build_packet(READ, [sd("dream", "d1", 0.5)], built_at=AT)
    b = build_packet(ReadInput("seed-2", None, "", "", "", ""), [sd("dream", "d1", 0.5), sd("dream", "d2", 0.5)],
                     built_at=AT)
    a2, b2 = with_recent_counts([a, b])
    assert a2.candidates[0].recent_packet_count == 1
    assert {c.ref_id: c.recent_packet_count for c in b2.candidates} == {"d1": 1, "d2": 0}


def test_prompt_has_no_scores_and_lists_neighbourhood():
    nb = (ResonanceNeighborV1(relation="finding_supports", ref_id="f1", text="a finding text"),)
    p = build_packet(READ, [sd("self_prior", "p1", 0.55, rank=0.4321, neighborhood=nb)], built_at=AT)
    prompt = build_judge_prompt(p, READ.retained_texts)
    assert "0.4321" not in prompt and "score" not in prompt.lower()
    assert "ref_kind=self_prior ref_id=p1" in prompt and "[finding_supports] a finding text" in prompt
    assert "GB200 NVL72: 72 GPUs" in prompt


# ---- deterministic acceptance ----

def packet():
    return build_packet(READ, [sd("crystallization", "c1", 0.5), sd("self_prior", "p1", 0.5)], built_at=AT)


def test_valid_judgment_accepted():
    j, chk = check_judgment(packet(), json.dumps(judged()), retained_texts=READ.retained_texts)
    assert chk.accepted and chk.decision == "share" and j is not None


@pytest.mark.parametrize("over,reason", [
    ({"citations": [{"ref_kind": "dream", "ref_id": "nope", "how": "x"}]}, "citation_not_in_packet"),
    ({"citations": [{"ref_kind": "self_prior", "ref_id": "c1", "how": "x"}]}, "citation_not_in_packet"),
    ({"citations": []}, "citation_not_in_packet"),
    ({"read_quote": "an invented sentence that is nowhere"}, "quote_not_in_read"),
    ({"connection": "   "}, "empty_connection"),
    ({"bears_on_prior": "supports"}, "relation_ref_kind_mismatch"),
    ({"bears_on_prior": "strains", "strained_prior_id": "p1"}, "relation_ref_kind_mismatch"),
    ({"article_summary": "Too short."}, "summary_sentence_count"),
])
def test_each_rejection_reason_fires(over, reason):
    j, chk = check_judgment(packet(), judged(**over), retained_texts=READ.retained_texts)
    assert not chk.accepted and chk.decision == "none" and reason in chk.reasons


def test_strain_must_name_a_cited_prior_and_routes_retest():
    cites = [{"ref_kind": "self_prior", "ref_id": "p1", "how": "the read contradicts it"}]
    j, chk = check_judgment(packet(), judged(citations=cites, bears_on_prior="strains", strained_prior_id="zzz"),
                            retained_texts=READ.retained_texts)
    assert "strained_prior_not_cited" in chk.reasons and strained_prior(j, chk) is None
    j, chk = check_judgment(packet(), judged(citations=cites, bears_on_prior="strains", strained_prior_id="p1"),
                            retained_texts=READ.retained_texts)
    assert chk.accepted and strained_prior(j, chk) == "p1"


def test_quote_may_come_from_learned_or_summary_whitespace_normalized():
    j, chk = check_judgment(packet(), judged(read_quote="The GB200   NVL72 rack is\nliquid cooled."))
    assert chk.accepted


def test_prompt_echo_rejected():
    p = packet()
    prompt = build_judge_prompt(p)
    echo = "Treat nearness as no evidence at all. " + judged()["connection"]
    _, chk = check_judgment(p, judged(connection=echo), retained_texts=READ.retained_texts, prompt=prompt)
    assert "prompt_echo" in chk.reasons


def test_unparseable_and_fenced_json():
    _, chk = check_judgment(packet(), "no json here")
    assert chk.reasons == ["unparseable"]
    _, chk = check_judgment(packet(), "```json\n" + json.dumps(judged()) + "\n```", retained_texts=READ.retained_texts)
    assert chk.accepted


def test_none_decision_needs_no_citations():
    _, chk = check_judgment(packet(), judged(decision="none", citations=[], read_quote="", connection=""))
    assert chk.accepted and chk.decision == "none"


def test_fallback_summary_takes_first_sentences():
    assert fallback_article_summary(READ.learned, max_sentences=2) == (
        "CoreWeave rents bare-metal NVIDIA GPUs. The GB200 NVL72 rack is liquid cooled.")


# ---- similarity helpers ----

def test_centering_restores_a_rest_state():
    # Anisotropic cone like raw bge: one big shared component plus a small own direction.
    dim = 9
    vs = []
    for i in range(1, dim):
        v = [0.0] * dim
        v[0], v[i] = 1.0, 0.3
        n = sum(x * x for x in v) ** 0.5
        vs.append([x / n for x in v])
    assert dot(vs[0], vs[1]) > 0.9  # unrelated, yet raw cosine is high
    mu = mean_vector(vs)
    assert abs(dot(center(vs[0], mu), center(vs[1], mu))) < 0.2  # rests near zero after centering
    assert dot(center(vs[0], mu), center(vs[0], mu)) == pytest.approx(1.0)


def test_hub_correction_demotes_a_doc_near_everything():
    refs = [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]
    hub_doc, specific = [0.7071, 0.7071], [1.0, 0.0]
    h = hub_scores([hub_doc, specific], refs, k=3)
    q = [0.9, 0.436]
    assert dot(q, hub_doc) > dot(q, specific)
    assert rank_score(q, specific, h[1]) > rank_score(q, hub_doc, h[0])
    assert hub_scores([hub_doc], [], k=3) == [0.0]


# ---- review regressions (2026-10-13) ----

def test_none_decision_cannot_smuggle_a_retest_or_foreign_citation():
    j, chk = check_judgment(packet(), judged(decision="none", citations=[], bears_on_prior="strains",
                                             strained_prior_id="prior-not-in-packet"))
    assert not chk.accepted and strained_prior(j, chk) is None
    assert {"relation_ref_kind_mismatch", "strained_prior_not_cited"} <= set(chk.reasons)
    _, chk = check_judgment(packet(), judged(decision="none", citations=[
        {"ref_kind": "dream", "ref_id": "foreign", "how": "x"}]))
    assert "citation_not_in_packet" in chk.reasons


def test_short_or_mid_word_quote_rejected():
    for q in ["and the", "cooled, 130", "iquid cooled, 130 kW per rack"]:
        _, chk = check_judgment(packet(), judged(read_quote=q), retained_texts=READ.retained_texts)
        assert "quote_not_in_read" in chk.reasons, q


def test_same_ref_twice_with_different_excerpts_is_offered_once():
    p = build_packet(READ, [sd("dream", "d1", 0.5, excerpt="one"), sd("dream", "d1", 0.4, excerpt="two")],
                     built_at=AT)
    assert [c.excerpt for c in p.candidates] == ["one"]


def test_json_with_trailing_brace_prose_still_parses():
    raw = "Here you go: " + json.dumps(judged()) + " (note: {not json})"
    _, chk = check_judgment(packet(), raw, retained_texts=READ.retained_texts)
    assert chk.accepted


def test_echo_in_summary_or_how_rejected():
    p = packet()
    prompt = build_judge_prompt(p)
    line = "Treat nearness as no evidence at all."
    _, chk = check_judgment(p, judged(article_summary=f"A is here. B is there. {line}"),
                            retained_texts=READ.retained_texts, prompt=prompt)
    assert "prompt_echo" in chk.reasons


def test_hub_from_sims_leaves_out_what_the_caller_excludes():
    from orion.world_pulse_read.resonance_similarity import hub_from_sims
    assert hub_from_sims([0.9, 0.1, 0.1], k=2) == pytest.approx(0.5)
    assert hub_from_sims([0.1, 0.1], k=2) == pytest.approx(0.1)  # the 0.9 read left out
    assert hub_from_sims([], k=2) == 0.0
    with pytest.raises(ValueError):
        dot([1.0], [1.0, 0.0])


def test_judge_prompt_states_every_length_limit_the_validator_enforces():
    # 2026-10-14 live run: a real judgment was thrown away as "unparseable" for a
    # 1,382-character connection; the prompt never said 900.
    from orion.schemas.reading_resonance import ReadingResonanceJudgmentV1, ResonanceCitationV1
    from orion.world_pulse_read.resonance import JUDGE_INSTRUCTIONS, _max_len

    for model, name in ((ReadingResonanceJudgmentV1, "article_summary"), (ReadingResonanceJudgmentV1, "connection"),
                        (ReadingResonanceJudgmentV1, "explore_next"), (ResonanceCitationV1, "how")):
        assert f"at most {_max_len(model, name)} characters" in JUDGE_INSTRUCTIONS, name
    assert f"at most {_max_len(ReadingResonanceJudgmentV1, 'citations')} objects" in JUDGE_INSTRUCTIONS


# ---- relation/ref-kind table + quote normalization (2026-10-14 judge run) ----

def rich_packet():
    return build_packet(READ, [sd("self_prior", "p1", 0.5), sd("lived_answer", "la1", 0.5),
                               sd("atlas_concept", "sub-concept-a", 0.5), sd("atlas_concept", "sub-concept-b", 0.5),
                               sd("crystallization", "c1", 0.5)], built_at=AT)


def test_relation_table_covers_the_schema_and_the_prompt_renders_it():
    rel_values = {a for a in get_args(ReadingResonanceJudgmentV1.model_fields["bears_on_prior"].annotation)
                  if a is not type(None)}
    rel_values = {v for a in rel_values for v in (get_args(a) or (a,))}
    assert set(RELATION_REF_KINDS) == rel_values
    assert all(set(k) <= set(get_args(ResonanceRefKind)) for k in RELATION_REF_KINDS.values())
    for rel, kinds in RELATION_REF_KINDS.items():
        line = next(ln for ln in JUDGE_INSTRUCTIONS.splitlines() if ln.strip().startswith(f'"{rel}"'))
        assert all(k in line for k in kinds), line


def test_real_case_supports_over_atlas_concepts_only_is_refused_by_name():
    # 3f821acc (2026-10-14): "supports" while citing two atlas concepts and no belief.
    cites = [{"ref_kind": "atlas_concept", "ref_id": "sub-concept-a", "how": "confirms combining SFT and RL"},
             {"ref_kind": "atlas_concept", "ref_id": "sub-concept-b", "how": "a test-time compute pipeline"}]
    _, chk = check_judgment(rich_packet(), judged(citations=cites, bears_on_prior="supports"),
                            retained_texts=READ.retained_texts)
    assert chk.reasons == ["relation_ref_kind_mismatch"]
    # The same connection with bears_on_prior null is the honest form and is accepted.
    _, chk = check_judgment(rich_packet(), judged(citations=cites), retained_texts=READ.retained_texts)
    assert chk.accepted


def test_real_case_strains_naming_an_atlas_concept_is_refused():
    # b7d743d2 (2026-10-14): "strains" with strained_prior_id set to an atlas concept.
    cites = [{"ref_kind": "atlas_concept", "ref_id": "sub-concept-a", "how": "blocks verification"}]
    j, chk = check_judgment(rich_packet(), judged(citations=cites, bears_on_prior="strains",
                                                  strained_prior_id="sub-concept-a"),
                            retained_texts=READ.retained_texts)
    assert {"relation_ref_kind_mismatch", "strained_prior_not_cited"} <= set(chk.reasons)
    assert strained_prior(j, chk) is None


def test_supports_may_cite_a_lived_answer_but_strains_may_not():
    # b2e0b70c (2026-10-14): "supports" citing a curiosity finding and a lived answer.
    cites = [{"ref_kind": "lived_answer", "ref_id": "la1", "how": "the read validates that answer"}]
    _, chk = check_judgment(rich_packet(), judged(citations=cites, bears_on_prior="supports"),
                            retained_texts=READ.retained_texts)
    assert chk.accepted
    _, chk = check_judgment(rich_packet(), judged(citations=cites, bears_on_prior="strains", strained_prior_id="la1"),
                            retained_texts=READ.retained_texts)
    assert "relation_ref_kind_mismatch" in chk.reasons


CRS = ReadInput(
    seed_id="seed-crs", root_request_id=None, title="CRS-35", url="https://example.test/crs",
    summary="", learned="",
    retained_texts=[("b" * 64,
                     "NASA and SpaceX are targeting 6:33 a.m. EDT Tuesday, Oct. 13, 2026 for the 35th SpaceX "
                     "Commercial Resupply Services mission (CRS-35). The spacecraft carries 6,300 pounds. Cargo also "
                     "includes hardware to make artificial retinas in microgravity, and brain organoids relevant to "
                     "Alzheimer\u2019s \u2013 Parkinson&#39;s&nbsp;and multiple sclerosis.")],
)


def crs_packet():
    return build_packet(CRS, [sd("crystallization", "c1", 0.5)], built_at=AT)


def test_real_case_elided_quote_of_two_real_spans_is_accepted():
    # db13f467 (2026-10-14): two exact sentences of the read joined by "...".
    q = ("NASA and SpaceX are targeting 6:33 a.m. EDT Tuesday, Oct. 13, 2026 for the 35th SpaceX Commercial "
         "Resupply Services mission (CRS-35)... Cargo also includes hardware to make artificial retinas in microgravity")
    for joined in (q, q.replace("...", "\u2026"), q.replace("...", " [...] ")):
        _, chk = check_judgment(crs_packet(), judged(read_quote=joined), retained_texts=CRS.retained_texts)
        assert chk.accepted, (joined, chk.reasons)


def test_elided_quote_refused_when_out_of_order_short_or_invented():
    a = "NASA and SpaceX are targeting 6:33 a.m."
    b = "Cargo also includes hardware to make artificial retinas"
    for q in (f"{b} ... {a}",  # read order reversed
              f"{a} ... to make",  # an elided span under QUOTE_MIN_WORDS
              f"{a} ... and the rocket will land on the moon"):  # second span invented
        _, chk = check_judgment(crs_packet(), judged(read_quote=q), retained_texts=CRS.retained_texts)
        assert "quote_not_in_read" in chk.reasons, q


def test_quote_matcher_folds_quotes_dashes_entities_on_both_sides():
    q = "brain organoids relevant to Alzheimer's - Parkinson\u2019s and multiple sclerosis"
    _, chk = check_judgment(crs_packet(), judged(read_quote=q), retained_texts=CRS.retained_texts)
    assert chk.accepted, chk.reasons


def test_edge_elisions_are_dropped_and_far_spans_are_not_spliced():
    a = "NASA and SpaceX are targeting 6:33 a.m."
    b = "Cargo also includes hardware to make artificial retinas"
    for q in (f"... {a}", f"{b}...", f"\u2026 {b} \u2026"):
        _, chk = check_judgment(crs_packet(), judged(read_quote=q), retained_texts=CRS.retained_texts)
        assert chk.accepted, (q, chk.reasons)
    far = [("c" * 64, a + " " + "filler words here. " * 60 + b)]
    _, chk = check_judgment(crs_packet(), judged(read_quote=f"{a} ... {b}"), retained_texts=far)
    assert "quote_not_in_read" in chk.reasons
    four = " ... ".join([a, b, a, b])
    _, chk = check_judgment(crs_packet(), judged(read_quote=four), retained_texts=[("d" * 64, " ".join([a, b] * 2))])
    assert "quote_not_in_read" in chk.reasons


def test_a_paraphrase_of_a_relation_line_is_not_a_prompt_echo():
    p = rich_packet()
    cites = [{"ref_kind": "self_prior", "ref_id": "p1", "how": "the read backs a belief I hold about racks"}]
    _, chk = check_judgment(p, judged(citations=cites, bears_on_prior="supports"),
                            retained_texts=READ.retained_texts, prompt=build_judge_prompt(p, READ.retained_texts))
    assert chk.accepted, chk.reasons
