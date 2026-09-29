"""What recall searches for during a reading turn (recall retrieval design phase 3, 2026-09-29)."""

from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.world_pulse_read.documents import document_ref
from orion.world_pulse_read.durable import reading_retrieval_query


def _seed(**over):
    kw = dict(seed_id="s", kind="finding", run_id="r", url="https://ex.com/a", title="Chip packaging")
    kw.update(over)
    return WorldPulseReadSeedV1(**kw)


def test_stage2_is_title_dash_claim():
    assert reading_retrieval_query(_seed(), "Advanced packaging is the new bottleneck.") == (
        "Chip packaging — Advanced packaging is the new bottleneck."
    )


def test_stage1_has_no_claim_yet_so_title_only():
    assert reading_retrieval_query(_seed()) == "Chip packaging"


def test_untitled_url_seed_falls_back_to_url():
    assert reading_retrieval_query(_seed(title="")) == "https://ex.com/a"


def test_untitled_document_seed_does_not_search_an_opaque_ref():
    seed = _seed(title="", url=document_ref("/data/docs/paper.pdf", "a" * 64))
    assert reading_retrieval_query(seed) is None
    assert reading_retrieval_query(seed, "claim") == "claim"


def test_capped_at_1000_chars_and_whitespace_collapsed():
    q = reading_retrieval_query(_seed(title="  Chip\n packaging "), "x" * 5000)
    assert q.startswith("Chip packaging — xx")
    assert len(q) == 1000


def test_brief_carries_it_and_old_briefs_still_validate():
    brief = ReadingRunBriefV1(seed_id="s", stage=2, prompt="p", session_id="x", timeout_sec=1.0, retrieval_query="q")
    assert brief.retrieval_query == "q"
    old = {"seed_id": "s", "stage": 1, "prompt": "p", "session_id": "x", "timeout_sec": 1.0}
    assert ReadingRunBriefV1.model_validate(old).retrieval_query is None
