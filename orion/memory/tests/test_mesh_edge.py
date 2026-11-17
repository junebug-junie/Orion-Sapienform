"""Issue #3409: the mesh-edge check refuses public text that quotes a core-page or lived-memory
sentence verbatim. Made-up text only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.memory import mesh_edge
from orion.memory.mesh_edge import (
    MeshEdgeGuard,
    MeshEdgeLeak,
    MeshEdgeUnavailable,
    check_outbound,
    guard_texts_from_rows,
    normalize,
)

PAGE = (
    "Robin is my oldest friend and keeps bees on the roof [episode_memory:abc]. "
    "She hates cold coffee. Robin told me the blue kite got stuck in the old oak tree."
)
MEMORY = "Juniper told me her neighbour's cat learned to open the pantry door last spring."


def _guard() -> MeshEdgeGuard:
    rows = [{"concept_id": "page:juniper", "content": PAGE}]
    recs = [SimpleNamespace(doc_id="episode_memory:11111111-2222", text=MEMORY)]
    return MeshEdgeGuard.from_texts(guard_texts_from_rows(rows, recs))


def test_normalize_folds_quotes_case_punctuation_and_whitespace():
    assert normalize("She’s  \"HERE\",\n  now!") == "shes here now"


def test_a_verbatim_page_sentence_is_refused_even_reformatted():
    body = "Release notes.\n\n> ROBIN TOLD ME the blue kite got   stuck in the old oak tree!\n"
    with pytest.raises(MeshEdgeLeak) as err:
        _guard().check(body, surface="github")
    assert err.value.hits[0].source == "page:juniper"
    # The refusal names the source, never the sentence.
    assert "kite" not in str(err.value)


def test_a_verbatim_lived_memory_sentence_is_refused():
    hits = _guard().find("As noted: Juniper told me her neighbour's cat learned to open the pantry door last spring.")
    assert [h.source for h in hits] == ["episode_memory:11111111-2222"]


def test_short_sentences_and_paraphrases_pass():
    guard = _guard()
    assert "she hates cold coffee" not in guard.phrases          # 4 words: under MIN_WORDS
    assert guard.find("Fixed the bee-keeping roof sensor; a kite was stuck in a tree.") == []
    assert guard.find("She hates cold coffee.") == []


def test_stored_cites_are_stripped_before_matching():
    assert _guard().find("Robin is my oldest friend and keeps bees on the roof.")


def test_match_needs_word_boundaries():
    guard = MeshEdgeGuard.from_texts([("page:anchor", "alpha beta gamma delta epsilon zeta")])
    assert guard.find("xalpha beta gamma delta epsilon zeta") == []
    assert guard.find("(alpha beta gamma delta epsilon zeta)")


def test_check_outbound_fails_closed_without_a_dsn(monkeypatch):
    for key in mesh_edge.DSN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MeshEdgeUnavailable):
        check_outbound("anything at all", surface="github")


def test_load_guard_wraps_driver_errors_as_unavailable(monkeypatch):
    def boom(_dsn):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mesh_edge, "_load_sync", boom)
    with pytest.raises(MeshEdgeUnavailable):
        mesh_edge.load_guard("postgresql://nobody@127.0.0.1:1/x", ttl_sec=0)


@pytest.mark.parametrize("encoded", [
    "Robin doesn'\\''t like loud noises in the morning at all.",      # shell quoting
    "Robin doesn\\u2019t like loud noises in the morning at all.",    # JSON escape
    "Robin doesn&#39;t like loud noises in the morning at all.",      # HTML entity
    "Robin doesn’t like loud\\nnoises in the morning at all.",        # curly quote + escaped newline
])
def test_encoded_apostrophes_still_match(encoded):
    guard = MeshEdgeGuard.from_texts([("episode_memory:x", "Robin doesn't like loud noises in the morning at all.")])
    assert guard.find(f"body={encoded}")


def test_guard_sql_covers_retired_rows_and_old_page_versions():
    assert "DISTINCT ON" not in mesh_edge.CORE_PAGES_SQL
    for table in ("episode_memory", "memory_crystallizations", "person_fact"):
        assert table in mesh_edge.GUARD_MEMORY_SQL
    assert "status" not in mesh_edge.GUARD_MEMORY_SQL and "removed_at" not in mesh_edge.GUARD_MEMORY_SQL
