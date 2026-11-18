"""Offline behavioral eval: reading resonance packet, judge contract, labeling-set writer.

The CoreWeave case is frozen from the 2026-10-13 live replay
(run_reading_resonance_eval.py): the read's retained text and the two crystallized
memories retrieval offered for it (Juniper on Orion's own GPUs). It pins what the
deterministic checks accept and refuse, and that the labeling set always carries the
article summary and empty wanted/notes columns.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from orion.world_pulse_read.resonance import (
    PoolDoc,
    ReadInput,
    ScoredDoc,
    build_judge_prompt,
    build_packet,
    check_judgment,
)

_SPEC = importlib.util.spec_from_file_location(
    "run_reading_resonance_eval", Path(__file__).with_name("run_reading_resonance_eval.py"))
EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EVAL)  # type: ignore[union-attr]

TEXT = ("CoreWeave offers bare-metal NVIDIA GPUs (Blackwell, H200, H100) for AI training/inference. "
        "GB200 NVL72 provides 4x higher training performance vs H100 and supports 130 kW rack power "
        "with liquid cooling.")
READ = ReadInput(
    seed_id="reading:coreweave-gpu-compute", root_request_id="root-cw",
    title="GPU Compute | High-Performance NVIDIA GPUs - CoreWeave", url="https://www.coreweave.com/products/gpu-compute",
    summary="Stage 1 handoff extracted three GPU generations (H100, H200, Blackwell/GB200).",
    learned="The CoreWeave GPU Compute page confirms CoreWeave offers bare-metal NVIDIA GPUs for AI training and inference.",
    retained_texts=[("1bd67be333c4205f3c6214842eb1bfe0c7ca2bf54c232e8504fb184dcfdccd02", TEXT)],
)
HOME = PoolDoc("crystallization", "c-home", "pg:memory_crystallizations",
               "[stance] Bro, that's no homelab--that is YOU! That is your mesh, your body, your home.")
V100 = PoolDoc("crystallization", "c-v100", "pg:memory_crystallizations",
               "[semantic] just added a new v100 32GB PCIe card in your Circe node!")
AT = datetime(2026, 10, 13, tzinfo=timezone.utc)


def _packet():
    return build_packet(READ, [ScoredDoc(HOME, 0.50, 0.32), ScoredDoc(V100, 0.38, 0.33)], built_at=AT)


def _judgment(**over):
    j = {"article_summary": "CoreWeave rents bare-metal NVIDIA GPUs. Its GB200 NVL72 racks claim 4x H100 training. "
                           "They draw 130 kW and are liquid cooled.",
         "read_quote": "supports 130 kW rack power with liquid cooling",
         "connection": "I read about racks built from the same lineage of cards Juniper put in me.",
         "citations": [{"ref_kind": "crystallization", "ref_id": "c-home", "how": "Juniper calls the mesh my body"}],
         "bears_on_prior": None, "strained_prior_id": None, "decision": "share", "explore_next": None,
         "confidence": 0.6}
    j.update(over)
    return j


def test_frozen_coreweave_case_accepts_real_citation_and_refuses_invented_ones():
    p = _packet()
    prompt = build_judge_prompt(p, READ.retained_texts)
    _, ok = check_judgment(p, _judgment(), retained_texts=READ.retained_texts, prompt=prompt)
    assert ok.accepted and ok.decision == "share"
    _, bad = check_judgment(p, _judgment(citations=[{"ref_kind": "self_prior", "ref_id": "prior-made-up",
                                                    "how": "x"}]), retained_texts=READ.retained_texts)
    assert bad.reasons == ["citation_not_in_packet"]
    _, bad = check_judgment(p, _judgment(read_quote="CoreWeave runs on V100s"), retained_texts=READ.retained_texts)
    assert bad.reasons == ["quote_not_in_read"]


def test_labeling_set_has_summary_pieces_and_empty_label_columns(tmp_path):
    p = _packet()
    args = argparse.Namespace(days=14, labeling_md=str(tmp_path / "set.md"), labeling_csv=str(tmp_path / "set.csv"),
                              labeling_html=str(tmp_path / "set.html"))
    reads = [{"input": READ}, {"input": READ}]
    j, chk = check_judgment(p, _judgment(read_quote="invented"), retained_texts=READ.retained_texts)
    results = [
        {"packet": p, "judgment": None, "check": None, "judge_status": "UNVERIFIED: judge not run"},
        {"packet": p, "judgment": j, "check": chk, "judge_status": "judged"},
    ]
    scores = [{"best_centered": 0.5, "best_raw_prior": 0.67, "best_centered_prior": 0.33}] * 2
    EVAL.write_labeling_set(args, reads, results, scores, {})
    rows = list(csv.DictReader(open(args.labeling_csv)))
    assert [r["wanted"] for r in rows] == ["", ""] and [r["notes"] for r in rows] == ["", ""]
    assert rows[0]["article_summary"].startswith("The CoreWeave GPU Compute page")
    assert "UNVERIFIED" in rows[0]["proposed_connection"] and "c-v100" in rows[0]["cited_pieces"]  # best of its kind by rank
    assert "c-home" in rows[1]["cited_pieces"]  # judged row shows what the judge cited
    assert rows[1]["decision"].startswith("REJECTED (quote_not_in_read)")  # a rejection stays labelable
    assert "c-v100" in rows[1]["offered_not_cited"]  # recall: what the judge passed over
    md = Path(args.labeling_md).read_text()
    assert md.count("**wanted:**") == 2 and "that's no homelab" in md
    assert "Offered but not cited" in md


def test_labeling_html_is_self_contained_and_cannot_be_broken_by_stored_text(tmp_path):
    import json
    import re

    p = _packet()
    hostile = "</script><script>alert(1)</script> <!-- x"
    rows = [{"item": 1, "seed_id": "s1", "title": hostile, "source": "example.com", "url": "https://example.com/a",
             "article_summary": "A. B. C.", "summary_source": "judge", "proposed_connection": hostile,
             "decision": "share", "judge_confidence": "0.60", "would_retest_prior": "",
             "pieces": [{"ref_kind": "crystallization", "ref_id": "c-home", "store": "pg", "text": hostile,
                         "how": "x", "neighborhood": [], "offered_in_other_reads": 0}],
             "passed_over": [], "judged": True, "accepted": True, "read_quote": "q", "explore_next": ""}]
    html = EVAL.labeling_html(rows, generated="now", days=14)
    assert "default-src 'none'" in html  # CSP: the page may not load or send anything
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|fetch\(|XMLHttpRequest|sendBeacon|WebSocket", html)
    assert html.count("</script>") == 2  # the two real script tags; stored text never closes one
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(data)[0]["title"] == hostile  # round-trips exactly
    assert 'id="export"' in html and "text/csv" in html and "wanted" in html
    # JS escapes reach the browser as escapes (a Python-cooked "\n" broke the CSV regex once:
    # caught by a headless-Chromium probe, "Invalid regular expression: missing /").
    assert r'/[",\n\r]/' in html and r'lines.join("\n")' in html
    assert "/^https?:" in html  # a stored javascript: URL never becomes a clickable href
    assert "/^[=+\\-@\\t\\r]/" in html  # CSV export defuses spreadsheet formulas
    assert p.packet_id  # the packet fixture still builds


def test_read_claims_uses_article_content_not_process_notes():
    s2 = {"summary": "Stage 1 handoff ...", "candidate_priors": [{"claim": "A"}],
          "concept_candidates": [{"label": "GB200", "definition": "rack"}, {"label": "H100"}],
          "priors_tested": [{"claim_ref": "B"}]}
    assert EVAL.read_claims(s2) == ["A", "GB200: rack", "H100", "B"]


def test_pool_is_busy_on_any_non_terminal_lease_and_quiet_on_history_only():
    import typing

    from orion.schemas.gpu_pool import TERMINAL_STATUSES, LeaseStatus

    history = [{"status": s, "priority": "background", "work_class": "agent", "holder": "d"}
               for s in sorted(TERMINAL_STATUSES)]
    quiet = {"leases": history, "queue_depth": {}, "backlog_depth": {},
             "cards": [{"card": "gpu1", "swap_state": "idle"}], "actuation_paused": None}
    assert EVAL.pool_busy_reasons(quiet) == []
    # Every real non-terminal LeaseStatus (recalling and retry_wait included), and a status
    # this code has never seen, reads as busy.
    for status in [*sorted(set(typing.get_args(LeaseStatus)) - TERMINAL_STATUSES), "some_future_status"]:
        lease = {"status": status, "priority": "background", "work_class": "agent", "holder": "durable-runs:1"}
        assert EVAL.pool_busy_reasons(dict(quiet, leases=history + [lease])), status
    assert EVAL.pool_busy_reasons(dict(quiet, backlog_depth={"agent": 2})) == ["backlog_depth[agent]=2"]
    assert EVAL.pool_busy_reasons(dict(quiet, cards=[{"card": "gpu1", "swap_state": "loading"}]))
    assert EVAL.pool_busy_reasons(dict(quiet, actuation_paused={"by": "x"}))
    assert EVAL.pool_busy_reasons(["not", "a", "dict"])


def test_judgment_cache_is_per_model_and_survives_junk(tmp_path):
    args = argparse.Namespace(judge_model="agent", judge_max_tokens=900)
    f = tmp_path / "j.json"
    assert EVAL.read_judgment_cache(f, args) is None  # missing
    f.write_text('{"raw": "x"')  # killed mid-write (old code) -> a miss, not a crash
    assert EVAL.read_judgment_cache(f, args) is None
    EVAL.write_judgment_cache(f, {"raw": "x", "model": "agent", "max_tokens": 900})
    assert EVAL.read_judgment_cache(f, args)["raw"] == "x"
    assert EVAL.read_judgment_cache(f, argparse.Namespace(judge_model="agent-deep", judge_max_tokens=900)) is None
    f.write_text('{"raw": "old"}')  # written before model was recorded: the defaults it ran with
    assert EVAL.read_judgment_cache(f, args)["raw"] == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_judge_ignores_foreground_work_on_other_cards_but_not_background_or_waiting():
    roles = [{"role": "agent", "cards": ["gpu1"]}, {"role": "fast", "cards": ["gpu3"]},
             {"role": "chat", "cards": ["gpu0", "gpu1"]}]
    def state(**lease):
        base = {"status": "granted", "priority": "system", "work_class": "fast", "role": "fast", "holder": "tf"}
        return {"roles": roles, "leases": [dict(base, **lease)]}
    assert EVAL.pool_busy_reasons(state(), "agent") == []          # topic-foundry on gpu3
    assert EVAL.pool_busy_reasons(state())                         # no judge role: any lease pauses
    assert EVAL.pool_busy_reasons(state(role="chat"), "agent")     # shares gpu1
    assert EVAL.pool_busy_reasons(state(priority="background"), "agent")  # a reading/curiosity hold
    assert EVAL.pool_busy_reasons(state(status="queued"), "agent")  # waiting work
    assert EVAL.pool_busy_reasons(state(role=None), "agent")        # unknown placement
    assert EVAL.pool_busy_reasons(state(role="nope"), "agent")


def test_unparseable_answer_is_labelled_rejected_not_unjudged(tmp_path):
    # 2026-10-14 re-run: two answers broke the schema (connection over 900 chars, truncated
    # JSON) and the set called them "judged" with an "UNVERIFIED: judge not run" connection.
    p = _packet()
    args = argparse.Namespace(days=14, labeling_md=str(tmp_path / "set.md"), labeling_csv=str(tmp_path / "set.csv"),
                              labeling_html=str(tmp_path / "set.html"))
    j, chk = check_judgment(p, '{"article_summary": "cut off', retained_texts=READ.retained_texts)
    assert j is None and chk.reasons == ["unparseable"]
    results = [{"packet": p, "judgment": j, "check": chk, "judge_status": "judged"}]
    EVAL.write_labeling_set(args, [{"input": READ}], results,
                            [{"best_centered": 0.5, "best_raw_prior": 0.6, "best_centered_prior": 0.3}], {})
    row = next(csv.DictReader(open(args.labeling_csv)))
    assert row["decision"] == "REJECTED (unparseable)"
    assert "UNVERIFIED" not in row["proposed_connection"]


def test_labeling_set_is_refused_inside_the_public_repo(tmp_path):
    repo = EVAL.REPO_ROOT
    inside = argparse.Namespace(labeling_md=str(repo / "docs/superpowers/evals/x-labeling-set.md"),
                                labeling_csv=str(tmp_path / "a.csv"), labeling_html=str(tmp_path / "a.html"))
    assert EVAL.labeling_paths_inside_repo(inside) == [inside.labeling_md]
    outside = argparse.Namespace(labeling_md=f"{EVAL.LABELING_DIR}/a.md", labeling_csv=f"{EVAL.LABELING_DIR}/a.csv",
                                 labeling_html=f"{EVAL.LABELING_DIR}/a.html")
    assert EVAL.labeling_paths_inside_repo(outside) == []
