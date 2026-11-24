"""Reveries that ask, Patch 1: gate tests for the spec's acceptance checks 2-6 and 8
(docs/superpowers/specs/2026-11-18-reverie-questions-design.md).

2 quiet gate (pure); 3 each seed reader from a fixture; 4 refusals (bad ref, near-duplicate, machinery
words); 5 focus routing (cap, origin, keep-pending: Hub half in services/orion-hub/tests/test_curiosity_focus.py);
6 far memory never serves a superseded memory; 8 the retired producers are gone (static).
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]


def seeds():
    return importlib.import_module("app.reverie_seeds")


def rev():
    return importlib.import_module("app.reverie")


def gate(**kw):
    s = seeds()
    base = dict(now=NOW, arousal_level="idle", last_winner_at=NOW - timedelta(minutes=20),
                latest_tick_at=NOW - timedelta(seconds=20), rows_since_stretch=0, rows_last_24h=0)
    base.update(kw)
    return s.quiet_gate(s.QuietGateInputs(**base), min_calm_sec=600, max_per_day=8)


# --- 2. the quiet gate -----------------------------------------------------------------------------


def test_gate_opens_when_idle_and_calm_long_enough():
    d = gate()
    assert d.open and d.reason == "open" and d.calm_sec == 1200
    assert d.stretch_start == NOW - timedelta(minutes=20)


def test_gate_shut_when_engaged_even_if_calm():
    assert gate(arousal_level="engaged").reason == "arousal_not_idle"
    assert gate(arousal_level="strained").reason == "arousal_not_idle"


def test_gate_shut_when_arousal_unknown():
    assert gate(arousal_level=None).reason == "arousal_unknown"
    assert gate(arousal_level="unknown").reason == "arousal_unknown"


def test_gate_shut_when_idle_but_something_won_attention_recently():
    d = gate(last_winner_at=NOW - timedelta(seconds=30))
    assert not d.open and d.reason == "calm_too_short"


def test_gate_shut_for_a_short_calm_stretch():
    d = gate(last_winner_at=NOW - timedelta(minutes=9, seconds=59))
    assert not d.open and d.reason == "calm_too_short"


def test_gate_shut_when_the_broadcast_is_dead_not_calm():
    assert gate(latest_tick_at=NOW - timedelta(minutes=6)).reason == "broadcast_stale"
    assert gate(latest_tick_at=None).reason == "broadcast_stale"


def test_one_reverie_per_stretch_and_daily_cap_and_off_switch():
    assert gate(rows_since_stretch=1).reason == "stretch_spent"
    assert gate(rows_last_24h=8).reason == "daily_cap"
    s = seeds()
    off = s.quiet_gate(s.QuietGateInputs(now=NOW, arousal_level="idle", last_winner_at=None,
                                         latest_tick_at=NOW, rows_since_stretch=0, rows_last_24h=0),
                       min_calm_sec=600, max_per_day=0)
    assert off.reason == "disabled"


def test_three_lane_refusals_give_up_the_stretch():
    assert gate(lane_refusals_this_stretch=2).open
    assert gate(lane_refusals_this_stretch=3).reason == "lane_refused_stretch"


def test_no_winner_in_a_day_starts_the_stretch_at_the_lookback_edge():
    d = gate(last_winner_at=None)
    assert d.open and d.stretch_start == NOW - seeds().STRETCH_LOOKBACK


# --- 3. seed readers from fixtures -----------------------------------------------------------------


def _miss(eid, seen, label="Neighbor A"):
    start = datetime(2026, 11, 24, 14, 0, tzinfo=timezone.utc)  # 07:00 Denver
    return {"expectation_id": eid, "stream_id": "walkway", "subject_key": f"individual:{eid}", "label": label,
            "day_kind": "weekday", "window_start": start, "window_end": start + timedelta(hours=1),
            "support_days": 14, "persons_seen": seen}


def test_absence_clean_and_unrecognised_are_worded_apart():
    out = seeds().absence_seeds([_miss("e1", 0), _miss("e2", 5)], tz_name="America/Denver", spent_refs=set())
    clean, unrec = out
    assert clean.evidence == "clean" and clean.refs == ["vision_percept_expectation:e1"]
    assert "between 07:00 and 08:00 on weekdays (14 days of support)" in clean.material
    assert "nobody passed" in clean.material
    assert unrec.evidence == "unrecognised"
    assert "detected a person in 5 frame(s)" in unrec.material and "Whether Neighbor A came is unknown" in unrec.material
    # Never phrased as an absence: no "did not come", "didn't come", "nobody passed".
    for words in ("did not come", "didn't come", "nobody passed", "no person at all"):
        assert words not in unrec.material
    assert clean.thread_key == "absence:individual:e1"


def test_absence_skips_spent_refs_and_unnamed_rows():
    out = seeds().absence_seeds([_miss("e1", 0), _miss("e2", 0, label="  ")], tz_name="America/Denver",
                                spent_refs={"vision_percept_expectation:e1"})
    assert out == []


def _bin(faculty, b, n, p, rate):
    return {"faculty": faculty, "bin": b, "n": n, "mean_p": p, "rate": rate}


def test_lost_bet_only_real_calibration_gaps():
    rows = [
        _bin("more_steps_than_usual", 1, 60, 0.15, 0.37),  # gap 0.22, se ~0.046 -> gap
        _bin("more_steps_than_usual", 8, 60, 0.84, 0.54),  # gap 0.30 -> gap
        _bin("more_steps_than_usual", 5, 60, 0.55, 0.58),  # calibrated
        _bin("asks_a_question", 1, 19, 0.15, 0.60),  # n < 20: one bad roll is not a gap
        _bin("asks_a_question", 3, 400, 0.35, 0.45),  # gap 0.10 < 0.15
    ]
    out = seeds().lost_bet_seeds(rows, spent_refs=set())
    assert [s.refs[0] for s in out] == ["self_bet_bin:more_steps_than_usual:0.8-0.9",
                                        "self_bet_bin:more_steps_than_usual:0.1-0.2"]
    top = out[0]
    assert "stated 0.8-0.9 (on average 0.84): it happened 54% of the time, n=60" in top.material
    assert "0.1-0.2" in top.material  # the faculty's other bins are shown
    spent = seeds().lost_bet_seeds(rows, spent_refs={"self_bet_bin:more_steps_than_usual:0.8-0.9"})
    assert [s.refs[0] for s in spent] == ["self_bet_bin:more_steps_than_usual:0.1-0.2"]


def test_lost_bet_gap_must_beat_two_standard_errors():
    # gap 0.16 but n=20 at p=0.5: se = 0.112, 2 se = 0.224 > 0.16 -> noise.
    assert seeds().lost_bet_seeds([_bin("f", 5, 20, 0.5, 0.66)], spent_refs=set()) == []


# --- 6. far memory: centered cosine, band, live records only ---------------------------------------


def _far_fixture():
    s = seeds()
    day = datetime(2026, 1, 1, tzinfo=timezone.utc)
    docs, vectors = {}, {}
    # anchor: newest record. Others: decreasing similarity along one axis, plus noise dims.
    docs["episode_memory:anchor"] = s.LivedDoc("episode_memory:anchor", "the newest memory", day + timedelta(days=300),
                                               day + timedelta(days=300))
    vectors["episode_memory:anchor"] = [1.0, 0.0, 0.0]
    for i in range(1, 31):
        did = f"episode_memory:m{i:02d}"
        docs[did] = s.LivedDoc(did, f"memory {i}", day + timedelta(days=i), day + timedelta(days=i))
        vectors[did] = [1.0 - i / 31.0, (i % 3) / 10.0, i / 31.0]
    return docs, vectors


def test_centered_cosine_ranks_nearest_first():
    docs, vectors = _far_fixture()
    ranked = seeds().centered_cosine_ranks("episode_memory:anchor", vectors)
    assert len(ranked) == 30 and "episode_memory:anchor" not in [d for d, _ in ranked]
    assert all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1))


def test_far_memory_seed_takes_the_band_and_the_oldest_in_it():
    docs, vectors = _far_fixture()
    seed = seeds().far_memory_seed(docs, vectors, band=(5, 25), spent_refs=set())
    ranked = [d for d, _ in seeds().centered_cosine_ranks("episode_memory:anchor", vectors)]
    band = ranked[4:25]
    assert seed.refs[0] == "episode_memory:anchor" and seed.refs[1] in band
    assert seed.refs[1] == min(band, key=lambda d: docs[d].occurred_at)
    assert "related to it but not the same" in seed.material


def test_far_memory_never_serves_a_superseded_memory():
    """A vector whose doc is not in the live snapshot (superseded, rejected, expired, removed) is dropped
    before ranking: it can never be the partner, even when it sits right in the band."""
    docs, vectors = _far_fixture()
    s = seeds()
    first = s.far_memory_seed(docs, vectors, band=(5, 25), spent_refs=set())
    superseded = first.refs[1]
    live = {k: v for k, v in docs.items() if k != superseded}  # the serving gate dropped it
    for _ in range(25):
        seed = s.far_memory_seed(live, vectors, band=(5, 25), spent_refs=set())
        if seed is None:
            break
        assert superseded not in seed.refs
        live = {k: v for k, v in live.items() if k != seed.refs[1]}
    # And an anchor that is not live is never used either.
    no_anchor = {k: v for k, v in docs.items() if k != "episode_memory:anchor"}
    seed = s.far_memory_seed(no_anchor, vectors, band=(5, 25), spent_refs=set())
    assert seed is None or "episode_memory:anchor" not in seed.refs


def test_far_memory_live_read_goes_through_the_recall_serving_gate():
    src = (SERVICE / "app" / "store.py").read_text()
    assert "from orion.memory.lived_memory import fetch_live_records" in src
    rsrc = (SERVICE / "app" / "reverie.py").read_text()
    assert "load_live_lived_memory()" in rsrc and "docs = {r.doc_id" in rsrc


# --- the picker --------------------------------------------------------------------------------------


def test_picker_takes_the_least_recently_used_kind_with_material():
    s = seeds()
    a = s.absence_seeds([_miss("e1", 0)], tz_name="America/Denver", spent_refs=set())
    b = s.lost_bet_seeds([_bin("f", 1, 60, 0.15, 0.40)], spent_refs=set())
    last = {"absence": NOW - timedelta(hours=1), "lost_bet": NOW - timedelta(days=2), "far_memory": NOW}
    assert s.pick_seed({"absence": a, "lost_bet": b}, last).kind == "lost_bet"
    assert s.pick_seed({"absence": a, "lost_bet": []}, last).kind == "absence"
    assert s.pick_seed({}, last) is None
    assert s.kinds_by_least_recent({})[0] == "absence"  # never used first, fixed order on ties


def test_lazy_picker_reads_a_kind_only_when_needed_and_survives_a_failing_reader():
    s, r = seeds(), rev()
    calls = []

    async def boom(spent):
        calls.append("absence")
        raise RuntimeError("db down")

    async def bets(spent):
        calls.append("lost_bet")
        return s.lost_bet_seeds([_bin("f", 1, 60, 0.15, 0.40)], spent_refs=spent)

    async def far(spent):
        calls.append("far_memory")
        return []

    seed = asyncio.run(r.pick_seed_lazily({}, {"absence": boom, "lost_bet": bets, "far_memory": far}, set()))
    assert seed.kind == "lost_bet" and calls == ["absence", "lost_bet"]


# --- 4. refusals ---------------------------------------------------------------------------------------


def _seed():
    return seeds().lost_bet_seeds([_bin("more_steps_than_usual", 1, 60, 0.15, 0.37)], spent_refs=set())[0]


REF = "self_bet_bin:more_steps_than_usual:0.1-0.2"


def check(answer, **kw):
    return rev().check_question(answer, seed=_seed(), **kw)


def test_an_answer_naming_its_seed_ref_passes():
    out = check({"question": "Which kinds of turns do I call short that run long?", "why": "my bets miss",
                 "refs": [REF], "cites": [], "none_reason": None})
    assert out["refused_reason"] is None and out["refs"] == [REF]


def test_a_ref_not_in_the_seed_is_refused():
    out = check({"question": "Why?", "refs": [REF, "episode_memory:x"]})
    assert out["refused_reason"] == "ref_not_in_seed" and "episode_memory:x" in out["refused_detail"]
    assert check({"question": "Why?", "refs": []})["refused_reason"] == "no_refs"


def test_a_near_duplicate_of_an_open_question_is_refused():
    open_texts = ["Which kinds of turns do I call short that actually run long?"]
    out = check({"question": "Which kinds of turns do I call short that run long?", "refs": [REF]},
                open_texts=open_texts)
    assert out["refused_reason"] == "near_duplicate"
    assert check({"question": "What changed in the walkway camera's angle last week?", "refs": [REF]},
                 open_texts=open_texts)["refused_reason"] is None


def test_machinery_words_are_refused():
    out = check({"question": "Why does the coalition keep winning the substrate?", "refs": [REF]})
    assert out["refused_reason"] == "machinery_words"


def test_too_long_unparseable_and_none_reason():
    assert check({"question": "x" * 301, "refs": [REF]})["refused_reason"] == "too_long"
    assert rev().check_question(None, seed=_seed())["refused_reason"] == "unparseable"
    assert check({"question": None, "none_reason": None})["refused_reason"] == "unparseable"
    none = check({"question": None, "none_reason": "Too few bets to tell."})
    assert none["refused_reason"] is None and none["none_reason"] == "Too few bets to tell."


def test_cites_are_only_shown_experience_ids():
    out = check({"question": "Q about my bets?", "refs": [REF], "cites": ["experience:a", "experience:zzz"]},
                now_event_ids=["a", "b"])
    assert out["cites"] == ["a"]


def test_parse_model_answer_reads_json_inside_text():
    r = rev()
    assert r.parse_model_answer('noise {"question": "q", "refs": []} trailing') == {"question": "q", "refs": []}
    assert r.parse_model_answer("no json") is None
    assert r.parse_model_answer({"a": 1}) == {"a": 1}


# --- the question call --------------------------------------------------------------------------------


def test_plan_request_is_the_reverie_question_verb_on_the_agent_lane_with_seed_and_now_only():
    r = rev()
    req = r.build_question_plan_request(_seed(), correlation_id="c1", orion_now="[experience:a] x")
    assert req.plan.verb_name == "reverie_question"
    assert req.args.extra["llm_route"] == "agent" and req.args.extra["execution_lane"] == "agent"
    assert set(req.context) == {"seed_kind", "seed_material", "seed_refs", "seed_evidence", "user_message",
                                "mode", "orion_now"}
    assert req.plan.metadata["recall_enabled_default"] == "false"


def test_prompt_renders_the_seed_and_warns_on_unrecognised():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(REPO / "orion" / "cognition" / "prompts")))
    s = seeds().absence_seeds([_miss("e2", 3)], tz_name="America/Denver", spent_refs=set())[0]
    text = env.get_template("reverie_question.j2").render(seed_kind=s.kind, seed_material=s.material,
                                                          seed_refs=s.refs, seed_evidence=s.evidence)
    assert "vision_percept_expectation:e2" in text and "Do not say they did not come" in text
    assert '"none_reason"' in text and "orion_now" not in text


# --- one poll, end to end with fakes --------------------------------------------------------------------


class Fake:
    def __init__(self, *, level="idle", ask_raises=None, answer=None, routes=0):
        self.level, self.ask_raises, self.routes = level, ask_raises, routes
        self.answer = answer if answer is not None else {
            "question": "Which kinds of turns do I call short that run long?", "why": "bets miss",
            "refs": [REF], "cites": ["experience:e1"], "none_reason": None}
        self.persisted, self.routed, self.updates, self.uses, self.asked = [], [], [], [], 0

    def deps(self):
        r = rev()

        async def arousal():
            return self.level

        async def gate_rows():
            return {"last_winner_at": NOW - timedelta(minutes=30), "latest_tick_at": NOW,
                    "rows_since_stretch": len(self.persisted), "rows_last_24h": len(self.persisted)}

        async def state():
            return {"last_used": {}, "spent_refs": set(), "threads": {}}

        async def bets(spent):
            return [_seed()]

        async def empty(spent):
            return []

        async def open_texts():
            return []

        async def now_cue(corr, run_ref):
            return r.NowCue(now_block="[experience:e1] x", shown_id="s1", shown_event_ids=("e1",))

        async def ask(seed, corr, now_block):
            self.asked += 1
            if self.ask_raises:
                raise self.ask_raises
            return self.answer, "model-x"

        async def persist(q):
            self.persisted.append(q)
            return True

        async def routes():
            return self.routes

        async def route(q):
            self.routed.append(q)
            return "focus:f1", "queued"

        async def update(qid, to, note):
            self.updates.append((qid, to, note))
            return True

        async def use(cue, q):
            self.uses.append((cue.shown_id, q.cites))

        return r.ReverieDeps(read_arousal=arousal, gate_rows=gate_rows, seed_state=state,
                             seed_readers={"absence": empty, "lost_bet": bets, "far_memory": empty},
                             open_texts=open_texts, now_cue=now_cue, ask=ask, persist=persist,
                             focus_routes_24h=routes, route=route, update_route=update, record_use=use,
                             now=lambda: NOW)


@pytest.fixture
def clean_state():
    r = rev()
    r._STRETCH_REFUSALS.clear()
    r._STRETCH_NO_MATERIAL.clear()
    yield r


def test_a_quiet_poll_writes_one_question_routes_it_and_records_the_cites(clean_state):
    f = Fake()
    q = asyncio.run(clean_state.run_reverie_once(f.deps()))
    assert q.accepted and q.seed_kind == "lost_bet" and q.routed_to == "focus:f1" and q.cites == ["e1"]
    assert len(f.persisted) == 1 and f.updates == [(q.question_id, "focus:f1", "queued")]
    assert f.uses == [("s1", ["e1"])]
    # The stretch is spent: the next poll writes nothing.
    assert asyncio.run(clean_state.run_reverie_once(f.deps())) is None and len(f.persisted) == 1


def test_engaged_arousal_writes_nothing(clean_state):
    f = Fake(level="engaged")
    assert asyncio.run(clean_state.run_reverie_once(f.deps())) is None and f.asked == 0


def test_a_refused_lane_writes_no_row_and_three_refusals_give_up_the_stretch(clean_state):
    f = Fake(ask_raises=clean_state.ReverieLlmUnavailable("gpu_pool_unavailable:shed:cabinet_hot"))
    for _ in range(5):
        assert asyncio.run(clean_state.run_reverie_once(f.deps())) is None
    assert f.persisted == [] and f.asked == 3
    assert clean_state.REVERIE_SKIP_COUNTS["lane_refused_stretch"] >= 1


def test_the_focus_cap_keeps_the_question_but_does_not_route_it(clean_state, monkeypatch):
    monkeypatch.setattr(clean_state.settings, "reverie_focus_max_per_day", 3)
    f = Fake(routes=3)
    q = asyncio.run(clean_state.run_reverie_once(f.deps()))
    assert q.accepted and q.routed_to == "none" and "cap reached (3 of 3" in q.route_note and f.routed == []


def test_a_refused_question_is_stored_and_never_routed(clean_state):
    f = Fake(answer={"question": "Why does the substrate keep winning?", "refs": [REF]})
    q = asyncio.run(clean_state.run_reverie_once(f.deps()))
    assert q.refused_reason == "machinery_words" and not q.accepted and f.routed == [] and len(f.persisted) == 1


def test_focus_why_names_the_seed_and_refs():
    r = rev()
    from orion.schemas.reverie import ReverieQuestionV1

    q = ReverieQuestionV1(question_id="q1", thread_id="t", thread_key="k", seed_kind="lost_bet", question="Q?",
                          why="bets miss", refs=[REF])
    assert r.focus_why(q) == f"bets miss (reverie lost_bet; reverie_question:q1; refs: {REF})"


# --- 8. the retired producers are gone ---------------------------------------------------------------


def test_the_narration_chain_is_retired():
    app = SERVICE / "app"
    for gone in ("chain.py", "resonance_monitor.py", "reverie_health_monitor.py", "grounding.py",
                 "broadcast_reader.py", "vision_reader.py"):
        assert not (app / gone).exists(), gone
    for verb in ("reverie_narrate", "reverie_expectation_judge"):
        assert not (REPO / "orion" / "cognition" / "verbs" / f"{verb}.yaml").exists()
    src = "\n".join(p.read_text() for p in app.glob("*.py"))
    for word in ("reverie.chain.v1", "persist_reverie_thought", "expectation_verdict", '"reverie_narrate"',
                 "substrate_reverie_refractory", "persist_reverie_chain", "run_reverie_chain"):
        assert word not in src, word


def test_settings_ship_on_with_the_spec_defaults(monkeypatch):
    for key in ("ORION_REVERIE_ENABLED", "ORION_REVERIE_INTERVAL_SEC", "ORION_REVERIE_QUIET_MIN_CALM_SEC",
                "ORION_REVERIE_MAX_PER_DAY", "ORION_REVERIE_FOCUS_MAX_PER_DAY", "ORION_REVERIE_FAR_MEMORY_BAND",
                "ORION_REVERIE_THREAD_LAPSE_DAYS"):
        monkeypatch.delenv(key, raising=False)
    from app.settings import ThoughtSettings

    s = ThoughtSettings(_env_file=None)
    assert s.reverie_enabled is True and s.reverie_interval_sec == 60 and s.reverie_quiet_min_calm_sec == 600
    assert s.reverie_max_per_day == 8 and s.reverie_focus_max_per_day == 3
    assert s.reverie_far_memory_band_ranks == (5, 25) and s.reverie_thread_lapse_days == 14


def test_far_memory_anchor_prefers_orions_own_memories():
    s = seeds()
    day = datetime(2026, 1, 1, tzinfo=timezone.utc)
    docs = {
        "person_fact:p": s.LivedDoc("person_fact:p", "a fact", day + timedelta(days=9), day + timedelta(days=9)),
        "episode_memory:e": s.LivedDoc("episode_memory:e", "mine", day, day),
    }
    assert s.pick_anchor(docs, set(docs)).doc_id == "episode_memory:e"
    assert s.pick_anchor({"person_fact:p": docs["person_fact:p"]}, {"person_fact:p"}).doc_id == "person_fact:p"
