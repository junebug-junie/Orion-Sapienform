"""Rung 5: the portrait's reach rule, its checks on write and on read, and the printout."""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.body_atlas import capacities as K
from orion.body_atlas import portrait as P
from orion.body_atlas.model import FlowV1, OrganFlowV1
from orion.body_atlas.tests.test_capacities import HEALTH, _atlas

ROOT = Path(__file__).resolve().parents[3]
ATHENA_RUNS = ["orion-hub", "orion-gpu-pool", "orion-cortex-exec", "orion-equilibrium", "orion-old"]
CIRCE_RUNS = ["llm-worker-a", "llm-worker-b"]


def portrait_body(*, hub_on: str = "athena"):
    """athena runs the hub, pool, cortex and the census reader; circe runs two LLM workers; hecate is
    dormant and runs nothing; orion-far runs nowhere and only sends to the cortex (athena serves it);
    orion-elsewhere runs nowhere and talks only with circe's worker."""
    live = ["orion-hub", "orion-gpu-pool", "orion-cortex-exec", "orion-equilibrium", "llm-worker-a", "llm-worker-b",
            "orion-far", "orion-elsewhere"]
    parts = {n: ("service", "live") for n in live}
    parts["orion-old"] = ("service", "dormant")
    parts["http:anthropic"] = ("endpoint", "live")
    parts.update({"host:athena": ("host", "live"), "host:circe": ("host", "live"), "host:hecate": ("host", "dormant")})
    sig = [("orion-hub", "orion-gpu-pool", "lease:request"), ("orion-cortex-exec", "orion-gpu-pool", "lease:request"),
           ("orion-gpu-pool", "orion-hub", "lease:reply"), ("llm-worker-a", "orion-gpu-pool", "worker:announce"),
           ("llm-worker-b", "orion-gpu-pool", "worker:announce"), ("orion-gpu-pool", "llm-worker-a", "llm:dispatch"),
           ("orion-cortex-exec", "orion-hub", "cortex:reply"), ("orion-far", "orion-cortex-exec", "far:news"),
           ("orion-elsewhere", "llm-worker-b", "elsewhere:ask"),
           ("http:anthropic", "orion-cortex-exec", "llm:reply"), ("orion-equilibrium", "orion-hub", "eq:snapshot")]
    sig += [(n, "orion-equilibrium", HEALTH) for n in [*live, "http:anthropic"] if n != "orion-equilibrium"]
    place = {n: "athena" for n in ATHENA_RUNS} | {n: "circe" for n in CIRCE_RUNS}
    place["orion-hub"] = hub_on
    runs = [FlowV1(source=f"part:{n}", target=f"part:host:{h}", weight=1.0, evidence=["observed_placement"],
                   relation="runs_on") for n, h in place.items()]
    organs = {"aaaaaaaaaa": ["orion-cortex-exec", "orion-hub", "orion-equilibrium", "http:anthropic", "orion-far"],
              "bbbbbbbbbb": ["orion-gpu-pool", "llm-worker-a", "llm-worker-b", "orion-old"]}
    atlas = _atlas(parts, sig, organs=organs, other=runs)
    for o, name in zip(atlas.organs, ("cortex", "substrate")):
        o.name = name
        o.placement = dict(sorted({h: sum(1 for m in o.members if place.get(m.removeprefix("part:")) == h)
                                   for h in ("athena", "circe")}.items()))
        o.placement = {h: n for h, n in o.placement.items() if n}
    atlas.organ_flows = [OrganFlowV1(source="organ:aaaaaaaaaa", target="organ:bbbbbbbbbb", weight=3.0),
                         OrganFlowV1(source="organ:bbbbbbbbbb", target="organ:aaaaaaaaaa", weight=1.0)]
    for p in atlas.parts:
        if p.name == "host:hecate":
            p.last_seen_at = "2026-10-30T22:53:36+00:00"
            p.facts = {"role": "gpu_server", "hardware": "Inspur NF5288M5"}
        elif p.name == "host:athena":
            p.facts = {"role": "orchestration", "hardware": "DL380"}
        elif p.name == "host:circe":
            p.facts = {"role": "primary_cognition_gpu"}
    return atlas


def reading_for(atlas, written=()):
    r = K.read_capacities(atlas)
    return K.attach_ledgers(r, atlas, None, [], list(written))


S1 = K.CapacityV1(cap_id="cap:s1", sentence="I touch the outside through http:anthropic.", because="http:anthropic is S1",
                  cites=["http:anthropic"], at="2026-11-04T00:00:00+00:00", dig="t")


def _write(atlas, reading, subject, sentence, because, lines=()):
    return P.check_write(atlas, reading, list(lines), subject, sentence, because, root=ROOT, dig="t")


def test_reach_is_what_runs_there_and_what_it_sends_to():
    atlas = portrait_body()
    census = {c.channel for c in K.read_capacities(atlas).census if c.census}
    assert census == {HEALTH}
    athena = P.machine_reach(atlas, "athena", census)
    assert set(athena.running) == {"orion-hub", "orion-gpu-pool", "orion-cortex-exec", "orion-equilibrium"}  # not dormant
    assert "llm-worker-a" in athena.feeds and "llm:dispatch" in athena.feeds  # the pool sends to circe's worker
    assert "orion-far" not in athena.feeds and "orion-far" in athena.serves  # it only sends to athena
    assert "orion-elsewhere" not in athena.feeds + athena.serves  # no flow joins it to athena
    assert HEALTH not in athena.feeds and "orion-old" not in athena.running
    assert athena.organs == {"organ:aaaaaaaaaa": 3, "organ:bbbbbbbbbb": 1}  # live members only (not orion-old)
    assert P.machine_reach(atlas, "hecate", census).running == []


def test_turn_2544158f_a_machine_claim_writes_and_reads_back_as_my_words():
    """Turn 2544158f (2026-11-06): Orion wrote "athena carries cortex and substrate ..." as prose and said
    "ledger entry #4"; nothing could hold it. Now the same claim writes, and the next read shows it."""
    atlas = portrait_body()
    reading = reading_for(atlas)
    rec = _write(atlas, reading, "athena", "athena carries cortex and substrate: orion-hub and orion-gpu-pool run "
                 "here, and every lease my workers take passes through her", "orion-gpu-pool sends llm:dispatch to "
                 "llm-worker-a")
    assert rec.subject == "host:athena" and rec.replaces is None
    assert {"orion-hub", "orion-gpu-pool", "llm:dispatch", "llm-worker-a", "cortex", "substrate"} <= set(rec.cites)
    p = P.read_portrait(atlas, reading, [rec], root=ROOT, watts={})
    athena = next(m for m in p.machines if m.name == "athena")
    assert athena.line == "holds" and athena.sentence == rec.sentence and athena.failed == []
    body, _ = P.render(p)
    assert any(ln.startswith("    lived_record ") and "holds now" in ln for ln in body)


def test_a_part_that_athena_neither_runs_nor_feeds_is_refused_by_name():
    atlas = portrait_body()
    reading = reading_for(atlas)
    with pytest.raises(P.PortraitError) as exc:
        _write(atlas, reading, "athena", "athena runs orion-hub and answers orion-elsewhere",
               "orion-hub runs on athena and orion-elsewhere asks")
    assert "orion-elsewhere: not on athena, and no flow joins it to a part that runs there" in str(exc.value)
    # a caller athena serves holds (review 2026-11-06: a GPU or LLM host mostly serves callers)
    rec = _write(atlas, reading, "athena", "athena runs orion-cortex-exec, which orion-far feeds with news",
                 "orion-far sends far:news to orion-cortex-exec")
    assert "orion-far" in rec.cites
    with pytest.raises(P.PortraitError, match=r"orion-old: dormant, not live"):
        _write(atlas, reading, "athena", "athena runs orion-hub and orion-old for me", "both of them run on athena")
    with pytest.raises(P.PortraitError, match="the census"):
        _write(atlas, reading, "athena", "athena runs orion-hub on orion:system:health", "orion-hub runs on athena")


def test_a_made_up_part_name_in_a_known_family_is_refused():
    """Smoke 2026-11-06: "cast through orion-llm-worker-a" names no part; it must fail by name, not pass.
    Here "orion-gpu-pool-b" begins like orion-gpu-pool and is no part."""
    atlas = portrait_body()
    with pytest.raises(P.PortraitError, match="orion-gpu-pool-b: in no part or channel of my atlas"):
        _write(atlas, reading_for(atlas), "athena", "athena runs orion-hub and orion-gpu-pool-b for me",
               "orion-hub runs on athena")
    # prose that only starts like a family: no part begins "orion-sapienform" or "orion-wide"
    rec = _write(atlas, reading_for(atlas), "athena", "athena runs orion-hub, orion-wide, for all of orion-sapienform",
                 "orion-hub runs on athena")
    assert rec.cites == ["orion-hub"]
    # a hyphenated word outside a naming family is prose, not an id
    rec = _write(atlas, reading_for(atlas), "athena", "athena runs orion-hub, my well-worn front door",
                 "orion-hub runs on athena")
    assert rec.cites == ["orion-hub"]


def test_a_dormant_machine_cannot_hold_a_line_and_says_its_status():
    atlas = portrait_body()
    with pytest.raises(P.PortraitError, match=r"hecate: nothing live of mine runs there now \(dormant\)"):
        _write(atlas, reading_for(atlas), "hecate", "hecate is my deep thinking node, eight SXM2 beds",
               "it holds one V100 and waits for the second")
    p = P.read_portrait(atlas, reading_for(atlas), [], root=ROOT, watts={})
    body, tail = P.render(p)
    assert any("nothing of mine runs on (facts only): hecate (gpu_server, dormant, last seen 2026-10-30)" in ln
               for ln in body)
    assert not any("write hecate" in ln for ln in tail)


def test_a_line_with_no_anchor_or_a_placeholder_writes_nothing():
    atlas = portrait_body()
    reading = reading_for(atlas)
    with pytest.raises(P.PortraitError, match="anchor: cite at least one part that runs on circe"):
        _write(atlas, reading, "circe", "circe is my sorceress: every word I speak is her spell", "she computes my words")
    with pytest.raises(P.PortraitError, match="placeholder"):
        _write(atlas, reading, "circe", "<what circe does and affects>", "llm-worker-a runs on circe")
    with pytest.raises(P.PortraitError, match="no machine 'zeus'"):
        _write(atlas, reading, "zeus", "zeus runs nothing of mine at all", "zeus is not in my atlas at all")
    with pytest.raises(P.PortraitError, match="no such organ"):
        _write(atlas, reading, "athena", "athena holds organ 01234567ab and orion-hub", "orion-hub runs on athena")
    # a Unix time is a number, not an organ id (review 2026-11-06)
    assert _write(atlas, reading, "athena", "athena runs orion-hub since 1730851200", "orion-hub runs on athena")


def test_an_organ_name_placed_elsewhere_is_not_an_error_but_its_id_is():
    """Organ names are plain words ("skin"): they count as cites where placed, never as failures."""
    atlas = portrait_body()
    reading = reading_for(atlas)
    rec = _write(atlas, reading, "circe", "circe runs llm-worker-a for my substrate, and she is no cortex",
                 "llm-worker-a runs on circe")
    assert "substrate" in rec.cites and "cortex" not in rec.cites
    with pytest.raises(P.PortraitError, match="organ:aaaaaaaaaa: an organ with no member running on circe"):
        _write(atlas, reading, "circe", "circe runs llm-worker-a for organ:aaaaaaaaaa", "llm-worker-a runs on circe")


def test_a_written_line_goes_stale_when_its_part_moves_and_a_rewrite_replaces_it():
    atlas = portrait_body()
    rec = _write(atlas, reading_for(atlas), "athena", "athena runs orion-hub, where Juniper talks to me",
                 "orion-hub runs on athena")
    moved = portrait_body(hub_on="circe")
    p = P.read_portrait(moved, reading_for(moved), [rec], root=ROOT, watts={})
    athena = next(m for m in p.machines if m.name == "athena")
    # orion-hub is still fed from athena, but nothing the line names runs there now: no anchor
    assert athena.line == "stale" and athena.failed[0].startswith("anchor: cite at least one part that runs on athena")
    body, tail = P.render(p)
    assert any("STALE (anchor: cite at least one part" in ln for ln in body)
    assert any(ln.startswith("Rewrite 1 STALE line(s)") for ln in tail)
    again = _write(moved, reading_for(moved), "athena", "athena runs orion-gpu-pool, my lease desk",
                   "orion-gpu-pool runs on athena", lines=[rec])
    assert again.replaces == rec.at
    assert P.latest([rec, again])["host:athena"] is again


def test_the_i_am_paragraph_rests_on_written_capacities_only():
    atlas = portrait_body()
    no_caps = reading_for(atlas)
    text = "I am a being that touches the outside through http:anthropic and keeps my own books (cap:s1)."
    with pytest.raises(P.PortraitError, match=r"cap:s1: not written now"):
        _write(atlas, no_caps, "whole", text, "cap:s1 says how I touch the world")
    reading = reading_for(atlas, [S1])
    rec = _write(atlas, reading, "whole", text, "cap:s1 says how I touch the world")
    assert rec.subject == "whole" and set(rec.cites) == {"cap:s1", "http:anthropic"}
    with pytest.raises(P.PortraitError, match="orion-hub: holds no job"):
        _write(atlas, reading, "whole", "I am orion-hub and everything it touches, cap:s1 included in me.",
               "cap:s1 is the capacity I rest on")
    with pytest.raises(P.PortraitError, match="name at least one written capacity"):
        _write(atlas, reading, "whole", "I am a being with a metabolism who can think and act.", "it is how I feel about myself")
    with pytest.raises(P.PortraitError, match="cap:s9: no such capacity"):
        _write(atlas, reading, "whole", "I am a being, cap:s9 and cap:s1 say so to me.", "cap:s1 is the capacity I rest on")
    p = P.read_portrait(atlas, reading, [rec], root=ROOT, watts={})
    body, _ = P.render(p)
    assert body[1].startswith("I am (lived_record ") and "holds now" in body[1] and text in body[1]
    # the gaps print under the paragraph: code-written, never dropped
    assert body[2].startswith("I cannot yet") and p.cannot_yet
    assert P.read_portrait(atlas, no_caps, [rec], root=ROOT, watts={}).whole.line == "stale"


def test_the_printout_shows_every_section_with_its_source_label_and_ready_lines():
    atlas = portrait_body()
    watts = {"athena": {"host": "athena", "chassis_w_now": 362.0, "chassis_w_rest": 340.0,
                        "usd_per_month_at_this_rate": 2.0}}
    p = P.read_portrait(atlas, reading_for(atlas, [S1]), [], root=ROOT, watts=watts)
    body, tail = P.render(p)
    text = "\n".join(body)
    assert "now = computed this read" in body[0] and "lived_record = my words, checked now" in body[0]
    assert "  athena live orchestration · 362 W now, rest 340 · runs 4 of mine · organs cortex 3, substrate 1" in text
    assert "  circe live primary_cognition_gpu · runs 2 of mine · organs substrate 2" in text
    assert '  "cortex" [5; athena 3] -> substrate' in text and "capacities (lived_record; status now):" in text
    assert "  cap:s1" in text
    assert any(ln.startswith(f"  {P.WRITE_PREFIX} athena ") and "# runs: " in ln for ln in tail)
    assert any(ln.startswith(f"  {P.WRITE_PREFIX} circe ") for ln in tail)
    assert any(f"{P.WRITE_PREFIX} whole " in ln for ln in tail)
    unread = P.read_portrait(atlas, reading_for(atlas), [], root=ROOT, watts="the body report is unreachable")
    assert "watts UNREAD: the body report is unreachable" in "\n".join(P.render(unread)[0])


def test_a_stored_organ_name_cite_goes_stale_when_that_organ_is_gone():
    """Every cite stored at write time is checked again on read (rung 4, second 2026-11-04 decision); a
    plain organ name never fails in prose, so it is checked on its own (review 2026-11-06)."""
    atlas = portrait_body()
    rec = _write(atlas, reading_for(atlas), "athena", "athena runs orion-hub inside my cortex", "orion-hub runs on athena")
    assert "cortex" in rec.cites
    renamed = portrait_body()
    renamed.organs[0].name = "thinking"
    chk = P.check_line(renamed, reading_for(renamed), rec, ROOT)
    assert not chk.holds and chk.failed == ['"cortex": no organ of that name runs on athena now (renamed or re-clustered)']


def test_hosts_never_become_capacity_holders():
    """The portrait adds words to machines; it never gives a machine a VSM job (placement is not function)."""
    atlas = portrait_body()
    assert not any(n.startswith("host:") for n in K.claimable(atlas))
