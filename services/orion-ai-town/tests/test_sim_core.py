"""Gate: the AI Town simulation core (living-world design §1-3, §8; PR 2).

Runs the shipped overlay/convex/sim/core.ts + seed.ts under Node >= 22.6 with
type stripping (no Convex, no LLM): acceptance A1 (a seeded 24 game-hour run
moves stock, needs and trust; replay is exact) and the Juniper half of A7.
Skips without a new enough node unless AITOWN_REQUIRE_TOOLS=1 (CI:
aitown-patch-chain.yml). The other tests are plain Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_SERVICE = Path(__file__).resolve().parents[1]
_SIM = _SERVICE / "overlay" / "convex" / "sim"
_DRIVER = _SERVICE / "tests" / "fixtures" / "sim_core_driver.ts"
sys.path.insert(0, str(_SERVICE / "scripts"))
import generate_descriptions as gen  # noqa: E402


def _node() -> str | None:
    node = os.environ.get("AITOWN_TEST_NODE") or shutil.which("node")
    if not node:
        return None
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    m = re.match(r"v(\d+)\.(\d+)", out)
    if not m or (int(m[1]), int(m[2])) < (22, 6):
        return None
    return node


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    node = _node()
    if not node:
        if os.environ.get("AITOWN_REQUIRE_TOOLS") == "1":
            pytest.fail("needs node >= 22.6 (set AITOWN_TEST_NODE)")
        pytest.skip("needs node >= 22.6 (set AITOWN_TEST_NODE)")
    work = tmp_path_factory.mktemp("sim")
    for name in ("core.ts", "seed.ts"):
        shutil.copy2(_SIM / name, work / name)
    shutil.copy2(_DRIVER, work / "driver.ts")
    run = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings", "driver.ts"],
        cwd=work, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout)


def test_a1_seeded_day_moves_stock_needs_and_trust(report):
    a1 = report["a1"]
    assert a1["minute"] == 24 * 60
    assert a1["stock_changed"], "no stock moved in 24 game-hours"
    assert a1["trust_changed"], "no trust moved in 24 game-hours"
    residents = {"mara_vale", "nico_sable", "sofia_bell", "cam_lin"}
    assert residents <= set(a1["needs_changed"])
    assert {"meal", "gossip", "coffee_delivery"} <= set(a1["kinds"])
    assert a1["sources"] == ["sim"]


def test_a1_replay_is_exact(report):
    a1 = report["a1"]
    assert a1["replay_identical"]
    # 288 five-minute ticks with a Convex-style save (sorted keys) in between
    # give the same world and the same events as one 24-hour tick.
    assert a1["stepped_identical"]
    assert a1["other_seed_differs"]
    assert a1["input_untouched"]


def test_effects_clamp(report):
    assert report["clamp"] == {"rest": 1, "social": 0, "trust_up": 1, "trust_down": -1, "coffee": 0}


def test_invalid_effects_are_rejected_whole(report):
    r = report["rejects"]
    assert r["unknown_op"].startswith("SimError: unknown effect op teleport")
    assert "unknown resident ghost" in r["half_valid"]
    assert "money follows cash" in r["money_need"]
    assert "unknown resident orion" in r["visitor_need"]
    assert "unknown item" in r["unknown_item"]
    assert "unknown place bakery" in r["unknown_place"]
    assert "integer" in r["fractional_stock"]
    assert "scalar" in r["object_value"] or "number, string or boolean" in r["object_value"]
    assert r["untouched"]


def test_free_text_event_is_a_rumor_and_moves_no_numbers(report):
    f = report["free_text"]
    assert f["status"] == "rumor"
    assert f["fact"] == "Someone saw lights over the square."
    assert f["numbers_same"] and f["rng_same"]
    assert f["rumors_kept"] == 20 and f["oldest_kept"] == 6


def test_a7_juniper_source_accepted_others_refused(report):
    s = report["sources"]
    assert (s["juniper_status"], s["juniper_source"], s["juniper_fact"]) == ("applied", "juniper", True)
    for name in ("orion", "mirror", "missing"):
        assert s[name].startswith("SimError: unknown source"), name


def test_quest_succeeds_and_fails(report):
    q = report["quests"]
    assert q["opened_status"] == "quest_opened"
    assert q["open_after_hour"] == "open"
    assert q["cam_purpose_rose"]
    assert (q["succeeded"], q["success_fact"]) == ("succeeded", "succeeded")
    assert q["reward_trust"] == pytest.approx(0.2)
    assert q["reward_cash"] == 5
    assert q["resolved_event"] == ["sim"]
    assert (q["failed"], q["failed_at"]) == ("failed", 30)
    assert q["penalty_trust"]
    assert "unknown quest participants" in q["bad_participant"]
    assert "unknown comparison" in q["bad_cmp"]


def test_entity_rule(report):
    assert report["entities"] == ["Maggie", "blacksmith", "Tom"]


def test_tick_takes_whole_steps(report):
    assert "multiple of 5" in report["tick_bad_minutes"]


# ---- plain Python: the core stays importable by node, the seed tracks the cards ----

def test_core_imports_nothing():
    core = (_SIM / "core.ts").read_text(encoding="utf-8")
    assert not re.search(r"^\s*import\b", core, re.M)
    assert not re.search(r"\b(enum|namespace)\s+\w+", core)


def test_committed_seed_matches_example_cards():
    cards = yaml.safe_load(gen.CARDS_EXAMPLE.read_text(encoding="utf-8"))
    assert (_SIM / "seed.ts").read_text(encoding="utf-8") == gen.render_sim_seed(cards)


def test_seed_has_residents_and_visitors():
    cards = yaml.safe_load(gen.CARDS_EXAMPLE.read_text(encoding="utf-8"))
    seed = gen.render_sim_seed(cards)
    cast = json.loads(seed.split("export const SIM_CAST: Cast = ", 1)[1].rstrip().rstrip(";"))
    sims = {c["id"]: c["sim"] for c in cast["characters"]}
    assert all(sims[i] for i in gen.NPC_ORDER)
    assert sims["juniper_feld"] is None and sims["orion"] is None
    assert {sims[i]["work"] for i in gen.NPC_ORDER} <= set(cast["places"])


def test_local_cards_without_sim_blocks_borrow_the_example():
    cards = yaml.safe_load(gen.CARDS_EXAMPLE.read_text(encoding="utf-8"))
    stale = {**cards, "characters": [{k: v for k, v in c.items() if k != "sim"} for c in cards["characters"]]}
    del stale["sim_world"]
    assert gen.render_sim_seed(stale) == gen.render_sim_seed(cards)


def test_sim_block_missing_a_key_fails():
    cards = yaml.safe_load(gen.CARDS_EXAMPLE.read_text(encoding="utf-8"))
    del cards["characters"][0]["sim"]["wage"]
    with pytest.raises(SystemExit, match="missing wage"):
        gen.render_sim_seed(cards)
