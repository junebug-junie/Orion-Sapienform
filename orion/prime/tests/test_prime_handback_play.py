"""Game company patch 3 (spec 2026-11-16 rev 4, section 6 seam 3): play turns hand back to Prime as ``play``.

A line Orion speaks while someone plays runs in Juniper's own chat session, and a guest types on a browser
surface. Both must hand back as ``play``, checked before the rule for Juniper's chat, or Prime would record
them as Juniper speaking. A promise made during play still opens a concern Prime holds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.prime import handback as hb
from orion.prime.choice import option_text
from orion.prime.state import PrimeStateV1
from orion.schemas.cockpit_sighting import CockpitHopV1

T0 = datetime(2026, 11, 18, 19, 0, tzinfo=timezone.utc)
JUNIPER_SESSION = "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"
RETRO = {"surface": "retro", "input_modality": "typed", "retro_sitting_id": "s1"}
BROWSER = {"surface": "hub_desktop", "input_modality": "typed"}
QUOTE = "I will tell you when you reach the boss"


def test_a_line_turn_is_play_even_in_juniper_session():
    assert hb.classify_who(JUNIPER_SESSION, {"mode": "orion", "source": "retro_commentary"}) == "play"
    # even if a surface were attached, the source decides first
    assert hb.classify_who(JUNIPER_SESSION, {"source": "retro_commentary", "surface_context": BROWSER}) == "play"


def test_a_guest_message_from_the_game_page_is_play_not_juniper():
    raw = {"mode": "orion", "surface_context": RETRO, "chat_history_source": "retro_guest"}
    assert hb.classify_who("retro-guest-s1", raw) == "play"
    assert hb.classify_who(JUNIPER_SESSION, raw) == "play"


def test_an_ordinary_browser_turn_is_still_juniper():
    assert hb.classify_who(JUNIPER_SESSION, {"mode": "orion", "surface_context": BROWSER}) == "chat_juniper"
    # Juniper typing on the game page (stated): chat_history_source is hub_orion, so it is her chat.
    assert hb.classify_who(JUNIPER_SESSION, {"mode": "orion", "surface_context": RETRO,
                                             "chat_history_source": "hub_orion"}) == "chat_juniper"
    # other sources are not play
    assert hb.classify_who(JUNIPER_SESSION, {"mode": "orion", "surface_context": BROWSER,
                                             "source": "hub_orion"}) == "chat_juniper"


def _hops(corr: str, raw_extra: dict) -> list[dict]:
    ingress = {"session_id": JUNIPER_SESSION, "user_message": "say something about the game", **raw_extra}
    fin = {"correlation_id": corr, "final_text": f"Nice sword. {QUOTE}.", "compliance_verdict": "completed",
           "reflection": {"promises_made": [{"offer": False, "quote": QUOTE}], "promises_kept": []}}
    return [CockpitHopV1(correlation_id=corr, seq=0, ts=T0, stage="ingress", visor_line="ingress", status="ok",
                         summary={}, raw=ingress, producer="orion-hub").model_dump(mode="json"),
            CockpitHopV1(correlation_id=corr, seq=9, ts=T0 + timedelta(seconds=40), stage="finalize",
                         visor_line="finalize", status="ok", summary={}, raw=fin,
                         producer="orion-hub").model_dump(mode="json")]


def test_a_promise_made_during_play_opens_a_concern_never_labelled_as_one_to_juniper():
    signals = [s for s in (hb.signal_from_hop(h) for h in _hops("corr-line-1", {"source": "retro_commentary"}))
               if s is not None]
    state, turns, _ = hb.fold_turn_signals(PrimeStateV1(), hb.PrimeTurnsStateV1(), signals,
                                           T0 + timedelta(minutes=1))
    (turn,) = turns.recent
    assert turn.who == "play" and not turn.gist.startswith("Juniper:")
    (c,) = state.concerns
    assert c.promise_text.startswith("a promise I made during a game sitting at ")
    assert c.text.startswith("a promise I made in a game sitting") and "Juniper" not in c.text
    assert "Juniper" not in option_text(c)


def test_every_kind_of_turn_still_has_words():
    from typing import get_args

    assert "play" in get_args(hb.TurnWho) and set(hb.PROMISE_TO) == set(hb.WHO_WORDS) == set(get_args(hb.TurnWho))
