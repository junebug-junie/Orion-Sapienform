"""The chat prompt's play line from the sitting record (game company patch 1).

Lives outside tests/retro: it imports orion.hub.turn_orchestrator, which needs the full Hub
dependencies (the native-boundary job runs tests/retro with a small set)."""
import asyncio
import json
from types import SimpleNamespace

from orion.retro.sitting import SITTING_KEY


class Redis:
    def __init__(self): self.data = {}
    async def get(self, key): return self.data.get(key)


def test_the_chat_prompt_gets_one_play_line_only_while_someone_plays():
    from orion.hub import turn_orchestrator as orch

    open_ = {"sitting_id": "s", "started_at": 0.0, "last_input_at": 60.0 * 12, "ended_at": None,
             "game_title": "A Link to the Past", "player_stated": "juniper"}
    on = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True)

    async def check():
        r = Redis(); bus = SimpleNamespace(redis=r)
        assert await orch._situation_with_game_sitting("Situation: midday.", bus, settings=on) == "Situation: midday."
        r.data[SITTING_KEY] = json.dumps(open_)
        text = await orch._situation_with_game_sitting("Situation: midday.", bus, settings=on, correlation_id="c1")
        assert text.startswith("Situation: midday.\nPlay: Juniper is playing A Link to the Past on your Retro service, for 12 min so far.")
        assert (await orch._situation_with_game_sitting(None, bus, settings=on)).startswith("Play: Juniper is playing")
        off = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=False)
        assert await orch._situation_with_game_sitting("Situation: midday.", bus, settings=off) == "Situation: midday."
        # a closed sitting, an unconnected bus and a broken Redis all leave the prompt as it was
        r.data[SITTING_KEY] = json.dumps(dict(open_, ended_at=800.0))
        assert await orch._situation_with_game_sitting("S", bus, settings=on) == "S"

        class NotConnected:
            @property
            def redis(self): raise RuntimeError("bus not connected")
        assert await orch._situation_with_game_sitting("S", NotConnected(), settings=on) == "S"
        assert await orch._situation_with_game_sitting("S", None, settings=on) == "S"
    asyncio.run(check())
