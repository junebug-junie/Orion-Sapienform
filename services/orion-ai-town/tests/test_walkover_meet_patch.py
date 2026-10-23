"""Gate: walk-over meets across a wall, and gives up eventually (2026-10-23).

Live: both NPC pairs stood on either side of a wall, walkingOver, for good.
The midpoint was the wall tile, findRoute failed every tick, and the
conversation-proximity patch had removed the only timeout.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-walkover-meet.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def _added() -> str:
    return "\n".join(
        line[1:]
        for line in _PATCH.read_text(encoding="utf-8").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_registered_after_proximity_patch():
    order = [
        line.strip().strip('",')
        for line in _APPLY.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert order.index("orion-conversation-proximity.patch") < order.index("orion-walkover-meet.patch")


def test_blocked_midpoint_sends_one_player_next_to_the_other():
    added = _added()
    assert "if (blockedWithPositions(destination, [], game.worldMap) !== null) {" in added
    # Exactly one of the pair waits, so they never chase each other's tiles.
    assert "if (player.id > otherPlayer.id) {" in added
    assert "destination = freeTileNextTo(game, now, otherPlayer.position, player.id) ?? destination;" in added
    assert "if (blocked(game, now, tile, playerId) === null) {" in added


def test_walk_over_has_a_backstop_timeout():
    added = _added()
    assert "export const WALK_OVER_TIMEOUT = 3 * 60000;" in added
    assert "if (member.invited + WALK_OVER_TIMEOUT < now) {" in added
    assert "conversation.leave(game, now, player);" in added
