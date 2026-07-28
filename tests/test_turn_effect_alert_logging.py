from orion.schemas.telemetry.turn_effect import should_emit_turn_effect_alert

# 2026-07-28 (spark-introspector retirement): every other test in this file
# hard-imported services/orion-spark-introspector/app/worker.py's
# turn-effect-alert helpers (_log_turn_effect_alert,
# _is_turn_effect_alert_blocked, _alert_severity, _is_dedupe_suppressed) --
# all deleted with the service. should_emit_turn_effect_alert itself is a
# real, shared function (orion/schemas/telemetry/turn_effect.py) with live
# callers elsewhere (services/orion-memory-consolidation/app/worker.py,
# services/orion-cortex-exec/app/executor.py), so this one test survives.


def test_turn_effect_alert_cooldown_blocks_second_emit():
    now = 100.0
    assert should_emit_turn_effect_alert(None, now, 120) is True
    assert should_emit_turn_effect_alert(now, now + 1, 120) is False
