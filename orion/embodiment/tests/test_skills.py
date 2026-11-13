from orion.embodiment.skills import SKILLS, context, count_band, health_band, threat_band


def test_health_band_edges():
    assert health_band(3, 3) == 'full' and health_band(3.5, 3) == 'full'
    assert health_band(2.5, 3) == 'hurt' and health_band(1.5, 3) == 'hurt'
    assert health_band(1, 3) == 'low' and health_band(0.5, 3) == 'low'
    assert health_band(None, 3) == 'unknown' and health_band(2, 0) == 'unknown'


def test_threat_band_edges():
    assert threat_band([47, 200], 48) == 'near' and threat_band([48], 48) == 'near'
    assert threat_band([49], 48) == 'none' and threat_band([], 48) == 'none'


def test_context_is_sorted_and_drops_none():
    assert list(context(threat='none', health='full', side=None)) == ['health', 'threat']
    assert count_band(0) == 'none' and count_band(1) == 'one' and count_band(3) == 'many'
    assert SKILLS[0] == 'traverse' and len(SKILLS) == 6
