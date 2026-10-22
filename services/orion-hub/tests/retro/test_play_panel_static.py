"""The play panel is in the rendered template and wired in the linked asset (browser eval:
evals/retro_play_panel.py exercises the interaction against a fixture Retro)."""
from pathlib import Path

HUB = Path(__file__).resolve().parents[2]


def test_template_has_play_panel_and_current_asset():
    html = (HUB / 'templates/retro.html').read_text()
    for marker in ('id="play"', 'id="atlas"', 'id="leisure"', 'id="zgoal"', 'id="zprediction"',
                   'id="verdicts"', 'id="takeover"', '/static/js/retro-panel.js?v=8'):
        assert marker in html, marker


def test_asset_reads_leisure_atlas_and_shed_and_watches_without_a_controller():
    js = (HUB / 'static/js/retro-panel.js').read_text()
    for marker in ("api('leisure')", '/atlas`', "'gpu_pool_unavailable'", 'retry_after_sec',
                   'function watch()', "'Orion is waiting: '", "'zelda-navigator-v1'", 'last_finished'):
        assert marker in js, marker
    watch = js[js.index('function watch()'):js.index('let atlasAt')]
    assert '/input' not in watch and 'grant(' not in watch   # watching never takes a controller
